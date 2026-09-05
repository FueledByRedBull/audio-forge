"""Offline checks for corresponding-source manifest hydration."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tomllib
from typing import Any, cast
import urllib.error
import urllib.request

import pytest

import source_distribution as source_tool
from source_distribution import (
    SourceDistributionError,
    _archive_target,
    _recipe_sha256,
    _read_locked_versions,
    _write_receipt,
    download_sources,
    verify_sources,
)


def _entry(payload: bytes = b"source archive") -> dict[str, str]:
    return {
        "id": "python-example-1.0",
        "kind": "python-sdist",
        "name": "Example",
        "version": "1.0",
        "filename": "example-1.0.tar.gz",
        "url": "https://files.pythonhosted.org/packages/example-1.0.tar.gz",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _mock_urlopen(monkeypatch, payload: bytes, final_url: str | None = None):
    class Response:
        def __init__(self):
            self._remaining = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def geturl(self):
            return final_url or "https://files.pythonhosted.org/packages/example-1.0.tar.gz"

        def read(self, size: int = -1):
            value, self._remaining = self._remaining, b""
            return value

    monkeypatch.setattr(source_tool, "_open_source_url", lambda *args, **kwargs: Response())


def test_locked_versions_ignore_comments_and_preserve_exact_pins(tmp_path: Path):
    lock = tmp_path / "requirements.txt"
    lock.write_text(
        "Example-Package==1.2.3 \\\n    --hash=sha256:deadbeef\n"
        "# ignored==9.9.9\n"
        "local @ file:///tmp/local.whl\n",
        encoding="utf-8",
    )

    assert _read_locked_versions(lock) == {"example-package": "1.2.3"}


def test_download_and_verify_source_archive(tmp_path: Path, monkeypatch):
    payload = b"source archive"
    entry = _entry(payload)
    manifest = {"entries": [entry], "blockers": []}
    destination = tmp_path / "hydrated"
    _mock_urlopen(monkeypatch, payload)

    downloaded = download_sources(manifest, destination)
    assert downloaded == [_archive_target(destination, entry)]
    verify_sources(manifest, destination)


def test_download_refuses_existing_hash_mismatch(tmp_path: Path):
    entry = _entry()
    manifest = {"entries": [entry], "blockers": []}
    target = _archive_target(tmp_path / "hydrated", entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tampered")

    with pytest.raises(SourceDistributionError, match="hash mismatch"):
        download_sources(manifest, tmp_path / "hydrated")

    assert target.read_bytes() == b"tampered"


def test_recipe_digest_normalizes_checkout_line_endings(tmp_path: Path):
    recipe = tmp_path / "recipe.ps1"
    recipe.write_bytes(b"one\r\ntwo\r\n")
    lf_digest = _recipe_sha256(recipe)
    recipe.write_bytes(b"one\ntwo\n")
    assert _recipe_sha256(recipe) == lf_digest
    recipe.write_bytes(b"one\ntwo!\n")
    assert _recipe_sha256(recipe) != lf_digest


def test_download_wraps_network_failure_with_entry_id(tmp_path: Path, monkeypatch):
    entry = _entry()

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(entry["url"], 403, "forbidden", cast(Any, {}), None)

    monkeypatch.setattr(source_tool, "_open_source_url", fail)
    target = _archive_target(tmp_path / "hydrated", entry)

    with pytest.raises(SourceDistributionError, match="python-example-1.0"):
        source_tool._download(entry, target)

    assert not list(target.parent.glob("*.part"))


def test_verify_requires_explicit_incomplete_override(tmp_path: Path):
    entry = _entry()
    entry["status"] = "blocked"
    manifest = {
        "entries": [entry],
        "blockers": ["df.dll source is unresolved"],
    }

    with pytest.raises(SourceDistributionError, match="remains incomplete"):
        verify_sources(manifest, tmp_path)

    verify_sources(manifest, tmp_path, allow_incomplete=True)


def test_verify_rejects_stale_archive_in_hydration(tmp_path: Path):
    payload = b"source archive"
    entry = _entry(payload)
    manifest = {"entries": [entry], "blockers": []}
    target = _archive_target(tmp_path, entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    (target.parent / "stale--old.tar.gz").write_bytes(b"old")

    with pytest.raises(SourceDistributionError, match="unexpected source archive"):
        verify_sources(manifest, tmp_path)


def test_release_manifest_cannot_claim_complete_with_blocked_entry():
    entry = _entry()
    entry["status"] = "blocked"
    with pytest.raises(SourceDistributionError, match="Complete source manifest"):
        source_tool._validate_manifest(
            {
                "schema_version": 1,
                "project": "AudioForge",
                "project_version": "2.0.0",
                "status": "complete",
                "blockers": [],
                "entries": [entry],
            },
            release=True,
        )


def test_release_manifest_rejects_stale_qt_source_identity():
    manifest_path = Path(__file__).resolve().parents[2] / "licenses" / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qt_entry = next(entry for entry in manifest["entries"] if entry["kind"] == "qt-source")
    qt_entry["sha256"] = "0" * 64

    with pytest.raises(SourceDistributionError, match="stale: qt-"):
        source_tool._validate_manifest(manifest, release=True)


def test_validate_project_archive_accepts_real_git_archive(tmp_path: Path):
    revision = source_tool._git_text("rev-parse", "--verify", "HEAD^{commit}")
    project_version = tomllib.loads(
        source_tool._git_text("show", f"{revision}:pyproject.toml")
    )["project"]["version"]
    archive_bytes = source_tool._git_archive_bytes(revision, project_version)

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        root = archive.getmembers()[0]
        assert root.name == f"AudioForge-{project_version}-{revision[:12]}"
        assert root.isdir()

    archive_path = tmp_path / "AudioForge-project-source.tar"
    archive_path.write_bytes(archive_bytes)
    digest = source_tool._validate_project_archive(
        archive_path,
        {"project_version": project_version},
        revision,
    )

    assert digest == hashlib.sha256(archive_bytes).hexdigest()


def test_source_receipt_binds_manifest_and_revision(tmp_path: Path, monkeypatch):
    payload = b"source archive"
    entry = _entry(payload)
    manifest = {
        "project": "AudioForge",
        "project_version": "2.0.0",
        "entries": [entry],
        "blockers": [],
    }
    archive = _archive_target(tmp_path, entry)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(payload)
    (tmp_path / "AudioForge-project-source.tar").write_bytes(b"project archive")
    monkeypatch.setattr(source_tool, "_git_text", lambda *args: "0123456789abcdef")

    receipt_path = _write_receipt(
        tmp_path,
        manifest,
        revision="v2.0.0",
        include_runtime_assets=False,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["revision"] == "0123456789abcdef"
    assert receipt["manifest_sha256"] == source_tool._manifest_digest(manifest)
    assert receipt["archives"][0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_verify_rejects_malformed_manifest_entry(tmp_path: Path):
    with pytest.raises(SourceDistributionError, match="non-object entry"):
        verify_sources({"entries": [None], "blockers": []}, tmp_path)


def test_blocked_runtime_asset_is_skipped_by_default(tmp_path: Path):
    blocked = {
        "id": "runtime-df.dll",
        "kind": "runtime-asset",
        "status": "blocked",
        "name": "df.dll",
        "version": "pinned",
        "filename": "df.dll",
        "url": "https://github.com/example/df.dll",
        "sha256": "0" * 64,
    }
    manifest = {"entries": [blocked], "blockers": ["df.dll source is unresolved"]}

    assert download_sources(manifest, tmp_path) == []
    verify_sources(manifest, tmp_path, allow_incomplete=True)


def test_source_built_runtime_output_is_attested_separately(tmp_path: Path):
    entry = {
        "id": "runtime-df.dll",
        "kind": "runtime-asset",
        "status": "available",
        "source_build": True,
        "name": "df.dll",
        "version": "pinned",
        "filename": "df.dll",
        "url": "https://github.com/Rikorose/DeepFilterNet/tree/commit",
        "sha256": "0" * 64,
    }
    manifest = {"entries": [entry], "blockers": []}

    assert download_sources(manifest, tmp_path, include_runtime_assets=True) == []
    verify_sources(manifest, tmp_path, include_runtime_assets=True)


def test_download_requires_trusted_https_and_redirect_target(tmp_path: Path, monkeypatch):
    payload = b"source archive"
    entry = _entry(payload)
    entry["url"] = "http://files.pythonhosted.org/packages/example-1.0.tar.gz"
    with pytest.raises(SourceDistributionError, match="trusted HTTPS"):
        download_sources({"entries": [entry], "blockers": []}, tmp_path)

    entry["url"] = "https://files.pythonhosted.org/packages/example-1.0.tar.gz"
    _mock_urlopen(monkeypatch, payload, "https://evil.example/source.tar.gz")
    with pytest.raises(SourceDistributionError, match="trusted HTTPS"):
        download_sources({"entries": [entry], "blockers": []}, tmp_path)


def test_redirect_handler_rejects_untrusted_intermediate_hop():
    handler = source_tool._TrustedRedirectHandler()
    request = urllib.request.Request(
        "https://files.pythonhosted.org/packages/example-1.0.tar.gz"
    )
    with pytest.raises(SourceDistributionError, match="trusted HTTPS"):
        handler.redirect_request(
            request,
            cast(Any, None),
            302,
            "Found",
            cast(Any, {}),
            "https://evil.example/source.tar.gz",
        )

def test_manifest_records_current_incomplete_status():
    manifest_path = Path(__file__).resolve().parents[2] / "licenses" / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = source_tool.load_manifest(manifest_path)

    assert manifest["status"] in {"complete", "incomplete"}
    assert bool(manifest["blockers"]) is (manifest["status"] == "incomplete")
    assert loaded["project_version"] == "2.0.0"
    assert len(manifest["entries"]) >= 100
    assert loaded["recipes"]
    assert all(
        entry.get("filename")
        for entry in manifest["entries"]
        if entry.get("kind") == "runtime-asset"
    )
    directml = next(
        entry for entry in manifest["entries"] if entry["id"] == "runtime-DirectML.dll"
    )
    assert directml["sha256"] == (
        "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"
    )
    assert directml["asset_sha256"] == (
        "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1"
    )
