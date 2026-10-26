"""Offline checks for corresponding-source manifest hydration."""

import functools
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib
from typing import Any, cast
from unittest.mock import Mock
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


@pytest.fixture(autouse=True)
def parsed_cargo_locks(monkeypatch):
    """Validating a manifest re-reads a Cargo.lock for every crate it lists; read each once."""
    monkeypatch.setattr(source_tool, "_cargo_lock_hashes", functools.cache(source_tool._cargo_lock_hashes))


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


def test_download_deduplicates_identical_source_content(tmp_path: Path, monkeypatch):
    payload = b"same source archive"
    first = _entry(payload)
    second = {
        **_entry(payload),
        "id": "python-example-copy-1.0",
        "filename": "example-copy-1.0.zip",
    }
    manifest = {
        "project": "AudioForge",
        "project_version": "1.12.0",
        "entries": [first, second],
        "blockers": [],
    }
    destination = tmp_path / "hydrated"
    _mock_urlopen(monkeypatch, payload)
    open_source = Mock(wraps=source_tool._open_source_url)
    monkeypatch.setattr(source_tool, "_open_source_url", open_source)

    downloaded = download_sources(manifest, destination)

    assert downloaded == [
        _archive_target(destination, first),
        _archive_target(destination, second),
    ]
    open_source.assert_called_once()
    assert downloaded[0] == downloaded[1]
    assert list((destination / "archives").iterdir()) == [downloaded[0]]
    verify_sources(manifest, destination)
    (destination / "AudioForge-project-source.tar").write_bytes(b"project archive")
    monkeypatch.setattr(source_tool, "_git_text", lambda *args: "0" * 40)
    receipt = json.loads(
        source_tool._write_receipt(
            destination,
            manifest,
            revision="HEAD",
            include_runtime_assets=False,
        ).read_text(encoding="utf-8")
    )
    assert [record["id"] for record in receipt["archives"]] == [
        first["id"],
        second["id"],
    ]
    assert {record["path"] for record in receipt["archives"]} == {
        _archive_target(destination, first).relative_to(destination).as_posix(),
        _archive_target(destination, second).relative_to(destination).as_posix(),
    }


def test_download_refuses_existing_hash_mismatch(tmp_path: Path):
    entry = _entry()
    manifest = {"entries": [entry], "blockers": []}
    target = _archive_target(tmp_path / "hydrated", entry)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tampered")

    with pytest.raises(SourceDistributionError, match="hash mismatch"):
        download_sources(manifest, tmp_path / "hydrated")

    assert target.read_bytes() == b"tampered"


def test_download_refuses_response_over_byte_cap(tmp_path: Path, monkeypatch):
    payload = b"x" * 33
    entry = _entry(payload)
    _mock_urlopen(monkeypatch, payload)
    monkeypatch.setattr(source_tool, "DOWNLOAD_MAX_BYTES", 32)
    target = _archive_target(tmp_path / "hydrated", entry)

    with pytest.raises(SourceDistributionError, match="exceeded 32 bytes"):
        source_tool._download(entry, target)

    assert not target.exists()
    assert not list(target.parent.glob("*.part"))


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
                "project_version": "1.12.0",
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


def test_qt_sources_use_fixed_trusted_mirror():
    runtime = _read_locked_versions(source_tool.ROOT / "requirements" / "runtime.txt")
    entries = source_tool._expected_static_source_entries(runtime)
    qt_entries = [
        entry for identifier, entry in entries.items()
        if identifier.startswith(("qt-", "python-pyside6-", "python-shiboken6-"))
    ]
    assert len(qt_entries) == 7
    for entry in qt_entries:
        assert entry["url"].startswith("https://qt.mirror.constant.com/")
        assert entry["source_of_truth"].startswith("https://download.qt.io/")


@pytest.mark.parametrize("name", source_tool.PYSIDE_PACKAGES)
def test_pyside_mapping_rejects_missing_or_different_wheel_versions(name):
    runtime = {package.casefold(): "6.11.1" for package in source_tool.PYSIDE_PACKAGES}
    for wrong in (None, "6.11.0", "6.12.0"):
        changed = dict(runtime)
        if wrong is None:
            del changed[name.casefold()]
        else:
            changed[name.casefold()] = wrong
        with pytest.raises(SourceDistributionError, match="verified source mapping"):
            source_tool._pyside_source_entries(changed)


@pytest.mark.parametrize("field", ["sha256", "url", "source_of_truth"])
def test_release_manifest_rejects_changed_pyside_archive_identity(field):
    manifest = json.loads(source_tool.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["entries"] if item["name"] == "PySide6-Essentials")
    entry[field] = "0" * 64 if field == "sha256" else "https://download.qt.io/changed.tar.xz"
    with pytest.raises(SourceDistributionError, match="stale: python-pyside6-"):
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


def _rewrite_tar(archive_bytes: bytes, change: str) -> bytes:
    """Re-encode a tar with new timestamps and owners, plus an optional content change."""
    output = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as source,
        tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as target,
    ):
        root = source.getnames()[0]
        for member in source.getmembers():
            if change == "missing" and member.name == f"{root}/LICENSE":
                continue
            body = source.extractfile(member) if member.isfile() else None
            data = body.read() if body else None
            if change == "changed" and member.name == f"{root}/pyproject.toml":
                data = (data or b"") + b"# tampered\n"
                member.size = len(data)
            member.mtime, member.uid, member.gid, member.uname, member.gname = 1, 7, 7, "x", "x"
            target.addfile(member, io.BytesIO(data) if data is not None else None)
        if change == "extra":
            extra = tarfile.TarInfo(f"{root}/extra.txt")
            extra.size = 1
            target.addfile(extra, io.BytesIO(b"x"))
    return output.getvalue()


@pytest.mark.parametrize("change", ["same", "changed", "extra", "missing"])
def test_validate_project_archive_compares_contents_not_bytes(tmp_path: Path, change: str):
    revision = source_tool._git_text("rev-parse", "--verify", "HEAD^{commit}")
    project_version = tomllib.loads(
        source_tool._git_text("show", f"{revision}:pyproject.toml")
    )["project"]["version"]
    genuine = source_tool._git_archive_bytes(revision, project_version)
    rewritten = _rewrite_tar(genuine, change)
    assert rewritten != genuine
    archive_path = tmp_path / "AudioForge-project-source.tar"
    archive_path.write_bytes(rewritten)

    def validate() -> str:
        return source_tool._validate_project_archive(
            archive_path, {"project_version": project_version}, revision
        )

    if change == "same":
        assert validate() == hashlib.sha256(rewritten).hexdigest()
    else:
        with pytest.raises(SourceDistributionError, match="does not match the recorded git revision"):
            validate()


def test_source_receipt_binds_manifest_and_revision(tmp_path: Path, monkeypatch):
    payload = b"source archive"
    entry = _entry(payload)
    manifest = {
        "project": "AudioForge",
        "project_version": "1.12.0",
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
        revision="v1.12.0",
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


# Crates Cargo.lock lists that `cargo metadata --filter-platform x86_64-pc-windows-msvc --features
# extension-module` leaves out: other operating systems, wasm, and features the extension does not
# turn on. Cargo.lock cannot say which those are, so a crate the lock gains that Windows does not
# build is listed here once; a crate Windows does build must appear in the source manifest instead.
NOT_BUILT_FOR_WINDOWS = {line.strip() for line in """
    alsa 0.11.0
    alsa-sys 0.4.0
    bitflags 2.11.0
    block2 0.6.2
    bumpalo 3.20.2
    bytes 1.11.1
    cesu8 1.1.0
    combine 4.6.7
    coreaudio-rs 0.14.2
    dispatch2 0.3.0
    equivalent 1.0.2
    futures-core 0.3.32
    futures-task 0.3.32
    futures-util 0.3.32
    hashbrown 0.16.1
    indexmap 2.13.0
    jni 0.21.1
    jni-sys 0.3.0
    js-sys 0.3.90
    log 0.4.29
    mach2 0.5.0
    memchr 2.8.0
    ndk 0.9.0
    ndk-context 0.1.1
    ndk-sys 0.6.0+11769913
    num-derive 0.4.2
    num_enum 0.7.5
    num_enum_derive 0.7.5
    objc2 0.6.3
    objc2-audio-toolbox 0.3.2
    objc2-avf-audio 0.3.2
    objc2-core-audio 0.3.2
    objc2-core-audio-types 0.3.2
    objc2-core-foundation 0.3.2
    objc2-encode 4.1.0
    objc2-foundation 0.3.2
    pkg-config 0.3.32
    proc-macro-crate 3.4.0
    redox_syscall 0.5.18
    rustversion 1.0.22
    same-file 1.0.6
    serde_derive 1.0.228
    slab 0.4.12
    thread-priority 1.2.0
    toml_datetime 0.7.5+spec-1.1.0
    toml_edit 0.23.10+spec-1.0.0
    toml_parser 1.0.9+spec-1.1.0
    walkdir 2.5.0
    wasm-bindgen 0.2.113
    wasm-bindgen-futures 0.4.63
    wasm-bindgen-macro 0.2.113
    wasm-bindgen-macro-support 0.2.113
    wasm-bindgen-shared 0.2.113
    web-sys 0.3.90
    winapi 0.3.9
    winapi-i686-pc-windows-gnu 0.4.0
    winapi-util 0.1.11
    winapi-x86_64-pc-windows-gnu 0.4.0
    windows-sys 0.45.0
    windows-sys 0.61.2
    windows-targets 0.42.2
    windows_aarch64_gnullvm 0.42.2
    windows_aarch64_msvc 0.42.2
    windows_i686_gnu 0.42.2
    windows_i686_msvc 0.42.2
    windows_x86_64_gnu 0.42.2
    windows_x86_64_gnullvm 0.42.2
    windows_x86_64_msvc 0.42.2
    winnow 0.7.14
""".strip().splitlines()}


def _answer_cargo_metadata_from_the_lock(monkeypatch) -> None:
    """Answer `cargo metadata` from Cargo.lock minus NOT_BUILT_FOR_WINDOWS, so no Rust toolchain installs.

    A crate added to the lock, dropped from it, re-versioned or given another checksum then fails
    the manifest check as it would with the real resolution.
    """
    lock = tomllib.loads((source_tool.ROOT / "Cargo.lock").read_text(encoding="utf-8"))["package"]
    packages = [
        {
            "id": f"{package['name']} {package['version']}", "name": package["name"],
            "version": package["version"], "source": package.get("source"), "license": None,
        }
        for package in lock
        if f"{package['name']} {package['version']}" not in NOT_BUILT_FOR_WINDOWS
    ]
    metadata = json.dumps({"packages": packages, "resolve": {"nodes": [{"id": package["id"]} for package in packages]}})
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        if list(command[:2]) != ["cargo", "metadata"]:
            return real_run(command, *args, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=metadata)

    monkeypatch.setattr(source_tool.subprocess, "run", run)


def test_manifest_records_current_incomplete_status(monkeypatch):
    manifest_path = Path(__file__).resolve().parents[2] / "licenses" / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _answer_cargo_metadata_from_the_lock(monkeypatch)
    try:
        loaded = source_tool.load_manifest(manifest_path)
    except SourceDistributionError as error:
        pytest.fail(
            f"{error}. If Cargo.lock gained or re-versioned a crate that Windows does not build, list it in "
            "NOT_BUILT_FOR_WINDOWS; otherwise regenerate licenses/source-manifest.json."
        )

    assert manifest["status"] in {"complete", "incomplete"}
    assert bool(manifest["blockers"]) is (manifest["status"] == "incomplete")
    assert loaded["project_version"] == source_tool._read_project_version()
    assert len(manifest["entries"]) >= 100
    assert loaded["recipes"]
    assert all(
        entry.get("filename")
        for entry in manifest["entries"]
        if entry.get("kind") == "runtime-asset"
    )
    ort = next(
        entry for entry in manifest["entries"] if entry["id"] == "runtime-onnxruntime.dll"
    )
    assets = json.loads((manifest_path.parents[1] / "release-assets.json").read_text(encoding="utf-8"))["assets"]
    runtime = next(asset for asset in assets if asset["path"] == "target/onnxruntime-cpu/lib/onnxruntime.dll")
    assert ort["sha256"] == runtime["origin"]["archive_sha256"]
    assert ort["asset_sha256"] == runtime["sha256"]
    assert not any(entry["id"].startswith("runtime-DirectML") for entry in manifest["entries"])
    assert any(entry["id"] == "onnxruntime-source-a83fc4d58cb4" for entry in manifest["entries"])


@pytest.mark.parametrize("damaged", [None, "checksum", "archive", "commit", "version", "source_manifest_sha256", "source_receipt_sha256", "project_source_sha256", "project_source_name"])
def test_source_archive_sidecars_bind_every_release_input(tmp_path, monkeypatch, damaged):
    archive = tmp_path / "source.7z"
    checksum = tmp_path / "source.7z.sha256"
    metadata_path = tmp_path / "source.7z.metadata.json"
    manifest_path = tmp_path / "manifest.json"
    receipt = tmp_path / "source-receipt.json"
    project = tmp_path / "AudioForge-project-source.tar"
    for path in (archive, manifest_path, receipt, project):
        path.write_bytes(path.name.encode())
    digest = source_tool._sha256(archive)
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    metadata = {
        "archive": {"name": archive.name, "sha256": digest},
        "commit": "a" * 40, "version": "1.12.0",
        "source_manifest_sha256": source_tool._sha256(manifest_path),
        "source_receipt_sha256": source_tool._sha256(receipt),
        "project_source_sha256": source_tool._sha256(project),
        "project_source_name": project.name,
    }
    monkeypatch.setattr(source_tool, "_git_text", lambda *args: "a" * 40)
    if damaged == "checksum":
        checksum.write_text(f"{digest}  wrong-name.7z\n", encoding="utf-8")
    elif damaged:
        metadata[damaged] = "wrong"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8-sig")
    args = (archive, checksum, metadata_path, manifest_path, tmp_path)
    if damaged:
        with pytest.raises(SourceDistributionError):
            source_tool.verify_archive_sidecars(*args, version="1.12.0", revision="HEAD")
    else:
        source_tool.verify_archive_sidecars(*args, version="1.12.0", revision="HEAD")


def test_archive_cli_requires_all_sidecars_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(source_tool, "load_manifest", lambda path: {})
    assert source_tool.main(["verify", "--archive", str(tmp_path / "source.7z")]) == 2
