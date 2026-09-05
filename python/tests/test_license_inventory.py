"""License collection preserves notices without exposing machine paths."""

import hashlib
import json
from pathlib import Path
import runpy
import tarfile
from types import SimpleNamespace

import pytest

import license_inventory
from license_inventory import copy_notices


def test_same_basename_different_notices_are_preserved(tmp_path: Path):
    files = [tmp_path / "one" / "LICENSE", tmp_path / "two" / "LICENSE"]
    for index, file in enumerate(files):
        file.parent.mkdir()
        file.write_text(f"Copyright holder {index}\n", encoding="utf-8")
    destination = tmp_path / "notices"
    notices = copy_notices(files + files, destination)
    assert len(notices) == 2
    for notice in notices:
        file = tmp_path / notice["file"]
        assert file.is_file()
        assert not Path(notice["file"]).is_absolute()
        assert hashlib.sha256(file.read_bytes()).hexdigest() == notice["sha256"]


def test_spec_bundles_only_current_verified_notices(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path / "build" / "dependency-licenses"
    directory.mkdir(parents=True)
    notice = directory / "current.txt"
    notice.write_bytes(b"Current notice")
    (directory / "stale.txt").write_bytes(b"Old cached notice")
    (directory / "inventory.json").write_text(json.dumps({
        "python": {"notices": [{
            "file": notice.name,
            "sha256": hashlib.sha256(notice.read_bytes()).hexdigest(),
        }]},
        "python_components": [], "rust_components": [], "native_components": [],
    }), encoding="utf-8")
    stages: dict[str, object] = {
        name: lambda *args, **kwargs: SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[])
        for name in ("Analysis", "PYZ", "EXE", "COLLECT")
    }
    stages["SPECPATH"] = str(tmp_path)
    result = runpy.run_path(str(root / "AudioForge.spec"), init_globals=stages)
    bundled = {Path(path).name for path, _destination in result["datas"]}
    assert {"inventory.json", "current.txt"} <= bundled
    assert "stale.txt" not in bundled
    notice.write_bytes(b"Changed after inventory collection")
    with pytest.raises(ValueError, match="digest mismatch"):
        runpy.run_path(str(root / "AudioForge.spec"), init_globals=stages)


def test_source_status_requires_verified_receipt_and_keeps_native_records(monkeypatch):
    manifest = {
        "status": "complete",
        "blockers": [],
        "entries": [{
            "kind": "native-build-source",
            "id": "deepfilter-source",
            "name": "DeepFilterNet",
            "version": "abc123",
            "source_of_truth": "https://github.com/example/project/commit/abc123",
            "url": "https://github.com/example/project/archive/abc123.tar.gz",
        }],
    }
    monkeypatch.setattr(license_inventory, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(license_inventory, "verify_sources", lambda *args, **kwargs: None)
    monkeypatch.setenv("AUDIOFORGE_SOURCE_DIR", "build/source-distribution")
    monkeypatch.setenv("AUDIOFORGE_SOURCE_REVISION", "v2.0.0")

    result = license_inventory.source_distribution_status()

    assert result["status"] == "complete"
    assert result["source_dir"] == "build/source-distribution"
    assert result["native_components"][0]["name"] == "DeepFilterNet"
    assert result["native_components"][0]["notices"] == []
    assert not any(Path(str(value)).is_absolute() for value in result.values() if isinstance(value, str))


def test_source_archive_notice_extraction_uses_declared_members(tmp_path: Path):
    archive_path = tmp_path / "archives" / "native-openssl--native.tar.gz"
    archive_path.parent.mkdir(parents=True)
    member_name = "openssl-3.0.16/LICENSE.txt"
    payload = b"OpenSSL license\n"
    source_file = tmp_path / "license.txt"
    source_file.write_bytes(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        with source_file.open("rb") as handle:
            archive.addfile(info, handle)

    manifest = {
        "entries": [{
            "id": "native-openssl",
            "kind": "native-build-source",
            "license_paths": [member_name],
            "filename": "native.tar.gz",
        }]
    }
    notices = license_inventory._source_archive_notices(
        manifest,
        tmp_path,
        tmp_path / "notices",
    )
    record = notices["native-openssl"][0]
    copied = tmp_path / "notices" / record["file"]
    assert copied.read_bytes() == payload
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
