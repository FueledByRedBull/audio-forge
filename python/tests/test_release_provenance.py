"""Tests for exact-artifact release provenance."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).parent.parent / "tools" / "release_provenance.py"
SPEC = importlib.util.spec_from_file_location("release_provenance", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
release_provenance = importlib.util.module_from_spec(SPEC)
sys.modules["release_provenance"] = release_provenance
SPEC.loader.exec_module(release_provenance)


@pytest.fixture(autouse=True)
def _clean_source_tree(monkeypatch):
    monkeypatch.setattr(release_provenance, "_git_is_dirty", lambda: False)


def _bundle(root: Path) -> Path:
    bundle = root / "AudioForge"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"exe")
    (bundle / "_internal" / "asset.bin").write_bytes(b"asset")
    return bundle


def test_git_commit_is_derived_from_head_and_cross_checks_workflow_sha(
    monkeypatch,
) -> None:
    head = "a" * 40
    monkeypatch.setattr(release_provenance, "_git_head", lambda: head)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    assert release_provenance._git_commit() == head

    monkeypatch.setenv("GITHUB_SHA", head.upper())
    assert release_provenance._git_commit() == head

    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    with pytest.raises(RuntimeError, match="does not match"):
        release_provenance._git_commit()


def test_bundle_manifest_is_normalized_and_deterministic(tmp_path):
    bundle = _bundle(tmp_path)

    first = release_provenance.build_bundle_manifest(bundle)
    second = release_provenance.build_bundle_manifest(bundle)

    assert first == second
    assert [entry["path"] for entry in first["files"]] == [
        "_internal/asset.bin",
        "AudioForge.exe",
    ]
    assert first["file_count"] == 2
    assert first["total_bytes"] == 8


def test_path_baseline_reports_additions_and_removals(tmp_path):
    manifest = release_provenance.build_bundle_manifest(_bundle(tmp_path))
    baseline = {"schema_version": 1, "paths": ["AudioForge.exe", "old.dll"]}

    additions, removals = release_provenance.compare_path_baseline(
        manifest, baseline
    )

    assert additions == ["_internal/asset.bin"]
    assert removals == ["old.dll"]


def test_create_and_verify_sidecars_bind_exact_archive_and_bundle(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)

    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path
    )

    assert (
        release_provenance.verify_sidecars(
            archive,
            checksum,
            manifest,
            metadata,
            bundle=bundle,
        )
        == []
    )
    metadata_json = json.loads(metadata.read_text(encoding="utf-8"))
    assert metadata_json["archive"]["name"] == archive.name
    assert metadata_json["bundle"]["file_count"] == 2
    assert metadata_json["commit"] == "a" * 40
    assert metadata_json["source_dirty"] is False


def test_sidecar_creation_rejects_dirty_source_unless_explicitly_local(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(release_provenance, "_git_is_dirty", lambda: True)

    with pytest.raises(ValueError, match="refuses a dirty source tree"):
        release_provenance.create_sidecars(bundle, archive, tmp_path)

    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path, allow_dirty=True
    )
    assert json.loads(metadata.read_text(encoding="utf-8"))["source_dirty"] is True
    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        expected_commit="a" * 40,
    )
    assert any("cannot be promoted" in error for error in errors)


def test_verifier_accepts_exact_files_under_a_different_extraction_root(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)
    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path
    )
    extracted = tmp_path / "audioforge-candidate"
    shutil.copytree(bundle, extracted)

    assert (
        release_provenance.verify_sidecars(
            archive,
            checksum,
            manifest,
            metadata,
            bundle=extracted,
        )
        == []
    )


def test_verifier_rejects_changed_archive_and_extracted_bundle(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)
    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path
    )
    archive.write_bytes(b"tampered")
    (bundle / "_internal" / "asset.bin").write_bytes(b"tampered")

    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
    )

    assert any("checksum sidecar" in error for error in errors)
    assert any("archive SHA-256" in error for error in errors)
    assert any("per-file manifest" in error for error in errors)


def test_verifier_binds_promotion_digest_commit_and_reports(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)
    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path
    )
    report = tmp_path / "qualification.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "passed": True,
                "commit": "a" * 40,
                "artifact": {"sha256": release_provenance.sha256_file(archive)},
            }
        ),
        encoding="utf-8",
    )

    assert (
        release_provenance.verify_sidecars(
            archive,
            checksum,
            manifest,
            metadata,
            bundle=bundle,
            expected_archive_sha256=release_provenance.sha256_file(archive),
            expected_commit="a" * 40,
            reports=[report],
        )
        == []
    )

    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "artifact": {"sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        expected_archive_sha256="c" * 64,
        expected_commit="d" * 40,
        reports=[report],
    )

    assert any("promotion SHA-256" in error for error in errors)
    assert any("release tag commit" in error for error in errors)
    assert any("different release artifact" in error for error in errors)
    assert any("not a passing qualification report" in error for error in errors)


def test_verifier_accepts_hardware_report_archive_hash_shape(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)
    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path
    )
    report = tmp_path / "hardware.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "passed",
                "passed": True,
                "qualification_kind": "exact-artifact-hardware",
                "artifact": {
                    "archive_sha256": release_provenance.sha256_file(archive)
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        release_provenance.verify_sidecars(
            archive,
            checksum,
            manifest,
            metadata,
            bundle=bundle,
            reports=[report],
        )
        == []
    )


def test_verifier_requires_both_pass_status_and_boolean(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)
    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle, archive, tmp_path
    )
    report = tmp_path / "qualification.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": False,
                "artifact": {"sha256": release_provenance.sha256_file(archive)},
            }
        ),
        encoding="utf-8",
    )

    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        reports=[report],
    )
    assert any("not a passing qualification report" in error for error in errors)
