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
    inventory = bundle / "_internal" / "licenses" / "dependencies" / "inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_distribution": {"status": "complete", "blockers": []},
            }
        ),
        encoding="utf-8",
    )
    return bundle


def _producer() -> dict[str, str]:
    return {
        "repository": "owner/repo",
        "workflow": "Release package",
        "run_id": "123",
        "run_attempt": "1",
        "event": "push",
        "head_sha": "a" * 40,
        "ref": "refs/tags/v1.2.3",
    }


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
        "_internal/licenses/dependencies/inventory.json",
        "AudioForge.exe",
    ]
    assert first["file_count"] == 3
    assert first["total_bytes"] == 92


def test_path_baseline_reports_additions_and_removals(tmp_path):
    manifest = release_provenance.build_bundle_manifest(_bundle(tmp_path))
    baseline = {"schema_version": 1, "paths": ["AudioForge.exe", "old.dll"]}

    additions, removals = release_provenance.compare_path_baseline(
        manifest, baseline
    )

    assert additions == [
        "_internal/asset.bin",
        "_internal/licenses/dependencies/inventory.json",
    ]
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
    assert metadata_json["bundle"]["file_count"] == 3
    assert metadata_json["commit"] == "a" * 40
    assert metadata_json["source_dirty"] is False


def test_sidecars_bind_deepfilter_attestation_to_candidate(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    dll = bundle / "_internal" / "df.dll"
    dll.write_bytes(b"source-built-df")

    monkeypatch.setattr(release_provenance, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release_provenance, "_project_version", lambda: "1.2.3")
    monkeypatch.setattr(release_provenance, "_git_commit", lambda: "a" * 40)

    for relative in release_provenance.DEEPFILTER_RECIPE_FILES:
        if relative.startswith("models/"):
            path = bundle / "_internal" / relative
        else:
            path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"recipe-line-1\r\nrecipe-line-2\r\n"
            if relative == "build_deepfilter.ps1"
            else relative.encode("utf-8")
        )
    provenance = {
        "schema_version": 1,
        "upstream": {
            "repository": "https://github.com/Rikorose/DeepFilterNet.git",
            "commit": "b" * 40,
        },
        "build": {
            "target": "x86_64-pc-windows-msvc",
            "profile": "release-lto",
            "features": ["capi"],
            "default_features": False,
            "required_exports": ["df_create"],
            "tested_rust": "1.94.0 (x86_64-pc-windows-msvc; LLVM 21.1.8)",
        },
        "tract_linalg_patch": {
            "archive_sha256": "a" * 64,
            "patched_cargo_toml_sha256": "b" * 64,
            "build_rs_sha256": "c" * 64,
        },
    }
    (tmp_path / "build-support/deepfilter/provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (tmp_path / "release-assets.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "df.dll",
                        "origin": {
                            "status": "verified-source-build",
                            "repository": provenance["upstream"]["repository"],
                            "commit": provenance["upstream"]["commit"],
                        },
                    },
                    *[
                        {
                            "path": relative,
                            "sha256": release_provenance.sha256_file(
                                bundle / "_internal" / relative
                            ),
                        }
                        for relative in release_provenance.DEEPFILTER_RECIPE_FILES
                        if relative.startswith("models/")
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    attestation = tmp_path / "AudioForge-v1.2.3-deepfilter.provenance.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "audioforge.deepfilter.build",
                "output": {
                    "name": "df.dll",
                    "bytes": dll.stat().st_size,
                    "sha256": release_provenance.sha256_file(dll),
                },
                "source": provenance["upstream"],
                "recipe": {
                    "files": {
                        relative: release_provenance._deepfilter_recipe_sha256(
                            bundle / "_internal" / relative
                            if relative.startswith("models/")
                            else tmp_path / relative
                        )
                        for relative in release_provenance.DEEPFILTER_RECIPE_FILES
                    },
                    "target": provenance["build"]["target"],
                    "profile": provenance["build"]["profile"],
                        "features": provenance["build"]["features"],
                        "default_features": provenance["build"]["default_features"],
                        "tract_linalg_archive_sha256": provenance["tract_linalg_patch"]["archive_sha256"],
                        "tract_linalg_patched_manifest_sha256": provenance["tract_linalg_patch"]["patched_cargo_toml_sha256"],
                        "tract_linalg_build_rs_sha256": provenance["tract_linalg_patch"]["build_rs_sha256"],
                    },
                    "toolchain": {"rustc": "rustc 1.94.0 (test)"},
                    "abi": {"required_exports": provenance["build"]["required_exports"]},
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "AudioForge-v1.2.3-win64-ultra.7z"
    archive.write_bytes(b"archive")
    checksum, manifest, metadata = release_provenance.create_sidecars(
        bundle,
        archive,
        tmp_path,
        native_attestation=attestation,
    )

    assert release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        native_attestation=attestation,
    ) == []
    metadata_json = json.loads(metadata.read_text(encoding="utf-8"))
    assert metadata_json["native_attestation"]["deepfilter"]["name"] == attestation.name

    dll.write_bytes(b"tampered")
    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        native_attestation=attestation,
    )
    assert any("output hash" in error for error in errors)


def test_source_distribution_completion_is_only_required_for_publication(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path)
    inventory_path = bundle / "_internal" / "licenses" / "dependencies" / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_distribution": {
                    "status": "pending",
                    "blockers": ["inherited native artifact source is unresolved"],
                },
            }
        ),
        encoding="utf-8",
    )
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
    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        require_source_distribution=True,
    )
    assert any("publication is blocked" in error for error in errors)


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
                "qualification_kind": "exact-artifact-package",
                "status": "passed",
                "passed": True,
                "commit": "a" * 40,
                "artifact": {"sha256": release_provenance.sha256_file(archive)},
                "producer": _producer(),
                "checks": {
                    "provenance": "passed",
                    "package_smoke": "passed",
                    "hidden_exe_startup": "passed",
                    "installer_provenance": "passed",
                    "installer_smoke": "passed",
                    "installer_upgrade": "passed",
                    "source_distribution": "passed",
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
            expected_archive_sha256=release_provenance.sha256_file(archive),
            expected_commit="a" * 40,
            reports=[report],
        )
        == []
    )

    report_data = json.loads(report.read_text(encoding="utf-8"))
    del report_data["checks"]["source_distribution"]
    report.write_text(json.dumps(report_data), encoding="utf-8")
    incomplete_errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        reports=[report],
    )
    assert any("qualification checks are incomplete" in error for error in incomplete_errors)

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
                "schema_version": 3,
                "status": "passed",
                "passed": True,
                "qualification_kind": "exact-artifact-hardware",
                "artifact": {
                    "archive_sha256": release_provenance.sha256_file(archive)
                },
                "producer": _producer(),
                "source_revision": "a" * 40,
                "case": {
                    "id": "win11-virtual-baseline",
                    "device_class": "virtual",
                    "nominal_sample_rate_hz": 48_000,
                    "scenario": "baseline",
                    "evidence_kind": "automated",
                    "scenario_evidence_valid": True,
                },
                "machine": {"release": "11"},
                "requested_health_duration_seconds": 1800.0,
                "package_smoke": {"passed": True},
                "executable_startup": {"passed": True},
                "model_discovery": {"passed": True},
                "selected_route_correlation": {"passed": True},
                "sustained_health": {"passed": True},
                "routes": {
                    "correlation": {
                        "input": "device-0123456789abcdef",
                        "output": "device-fedcba9876543210",
                    },
                    "sustained_health": {
                        "input": "device-0123456789abcdef",
                        "output": "device-fedcba9876543210",
                    },
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


def test_verifier_checks_matrix_source_report_hashes_and_schema(
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

    source_root = tmp_path / "case-reports"
    source = source_root / "run-123" / "release-hardware-qualification.json"
    source.parent.mkdir(parents=True)
    source_report = {
        "schema_version": 3,
        "qualification_kind": "exact-artifact-hardware",
        "status": "passed",
        "passed": True,
        "source_revision": "a" * 40,
        "artifact": {"archive_sha256": release_provenance.sha256_file(archive)},
        "producer": _producer(),
        "case": {
            "id": "win11-virtual-baseline",
            "device_class": "virtual",
            "nominal_sample_rate_hz": 48_000,
            "scenario": "baseline",
            "evidence_kind": "automated",
            "scenario_evidence_valid": True,
        },
        "machine": {"release": "11"},
        "requested_health_duration_seconds": 1800,
        "package_smoke": {"passed": True},
        "executable_startup": {"passed": True},
        "model_discovery": {"passed": True},
        "selected_route_correlation": {"passed": True},
        "sustained_health": {"passed": True},
        "routes": {
            "correlation": {
                "input": "device-0123456789abcdef",
                "output": "device-fedcba9876543210",
            },
            "sustained_health": {
                "input": "device-0123456789abcdef",
                "output": "device-fedcba9876543210",
            },
        },
    }
    source.write_text(json.dumps(source_report), encoding="utf-8")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "qualification_kind": "exact-artifact-hardware-matrix",
                "status": "passed",
                "passed": True,
                "source_revision": "a" * 40,
                "artifact": {
                    "archive_sha256": release_provenance.sha256_file(archive)
                },
                "producer": _producer(),
                "coverage": {
                    "missing": {
                        "automated_baseline_cases": 0,
                        "os_releases": [],
                        "device_classes": [],
                        "nominal_sample_rates_hz": [],
                        "scenarios": [],
                    },
                    "required": {
                        "required_os_releases": ["10", "11"],
                        "required_device_classes": ["built_in", "usb", "virtual"],
                        "required_nominal_sample_rates_hz": [44_100, 48_000],
                        "required_scenarios": [
                            "baseline",
                            "default_device_change",
                            "device_reconnect",
                            "model_configuration_change",
                            "sleep_resume",
                        ],
                    },
                },
                "cases": [
                    {
                        "id": "win11-virtual-baseline",
                        "report_file": "run-123/release-hardware-qualification.json",
                        "report_sha256": release_provenance.sha256_file(source),
                        "os_release": "11",
                        "device_class": "virtual",
                        "nominal_sample_rate_hz": 48_000,
                        "scenario": "baseline",
                        "evidence_kind": "automated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    forged_errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        expected_commit="a" * 40,
        reports=[matrix],
        matrix_report_root=source_root,
    )
    assert any("coverage.missing" in error for error in forged_errors)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    errors = release_provenance.verify_sidecars(
        archive,
        checksum,
        manifest,
        metadata,
        bundle=bundle,
        expected_commit="a" * 40,
        reports=[matrix],
        matrix_report_root=source_root,
    )
    assert any("source report hash" in error for error in errors)


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
