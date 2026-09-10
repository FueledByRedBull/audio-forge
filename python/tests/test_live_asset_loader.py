from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import evaluate_onnxruntime_probe as probe
import fetch_release_assets
import source_distribution
from verify_release_assets import load_asset_manifest, verify_assets


def _cpu_manifest(tmp_path: Path) -> Path:
    archive_sha256 = "a" * 64
    names = (
        "onnxruntime.dll",
        "onnxruntime.lib",
        "onnxruntime_providers_shared.dll",
    )
    assets = []
    for index, name in enumerate(names, start=1):
        assets.append(
            {
                "path": f"target/onnxruntime-cpu/lib/{name}",
                "size": index,
                "sha256": f"{index:064x}",
                "source": "https://github.com/example/runtime.zip",
                "origin": {
                    "status": "verified-upstream-archive",
                    "archive_sha256": archive_sha256,
                    "archive_size": 123,
                    "archive_member": f"package/{name}",
                    "runtime": "CPU-only Windows x64",
                },
                "license": "MIT",
            }
        )
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(json.dumps({"assets": assets}), encoding="utf-8")
    return manifest


def test_loader_rejects_duplicate_paths_and_invalid_digests(tmp_path: Path) -> None:
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {"path": "runtime.dll", "size": 1, "sha256": "0" * 64},
                    {"path": "runtime.dll", "size": 1, "sha256": "bad"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        load_asset_manifest(manifest)

    message = str(raised.value)
    assert "repeats asset path runtime.dll" in message
    assert "manifest sha256 must be a 64-character hex string" in message


def test_probe_and_fetch_use_live_cpu_manifest_values(tmp_path: Path) -> None:
    manifest_path = _cpu_manifest(tmp_path)
    manifest = load_asset_manifest(manifest_path)

    runtime_files, archive_sha256 = probe._load_cpu_ort_expectations(manifest_path)

    assert archive_sha256 == "a" * 64
    assert runtime_files["onnxruntime.dll"]["size"] == 1
    assert runtime_files["onnxruntime.dll"]["sha256"] == "1".zfill(64)
    assets = fetch_release_assets._manifest_assets(manifest, only_cpu_runtime=True)
    assert [asset["name"] for asset in assets] == [
        "onnxruntime.dll",
        "onnxruntime.lib",
        "onnxruntime_providers_shared.dll",
    ]


def test_verify_assets_keeps_missing_file_error(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {"assets": [{"path": "missing.dll", "size": 1, "sha256": "0" * 64}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("verify_release_assets.REPO_ROOT", tmp_path)

    assert verify_assets(manifest) == ["missing.dll: missing"]


@pytest.mark.parametrize(
    ("status", "blocker_fragment"),
    [
        (
            "inherited-binary-build-identity-unresolved",
            "identity is unresolved",
        ),
        ("license-restricted", "terms are not cleared"),
        ("proprietary", "terms are not cleared"),
        ("distribution-blocked", "terms are not cleared"),
    ],
)
def test_blocked_origin_statuses_feed_source_blockers_but_fail_release_verification(
    tmp_path: Path, monkeypatch, status: str, blocker_fragment: str
) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"blocked")
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "asset.bin",
                        "size": asset.stat().st_size,
                        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                        "origin": {
                            "status": status
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_asset_manifest(manifest).assets[0]["path"] == "asset.bin"
    monkeypatch.setattr(source_distribution, "ROOT", tmp_path)
    entries, blockers = source_distribution._native_asset_entries()
    assert entries[0]["status"] == "blocked"
    assert any(blocker_fragment in blocker for blocker in blockers)

    monkeypatch.setattr("verify_release_assets.REPO_ROOT", tmp_path)
    errors = verify_assets(manifest)
    assert errors == ["asset.bin: origin status is not releasable"]
