"""Tests for packaging smoke and release asset verification helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).parent.parent / "tools"


def _load_tool(name: str):
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package_smoke = _load_tool("package_smoke")
verify_release_assets = _load_tool("verify_release_assets")


def _write_bundle_file(bundle: Path, relative_path: str) -> None:
    path = bundle / "_internal" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_package_smoke_rejects_empty_models_directory(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "_internal" / "models").mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    _write_bundle_file(bundle, "df.dll")
    _write_bundle_file(bundle, "DirectML.dll")
    _write_bundle_file(bundle, "mic_eq_core.cp312-win_amd64.pyd")
    (bundle / "_internal" / "example.dist-info").mkdir()

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("DeepFilterNet3_ll_onnx.tar.gz" in error for error in errors)
    assert any("DeepFilterNet3_onnx.tar.gz" in error for error in errors)
    assert any("silero_vad.onnx" in error for error in errors)


def test_package_smoke_accepts_required_assets_and_metadata(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, "mic_eq_core.cp312-win_amd64.pyd")
    (bundle / "_internal" / "example.dist-info").mkdir()

    assert package_smoke.check_dist_bundle(bundle) == []


def test_package_smoke_rejects_bundle_without_license_metadata(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, "mic_eq_core.cp312-win_amd64.pyd")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("dependency dist-info" in error for error in errors)


def test_verify_release_assets_reports_missing_and_hash_mismatch(tmp_path, monkeypatch):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"actual")
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "asset.bin",
                        "size": 6,
                        "sha256": "0" * 64,
                    },
                    {
                        "path": "missing.bin",
                        "sha256": "0" * 64,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)

    errors = verify_release_assets.verify_assets(manifest)

    assert any("sha256 mismatch" in error for error in errors)
    assert any("missing.bin: missing" in error for error in errors)
