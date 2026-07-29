"""Tests for packaging smoke and release asset verification helpers."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


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
prune_bundle = _load_tool("prune_bundle")
verify_release_assets = _load_tool("verify_release_assets")
fetch_release_assets = _load_tool("fetch_release_assets")
check_versions = _load_tool("check_versions")
run_semgrep = _load_tool("run_semgrep")


def _write_bundle_file(bundle: Path, relative_path: str) -> None:
    path = bundle / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _write_valid_build_info(bundle: Path) -> None:
    path = bundle / "_internal" / "audioforge-build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "version": package_smoke._expected_version()}),
        encoding="utf-8",
    )


def test_package_smoke_source_packaging_checks_pass():
    assert package_smoke.check_source_packaging() == []


def test_semgrep_gate_uses_rule_default_severity_when_result_omits_level(
    tmp_path,
):
    sarif = tmp_path / "results.sarif"
    sarif.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "rules": [
                                    {
                                        "id": "error-rule",
                                        "defaultConfiguration": {"level": "error"},
                                    },
                                    {
                                        "id": "warning-rule",
                                        "defaultConfiguration": {"level": "warning"},
                                    },
                                ]
                            }
                        },
                        "results": [
                            {"ruleId": "error-rule"},
                            {"ruleId": "warning-rule"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert run_semgrep._error_findings(sarif) == ["error-rule"]


def test_package_smoke_rejects_empty_models_directory(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "_internal" / "models").mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    _write_bundle_file(bundle, "_internal/df.dll")
    _write_bundle_file(bundle, "_internal/DirectML.dll")
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd")
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
    _write_valid_build_info(bundle)
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd")

    assert package_smoke.check_dist_bundle(bundle) == []


def test_package_smoke_rejects_duplicate_native_extension(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd")
    _write_bundle_file(bundle, "_internal/mic_eq_core/mic_eq_core.cp312-win_amd64.pyd")
    (bundle / "_internal" / "example.dist-info").mkdir()

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("_internal/mic_eq_core/mic_eq_core*.pyd" in error for error in errors)


def test_prune_bundle_removes_duplicate_native_extension_only_when_packaged_copy_exists(
    tmp_path,
):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd")
    _write_bundle_file(bundle, "_internal/mic_eq_core/mic_eq_core.cp312-win_amd64.pyd")

    prune_bundle.prune_bundle(bundle)

    assert (
        bundle / "_internal" / "mic_eq" / "mic_eq_core.cp312-win_amd64.pyd"
    ).is_file()
    assert not (bundle / "_internal" / "mic_eq_core").exists()


def test_prune_bundle_keeps_top_level_native_extension_without_packaged_copy(tmp_path):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, "_internal/mic_eq_core/mic_eq_core.cp312-win_amd64.pyd")

    prune_bundle.prune_bundle(bundle)

    assert (
        bundle / "_internal" / "mic_eq_core" / "mic_eq_core.cp312-win_amd64.pyd"
    ).is_file()


def test_package_smoke_rejects_misplaced_decoy_assets(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        decoy_path = bundle / "_internal" / "decoys" / Path(relative_path).name
        decoy_path.parent.mkdir(parents=True, exist_ok=True)
        decoy_path.write_bytes(b"x")
    _write_bundle_file(bundle, "_internal/decoys/mic_eq_core.cp312-win_amd64.pyd")
    (bundle / "_internal" / "example.dist-info").mkdir()

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("_internal/df.dll" in error for error in errors)
    assert any("_internal/mic_eq/mic_eq_core*.pyd" in error for error in errors)


def test_package_smoke_rejects_bundle_without_required_license_notice(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        if relative_path == "_internal/licenses/DirectML-LICENSE.txt":
            continue
        _write_bundle_file(bundle, relative_path)
    _write_valid_build_info(bundle)
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("DirectML-LICENSE.txt" in error for error in errors)


def test_package_smoke_rejects_stale_bundle_version(tmp_path):
    bundle = tmp_path / "AudioForge"
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES:
        _write_bundle_file(bundle, relative_path)
    build_info = bundle / "_internal" / "audioforge-build.json"
    build_info.write_text(
        json.dumps({"schema_version": 1, "version": "0.0.0"}),
        encoding="utf-8",
    )
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("reports version '0.0.0'" in error for error in errors)


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


def test_verify_release_assets_rejects_absolute_and_traversal_paths(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": str(tmp_path / "asset.bin"),
                        "sha256": "0" * 64,
                    },
                    {
                        "path": "../asset.bin",
                        "sha256": "0" * 64,
                    },
                    {
                        "path": "asset.bin",
                        "bundle_path": "../bundle.bin",
                        "sha256": "0" * 64,
                    },
                    {
                        "path": r"C:\tmp\asset.bin",
                        "sha256": "0" * 64,
                    },
                    {
                        "path": r"models\..\asset.bin",
                        "sha256": "0" * 64,
                    },
                    {
                        "path": "asset.bin",
                        "bundle_path": r"models\..\bundle.bin",
                        "sha256": "0" * 64,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)

    errors = verify_release_assets.verify_assets(manifest)

    assert any("absolute path" in error for error in errors)
    assert any("'..' traversal" in error for error in errors)
    assert any("bundle_path" in error and "'..' traversal" in error for error in errors)
    assert any(
        r"C:\tmp\asset.bin" in error and "absolute path" in error for error in errors
    )
    assert any(r"models\..\asset.bin" in error for error in errors)
    assert any(r"models\..\bundle.bin" in error for error in errors)


def test_fetch_release_assets_direct_download_writes_response(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_release_assets.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(b"pinned-model"),
    )
    destination = tmp_path / "silero_vad.onnx"

    fetch_release_assets._download_direct_url(
        "https://example.invalid/silero_vad.onnx",
        destination,
    )

    assert destination.read_bytes() == b"pinned-model"


def test_fetch_release_assets_default_tag_comes_from_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps({"fallback_release_tag": "v9.8.7", "assets": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_release_assets, "MANIFEST_PATH", manifest)

    assert fetch_release_assets._default_asset_source_tag() == "v9.8.7"


def test_version_check_rejects_stale_readme_hydration_tag(tmp_path, monkeypatch):
    (tmp_path / "python" / "tools").mkdir(parents=True)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "release-assets.json").write_text(
        json.dumps({"fallback_release_tag": "v1.10.0"}),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "The fallback release is pinned once in `release-assets.json`.\n"
        "fetch_release_assets.py --release-tag v1.8.0\n",
        encoding="utf-8",
    )
    (tmp_path / "RELEASING.md").write_text(
        "fetch_release_assets.py\n",
        encoding="utf-8",
    )
    (tmp_path / "python" / "tools" / "fetch_release_assets.py").write_text(
        "default=_default_asset_source_tag()\n",
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows" / "release-package.yml").write_text(
        ").fallback_release_tag\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_versions, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="stale fallback tag"):
        check_versions._check_release_asset_hydration()


def test_version_check_rejects_stale_releasing_hydration_tag(
    tmp_path, monkeypatch
):
    (tmp_path / "python" / "tools").mkdir(parents=True)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "release-assets.json").write_text(
        json.dumps({"fallback_release_tag": "v1.10.0"}),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "The fallback release is pinned once in `release-assets.json`.\n",
        encoding="utf-8",
    )
    (tmp_path / "RELEASING.md").write_text(
        "fetch_release_assets.py --release-tag v1.8.0\n",
        encoding="utf-8",
    )
    (tmp_path / "python" / "tools" / "fetch_release_assets.py").write_text(
        "default=_default_asset_source_tag()\n",
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows" / "release-package.yml").write_text(
        ").fallback_release_tag\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_versions, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="RELEASING.md.*stale fallback tag"):
        check_versions._check_release_asset_hydration()
