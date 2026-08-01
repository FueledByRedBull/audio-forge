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


prune_bundle = _load_tool("prune_bundle")
package_smoke = _load_tool("package_smoke")
verify_release_assets = _load_tool("verify_release_assets")
fetch_release_assets = _load_tool("fetch_release_assets")
check_versions = _load_tool("check_versions")
run_semgrep = _load_tool("run_semgrep")
check_workflows = _load_tool("check_workflows")


def _write_bundle_file(bundle: Path, relative_path: str) -> None:
    path = bundle / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _write_valid_build_info(bundle: Path, *, version: str | None = None) -> None:
    path = bundle / "_internal" / "audioforge-build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "version": version or package_smoke._expected_version(),
            }
        ),
        encoding="utf-8",
    )


def test_package_smoke_source_packaging_checks_pass():
    assert package_smoke.check_source_packaging() == []


def test_workflow_action_parser_covers_inline_and_named_steps():
    source = (
        "      - uses: actions/checkout@" + "a" * 40 + "\n"
        "      - name: Audit\n"
        "        uses: rustsec/audit-check@" + "b" * 40 + "\n"
    )

    assert check_workflows.ACTION_REF.findall(source) == [
        ("actions/checkout", "a" * 40),
        ("rustsec/audit-check", "b" * 40),
    ]


def test_repository_workflow_release_gates_are_current():
    assert check_workflows.check_workflows() == []


def test_dependabot_checker_rejects_routine_version_updates(monkeypatch, tmp_path):
    path = tmp_path / "dependabot.yml"
    path.write_text(
        """version: 2
updates:
  - package-ecosystem: pip
    allow:
      - dependency-name: "*"
        update-types: ["version-update:semver-patch"]
    groups:
      python-lock:
        patterns: ["*"]
  - package-ecosystem: cargo
    allow:
      - dependency-name: "*"
        update-types: ["version-update:semver-patch"]
    groups:
      rust-lock:
        patterns: ["*"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_workflows, "DEPENDABOT_PATH", path)
    errors: list[str] = []

    check_workflows._check_dependabot(errors)

    assert errors == [
        "dependabot.yml: pip routine version updates must be disabled",
        "dependabot.yml: pip must not define routine update groups",
        "dependabot.yml: cargo routine version updates must be disabled",
        "dependabot.yml: cargo must not define routine update groups",
    ]


def test_release_workflow_checker_rejects_dirty_source_override():
    path = check_workflows.WORKFLOW_DIR / "release-package.yml"
    source = path.read_text(encoding="utf-8") + "\n--allow-dirty\n"
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any("must fail closed on dirty source trees" in error for error in errors)


def test_release_workflow_checker_rejects_asset_clobbering():
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    source = path.read_text(encoding="utf-8") + "\n--clobber\n"
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any("must not overwrite published release assets" in error for error in errors)


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


def test_semgrep_output_path_creates_parent_and_removes_stale_file(tmp_path):
    sarif = tmp_path / "nested" / "results.sarif"
    sarif.parent.mkdir()
    sarif.write_text("stale", encoding="utf-8")

    prepared = run_semgrep._prepare_sarif_path(sarif)

    assert prepared == sarif.resolve()
    assert prepared.parent.is_dir()
    assert not prepared.exists()


def test_semgrep_scan_includes_untracked_source_and_excludes_generated_reports(
    tmp_path,
    monkeypatch,
):
    rulesets = tmp_path / "semgrep-rulesets.txt"
    rulesets.write_text("p/default\n", encoding="utf-8")
    monkeypatch.setattr(run_semgrep, "RULESET_FILE", rulesets)
    monkeypatch.setattr(run_semgrep, "_semgrep_executable", lambda: "semgrep")

    command = run_semgrep._scan_command(tmp_path / "results.sarif")

    assert "--no-git-ignore" in command
    exclusions = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--exclude"
    ]
    assert "*.sarif" in exclusions
    assert "models" in exclusions


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


def test_prune_bundle_removes_system_ucrt_and_package_smoke_rejects_it(tmp_path):
    bundle = tmp_path / "AudioForge"
    ucrt = bundle / "_internal" / "ucrtbase.dll"
    api_set = bundle / "_internal" / "api-ms-win-crt-runtime-l1-1-0.dll"
    unrelated = bundle / "_internal" / "runtime.dll"
    for path in (ucrt, api_set, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    errors = package_smoke.check_dist_bundle(bundle)
    assert any("app-local UCRT/API-set" in error for error in errors)

    removed = prune_bundle.prune_bundle(bundle)

    assert sorted(path.as_posix() for path in removed) == [
        "_internal/api-ms-win-crt-runtime-l1-1-0.dll",
        "_internal/ucrtbase.dll",
    ]
    assert not ucrt.exists()
    assert not api_set.exists()
    assert unrelated.is_file()


def test_package_smoke_historical_ucrt_exception_is_exact_and_version_bound(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(package_smoke, "_expected_version", lambda: "1.10.1")
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, "AudioForge.exe")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_valid_build_info(bundle, version="1.10.1")
    _write_bundle_file(
        bundle, "_internal/mic_eq/mic_eq_core.cp312-win_amd64.pyd"
    )
    for index in range(45):
        _write_bundle_file(
            bundle,
            f"_internal/api-ms-win-crt-historical-{index:02d}.dll",
        )
    _write_bundle_file(bundle, "_internal/ucrtbase.dll")

    assert (
        package_smoke.check_dist_bundle(
            bundle,
            allow_historical_ucrt_for_version="1.10.1",
        )
        == []
    )
    (bundle / "_internal/api-ms-win-crt-historical-00.dll").unlink()
    errors = package_smoke.check_dist_bundle(
        bundle,
        allow_historical_ucrt_for_version="1.10.1",
    )
    assert any("app-local UCRT/API-set" in error for error in errors)


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
        "https://raw.githubusercontent.com/example/project/revision/silero_vad.onnx",
        destination,
    )

    assert destination.read_bytes() == b"pinned-model"


def test_fetch_release_assets_rejects_untrusted_direct_download_url(tmp_path):
    with pytest.raises(ValueError, match="trusted raw.githubusercontent.com HTTPS"):
        fetch_release_assets._download_direct_url(
            "https://example.invalid/silero_vad.onnx",
            tmp_path / "silero_vad.onnx",
        )


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


def test_version_check_rejects_static_current_archive_claims(tmp_path, monkeypatch):
    (tmp_path / "release-notes").mkdir()
    (tmp_path / "README.md").write_text(
        "The exact release archive is 123,456 bytes.\n", encoding="utf-8"
    )
    (tmp_path / "release-notes" / "release-notes-v9.8.7.md").write_text(
        "Use the generated checksum sidecar.\n", encoding="utf-8"
    )
    monkeypatch.setattr(check_versions, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="exact release archive"):
        check_versions._check_no_static_current_archive_claims("9.8.7")


def test_version_check_accepts_generated_archive_sidecar_references(
    tmp_path, monkeypatch
):
    (tmp_path / "release-notes").mkdir()
    for path in (
        tmp_path / "README.md",
        tmp_path / "release-notes" / "release-notes-v9.8.7.md",
    ):
        path.write_text(
            "Use generated archive metadata and checksum sidecars.\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(check_versions, "REPO_ROOT", tmp_path)

    check_versions._check_no_static_current_archive_claims("9.8.7")
