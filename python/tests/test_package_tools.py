"""Tests for packaging smoke and release asset verification helpers."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import stat
import subprocess
import sys
import zipfile
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


def test_release_outputs_are_ignored_without_hiding_source(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(
        (TOOLS_DIR.parents[1] / ".gitignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    outputs = ["AudioForge-v9.8.7-deepfilter.provenance.json"]
    for artifact in ("win64-ultra.7z", "win64.msi", "source.7z"):
        for suffix in ("", ".sha256", ".metadata.json"):
            outputs.append(f"AudioForge-v9.8.7-{artifact}{suffix}")
    outputs.extend([
        "AudioForge-v9.8.7-win64-ultra.7z.manifest.json",
        "AudioForge-v9.8.7-win64.msi.manifest.json",
    ])
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *outputs, "python/mic_eq/new_source.py"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert set(result.stdout.splitlines()) == set(outputs)


def _write_bundle_file(bundle: Path, relative_path: str) -> None:
    path = bundle / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _native_extension_name() -> str:
    return "mic_eq_core" + package_smoke._expected_extension_suffix()


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


def test_runtime_analysis_without_optional_development_packages():
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys

class ExcludeOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'cffi', 'pycparser', 'charset_normalizer', 'typing_extensions', 'yaml'
        }:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, ExcludeOptional())
from mic_eq.ui.main_window import MainWindow
import numpy as np
from scipy.signal import correlate, lfilter, resample_poly
from scipy.optimize import least_squares, minimize

x = np.random.default_rng(42).normal(size=480)
assert np.isfinite(lfilter([0.5, 0.5], [1], x)).all()
assert resample_poly(x, 1, 3).shape == (160,)
assert correlate(x, x).argmax() == len(x) - 1
assert least_squares(lambda p: p - 2, [0.0]).success
assert minimize(lambda p: float((p[0] - 2) ** 2), [0.0]).success
assert MainWindow is not None
"""],
        cwd=TOOLS_DIR.parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows inherited stderr handle")
def test_noise_model_discovery_with_closed_stderr_pipe():
    result = subprocess.run(
        [sys.executable, "-c", """
import ctypes
from ctypes import wintypes
import msvcrt
import os
from mic_eq import AudioProcessor
from mic_eq.ui.app_bootstrap import configure_deepfilter_env

configure_deepfilter_env()
os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = "1"
processor = AudioProcessor()
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetStdHandle.argtypes = [wintypes.DWORD]
kernel.GetStdHandle.restype = wintypes.HANDLE
kernel.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
kernel.SetStdHandle.restype = wintypes.BOOL
previous = kernel.GetStdHandle(-12)
reader, writer = os.pipe()
os.close(reader)
assert kernel.SetStdHandle(-12, msvcrt.get_osfhandle(writer))
try:
    assert processor.list_noise_models()
finally:
    assert kernel.SetStdHandle(-12, previous)
    os.close(writer)
"""],
        cwd=TOOLS_DIR.parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


def test_build_script_propagates_pyinstaller_failure_code():
    source = (check_workflows.REPO_ROOT / "build_exe.ps1").read_text(encoding="utf-8")

    assert "$pyinstallerExitCode = $LASTEXITCODE" in source
    assert "exit $pyinstallerExitCode" in source


def test_msi_builder_accepts_explicit_python_path():
    source = (check_workflows.REPO_ROOT / "build_msi.ps1").read_text(
        encoding="utf-8"
    )

    assert '[string]$PythonPath = ""' in source
    assert "Resolve-Path -LiteralPath $buildPython" in source
    assert "& $buildPython -c" in source


def test_release_build_selects_python_313_and_pinned_cpu_ort_explicitly():
    workflow = (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(
        encoding="utf-8"
    )
    errors: list[str] = []

    check_workflows._check_python_runtime("release-package.yml", workflow, errors)

    assert errors == []


def test_ci_workflow_hydrates_cpu_ort_before_both_build_jobs():
    source = (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    errors: list[str] = []

    check_workflows._check_required_gates("ci.yml", source, errors)

    assert errors == []
    assert source.count("fetch_release_assets.py --only-cpu-runtime --force") == 2


def test_release_workflow_binds_existing_tag_to_checked_out_commit():
    source = (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(
        encoding="utf-8"
    )
    errors: list[str] = []

    check_workflows._check_required_gates("release-package.yml", source, errors)

    assert errors == []


def test_release_candidate_can_be_validated_before_tagging():
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(encoding="utf-8")
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    jobs = workflow["jobs"]
    assert jobs["package-windows"]["outputs"]["source_revision"] == "${{ steps.meta.outputs.source_revision }}"
    steps = jobs["package-windows"]["steps"]
    binding = next(step for step in steps if step.get("name") == "Bind existing release tag to checked-out commit")
    assert binding["if"] == "inputs.release_tag != '' || github.ref_type == 'tag'"
    validation = jobs["validate-candidate"]["steps"]
    assert validation[0]["with"]["ref"] == "${{ needs.package-windows.outputs.source_revision }}"
    install_index = next(i for i, step in enumerate(validation) if step.get("name") == "Install pinned validation runtime")
    smoke_index = next(i for i, step in enumerate(validation) if step.get("name") == "Verify, extract, and smoke-test exact candidate")
    assert install_index < smoke_index
    assert "--require-hashes -r requirements/runtime.txt" in validation[install_index]["run"]


def test_workflow_checker_rejects_legacy_python_pin():
    errors: list[str] = []

    check_workflows._check_python_runtime(
        "ci.yml",
        'python-version: "3.12.10"\n',
        errors,
    )

    assert any("must pin CPython 3.13.15" in error for error in errors)


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


def test_promotion_keeps_package_gates_without_hardware_runner():
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    source = path.read_text(encoding="utf-8")
    errors: list[str] = []
    check_workflows._check_required_gates(path.name, source, errors)
    assert errors == []
    assert "hardware_matrix_run_id" not in source
    assert "self-hosted" not in source
    assert "--require-hashes -r requirements/runtime.txt" in source
    assert "git fetch --no-tags origin $env:GITHUB_SHA --depth=1" in source


def test_release_workflow_checker_rejects_asset_clobbering():
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    source = path.read_text(encoding="utf-8") + "\ngh release upload --clobber\n"
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any("must not overwrite published release assets" in error for error in errors)


def test_release_workflow_checker_requires_active_package_smoke_gate():
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    source = path.read_text(encoding="utf-8").replace(
        "python python/tools/package_smoke.py --dist $extract",
        "# python python/tools/package_smoke.py --dist $extract",
        1,
    )
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any("package_smoke.py --dist" in error for error in errors)


def test_release_workflow_checker_requires_source_distribution_publication_gate():
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    source = path.read_text(encoding="utf-8").replace(
        "--require-source-distribution",
        "--removed-source-distribution-gate",
    )
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any("--require-source-distribution" in error for error in errors)


def test_workflow_gate_parser_excludes_disabled_and_non_blocking_steps():
    document = {
        "jobs": {
            "job": {
                "steps": [
                    {"run": "python enabled.py"},
                    {"if": False, "run": "python disabled.py"},
                    {"continue-on-error": True, "run": "python tolerated.py"},
                    {
                        "continue-on-error": "${{ true }}",
                        "run": "python expression-tolerated.py",
                    },
                    {"run": "# python comment.py\npython active.py"},
                ],
            },
            "disabled-job": {
                "if": "${{ false }}",
                "steps": [{"run": "python disabled-job.py"}],
            },
            "tolerated-job": {
                "continue-on-error": "${{ true }}",
                "steps": [{"run": "python tolerated-job.py"}],
            },
        }
    }

    active = check_workflows._active_run_source(document)

    assert "enabled.py" in active
    assert "active.py" in active
    assert "disabled.py" not in active
    assert "tolerated.py" not in active
    assert "expression-tolerated.py" not in active
    assert "disabled-job.py" not in active
    assert "tolerated-job.py" not in active
    assert "comment.py" not in active


@pytest.mark.parametrize(
    "workflow_name",
    (
        "release-hardware-qualify.yml",
        "release-hardware-matrix.yml",
        "release-promote.yml",
    ),
)
def test_release_workflow_checker_rejects_tag_after_path_separator(workflow_name):
    path = check_workflows.WORKFLOW_DIR / workflow_name
    source = path.read_text(encoding="utf-8").replace(
        "git rev-list -n 1 $env:RELEASE_TAG --",
        "git rev-list -n 1 -- $env:RELEASE_TAG",
    )
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any("git rev-list" in error for error in errors)


@pytest.mark.parametrize(
    "required_environment",
    (
        "PYTHONPATH: ${{ github.workspace }}/python",
        "VAD_MODEL_PATH: ${{ steps.candidate.outputs.vad_model }}",
        "${{ runner.temp }}/audioforge-release-candidate",
        "${{ runner.temp }}/release-hardware-qualification.json",
    ),
)
def test_release_workflow_checker_requires_isolated_hardware_harness(
    required_environment,
):
    path = check_workflows.WORKFLOW_DIR / "release-hardware-qualify.yml"
    source = path.read_text(encoding="utf-8").replace(required_environment, "removed", 1)
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source, errors)

    assert any(required_environment in error for error in errors)


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


def test_semgrep_error_findings_honor_only_in_source_suppressions(tmp_path):
    sarif = tmp_path / "results.sarif"
    sarif.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "tool": {"driver": {"rules": []}},
                        "results": [
                            {
                                "ruleId": "in-source-error",
                                "level": "error",
                                "suppressions": [{"kind": "inSource"}],
                            },
                            {
                                "ruleId": "external-error",
                                "level": "error",
                                "suppressions": [{"kind": "external"}],
                            },
                            {
                                "ruleId": "rejected-error",
                                "level": "error",
                                "suppressions": [
                                    {"kind": "inSource", "status": "rejected"}
                                ],
                            },
                            {
                                "ruleId": "under-review-error",
                                "level": "error",
                                "suppressions": [
                                    {"kind": "inSource", "status": "underReview"}
                                ],
                            },
                            {"ruleId": "active-error", "level": "error"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert run_semgrep._error_findings(sarif) == [
        "external-error",
        "rejected-error",
        "under-review-error",
        "active-error",
    ]


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
        value.removeprefix("--exclude=")
        for value in command
        if value.startswith("--exclude=")
    ]
    assert "*.sarif" in exclusions
    assert "models" in exclusions
    assert ".venv*" in exclusions
    assert ".venv" not in exclusions
    for secret_pattern in (".env", ".env.*", "credentials.*", "secrets.*"):
        assert secret_pattern in exclusions


def test_cpython313_offline_imports_work_without_ssl(tmp_path):
    if sys.version_info < (3, 13):
        pytest.skip("portable runtime is CPython 3.13+")

    code = r'''
import importlib.abc
import logging
import os
import sys
import tempfile


class BlockSSL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_hashlib", "ssl", "_ssl"}:
            raise ModuleNotFoundError(f"blocked test import: {fullname}")
        return None


sys.meta_path.insert(0, BlockSSL())
sys.path.insert(0, "python")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

with tempfile.TemporaryDirectory(prefix="audioforge-ssl-regression-") as root:
    os.environ["APPDATA"] = root
    import PyQt6.QtCore
    import PyQt6.QtGui
    import PyQt6.QtWidgets
    import logging.handlers
    import hashlib
    import hmac
    import mic_eq.ui.app_bootstrap
    from mic_eq.app_logging import configure_app_logging

    assert hashlib.sha256(b"audioforge").hexdigest() == (
        "45c955234cd1a7df4065ce3e6c962fb24ae7bd0131330dd85ba8ee9d634418be"
    )
    assert hmac.new(b"key", b"audioforge", hashlib.sha256).hexdigest() == (
        "3a84935d64d89fb8cf1fb3f0688e33b9b5c299d07cf16acbf7c51bb354cb8b46"
    )
    log_file = configure_app_logging()
    assert log_file.is_file()
    for handler in list(logging.getLogger().handlers):
        handler.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_smoke.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_package_smoke_rejects_empty_models_directory(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "_internal" / "models").mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    _write_bundle_file(bundle, "_internal/df.dll")
    _write_bundle_file(bundle, "_internal/onnxruntime.dll")
    _write_bundle_file(bundle, "_internal/onnxruntime_providers_shared.dll")
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
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
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")

    assert package_smoke.check_dist_bundle(bundle) == []


def test_package_smoke_rejects_duplicate_native_extension(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    _write_bundle_file(bundle, f"_internal/mic_eq_core/{_native_extension_name()}")
    (bundle / "_internal" / "example.dist-info").mkdir()

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("_internal/mic_eq_core/mic_eq_core*.pyd" in error for error in errors)


def test_package_smoke_rejects_foreign_python_abi_extension(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    _write_bundle_file(bundle, "_internal/mic_eq/mic_eq_core.foreign.pyd")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("foreign Python ABI extensions" in error for error in errors)


def test_package_smoke_rejects_retired_directml_payload(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    _write_bundle_file(bundle, "_internal/DirectML.dll")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("retired DirectML payload" in error for error in errors)


def test_package_smoke_rejects_retired_directml_notice(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    _write_bundle_file(bundle, "_internal/licenses/DirectML-LICENSE.txt")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("retired DirectML license notice" in error for error in errors)


def test_package_smoke_rejects_excluded_openssl_payload(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    for relative_path in (
        "_internal/_ssl.pyd",
        "_internal/_hashlib.pyd",
        "_internal/libssl-3.dll",
        "_internal/libcrypto-3.dll",
    ):
        _write_bundle_file(bundle, relative_path)

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("excluded OpenSSL payload" in error for error in errors)


def test_prune_bundle_removes_duplicate_native_extension_only_when_packaged_copy_exists(
    tmp_path,
):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    _write_bundle_file(bundle, f"_internal/mic_eq_core/{_native_extension_name()}")

    prune_bundle.prune_bundle(bundle)

    assert (
        bundle / "_internal" / "mic_eq" / _native_extension_name()
    ).is_file()
    assert not (bundle / "_internal" / "mic_eq_core").exists()


def test_prune_bundle_removes_foreign_python_abi_extension(tmp_path):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    foreign = bundle / "_internal" / "mic_eq" / "mic_eq_core.foreign.pyd"
    foreign.write_bytes(b"x")

    prune_bundle.prune_bundle(bundle)

    assert not foreign.exists()


def test_prune_bundle_keeps_top_level_native_extension_without_packaged_copy(tmp_path):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, f"_internal/mic_eq_core/{_native_extension_name()}")

    prune_bundle.prune_bundle(bundle)

    assert (
        bundle / "_internal" / "mic_eq_core" / _native_extension_name()
    ).is_file()


def test_prune_bundle_removes_excluded_openssl_payload(tmp_path):
    bundle = tmp_path / "AudioForge"
    relative_paths = (
        "_internal/_ssl.pyd",
        "_internal/_hashlib.pyd",
        "_internal/libssl-3.dll",
        "_internal/libcrypto-3.dll",
    )
    for relative_path in relative_paths:
        _write_bundle_file(bundle, relative_path)

    removed = prune_bundle.prune_bundle(bundle)

    assert sorted(path.as_posix() for path in removed) == sorted(relative_paths)
    assert not any((bundle / relative_path).exists() for relative_path in relative_paths)


def test_package_smoke_rejects_external_windows_icu(tmp_path):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, "_internal/ICUUC.dll")
    assert any("app-local Windows ICU" in error for error in package_smoke.check_dist_bundle(bundle))


def test_prune_removes_image_plugins_with_excluded_qt_modules(tmp_path):
    bundle = tmp_path / "AudioForge"
    for plugin in ("qpdf.dll", "qsvg.dll", "qico.dll"):
        _write_bundle_file(bundle, f"_internal/PyQt6/Qt6/plugins/imageformats/{plugin}")
    assert any("without its Qt module" in error for error in package_smoke.check_dist_bundle(bundle))
    prune_bundle.prune_bundle(bundle)
    assert not any("without its Qt module" in error for error in package_smoke.check_dist_bundle(bundle))
    assert (bundle / "_internal/PyQt6/Qt6/plugins/imageformats/qico.dll").is_file()


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
        bundle, f"_internal/mic_eq/{_native_extension_name()}"
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
    _write_bundle_file(bundle, "_internal/decoys/mic_eq_core.foreign.pyd")
    (bundle / "_internal" / "example.dist-info").mkdir()

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("_internal/df.dll" in error for error in errors)
    assert any("does not contain _internal/mic_eq/mic_eq_core" in error for error in errors)


def test_package_smoke_rejects_bundle_without_required_license_notice(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        if relative_path == "_internal/licenses/ONNXRuntime-LICENSE.txt":
            continue
        _write_bundle_file(bundle, relative_path)
    _write_valid_build_info(bundle)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("ONNXRuntime-LICENSE.txt" in error for error in errors)


def test_package_smoke_rejects_stale_bundle_version(tmp_path):
    bundle = tmp_path / "AudioForge"
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES:
        _write_bundle_file(bundle, relative_path)
    build_info = bundle / "_internal" / "audioforge-build.json"
    build_info.write_text(
        json.dumps({"schema_version": 1, "version": "0.0.0"}),
        encoding="utf-8",
    )
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")

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


def test_verify_release_assets_can_limit_to_hydrated_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.dll"
    runtime.write_bytes(b"runtime")
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "runtime.dll",
                        "size": runtime.stat().st_size,
                        "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                    },
                    {"path": "source-build.dll", "sha256": "0" * 64},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)

    assert verify_release_assets.verify_assets(manifest, {"runtime.dll"}) == []


def test_verify_release_assets_selected_paths_require_manifest_entries(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "runtime.dll"
    runtime.write_bytes(b"runtime")
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "runtime.dll",
                        "size": runtime.stat().st_size,
                        "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)

    errors = verify_release_assets.verify_assets(
        manifest, {"runtime.dll", "target/onnxruntime-cpu/lib/onnxruntime.dll"}
    )

    assert errors == [
        "target/onnxruntime-cpu/lib/onnxruntime.dll: manifest entry missing"
    ]


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


def test_verify_release_assets_accepts_attested_source_build(tmp_path, monkeypatch):
    dll = tmp_path / "df.dll"
    dll.write_bytes(b"verified-source-build")
    for relative in verify_release_assets.SOURCE_RECIPE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative != "build-support/deepfilter/provenance.json":
            path.write_bytes(relative.encode("utf-8"))

    provenance = {
        "schema_version": 1,
        "upstream": {
            "repository": "https://github.com/Rikorose/DeepFilterNet.git",
            "commit": "a" * 40,
        },
        "build": {
            "target": "x86_64-pc-windows-msvc",
            "profile": "release-lto",
            "features": ["capi"],
            "default_features": False,
            "required_exports": ["df_create"],
            "tested_rust": "1.94.0 (x86_64-pc-windows-msvc)",
        },
        "tract_linalg_patch": {
            "archive_sha256": "b" * 64,
            "patched_cargo_toml_sha256": "c" * 64,
            "build_rs_sha256": "d" * 64,
        },
    }
    (tmp_path / "build-support/deepfilter/provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    recipe_files = {
        relative: verify_release_assets._sha256(tmp_path / relative)
        for relative in verify_release_assets.SOURCE_RECIPE_FILES
    }
    attestation = {
        "schema_version": 1,
        "kind": "audioforge.deepfilter.build",
        "output": {
            "name": "df.dll",
            "bytes": dll.stat().st_size,
            "sha256": verify_release_assets._sha256(dll),
        },
        "source": {
            "repository": provenance["upstream"]["repository"],
            "commit": provenance["upstream"]["commit"],
        },
        "recipe": {
            "files": recipe_files,
            "tract_linalg_archive_sha256": provenance["tract_linalg_patch"]["archive_sha256"],
            "tract_linalg_patched_manifest_sha256": provenance["tract_linalg_patch"]["patched_cargo_toml_sha256"],
            "tract_linalg_build_rs_sha256": provenance["tract_linalg_patch"]["build_rs_sha256"],
            "target": provenance["build"]["target"],
            "profile": provenance["build"]["profile"],
            "features": provenance["build"]["features"],
            "default_features": provenance["build"]["default_features"],
        },
        "toolchain": {"rustc": "rustc 1.94.0 (test)"},
        "abi": {"required_exports": provenance["build"]["required_exports"]},
    }
    (tmp_path / "attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "df.dll",
                        "size": 1,
                        "sha256": "0" * 64,
                        "origin": {
                            "status": "verified-source-build",
                            "attestation_path": "attestation.json",
                            "provenance": "build-support/deepfilter/provenance.json",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)

    assert verify_release_assets.verify_assets(manifest) == []

    (tmp_path / "models/DeepFilterNet3_onnx.tar.gz").write_bytes(b"tampered")
    errors = verify_release_assets.verify_assets(manifest)
    assert any("recipe hash mismatch" in error for error in errors)


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


def test_fetch_pinned_archive_rejects_oversized_response(tmp_path, monkeypatch):
    payload = b"too-large"

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def geturl(self):
            return "https://github.com/example/project/releases/download/v1/runtime.zip"

    class Opener:
        def open(self, request, timeout):
            return Response(payload)

    monkeypatch.setattr(fetch_release_assets.urllib.request, "build_opener", lambda *_: Opener())
    entry = {
        "source": "https://github.com/example/project/releases/download/v1/runtime.zip",
        "origin": {
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "archive_size": len(payload) - 1,
        },
    }

    with pytest.raises(RuntimeError, match="exceeded its declared size"):
        fetch_release_assets._download_pinned_archive(entry, tmp_path, {})


def test_fetch_pinned_archive_rejects_unsafe_or_symlink_members(tmp_path):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.dll", b"escape")

    with pytest.raises(RuntimeError, match="unsafe member path"):
        fetch_release_assets._extract_pinned_zip_member(
            archive, tmp_path / "extract", "package/runtime.dll"
        )

    symlink_archive = tmp_path / "symlink.zip"
    symlink = zipfile.ZipInfo("package/runtime.dll")
    symlink.create_system = 3
    symlink.external_attr = stat.S_IFLNK << 16
    with zipfile.ZipFile(symlink_archive, "w") as output:
        output.writestr(symlink, b"target")

    with pytest.raises(RuntimeError, match="symlink member"):
        fetch_release_assets._extract_pinned_zip_member(
            symlink_archive, tmp_path / "extract-symlink", "package/runtime.dll"
        )


def test_verify_release_assets_validates_pinned_archive_contract(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.dll"
    runtime.write_bytes(b"runtime")
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "path": "runtime.dll",
                        "size": runtime.stat().st_size,
                        "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                        "source": "https://github.com/example/project/releases/download/v1/runtime.zip",
                        "origin": {
                            "status": "verified-upstream-archive",
                            "runtime": "CPU-only Windows x64",
                            "archive_sha256": "a" * 64,
                            "archive_size": 123,
                            "archive_member": "package/runtime.dll",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)

    assert verify_release_assets.verify_assets(manifest) == []

    invalid = json.loads(manifest.read_text(encoding="utf-8"))
    invalid["assets"][0]["origin"]["runtime"] = 1
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    errors = verify_release_assets.verify_assets(manifest)
    assert any("CPU-only runtime" in error for error in errors)


def test_fetch_release_assets_default_tag_comes_from_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(
        json.dumps({"fallback_release_tag": "v9.8.7", "assets": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(fetch_release_assets, "MANIFEST_PATH", manifest)

    assert fetch_release_assets._default_asset_source_tag() == "v9.8.7"


def test_fetch_source_asset_invokes_pinned_builder_without_release_fallback(
    tmp_path, monkeypatch
):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-OutputPath") + 1])
        attestation = Path(command[command.index("-AttestationPath") + 1])
        output.write_bytes(b"source-built")
        attestation.parent.mkdir(parents=True, exist_ok=True)
        attestation.write_text("{}", encoding="utf-8")
        return ""

    monkeypatch.setattr(fetch_release_assets, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fetch_release_assets.shutil, "which", lambda name: "pwsh.exe")
    monkeypatch.setattr(fetch_release_assets, "_run", fake_run)
    asset = {"name": "df.dll", "destination": Path("df.dll")}
    entry = {
        "path": "df.dll",
        "origin": {
            "status": "verified-source-build",
            "attestation_path": "target/deepfilter/df.dll.provenance.json",
        },
    }

    output = fetch_release_assets._build_source_asset(asset, entry, tmp_path / "tmp")

    assert output.read_bytes() == b"source-built"
    assert len(commands) == 1
    assert "build_deepfilter.ps1" in " ".join(commands[0])
    assert "gh" not in commands[0]


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
