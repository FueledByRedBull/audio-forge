"""Tests for packaging smoke and release asset verification helpers."""

from __future__ import annotations

import copy
import hashlib
import io
import os
import runpy
import shutil
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


import prune_bundle
import package_smoke
import verify_release_assets
import fetch_release_assets
import release_provenance
import check_versions
import run_semgrep
import check_workflows


TOOLS_DIR = Path(__file__).parent.parent / "tools"
QT_PLUGINS = "_internal/PySide6/plugins"
UNUSED_QT_PLUGIN_FILES = (
    f"{QT_PLUGINS}/platforms/qdirect2d.dll",
    *(
        f"{QT_PLUGINS}/imageformats/{name}.dll"
        for name in ("qgif", "qicns", "qjpeg", "qtga", "qtiff", "qwbmp", "qwebp")
    ),
    *(
        f"{QT_PLUGINS}/tls/{name}.dll"
        for name in ("qcertonlybackend", "qopensslbackend", "qschannelbackend")
    ),
    f"{QT_PLUGINS}/generic/qtuiotouchplugin.dll",
    f"{QT_PLUGINS}/networkinformation/qnetworklistmanager.dll",
)
# The Qt files and NumPy BLAS that the trimmed bundle must keep.
KEPT_PAYLOAD_FILES = (
    "_internal/PySide6/Qt6OpenGL.dll",
    "_internal/PySide6/QtOpenGL.pyd",
    f"{QT_PLUGINS}/platforms/qwindows.dll",
    f"{QT_PLUGINS}/platforms/qoffscreen.dll",
    f"{QT_PLUGINS}/platforms/qminimal.dll",
    f"{QT_PLUGINS}/imageformats/qico.dll",
    f"{QT_PLUGINS}/styles/qmodernwindowsstyle.dll",
    f"{QT_PLUGINS}/multimedia/windowsmediaplugin.dll",
    # NumPy ships its BLAS under a scipy-openblas name; it is not SciPy payload.
    "_internal/numpy.libs/libscipy_openblas64_-b788215d9d47792bcba3a2e2a7114320.dll",
)
REMOVED_BUNDLE_FILES = (
    *UNUSED_QT_PLUGIN_FILES,
    "_internal/scipy/optimize/_slsqplib.cp313-win_amd64.pyd",
    "_internal/scipy.libs/libscipy_openblas-197ee2fc9b4d071f7e048078cac74115.dll",
)


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


def test_bundle_path_baseline_includes_required_runtime_files():
    baseline_path = TOOLS_DIR.parents[1] / "evaluation" / "release-bundle-path-baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert set(package_smoke.REQUIRED_BUNDLE_FILES) <= set(baseline["paths"])


def test_bundle_path_baseline_omits_payload_removed_from_the_bundle():
    baseline_path = TOOLS_DIR.parents[1] / "evaluation" / "release-bundle-path-baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    removed = (
        *(path.as_posix() for path in prune_bundle.UNUSED_QT_PLUGIN_PATHS),
        *package_smoke.SCIPY_BUNDLE_DIRS,
    )

    assert not [
        path
        for path in baseline["paths"]
        if any(path == root or path.startswith(f"{root}/") for root in removed)
    ]
    assert set(KEPT_PAYLOAD_FILES) <= set(baseline["paths"])


def test_package_smoke_requires_spec_to_exclude_scipy():
    spec = (TOOLS_DIR.parents[1] / "AudioForge.spec").read_text(encoding="utf-8")
    assert package_smoke._spec_module_errors(spec) == []

    without = spec.replace('        "scipy",\n', "", 1)
    assert without != spec
    expected = ["AudioForge.spec must exclude scipy"]
    assert package_smoke._spec_module_errors(without) == expected
    # Listing it as a hidden import instead does not count as excluding it.
    hidden = without.replace('"json",', '"json", "scipy",', 1)
    assert package_smoke._spec_module_errors(hidden) == expected


def test_package_smoke_rejects_spec_that_excludes_or_drops_the_qtopengl_binding():
    spec = (TOOLS_DIR.parents[1] / "AudioForge.spec").read_text(encoding="utf-8")
    hidden_import = '        "PySide6.QtOpenGL",\n'
    assert spec.count(hidden_import) == 1

    excluded = spec.replace('        "ssl",\n', f'{hidden_import}        "ssl",\n', 1)
    dropped = spec.replace(hidden_import, "", 1)
    moved = excluded.replace(hidden_import, "", 1)

    for broken in (excluded, dropped, moved):
        (error,) = package_smoke._spec_module_errors(broken)
        assert "PySide6.QtOpenGL in hiddenimports and not exclude it" in error


@pytest.mark.parametrize(
    "statement",
    [
        "import scipy",
        "import scipy.signal",
        "from scipy.signal import resample_poly",
        "from scipy import signal",
        "from scipy.optimize import least_squares",
        "import scipy.io as sio",
    ],
)
def test_package_smoke_rejects_app_imports_of_scipy(tmp_path, statement):
    package = tmp_path / "python" / "mic_eq" / "analysis"
    package.mkdir(parents=True)
    (package / "fine.py").write_text(
        "import numpy\nfrom mic_eq.analysis import signal_processing\n",
        encoding="utf-8",
    )
    assert package_smoke._scipy_imports(tmp_path) == []

    (package / "bad.py").write_text(f"def fit():\n    {statement}\n", encoding="utf-8")

    errors = package_smoke._scipy_imports(tmp_path)
    assert len(errors) == 1
    assert errors[0].startswith("python/mic_eq/analysis/bad.py imports scipy")
    assert errors[0].endswith("which is a development-only dependency")


def test_qml_hook_keeps_runtime_modules_without_editor_type_descriptions():
    hook = runpy.run_path(
        str(TOOLS_DIR.parents[1] / "pyinstaller-hooks" / "hook-PySide6.QtQml.py")
    )

    assert not any(Path(source).suffix == ".qmltypes" for source, _ in hook["datas"])
    for module in hook["QML_MODULES"]:
        module_source = hook["_source"] / module
        assert (str(module_source / "qmldir"), str(hook["_destination"] / module)) in hook["datas"]
        assert any(Path(source).parent == module_source for source, _ in hook["binaries"])


def test_runtime_imports_work_without_scipy_and_optional_development_packages():
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys

class ExcludeOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        package = fullname.split('.')[0]
        if package in {
            'scipy', 'cffi', 'pycparser', 'charset_normalizer', 'typing_extensions', 'yaml'
        }:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, ExcludeOptional())
from mic_eq.ui.main_window import MainWindow
import numpy as np
from mic_eq.analysis.signal_processing import resample_poly

x = np.random.default_rng(42).normal(size=480)
assert resample_poly(x, 1, 3).shape == (160,)
assert 'scipy' not in sys.modules
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


def test_workflow_policy_rejects_mutable_action_reference(tmp_path, monkeypatch):
    source = (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    action, ref = check_workflows.ACTION_REF.findall(source)[0]
    (tmp_path / "ci.yml").write_text(source.replace(f"{action}@{ref}", f"{action}@v7"), encoding="utf-8")
    monkeypatch.setattr(check_workflows, "WORKFLOW_DIR", tmp_path)
    assert any("not pinned to a commit SHA" in error for error in check_workflows.check_workflows())


def test_repository_workflow_release_gates_are_current():
    assert check_workflows.check_workflows() == []


def test_ci_uploads_semgrep_results_with_scoped_code_scanning_permission():
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"push", "pull_request"}
    assert triggers["push"] == {
        "branches": ["master"],
    }
    static_job = workflow["jobs"]["static"]
    assert static_job["permissions"] == {
        "contents": "read",
        "security-events": "write",
    }
    upload = next(
        step
        for step in static_job["steps"]
        if step.get("name") == "Upload Semgrep results to GitHub Code Scanning"
    )
    assert upload["uses"].startswith("github/codeql-action/upload-sarif@")
    assert upload["with"] == {
        "sarif_file": "semgrep-results.sarif",
        "category": "semgrep",
    }
    assert upload["if"] == (
        "always() && github.event_name == 'push' && "
        "hashFiles('semgrep-results.sarif') != ''"
    )
    artifact = next(
        step
        for step in static_job["steps"]
        if step.get("name") == "Upload Semgrep SARIF"
    )
    assert artifact["uses"].startswith("actions/upload-artifact@")
    assert artifact["if"] == "always() && hashFiles('semgrep-results.sarif') != ''"
    assert artifact["with"]["path"] == "semgrep-results.sarif"
    assert artifact["with"]["if-no-files-found"] == "error"

    errors: list[str] = []
    check_workflows._check_permissions("ci.yml", workflow, errors)
    assert errors == []


@pytest.mark.parametrize(
    ("scope", "permissions"),
    [
        ("top", {"contents": "write"}),
        ("top", "write-all"),
        ("static", {"contents": "read", "security-events": "write", "actions": "write"}),
        ("static", "write-all"),
        ("pytest", {"security-events": "write"}),
        ("python", {"contents": "write"}),
        ("rust-stress", "write-all"),
    ],
)
def test_workflow_permission_checker_rejects_unapproved_writes(scope, permissions):
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    )
    target = workflow if scope == "top" else workflow["jobs"][scope]
    target["permissions"] = permissions
    errors: list[str] = []
    check_workflows._check_permissions("ci.yml", workflow, errors)
    assert errors


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


def test_release_metadata_checks_each_native_version_command_immediately():
    workflow = (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(
        encoding="utf-8"
    )
    lines = workflow.splitlines()

    msi_index = next(
        index
        for index, line in enumerate(lines)
        if "$msiVersion =" in line
    )
    prerelease_index = next(
        index
        for index, line in enumerate(lines)
        if "$isPrerelease =" in line
    )

    assert lines[msi_index + 1].strip().startswith("if ($LASTEXITCODE -ne 0")
    assert lines[prerelease_index + 1].strip().startswith("if ($LASTEXITCODE -ne 0")
    assert '$isPrerelease -notin @("true", "false")' in lines[prerelease_index + 1]
    assert (
        "if ($LASTEXITCODE -ne 0 -or $sourceRevision -notmatch"
        in workflow
    )
    assert (
        'if ($LASTEXITCODE -ne 0 -or $tagCommit -ne "${{ needs.package-windows.outputs.source_revision }}")'
        in workflow
    )


def test_ci_workflow_hydrates_cpu_ort_in_every_native_job():
    source = (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    errors: list[str] = []

    check_workflows._check_required_gates("ci.yml", source, errors)

    assert errors == []
    assert source.count("fetch_release_assets.py --only-cpu-runtime --force") == 3


def _ci_errors(mutate):
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    )
    mutate(workflow["jobs"])
    errors: list[str] = []
    check_workflows._check_required_gates(
        "ci.yml", check_workflows.yaml.safe_dump(workflow), errors
    )
    return "; ".join(errors)


def _drop_step(job, name):
    job["steps"] = [step for step in job["steps"] if step.get("name") != name]


def _set_step_run(job, name, run):
    next(step for step in job["steps"] if step.get("name") == name)["run"] = run


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda jobs: jobs.pop("python"), "required gate job python is missing"),
        (lambda jobs: jobs["rust"].update(needs=["rust-tests"]), "rust must need exactly"),
        (lambda jobs: jobs["python"].pop("if"), "python must use if: always()"),
        (lambda jobs: jobs["python"].update(env={}), "does not check the result of static"),
        (lambda jobs: jobs["static"].update({"continue-on-error": True}), "must not set continue-on-error"),
        (lambda jobs: jobs.update(extra=copy.deepcopy(jobs["static"])), "extra is not part of a required gate"),
        (
            lambda jobs: _drop_step(jobs["pytest"], "Hydrate pinned CPU ONNX Runtime"),
            "pytest must hydrate",
        ),
        (
            lambda jobs: _set_step_run(
                jobs["rust-tests"], "Rust tests", "cargo test --locked -p mic_eq_core -- --skip other_test --exact"
            ),
            "may skip only seeded_control_and_dsp_loops_remain_finite_under_contention",
        ),
        (
            lambda jobs: _set_step_run(
                jobs["rust-tests"],
                "Rust tests",
                "cargo test --locked -p mic_eq_core -- --skip seeded_control_and_dsp_loops_remain_finite_under_contention",
            ),
            "(with --exact)",
        ),
    ],
)
def test_ci_gate_jobs_and_job_split_are_enforced(mutate, message):
    assert message in _ci_errors(mutate)


def test_ci_pytest_shard_lists_reject_renamed_duplicate_and_unmatched_files(
    tmp_path, monkeypatch
):
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    )
    real = json.loads(check_workflows.PYTEST_SHARDS_PATH.read_text(encoding="utf-8"))
    last_listed = [path for files in real.values() for path in files][-1]
    first = next(iter(real))
    path = tmp_path / "pytest-shards.json"
    monkeypatch.setattr(check_workflows, "PYTEST_SHARDS_PATH", path)

    def errors_for(shards):
        path.write_text(json.dumps(shards), encoding="utf-8")
        errors: list[str] = []
        check_workflows._check_pytest_shards(workflow, errors)
        return "; ".join(errors)

    assert errors_for(real) == ""
    renamed = {**real, first: ["python/tests/test_renamed_away.py"]}
    assert "does not exist" in errors_for(renamed)
    assert "listed more than once" in errors_for({**real, first: [*real[first], last_listed]})
    assert "matrix shard must be" in errors_for({**real, "extra": ["python/tests/test_spectrum.py"]})
    assert "other than 'rest'" in errors_for({**real, "rest": ["python/tests/test_spectrum.py"]})


def test_release_workflow_binds_existing_tag_to_checked_out_commit():
    source = (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(
        encoding="utf-8"
    )
    errors: list[str] = []

    check_workflows._check_required_gates("release-package.yml", source, errors)

    assert errors == []


@pytest.mark.parametrize("location", ["comment", "other-step", "disabled-step"])
def test_release_rust_tests_require_pythonpath_on_the_test_step(location):
    source = (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(
        encoding="utf-8"
    )
    workflow = check_workflows.yaml.safe_load(source)
    steps = workflow["jobs"]["rust-tests"]["steps"]
    rust_tests = next(step for step in steps if step.get("name") == "Rust tests")
    pythonpath = rust_tests["env"].pop("PYTHONPATH")
    if location != "comment":
        steps.append({
            "run": "cargo test --locked -p mic_eq_core" if location == "disabled-step" else "echo ready",
            "env": {"PYTHONPATH": pythonpath},
            "if": location != "disabled-step",
        })
    source = check_workflows.yaml.safe_dump(workflow) + f"\n# PYTHONPATH: {pythonpath}\n"
    errors: list[str] = []

    check_workflows._check_required_gates("release-package.yml", source, errors)

    assert any("PYTHONPATH" in error for error in errors)


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


def _release_job_errors(mutate):
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(encoding="utf-8")
    )
    mutate(workflow["jobs"])
    errors: list[str] = []
    check_workflows._check_release_jobs(workflow, errors)
    return "; ".join(errors)


def _drop_steps(job, predicate):
    job["steps"] = [step for step in job["steps"] if not predicate(step)]


def _runs(marker):
    return lambda step: marker in str(step.get("run", ""))


def _uses(action):
    return lambda step: str(step.get("uses", "")).startswith(action)


def _artifact(name):
    return lambda step: (step.get("with") or {}).get("name") == name


def _step_named(job, name):
    return next(step for step in job["steps"] if step.get("name") == name)


def _set_checkout_ref(job, ref):
    for step in job["steps"]:
        if _uses("actions/checkout@")(step):
            step["with"]["ref"] = ref


def _move_to_end(job, name):
    step = _step_named(job, name)
    job["steps"].remove(step)
    job["steps"].append(step)


def test_release_job_layout_is_current():
    assert _release_job_errors(lambda jobs: None) == ""


@pytest.mark.parametrize(
    ("job", "marker"),
    [
        (job, marker)
        for job, markers in check_workflows.RELEASE_JOB_GATES.items()
        for marker in markers
    ],
)
def test_release_gate_must_run_in_its_job(job, marker):
    errors = _release_job_errors(lambda jobs: _drop_steps(jobs[job], _runs(marker)))

    assert f"{job} must run {marker!r}" in errors


def test_release_gate_must_be_enabled_and_blocking():
    def disable(jobs):
        _step_named(jobs["static"], "Ruff")["if"] = False
        _step_named(jobs["pytest"], "Pytest")["continue-on-error"] = True

    errors = _release_job_errors(disable)

    assert "static must run '-m ruff check'" in errors
    assert "pytest must run '-m pytest'" in errors


def test_release_shell_text_is_not_a_gate():
    lines = ["echo python -m ruff check", "Write-Host cargo fmt --check"]

    assert not check_workflows._active_run_has_marker(lines, "-m ruff check")
    assert not check_workflows._active_run_has_marker(lines, "cargo fmt --check")
    assert check_workflows._active_run_has_marker(
        [r".\.venv\Scripts\python.exe -m ruff check python"], "-m ruff check"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda jobs: jobs.pop("pytest"), "required release job pytest is missing"),
        (lambda jobs: jobs.pop("static"), "required release job static is missing"),
        (lambda jobs: jobs["pytest"].update(needs=["static"]), "pytest must need exactly package-windows"),
        (lambda jobs: jobs["rust-stress"].pop("needs"), "rust-stress must need exactly package-windows"),
        (
            lambda jobs: jobs["validate-candidate"].update(needs=["package-windows", "static"]),
            "validate-candidate must need exactly package-windows",
        ),
        (
            lambda jobs: jobs["pytest"].update({"if": "github.event_name == 'push'"}),
            "job pytest must be unconditional and blocking",
        ),
        (
            lambda jobs: jobs["rust-tests"].update({"continue-on-error": True}),
            "job rust-tests must be unconditional and blocking",
        ),
        (
            lambda jobs: _set_checkout_ref(jobs["pytest"], "${{ github.ref }}"),
            "pytest must check out ref ${{ needs.package-windows.outputs.source_revision }}",
        ),
        (
            lambda jobs: _set_checkout_ref(jobs["validate-candidate"], "${{ github.sha }}"),
            "validate-candidate must check out ref",
        ),
        (
            lambda jobs: _set_checkout_ref(jobs["static"], "${{ github.ref }}"),
            "static must check out ref ${{ inputs.release_tag || github.sha }}",
        ),
        (
            lambda jobs: _set_checkout_ref(jobs["package-windows"], "${{ github.ref }}"),
            "package-windows must check out ref ${{ inputs.release_tag || github.sha }}",
        ),
        (
            lambda jobs: _drop_steps(jobs["rust-tests"], _uses("actions/download-artifact@")),
            "rust-tests must download the audioforge-release-runtime artifact",
        ),
        (
            lambda jobs: _drop_steps(jobs["rust-stress"], _runs("verify_release_assets.py")),
            "rust-stress must verify the downloaded runtime assets",
        ),
        (
            lambda jobs: _step_named(jobs["pytest"], "Verify the packaged runtime files")["env"].clear(),
            "pytest must verify the downloaded runtime assets and the native module's SHA-256",
        ),
        (
            lambda jobs: _move_to_end(jobs["rust-tests"], "Verify the packaged runtime files"),
            "rust-tests must run 'cargo test --locked -p mic_eq_core' after verifying the runtime files",
        ),
        (
            lambda jobs: _move_to_end(jobs["pytest"], "Verify the packaged runtime files"),
            "pytest must run '-m pytest' after verifying the runtime files",
        ),
        (
            lambda jobs: _step_named(
                jobs["validate-candidate"], "Verify, extract, and smoke-test exact candidate"
            )["env"].clear(),
            "validate-candidate must compare the archive's native module",
        ),
        (
            lambda jobs: jobs["package-windows"]["outputs"].pop("native_sha256"),
            "package-windows must output the packaged native module's SHA-256",
        ),
        (
            lambda jobs: _drop_steps(jobs["package-windows"], _artifact("audioforge-release-runtime")),
            "package-windows must fetch the runtime assets, build the extension and the bundle",
        ),
        (
            lambda jobs: _move_to_end(
                jobs["package-windows"], "Download and verify pinned release runtime assets"
            ),
            "package-windows must fetch the runtime assets, build the extension and the bundle",
        ),
        (
            lambda jobs: _step_named(jobs["package-windows"], "Stage the packaged native module for the test jobs").update(
                run="exit 0"
            ),
            "package-windows must hash the native module packaged in dist/AudioForge",
        ),
        (
            lambda jobs: next(
                step for step in jobs["package-windows"]["steps"]
                if _artifact("audioforge-release-runtime")(step)
            )["with"].update(path="df.dll\npython/mic_eq/mic_eq_core*.pyd"),
            "artifact must contain models/silero_vad.onnx",
        ),
        (
            lambda jobs: jobs["package-windows"]["steps"].append(
                {"uses": "actions/cache@" + "a" * 40, "with": {"path": "target/deepfilter", "key": "k"}}
            ),
            "package-windows must not restore target/deepfilter from the Actions cache",
        ),
        (
            lambda jobs: jobs["package-windows"]["steps"].append(
                {"uses": "actions/cache@" + "a" * 40, "with": {"path": "target/release", "key": "k"}}
            ),
            "package-windows must not restore target/release from the Actions cache",
        ),
        (
            lambda jobs: jobs["package-windows"]["steps"].append(
                {"uses": "actions/cache/restore@" + "a" * 40, "with": {"path": "df.dll", "key": "k"}}
            ),
            "must not restore df.dll from the Actions cache",
        ),
        (
            lambda jobs: jobs["pytest"]["steps"].append(
                {"uses": "actions/cache/restore@" + "a" * 40, "with": {"path": "python/mic_eq/mic_eq_core*.pyd", "key": "k"}}
            ),
            "pytest must not restore python/mic_eq/mic_eq_core*.pyd from the Actions cache",
        ),
        (
            lambda jobs: jobs["pytest"]["strategy"]["matrix"].update(shard=["rest"]),
            "release-package.yml: pytest matrix shard must be",
        ),
        (
            lambda jobs: _step_named(jobs["pytest"], "Pytest").update(run="python -m pytest python/tests -q"),
            "pytest must run the files in pytest-shards.json",
        ),
    ],
)
def test_release_job_layout_is_enforced(mutate, message):
    assert message in _release_job_errors(mutate)


def test_release_job_layout_is_part_of_the_release_gate_check():
    workflow = check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(encoding="utf-8")
    )
    _drop_steps(workflow["jobs"]["static"], _runs("-m ruff check"))
    errors: list[str] = []

    check_workflows._check_required_gates(
        "release-package.yml", check_workflows.yaml.safe_dump(workflow), errors
    )

    assert "release-package.yml: static must run '-m ruff check'" in errors


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


@pytest.mark.parametrize("command", ["build", "test", "clippy", "run"])
def test_cargo_commands_must_pass_locked(command):
    def errors_for(line):
        document = {"jobs": {"j": {"steps": [{"run": line}]}}}
        errors: list[str] = []
        check_workflows._check_cargo_locked("x.yml", document, errors)
        return errors

    assert len(errors_for(f"cargo {command} -p mic_eq_core")) == 1
    assert errors_for(f"cargo {command} --locked -p mic_eq_core") == []
    assert len(errors_for(f"cargo {command} -p mic_eq_core -- --locked")) == 1


@pytest.mark.parametrize(
    "needle", ['$run.event -ne "workflow_dispatch"', "git cat-file -t $tagRef"]
)
def test_promotion_requires_dispatch_candidate_and_annotated_tag(needle):
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    source = path.read_text(encoding="utf-8")
    assert needle in source
    errors: list[str] = []

    check_workflows._check_required_gates(path.name, source.replace(needle, "true"), errors)

    assert any(needle in error for error in errors)


def test_promotion_binds_portable_and_msi_to_the_candidate_run():
    path = check_workflows.WORKFLOW_DIR / "release-promote.yml"
    document = check_workflows.yaml.safe_load(path.read_text(encoding="utf-8"))
    prefix = "python python/tools/release_provenance.py verify"
    script = "\n".join(
        step["run"]
        for step in check_workflows._active_steps(document)
        if isinstance(step.get("run"), str)
    )
    commands = [part.split("if ($LASTEXITCODE", 1)[0] for part in script.split(prefix)[1:]]

    assert len(commands) == 2
    for command in commands:
        assert (
            '--expected-producer "Release package" $env:CANDIDATE_RUN_ID '
            "$env:CANDIDATE_RUN_ATTEMPT"
        ) in command
        assert "--expected-commit" in command
        assert "--require-source-distribution" in command
        assert "--native-attestation" in command
    assert "--expected-archive-sha256" in commands[0]
    assert "--report validation/release-qualification.json" in commands[0]
    assert "--expected-archive-sha256" not in commands[1]
    assert "--report" not in commands[1]


@pytest.mark.parametrize(
    ("is_draft", "should_update"), [(True, True), (False, False)]
)
def test_promotion_refreshes_release_metadata_only_while_draft(
    tmp_path, is_draft, should_update
):
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required for the Windows promotion workflow")
    source = (check_workflows.WORKFLOW_DIR / "release-promote.yml").read_text()
    start = source.index("          $releaseJson = gh release view")
    end = source.index("          function Get-ReleaseAssets", start)
    script = source[start:end]
    notes = tmp_path / "release-notes.md"
    notes.write_text("# AudioForge 1.13.0\n\nSelected notes.\n", encoding="utf-8")
    setup = f'''
$ErrorActionPreference = 'Stop'
$version = '1.13.0'
$expectedPrerelease = $false
$notesPath = '{notes.as_posix()}'
$env:RELEASE_TAG = 'v1.13.0'
$env:GITHUB_REPOSITORY = 'example/audio-forge'
$script:editArgs = @()
function gh {{
  if ($args[1] -eq 'view') {{
    $global:LASTEXITCODE = 0
    return '{{"isDraft":{str(is_draft).lower()},"isPrerelease":false,"assets":[]}}'
  }}
  if ($args[1] -eq 'edit') {{
    $script:editArgs = @($args)
    $global:LASTEXITCODE = 0
  }}
}}
'''
    expected_update = "$true" if should_update else "$false"
    assertions = f'''
$hasExpectedMetadata =
  $script:editArgs -contains '--title' -and
  $script:editArgs -contains 'AudioForge 1.13.0' -and
  $script:editArgs -contains '--notes-file' -and
  $script:editArgs -contains $notesPath
if ($hasExpectedMetadata -ne {expected_update}) {{ exit 1 }}
'''
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            setup + script + assertions,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_promotion_rejects_extra_and_case_mismatched_asset_names():
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required for the Windows promotion workflow")
    source = (check_workflows.WORKFLOW_DIR / "release-promote.yml").read_text()
    start = source.find("          function Assert-ReleaseAssetNames")
    assert start >= 0, "promotion must enforce exact release asset names"
    end = source.index("          function Assert-AssetMatches", start)
    helper = source[start:end]
    assertions = r'''
$expected = @('AudioForge.7z')
$matching = [pscustomobject]@{ assets = @([pscustomobject]@{ name = 'AudioForge.7z' }) }
$partial = [pscustomobject]@{ assets = @() }
$extra = [pscustomobject]@{ assets = @(
  [pscustomobject]@{ name = 'AudioForge.7z' },
  [pscustomobject]@{ name = 'unexpected.txt' }
) }
$caseMismatch = [pscustomobject]@{ assets = @([pscustomobject]@{ name = 'audioforge.7z' }) }
Assert-ReleaseAssetNames $matching $expected
Assert-ReleaseAssetNames $partial $expected -AllowMissing
foreach ($release in @($extra, $caseMismatch)) {
  $rejected = $false
  try { Assert-ReleaseAssetNames $release $expected -AllowMissing }
  catch { $rejected = $true }
  if (-not $rejected) { exit 1 }
}
$rejected = $false
try { Assert-ReleaseAssetNames $partial $expected }
catch { $rejected = $true }
if (-not $rejected) { exit 1 }
'''
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            helper + assertions,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_promotion_checks_asset_names_before_publishing():
    source = (check_workflows.WORKFLOW_DIR / "release-promote.yml").read_text()

    allow_missing = source.index(
        "Assert-ReleaseAssetNames $release $expectedAssetNames -AllowMissing"
    )
    upload = source.index("gh release upload $env:RELEASE_TAG")
    exact_set = source.index(
        "Assert-ReleaseAssetNames $release $expectedAssetNames", allow_missing + 1
    )
    publish = source.index(
        "gh release edit $env:RELEASE_TAG --repo $env:GITHUB_REPOSITORY --draft=false"
    )
    assert allow_missing < upload < exact_set < publish


@pytest.mark.parametrize("digest", ["matching", "sha256:" + "0" * 64, ""])
def test_promotion_preflight_checks_digest_without_downloading(tmp_path, digest):
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required for the Windows promotion workflow")
    source = (check_workflows.WORKFLOW_DIR / "release-promote.yml").read_text()
    script = source[source.index("          $toUpload = @()"):]
    script = script[:script.index("          if ($toUpload.Count -gt 0)")]
    asset = tmp_path / "asset.7z"
    asset.write_bytes(b"verified candidate")
    if digest == "matching":
        digest = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    setup = """
$ErrorActionPreference = 'Stop'
$assetPaths = @($env:TEST_ASSET)
$remoteAssets = @{ 'asset.7z' = @{ digest = $env:TEST_DIGEST } }
function gh { throw 'Preflight must not download assets' }
"""
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", setup + script],
        env={**os.environ, "TEST_ASSET": str(asset), "TEST_DIGEST": digest},
        capture_output=True, text=True,
    )
    expected = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    assert (result.returncode == 0) == (digest == expected), result.stderr


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
                    {
                        "continue-on-error": "${{true}}",
                        "run": "python compact-expression-tolerated.py",
                    },
                    {
                        "if": "${{false}}",
                        "run": "python compact-expression-disabled.py",
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
    assert "compact-expression-tolerated.py" not in active
    assert "compact-expression-disabled.py" not in active
    assert "disabled-job.py" not in active
    assert "tolerated-job.py" not in active
    assert "comment.py" not in active


def test_workflow_gate_parser_excludes_disabled_and_non_blocking_actions():
    document = {
        "jobs": {
            "active": {
                "steps": [
                    {"uses": "actions/download-artifact@" + "a" * 40},
                    {"if": False, "uses": "actions/checkout@" + "b" * 40},
                    {
                        "continue-on-error": "${{true}}",
                        "uses": "actions/upload-artifact@" + "c" * 40,
                    },
                ]
            },
            "disabled-job": {
                "if": "${{false}}",
                "steps": [{"uses": "actions/download-artifact@" + "d" * 40}],
            },
        }
    }

    active = check_workflows._active_uses(document)

    assert active == ["actions/download-artifact@" + "a" * 40]


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
                                    {
                                        "id": "note-rule",
                                        "defaultConfiguration": {"level": "note"},
                                    },
                                ]
                            }
                        },
                        "results": [
                            {"ruleId": "error-rule"},
                            {"ruleId": "warning-rule"},
                            {"ruleId": "note-rule"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert run_semgrep._blocking_findings(sarif) == ["error-rule", "warning-rule"]


def test_semgrep_blocking_findings_honor_only_in_source_suppressions(tmp_path):
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

    assert run_semgrep._blocking_findings(sarif) == [
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
    for temp_pattern in (".tmp*", ".pytest-tmp*", "IGNORE*"):
        assert temp_pattern in exclusions


def test_semgrep_publishes_windows_file_uris_without_changing_findings(
    tmp_path, monkeypatch,
):
    source_uri = r"D:\\a\\audio-forge\\audio-forge\\python\\tools\\example.py"
    trace_uri = r"C:\work space\café#100%.py"
    payload = {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "Semgrep"}},
            "results": [{
                "ruleId": "example-rule",
                "level": "note",
                "message": {"text": source_uri},
                "locations": [{"physicalLocation": {
                    "artifactLocation": {"uri": source_uri, "uriBaseId": "%SRCROOT%"},
                    "region": {"startLine": 42},
                }}],
                "relatedLocations": [{"physicalLocation": {
                    "artifactLocation": {"uri": trace_uri},
                }}],
                "partialFingerprints": {"matchBasedId/v1": "original-fingerprint"},
                "suppressions": [{"kind": "external", "status": "accepted"}],
            }],
            "artifacts": [
                {"location": {"uri": r"\\server\share\source file.py"}},
                {"location": {"uri": "file:///D:/already%20encoded.py"}},
                {"location": {"uri": "https://example.com/source.py"}},
                {"location": {"uri": "python/relative.py"}},
            ],
        }],
    }
    output = tmp_path / "published.sarif"
    monkeypatch.setattr(run_semgrep, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_semgrep, "_scan_command", lambda path: ["semgrep"])
    monkeypatch.setattr(sys, "argv", ["run_semgrep.py", "--sarif", str(output)])

    def scan(command, *, cwd, env, check):
        assert cwd == tmp_path
        (cwd / ".semgrep-results.sarif").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_semgrep.subprocess, "run", scan)
    assert run_semgrep.main() == 0
    result = payload["runs"][0]["results"][0]
    result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] = (
        "file:///D:/a/audio-forge/audio-forge/python/tools/example.py"
    )
    result["relatedLocations"][0]["physicalLocation"]["artifactLocation"]["uri"] = (
        "file:///C:/work%20space/caf%C3%A9%23100%25.py"
    )
    payload["runs"][0]["artifacts"][0]["location"]["uri"] = (
        "file://server/share/source%20file.py"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / ".semgrep-results.sarif").exists()


def test_semgrep_publishes_only_findings_not_suppressed_in_source(
    tmp_path, monkeypatch,
):
    results = [
        {"ruleId": "accepted", "suppressions": [{"kind": "inSource", "status": "accepted"}]},
        {"ruleId": "implicit", "suppressions": [{"kind": "inSource"}]},
        {"ruleId": "rejected", "suppressions": [{"kind": "inSource", "status": "rejected"}]},
        {"ruleId": "external", "suppressions": [{"kind": "external"}]},
        {"ruleId": "active"},
    ]
    output = tmp_path / "published.sarif"
    monkeypatch.setattr(run_semgrep, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_semgrep, "_scan_command", lambda path: ["semgrep"])
    monkeypatch.setattr(sys, "argv", ["run_semgrep.py", "--sarif", str(output)])

    def scan(command, *, cwd, env, check):
        (cwd / ".semgrep-results.sarif").write_text(
            json.dumps({"runs": [{"results": results}]}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_semgrep.subprocess, "run", scan)
    assert run_semgrep.main() == 0
    published = json.loads(output.read_text(encoding="utf-8"))["runs"][0]["results"]
    assert [result["ruleId"] for result in published] == ["rejected", "external", "active"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"ruleId": "r", "level": "error"}, 1),
        ({"ruleId": "r", "level": "warning"}, 1),
        ({"ruleId": "r", "level": "note"}, 0),
        (
            {
                "ruleId": "r",
                "level": "warning",
                "suppressions": [{"kind": "inSource"}],
            },
            0,
        ),
    ],
)
def test_semgrep_exit_code_fails_on_unsuppressed_errors_and_warnings(
    tmp_path, monkeypatch, result, expected,
):
    monkeypatch.setattr(run_semgrep, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_semgrep, "_scan_command", lambda path: ["semgrep"])
    monkeypatch.setattr(
        sys, "argv", ["run_semgrep.py", "--sarif", str(tmp_path / "out.sarif")]
    )

    def scan(command, *, cwd, env, check):
        (cwd / ".semgrep-results.sarif").write_text(
            json.dumps({"runs": [{"results": [result]}]}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_semgrep.subprocess, "run", scan)
    assert run_semgrep.main() == expected


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
    import PySide6.QtCore
    import PySide6.QtGui
    import PySide6.QtWidgets
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


def test_package_smoke_requires_windows_audio_backend(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        if relative_path.endswith("windowsmediaplugin.dll"):
            continue
        _write_bundle_file(bundle, relative_path)
    _write_valid_build_info(bundle)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("windowsmediaplugin.dll" in error for error in errors)


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


def test_package_smoke_rejects_unused_qt_ffmpeg_payload(tmp_path):
    bundle = tmp_path / "AudioForge"
    (bundle / "AudioForge.exe").parent.mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"x")
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES[1:]:
        _write_bundle_file(bundle, relative_path)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")
    _write_bundle_file(bundle, "_internal/PySide6/avcodec-61.dll")
    _write_bundle_file(bundle, "_internal/PySide6/plugins/multimedia/ffmpegmediaplugin.dll")

    errors = package_smoke.check_dist_bundle(bundle)

    assert any("unused Qt FFmpeg payload" in error for error in errors)


def test_package_smoke_rejects_pythonwin_mfc(tmp_path):
    payload = tmp_path / "_internal" / "Pythonwin"
    payload.mkdir(parents=True)
    for name in ("win32ui.pyd", "mfc140u.dll"):
        (payload / name).write_bytes(b"unused")
    errors = package_smoke.check_dist_bundle(tmp_path)
    error = next(error for error in errors if "Pythonwin/MFC" in error)
    assert "win32ui.pyd" in error and "mfc140u.dll" in error


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


def test_prune_bundle_removes_unused_qt_ffmpeg_payload_and_keeps_windows_backend(tmp_path):
    bundle = tmp_path / "AudioForge"
    relative_paths = (
        "_internal/PySide6/avcodec-61.dll",
        "_internal/PySide6/avformat-61.dll",
        "_internal/PySide6/avutil-59.dll",
        "_internal/PySide6/swresample-5.dll",
        "_internal/PySide6/swscale-8.dll",
        "_internal/PySide6/plugins/multimedia/ffmpegmediaplugin.dll",
    )
    for relative_path in relative_paths:
        _write_bundle_file(bundle, relative_path)
    windows_backend = bundle / "_internal/PySide6/plugins/multimedia/windowsmediaplugin.dll"
    windows_backend.parent.mkdir(parents=True, exist_ok=True)
    windows_backend.write_bytes(b"x")

    removed = prune_bundle.prune_bundle(bundle)

    assert sorted(path.as_posix() for path in removed) == sorted(relative_paths)
    assert not any((bundle / relative_path).exists() for relative_path in relative_paths)
    assert windows_backend.is_file()


def test_prune_bundle_keeps_dependency_dist_info_metadata(tmp_path):
    bundle = tmp_path / "AudioForge"
    metadata = bundle / "_internal/example.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: example\n", encoding="utf-8")

    prune_bundle.prune_bundle(bundle)

    assert metadata.is_file()


def test_package_smoke_rejects_external_windows_icu(tmp_path):
    bundle = tmp_path / "AudioForge"
    _write_bundle_file(bundle, "_internal/ICUUC.dll")
    assert any("app-local Windows ICU" in error for error in package_smoke.check_dist_bundle(bundle))


def test_prune_removes_image_plugins_with_excluded_qt_modules(tmp_path):
    bundle = tmp_path / "AudioForge"
    for plugin in ("qpdf.dll", "qsvg.dll", "qico.dll"):
        _write_bundle_file(bundle, f"_internal/PySide6/plugins/imageformats/{plugin}")
    assert any("without its Qt module" in error for error in package_smoke.check_dist_bundle(bundle))
    prune_bundle.prune_bundle(bundle)
    assert not any("without its Qt module" in error for error in package_smoke.check_dist_bundle(bundle))
    assert (bundle / "_internal/PySide6/plugins/imageformats/qico.dll").is_file()


def test_qt_inventory_rejects_unselected_modules_and_prunes_virtual_keyboard(tmp_path):
    bundle = tmp_path / "AudioForge"
    unused = (
        "Qt6VirtualKeyboard.dll", "Qt6MultimediaWidgets.dll",
        "QtMultimediaWidgets.pyd", "Qt6Quick3DUtils.dll",
        "plugins/qmltooling/qmldbg_debugger.dll",
        "plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
    )
    for relative in (
        *unused, "Qt6Gui.dll", "Qt6Quick.dll", "plugins/platforms/qwindows.dll"
    ):
        _write_bundle_file(bundle, f"_internal/PySide6/{relative}")
    assert any("unselected Qt module" in error for error in package_smoke.check_dist_bundle(bundle))
    prune_bundle.prune_bundle(bundle)
    assert all(not (bundle / "_internal/PySide6" / relative).exists() for relative in unused)
    assert (bundle / "_internal/PySide6/Qt6Gui.dll").is_file()
    assert (bundle / "_internal/PySide6/Qt6Quick.dll").is_file()
    assert (bundle / "_internal/PySide6/plugins/platforms/qwindows.dll").is_file()
    assert not any("unselected Qt module" in error for error in package_smoke.check_dist_bundle(bundle))
    _write_bundle_file(bundle, "_internal/PySide6/Qt6Unexpected.dll")
    assert any("unselected Qt module" in error for error in package_smoke.check_dist_bundle(bundle))


def _write_valid_bundle(bundle: Path) -> None:
    for relative_path in package_smoke.REQUIRED_BUNDLE_FILES:
        _write_bundle_file(bundle, relative_path)
    _write_valid_build_info(bundle)
    _write_bundle_file(bundle, f"_internal/mic_eq/{_native_extension_name()}")


@pytest.mark.parametrize("returned", REMOVED_BUNDLE_FILES)
def test_package_smoke_rejects_payload_removed_from_the_bundle(tmp_path, returned):
    bundle = tmp_path / "AudioForge"
    _write_valid_bundle(bundle)
    for kept in KEPT_PAYLOAD_FILES:
        _write_bundle_file(bundle, kept)
    assert package_smoke.check_dist_bundle(bundle) == []

    _write_bundle_file(bundle, returned)

    (error,) = package_smoke.check_dist_bundle(bundle)
    assert "payload removed from the bundle" in error
    listed = error.split("payload removed from the bundle: ", 1)[1].split(", ")
    assert any(returned == item or returned.startswith(f"{item}/") for item in listed)


@pytest.mark.parametrize(
    "required",
    [
        # Loads the window icon; a missing plugin drops it silently.
        f"{QT_PLUGINS}/imageformats/qico.dll",
        # libshiboken imports it when the Qt Quick bindings load.
        "_internal/PySide6/QtOpenGL.pyd",
    ],
)
def test_package_smoke_requires_the_qt_files_the_trim_must_keep(tmp_path, required):
    bundle = tmp_path / "AudioForge"
    _write_valid_bundle(bundle)
    (bundle / required).unlink()

    errors = package_smoke.check_dist_bundle(bundle)

    assert errors == [f"{bundle} does not contain {required}"]


def test_prune_bundle_removes_unused_qt_plugins_and_keeps_the_used_ones(tmp_path):
    bundle = tmp_path / "AudioForge"
    _write_valid_bundle(bundle)
    for relative_path in (*UNUSED_QT_PLUGIN_FILES, *KEPT_PAYLOAD_FILES):
        _write_bundle_file(bundle, relative_path)
    assert any(
        "payload removed from the bundle" in error
        for error in package_smoke.check_dist_bundle(bundle)
    )

    prune_bundle.prune_bundle(bundle)

    assert not any((bundle / relative_path).exists() for relative_path in UNUSED_QT_PLUGIN_FILES)
    assert not (bundle / QT_PLUGINS / "tls").exists()
    assert all((bundle / relative_path).is_file() for relative_path in KEPT_PAYLOAD_FILES)
    assert package_smoke.check_dist_bundle(bundle) == []


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
    for relative in release_provenance.DEEPFILTER_RECIPE_FILES:
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
        for relative in release_provenance.DEEPFILTER_RECIPE_FILES
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

    # A source build's DLL and receipt must both validate before either replaces
    # the installed pair. Recipe metadata is still checked against the checkout.
    staging = tmp_path / "staging"
    staging.mkdir()
    staged_dll = staging / "df.dll"
    staged_receipt = staging / "attestation.json"
    staged_dll.write_bytes(dll.read_bytes())
    valid_receipt = (tmp_path / "attestation.json").read_bytes()
    staged_receipt.write_text("{}")
    dll.write_bytes(b"previous-good-build")
    installed_receipt = tmp_path / "attestation.json"
    installed_receipt.write_bytes(b"previous-good-receipt")
    monkeypatch.setattr(fetch_release_assets, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fetch_release_assets, "MANIFEST_PATH", manifest)
    with pytest.raises(RuntimeError, match="verification"):
        fetch_release_assets._install_verified_asset(
            staged_dll, dll, attestation=(staged_receipt, installed_receipt)
        )
    assert dll.read_bytes() == b"previous-good-build"
    assert installed_receipt.read_bytes() == b"previous-good-receipt"
    staged_receipt.write_bytes(valid_receipt)
    fetch_release_assets._install_verified_asset(
        staged_dll, dll, attestation=(staged_receipt, installed_receipt)
    )
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
    assert not (tmp_path / "target/deepfilter/df.dll.provenance.json").exists()
    assert (tmp_path / "tmp/target/deepfilter/df.dll.provenance.json").is_file()


@pytest.mark.parametrize("payload", [b"", b"evil", b"too-long-payload"])
@pytest.mark.parametrize("source_kind", ["direct", "release", "archive", "pinned"])
def test_forced_hydration_rejects_bad_bytes_before_replacing_asset(
    tmp_path, monkeypatch, payload, source_kind
):
    installed = tmp_path / "model.bin"
    installed.write_bytes(b"good")
    entry = {
        "path": "model.bin", "size": 4,
        "sha256": hashlib.sha256(b"good").hexdigest(),
    }
    if source_kind == "direct":
        entry.update(source="https://example.com/model.bin",
                     origin={"status": "pinned-upstream-model"})
    elif source_kind == "pinned":
        entry.update(source="https://github.com/example/runtime/releases/download/v1/runtime.zip",
                     origin={"status": "verified-upstream-archive", "runtime": "CPU-only Windows x64",
                             "archive_sha256": "a" * 64, "archive_size": 123,
                             "archive_member": "package/model.bin"})
    manifest = tmp_path / "release-assets.json"
    manifest.write_text(json.dumps({"fallback_release_tag": "v1.0.0", "assets": [entry]}))
    monkeypatch.setattr(fetch_release_assets, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verify_release_assets, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fetch_release_assets, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(fetch_release_assets.sys, "argv", ["fetch", "--force"])
    monkeypatch.setattr(fetch_release_assets.shutil, "which", lambda _: "gh")
    monkeypatch.setattr(fetch_release_assets, "_release_asset_names", lambda *_: {
        "model.bin" if source_kind == "release" else "AudioForge-test-win64-ultra.7z"
    })
    monkeypatch.setattr(fetch_release_assets, "_download_direct_url",
                        lambda _url, dest: dest.write_bytes(payload))
    monkeypatch.setattr(fetch_release_assets, "_download_asset",
                        lambda _tag, _repo, name, dest: (dest / name).write_bytes(payload))

    def extract(_archive, root, relative):
        output = root / relative
        output.parent.mkdir(parents=True)
        output.write_bytes(payload)
        return output

    monkeypatch.setattr(fetch_release_assets, "_extract_archive_asset", extract)
    monkeypatch.setattr(fetch_release_assets, "_download_pinned_archive",
                        lambda _entry, root, _cache: root / "archive.zip")
    monkeypatch.setattr(fetch_release_assets, "_extract_pinned_zip_member", extract)
    with pytest.raises(RuntimeError, match="verification"):
        fetch_release_assets.main()
    assert installed.read_bytes() == b"good"


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


def test_version_check_rejects_extra_release_notes(tmp_path, monkeypatch):
    notes = tmp_path / "release-notes"
    notes.mkdir()
    (notes / "release-notes-v9.8.7.md").write_text("Notes.\n", encoding="utf-8")
    monkeypatch.setattr(check_versions, "REPO_ROOT", tmp_path)
    version = check_versions.parse_version("9.8.7")

    check_versions._check_only_current_release_notes(version)

    (notes / "release-notes-v9.8.6.md").write_text("Old.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must hold only release-notes-v9.8.7.md"):
        check_versions._check_only_current_release_notes(version)
