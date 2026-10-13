"""Validate workflow YAML, immutable action pins, and release permissions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"
PYTEST_SHARDS_PATH = REPO_ROOT / ".github" / "pytest-shards.json"
# Required status checks are the gate jobs; each must wait for all of its parts.
CI_GATES = {
    "python": ("static", "pytest"),
    "rust": ("rust-tests", "rust-stress"),
}
CI_CPU_RUNTIME_JOBS = ("pytest", "rust-tests", "rust-stress")
CI_CPU_RUNTIME_HYDRATION = "fetch_release_assets.py --only-cpu-runtime --force"
# The debug-mode Rust run skips only this test; rust-stress runs it in release.
CI_DEBUG_SKIPPED_TEST = "seeded_control_and_dsp_loops_remain_finite_under_contention"
CI_LAST_PYTEST_SHARD = "rest"
# release-package.yml: `static` needs no built file and runs beside
# `package-windows`; the test jobs run on the runtime files it produced and
# `validate-candidate` compares the archive's native module with the tested one.
RELEASE_RUNTIME_ARTIFACT = "audioforge-release-runtime"
RELEASE_SOURCE_REF = "${{ inputs.release_tag || github.sha }}"
RELEASE_PACKAGED_REF = "${{ needs.package-windows.outputs.source_revision }}"
RELEASE_NATIVE_SHA256 = "${{ needs.package-windows.outputs.native_sha256 }}"
RELEASE_RUNTIME_FILES = (
    "df.dll",
    "target/deepfilter/df.dll.provenance.json",
    "target/onnxruntime-cpu/lib/onnxruntime.dll",
    "target/onnxruntime-cpu/lib/onnxruntime.lib",
    "target/onnxruntime-cpu/lib/onnxruntime_providers_shared.dll",
    "models/DeepFilterNet3_ll_onnx.tar.gz",
    "models/DeepFilterNet3_onnx.tar.gz",
    "models/silero_vad.onnx",
    "python/mic_eq/mic_eq_core*.pyd",
)
# Built or hydrated files never come from the Actions cache.
RELEASE_UNCACHEABLE = (
    "df.dll", "target/deepfilter", "target/onnxruntime", "models", "python/mic_eq", "dist",
)
RELEASE_TEST_JOBS = ("pytest", "rust-tests", "rust-stress")
RELEASE_JOB_GATES = {
    "static": (
        "python/tools/check_versions.py",
        "python/tools/check_evaluation_hygiene.py",
        "pip_audit --require-hashes -r requirements/runtime.txt",
        "pip_audit --require-hashes -r requirements/dev.txt",
        "-m ruff check",
        "-m pyright",
        "python/tools/check_workflows.py",
        "python/tools/package_smoke.py --source-only",
        "python/tools/run_semgrep.py",
        "cargo fmt --check",
        "cargo audit --deny warnings",
    ),
    "pytest": ("-m pytest",),
    "rust-tests": (
        "cargo test -p mic_eq_core",
        "cargo clippy -p mic_eq_core --all-targets -- -D warnings",
    ),
    "rust-stress": ("cargo test --release -p mic_eq_core --test stress_tests",),
}
ACTION_REF = re.compile(
    r"^\s*(?:-\s*)?uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE
)
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
PYTHON_VERSION = "3.13.15"
RUN_REQUIRED_MARKERS = frozenset(
    {
        "pip_audit --require-hashes",
        "run_semgrep.py",
        "cargo test",
        "cargo clippy",
        "cargo install cargo-audit",
        "cargo audit",
        "python/tools/check_versions.py",
        "python/tools/package_smoke.py",
        "package_smoke.py",
        "python/tools/verify_release_assets.py",
        "build_exe.ps1",
        "build_msi.ps1",
        "msi_smoke.py",
        "release_provenance.py",
        "gh release upload",
        "gh release download",
        "gh release create",
        "gh release edit",
        "gh run download",
        "evaluate_hardware_matrix.py",
        "evaluate_hardware_validation.py",
        "--smoke-test",
        "--upgrade-from",
    }
)


def _mapping(value: Any, context: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{context} must be a mapping")
        return {}
    return value


def _check_permissions(
    name: str,
    document: dict[str, Any],
    errors: list[str],
) -> None:
    top_permissions = _mapping(document.get("permissions"), f"{name}: permissions", errors)
    if top_permissions != {"contents": "read"}:
        errors.append(f"{name}: top-level permissions must be exactly contents: read")

    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    for job_name, raw_job in jobs.items():
        job = _mapping(raw_job, f"{name}: job {job_name}", errors)
        permissions = job.get("permissions")
        if name == "ci.yml" and job_name == "static":
            if permissions != {
                "contents": "read",
                "security-events": "write",
            }:
                errors.append(
                    "ci.yml: static must have only contents: read and "
                    "security-events: write"
                )
            continue
        if name == "release-promote.yml" and job_name == "promote-release":
            if permissions != {"actions": "read", "contents": "write"}:
                errors.append(
                    f"{name}: promote-release must have only actions: read and "
                    "contents: write"
                )
            continue
        if (
            name == "release-hardware-qualify.yml"
            and job_name == "qualify-hardware"
        ):
            if permissions != {"actions": "read", "contents": "read"}:
                errors.append(
                    f"{name}: qualify-hardware must have only actions: read and "
                    "contents: read"
                )
            continue
        if name == "release-hardware-matrix.yml" and job_name == "assemble":
            if permissions != {"actions": "read", "contents": "read"}:
                errors.append(
                    f"{name}: assemble must have only actions: read and contents: read"
                )
            continue
        if permissions == "write-all" or (
            isinstance(permissions, dict)
            and any(access == "write" for access in permissions.values())
        ):
            errors.append(f"{name}: job {job_name} must not request write permission")


def _is_boolean_expression(value: Any, expected: bool) -> bool:
    if value is expected:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    word = "true" if expected else "false"
    return normalized == word or re.fullmatch(
        rf"\$\{{\{{\s*{word}\s*\}}\}}", normalized
    ) is not None


def _active_steps(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled, blocking workflow steps."""
    active: list[dict[str, Any]] = []
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return active
    for raw_job in jobs.values():
        if not isinstance(raw_job, dict) or not isinstance(raw_job.get("steps"), list):
            continue
        if _is_boolean_expression(raw_job.get("if"), False) or _is_boolean_expression(
            raw_job.get("continue-on-error"), True
        ):
            continue
        for step in raw_job["steps"]:
            if not isinstance(step, dict) or not (
                isinstance(step.get("run"), str)
                or isinstance(step.get("uses"), str)
            ):
                continue
            if _is_boolean_expression(step.get("if"), False):
                continue
            if _is_boolean_expression(step.get("continue-on-error"), True):
                continue
            active.append(step)
    return active


def _active_run_lines(document: dict[str, Any]) -> list[str]:
    """Return executable run lines after excluding disabled/non-blocking jobs."""
    return [
        line
        for step in _active_steps(document)
        if isinstance(step.get("run"), str)
        for line in step["run"].splitlines()
        if not line.lstrip().startswith("#")
    ]


def _active_uses(document: dict[str, Any]) -> list[str]:
    """Return action references from enabled, blocking workflow steps."""
    return [
        step["uses"]
        for step in _active_steps(document)
        if isinstance(step.get("uses"), str)
    ]


def _active_run_source(document: dict[str, Any]) -> str:
    """Return executable run text after excluding disabled or non-blocking jobs."""
    active = _active_run_lines(document)
    return "\n".join(active)


def _active_run_has_marker(lines: list[str], marker: str) -> bool:
    """Match required commands as executable tokens, rather than printed text."""
    if marker.startswith("cargo "):
        command = marker.removeprefix("cargo ")
        return any(
            re.search(
                rf"(?i)(?:^\s*|[;&|]\s*)(?:&\s*)?(?:[^\s]+[/\\])?cargo(?:\.exe)?\s+{re.escape(command)}(?:\s|$)",
                line,
            )
            is not None
            for line in lines
        )
    if marker.startswith("gh "):
        command = marker.removeprefix("gh ")
        return any(
            re.search(
                rf"(?i)(?:^\s*|[;&|]\s*)(?:&\s*)?(?:[^\s]+[/\\])?gh(?:\.exe)?\s+{re.escape(command)}(?:\s|$)",
                line,
            )
            is not None
            for line in lines
        )
    if marker.startswith("pip_audit "):
        return any(
            re.search(
                r"(?i)(?:^\s*|[;&|]\s*)(?:&\s*)?(?:[^\s]+[/\\])?(?:python|python\.exe|py)(?:\s|$)",
                line,
            )
            and marker in line
            for line in lines
        )
    if marker.endswith(".ps1"):
        return any(
            re.search(r"(?i)(?:^\s*|[;&|]\s*)(?:&\s*)?(?:powershell|pwsh)(?:\.exe)?\b", line)
            and marker in line
            for line in lines
        )
    if ".py" in marker or marker.startswith("-m "):
        return any(
            re.search(
                r"(?i)(?:^\s*|[;&|]\s*)(?:&\s*)?(?:[^\s]+[/\\])?(?:python|python\.exe|py)(?:\s|$)",
                line,
            )
            is not None
            and marker in line
            for line in lines
        )
    return any(marker in line for line in lines)


def _check_required_gates(
    name: str,
    source: str,
    errors: list[str],
    *,
    document: dict[str, Any] | None = None,
) -> None:
    if document is None:
        try:
            document = _mapping(yaml.safe_load(source), name, errors)
        except yaml.YAMLError:
            document = {}
    active_run_lines = _active_run_lines(document)

    def has_gate(needle: str) -> bool:
        if needle.startswith("actions/") and needle.endswith("@"):
            return any(needle in action for action in _active_uses(document))
        if any(marker in needle for marker in RUN_REQUIRED_MARKERS):
            return _active_run_has_marker(active_run_lines, needle)
        return needle in source

    if name == "release-promote.yml":
        required = (
            "actions/download-artifact@",
            "release_tag must be the canonical vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rc.N tag",
            "git rev-list -n 1 $env:RELEASE_TAG --",
            "release_provenance.py verify",
            "--expected-archive-sha256",
            "--expected-commit",
            "--require-source-distribution",
            "git fetch --no-tags origin $env:GITHUB_SHA --depth=1",
            "Set-Content -LiteralPath $notesPath",
            "--report validation/release-qualification.json",
            "--native-attestation",
            "source_distribution.py verify",
            "--require-receipt",
            "validation/release-qualification.json",
            "evidenceArchive",
            "checksumsFile",
            "SHA256SUMS",
            "package_smoke.py --dist",
            "--smoke-test",
            "gh release upload",
            "gh release download",
            "Assert-AssetMatches $assetPath (Join-Path $verifyDir $assetName) $assetName",
            "gh release edit $env:RELEASE_TAG --repo $env:GITHUB_REPOSITORY "
            '--title "AudioForge $version" --notes-file $notesPath',
            "Assert-ReleaseAssetNames $release $expectedAssetNames -AllowMissing",
            "Assert-ReleaseAssetNames $release $expectedAssetNames",
            "Compare-Object -ReferenceObject $ExpectedNames -DifferenceObject $actualNames -CaseSensitive",
        )
        for needle in required:
            if not has_gate(needle):
                errors.append(
                    f"{name}: missing required promotion gate {needle!r}"
                )
        upload_lines = (
            line
            for line in source.splitlines()
            if re.search(r"\bgh\s+release\s+upload\b", line)
        )
        if any("--clobber" in line for line in upload_lines):
            errors.append(
                f"{name}: promotion must not overwrite published release assets"
            )
        return
    if name == "release-hardware-matrix.yml":
        required = (
            "gh run download",
            "release_tag must be the canonical vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rc.N tag",
            "git rev-list -n 1 $env:RELEASE_TAG --",
            "evaluate_hardware_matrix.py",
            "--expected-archive-sha256",
            "--expected-source-revision",
            "audioforge-release-hardware-matrix-",
        )
        for needle in required:
            if not has_gate(needle):
                errors.append(f"{name}: missing required matrix gate {needle!r}")
        return
    if name == "release-hardware-qualify.yml":
        required = (
            "runs-on: [self-hosted, windows, x64, audioforge-hardware]",
            "actions/download-artifact@",
            "release_tag must be the canonical vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rc.N tag",
            "git rev-list -n 1 $env:RELEASE_TAG --",
            "release_provenance.py verify",
            "--expected-archive-sha256",
            "--expected-commit",
            "--require-source-distribution",
            "--native-attestation",
            "evaluate_hardware_validation.py",
            "--confirm-scenario-observed",
            "explicit operator attestation",
            "--health-duration",
            "$duration -lt 1800",
            "PYTHONPATH: ${{ github.workspace }}/python",
            "VAD_MODEL_PATH: ${{ steps.candidate.outputs.vad_model }}",
            "${{ runner.temp }}/audioforge-release-candidate",
            "${{ runner.temp }}/release-hardware-qualification.json",
            "audioforge-release-hardware-validation-",
        )
        for needle in required:
            if not has_gate(needle):
                errors.append(
                    f"{name}: missing required hardware gate {needle!r}"
                )
        return

    shared = (
        "pip_audit --require-hashes -r requirements/runtime.txt",
        "pip_audit --require-hashes -r requirements/dev.txt",
        "run_semgrep.py",
        "cargo test --release -p mic_eq_core --test stress_tests",
        "cargo clippy -p mic_eq_core --all-targets -- -D warnings",
        "cargo install cargo-audit --version 0.22.2 --locked --force",
        "cargo audit --deny warnings",
    )
    required = shared
    if name == "release-package.yml":
        rust_test_steps = [
            step for step in _active_steps(document)
            if _active_run_has_marker(
                str(step.get("run", "")).splitlines(), "cargo test -p mic_eq_core"
            )
        ]
        pythonpath = (
            r"${{ github.workspace }}\.venv\Lib\site-packages;${{ github.workspace }}\python"
        )
        if not rust_test_steps or any(
            not isinstance(step.get("env"), dict)
            or step["env"].get("PYTHONPATH") != pythonpath
            for step in rust_test_steps
        ):
            errors.append(
                f"{name}: Rust test steps must set PYTHONPATH to the project venv and Python source"
            )
        required += (
            "python/tools/check_versions.py",
            "python/tools/package_smoke.py --source-only",
            "python/tools/verify_release_assets.py",
            "powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1",
            "actions/upload-artifact@",
            "AudioForge-$expectedTag-win64-ultra.7z",
            "release_provenance.py create",
            "release_provenance.py verify",
            "source_distribution.py bundle",
            "--include-runtime-assets",
            "--require-receipt",
            "release-bundle-path-baseline.json",
            "--smoke-test",
            "--upgrade-from",
            "--native-attestation",
            "--file build-support/deepfilter/Cargo.lock",
            "git fetch --no-tags origin $tagFetchSpec",
            "$tagCommit -ne $currentCommit",
        )
        if "--allow-dirty" in source:
            errors.append(
                f"{name}: release candidates must fail closed on dirty source trees"
            )
        _check_release_jobs(document, errors)
    for needle in required:
        if not has_gate(needle):
            errors.append(f"{name}: missing required release gate {needle!r}")

    # The release workflow's test jobs verify the runtime assets they download
    # instead (see _check_release_jobs).
    if name != "release-package.yml" and "cargo test -p mic_eq_core" in source:
        model_step = source.find("silero_vad.onnx")
        rust_tests = source.find("cargo test -p mic_eq_core")
        if model_step < 0 or model_step > rust_tests:
            errors.append(
                f"{name}: pinned Silero model must be available before Rust tests"
            )
    if name == "ci.yml":
        _check_ci_jobs(document, errors)
        _check_pytest_shards(document, errors)


def _is_always(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"(?:\$\{\{\s*)?always\(\)(?:\s*\}\})?", value.strip()
    ) is not None


def _check_ci_jobs(document: dict[str, Any], errors: list[str]) -> None:
    """Require gate jobs that fail unless every part succeeded."""
    jobs = _mapping(document.get("jobs"), "ci.yml: jobs", errors)
    parts = {part for group in CI_GATES.values() for part in group}
    for job_name, job in jobs.items():
        if job_name not in CI_GATES and job_name not in parts:
            errors.append(f"ci.yml: job {job_name} is not part of a required gate")
        if isinstance(job, dict) and "continue-on-error" in job:
            errors.append(f"ci.yml: job {job_name} must not set continue-on-error")
    for gate, expected in CI_GATES.items():
        job = jobs.get(gate)
        if not isinstance(job, dict):
            errors.append(f"ci.yml: required gate job {gate} is missing")
            continue
        needs = job.get("needs")
        if isinstance(needs, str):
            needs = [needs]
        if not isinstance(needs, list) or sorted(needs) != sorted(expected):
            errors.append(f"ci.yml: {gate} must need exactly {', '.join(expected)}")
        if not _is_always(job.get("if")):
            errors.append(
                f"ci.yml: {gate} must use if: always() so a skipped or cancelled part fails it"
            )
        gate_text = yaml.safe_dump(job)
        for part in expected:
            if f"needs.{part}.result" not in gate_text:
                errors.append(f"ci.yml: {gate} does not check the result of {part}")
    for job_name in CI_CPU_RUNTIME_JOBS:
        job = jobs.get(job_name)
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list) or not any(
            isinstance(step, dict) and CI_CPU_RUNTIME_HYDRATION in str(step.get("run", ""))
            for step in steps
        ):
            errors.append(f"ci.yml: {job_name} must hydrate the pinned CPU ONNX Runtime")
    for line in _active_run_lines(document):
        if _active_run_has_marker([line], "cargo test") and "--release" not in line:
            skipped = re.findall(r"--skip\s+(\S+)", line)
            if skipped and (
                skipped != [CI_DEBUG_SKIPPED_TEST] or "--exact" not in line.split()
            ):
                errors.append(
                    "ci.yml: the debug-mode Rust run may skip only "
                    f"{CI_DEBUG_SKIPPED_TEST} (with --exact)"
                )


def _step_lines(step: dict[str, Any]) -> list[str]:
    return [
        line
        for line in str(step.get("run", "")).splitlines()
        if not line.lstrip().startswith("#")
    ]


def _check_release_jobs(document: dict[str, Any], errors: list[str]) -> None:
    """Require the gate layout of release-package.yml.

    Promotion accepts only a successful run, so every job must be unconditional
    and blocking. The test jobs must run on the runtime files package-windows
    produced and verify them, including the packaged native module's SHA-256,
    before testing; validate-candidate compares the archive with that module.
    """
    name = "release-package.yml"
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)

    def fail(message: str) -> None:
        errors.append(f"{name}: {message}")

    def job_steps(job_name: str) -> list[dict[str, Any]]:
        job = jobs.get(job_name)
        return _active_steps({"jobs": {job_name: job}}) if isinstance(job, dict) else []

    def index(steps: list[dict[str, Any]], predicate: Any) -> int:
        return next((i for i, step in enumerate(steps) if predicate(step)), -1)

    def runs(marker: str) -> Any:
        return lambda step: _active_run_has_marker(_step_lines(step), marker)

    def uses(action: str, **expected: str) -> Any:
        return lambda step: str(step.get("uses", "")).startswith(f"{action}@") and all(
            (step.get("with") or {}).get(key) == value for key, value in expected.items()
        )

    def hashes_native_module(location: str) -> Any:
        def check(step: dict[str, Any]) -> bool:
            env = step.get("env")
            run = str(step.get("run", ""))
            return (
                isinstance(env, dict)
                and RELEASE_NATIVE_SHA256 in env.values()
                and all(part in run for part in ("Get-FileHash", "mic_eq_core*.pyd", location))
            )

        return check

    def needs_only_package(job_name: str) -> None:
        needs = (jobs.get(job_name) or {}).get("needs")
        if (needs if isinstance(needs, list) else [needs]) != ["package-windows"]:
            fail(f"{job_name} must need exactly package-windows")

    def checks_out(job_name: str, ref: str) -> None:
        checkouts = [s for s in job_steps(job_name) if uses("actions/checkout")(s)]
        if not checkouts or any((s.get("with") or {}).get("ref") != ref for s in checkouts):
            fail(f"{job_name} must check out ref {ref}")

    for job_name in ("static", "package-windows", *RELEASE_TEST_JOBS, "validate-candidate"):
        if not isinstance(jobs.get(job_name), dict):
            fail(f"required release job {job_name} is missing")
    for job_name, job in jobs.items():
        if isinstance(job, dict) and ("if" in job or "continue-on-error" in job):
            fail(
                f"job {job_name} must be unconditional and blocking; promotion "
                "requires every job of the run to succeed"
            )

    for job_name, markers in RELEASE_JOB_GATES.items():
        lines = [line for step in job_steps(job_name) for line in _step_lines(step)]
        for marker in markers:
            if not _active_run_has_marker(lines, marker):
                fail(f"{job_name} must run {marker!r}")
    shards = _read_pytest_shards()
    if shards is not None:
        _check_pytest_job(document, shards, name, errors)

    for job_name in ("static", "package-windows"):
        checks_out(job_name, RELEASE_SOURCE_REF)
    for job_name in (*RELEASE_TEST_JOBS, "validate-candidate"):
        needs_only_package(job_name)
        checks_out(job_name, RELEASE_PACKAGED_REF)
    for job_name in RELEASE_TEST_JOBS:
        steps = job_steps(job_name)
        download = index(steps, uses("actions/download-artifact", name=RELEASE_RUNTIME_ARTIFACT))
        verify = index(
            steps,
            lambda step: hashes_native_module("python/mic_eq")(step)
            and runs("python/tools/verify_release_assets.py")(step),
        )
        if download < 0:
            fail(f"{job_name} must download the {RELEASE_RUNTIME_ARTIFACT} artifact")
        if verify < download or verify < 0:
            fail(
                f"{job_name} must verify the downloaded runtime assets and the native "
                "module's SHA-256 after downloading them"
            )
        for marker in RELEASE_JOB_GATES[job_name]:
            if 0 <= index(steps, runs(marker)) <= max(verify, download):
                fail(f"{job_name} must run {marker!r} after verifying the runtime files")
    if index(job_steps("validate-candidate"), hashes_native_module("_internal/mic_eq")) < 0:
        fail("validate-candidate must compare the archive's native module with the tested SHA-256")

    package = job_steps("package-windows")
    outputs = (jobs.get("package-windows") or {}).get("outputs")
    if not isinstance(outputs, dict) or outputs.get("native_sha256") != (
        "${{ steps.runtime.outputs.native_sha256 }}"
    ):
        fail("package-windows must output the packaged native module's SHA-256")
    fetch = index(package, runs("python/tools/fetch_release_assets.py"))
    extension = index(package, runs("maturin develop --release"))
    build = index(package, runs("build_exe.ps1"))
    stage = index(package, lambda step: step.get("id") == "runtime")
    upload = index(package, uses("actions/upload-artifact", name=RELEASE_RUNTIME_ARTIFACT))
    if not 0 <= fetch < extension < build < stage < upload:
        fail(
            "package-windows must fetch the runtime assets, build the extension and the "
            "bundle, then stage and upload the runtime artifact, in that order"
        )
    if stage >= 0 and not all(
        part in str(package[stage].get("run", ""))
        for part in ("dist/AudioForge/_internal/mic_eq", "Get-FileHash", "native_sha256=")
    ):
        fail("package-windows must hash the native module packaged in dist/AudioForge")
    if upload >= 0:
        listed = str((package[upload].get("with") or {}).get("path", "")).split()
        for path in RELEASE_RUNTIME_FILES:
            if path not in listed:
                fail(f"the {RELEASE_RUNTIME_ARTIFACT} artifact must contain {path}")

    for job_name, job in jobs.items():
        for step in job.get("steps", []) if isinstance(job, dict) else []:
            if not str(step.get("uses", "")).startswith("actions/cache"):
                continue
            for path in str((step.get("with") or {}).get("path", "")).split():
                cached = path.replace("\\", "/").lstrip("./")
                if cached.startswith(RELEASE_UNCACHEABLE) or (
                    job_name == "package-windows" and cached.startswith("target")
                ):
                    fail(f"job {job_name} must not restore {path} from the Actions cache")


def _check_pytest_shards(document: dict[str, Any], errors: list[str]) -> None:
    """Keep the shard lists real files; the last shard runs everything else."""
    label = PYTEST_SHARDS_PATH.name
    try:
        shards = json.loads(PYTEST_SHARDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"{label}: cannot load pytest shard lists: {error}")
        return
    if not isinstance(shards, dict) or not shards or CI_LAST_PYTEST_SHARD in shards:
        errors.append(
            f"{label}: must map named shards other than {CI_LAST_PYTEST_SHARD!r} "
            "to test file lists"
        )
        return
    listed: set[str] = set()
    for shard, files in shards.items():
        if not isinstance(files, list) or not files:
            errors.append(f"{label}: shard {shard} must list at least one test file")
            continue
        for path in files:
            if not isinstance(path, str) or not re.fullmatch(
                r"python/tests/[^/\\]+\.py", path
            ):
                errors.append(f"{label}: shard {shard} entry {path!r} is not a python/tests/*.py path")
            elif not (REPO_ROOT / path).is_file():
                errors.append(f"{label}: shard {shard} lists {path}, which does not exist")
            elif path in listed:
                errors.append(f"{label}: {path} is listed more than once")
            else:
                listed.add(path)
    _check_pytest_job(document, shards, "ci.yml", errors)


def _check_pytest_job(
    document: dict[str, Any], shards: dict[str, Any], name: str, errors: list[str]
) -> None:
    """Require the pytest matrix of `name` to run the shards of pytest-shards.json."""
    label = PYTEST_SHARDS_PATH.name
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    job = jobs.get("pytest")
    if not isinstance(job, dict):
        return
    strategy = job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    expected = [*shards, CI_LAST_PYTEST_SHARD]
    if not isinstance(matrix, dict) or matrix.get("shard") != expected:
        errors.append(f"{name}: pytest matrix shard must be {expected}, as listed in {label}")
    steps = job.get("steps")
    run = "\n".join(
        str(step.get("run", "")) for step in steps if isinstance(step, dict)
    ) if isinstance(steps, list) else ""
    if label not in run or "--ignore" not in run:
        errors.append(
            f"{name}: pytest must run the files in {label} and ignore them in the last shard"
        )


def _read_pytest_shards() -> dict[str, Any] | None:
    try:
        shards = json.loads(PYTEST_SHARDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # _check_pytest_shards reports an unreadable shard file
    return shards if isinstance(shards, dict) else None


def _check_dependabot(errors: list[str]) -> None:
    try:
        document = yaml.safe_load(DEPENDABOT_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        errors.append(f"dependabot.yml: cannot load configuration: {error}")
        return
    document = _mapping(document, "dependabot.yml", errors)
    updates = document.get("updates")
    if not isinstance(updates, list):
        errors.append("dependabot.yml: updates must be a list")
        return
    for ecosystem in ("pip", "cargo"):
        matching = [
            item
            for item in updates
            if isinstance(item, dict) and item.get("package-ecosystem") == ecosystem
        ]
        if len(matching) != 1:
            errors.append(
                f"dependabot.yml: expected one {ecosystem} update configuration"
            )
            continue
        config = matching[0]
        if config.get("open-pull-requests-limit") != 0:
            errors.append(
                f"dependabot.yml: {ecosystem} routine version updates must be disabled"
            )
        if "allow" in config or "groups" in config:
                errors.append(
                    f"dependabot.yml: {ecosystem} must not define routine update groups"
                )


def _check_python_runtime(name: str, source: str, errors: list[str]) -> None:
    versions = re.findall(r"(?m)^\s*python-version:\s*[\"']([^\"']+)[\"']", source)
    for version in versions:
        if version != PYTHON_VERSION:
            errors.append(
                f"{name}: actions/setup-python must pin CPython {PYTHON_VERSION}, found {version}"
            )
    if "py -3.12" in source or "Python 3.12" in source:
        errors.append(f"{name}: legacy Python 3.12 runtime reference remains")
    if name == "release-package.yml":
        if "-PythonPath .\\.venv\\Scripts\\python.exe" not in source:
            errors.append(
                f"{name}: build_exe.ps1 must receive the project .venv interpreter explicitly"
            )
        if "ORT_LIB_LOCATION=$ortLib" not in source:
            errors.append(
                f"{name}: release package must export the pinned CPU ORT library path"
            )
        if "ORT_PREFER_DYNAMIC_LINK=1" not in source:
            errors.append(
                f"{name}: release package must require dynamic CPU ORT linking"
            )


def check_workflows() -> list[str]:
    errors: list[str] = []
    paths = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    if not paths:
        return ["no workflow YAML files found"]

    for path in paths:
        source = path.read_text(encoding="utf-8")
        try:
            document = yaml.safe_load(source)
        except yaml.YAMLError as error:
            errors.append(f"{path.name}: invalid YAML: {error}")
            continue
        document = _mapping(document, path.name, errors)
        _check_permissions(path.name, document, errors)
        _check_required_gates(path.name, source, errors, document=document)
        _check_python_runtime(path.name, source, errors)

        action_refs = ACTION_REF.findall(source)
        if not action_refs:
            errors.append(f"{path.name}: no GitHub Action references found")
        for action, ref in action_refs:
            if COMMIT_SHA.fullmatch(ref) is None:
                errors.append(
                    f"{path.name}: {action}@{ref} is not pinned to a commit SHA"
                )

    _check_dependabot(errors)

    return errors


def main() -> int:
    errors = check_workflows()
    if errors:
        print("Workflow validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(
        "Workflow YAML, Dependabot policy, action pins, permissions, and "
        "release gates are valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
