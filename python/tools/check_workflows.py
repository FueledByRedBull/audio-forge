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
        "cargo test --locked -p mic_eq_core",
        "cargo clippy --locked -p mic_eq_core --all-targets -- -D warnings",
    ),
    "rust-stress": ("cargo test --release --locked -p mic_eq_core --test stress_tests",),
}
DEV_LOCK_PATH = REPO_ROOT / "requirements" / "dev.txt"
DEEPFILTER_LOCK = "build-support/deepfilter/Cargo.lock"
DEEPFILTER_PROVENANCE_PATH = REPO_ROOT / "build-support" / "deepfilter" / "provenance.json"
# Workflows never run on triggers that give a fork's code a token.
FORBIDDEN_TRIGGERS = frozenset({"pull_request_target", "workflow_run", "issue_comment"})
# Run text may interpolate only step and job outputs; anything user-controlled
# (inputs, event fields, even through env) reaches a script as $env:NAME instead.
RUN_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
SAFE_RUN_EXPRESSION = re.compile(r"(?:steps|needs)\.[\w-]+\.outputs\.[\w-]+")
# release-package.yml hands step and job outputs to its scripts as AF_* variables
# (read as $env:AF_NAME); each such variable must be exactly one of those outputs.
RELEASE_OUTPUT_ENV_PREFIX = "AF_"
# A checkout of a tag input names refs/tags/ (a bare name can resolve to a
# branch of that name) and the job then asserts HEAD is the tag's commit.
TAG_INPUT = re.compile(r"inputs\.\w*tag\b")
TAG_CHECKOUT_REF = re.compile(r"refs/tags/\$\{\{\s*inputs\.\w+\s*\}\}")
HEAD_IS_TAG_COMMIT = re.compile(r"git rev-parse HEAD\)\.Trim\(\)\s+-ne\b")
# Allowed pip_audit --ignore-vuln ids, each with the dev-lock pin that causes it.
# The ignore stops being allowed once the lock moves off that pin.
# CVE-2026-59951: semgrep pins mcp==1.29.0; only its MCP server uses mcp.
# CVE-2026-114603: semgrep pins opentelemetry-api~=1.37.0 (through 1.182.1); the
# ReDoS is in TraceState.from_header for inbound tracestate headers, and semgrep
# only exports telemetry.
AUDIT_IGNORES = {
    "CVE-2026-59951": ("mcp", "1.29.0"),
    "CVE-2026-114603": ("opentelemetry-api", "1.37.0"),
}
CARGO_LOCKED_COMMANDS = ("build", "test", "clippy", "run")
ACTION_REF = re.compile(
    r"^\s*(?:-\s*)?uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE
)
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
PYTHON_VERSION = "3.13.17"
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
# release-promote.yml verifies in a read-only job that runs the candidate and
# publishes from a second job that never runs it; only the second can write.
PROMOTE_JOB_PERMISSIONS = {
    "verify-candidate": {"actions": "read", "contents": "read"},
    "publish-release": {"contents": "write"},
}
PROMOTE_PUBLISH_JOB = "publish-release"
# Candidate code is the executable, the MSI and the pinned pip packages. None of
# it may appear in the publishing job, and the verifying job runs the executable
# and the MSI only after the release files are staged and uploaded.
PROMOTE_CANDIDATE_RUNNERS = ("Start-Process", "msiexec", "msi_smoke.py", "package_smoke.py")
PROMOTE_PUBLISH_FORBIDDEN = (*PROMOTE_CANDIDATE_RUNNERS, "-m pip", "pip install")
PROMOTE_PUBLISH_ONLY_COMMANDS = tuple(
    f"gh release {verb}" for verb in ("create", "upload", "edit", "delete")
)
# The workflows that run privileged or physical work. Each of their jobs starts
# by refusing everyone but a maintainer dispatching from master.
DISPATCH_WORKFLOWS = (
    "release-promote.yml",
    "release-hardware-qualify.yml",
    "release-hardware-matrix.yml",
)
HARDWARE_RUNNER = ["self-hosted", "windows", "x64", "audioforge-hardware"]
# The jobs that declare `environment: release`. Repository settings limit that
# environment to master, which binds a branch copy of the workflow only while
# the copy keeps the line; no other job may use it.
RELEASE_ENVIRONMENT_JOBS = {
    "release-promote.yml": frozenset({PROMOTE_PUBLISH_JOB}),
    "release-hardware-qualify.yml": frozenset({"qualify-hardware"}),
    "release-hardware-matrix.yml": frozenset({"assemble"}),
}
# cache-cleanup.yml deletes a closed pull request's Actions caches. Its one job is
# the only one that may hold actions: write, and it runs no actions or checkout.
CACHE_CLEANUP_WORKFLOW = "cache-cleanup.yml"
CACHE_CLEANUP_JOB = "cleanup"
# weekly-checks.yml re-runs the checks that read the outside world, with nothing
# cached, and publishes nothing.
WEEKLY_WORKFLOW = "weekly-checks.yml"
DISPATCH_AUTHORIZATION = (
    "GH_TOKEN: ${{ github.token }}",
    "DISPATCHER: ${{ github.triggering_actor }}",
    'if ($env:GITHUB_REF -ne "refs/heads/master")',
    "/collaborators/",
    "/permission",
    '$permission.role_name -notin @("admin", "maintain")',
)
CANONICAL_TAG = (
    "release_tag must be the canonical vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rc.N tag"
)
TAG_COMMIT = "git rev-list -n 1 $env:RELEASE_TAG --"
PROVENANCE_GATES = (
    "release_provenance.py verify",
    "--expected-archive-sha256",
    "--expected-commit",
    "--require-source-distribution",
    "--native-attestation",
)
# Per workflow: a label for the messages and, per job, the steps that must exist
# in this order. A step is a tuple of needles that must all sit in one active,
# unconditional step: the first picks the step, "!x" must be absent from it.
# Needles are matched in a step's run lines (comments excluded), its action
# reference and its "KEY: value" env and with entries.
GATE_SEQUENCES: dict[str, tuple[str, dict[str, tuple[tuple[str, ...], ...]]]] = {
    "release-promote.yml": (
        "promotion",
        {
            "verify-candidate": (
                (
                    "actions/checkout@",
                    "ref: refs/tags/${{ inputs.release_tag }}",
                    "persist-credentials: false",
                ),
                (
                    CANONICAL_TAG,
                    TAG_COMMIT,
                    "git cat-file -t $tagRef",
                    '$run.event -ne "workflow_dispatch"',
                ),
                ("--require-hashes -r requirements/runtime.txt",),
                (
                    "actions/download-artifact@",
                    "name: audioforge-release-candidate-${{ inputs.release_tag }}",
                    "run-id: ${{ inputs.candidate_run_id }}",
                ),
                (
                    "actions/download-artifact@",
                    "name: audioforge-release-validation-${{ inputs.release_tag }}",
                    "run-id: ${{ inputs.candidate_run_id }}",
                ),
                # The five release files are staged, hashed and uploaded before the
                # step that runs the candidate's MSI and executable.
                ("Get-ChildItem -LiteralPath candidate", "-deepfilter.provenance.json"),
                (
                    "-evidence.7z",
                    "validation/release-qualification.json",
                    "SHA256SUMS",
                    "EXPECTED_SHA256: ${{ inputs.expected_archive_sha256 }}",
                    "asset_hashes=",
                ),
                (
                    "actions/upload-artifact@",
                    "name: audioforge-verified-release-${{ inputs.release_tag }}",
                    "retention-days: 7",
                ),
                (
                    *PROVENANCE_GATES,
                    "--report validation/release-qualification.json",
                    "source_distribution.py verify",
                    "--require-receipt",
                    "msi_smoke.py",
                    "package_smoke.py --dist",
                    "--smoke-test",
                ),
            ),
            PROMOTE_PUBLISH_JOB: (
                (
                    "actions/checkout@",
                    "ref: ${{ github.sha }}",
                    "persist-credentials: false",
                ),
                (
                    "actions/download-artifact@",
                    "name: audioforge-verified-release-${{ inputs.release_tag }}",
                ),
                (
                    "ASSET_HASHES: ${{ needs.verify-candidate.outputs.asset_hashes }}",
                    "EXPECTED_SHA256: ${{ inputs.expected_archive_sha256 }}",
                    "RELEASE_TAG: ${{ inputs.release_tag }}",
                    'archive = "AudioForge-$tag-win64-ultra.7z"',
                    'checksums = "AudioForge-$tag-SHA256SUMS.txt"',
                    "Get-FileHash",
                ),
                (
                    "Set-Content -LiteralPath $notesPath",
                    "gh release upload",
                    "gh release download",
                    "Assert-AssetMatches $assetPath (Join-Path $verifyDir $assetName) $assetName",
                    "gh release edit $env:RELEASE_TAG --repo $env:GITHUB_REPOSITORY "
                    '--title "AudioForge $version" --notes-file $notesPath',
                    "Assert-ReleaseAssetNames $release $expectedAssetNames -AllowMissing",
                    "Assert-ReleaseAssetNames $release $expectedAssetNames",
                    "Compare-Object -ReferenceObject $ExpectedNames -DifferenceObject $actualNames -CaseSensitive",
                ),
            ),
        },
    ),
    "release-hardware-matrix.yml": (
        "matrix",
        {
            "assemble": (
                ("actions/checkout@", "ref: refs/tags/${{ inputs.release_tag }}"),
                ("gh run download", CANONICAL_TAG, TAG_COMMIT),
                (
                    "evaluate_hardware_matrix.py",
                    "--expected-archive-sha256",
                    "--expected-source-revision",
                ),
                (
                    "actions/upload-artifact@",
                    "name: audioforge-release-hardware-matrix-${{ inputs.release_tag }}",
                ),
            ),
        },
    ),
    "release-hardware-qualify.yml": (
        "hardware",
        {
            "qualify-hardware": (
                ("actions/checkout@", "ref: refs/tags/${{ inputs.release_tag }}"),
                (
                    "actions/download-artifact@",
                    "path: ${{ runner.temp }}/audioforge-release-candidate",
                ),
                (CANONICAL_TAG, TAG_COMMIT, *PROVENANCE_GATES),
                (
                    "evaluate_hardware_validation.py",
                    "--confirm-scenario-observed",
                    "explicit operator attestation",
                    "--health-duration",
                    "$duration -lt 1800",
                    "PYTHONPATH: ${{ github.workspace }}/python",
                    "VAD_MODEL_PATH: ${{ steps.candidate.outputs.vad_model }}",
                ),
                (
                    "actions/upload-artifact@",
                    "name: audioforge-release-hardware-validation-"
                    "${{ inputs.release_tag }}-${{ inputs.case_id }}",
                    "path: ${{ runner.temp }}/release-hardware-qualification.json",
                ),
            ),
        },
    ),
    # Deletes the caches of a closed pull request with the gh CLI alone.
    CACHE_CLEANUP_WORKFLOW: (
        "cache cleanup",
        {
            CACHE_CLEANUP_JOB: (
                (
                    "gh cache list",
                    "PULL_REF: refs/pull/${{ github.event.pull_request.number }}/merge",
                    "gh cache delete",
                ),
            ),
        },
    ),
    # Re-reads sources, assets and advisories with nothing cached; it publishes nothing.
    WEEKLY_WORKFLOW: (
        "weekly",
        {
            "source": (
                (
                    "source_distribution.py bundle",
                    "source_distribution.py verify",
                    "--include-runtime-assets",
                    "--require-receipt",
                    "--revision",
                ),
            ),
            "assets": (
                (
                    "python/tools/fetch_release_assets.py",
                    "!--only-cpu-runtime",
                    "python/tools/verify_release_assets.py",
                    "GH_TOKEN: ${{ github.token }}",
                ),
            ),
            "audit": (
                ("pip install --require-hashes -r requirements/dev.txt",),
                (
                    "pip_audit --require-hashes -r requirements/runtime.txt",
                    "pip_audit --require-hashes -r requirements/dev.txt",
                ),
                ("cargo install cargo-audit --version 0.22.2 --locked --force",),
                (
                    "cargo audit --deny warnings",
                    "--file build-support/deepfilter/Cargo.lock",
                ),
            ),
        },
    ),
    # Builds and tests the bundle and the MSI of the pushed source; it publishes nothing.
    "package-check.yml": (
        "package check",
        {
            "package": (
                ("fetch_release_assets.py --force",),
                ("ORT_LIB_LOCATION=$ortLib", "ORT_PREFER_DYNAMIC_LINK=1"),
                ("maturin develop --release --locked",),
                ("python/tools/verify_release_assets.py",),
                ("build_exe.ps1", "-PythonPath .\\.venv\\Scripts\\python.exe"),
                ("python/tools/package_smoke.py",),
                ("--smoke-test",),
                (
                    "build_msi.ps1",
                    "-AcceptWixEula",
                    "python/tools/msi_smoke.py",
                    "--install-uninstall",
                    "--upgrade-from",
                ),
            ),
        },
    ),
}
# The debug-mode Rust run of ci.yml and release-package.yml enables DeepFilter
# and fails when a DeepFilter or Silero test prints that it skipped itself.
RUST_SKIP_SCAN_STEP = (
    "cargo test --locked -p mic_eq_core",
    "AUDIOFORGE_ENABLE_DEEPFILTER: 1",
    "--nocapture",
    # The step must fail: a non-zero test exit and any skip notice both throw.
    r"re:^\s*if \(\$LASTEXITCODE -ne 0\) \{ throw ",
    r"re:^\s*\$skipped = .*Select-String .*-Pattern '\\b\(skipped\|skipping\)\\b'",
    r"re:^\s*if \(\$skipped\.Count -gt 0\) \{ throw ",
)
# ci.yml's rust-tests also builds df.dll from the pinned source; the release
# workflow's test jobs run on the runtime files its package job built.
CI_RUST_DEEPFILTER_STEPS = (
    # df.dll and its attestation may be restored here (never in the release
    # workflow); the step after it still checks both against the recipe.
    (
        "actions/cache/restore@",
        "target/deepfilter/df.dll.provenance.json",
        "hashFiles('build_deepfilter.ps1', 'build-support/deepfilter/**', "
        "'release-assets.json', 'rust-toolchain.toml')",
    ),
    (
        "python/tools/fetch_release_assets.py",
        "!--only-cpu-runtime",
        "python/tools/verify_release_assets.py",
        "GH_TOKEN: ${{ github.token }}",
    ),
    RUST_SKIP_SCAN_STEP,
)
# ci.yml's static job lints the workflow definitions with the locked zizmor.
CI_STATIC_STEPS = (("zizmor --no-progress .github/workflows", "GH_TOKEN: ${{ github.token }}"),)


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
        if name == CACHE_CLEANUP_WORKFLOW and job_name == CACHE_CLEANUP_JOB:
            if permissions != {"actions": "write"}:
                errors.append(f"{name}: {job_name} must have only actions: write")
            continue
        if name == "release-promote.yml" and job_name in PROMOTE_JOB_PERMISSIONS:
            if permissions != PROMOTE_JOB_PERMISSIONS[job_name]:
                errors.append(
                    f"{name}: {job_name} must have only "
                    + " and ".join(
                        f"{scope}: {access}"
                        for scope, access in PROMOTE_JOB_PERMISSIONS[job_name].items()
                    )
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


def _job_steps(job_id: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one job's enabled, blocking workflow steps."""
    return _active_steps({"jobs": {job_id: job}})


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
    for program in ("cargo", "gh"):
        if marker.startswith(f"{program} "):
            command = marker.removeprefix(f"{program} ")
            return any(
                re.search(
                    # A regular expression: its operator runs are not obfuscated code.
                    rf"(?i)(?:^\s*|[;&|]\s*)(?:&\s*)?(?:[^\s]+[/\\])?{program}(?:\.exe)?\s+{re.escape(command)}(?:\s|$)",  # nosemgrep: generic-obfuscation-operators
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

    if name in GATE_SEQUENCES:
        label, sequence = GATE_SEQUENCES[name]
        _check_step_sequences(name, label, document, sequence, errors)
        if name in DISPATCH_WORKFLOWS:
            _check_dispatch_authorization(name, document, errors)
            _check_release_environment(name, document, errors)
        if name == "release-promote.yml":
            _check_promotion_jobs(name, document, errors)
            upload_lines = (
                line
                for line in source.splitlines()
                if re.search(r"\bgh\s+release\s+upload\b", line)
            )
            if any("--clobber" in line for line in upload_lines):
                errors.append(
                    f"{name}: promotion must not overwrite published release assets"
                )
        if name == "release-hardware-qualify.yml":
            job = _mapping(document.get("jobs"), f"{name}: jobs", errors).get("qualify-hardware")
            if not isinstance(job, dict) or job.get("runs-on") != HARDWARE_RUNNER:
                errors.append(
                    f"{name}: missing required hardware gate "
                    f"'runs-on: [{', '.join(HARDWARE_RUNNER)}]'"
                )
        if name == "package-check.yml":
            _check_package_check_publishes_nothing(name, document, errors)
        if name == CACHE_CLEANUP_WORKFLOW:
            _check_cache_cleanup(name, source, document, errors)
        if name == WEEKLY_WORKFLOW:
            _check_weekly_checks(name, document, errors)
        return

    shared = (
        "pip_audit --require-hashes -r requirements/runtime.txt",
        "pip_audit --require-hashes -r requirements/dev.txt",
        "run_semgrep.py",
        "cargo test --release --locked -p mic_eq_core --test stress_tests",
        "cargo clippy --locked -p mic_eq_core --all-targets -- -D warnings",
        "cargo install cargo-audit --version 0.22.2 --locked --force",
        "cargo audit --deny warnings",
    )
    required = shared
    if name == "release-package.yml":
        rust_test_steps = [
            step for step in _active_steps(document)
            if _active_run_has_marker(
                str(step.get("run", "")).splitlines(), "cargo test --locked -p mic_eq_core"
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
    if name != "release-package.yml" and "cargo test --locked -p mic_eq_core" in source:
        model_step = source.find("silero_vad.onnx")
        rust_tests = source.find("cargo test --locked -p mic_eq_core")
        if model_step < 0 or model_step > rust_tests:
            errors.append(
                f"{name}: pinned Silero model must be available before Rust tests"
            )
    if name == "ci.yml":
        _check_ci_jobs(document, errors)
        _check_pytest_shards(document, errors)


def _is_output_expression(value: Any) -> bool:
    """Whether `value` is exactly one step or job output expression."""
    match = RUN_EXPRESSION.fullmatch(str(value).strip())
    return match is not None and SAFE_RUN_EXPRESSION.fullmatch(match.group(1).strip()) is not None


def _step_text(step: dict[str, Any]) -> str:
    entries = [
        f"{key}: {str(value).lower() if isinstance(value, bool) else value}"
        for section in ("env", "with")
        if isinstance(step.get(section), dict)
        for key, value in step[section].items()
    ]
    return "\n".join([str(step.get("uses", "")), *entries, *_step_lines(step)])


def _step_has(step: dict[str, Any], needle: str) -> bool:
    """Whether an active step carries `needle`.

    "!x" means "x" is absent, "a@" an action, "re:x" a regular expression that
    matches one run line, and anything else a substring.
    """
    if needle.startswith("!"):
        return not _step_has(step, needle[1:])
    if needle.startswith("re:"):
        return any(re.search(needle[3:], line) for line in _step_lines(step))
    if needle.endswith("@"):
        return str(step.get("uses", "")).startswith(needle)
    if any(marker in needle for marker in RUN_REQUIRED_MARKERS):
        return _active_run_has_marker(_step_lines(step), needle)
    return needle in _step_text(step)


def _check_step_sequences(
    name: str,
    label: str,
    document: dict[str, Any],
    sequence: dict[str, tuple[tuple[str, ...], ...]],
    errors: list[str],
) -> None:
    """Require each job's gate steps to exist, be enabled, and run in the listed order."""
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    for job_id, step_specs in sequence.items():
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            errors.append(f"{name}: required job {job_id} is missing")
            continue
        steps = _job_steps(job_id, job)
        cursor = -1
        for needles in step_specs:
            selectors = [
                needle
                for position, needle in enumerate(needles)
                if position == 0 or needle.startswith("!")
            ]
            matches = [
                index
                for index, step in enumerate(steps)
                if all(_step_has(step, selector) for selector in selectors)
            ]
            if not matches:
                errors.append(f"{name}: {job_id}: missing required {label} gate {needles[0]!r}")
                continue
            later = [index for index in matches if index > cursor]
            if not later:
                errors.append(
                    f"{name}: {job_id}: {needles[0]!r} must run after the gate before it"
                )
                continue
            cursor = later[0]
            step = steps[cursor]
            if "if" in step and not _is_always(step["if"]):
                errors.append(
                    f"{name}: {job_id}: the {label} gate {needles[0]!r} must be unconditional"
                )
            errors.extend(
                f"{name}: {job_id}: missing required {label} gate {needle!r}"
                for needle in needles[1:]
                if not needle.startswith("!") and not _step_has(step, needle)
            )


def _check_dispatch_authorization(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    """Every job starts by refusing all but a maintainer dispatching from master."""
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    for job_id, job in jobs.items():
        job = job if isinstance(job, dict) else {}
        steps = job.get("steps")
        first = steps[0] if isinstance(steps, list) and steps and isinstance(steps[0], dict) else {}
        missing = [needle for needle in DISPATCH_AUTHORIZATION if not _step_has(first, needle)]
        if missing:
            errors.append(
                f"{name}: {job_id}: the first step must authorize the dispatch; missing {missing}"
            )
        if any(key in scope for scope in (first, job) for key in ("if", "continue-on-error")):
            errors.append(
                f"{name}: {job_id}: the dispatch authorization must be unconditional and blocking"
            )


def _check_release_environment(name: str, document: dict[str, Any], errors: list[str]) -> None:
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    for job_id, job in jobs.items():
        bound = job_id in RELEASE_ENVIRONMENT_JOBS[name]
        if (job.get("environment") if isinstance(job, dict) else None) != ("release" if bound else None):
            errors.append(
                f"{name}: {job_id} must {'use' if bound else 'not use'} environment: release"
            )


def _check_promotion_jobs(name: str, document: dict[str, Any], errors: list[str]) -> None:
    """Only the publishing job writes or runs gh release, and it never runs the candidate."""
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        publishes = job_id == PROMOTE_PUBLISH_JOB
        lines = _active_run_lines({"jobs": {job_id: job}})
        if publishes:
            needs = job.get("needs")
            if (needs if isinstance(needs, list) else [needs]) != ["verify-candidate"]:
                errors.append(f"{name}: {job_id} must need exactly verify-candidate")
            runs_candidate = [
                marker for marker in PROMOTE_PUBLISH_FORBIDDEN
                if any(marker in line for line in lines)
            ] + [
                use for use in _active_uses({"jobs": {job_id: job}})
                if use.startswith("actions/setup-python@")
            ]
            if runs_candidate:
                errors.append(f"{name}: {job_id} must not run candidate code ({runs_candidate})")
            continue
        errors.extend(
            f"{name}: only {PROMOTE_PUBLISH_JOB} may run {command!r}"
            for command in PROMOTE_PUBLISH_ONLY_COMMANDS
            if _active_run_has_marker(lines, command)
        )
        steps = _job_steps(job_id, job)
        staged = next(
            (
                index
                for index, step in enumerate(steps)
                if str(step.get("uses", "")).startswith("actions/upload-artifact@")
            ),
            len(steps),
        )
        early = [
            marker
            for marker in PROMOTE_CANDIDATE_RUNNERS
            if any(marker in line for step in steps[:staged] for line in _step_lines(step))
        ]
        if early:
            errors.append(
                f"{name}: {job_id} runs candidate code before the release files are "
                f"staged and uploaded ({early})"
            )


def _check_package_check_publishes_nothing(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    lines = _active_run_lines(document)
    forbidden = [
        marker for marker in ("gh release", "git push") if _active_run_has_marker(lines, marker)
    ] + [use for use in _active_uses(document) if use.startswith("actions/upload-artifact@")]
    if forbidden:
        errors.append(f"{name}: must not publish or upload anything ({forbidden})")


def _check_cache_cleanup(
    name: str, source: str, document: dict[str, Any], errors: list[str]
) -> None:
    """The cleanup job holds actions: write, so it stays a bare gh script."""
    triggers = next((value for key, value in document.items() if key in ("on", True)), None)
    if triggers != {"pull_request": {"types": ["closed"]}}:
        errors.append(f"{name}: must trigger only on pull_request closed")
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    if list(jobs) != [CACHE_CLEANUP_JOB]:
        errors.append(f"{name}: must define only the {CACHE_CLEANUP_JOB} job")
    if ACTION_REF.search(source):
        errors.append(f"{name}: must not use actions or check out code")


def _check_weekly_checks(name: str, document: dict[str, Any], errors: list[str]) -> None:
    """The weekly run is scheduled, publishes nothing and hydrates source uncached."""
    triggers = next((value for key, value in document.items() if key in ("on", True)), None)
    if not isinstance(triggers, dict) or sorted(triggers) != ["schedule", "workflow_dispatch"]:
        errors.append(f"{name}: must trigger only on schedule and workflow_dispatch")
    _check_package_check_publishes_nothing(name, document, errors)
    jobs = _mapping(document.get("jobs"), f"{name}: jobs", errors)
    source = jobs.get("source")
    if isinstance(source, dict) and any(
        use.startswith("actions/cache") for use in _active_uses({"jobs": {"source": source}})
    ):
        errors.append(f"{name}: source must hydrate without the Actions cache")


def _check_cargo_locked(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    """Cargo commands that resolve this repository's Cargo.lock must not update it."""
    for line in _active_run_lines(document):
        for command in CARGO_LOCKED_COMMANDS:
            # Arguments after a bare `--` go to the test binary, not to cargo.
            if _active_run_has_marker([line], f"cargo {command}") and (
                "--locked" not in line.split(" -- ")[0].split()
            ):
                errors.append(f"{name}: 'cargo {command}' must pass --locked: {line.strip()}")


def _joined_run_lines(document: dict[str, Any]) -> list[str]:
    """Run lines with PowerShell backtick continuations joined into one command."""
    joined = re.sub(r"`[ \t]*\n\s*", " ", "\n".join(_active_run_lines(document)))
    return joined.splitlines()


def _reviewed_advisories() -> set[str]:
    warnings = json.loads(DEEPFILTER_PROVENANCE_PATH.read_text(encoding="utf-8"))[
        "audit"
    ]["allowed_warnings"]
    return {warning["advisory"] for warning in warnings}


def _check_rustsec_ignores(
    name: str, document: dict[str, Any], reviewed: set[str], errors: list[str]
) -> None:
    """Allow only the DeepFilter advisories reviewed in provenance.json, and only there."""
    audits_deepfilter = False
    for line in _joined_run_lines(document):
        if not _active_run_has_marker([line], "cargo audit"):
            continue
        audits_deepfilter |= DEEPFILTER_LOCK in line
        ignored = set(re.findall(r"--ignore\s+(\S+)", line))
        expected = reviewed if DEEPFILTER_LOCK in line else set()
        if ignored != expected:
            errors.append(
                f"{name}: cargo audit ignores {sorted(ignored)}, expected {sorted(expected)}"
            )
    if name in ("ci.yml", "release-package.yml", WEEKLY_WORKFLOW) and not audits_deepfilter:
        errors.append(f"{name}: missing a cargo audit of {DEEPFILTER_LOCK}")


def _check_triggers_and_expressions(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    """Reject privileged triggers and user-controlled expressions in run text."""
    # YAML reads the bare key `on` as the boolean True.
    triggers = next((value for key, value in document.items() if key in ("on", True)), None)
    names = [triggers] if isinstance(triggers, str) else list(triggers or [])
    for trigger in names:
        if trigger in FORBIDDEN_TRIGGERS:
            errors.append(f"{name}: trigger {trigger} is not allowed")
    jobs = document.get("jobs")
    job_list = [job for job in jobs.values() if isinstance(job, dict)] if isinstance(jobs, dict) else []
    if name in DISPATCH_WORKFLOWS and names != ["workflow_dispatch"]:
        errors.append(
            f"{name}: only workflow_dispatch may trigger this workflow, found {names}"
        )
    if names != ["workflow_dispatch"] and any(
        "self-hosted" in str(job.get("runs-on")) for job in job_list
    ):
        errors.append(f"{name}: a self-hosted runner may be used only by workflow_dispatch")
    for job in job_list:
        for step in job.get("steps", []):
            for expression in RUN_EXPRESSION.findall(str(step.get("run", ""))):
                if SAFE_RUN_EXPRESSION.fullmatch(expression.strip()) is None:
                    errors.append(
                        f"{name}: run text must not interpolate {expression.strip()!r}; "
                        "pass it through env"
                    )


def _check_checkout_credentials(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    """Every job installs packages or runs repository code, so no checkout keeps its token."""
    jobs = document.get("jobs")
    for job_id, job in jobs.items() if isinstance(jobs, dict) else ():
        for step in (job.get("steps") if isinstance(job, dict) else None) or []:
            if (
                isinstance(step, dict)
                and str(step.get("uses", "")).startswith("actions/checkout@")
                and (step.get("with") or {}).get("persist-credentials") is not False
            ):
                errors.append(f"{name}: {job_id}: actions/checkout must set persist-credentials: false")


def _check_tag_checkouts(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    """Tag-input checkouts must use refs/tags/ and be followed by a HEAD-is-tag-commit assertion."""
    jobs = document.get("jobs")
    for job_id, job in jobs.items() if isinstance(jobs, dict) else ():
        steps = [step for step in (job.get("steps") if isinstance(job, dict) else None) or [] if isinstance(step, dict)]
        runs = "\n".join(str(step.get("run", "")) for step in steps)
        for step in steps:
            ref = str((step.get("with") or {}).get("ref", ""))
            # release-package.yml's optional-tag ref has its own tag-commit gate.
            if (
                not str(step.get("uses", "")).startswith("actions/checkout@")
                or TAG_INPUT.search(ref) is None
                or ref == RELEASE_SOURCE_REF
            ):
                continue
            if TAG_CHECKOUT_REF.fullmatch(ref) is None:
                errors.append(f"{name}: {job_id}: checkout of {ref} must use refs/tags/")
            elif HEAD_IS_TAG_COMMIT.search(runs) is None:
                errors.append(
                    f"{name}: {job_id}: checkout of {ref} must be followed by a check "
                    "that HEAD is the tag commit"
                )


def _check_audit_ignores(
    name: str, document: dict[str, Any], errors: list[str]
) -> None:
    """Allow only the pip_audit ignores in AUDIT_IGNORES, and only on the dev lock."""
    for line in _joined_run_lines(document):
        # Any pip_audit invocation is in scope, whatever order its flags take.
        if not _active_run_has_marker([line], "pip_audit "):
            continue
        tokens = [token.strip("\"'") for token in line.split()]
        ignored = []
        for position, token in enumerate(tokens):
            # argparse accepts any prefix of --ignore-vuln from --ig up.
            flag, equals, value = token.partition("=")
            if len(flag) >= 4 and "--ignore-vuln".startswith(flag):
                ignored.append(value if equals else "".join(tokens[position + 1:position + 2]))
        if ignored and re.search(r"requirements[/\\]runtime\.txt", line):
            errors.append(f"{name}: the runtime lock audit must not ignore {ignored}")
            continue
        for vuln in ignored:
            if vuln not in AUDIT_IGNORES:
                errors.append(f"{name}: {vuln} is not an allowed audit ignore")
                continue
            package, version = AUDIT_IGNORES[vuln]
            try:
                lock = DEV_LOCK_PATH.read_text(encoding="utf-8")
            except OSError as error:
                errors.append(f"{name}: cannot read {DEV_LOCK_PATH.name}: {error}")
                continue
            if re.search(rf"(?m)^{re.escape(package)}=={re.escape(version)}(?:\s|$)", lock) is None:
                errors.append(
                    f"{name}: audit ignore {vuln} needs {DEV_LOCK_PATH.name} to pin "
                    f"{package}=={version}; remove the ignore"
                )


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
    _check_step_sequences(
        "ci.yml", "DeepFilter", document, {"rust-tests": CI_RUST_DEEPFILTER_STEPS}, errors
    )
    _check_step_sequences(
        "ci.yml", "workflow lint", document, {"static": CI_STATIC_STEPS}, errors
    )
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
        return _job_steps(job_name, job) if isinstance(job, dict) else []

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
    _check_step_sequences(name, "DeepFilter", document, {"rust-tests": (RUST_SKIP_SCAN_STEP,)}, errors)

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

    for job_name in jobs:
        for step in job_steps(job_name):
            env = step.get("env")
            for key, value in (env.items() if isinstance(env, dict) else ()):
                if str(key).startswith(RELEASE_OUTPUT_ENV_PREFIX) and not _is_output_expression(value):
                    fail(f"{job_name}: {key} must be exactly one step or job output")

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
    versions = re.findall(r"(?m)^\s*python-version:\s*[\"']?([^\"'\s#]+)", source)
    setups = len(re.findall(r"(?m)^\s*(?:-\s*)?uses:\s*actions/setup-python@", source))
    if setups != len(versions):
        errors.append(f"{name}: every actions/setup-python step must set python-version")
    for version in versions:
        if version != PYTHON_VERSION:
            errors.append(
                f"{name}: actions/setup-python must pin CPython {PYTHON_VERSION}, found {version}"
            )
    if "py -3.12" in source or "Python 3.12" in source:
        errors.append(f"{name}: legacy Python 3.12 runtime reference remains")
    if name in ("release-package.yml", "package-check.yml"):
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

    try:
        reviewed = _reviewed_advisories()
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(f"cannot read the reviewed advisories in provenance.json: {error}")
        reviewed = None

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
        _check_cargo_locked(path.name, document, errors)
        _check_audit_ignores(path.name, document, errors)
        if reviewed is not None:
            _check_rustsec_ignores(path.name, document, reviewed, errors)
        _check_triggers_and_expressions(path.name, document, errors)
        _check_tag_checkouts(path.name, document, errors)
        _check_checkout_credentials(path.name, document, errors)
        _check_python_runtime(path.name, source, errors)

        action_refs = ACTION_REF.findall(source)
        if not action_refs and path.name != CACHE_CLEANUP_WORKFLOW:
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
