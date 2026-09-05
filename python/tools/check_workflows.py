"""Validate workflow YAML, immutable action pins, and release permissions."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"
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
        if isinstance(permissions, dict) and any(
            access == "write" for access in permissions.values()
        ):
            errors.append(f"{name}: job {job_name} must not request write permission")


def _is_boolean_expression(value: Any, expected: bool) -> bool:
    if value is expected:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    word = "true" if expected else "false"
    return normalized in {word, f"${{{{ {word} }}}}"}


def _active_run_lines(document: dict[str, Any]) -> list[str]:
    """Return executable run lines after excluding disabled/non-blocking jobs."""
    active: list[str] = []
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
            if not isinstance(step, dict) or not isinstance(step.get("run"), str):
                continue
            if _is_boolean_expression(step.get("if"), False):
                continue
            if _is_boolean_expression(step.get("continue-on-error"), True):
                continue
            active.extend(
                line
                for line in step["run"].splitlines()
                if not line.lstrip().startswith("#")
            )
    return active


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
    if ".py" in marker:
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
            "--report validation/release-qualification.json",
            "--report hardware-matrix/release-hardware-matrix.json",
            "--matrix-report-root hardware-matrix/case-reports",
            "--native-attestation",
            "source_distribution.py verify",
            "--require-receipt",
            "validation/release-qualification.json",
            "hardware-matrix/case-reports",
            "evidenceArchive",
            "evidenceChecksum",
            "release-evidence",
            "package_smoke.py --dist",
            "--smoke-test",
            "gh release upload",
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
        required += (
            "python/tools/check_versions.py",
            "python/tools/package_smoke.py --source-only",
            "python/tools/verify_release_assets.py",
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
    for needle in required:
        if not has_gate(needle):
            errors.append(f"{name}: missing required release gate {needle!r}")

    if "cargo test -p mic_eq_core" in source:
        model_step = source.find(
            "fetch_release_assets.py"
            if name == "release-package.yml"
            else "silero_vad.onnx"
        )
        rust_tests = source.find("cargo test -p mic_eq_core")
        if model_step < 0 or model_step > rust_tests:
            errors.append(
                f"{name}: pinned Silero model must be available before Rust tests"
            )
    if name == "release-package.yml":
        asset_fetch = source.find("fetch_release_assets.py")
        extension_build = source.find("maturin develop --release")
        if asset_fetch < 0 or asset_fetch > extension_build:
            errors.append(
                f"{name}: verified runtime assets must be fetched before extension build"
            )
    if name == "ci.yml":
        cpu_hydration = "fetch_release_assets.py --only-cpu-runtime --force"
        if source.count(cpu_hydration) < 2:
            errors.append(
                f"{name}: both Python and Rust jobs must hydrate the pinned CPU ONNX Runtime"
            )


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
