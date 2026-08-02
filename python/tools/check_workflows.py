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
RUSTSEC_NODE24_SHA = "858dc40f52ca2b8570b7a997c1c4e35c6fc9a432"


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


def _check_required_gates(name: str, source: str, errors: list[str]) -> None:
    if name == "release-promote.yml":
        required = (
            "actions/download-artifact@",
            "release_tag must be an exact vMAJOR.MINOR.PATCH tag",
            "git rev-list -n 1 $env:RELEASE_TAG --",
            "release_provenance.py verify",
            "--expected-archive-sha256",
            "--expected-commit",
            "--report validation/release-qualification.json",
            "--report hardware-matrix/release-hardware-matrix.json",
            "package_smoke.py --dist",
            "gh release upload",
        )
        for needle in required:
            if needle not in source:
                errors.append(
                    f"{name}: missing required promotion gate {needle!r}"
                )
        if "--clobber" in source:
            errors.append(
                f"{name}: promotion must not overwrite published release assets"
            )
        return
    if name == "release-hardware-matrix.yml":
        required = (
            "gh run download",
            "release_tag must be an exact vMAJOR.MINOR.PATCH tag",
            "git rev-list -n 1 $env:RELEASE_TAG --",
            "evaluate_hardware_matrix.py",
            "--expected-archive-sha256",
            "--expected-source-revision",
            "audioforge-release-hardware-matrix-",
        )
        for needle in required:
            if needle not in source:
                errors.append(f"{name}: missing required matrix gate {needle!r}")
        return
    if name == "release-hardware-qualify.yml":
        required = (
            "runs-on: [self-hosted, windows, x64, audioforge-hardware]",
            "actions/download-artifact@",
            "release_tag must be an exact vMAJOR.MINOR.PATCH tag",
            "git rev-list -n 1 $env:RELEASE_TAG --",
            "release_provenance.py verify",
            "--expected-archive-sha256",
            "--expected-commit",
            "evaluate_hardware_validation.py",
            "--confirm-scenario-observed",
            "explicit operator attestation",
            "--health-duration",
            "$duration -lt 1800",
            "audioforge-release-hardware-validation-",
        )
        for needle in required:
            if needle not in source:
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
        "rustsec/audit-check@",
    )
    required = shared
    if name == "release-package.yml":
        required += (
            "python/tools/check_versions.py",
            "python/tools/package_smoke.py --source-only",
            "python/tools/verify_release_assets.py",
            "release_provenance.py create",
            "release_provenance.py verify",
            "release-bundle-path-baseline.json",
        )
        if "--allow-dirty" in source:
            errors.append(
                f"{name}: release candidates must fail closed on dirty source trees"
            )
    for needle in required:
        if needle not in source:
            errors.append(f"{name}: missing required release gate {needle!r}")

    if f"rustsec/audit-check@{RUSTSEC_NODE24_SHA}" not in source:
        errors.append(f"{name}: RustSec must use the pinned Node 24 action revision")
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
        _check_required_gates(path.name, source, errors)

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
