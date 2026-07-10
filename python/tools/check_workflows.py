"""Validate workflow YAML, immutable action pins, and release permissions."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
ACTION_REF = re.compile(r"^\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


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
        if job_name == "publish-release":
            if permissions != {"contents": "write"}:
                errors.append(
                    f"{name}: publish-release must have only contents: write"
                )
            continue
        if isinstance(permissions, dict) and any(
            access == "write" for access in permissions.values()
        ):
            errors.append(f"{name}: job {job_name} must not request write permission")


def _check_required_gates(name: str, source: str, errors: list[str]) -> None:
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
        )
    for needle in required:
        if needle not in source:
            errors.append(f"{name}: missing required release gate {needle!r}")


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

    return errors


def main() -> int:
    errors = check_workflows()
    if errors:
        print("Workflow validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("Workflow YAML, action pins, permissions, and release gates are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
