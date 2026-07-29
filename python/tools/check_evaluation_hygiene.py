"""Validate portable, reproducible AudioForge evaluation evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = REPO_ROOT / "evaluation"
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")
POSIX_HOME = re.compile(r"^/(?:Users|home)/[^/]+(?:/|$)")
AUDIBLE_CONTRACT_FIELDS = {
    "configuration",
    "asset_hashes",
    "runtime",
    "latency",
    "clean_preservation",
    "listening_status",
}


def _walk_strings(value: Any, location: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((location, value))
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk_strings(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{location}[{index}]"))
    return found


def validate_report(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path}: invalid JSON: {error}"]

    if not isinstance(report, dict):
        return [f"{path}: report root must be an object"]

    for location, value in _walk_strings(report):
        if WINDOWS_ABSOLUTE.match(value) or POSIX_HOME.match(value):
            errors.append(f"{path}:{location}: machine-local absolute path: {value!r}")

    if report.get("audible_change") is True:
        if int(report.get("schema_version", 0)) < 2:
            errors.append(f"{path}: audible-change reports require schema_version >= 2")
        contract = report.get("evaluation_contract")
        if not isinstance(contract, dict):
            errors.append(f"{path}: audible-change report lacks evaluation_contract")
        else:
            missing = sorted(AUDIBLE_CONTRACT_FIELDS - contract.keys())
            if missing:
                errors.append(
                    f"{path}: evaluation_contract lacks {', '.join(missing)}"
                )
            runtime = contract.get("runtime")
            if not isinstance(runtime, dict) or "max_p99_frame_seconds" not in runtime:
                errors.append(
                    f"{path}: evaluation_contract.runtime lacks max_p99_frame_seconds"
                )
            listening = contract.get("listening_status")
            if not isinstance(listening, dict) or listening.get("status") not in {
                "passed",
                "failed",
                "not_run",
            }:
                errors.append(
                    f"{path}: listening_status must explicitly be passed, failed, or not_run"
                )
            if (
                isinstance(listening, dict)
                and listening.get("status") == "not_run"
                and not listening.get("reason")
            ):
                errors.append(
                    f"{path}: listening_status not_run requires an explanatory reason"
                )
    return errors


def validate_evaluation_tree(root: Path = DEFAULT_EVALUATION_ROOT) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.glob("*.json")):
        errors.extend(validate_report(path))
        if path.name == "release-trends.json":
            errors.extend(validate_release_trends(path))
    return errors


def validate_release_trends(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        trends = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path}: invalid release trends JSON: {error}"]
    releases = trends.get("releases")
    if not isinstance(releases, list):
        return [f"{path}: releases must be a list"]
    versions: set[str] = set()
    for index, release in enumerate(releases):
        location = f"{path}:releases[{index}]"
        if not isinstance(release, dict):
            errors.append(f"{location}: entry must be an object")
            continue
        version = release.get("version")
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            errors.append(f"{location}: version must be semantic X.Y.Z")
        elif version in versions:
            errors.append(f"{location}: duplicate version {version}")
        else:
            versions.add(version)
        for category in ("package", "runtime", "quality", "hardware"):
            if category not in release:
                errors.append(f"{location}: missing {category}")
        for metric_location, metric in _measurement_nodes(release):
            status = metric.get("status")
            if status not in {"measured", "not_measured"}:
                errors.append(f"{location}.{metric_location}: invalid measurement status")
            elif status == "measured" and "value" not in metric:
                errors.append(f"{location}.{metric_location}: measured value is missing")
            elif status == "not_measured" and not metric.get("reason"):
                errors.append(
                    f"{location}.{metric_location}: not_measured requires a reason"
                )
    return errors


def _measurement_nodes(
    release: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    nodes: list[tuple[str, dict[str, Any]]] = []
    for category in ("runtime", "quality", "hardware"):
        metric = release.get(category)
        if isinstance(metric, dict):
            nodes.append((category, metric))
    package = release.get("package")
    if isinstance(package, dict):
        for name in ("bundle", "archive"):
            metric = package.get(name)
            if isinstance(metric, dict):
                nodes.append((f"package.{name}", metric))
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    args = parser.parse_args()
    errors = validate_evaluation_tree(args.evaluation_root)
    if errors:
        print("Evaluation hygiene check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Evaluation hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
