"""Validate portable, reproducible AudioForge evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = REPO_ROOT / "evaluation"
MAX_TRACKED_REPORT_BYTES = 100_000
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")
POSIX_HOME = re.compile(r"^/(?:Users|home)/[^/]+(?:/|$)")
AUDIBLE_CONTRACT_FIELDS = {
    "configuration",
    "asset_hashes",
    "runtime",
    "latency",
    "clean_preservation",
}
DEVICE_PSEUDONYM = re.compile(r"^device-[0-9a-f]{16}$")
PORTABLE_TEXT_SUFFIXES = {
    ".bat",
    ".c",
    ".h",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".pyi",
    ".rs",
    ".toml",
    ".yaml",
    ".yml",
}


def _portable_source_sha256(path: Path) -> set[str]:
    data = path.read_bytes()
    hashes = {hashlib.sha256(data).hexdigest()}
    if path.suffix.casefold() not in PORTABLE_TEXT_SUFFIXES or b"\0" in data:
        return hashes
    lf = data.replace(b"\r\n", b"\n")
    hashes.add(hashlib.sha256(lf).hexdigest())
    hashes.add(hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())
    return hashes


def _declared_source_hashes(report: dict[str, Any]) -> list[tuple[str, str]]:
    containers: list[Any] = [report.get("source_sha256")]
    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        containers.append(provenance.get("source_hashes"))
    found: list[tuple[str, str]] = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        for raw_path, expected in container.items():
            if isinstance(raw_path, str) and isinstance(expected, str):
                found.append((raw_path, expected))
    return found


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
    if path.stat().st_size > MAX_TRACKED_REPORT_BYTES:
        errors.append(
            f"{path}: tracked report exceeds {MAX_TRACKED_REPORT_BYTES} bytes; "
            "move per-case detail to an ignored --details-output artifact"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path}: invalid JSON: {error}"]

    if not isinstance(report, dict):
        return [f"{path}: report root must be an object"]

    for location, value in _walk_strings(report):
        if WINDOWS_ABSOLUTE.match(value) or POSIX_HOME.match(value):
            errors.append(f"{path}:{location}: machine-local absolute path: {value!r}")

    declared_source_hashes = _declared_source_hashes(report)
    for raw_path, expected in declared_source_hashes:
        source_path = Path(raw_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append(f"{path}: non-portable source hash path: {raw_path!r}")
            continue
        resolved = REPO_ROOT / source_path
        if not resolved.is_file():
            errors.append(f"{path}: declared source file is missing: {raw_path}")
        elif not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{path}: invalid source SHA-256 for {raw_path}")
        elif expected not in _portable_source_sha256(resolved):
            errors.append(f"{path}: stale source SHA-256 for {raw_path}")

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
        if not declared_source_hashes:
            errors.append(
                f"{path}: audible-change report lacks verifiable source SHA-256 hashes"
            )
    if path.name.startswith("hardware-validation"):
        errors.extend(_validate_hardware_report_privacy(path, report))
    return errors


def _validate_hardware_report_privacy(
    path: Path | str,
    report: dict[str, Any],
) -> list[str]:
    routes = report.get("routes")
    if not isinstance(routes, dict):
        return []
    errors: list[str] = []
    for route_name, route in routes.items():
        if not isinstance(route, dict):
            errors.append(f"{path}: routes.{route_name} must be an object")
            continue
        for direction in ("input", "output"):
            value = route.get(direction)
            if not isinstance(value, str) or DEVICE_PSEUDONYM.fullmatch(value) is None:
                errors.append(
                    f"{path}: routes.{route_name}.{direction} must use a "
                    "report-local device pseudonym"
                )
    if int(report.get("schema_version", 0)) < 3:
        redaction = report.get("privacy_redaction")
        if not isinstance(redaction, dict) or redaction.get("applied") is not True:
            errors.append(
                f"{path}: historical hardware report lacks privacy-redaction provenance"
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
        release_status = release.get("status")
        if release_status not in {"candidate", "published"}:
            errors.append(f"{location}: status must be candidate or published")
        if release_status == "published" and not re.fullmatch(
            r"[0-9a-f]{40}", str(release.get("commit", ""))
        ):
            errors.append(f"{location}: published rows require an exact commit")
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
        hardware_metric = release.get("hardware")
        if (
            isinstance(hardware_metric, dict)
            and hardware_metric.get("status") == "measured"
        ):
            hardware_value = hardware_metric.get("value")
            if not isinstance(hardware_value, dict):
                errors.append(f"{location}.hardware: measured value must be an object")
            else:
                errors.extend(
                    _validate_hardware_report_privacy(
                        f"{location}.hardware.value", hardware_value
                    )
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
