"""Aggregate exact-artifact hardware cases into the autonomous release gate."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hardware_qualification import (
    MINIMUM_AUTOMATED_BASELINE_CASES,
    REQUIRED_DEVICE_CLASSES,
    REQUIRED_OS_RELEASES,
    REQUIRED_SAMPLE_RATES,
    REQUIRED_SCENARIOS,
    SUPPORTED_DEVICE_CLASSES,
    SUPPORTED_OS_RELEASES,
    SUPPORTED_SAMPLE_RATES,
    SUPPORTED_SCENARIOS,
    coverage_missing,
    validate_case,
)
from release_provenance import sha256_file


def _load_case(
    path: Path,
    expected_archive_sha256: str,
    expected_source_revision: str | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {}, [f"{path.name}: unreadable report ({error})"]
    if not isinstance(report, dict):
        return {}, [f"{path.name}: root must be an object"]
    errors.extend(
        f"{path.name}: {error}"
        for error in validate_case(
            report,
            expected_archive_sha256=expected_archive_sha256,
            expected_source_revision=expected_source_revision,
        )
    )
    return report, errors


def aggregate(
    report_paths: list[Path],
    *,
    expected_archive_sha256: str,
    expected_source_revision: str | None = None,
    output: Path,
    report_root: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    expected_hash = expected_archive_sha256.strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("expected archive SHA-256 must contain 64 lowercase hex digits")
    reports: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    case_ids: set[str] = set()

    def case_field(report: dict[str, Any], field: str, default: Any = "") -> Any:
        case = report.get("case")
        return case.get(field, default) if isinstance(case, dict) else default

    def machine_release(report: dict[str, Any]) -> str:
        machine = report.get("machine")
        value = machine.get("release") if isinstance(machine, dict) else ""
        return value if isinstance(value, str) else ""

    def sample_rate(report: dict[str, Any]) -> int:
        value = case_field(report, "nominal_sample_rate_hz", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    for path in report_paths:
        report, report_errors = _load_case(
            path, expected_hash, expected_source_revision
        )
        errors.extend(report_errors)
        case = report.get("case") if isinstance(report, dict) else None
        case_id = str(case.get("id", "")) if isinstance(case, dict) else ""
        if case_id in case_ids:
            errors.append(f"duplicate hardware case ID: {case_id}")
        elif case_id:
            case_ids.add(case_id)
        reports.append((path, report))

    source_revisions = {
        str(report.get("source_revision", ""))
        for _path, report in reports
        if str(report.get("source_revision", ""))
    }
    if len(source_revisions) > 1:
        errors.append("hardware reports contain multiple source revisions")

    os_releases = {
        machine_release(report)
        for _path, report in reports
        if machine_release(report)
    }
    device_classes = {
        str(case_field(report, "device_class", ""))
        for _path, report in reports
        if isinstance(report.get("case"), dict)
    }
    sample_rates = {
        sample_rate(report)
        for _path, report in reports
        if isinstance(report.get("case"), dict)
    }
    scenarios = {
        str(case_field(report, "scenario", ""))
        for _path, report in reports
        if isinstance(report.get("case"), dict)
    }
    missing = coverage_missing(report for _path, report in reports)
    complete = not errors and all(
        value == 0 or value == [] for value in missing.values()
    )
    resolved_report_root = report_root.resolve() if report_root is not None else None

    def report_label(path: Path) -> str:
        """Retain the downloaded run/artifact directory for source verification."""
        if resolved_report_root is not None:
            try:
                return path.resolve().relative_to(resolved_report_root).as_posix()
            except ValueError:
                errors.append(f"{path.name}: report is outside the declared report root")
        return (Path(path.parent.name) / path.name).as_posix()

    result = {
        "schema_version": 1,
        "qualification_kind": "exact-artifact-hardware-matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if complete else "incomplete",
        "passed": complete,
        "artifact": {"archive_sha256": expected_hash},
        "source_revision": (
            expected_source_revision
            if expected_source_revision is not None
            else next(iter(source_revisions), None)
        ),
        "coverage": {
            "observed": {
                "os_releases": sorted(os_releases),
                "device_classes": sorted(device_classes),
                "nominal_sample_rates_hz": sorted(sample_rates),
                "scenarios": sorted(scenarios),
            },
            "required": {
                "minimum_automated_baseline_cases": MINIMUM_AUTOMATED_BASELINE_CASES,
                "minimum_health_duration_seconds": 1_800,
                "required_os_releases": sorted(REQUIRED_OS_RELEASES),
                "required_device_classes": sorted(REQUIRED_DEVICE_CLASSES),
                "required_nominal_sample_rates_hz": sorted(REQUIRED_SAMPLE_RATES),
                "required_scenarios": sorted(REQUIRED_SCENARIOS),
                "supported_os_releases": sorted(SUPPORTED_OS_RELEASES),
                "supported_device_classes": sorted(SUPPORTED_DEVICE_CLASSES),
                "supported_nominal_sample_rates_hz": sorted(SUPPORTED_SAMPLE_RATES),
                "supported_scenarios": sorted(SUPPORTED_SCENARIOS),
            },
            "missing": missing,
        },
        "cases": [
            {
                "id": str(case_field(report, "id", "")),
                "report_file": report_label(path),
                "report_sha256": sha256_file(path),
                "os_release": machine_release(report),
                "device_class": str(case_field(report, "device_class", "")),
                "nominal_sample_rate_hz": sample_rate(report),
                "scenario": str(case_field(report, "scenario", "")),
                "evidence_kind": str(case_field(report, "evidence_kind", "")),
            }
            for path, report in reports
        ],
        "errors": errors,
        "limitations": [
            "Coverage is release-artifact and hardware specific; source-tree simulations do not satisfy this gate.",
            "Promotion requires the digest-bound baseline plus the required OS, device, sample-rate, and lifecycle dimensions listed in coverage.required.",
            "Optional non-baseline lifecycle cases combine automated health metrics with an explicit operator-observed event.",
            "No device names or endpoint IDs are retained in case or matrix reports.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not complete and not allow_incomplete:
        raise RuntimeError("hardware matrix is incomplete; inspect the generated report")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = aggregate(
        [path.resolve(strict=True) for path in args.reports],
        expected_archive_sha256=args.expected_archive_sha256,
        expected_source_revision=args.expected_source_revision,
        output=args.output.resolve(),
        report_root=args.report_root.resolve() if args.report_root else None,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps({"passed": result["passed"], "coverage": result["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
