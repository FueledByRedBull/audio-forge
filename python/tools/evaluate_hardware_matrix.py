"""Aggregate exact-artifact hardware cases into the autonomous release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_OS_RELEASES = frozenset({"10", "11"})
SUPPORTED_DEVICE_CLASSES = frozenset({"built_in", "usb", "virtual", "other"})
SUPPORTED_SAMPLE_RATES = frozenset({44_100, 48_000})
SUPPORTED_SCENARIOS = frozenset(
    {
        "baseline",
        "device_reconnect",
        "default_device_change",
        "sleep_resume",
        "buffer_negotiation",
        "route_change",
    }
)
MINIMUM_AUTOMATED_BASELINE_CASES = 1
PSEUDONYM = re.compile(r"^device-[0-9a-f]{16}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if report.get("schema_version") != 3:
        errors.append(f"{path.name}: hardware case schema must be 3")
    if report.get("qualification_kind") != "exact-artifact-hardware":
        errors.append(f"{path.name}: wrong qualification kind")
    if report.get("passed") is not True or report.get("status") != "passed":
        errors.append(f"{path.name}: case did not pass")
    source_revision = report.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        errors.append(f"{path.name}: source revision is missing")
    elif (
        expected_source_revision is not None
        and source_revision != expected_source_revision
    ):
        errors.append(f"{path.name}: source revision differs from the release tag")
    artifact = report.get("artifact")
    if not isinstance(artifact, dict):
        errors.append(f"{path.name}: exact artifact provenance is missing")
    elif str(artifact.get("archive_sha256", "")).casefold() != expected_archive_sha256:
        errors.append(f"{path.name}: archive SHA-256 differs from the matrix candidate")
    case = report.get("case")
    if not isinstance(case, dict) or not str(case.get("id", "")):
        errors.append(f"{path.name}: case metadata is missing")
    else:
        device_class = case.get("device_class")
        sample_rate = case.get("nominal_sample_rate_hz")
        scenario = case.get("scenario")
        evidence_kind = case.get("evidence_kind")
        if device_class not in SUPPORTED_DEVICE_CLASSES:
            errors.append(f"{path.name}: unsupported device class")
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            errors.append(f"{path.name}: unsupported nominal sample rate")
        if scenario not in SUPPORTED_SCENARIOS:
            errors.append(f"{path.name}: unsupported lifecycle scenario")
        if evidence_kind not in {"automated", "operator_observed"}:
            errors.append(f"{path.name}: unsupported evidence kind")
        if scenario == "baseline" and evidence_kind != "automated":
            errors.append(f"{path.name}: baseline case must use automated evidence")
        if scenario != "baseline" and evidence_kind != "operator_observed":
            errors.append(f"{path.name}: lifecycle scenario lacks operator evidence")
        if scenario != "baseline" and case.get("operator_attestation") is not True:
            errors.append(f"{path.name}: lifecycle scenario lacks operator attestation")
        if case.get("scenario_evidence_valid") is not True:
            errors.append(f"{path.name}: scenario evidence was not validated")
    machine = report.get("machine")
    if (
        not isinstance(machine, dict)
        or str(machine.get("release", "")) not in SUPPORTED_OS_RELEASES
    ):
        errors.append(f"{path.name}: unsupported or missing Windows release")
    duration = report.get("requested_health_duration_seconds")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration < 1_800.0
    ):
        errors.append(f"{path.name}: sustained health duration is below 1800 seconds")
    for field in (
        "package_smoke",
        "executable_startup",
        "model_discovery",
        "selected_route_correlation",
        "sustained_health",
    ):
        check = report.get(field)
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"{path.name}: {field} did not pass")
    routes = report.get("routes")
    if not isinstance(routes, dict):
        errors.append(f"{path.name}: route pseudonyms are missing")
    else:
        for route in routes.values():
            if not isinstance(route, dict):
                errors.append(f"{path.name}: invalid route record")
                continue
            for value in route.values():
                if not isinstance(value, str) or PSEUDONYM.fullmatch(value) is None:
                    errors.append(f"{path.name}: raw or invalid device identity in report")
    return report, errors


def aggregate(
    report_paths: list[Path],
    *,
    expected_archive_sha256: str,
    expected_source_revision: str | None = None,
    output: Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    expected_hash = expected_archive_sha256.strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("expected archive SHA-256 must contain 64 lowercase hex digits")
    reports: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    case_ids: set[str] = set()
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
        str(report.get("machine", {}).get("release", ""))
        for _path, report in reports
        if isinstance(report.get("machine"), dict)
    }
    device_classes = {
        str(report.get("case", {}).get("device_class", ""))
        for _path, report in reports
        if isinstance(report.get("case"), dict)
    }
    sample_rates = {
        int(report.get("case", {}).get("nominal_sample_rate_hz", 0) or 0)
        for _path, report in reports
        if isinstance(report.get("case"), dict)
    }
    scenarios = {
        str(report.get("case", {}).get("scenario", ""))
        for _path, report in reports
        if isinstance(report.get("case"), dict)
    }
    automated_baseline_cases = sum(
        1
        for _path, report in reports
        if report.get("case", {}).get("scenario") == "baseline"
        and report.get("case", {}).get("evidence_kind") == "automated"
    )
    missing = {
        "automated_baseline_cases": max(
            0, MINIMUM_AUTOMATED_BASELINE_CASES - automated_baseline_cases
        )
    }
    complete = not errors and missing["automated_baseline_cases"] == 0
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
                "supported_os_releases": sorted(SUPPORTED_OS_RELEASES),
                "supported_device_classes": sorted(SUPPORTED_DEVICE_CLASSES),
                "supported_nominal_sample_rates_hz": sorted(SUPPORTED_SAMPLE_RATES),
            },
            "missing": missing,
        },
        "cases": [
            {
                "id": str(report.get("case", {}).get("id", "")),
                "report_file": path.name,
                "report_sha256": _sha256(path),
                "os_release": str(report.get("machine", {}).get("release", "")),
                "device_class": str(
                    report.get("case", {}).get("device_class", "")
                ),
                "nominal_sample_rate_hz": int(
                    report.get("case", {}).get("nominal_sample_rate_hz", 0) or 0
                ),
                "scenario": str(report.get("case", {}).get("scenario", "")),
                "evidence_kind": str(
                    report.get("case", {}).get("evidence_kind", "")
                ),
            }
            for path, report in reports
        ],
        "errors": errors,
        "limitations": [
            "Coverage is release-artifact and hardware specific; source-tree simulations do not satisfy this gate.",
            "Promotion requires one digest-bound automated baseline; broader OS, device, rate, and lifecycle coverage is reported but is not inferred or required.",
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
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = aggregate(
        [path.resolve(strict=True) for path in args.reports],
        expected_archive_sha256=args.expected_archive_sha256,
        expected_source_revision=args.expected_source_revision,
        output=args.output.resolve(),
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps({"passed": result["passed"], "coverage": result["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
