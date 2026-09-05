"""Shared schema checks for exact-artifact hardware qualification reports."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
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
        "model_configuration_change",
    }
)
REQUIRED_OS_RELEASES = frozenset({"10", "11"})
REQUIRED_DEVICE_CLASSES = frozenset({"built_in", "usb", "virtual"})
REQUIRED_SAMPLE_RATES = frozenset({44_100, 48_000})
REQUIRED_SCENARIOS = frozenset(
    {
        "baseline",
        "device_reconnect",
        "default_device_change",
        "sleep_resume",
        "model_configuration_change",
    }
)
MINIMUM_AUTOMATED_BASELINE_CASES = 1
PSEUDONYM = re.compile(r"^device-[0-9a-f]{16}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def validate_case(
    report: dict[str, Any],
    *,
    expected_archive_sha256: str | None = None,
    expected_source_revision: str | None = None,
) -> list[str]:
    """Return all schema and evidence errors for one hardware report."""
    errors: list[str] = []
    if report.get("schema_version") != 3:
        errors.append("hardware case schema must be 3")
    if report.get("qualification_kind") != "exact-artifact-hardware":
        errors.append("wrong qualification kind")
    if report.get("passed") is not True or report.get("status") != "passed":
        errors.append("case did not pass")

    source_revision = report.get("source_revision")
    if not isinstance(source_revision, str) or COMMIT.fullmatch(source_revision.casefold()) is None:
        errors.append("source revision is missing or malformed")
    elif (
        expected_source_revision is not None
        and source_revision.casefold() != expected_source_revision.casefold()
    ):
        errors.append("source revision differs from the release tag")

    artifact = report.get("artifact")
    archive_hash = artifact.get("archive_sha256") if isinstance(artifact, dict) else None
    if not isinstance(archive_hash, str) or SHA256.fullmatch(archive_hash) is None:
        errors.append("exact artifact provenance is missing or malformed")
    elif (
        expected_archive_sha256 is not None
        and archive_hash.casefold() != expected_archive_sha256.casefold()
    ):
        errors.append("archive SHA-256 differs from the matrix candidate")

    case = report.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
        errors.append("case metadata is missing")
    else:
        device_class = case.get("device_class")
        sample_rate = case.get("nominal_sample_rate_hz")
        scenario = case.get("scenario")
        evidence_kind = case.get("evidence_kind")
        if not isinstance(device_class, str) or device_class not in SUPPORTED_DEVICE_CLASSES:
            errors.append("unsupported device class")
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate not in SUPPORTED_SAMPLE_RATES
        ):
            errors.append("unsupported nominal sample rate")
        if not isinstance(scenario, str) or scenario not in SUPPORTED_SCENARIOS:
            errors.append("unsupported lifecycle scenario")
        if not isinstance(evidence_kind, str) or evidence_kind not in {
            "automated",
            "operator_observed",
        }:
            errors.append("unsupported evidence kind")
        if scenario == "baseline" and evidence_kind != "automated":
            errors.append("baseline case must use automated evidence")
        if scenario != "baseline" and evidence_kind != "operator_observed":
            errors.append("lifecycle scenario lacks operator evidence")
        if scenario != "baseline" and case.get("operator_attestation") is not True:
            errors.append("lifecycle scenario lacks operator attestation")
        if case.get("scenario_evidence_valid") is not True:
            errors.append("scenario evidence was not validated")

    machine = report.get("machine")
    if (
        not isinstance(machine, dict)
        or not isinstance(machine.get("release"), str)
        or machine.get("release") not in SUPPORTED_OS_RELEASES
    ):
        errors.append("unsupported or missing Windows release")

    duration = report.get("requested_health_duration_seconds")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or duration < 1_800.0
    ):
        errors.append("sustained health duration is below 1800 seconds")

    for field in (
        "package_smoke",
        "executable_startup",
        "model_discovery",
        "selected_route_correlation",
        "sustained_health",
    ):
        check = report.get(field)
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"{field} did not pass")

    routes = report.get("routes")
    required_routes = ("correlation", "sustained_health")
    if not isinstance(routes, dict) or not routes:
        errors.append("route pseudonyms are missing")
    else:
        for route_name in required_routes:
            route = routes.get(route_name)
            if not isinstance(route, dict):
                errors.append(f"invalid {route_name} route record")
                continue
            for endpoint in ("input", "output"):
                value = route.get(endpoint)
                if not isinstance(value, str) or PSEUDONYM.fullmatch(value) is None:
                    errors.append("raw or invalid device identity in report")
    return errors


def coverage_missing(
    reports: Iterable[dict[str, Any]],
) -> dict[str, int | list[str] | list[int]]:
    """Compute the mandatory v2 matrix gaps from validated case-shaped reports."""
    reports = list(reports)
    os_releases: set[str] = set()
    device_classes: set[str] = set()
    sample_rates: set[int] = set()
    scenarios: set[str] = set()
    baseline_count = 0
    for report in reports:
        case = report.get("case")
        machine = report.get("machine")
        if not isinstance(case, dict):
            continue
        release = machine.get("release") if isinstance(machine, dict) else None
        if isinstance(release, str):
            os_releases.add(release)
        device_class = case.get("device_class")
        if isinstance(device_class, str):
            device_classes.add(device_class)
        sample_rate = case.get("nominal_sample_rate_hz")
        if isinstance(sample_rate, int) and not isinstance(sample_rate, bool):
            sample_rates.add(sample_rate)
        scenario = case.get("scenario")
        if isinstance(scenario, str):
            scenarios.add(scenario)
        if scenario == "baseline" and case.get("evidence_kind") == "automated":
            baseline_count += 1
    return {
        "automated_baseline_cases": max(
            0, MINIMUM_AUTOMATED_BASELINE_CASES - baseline_count
        ),
        "os_releases": sorted(REQUIRED_OS_RELEASES - os_releases),
        "device_classes": sorted(REQUIRED_DEVICE_CLASSES - device_classes),
        "nominal_sample_rates_hz": sorted(REQUIRED_SAMPLE_RATES - sample_rates),
        "scenarios": sorted(REQUIRED_SCENARIOS - scenarios),
    }
