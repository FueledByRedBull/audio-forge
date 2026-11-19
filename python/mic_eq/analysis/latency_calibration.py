"""
Latency calibration analysis utilities.

Uses repeated coded probes and robust correlation between a known probe and
recorded input to estimate the selected output-to-input route latency. The
probe and the analysis run in the native core.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ..mic_eq_core import analyze_latency as _native_analyze_latency
from ..mic_eq_core import generate_probe_signal


@dataclass
class LatencyCalibrationResult:
    """Result of latency calibration analysis."""

    success: bool
    measured_round_trip_ms: float
    estimated_one_way_ms: float
    applied_compensation_ms: float
    confidence: float
    peak_sample_offset: int
    message: str = ""
    repetition_count: int = 0
    agreement_ms: float = 0.0
    ambiguity_score: float = 0.0
    sub_sample_offset: float = 0.0
    route_latency_ms: float = 0.0
    directional_latency_ms: float | None = None
    route_kind: str = "output_to_input"
    compensation_basis: str = "measured_output_to_input_route"


def analyze_latency(
    reference_probe: np.ndarray | None,
    recorded_signal: np.ndarray | None,
    sample_rate: int = 48_000,
    min_search_ms: float = 5.0,
    max_search_ms: float = 500.0,
    expected_playback_start_ms: float | None = None,
    expected_playback_jitter_ms: float | None = None,
    expected_latency_min_ms: float | None = None,
    expected_latency_max_ms: float | None = None,
    route_kind: str = "output_to_input",
) -> LatencyCalibrationResult:
    """Estimate the selected output-to-input route latency.

    The calibration signal is played on the selected output and measured on
    the selected input. That is one concrete route, not a symmetric round trip
    by definition. The measured route delay is therefore applied directly;
    it is deliberately not divided by two. A directional one-way estimate
    requires an independent timing reference and is reported as unavailable.
    """
    return LatencyCalibrationResult(
        **_native_analyze_latency(
            reference_probe,
            recorded_signal,
            sample_rate,
            min_search_ms,
            max_search_ms,
            expected_playback_start_ms,
            expected_playback_jitter_ms,
            expected_latency_min_ms,
            expected_latency_max_ms,
            route_kind,
        )
    )


def result_to_profile(
    result: LatencyCalibrationResult,
    sample_rate: int = 48_000,
    *,
    engine_latency_ms: float = 0.0,
    engine_config_signature: str = "",
) -> dict:
    """Convert analysis result into persisted profile dictionary."""
    route_latency_ms = float(result.route_latency_ms)
    if route_latency_ms <= 0.0:
        route_latency_ms = max(
            0.0,
            float(result.measured_round_trip_ms),
            float(result.applied_compensation_ms),
        )
    engine_latency_ms = max(0.0, float(engine_latency_ms))
    return {
        "measured_round_trip_ms": float(result.measured_round_trip_ms),
        "estimated_one_way_ms": float(result.estimated_one_way_ms),
        "applied_compensation_ms": float(result.applied_compensation_ms),
        "route_latency_ms": route_latency_ms,
        "directional_latency_ms": (
            float(result.directional_latency_ms)
            if result.directional_latency_ms is not None
            else None
        ),
        "route_kind": str(result.route_kind),
        "compensation_basis": str(result.compensation_basis),
        "confidence": float(result.confidence),
        "agreement_ms": float(result.agreement_ms),
        "ambiguity_score": float(result.ambiguity_score),
        "repetition_count": int(result.repetition_count),
        "sample_rate": int(sample_rate),
        "engine_latency_ms": engine_latency_ms,
        "total_latency_ms": route_latency_ms + engine_latency_ms,
        "engine_config_signature": str(engine_config_signature),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "LatencyCalibrationResult",
    "analyze_latency",
    "generate_probe_signal",
    "result_to_profile",
]
