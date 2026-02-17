"""
Latency calibration analysis utilities.

Uses cross-correlation between a known probe and recorded input to estimate
round-trip and one-way latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.signal import chirp, correlate, correlation_lags


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


def generate_probe_signal(
    sample_rate: int = 48_000,
    duration_ms: float = 80.0,
    start_freq_hz: float = 1_500.0,
    end_freq_hz: float = 9_000.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate deterministic chirp probe for latency measurement."""
    total_samples = max(1, int(sample_rate * (duration_ms / 1000.0)))
    t = np.arange(total_samples, dtype=np.float64) / float(sample_rate)

    sweep = chirp(
        t,
        f0=start_freq_hz,
        f1=end_freq_hz,
        t1=max(t[-1], 1e-6),
        method="linear",
    )

    # Window probe to reduce edge energy and false peaks.
    window = np.hanning(total_samples)
    probe = sweep * window

    peak = np.max(np.abs(probe))
    if peak > 0.0:
        probe = (probe / peak) * float(amplitude)

    return probe.astype(np.float32)


def analyze_latency(
    reference_probe: np.ndarray,
    recorded_signal: np.ndarray,
    sample_rate: int = 48_000,
    min_search_ms: float = 5.0,
    max_search_ms: float = 500.0,
) -> LatencyCalibrationResult:
    """Estimate latency by cross-correlating recorded signal against probe."""
    if reference_probe is None or recorded_signal is None:
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=0.0,
            estimated_one_way_ms=0.0,
            applied_compensation_ms=0.0,
            confidence=0.0,
            peak_sample_offset=0,
            message="Missing probe or recording.",
        )

    ref = np.asarray(reference_probe, dtype=np.float64).flatten()
    rec = np.asarray(recorded_signal, dtype=np.float64).flatten()

    if ref.size < 16 or rec.size < ref.size:
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=0.0,
            estimated_one_way_ms=0.0,
            applied_compensation_ms=0.0,
            confidence=0.0,
            peak_sample_offset=0,
            message="Recording too short for reliable correlation.",
        )

    # Remove DC to improve correlation reliability.
    ref = ref - np.mean(ref)
    rec = rec - np.mean(rec)

    corr = correlate(rec, ref, mode="full", method="fft")
    lags = correlation_lags(rec.size, ref.size, mode="full")

    min_lag = int((min_search_ms / 1000.0) * sample_rate)
    max_lag = int((max_search_ms / 1000.0) * sample_rate)

    valid_mask = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(valid_mask):
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=0.0,
            estimated_one_way_ms=0.0,
            applied_compensation_ms=0.0,
            confidence=0.0,
            peak_sample_offset=0,
            message="Search window is outside valid lag range.",
        )

    corr_window = corr[valid_mask]
    lag_window = lags[valid_mask]
    magnitudes = np.abs(corr_window)

    peak_index = int(np.argmax(magnitudes))
    peak_value = float(magnitudes[peak_index])
    peak_lag_samples = int(lag_window[peak_index])

    # Confidence from peak-vs-background ratio and sidelobe separation.
    background = float(np.median(magnitudes) + 1e-9)
    top_two = np.partition(magnitudes, -2)[-2:]
    second_peak = float(np.min(top_two)) if top_two.size >= 2 else 0.0
    peak_to_floor = peak_value / background
    peak_separation = peak_value / (second_peak + 1e-9)
    confidence = min(1.0, max(0.0, 0.25 * np.log10(peak_to_floor + 1.0) + 0.5 * (peak_separation - 1.0)))

    measured_round_trip_ms = (peak_lag_samples * 1000.0) / float(sample_rate)
    estimated_one_way_ms = measured_round_trip_ms / 2.0

    success = confidence >= 0.25 and measured_round_trip_ms > 0.0
    message = "ok" if success else "Low confidence correlation peak."

    return LatencyCalibrationResult(
        success=success,
        measured_round_trip_ms=measured_round_trip_ms,
        estimated_one_way_ms=estimated_one_way_ms,
        applied_compensation_ms=estimated_one_way_ms,
        confidence=confidence,
        peak_sample_offset=peak_lag_samples,
        message=message,
    )


def result_to_profile(result: LatencyCalibrationResult, sample_rate: int = 48_000) -> dict:
    """Convert analysis result into persisted profile dictionary."""
    return {
        "measured_round_trip_ms": float(result.measured_round_trip_ms),
        "estimated_one_way_ms": float(result.estimated_one_way_ms),
        "applied_compensation_ms": float(result.applied_compensation_ms),
        "confidence": float(result.confidence),
        "sample_rate": int(sample_rate),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
