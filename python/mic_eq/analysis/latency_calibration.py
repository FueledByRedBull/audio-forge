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
    expected_playback_start_ms: float | None = None,
    expected_playback_jitter_ms: float | None = None,
    expected_latency_min_ms: float | None = None,
    expected_latency_max_ms: float | None = None,
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
    expected_window_used = expected_playback_start_ms is not None
    expected_min = expected_latency_min_ms if expected_latency_min_ms is not None else min_search_ms
    expected_max = expected_latency_max_ms if expected_latency_max_ms is not None else max_search_ms
    playback_min_ms = 0.0
    playback_max_ms = 0.0
    if expected_window_used:
        playback_jitter_ms = max(0.0, expected_playback_jitter_ms or 0.0)
        playback_min_ms = max(0.0, expected_playback_start_ms - playback_jitter_ms)
        playback_max_ms = max(playback_min_ms, expected_playback_start_ms + playback_jitter_ms)
        min_lag = int(((playback_min_ms + expected_min) / 1000.0) * sample_rate)
        max_lag = int(((playback_max_ms + expected_max) / 1000.0) * sample_rate)

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

    valid_overlap = (lag_window >= 0) & ((lag_window + ref.size) <= rec.size)
    if not np.any(valid_overlap):
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=0.0,
            estimated_one_way_ms=0.0,
            applied_compensation_ms=0.0,
            confidence=0.0,
            peak_sample_offset=0,
            message="Search window does not overlap captured audio.",
        )

    corr_window = corr_window[valid_overlap]
    lag_window = lag_window[valid_overlap]
    magnitudes = magnitudes[valid_overlap]

    ref_energy = float(np.sum(ref * ref) + 1e-12)
    rec_energy_prefix = np.concatenate(([0.0], np.cumsum(rec * rec)))
    window_energy = rec_energy_prefix[lag_window + ref.size] - rec_energy_prefix[lag_window]
    normalized_scores = magnitudes / np.sqrt(np.maximum(window_energy, 1e-12) * ref_energy)

    peak_index = int(np.argmax(normalized_scores))
    peak_value = float(normalized_scores[peak_index])
    peak_lag_samples = int(lag_window[peak_index])

    measured_round_trip_ms = (peak_lag_samples * 1000.0) / float(sample_rate)
    if expected_window_used:
        measured_round_trip_ms = max(0.0, measured_round_trip_ms - expected_playback_start_ms)
    estimated_one_way_ms = measured_round_trip_ms / 2.0

    peak_segment_end = peak_lag_samples + ref.size
    if peak_lag_samples < 0 or peak_segment_end > rec.size:
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=measured_round_trip_ms,
            estimated_one_way_ms=estimated_one_way_ms,
            applied_compensation_ms=estimated_one_way_ms,
            confidence=0.0,
            peak_sample_offset=peak_lag_samples,
            message="Peak segment is outside the captured recording.",
        )

    exclusion_radius = max(1, ref.size // 4)
    off_peak_mask = np.ones_like(normalized_scores, dtype=bool)
    off_peak_mask[max(0, peak_index - exclusion_radius) : peak_index + exclusion_radius + 1] = False
    off_peak_scores = normalized_scores[off_peak_mask]
    background = float(np.median(off_peak_scores)) if off_peak_scores.size else 0.0
    second_peak = float(np.max(off_peak_scores)) if off_peak_scores.size else 0.0
    peak_percentile = (
        float(np.percentile(off_peak_scores, 90)) if off_peak_scores.size else background
    )
    robust_mad = (
        float(np.median(np.abs(off_peak_scores - background))) if off_peak_scores.size else 0.0
    )
    robust_sigma = max(1e-6, 1.4826 * robust_mad)
    peak_z = max(0.0, (peak_value - background) / robust_sigma)
    prominence_ratio = max(0.0, (peak_value - peak_percentile) / (peak_value + 1e-6))
    margin_ratio = max(0.0, 1.0 - (second_peak / (peak_value + 1e-6)))

    absolute_peak_score = min(1.0, max(0.0, (peak_value - 0.08) / 0.22))
    peak_score = min(1.0, max(0.0, (peak_z - 3.0) / 5.0))
    prominence_score = min(1.0, max(0.0, prominence_ratio / 0.5))
    margin_score = min(1.0, max(0.0, margin_ratio / 0.3))
    alignment_score = 0.0

    if expected_window_used:
        expected_center_ms = 0.5 * (
            playback_min_ms + playback_max_ms + expected_min + expected_max
        )
        expected_center_samples = int((expected_center_ms / 1000.0) * sample_rate)
        half_width_samples = max(1, (max_lag - min_lag) // 2)
        alignment_score = max(
            0.0,
            1.0
            - (abs(peak_lag_samples - expected_center_samples) / float(half_width_samples)),
        )
        confidence = (
            0.30 * absolute_peak_score
            + 0.30 * peak_score
            + 0.20 * prominence_score
            + 0.10 * margin_score
            + 0.10 * alignment_score
        )
    else:
        confidence = (
            0.35 * absolute_peak_score
            + 0.35 * peak_score
            + 0.20 * prominence_score
            + 0.10 * margin_score
        )

    success = (
        confidence >= 0.25
        and measured_round_trip_ms > 0.0
        and peak_value >= 0.07
        and margin_ratio >= 0.01
    )
    message = "ok" if success else "Low confidence or ambiguous correlation peak."

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
