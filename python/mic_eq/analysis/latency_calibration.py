"""
Latency calibration analysis utilities.

Uses repeated coded probes and robust correlation between a known probe and
recorded input to estimate round-trip and one-way latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.signal import correlate, correlation_lags


BARKER_13 = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    dtype=np.float64,
)
DEFAULT_REPETITIONS = 4


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


def generate_probe_signal(
    sample_rate: int = 48_000,
    duration_ms: float = 80.0,
    start_freq_hz: float = 1_500.0,
    end_freq_hz: float = 9_000.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a deterministic repeated Barker-coded probe for latency measurement.

    ``start_freq_hz`` and ``end_freq_hz`` are retained for API compatibility;
    the current probe uses coded wideband bursts rather than a chirp sweep.
    """
    del start_freq_hz, end_freq_hz

    total_samples = max(1, int(sample_rate * (duration_ms / 1000.0)))
    burst, offsets = _coded_probe_components(sample_rate, total_samples)
    probe = np.zeros(total_samples, dtype=np.float64)
    for offset in offsets:
        end = min(total_samples, offset + burst.size)
        if end > offset:
            probe[offset:end] += burst[: end - offset]

    peak = np.max(np.abs(probe))
    if peak > 0.0:
        probe = (probe / peak) * float(amplitude)

    return probe.astype(np.float32)


def _coded_probe_components(
    sample_rate: int,
    total_samples: int,
    repetitions: int = DEFAULT_REPETITIONS,
) -> tuple[np.ndarray, list[int]]:
    """Return a single coded burst and start offsets inside the full probe."""
    repetitions = max(1, int(repetitions))
    chip_samples = max(4, int(round(sample_rate * 0.0005)))
    min_spacing = max(chip_samples, int(round(sample_rate * 0.006)))

    while chip_samples > 4:
        burst_len = BARKER_13.size * chip_samples
        required = repetitions * burst_len + (repetitions - 1) * min_spacing
        if required <= total_samples:
            break
        chip_samples -= 1

    code = np.repeat(BARKER_13, chip_samples)
    window = np.hanning(code.size)
    burst = code * window
    burst -= float(np.mean(burst))
    peak = float(np.max(np.abs(burst)))
    if peak > 0.0:
        burst /= peak

    burst_len = burst.size
    if repetitions == 1 or total_samples <= burst_len:
        return burst[:total_samples], [0]

    available_gap = max(0, total_samples - repetitions * burst_len)
    spacing = max(min_spacing, available_gap // (repetitions - 1))
    offsets: list[int] = []
    cursor = 0
    for _ in range(repetitions):
        if cursor + burst_len > total_samples:
            break
        offsets.append(cursor)
        cursor += burst_len + spacing

    if not offsets:
        offsets = [0]
    return burst, offsets


def _normalize_audio(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).flatten()
    if signal.size == 0:
        return signal
    return signal - float(np.mean(signal))


def _parabolic_peak_offset(scores: np.ndarray, index: int) -> float:
    if index <= 0 or index >= scores.size - 1:
        return 0.0
    left = float(scores[index - 1])
    center = float(scores[index])
    right = float(scores[index + 1])
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def _normalized_correlation_scores(
    rec: np.ndarray,
    ref: np.ndarray,
    *,
    min_lag: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr = correlate(rec, ref, mode="full", method="fft")
    lags = correlation_lags(rec.size, ref.size, mode="full")
    valid_mask = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(valid_mask):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    corr_window = corr[valid_mask]
    lag_window = lags[valid_mask]
    valid_overlap = (lag_window >= 0) & ((lag_window + ref.size) <= rec.size)
    if not np.any(valid_overlap):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    corr_window = corr_window[valid_overlap]
    lag_window = lag_window[valid_overlap]
    magnitudes = np.abs(corr_window)

    ref_energy = float(np.sum(ref * ref) + 1e-12)
    rec_energy_prefix = np.concatenate(([0.0], np.cumsum(rec * rec)))
    window_energy = rec_energy_prefix[lag_window + ref.size] - rec_energy_prefix[lag_window]
    normalized_scores = magnitudes / np.sqrt(np.maximum(window_energy, 1e-12) * ref_energy)
    return lag_window, normalized_scores


def _phat_lag_hint(
    rec: np.ndarray,
    ref: np.ndarray,
    *,
    min_lag: int,
    max_lag: int,
) -> int | None:
    """Return a GCC-PHAT lag hint inside the search range."""
    if rec.size < ref.size or ref.size < 2:
        return None

    n = 1
    while n < rec.size + ref.size:
        n <<= 1
    rec_fft = np.fft.rfft(rec, n=n)
    ref_fft = np.fft.rfft(ref, n=n)
    cross = rec_fft * np.conj(ref_fft)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft(cross, n=n)
    lags = np.arange(corr.size)
    wrapped = lags.copy()
    wrapped[wrapped > n // 2] -= n
    valid = (wrapped >= min_lag) & (wrapped <= max_lag)
    if not np.any(valid):
        return None
    valid_scores = np.abs(corr[valid])
    valid_lags = wrapped[valid]
    return int(valid_lags[int(np.argmax(valid_scores))])


def _pick_peak(
    lags: np.ndarray,
    scores: np.ndarray,
    *,
    direct_path_bias: float = 0.82,
) -> tuple[float, float, float, float, float]:
    """Pick earliest strong peak and return lag, score, margin, ambiguity, background."""
    if lags.size == 0 or scores.size == 0:
        return 0.0, 0.0, 0.0, 1.0, 0.0

    max_index = int(np.argmax(scores))
    max_score = float(scores[max_index])
    threshold = max_score * float(direct_path_bias)
    strong = np.flatnonzero(scores >= threshold)
    peak_index = int(strong[0]) if strong.size else max_index
    peak_score = float(scores[peak_index])
    peak_lag = float(lags[peak_index]) + _parabolic_peak_offset(scores, peak_index)

    exclusion_radius = max(1, min(128, scores.size // 50))
    off_peak_mask = np.ones_like(scores, dtype=bool)
    off_peak_mask[max(0, peak_index - exclusion_radius) : peak_index + exclusion_radius + 1] = False
    off_peak_scores = scores[off_peak_mask]
    background = float(np.median(off_peak_scores)) if off_peak_scores.size else 0.0
    second_peak = float(np.max(off_peak_scores)) if off_peak_scores.size else 0.0
    margin_ratio = max(0.0, 1.0 - (second_peak / (peak_score + 1e-6)))
    ambiguity = float(np.clip(second_peak / (peak_score + 1e-6), 0.0, 1.0))
    return peak_lag, peak_score, margin_ratio, ambiguity, background


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
    """Estimate latency from repeated coded probes in captured audio."""
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

    ref = _normalize_audio(reference_probe)
    rec = _normalize_audio(recorded_signal)

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

    if max_lag <= min_lag:
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=0.0,
            estimated_one_way_ms=0.0,
            applied_compensation_ms=0.0,
            confidence=0.0,
            peak_sample_offset=0,
            message="Search window is outside valid lag range.",
        )

    burst, offsets = _coded_probe_components(sample_rate, ref.size)
    if burst.size < 16 or not offsets:
        burst = ref
        offsets = [0]

    full_lags, full_scores = _normalized_correlation_scores(
        rec,
        ref,
        min_lag=min_lag,
        max_lag=max_lag,
    )
    if full_lags.size == 0:
        return LatencyCalibrationResult(
            success=False,
            measured_round_trip_ms=0.0,
            estimated_one_way_ms=0.0,
            applied_compensation_ms=0.0,
            confidence=0.0,
            peak_sample_offset=0,
            message="Search window does not overlap captured audio.",
        )
    coarse_start, full_peak, full_margin, full_ambiguity, _background = _pick_peak(
        full_lags,
        full_scores,
        direct_path_bias=0.985,
    )
    local_radius = max(int(round(sample_rate * 0.010)), burst.size)

    estimates: list[float] = []
    peak_values: list[float] = [full_peak]
    margins: list[float] = [full_margin]
    ambiguities: list[float] = [full_ambiguity]
    phat_matches: list[float] = []

    for offset in offsets:
        expected_lag = coarse_start + float(offset)
        lag_min = max(min_lag + int(offset), int(round(expected_lag - local_radius)))
        lag_max = min(max_lag + int(offset), int(round(expected_lag + local_radius)))
        lag_window, scores = _normalized_correlation_scores(
            rec,
            burst,
            min_lag=lag_min,
            max_lag=lag_max,
        )
        if lag_window.size == 0:
            continue

        peak_lag, peak_value, margin_ratio, ambiguity, _background = _pick_peak(
            lag_window,
            scores,
            direct_path_bias=0.94,
        )
        if peak_value < 0.035:
            continue

        start_estimate = peak_lag - float(offset)
        estimates.append(start_estimate)
        peak_values.append(peak_value)
        margins.append(margin_ratio)
        ambiguities.append(ambiguity)

        phat_hint = _phat_lag_hint(rec, burst, min_lag=lag_min, max_lag=lag_max)
        if phat_hint is not None:
            phat_start = float(phat_hint - int(offset))
            phat_matches.append(
                max(0.0, 1.0 - abs(phat_start - start_estimate) / max(1.0, sample_rate * 0.006))
            )

    if not estimates:
        lag_window, scores = _normalized_correlation_scores(
            rec,
            ref,
            min_lag=min_lag,
            max_lag=max_lag,
        )
        if lag_window.size == 0:
            return LatencyCalibrationResult(
                success=False,
                measured_round_trip_ms=0.0,
                estimated_one_way_ms=0.0,
                applied_compensation_ms=0.0,
                confidence=0.0,
                peak_sample_offset=0,
                message="Search window does not overlap captured audio.",
            )
        peak_lag, peak_value, margin_ratio, ambiguity, _background = _pick_peak(
            lag_window,
            scores,
            direct_path_bias=0.985,
        )
        estimates = [peak_lag]
        peak_values = [peak_value]
        margins = [margin_ratio]
        ambiguities = [ambiguity]

    estimates_arr = np.asarray(estimates, dtype=np.float64)
    median_start_samples = float(np.median(estimates_arr))
    deviations = np.abs(estimates_arr - median_start_samples)
    agreement_samples = float(np.percentile(deviations, 75)) if deviations.size else 0.0
    agreement_ms = (agreement_samples * 1000.0) / float(sample_rate)

    measured_round_trip_ms = (median_start_samples * 1000.0) / float(sample_rate)
    if expected_window_used:
        measured_round_trip_ms = max(0.0, measured_round_trip_ms - expected_playback_start_ms)
    estimated_one_way_ms = measured_round_trip_ms / 2.0

    peak_value = float(np.median(peak_values)) if peak_values else 0.0
    margin_ratio = float(np.median(margins)) if margins else 0.0
    ambiguity_score = float(np.median(ambiguities)) if ambiguities else 1.0
    phat_score = float(np.median(phat_matches)) if phat_matches else 0.5

    peak_strength_score = float(np.clip((peak_value - 0.06) / 0.24, 0.0, 1.0))
    agreement_score = float(np.clip(1.0 - agreement_ms / 4.0, 0.0, 1.0))
    repetition_score = float(np.clip(len(estimates) / min(3, max(1, len(offsets))), 0.0, 1.0))
    margin_score = float(np.clip(margin_ratio / 0.28, 0.0, 1.0))
    ambiguity_confidence = float(np.clip(1.0 - ambiguity_score, 0.0, 1.0))

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
            - (abs(median_start_samples - expected_center_samples) / float(half_width_samples)),
        )

    confidence = (
        0.24 * peak_strength_score
        + 0.24 * agreement_score
        + 0.18 * repetition_score
        + 0.14 * margin_score
        + 0.12 * ambiguity_confidence
        + 0.08 * phat_score
    )
    if expected_window_used:
        confidence = 0.88 * confidence + 0.12 * alignment_score

    success = (
        confidence >= 0.32
        and measured_round_trip_ms > 0.0
        and peak_value >= 0.07
        and ambiguity_score < 0.90
        and len(estimates) >= min(2, len(offsets))
        and agreement_ms <= 6.0
    )
    if success:
        message = "ok"
    elif agreement_ms > 6.0 and len(estimates) > 1:
        message = "Repeated probes disagree; echoes or bleed make latency ambiguous."
    elif ambiguity_score > 0.82:
        message = "Echo ambiguity: competing correlation peaks are too close."
    else:
        message = "Low confidence or ambiguous coded-probe correlation."

    return LatencyCalibrationResult(
        success=success,
        measured_round_trip_ms=measured_round_trip_ms,
        estimated_one_way_ms=estimated_one_way_ms,
        applied_compensation_ms=estimated_one_way_ms,
        confidence=confidence,
        peak_sample_offset=int(round(median_start_samples)),
        message=message,
        repetition_count=len(estimates),
        agreement_ms=agreement_ms,
        ambiguity_score=ambiguity_score,
        sub_sample_offset=median_start_samples,
    )


def result_to_profile(result: LatencyCalibrationResult, sample_rate: int = 48_000) -> dict:
    """Convert analysis result into persisted profile dictionary."""
    return {
        "measured_round_trip_ms": float(result.measured_round_trip_ms),
        "estimated_one_way_ms": float(result.estimated_one_way_ms),
        "applied_compensation_ms": float(result.applied_compensation_ms),
        "confidence": float(result.confidence),
        "agreement_ms": float(result.agreement_ms),
        "ambiguity_score": float(result.ambiguity_score),
        "repetition_count": int(result.repetition_count),
        "sample_rate": int(sample_rate),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
