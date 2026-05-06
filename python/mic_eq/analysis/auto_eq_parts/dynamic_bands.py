"""Dynamic band selection and reliability helpers for Auto-EQ."""

import numpy as np

from mic_eq.config import EQ_FREQUENCIES

from .constants import (
    DENSE_GRID_POINTS,
    DYNAMIC_CENTER_REFINE_PCT,
    DYNAMIC_HIGH_SHELF_RANGE_HZ,
    DYNAMIC_LOW_SHELF_RANGE_HZ,
    DYNAMIC_MEANINGFUL_CORRECTION_DB,
    DYNAMIC_PEAK_MIN_SEPARATION_OCT,
    DYNAMIC_PEAK_RANGE_HZ,
    DYNAMIC_SHELF_CENTER_REFINE_PCT,
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    LOW_BAND_Q_MAX,
    LOW_BAND_Q_MAX_HZ,
    NUM_EQ_BANDS,
    OUT_OF_BAND_WEIGHT,
    Q_MAX,
    Q_MIN,
    Q_PRIOR,
    SNR_FULL_DB,
    SNR_LOW_RELIABILITY_MAX_BOOST_DB,
    SNR_LOW_RELIABILITY_WEIGHT,
    SNR_MIN_DB,
    TILT_FIT_MAX_HZ,
    TILT_FIT_MIN_HZ,
    TILT_MIN_FIT_R2,
    VOICE_WEIGHT,
)

def _build_dense_log_grid(freqs: np.ndarray) -> np.ndarray:
    freq_min = max(20.0, float(np.min(freqs)))
    freq_max = min(20_000.0, float(np.max(freqs)))
    if freq_max <= freq_min:
        freq_max = max(freq_min * 1.001, freq_min + 1.0)
    return np.logspace(np.log10(freq_min), np.log10(freq_max), DENSE_GRID_POINTS)


def _voice_weights(freqs: np.ndarray) -> np.ndarray:
    weights = np.full_like(freqs, OUT_OF_BAND_WEIGHT, dtype=float)
    voice_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    weights[voice_mask] = VOICE_WEIGHT
    return weights


def _q_bounds(center_freqs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    q_low = np.full(NUM_EQ_BANDS, Q_MIN, dtype=float)
    q_high = np.full(NUM_EQ_BANDS, Q_MAX, dtype=float)
    for i, fc in enumerate(center_freqs):
        if fc < LOW_BAND_Q_MAX_HZ:
            q_high[i] = LOW_BAND_Q_MAX
    return q_low, q_high


def _role_center_limits() -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(NUM_EQ_BANDS, DYNAMIC_PEAK_RANGE_HZ[0], dtype=float)
    upper = np.full(NUM_EQ_BANDS, DYNAMIC_PEAK_RANGE_HZ[1], dtype=float)
    lower[0], upper[0] = DYNAMIC_LOW_SHELF_RANGE_HZ
    lower[-1], upper[-1] = DYNAMIC_HIGH_SHELF_RANGE_HZ
    return lower, upper


def _center_bounds(base_centers_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    role_low, role_high = _role_center_limits()
    refine_pct = np.full(NUM_EQ_BANDS, DYNAMIC_CENTER_REFINE_PCT, dtype=float)
    refine_pct[0] = DYNAMIC_SHELF_CENTER_REFINE_PCT
    refine_pct[-1] = DYNAMIC_SHELF_CENTER_REFINE_PCT

    center_low = np.maximum(role_low, base_centers_hz * (1.0 - refine_pct))
    center_high = np.minimum(role_high, base_centers_hz * (1.0 + refine_pct))

    # Keep optimizer bounds ordered by using geometric midpoints between
    # selected centers. This prevents the peaking bands from swapping roles.
    for i in range(NUM_EQ_BANDS - 1):
        midpoint = float(np.sqrt(base_centers_hz[i] * base_centers_hz[i + 1]))
        center_high[i] = min(center_high[i], midpoint * 0.999)
        center_low[i + 1] = max(center_low[i + 1], midpoint * 1.001)

    for i, center in enumerate(base_centers_hz):
        if center_low[i] >= center_high[i]:
            center_low[i] = max(role_low[i], center * 0.995)
            center_high[i] = min(role_high[i], center * 1.005)

    return center_low, center_high


def _mask_range(freqs: np.ndarray, lower_hz: float, upper_hz: float) -> np.ndarray:
    return (freqs >= lower_hz) & (freqs <= upper_hz)


def _max_score_frequency(
    freqs: np.ndarray,
    scores: np.ndarray,
    lower_hz: float,
    upper_hz: float,
    fallback_hz: float,
) -> float:
    mask = _mask_range(freqs, lower_hz, upper_hz)
    if not np.any(mask):
        return fallback_hz

    local_freqs = freqs[mask]
    local_scores = scores[mask]
    idx = int(np.argmax(local_scores))
    return float(local_freqs[idx])


def _is_separated_octaves(freq_hz: float, selected: list[float], min_octaves: float) -> bool:
    return all(abs(np.log2(freq_hz / existing)) >= min_octaves for existing in selected)


def _fallback_peak_centers(
    freqs: np.ndarray,
    scores: np.ndarray,
    selected: list[float],
) -> list[float]:
    lower_hz, upper_hz = DYNAMIC_PEAK_RANGE_HZ
    edges = np.geomspace(lower_hz, upper_hz, NUM_EQ_BANDS)
    fillers: list[float] = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        center = _max_score_frequency(freqs, scores, float(lo), float(hi), float(np.sqrt(lo * hi)))
        if _is_separated_octaves(
            center,
            selected + fillers,
            DYNAMIC_PEAK_MIN_SEPARATION_OCT * 0.75,
        ):
            fillers.append(center)
        if len(selected) + len(fillers) >= NUM_EQ_BANDS - 2:
            break

    if len(selected) + len(fillers) < NUM_EQ_BANDS - 2:
        for center in np.geomspace(lower_hz, upper_hz, NUM_EQ_BANDS - 2):
            center = float(center)
            if _is_separated_octaves(
                center,
                selected + fillers,
                DYNAMIC_PEAK_MIN_SEPARATION_OCT * 0.5,
            ):
                fillers.append(center)
            if len(selected) + len(fillers) >= NUM_EQ_BANDS - 2:
                break

    return fillers


def _estimate_q_from_residual(
    dense_freqs: np.ndarray,
    residual_db: np.ndarray,
    center_hz: float,
    q_min: float,
    q_max: float,
    fallback_q: float,
) -> float:
    idx = int(np.argmin(np.abs(dense_freqs - center_hz)))
    peak = float(residual_db[idx])
    peak_abs = abs(peak)
    if peak_abs < DYNAMIC_MEANINGFUL_CORRECTION_DB:
        return float(np.clip(fallback_q, q_min, q_max))

    sign = 1.0 if peak >= 0.0 else -1.0
    threshold = max(peak_abs * 0.5, DYNAMIC_MEANINGFUL_CORRECTION_DB)
    left = idx
    right = idx

    while left > 0:
        value = float(residual_db[left - 1])
        if value * sign <= 0.0 or abs(value) < threshold:
            break
        left -= 1

    while right < dense_freqs.size - 1:
        value = float(residual_db[right + 1])
        if value * sign <= 0.0 or abs(value) < threshold:
            break
        right += 1

    f_left = float(dense_freqs[left])
    f_right = float(dense_freqs[right])
    bandwidth_hz = max(f_right - f_left, center_hz * 0.04)
    estimated_q = center_hz / bandwidth_hz
    return float(np.clip(estimated_q, q_min, q_max))


def _select_dynamic_band_layout(
    dense_freqs: np.ndarray,
    residual_db: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(residual_db) * weights
    low_center = _max_score_frequency(
        dense_freqs,
        scores,
        DYNAMIC_LOW_SHELF_RANGE_HZ[0],
        DYNAMIC_LOW_SHELF_RANGE_HZ[1],
        EQ_FREQUENCIES[0],
    )
    high_center = _max_score_frequency(
        dense_freqs,
        scores,
        DYNAMIC_HIGH_SHELF_RANGE_HZ[0],
        DYNAMIC_HIGH_SHELF_RANGE_HZ[1],
        EQ_FREQUENCIES[-1],
    )

    peak_mask = _mask_range(dense_freqs, DYNAMIC_PEAK_RANGE_HZ[0], DYNAMIC_PEAK_RANGE_HZ[1])
    peak_indices = np.flatnonzero(peak_mask)
    candidate_indices: list[int] = []
    for idx in peak_indices:
        if idx == 0 or idx >= scores.size - 1:
            continue
        if scores[idx] >= scores[idx - 1] and scores[idx] >= scores[idx + 1]:
            candidate_indices.append(int(idx))

    if not candidate_indices:
        candidate_indices = [int(idx) for idx in peak_indices]

    candidate_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)
    selected_peaks: list[float] = []
    for idx in candidate_indices:
        center = float(dense_freqs[idx])
        if (
            abs(float(residual_db[idx])) >= DYNAMIC_MEANINGFUL_CORRECTION_DB
            and _is_separated_octaves(
                center,
                selected_peaks,
                DYNAMIC_PEAK_MIN_SEPARATION_OCT,
            )
        ):
            selected_peaks.append(center)
        if len(selected_peaks) >= NUM_EQ_BANDS - 2:
            break

    if len(selected_peaks) < NUM_EQ_BANDS - 2:
        selected_peaks.extend(_fallback_peak_centers(dense_freqs, scores, selected_peaks))
    if len(selected_peaks) < NUM_EQ_BANDS - 2:
        for center in np.geomspace(
            DYNAMIC_PEAK_RANGE_HZ[0],
            DYNAMIC_PEAK_RANGE_HZ[1],
            NUM_EQ_BANDS - 2,
        ):
            center = float(center)
            if not any(abs(np.log2(center / existing)) < 0.02 for existing in selected_peaks):
                selected_peaks.append(center)
            if len(selected_peaks) >= NUM_EQ_BANDS - 2:
                break

    selected_peaks = sorted(selected_peaks[:NUM_EQ_BANDS - 2])
    centers = np.asarray([low_center, *selected_peaks, high_center], dtype=float)

    role_low, role_high = _role_center_limits()
    centers = np.clip(centers, role_low, role_high)
    centers[1:-1] = np.sort(centers[1:-1])

    q_low, q_high = _q_bounds(centers.tolist())
    fallback_q = np.clip(np.full(NUM_EQ_BANDS, Q_PRIOR, dtype=float), q_low, q_high)
    q_prior = np.asarray(
        [
            _estimate_q_from_residual(
                dense_freqs,
                residual_db,
                float(center),
                float(q_low[i]),
                float(q_high[i]),
                float(fallback_q[i]),
            )
            for i, center in enumerate(centers)
        ],
        dtype=float,
    )
    return centers, q_prior


def _enforce_adjacent_gain_limit(gains: np.ndarray, max_diff_db: float) -> np.ndarray:
    bounded = np.asarray(gains, dtype=float).copy()
    bounded = np.clip(bounded, GAIN_MIN_DB, GAIN_MAX_DB)

    for i in range(1, bounded.size):
        lo = bounded[i - 1] - max_diff_db
        hi = bounded[i - 1] + max_diff_db
        bounded[i] = np.clip(bounded[i], lo, hi)

    for i in range(bounded.size - 2, -1, -1):
        lo = bounded[i + 1] - max_diff_db
        hi = bounded[i + 1] + max_diff_db
        bounded[i] = np.clip(bounded[i], lo, hi)

    return np.clip(bounded, GAIN_MIN_DB, GAIN_MAX_DB)


def _remove_spectral_tilt(freqs: np.ndarray, measured_db: np.ndarray) -> tuple[np.ndarray, float]:
    mask = (freqs >= TILT_FIT_MIN_HZ) & (freqs <= TILT_FIT_MAX_HZ)
    if np.sum(mask) < 2:
        return measured_db, 0.0

    x = np.log10(freqs[mask])
    y = measured_db[mask]
    x_center = x - np.mean(x)
    denom = float(np.sum(x_center ** 2))
    if denom <= 0.0:
        return measured_db, 0.0

    slope = float(np.dot(x_center, y) / denom)
    y_center = y - np.mean(y)
    ss_tot = float(np.sum(y_center ** 2))
    if ss_tot <= 1e-12:
        return measured_db, 0.0

    fit_y = slope * x_center
    ss_res = float(np.sum((y - fit_y) ** 2))
    fit_r2 = 1.0 - (ss_res / ss_tot)
    if not np.isfinite(fit_r2) or fit_r2 < TILT_MIN_FIT_R2:
        return measured_db, 0.0

    all_x_center = np.log10(freqs) - np.mean(x)
    tilt_component = slope * all_x_center
    return measured_db - tilt_component, slope


def _snr_reliability(snr_db: np.ndarray) -> np.ndarray:
    reliability = (snr_db - SNR_MIN_DB) / (SNR_FULL_DB - SNR_MIN_DB)
    return np.clip(reliability, 0.0, 1.0)


def _snr_aware_gain_upper_bounds(snr_db: np.ndarray) -> np.ndarray:
    reliability = _snr_reliability(snr_db)
    return (
        SNR_LOW_RELIABILITY_MAX_BOOST_DB
        + reliability * (GAIN_MAX_DB - SNR_LOW_RELIABILITY_MAX_BOOST_DB)
    )


def _snr_weight_scale_dense(
    dense_freqs: np.ndarray,
    band_centers_hz: np.ndarray,
    band_snr_db: np.ndarray,
) -> np.ndarray:
    reliability = _snr_reliability(band_snr_db)
    band_scale = SNR_LOW_RELIABILITY_WEIGHT + reliability * (1.0 - SNR_LOW_RELIABILITY_WEIGHT)
    return np.interp(
        dense_freqs,
        band_centers_hz,
        band_scale,
        left=float(band_scale[0]),
        right=float(band_scale[-1]),
    )


def _estimate_band_snr_db(
    dense_freqs: np.ndarray,
    measured_dense_db: np.ndarray,
    band_centers_hz: np.ndarray,
) -> np.ndarray:
    voice_mask = (dense_freqs >= TILT_FIT_MIN_HZ) & (dense_freqs <= TILT_FIT_MAX_HZ)
    if np.any(voice_mask):
        noise_floor_db = float(np.percentile(measured_dense_db[voice_mask], 20.0))
    else:
        noise_floor_db = float(np.percentile(measured_dense_db, 20.0))

    band_snr = np.empty(band_centers_hz.size, dtype=float)
    half_oct = 2 ** (1.0 / 6.0)
    for i, fc in enumerate(band_centers_hz):
        band_mask = (dense_freqs >= fc / half_oct) & (dense_freqs <= fc * half_oct)
        if np.any(band_mask):
            band_peak_db = float(np.max(measured_dense_db[band_mask]))
        else:
            band_peak_db = float(np.interp(fc, dense_freqs, measured_dense_db))
        band_snr[i] = band_peak_db - noise_floor_db

    return band_snr
