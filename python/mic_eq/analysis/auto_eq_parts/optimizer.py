"""Constrained least-squares optimizer for Auto-EQ."""

from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize

from .constants import (
    DEBUG,
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    LAMBDA_CENTER,
    LAMBDA_COUPLING,
    LAMBDA_G,
    LAMBDA_Q,
    LAMBDA_TILT,
    MAX_ADJ_GAIN_DIFF_DB,
    MAX_GAIN_SLOPE_DB_PER_OCTAVE,
    NUM_EQ_BANDS,
    debug_log,
)
from .dynamic_bands import (
    _build_dense_log_grid,
    _center_bounds,
    _enforce_adjacent_gain_limit,
    _estimate_band_snr_db,
    _q_bounds,
    _remove_spectral_tilt,
    _select_dynamic_band_layout,
    _snr_aware_gain_upper_bounds,
    _snr_weight_scale_dense,
    _spectral_tilt_fit,
    _voice_weights,
)
from .response import _predict_eq_response
from ..eq_quality import evaluate_eq_quality, weighted_target_error

def _gain_only_residuals(
    gains: np.ndarray,
    dense_freqs: np.ndarray,
    measured_dense_db: np.ndarray,
    target_dense_db: np.ndarray,
    center_freqs: list[float],
    fixed_qs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    eq_response = _predict_eq_response(dense_freqs, gains, fixed_qs, center_freqs)
    error = target_dense_db - (measured_dense_db + eq_response)
    return np.sqrt(weights) * error


def _log_frequency_gain_curvature(
    gains: np.ndarray,
    centers_hz: np.ndarray,
) -> np.ndarray:
    """Return slope changes on a non-uniform log-frequency grid.

    A gain line that is linear in octaves has zero curvature regardless of how
    the dynamic EQ centers are spaced.
    """
    gains_arr = np.asarray(gains, dtype=float)
    log_centers = np.log2(np.clip(np.asarray(centers_hz, dtype=float), 1e-6, None))
    if gains_arr.size < 3 or log_centers.size != gains_arr.size:
        return np.empty(0, dtype=float)
    spacing = np.maximum(np.diff(log_centers), 1e-6)
    slopes = np.diff(gains_arr) / spacing
    local_span = 0.5 * (spacing[:-1] + spacing[1:])
    return np.diff(slopes) * local_span


def _adjacent_gain_limits(centers_hz: np.ndarray) -> np.ndarray:
    octave_spacing = np.maximum(
        np.diff(np.log2(np.clip(np.asarray(centers_hz, dtype=float), 1e-6, None))),
        1e-6,
    )
    return np.minimum(
        MAX_ADJ_GAIN_DIFF_DB,
        MAX_GAIN_SLOPE_DB_PER_OCTAVE * octave_spacing,
    )


def _joint_gain_q_residuals(
    params: np.ndarray,
    dense_freqs: np.ndarray,
    measured_dense_db: np.ndarray,
    target_dense_db: np.ndarray,
    base_centers_hz: np.ndarray,
    weights: np.ndarray,
    q_prior: np.ndarray,
) -> np.ndarray:
    gains = params[:NUM_EQ_BANDS]
    qs = params[NUM_EQ_BANDS:2 * NUM_EQ_BANDS]
    centers_hz = params[2 * NUM_EQ_BANDS:]

    eq_response = _predict_eq_response(dense_freqs, gains, qs, centers_hz)
    error = target_dense_db - (measured_dense_db + eq_response)

    q_regularization = np.log(qs / q_prior)
    gain_ripple = _log_frequency_gain_curvature(gains, centers_hz)
    center_regularization = np.log(centers_hz / base_centers_hz)
    gain_coupling_excess = np.maximum(
        0.0,
        np.abs(np.diff(gains)) - _adjacent_gain_limits(centers_hz),
    )

    log_centers = np.log10(centers_hz)
    centered_log_centers = log_centers - np.mean(log_centers)
    denom = float(np.sum(centered_log_centers ** 2))
    tilt_slope = float(np.dot(centered_log_centers, gains) / denom) if denom > 0.0 else 0.0

    return np.concatenate(
        [
            np.sqrt(weights) * error,
            np.sqrt(LAMBDA_Q) * q_regularization,
            np.sqrt(LAMBDA_G) * gain_ripple,
            np.sqrt(LAMBDA_CENTER) * center_regularization,
            np.sqrt(LAMBDA_COUPLING) * gain_coupling_excess,
            np.array([np.sqrt(LAMBDA_TILT) * tilt_slope]),
        ]
    )


def _band_confidence(
    dense_freqs: np.ndarray,
    centers_hz: np.ndarray,
    residual_db: np.ndarray,
    band_snr_db: np.ndarray,
    voiced_window_ratio: float,
    repeatability_dense: np.ndarray | None,
    active_gains: np.ndarray | None = None,
    *,
    snr_available: bool = True,
) -> np.ndarray:
    if snr_available:
        snr_reliability = np.clip((band_snr_db - 3.0) / 10.0, 0.0, 1.0)
    else:
        snr_reliability = np.full_like(centers_hz, 0.45, dtype=float)
    residual_at_centers = np.abs(np.interp(centers_hz, dense_freqs, residual_db))
    if active_gains is None:
        active_mask = residual_at_centers >= 0.75
    else:
        active_mask = np.abs(active_gains) >= 0.25
    correction_support = np.where(
        active_mask,
        np.clip(residual_at_centers / 2.0, 0.55, 1.0),
        0.55,
    )
    if repeatability_dense is None:
        repeatability = np.full_like(centers_hz, 0.60, dtype=float)
    else:
        repeatability = np.interp(
            centers_hz,
            dense_freqs,
            repeatability_dense,
            left=float(repeatability_dense[0]),
            right=float(repeatability_dense[-1]),
        )
    coverage = np.clip(voiced_window_ratio / 0.55, 0.0, 1.0)
    confidence = (
        0.25 * snr_reliability
        + 0.35 * repeatability
        + 0.25 * correction_support
        + 0.15 * coverage
    )
    return np.clip(confidence, 0.0, 1.0)


def _constrained_gain_refinement(
    gains: np.ndarray,
    dense_freqs: np.ndarray,
    measured_dense_db: np.ndarray,
    target_dense_db: np.ndarray,
    qs: np.ndarray,
    centers_hz: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Re-optimize confidence-scaled gains under explicit adjacency bounds."""
    gains_arr = np.asarray(gains, dtype=float)
    adjacent_limits = _adjacent_gain_limits(centers_hz)
    projected = _enforce_adjacent_gain_limit(gains_arr, adjacent_limits)
    lower = np.minimum(gains_arr, 0.0)
    upper = np.maximum(gains_arr, 0.0)
    projected = np.clip(projected, lower, upper)

    def objective(candidate: np.ndarray) -> float:
        response = _predict_eq_response(dense_freqs, candidate, qs, centers_hz)
        error = target_dense_db - (measured_dense_db + response)
        curvature = _log_frequency_gain_curvature(candidate, centers_hz)
        log_centers = np.log10(np.clip(centers_hz, 1e-6, None))
        centered_log_centers = log_centers - float(np.mean(log_centers))
        denominator = float(np.sum(centered_log_centers**2))
        tilt = (
            float(np.dot(centered_log_centers, candidate) / denominator)
            if denominator > 0.0
            else 0.0
        )
        return float(
            np.sum(weights * error * error)
            + LAMBDA_G * np.sum(curvature * curvature)
            + LAMBDA_TILT * tilt * tilt
        )

    def adjacency_slack(candidate: np.ndarray) -> np.ndarray:
        return adjacent_limits - np.abs(np.diff(candidate))

    solver_bounds: Any = list(zip(lower.tolist(), upper.tolist(), strict=True))
    solver_constraints: Any = (
        {
            "type": "ineq",
            "fun": adjacency_slack,
        },
    )
    result = minimize(
        objective,
        projected,
        method="SLSQP",
        bounds=solver_bounds,
        constraints=solver_constraints,
        options={"ftol": 1e-7, "maxiter": 120, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        candidate = np.asarray(result.x, dtype=float)
        if np.all(np.abs(np.diff(candidate)) <= adjacent_limits + 1e-6):
            return candidate, True
    return projected, False


def _regularize_q_for_confidence(
    qs: np.ndarray,
    gains: np.ndarray,
    centers_hz: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    q_low, q_high = _q_bounds(centers_hz.tolist())
    bounded = np.clip(qs, q_low, q_high)
    for i, gain in enumerate(gains):
        conf = float(confidence[i])
        if abs(gain) < 0.25:
            continue
        if conf < 0.65:
            bounded[i] = min(bounded[i], 1.0 + conf * 2.0)
        if gain > 0.0:
            bounded[i] = min(bounded[i], 4.2 if conf > 0.75 else 2.8)
        if centers_hz[i] < 250.0:
            bounded[i] = min(bounded[i], 1.8 if gain > 0.0 else 2.2)

    for i in range(1, bounded.size):
        if gains[i - 1] > 2.0 and gains[i] > 2.0:
            octave_gap = abs(float(np.log2(centers_hz[i] / centers_hz[i - 1])))
            if octave_gap < 0.45:
                bounded[i - 1] = min(bounded[i - 1], 2.5)
                bounded[i] = min(bounded[i], 2.5)
    return np.clip(bounded, q_low, q_high)


def _apply_confidence_gain_scaling(
    gains: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    scaled = gains.copy()
    for i, gain in enumerate(scaled):
        conf = float(confidence[i])
        if gain > 0.0:
            max_boost = 0.35 + conf * conf * 7.65
            scaled[i] = min(gain * (0.35 + 0.65 * conf), max_boost)
        else:
            scaled[i] = gain * (0.55 + 0.45 * conf)
        if conf < 0.20:
            scaled[i] *= 0.15
    return scaled


def _validation_confidence(
    before_error: float,
    after_error: float,
    validation_gain_scale: float,
) -> float:
    if before_error <= 1e-9:
        improvement_score = 1.0
    else:
        improvement_ratio = max(0.0, (before_error - after_error) / before_error)
        improvement_score = np.clip(improvement_ratio / 0.20, 0.0, 1.0)
    return float(
        np.clip(
            0.35 + 0.35 * improvement_score + 0.30 * float(validation_gain_scale),
            0.0,
            1.0,
        )
    )


def _smooth_log_frequency_values(
    freqs: np.ndarray,
    values: np.ndarray,
    width_octaves: float,
) -> np.ndarray:
    safe_freqs = np.clip(np.asarray(freqs, dtype=float), 20.0, None)
    values = np.asarray(values, dtype=float)
    log_freqs = np.log2(safe_freqs)
    smoothed = np.empty_like(values)
    width = max(float(width_octaves), 1e-3)
    for index, center in enumerate(log_freqs):
        distance = (log_freqs - center) / width
        weights = np.exp(-0.5 * distance * distance)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            smoothed[index] = values[index]
        else:
            smoothed[index] = float(np.dot(weights, values) / weight_sum)
    return smoothed


def _regularize_correction_residual(
    dense_freqs: np.ndarray,
    residual_db: np.ndarray,
    smoothing_strength: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
    strength = str(smoothing_strength or "conservative").strip().lower()
    if strength not in {"off", "balanced", "conservative", "broad"}:
        strength = "conservative"
    residual_db = np.asarray(residual_db, dtype=float)
    if strength == "off":
        return residual_db.copy(), {
            "smoothing_strength": "off",
            "max_requested_correction_db": float(np.max(np.abs(residual_db))),
            "max_regularized_correction_db": float(np.max(np.abs(residual_db))),
            "max_narrow_residual_db": 0.0,
        }

    medium = _smooth_log_frequency_values(dense_freqs, residual_db, 0.16)
    broad_width = 0.40 if strength == "conservative" else 0.55 if strength == "broad" else 0.28
    broad = _smooth_log_frequency_values(dense_freqs, residual_db, broad_width)
    max_local_excursion = 3.0 if strength == "conservative" else 2.0 if strength == "broad" else 5.0
    broad_blend = 0.35 if strength == "conservative" else 0.55 if strength == "broad" else 0.18

    local = np.clip(residual_db - medium, -max_local_excursion, max_local_excursion)
    clamped = medium + local
    regularized = (1.0 - broad_blend) * clamped + broad_blend * broad
    return regularized, {
        "smoothing_strength": strength,
        "max_requested_correction_db": float(np.max(np.abs(residual_db))),
        "max_regularized_correction_db": float(np.max(np.abs(regularized))),
        "max_narrow_residual_db": float(np.max(np.abs(residual_db - broad))),
    }


def _overall_confidence(
    band_confidences: np.ndarray,
    gains: np.ndarray,
    capture_confidence: float | None,
    validation_confidence: float,
) -> tuple[float, float, float]:
    active_mask = np.abs(gains) >= 0.25
    if np.any(active_mask):
        eq_confidence = float(np.mean(band_confidences[active_mask]))
    else:
        eq_confidence = float(np.mean(band_confidences))
    capture_score = float(capture_confidence) if capture_confidence is not None else 1.0
    overall = float(
        np.clip(
            0.55 * eq_confidence + 0.25 * capture_score + 0.20 * validation_confidence,
            0.0,
            1.0,
        )
    )
    return overall, eq_confidence, capture_score


def _validate_and_attenuate_solution(
    dense_freqs: np.ndarray,
    measured_dense_db: np.ndarray,
    target_dense_db: np.ndarray,
    gains: np.ndarray,
    qs: np.ndarray,
    centers_hz: np.ndarray,
    confidence: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    before_error = weighted_target_error(
        dense_freqs,
        measured_dense_db,
        target_dense_db,
        np.zeros_like(gains),
        qs,
        centers_hz,
        weights,
    )
    best_gains = gains.copy()
    best_error = float("inf")
    best_scale = 1.0
    best_metrics = evaluate_eq_quality(centers_hz, best_gains, qs).to_dict()

    for scale in (1.0, 0.85, 0.70, 0.55, 0.40, 0.25):
        candidate = gains * scale
        metrics = evaluate_eq_quality(centers_hz, candidate, qs)
        after_error = weighted_target_error(
            dense_freqs,
            measured_dense_db,
            target_dense_db,
            candidate,
            qs,
            centers_hz,
            weights,
        )
        if after_error < best_error and metrics.risk_score < 1.8:
            best_error = after_error
            best_gains = candidate
            best_scale = scale
            best_metrics = metrics.to_dict()
        if after_error <= before_error * 0.98 and metrics.risk_score < 1.0:
            return candidate, before_error, after_error, scale, metrics.to_dict()

    if best_error > before_error:
        candidate = best_gains.copy()
        harmful_order = np.argsort(confidence)
        for idx in harmful_order[:3]:
            if candidate[idx] > 0.0:
                candidate[idx] = 0.0
                candidate_error = weighted_target_error(
                    dense_freqs,
                    measured_dense_db,
                    target_dense_db,
                    candidate,
                    qs,
                    centers_hz,
                    weights,
                )
                candidate_metrics = evaluate_eq_quality(centers_hz, candidate, qs)
                if candidate_error <= before_error and candidate_metrics.risk_score < 1.5:
                    return (
                        candidate,
                        before_error,
                        candidate_error,
                        best_scale,
                        candidate_metrics.to_dict(),
                    )

    return best_gains, before_error, best_error, best_scale, best_metrics


def calculate_eq_bands(
    freqs,
    measured_db,
    target_db,
    *,
    spectral_repeatability=None,
    voiced_window_ratio=1.0,
    analysis_confidence=None,
    global_snr_db=None,
    spectral_snr_db=None,
    noise_reference_source="unavailable",
    target_profile="static",
    used_spectrum_fallback=False,
    smoothing_strength="conservative",
    tilt_policy="preserve",
):
    """
    Calculate optimal 10-band EQ settings using least-squares optimization.

    Finds gains that minimize error between target curve and
    (measured spectrum + EQ response). Accounts for band interaction.

    Args:
        freqs: Frequency array (Hz)
        measured_db: Measured spectrum in dBFS (dB relative to full scale)
        target_db: Target curve in dB (relative adjustments)

    Returns:
        eq_settings: Dict with 'band_gains' and 'band_qs' (10-element lists)
    """
    # DEBUG: Log what we're working with
    debug_log(f"[EQ_CALC] Measured spectrum range: [{measured_db.min():.1f}, {measured_db.max():.1f}] dB")
    debug_log(f"[EQ_CALC] Target curve range: [{target_db.min():.1f}, {target_db.max():.1f}] dB")

    # CRITICAL FIX: Normalize measured spectrum to relative dB
    # The measured spectrum is in dBFS (always negative for speech)
    # The target curve is relative adjustments (0 to +4 dB)
    # We need to normalize the measured spectrum to compare like-to-like
    #
    # Approach: Find the average level in the voice range (100-8000 Hz)
    # and normalize relative to that average
    voice_range_mask = (freqs >= 100) & (freqs <= 8000)
    if np.any(voice_range_mask):
        voice_avg = np.mean(measured_db[voice_range_mask])
    else:
        voice_avg = np.mean(measured_db)

    # Normalize: subtract the average to get relative dB
    measured_db_normalized = measured_db - voice_avg

    debug_log(f"[EQ_CALC] Voice range average: {voice_avg:.1f} dB")
    debug_log(
        f"[EQ_CALC] Normalized measured range: [{measured_db_normalized.min():.1f}, {measured_db_normalized.max():.1f}] dB"
    )
    debug_log(
        f"[EQ_CALC] Difference (target - normalized): avg {(target_db - measured_db_normalized).mean():.2f} dB"
    )

    # Preserve broad microphone/voice tilt unless the caller explicitly opts
    # into the legacy detrending experiment. Silent detrending can erase the
    # exact dark/bright response a static target is intended to correct.
    measured_db = measured_db_normalized
    tilt_policy = str(tilt_policy or "preserve").strip().lower()
    if tilt_policy not in {"preserve", "detrend"}:
        raise ValueError(f"Unknown spectral tilt policy: {tilt_policy}")
    _tilt_component, tilt_slope, tilt_fit_r2 = _spectral_tilt_fit(freqs, measured_db)
    if tilt_policy == "detrend":
        measured_db, tilt_slope = _remove_spectral_tilt(freqs, measured_db)
    debug_log(
        f"[EQ_CALC] Spectral tilt: {tilt_slope:.3f} dB/log10(Hz), "
        f"R2={tilt_fit_r2:.3f}, policy={tilt_policy}"
    )

    # Use a dense log-spaced frequency grid for optimization to reduce center-only artifacts.
    dense_freqs = _build_dense_log_grid(freqs)
    measured_dense_db = np.interp(dense_freqs, freqs, measured_db)
    target_dense_db = np.interp(dense_freqs, freqs, target_db)
    target_residual_dense = target_dense_db - measured_dense_db
    target_residual_dense, residual_regularization = _regularize_correction_residual(
        dense_freqs,
        target_residual_dense,
        smoothing_strength,
    )
    target_dense_db = measured_dense_db + target_residual_dense
    repeatability_dense = None
    if spectral_repeatability is not None:
        repeatability_arr = np.asarray(spectral_repeatability, dtype=float)
        if repeatability_arr.shape == np.asarray(freqs).shape:
            repeatability_dense = np.interp(
                dense_freqs,
                freqs,
                np.clip(repeatability_arr, 0.0, 1.0),
            )
    spectral_snr_dense = None
    if spectral_snr_db is not None:
        spectral_snr_arr = np.asarray(spectral_snr_db, dtype=float)
        if spectral_snr_arr.shape == np.asarray(freqs).shape:
            spectral_snr_dense = np.interp(
                dense_freqs,
                freqs,
                spectral_snr_arr,
            )
    center_selection_weights = _voice_weights(dense_freqs)
    base_centers_hz, q_initial = _select_dynamic_band_layout(
        dense_freqs,
        target_dense_db - measured_dense_db,
        center_selection_weights,
    )
    center_freqs = base_centers_hz.tolist()
    qs_stage1 = q_initial

    band_snr_db = _estimate_band_snr_db(
        dense_freqs,
        spectral_snr_dense,
        base_centers_hz,
    )
    snr_available = bool(np.any(np.isfinite(band_snr_db)))
    effective_band_snr_db = np.where(
        np.isfinite(band_snr_db),
        band_snr_db,
        18.0,
    )
    measurement_metadata_available = bool(
        spectral_repeatability is not None or analysis_confidence is not None
    )
    preliminary_confidence = _band_confidence(
        dense_freqs,
        base_centers_hz,
        target_dense_db - measured_dense_db,
        effective_band_snr_db,
        float(voiced_window_ratio),
        repeatability_dense,
        snr_available=snr_available,
    )
    dynamic_gain_upper = (
        _snr_aware_gain_upper_bounds(effective_band_snr_db)
        if snr_available
        else np.full(NUM_EQ_BANDS, GAIN_MAX_DB, dtype=float)
    )
    if measurement_metadata_available:
        dynamic_gain_upper = np.minimum(
            dynamic_gain_upper,
            0.35 + preliminary_confidence * preliminary_confidence * (GAIN_MAX_DB - 0.35),
        )
    weights = _voice_weights(dense_freqs)
    if snr_available:
        weights = weights * _snr_weight_scale_dense(
            dense_freqs,
            base_centers_hz,
            effective_band_snr_db,
        )
    if repeatability_dense is not None:
        weights = weights * (0.35 + 0.65 * repeatability_dense)

    measured_db_at_centers = np.interp(center_freqs, dense_freqs, measured_dense_db)
    target_db_at_centers = np.interp(center_freqs, dense_freqs, target_dense_db)
    desired_gains = target_db_at_centers - measured_db_at_centers
    gain_lower = np.full(NUM_EQ_BANDS, GAIN_MIN_DB, dtype=float)
    gains_initial = np.clip(desired_gains, gain_lower, dynamic_gain_upper)

    verbose_level = 2 if DEBUG else 0

    # Stage 1: stable gain-only solve with fixed Q prior.
    stage1 = least_squares(
        _gain_only_residuals,
        gains_initial,
        args=(
            dense_freqs,
            measured_dense_db,
            target_dense_db,
            center_freqs,
            qs_stage1,
            weights,
        ),
        bounds=(gain_lower, dynamic_gain_upper),
        method="trf",
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-6,
        max_nfev=120,
        verbose=verbose_level,
    )
    gains_stage1 = stage1.x

    # Stage 2: refine gains + Q with bounded Q and local center refinement.
    q_low, q_high = _q_bounds(center_freqs)
    center_low, center_high = _center_bounds(base_centers_hz)
    q_prior = np.clip(q_initial, q_low, q_high)
    params_initial = np.concatenate([gains_stage1, q_prior, base_centers_hz])
    params_lower = np.concatenate(
        [gain_lower, q_low, center_low]
    )
    params_upper = np.concatenate(
        [dynamic_gain_upper, q_high, center_high]
    )
    stage2 = least_squares(
        _joint_gain_q_residuals,
        params_initial,
        args=(
            dense_freqs,
            measured_dense_db,
            target_dense_db,
            base_centers_hz,
            weights,
            q_prior,
        ),
        bounds=(params_lower, params_upper),
        method="trf",
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-6,
        max_nfev=180,
        verbose=verbose_level,
    )
    optimal_gains = stage2.x[:NUM_EQ_BANDS]
    optimal_qs = stage2.x[NUM_EQ_BANDS:2 * NUM_EQ_BANDS]
    optimal_centers_hz = stage2.x[2 * NUM_EQ_BANDS:]
    band_confidences = _band_confidence(
        dense_freqs,
        optimal_centers_hz,
        target_dense_db - measured_dense_db,
        effective_band_snr_db,
        float(voiced_window_ratio),
        repeatability_dense,
        active_gains=optimal_gains,
        snr_available=snr_available,
    )

    debug_log(
        f"[EQ_CALC] Dynamic base centers: {[round(fc, 1) for fc in base_centers_hz]}"
    )
    debug_log(
        f"[EQ_CALC] Dynamic Q priors: {[round(q, 3) for q in q_prior]}"
    )
    debug_log(
        f"[EQ_CALC] Stage1 gains: {[round(g, 2) for g in gains_stage1]}"
    )
    debug_log(
        f"[EQ_CALC] Stage2 gains (raw): {[round(g, 2) for g in optimal_gains]}"
    )
    debug_log(
        f"[EQ_CALC] Stage2 Qs: {[round(q, 3) for q in optimal_qs]}"
    )
    debug_log(
        f"[EQ_CALC] Stage2 centers: {[round(fc, 1) for fc in optimal_centers_hz]}"
    )
    debug_log(
        f"[EQ_CALC] Band SNR dB: "
        f"{[round(v, 1) if np.isfinite(v) else None for v in band_snr_db]}"
    )
    debug_log(
        f"[EQ_CALC] Dynamic max boosts: {[round(v, 2) for v in dynamic_gain_upper]}"
    )
    debug_log(f"[EQ_CALC] Stage2 success: {stage2.success}")
    if hasattr(stage2, "message"):
        debug_log(f"[EQ_CALC] Stage2 message: {stage2.message}")

    # Apply measurement confidence only when this came from the production
    # analysis pipeline. Low-level synthetic callers do not have capture
    # metadata and retain the solver's ordinary bounds.
    if measurement_metadata_available:
        optimal_gains = _apply_confidence_gain_scaling(optimal_gains, band_confidences)
    optimal_qs = _regularize_q_for_confidence(
        optimal_qs,
        optimal_gains,
        optimal_centers_hz,
        band_confidences,
    )

    # Re-optimize confidence-scaled gains under explicit adjacent-band
    # constraints. The projection is only a feasible starting point/fallback;
    # successful output is a constrained optimum rather than a post-hoc clamp.
    optimal_gains = np.clip(optimal_gains, gain_lower, dynamic_gain_upper)
    optimal_gains, constraint_solver_success = _constrained_gain_refinement(
        optimal_gains,
        dense_freqs,
        measured_dense_db,
        target_dense_db,
        optimal_qs,
        optimal_centers_hz,
        weights,
    )
    (
        optimal_gains,
        before_error,
        after_error,
        validation_gain_scale,
        quality_metrics,
    ) = _validate_and_attenuate_solution(
        dense_freqs,
        measured_dense_db,
        target_dense_db,
        optimal_gains,
        optimal_qs,
        optimal_centers_hz,
        band_confidences,
        weights,
    )
    inactive_mask = np.abs(optimal_gains) < 0.25
    if np.any(inactive_mask):
        optimal_gains = optimal_gains.copy()
        optimal_gains[inactive_mask] = 0.0
        optimal_gains, inactive_constraint_solver_success = (
            _constrained_gain_refinement(
                optimal_gains,
                dense_freqs,
                measured_dense_db,
                target_dense_db,
                optimal_qs,
                optimal_centers_hz,
                weights,
            )
        )
        constraint_solver_success = bool(
            constraint_solver_success and inactive_constraint_solver_success
        )
        after_error = weighted_target_error(
            dense_freqs,
            measured_dense_db,
            target_dense_db,
            optimal_gains,
            optimal_qs,
            optimal_centers_hz,
            weights,
        )

    validation_conf = _validation_confidence(before_error, after_error, validation_gain_scale)
    overall_confidence, eq_confidence, capture_confidence = _overall_confidence(
        band_confidences,
        optimal_gains,
        analysis_confidence,
        validation_conf,
    )
    low_confidence_active_bands = int(
        np.sum((np.abs(optimal_gains) >= 0.25) & (band_confidences < 0.45))
    )
    recommendation_status = "apply"
    abstention_reasons: list[str] = []
    if used_spectrum_fallback:
        abstention_reasons.append("insufficient repeatable voiced windows")
    if analysis_confidence is not None and float(analysis_confidence) < 0.35:
        abstention_reasons.append("capture quality score is too low")
    if snr_available and np.nanmedian(band_snr_db) < 3.0:
        abstention_reasons.append("noise-referenced SNR is too low")
    if low_confidence_active_bands >= 3:
        abstention_reasons.append("too many active bands lack measurement support")
    if abstention_reasons:
        recommendation_status = "abstain"
        optimal_gains = np.zeros_like(optimal_gains)
        after_error = before_error
    elif overall_confidence < 0.60 or validation_gain_scale < 0.70:
        recommendation_status = "reduced"

    debug_log(f"[EQ_CALC] Final gains: {[round(g, 2) for g in optimal_gains]}")

    return {
        'band_gains': optimal_gains.tolist(),
        'band_qs': optimal_qs.tolist(),
        'band_freqs': optimal_centers_hz.tolist(),
        'band_confidences': band_confidences.tolist(),
        'band_snr_db': [
            float(value) if np.isfinite(value) else None
            for value in band_snr_db
        ],
        'noise_referenced_snr_db': (
            float(global_snr_db)
            if snr_available and global_snr_db is not None
            else None
        ),
        'analysis_confidence': overall_confidence,
        'eq_confidence': eq_confidence,
        'capture_confidence': capture_confidence,
        'validation_confidence': validation_conf,
        'low_confidence_active_bands': low_confidence_active_bands,
        'active_band_count': int(np.sum(np.abs(optimal_gains) >= 0.25)),
        'recommendation_status': recommendation_status,
        'apply_recommended': recommendation_status != "abstain",
        'abstention_reasons': abstention_reasons,
        'confidence_semantics': 'bounded_quality_score',
        'snr_reference_available': snr_available,
        'noise_reference_source': (
            str(noise_reference_source)
            if snr_available
            else "unavailable"
        ),
        'spectral_tilt_policy': tilt_policy,
        'spectral_tilt_slope_db_per_decade': tilt_slope,
        'spectral_tilt_fit_r2': tilt_fit_r2,
        'constraint_solver_success': constraint_solver_success,
        'max_adjacent_gain_difference_db': float(
            np.max(np.abs(np.diff(optimal_gains)))
        ),
        'max_adjacent_gain_slope_db_per_octave': float(
            np.max(
                np.abs(np.diff(optimal_gains))
                / np.maximum(
                    np.diff(
                        np.log2(
                            np.clip(
                                optimal_centers_hz,
                                1e-6,
                                None,
                            )
                        )
                    ),
                    1e-6,
                )
            )
        ),
        'validation_before_error_db': before_error,
        'validation_after_error_db': after_error,
        'validation_gain_scale': validation_gain_scale,
        'target_profile': target_profile,
        'smoothing_strength': residual_regularization["smoothing_strength"],
        'residual_regularization': residual_regularization,
        'used_spectrum_fallback': bool(used_spectrum_fallback),
        'eq_quality': quality_metrics,
    }
