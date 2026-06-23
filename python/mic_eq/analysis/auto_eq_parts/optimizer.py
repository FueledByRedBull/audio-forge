"""Constrained least-squares optimizer for Auto-EQ."""

import numpy as np
from scipy.optimize import least_squares

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
    gain_ripple = np.diff(gains, n=2)
    center_regularization = np.log(centers_hz / base_centers_hz)
    gain_coupling_excess = np.maximum(
        0.0, np.abs(np.diff(gains)) - MAX_ADJ_GAIN_DIFF_DB
    )

    log_centers = np.log10(base_centers_hz)
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
) -> np.ndarray:
    snr_reliability = np.clip((band_snr_db - 3.0) / 10.0, 0.0, 1.0)
    residual_at_centers = np.abs(np.interp(centers_hz, dense_freqs, residual_db))
    if active_gains is None:
        active_mask = residual_at_centers >= 0.75
    else:
        active_mask = np.abs(active_gains) >= 0.25
    correction_support = np.where(
        active_mask,
        np.clip(residual_at_centers / 2.0, 0.55, 1.0),
        0.85,
    )
    if repeatability_dense is None:
        repeatability = np.full_like(centers_hz, 0.90, dtype=float)
        snr_reliability = np.maximum(snr_reliability, 0.75)
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
    target_profile="static",
    used_spectrum_fallback=False,
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

    # Use normalized measured spectrum for optimization.
    measured_db = measured_db_normalized
    measured_db, tilt_slope = _remove_spectral_tilt(freqs, measured_db)
    debug_log(f"[EQ_CALC] Removed tilt slope: {tilt_slope:.3f} dB/log10(Hz)")

    # Use a dense log-spaced frequency grid for optimization to reduce center-only artifacts.
    dense_freqs = _build_dense_log_grid(freqs)
    measured_dense_db = np.interp(dense_freqs, freqs, measured_db)
    target_dense_db = np.interp(dense_freqs, freqs, target_db)
    repeatability_dense = None
    if spectral_repeatability is not None:
        repeatability_arr = np.asarray(spectral_repeatability, dtype=float)
        if repeatability_arr.shape == np.asarray(freqs).shape:
            repeatability_dense = np.interp(
                dense_freqs,
                freqs,
                np.clip(repeatability_arr, 0.0, 1.0),
            )
    center_selection_weights = _voice_weights(dense_freqs)
    base_centers_hz, q_initial = _select_dynamic_band_layout(
        dense_freqs,
        target_dense_db - measured_dense_db,
        center_selection_weights,
    )
    center_freqs = base_centers_hz.tolist()
    qs_stage1 = q_initial

    band_snr_db = _estimate_band_snr_db(dense_freqs, measured_dense_db, base_centers_hz)
    if global_snr_db is not None:
        band_snr_db = np.maximum(band_snr_db, float(global_snr_db) - 6.0)
    preliminary_confidence = _band_confidence(
        dense_freqs,
        base_centers_hz,
        target_dense_db - measured_dense_db,
        band_snr_db,
        float(voiced_window_ratio),
        repeatability_dense,
    )
    dynamic_gain_upper = _snr_aware_gain_upper_bounds(band_snr_db)
    dynamic_gain_upper = np.minimum(
        dynamic_gain_upper,
        0.35 + preliminary_confidence * preliminary_confidence * (GAIN_MAX_DB - 0.35),
    )
    weights = _voice_weights(dense_freqs) * _snr_weight_scale_dense(
        dense_freqs, base_centers_hz, band_snr_db
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
        band_snr_db,
        float(voiced_window_ratio),
        repeatability_dense,
        active_gains=optimal_gains,
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
        f"[EQ_CALC] Band SNR dB: {[round(v, 1) for v in band_snr_db]}"
    )
    debug_log(
        f"[EQ_CALC] Dynamic max boosts: {[round(v, 2) for v in dynamic_gain_upper]}"
    )
    debug_log(f"[EQ_CALC] Stage2 success: {stage2.success}")
    if hasattr(stage2, "message"):
        debug_log(f"[EQ_CALC] Stage2 message: {stage2.message}")

    # Apply confidence scaling once and let validation decide whether more attenuation is needed.
    optimal_gains = _apply_confidence_gain_scaling(optimal_gains, band_confidences)
    optimal_qs = _regularize_q_for_confidence(
        optimal_qs,
        optimal_gains,
        optimal_centers_hz,
        band_confidences,
    )

    # Clip to SNR-aware boost limits and enforce adjacent-band coupling.
    optimal_gains = np.clip(optimal_gains, gain_lower, dynamic_gain_upper)
    optimal_gains = _enforce_adjacent_gain_limit(optimal_gains, MAX_ADJ_GAIN_DIFF_DB)
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
        band_confidences = band_confidences.copy()
        band_confidences[inactive_mask] = np.maximum(band_confidences[inactive_mask], 0.75)
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

    debug_log(f"[EQ_CALC] Final gains: {[round(g, 2) for g in optimal_gains]}")

    return {
        'band_gains': optimal_gains.tolist(),
        'band_qs': optimal_qs.tolist(),
        'band_freqs': optimal_centers_hz.tolist(),
        'band_confidences': band_confidences.tolist(),
        'analysis_confidence': overall_confidence,
        'eq_confidence': eq_confidence,
        'capture_confidence': capture_confidence,
        'validation_confidence': validation_conf,
        'low_confidence_active_bands': low_confidence_active_bands,
        'active_band_count': int(np.sum(np.abs(optimal_gains) >= 0.25)),
        'validation_before_error_db': before_error,
        'validation_after_error_db': after_error,
        'validation_gain_scale': validation_gain_scale,
        'target_profile': target_profile,
        'used_spectrum_fallback': bool(used_spectrum_fallback),
        'eq_quality': quality_metrics,
    }
