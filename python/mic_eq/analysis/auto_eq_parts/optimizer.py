"""Constrained least-squares optimizer for Auto-EQ."""

import numpy as np
from scipy.optimize import least_squares

from .constants import (
    DEBUG,
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


def calculate_eq_bands(freqs, measured_db, target_db):
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
    center_selection_weights = _voice_weights(dense_freqs)
    base_centers_hz, q_initial = _select_dynamic_band_layout(
        dense_freqs,
        target_dense_db - measured_dense_db,
        center_selection_weights,
    )
    center_freqs = base_centers_hz.tolist()
    qs_stage1 = q_initial

    band_snr_db = _estimate_band_snr_db(dense_freqs, measured_dense_db, base_centers_hz)
    dynamic_gain_upper = _snr_aware_gain_upper_bounds(band_snr_db)
    weights = _voice_weights(dense_freqs) * _snr_weight_scale_dense(
        dense_freqs, base_centers_hz, band_snr_db
    )

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

    # Apply 70% correction factor (prevents over-compensation)
    optimal_gains = optimal_gains * 0.7

    # Clip to SNR-aware boost limits and enforce adjacent-band coupling.
    optimal_gains = np.clip(optimal_gains, gain_lower, dynamic_gain_upper)
    optimal_gains = _enforce_adjacent_gain_limit(optimal_gains, MAX_ADJ_GAIN_DIFF_DB)

    debug_log(f"[EQ_CALC] Final gains (after 70% correction): {[round(g, 2) for g in optimal_gains]}")

    return {
        'band_gains': optimal_gains.tolist(),
        'band_qs': optimal_qs.tolist(),
        'band_freqs': optimal_centers_hz.tolist(),
    }
