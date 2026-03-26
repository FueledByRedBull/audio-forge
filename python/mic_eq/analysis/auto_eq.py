"""
Auto-EQ calculation using constrained least-squares fitting.

Uses a two-stage optimization strategy:
1) optimize gains with fixed Q (stable coarse solve)
2) refine gains + Q jointly on a dense log-frequency grid with regularization
"""
import os

import numpy as np
from scipy.optimize import least_squares

from mic_eq.config import (
    TARGET_CURVES,
    EQ_FREQUENCIES,
    AUTO_EQ_DEFAULT_Q
)
DEBUG = os.environ.get("AUDIOFORGE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
SAMPLE_RATE = 48_000.0
NUM_EQ_BANDS = 10
GAIN_MIN_DB = -12.0
GAIN_MAX_DB = 12.0
Q_PRIOR = AUTO_EQ_DEFAULT_Q
Q_MIN = 0.3
Q_MAX = 6.0
LOW_BAND_Q_MAX = 2.5
LOW_BAND_Q_MAX_HZ = 250.0
DENSE_GRID_POINTS = 256
VOICE_WEIGHT = 2.0
OUT_OF_BAND_WEIGHT = 0.8
LAMBDA_Q = 10.0
LAMBDA_G = 0.35
LAMBDA_CENTER = 16.0
LAMBDA_TILT = 0.08
LAMBDA_COUPLING = 8.0
CENTER_NUDGE_PCT = 0.15
LOW_BAND_CENTER_NUDGE_PCT = 0.10
MAX_ADJ_GAIN_DIFF_DB = 6.0
TILT_FIT_MIN_HZ = 100.0
TILT_FIT_MAX_HZ = 8000.0
TILT_MIN_FIT_R2 = 0.65
SNR_MIN_DB = 3.0
SNR_FULL_DB = 18.0
SNR_LOW_RELIABILITY_WEIGHT = 0.35
SNR_LOW_RELIABILITY_MAX_BOOST_DB = 3.0


def _debug_log(message: str) -> None:
    if DEBUG:
        print(message)


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


def _center_bounds(base_centers_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center_low = base_centers_hz * (1.0 - CENTER_NUDGE_PCT)
    center_high = base_centers_hz * (1.0 + CENTER_NUDGE_PCT)

    low_mask = base_centers_hz < LOW_BAND_Q_MAX_HZ
    center_low[low_mask] = base_centers_hz[low_mask] * (1.0 - LOW_BAND_CENTER_NUDGE_PCT)
    center_high[low_mask] = base_centers_hz[low_mask] * (1.0 + LOW_BAND_CENTER_NUDGE_PCT)
    return center_low, center_high


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


def get_target_curve(freqs, target_preset='broadcast'):
    """
    Get target curve values at specified frequencies.

    Args:
        freqs: Frequency array (Hz)
        target_preset: Target curve name ('broadcast', 'podcast', 'streaming', 'flat')

    Returns:
        target_db: Target dB values at each frequency
    """
    if target_preset not in TARGET_CURVES:
        raise ValueError(f"Unknown target preset: {target_preset}")

    target_curve = TARGET_CURVES[target_preset]

    # Interpolate target curve to requested frequencies
    target_db = np.interp(
        freqs,
        EQ_FREQUENCIES,
        target_curve.band_targets,
        left=target_curve.band_targets[0],
        right=target_curve.band_targets[-1]
    )

    return target_db


def _predict_eq_response(freqs, gains, qs, center_freqs):
    """
    Predict how EQ settings affect frequency response.

    Simulates the combined effect of all 10 parametric EQ bands.
    This accounts for band interaction - each band affects neighboring
    frequencies based on its Q factor.

    Args:
        freqs: Frequency array (Hz)
        gains: List of 10 gain values (dB)
        qs: List of 10 Q values
        center_freqs: List of 10 center frequencies (Hz)

    Returns:
        response_db: Combined EQ response in dB at each frequency
    """
    # Initialize response in linear domain (1.0 = unity gain = 0 dB)
    response_linear = np.ones_like(freqs, dtype=float)

    for gain_db, q, fc in zip(gains, qs, center_freqs):
        # Skip bands with no gain (optimization)
        if abs(gain_db) < 0.01:
            continue

        # Convert dB gain to linear amplitude
        # For biquad filters: A = 10^(dB/40)
        # gain_db = 0  -> A = 1 (no change)
        # gain_db = 6  -> A ≈ 1.995 (2x amplitude = 6dB boost)
        A = 10 ** (gain_db / 40.0)

        # Compute peaking EQ coefficients (Audio EQ Cookbook, RBJ).
        # At center frequency: magnitude response = A^2 = 10^(dB/20)
        w0 = 2 * np.pi * fc / SAMPLE_RATE
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)

        # Biquad coefficients for peaking EQ
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        # Evaluate transfer function H(z) at each frequency
        # H(e^(j*w)) = (b0 + b1*e^(-j*w) + b2*e^(-j*2w)) / (a0 + a1*e^(-j*w) + a2*e^(-j*2w))

        # Normalized frequency: w = 2*pi*f/fs
        w = 2 * np.pi * freqs / SAMPLE_RATE
        z_inv = np.exp(-1j * w)  # z^(-1)
        z_inv_2 = z_inv ** 2      # z^(-2)

        # Numerator and denominator of transfer function
        numerator = b0 + b1 * z_inv + b2 * z_inv_2
        denominator = a0 + a1 * z_inv + a2 * z_inv_2

        # Magnitude response = |H(e^(j*w))|
        magnitude = np.abs(numerator / denominator)

        # Accumulate in linear domain (multiply responses)
        response_linear *= magnitude

    # Convert back to dB
    response_db = 20 * np.log10(np.maximum(response_linear, 1e-12))

    return response_db


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
    _debug_log(f"[EQ_CALC] Measured spectrum range: [{measured_db.min():.1f}, {measured_db.max():.1f}] dB")
    _debug_log(f"[EQ_CALC] Target curve range: [{target_db.min():.1f}, {target_db.max():.1f}] dB")

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

    _debug_log(f"[EQ_CALC] Voice range average: {voice_avg:.1f} dB")
    _debug_log(
        f"[EQ_CALC] Normalized measured range: [{measured_db_normalized.min():.1f}, {measured_db_normalized.max():.1f}] dB"
    )
    _debug_log(
        f"[EQ_CALC] Difference (target - normalized): avg {(target_db - measured_db_normalized).mean():.2f} dB"
    )

    # Use normalized measured spectrum for optimization.
    measured_db = measured_db_normalized
    measured_db, tilt_slope = _remove_spectral_tilt(freqs, measured_db)
    _debug_log(f"[EQ_CALC] Removed tilt slope: {tilt_slope:.3f} dB/log10(Hz)")

    center_freqs = EQ_FREQUENCIES
    base_centers_hz = np.asarray(center_freqs, dtype=float)
    qs_stage1 = np.full(NUM_EQ_BANDS, AUTO_EQ_DEFAULT_Q, dtype=float)

    # Use a dense log-spaced frequency grid for optimization to reduce center-only artifacts.
    dense_freqs = _build_dense_log_grid(freqs)
    measured_dense_db = np.interp(dense_freqs, freqs, measured_db)
    target_dense_db = np.interp(dense_freqs, freqs, target_db)
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

    # Stage 2: refine gains + Q with bounded Q and regularization.
    q_low, q_high = _q_bounds(center_freqs)
    center_low, center_high = _center_bounds(base_centers_hz)
    q_prior = np.clip(np.full(NUM_EQ_BANDS, Q_PRIOR, dtype=float), q_low, q_high)
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

    _debug_log(
        f"[EQ_CALC] Stage1 gains: {[round(g, 2) for g in gains_stage1]}"
    )
    _debug_log(
        f"[EQ_CALC] Stage2 gains (raw): {[round(g, 2) for g in optimal_gains]}"
    )
    _debug_log(
        f"[EQ_CALC] Stage2 Qs: {[round(q, 3) for q in optimal_qs]}"
    )
    _debug_log(
        f"[EQ_CALC] Stage2 centers: {[round(fc, 1) for fc in optimal_centers_hz]}"
    )
    _debug_log(
        f"[EQ_CALC] Band SNR dB: {[round(v, 1) for v in band_snr_db]}"
    )
    _debug_log(
        f"[EQ_CALC] Dynamic max boosts: {[round(v, 2) for v in dynamic_gain_upper]}"
    )
    _debug_log(f"[EQ_CALC] Stage2 success: {stage2.success}")
    if hasattr(stage2, "message"):
        _debug_log(f"[EQ_CALC] Stage2 message: {stage2.message}")

    # Apply 70% correction factor (prevents over-compensation)
    optimal_gains = optimal_gains * 0.7

    # Clip to SNR-aware boost limits and enforce adjacent-band coupling.
    optimal_gains = np.clip(optimal_gains, gain_lower, dynamic_gain_upper)
    optimal_gains = _enforce_adjacent_gain_limit(optimal_gains, MAX_ADJ_GAIN_DIFF_DB)

    _debug_log(f"[EQ_CALC] Final gains (after 70% correction): {[round(g, 2) for g in optimal_gains]}")

    return {
        'band_gains': optimal_gains.tolist(),
        'band_qs': optimal_qs.tolist(),
        'band_freqs': optimal_centers_hz.tolist(),
    }


def analyze_auto_eq(audio_data, sample_rate, target_preset='broadcast'):
    """
    Complete auto-EQ analysis pipeline.

    High-level function that runs the full analysis pipeline:
    1. Compute spectrum with Hamming window
    2. Apply 1/6 octave smoothing
    3. Get target curve
    4. Calculate optimal EQ bands using least-squares
    5. Validate results

    Args:
        audio_data: Recorded audio samples (float32 NumPy array)
        sample_rate: Sample rate in Hz (should be 48000)
        target_preset: Target curve name ('broadcast', 'podcast', 'streaming', 'flat')

    Returns:
        result: Tuple of (eq_settings, validation_result)
            - eq_settings: Dict with 'band_gains' and 'band_qs' (10-element lists)
            - validation_result: ValidationResult with passed flag

    Raises:
        ValueError: If validation fails (with generic user message)
    """
    from .spectrum import compute_voice_spectrum, smooth_spectrum_octave
    from .failure_detection import validate_analysis

    # Step 1: Compute spectrum
    freqs, spectrum_db = compute_voice_spectrum(audio_data, sample_rate)

    # Step 2: Apply smoothing
    spectrum_smoothed = smooth_spectrum_octave(freqs, spectrum_db, fraction=6)

    # Step 3: Get target curve
    target_db = get_target_curve(freqs, target_preset)

    # Step 4: Calculate optimal EQ bands using least-squares
    eq_settings = calculate_eq_bands(freqs, spectrum_smoothed, target_db)

    # Step 5: Validate results
    validation = validate_analysis(eq_settings, spectrum_smoothed, freqs)

    if not validation.passed:
        # Raise with generic user message (not technical details)
        raise ValueError(validation.reason)

    return eq_settings, validation
