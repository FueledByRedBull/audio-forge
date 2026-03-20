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
    center_freqs: list[float],
    weights: np.ndarray,
    q_prior: np.ndarray,
) -> np.ndarray:
    gains = params[:NUM_EQ_BANDS]
    qs = params[NUM_EQ_BANDS:]

    eq_response = _predict_eq_response(dense_freqs, gains, qs, center_freqs)
    error = target_dense_db - (measured_dense_db + eq_response)

    q_regularization = np.log(qs / q_prior)
    gain_ripple = np.diff(gains, n=2)

    return np.concatenate(
        [
            np.sqrt(weights) * error,
            np.sqrt(LAMBDA_Q) * q_regularization,
            np.sqrt(LAMBDA_G) * gain_ripple,
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

    # Use normalized measured spectrum for optimization
    measured_db = measured_db_normalized

    center_freqs = EQ_FREQUENCIES
    qs_stage1 = np.full(NUM_EQ_BANDS, AUTO_EQ_DEFAULT_Q, dtype=float)

    # Use a dense log-spaced frequency grid for optimization to reduce center-only artifacts.
    dense_freqs = _build_dense_log_grid(freqs)
    measured_dense_db = np.interp(dense_freqs, freqs, measured_db)
    target_dense_db = np.interp(dense_freqs, freqs, target_db)
    weights = _voice_weights(dense_freqs)

    measured_db_at_centers = np.interp(center_freqs, dense_freqs, measured_dense_db)
    target_db_at_centers = np.interp(center_freqs, dense_freqs, target_dense_db)
    desired_gains = target_db_at_centers - measured_db_at_centers
    gains_initial = np.clip(desired_gains, GAIN_MIN_DB, GAIN_MAX_DB)

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
        bounds=(GAIN_MIN_DB, GAIN_MAX_DB),
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
    q_prior = np.clip(np.full(NUM_EQ_BANDS, Q_PRIOR, dtype=float), q_low, q_high)
    params_initial = np.concatenate([gains_stage1, q_prior])
    params_lower = np.concatenate(
        [np.full(NUM_EQ_BANDS, GAIN_MIN_DB, dtype=float), q_low]
    )
    params_upper = np.concatenate(
        [np.full(NUM_EQ_BANDS, GAIN_MAX_DB, dtype=float), q_high]
    )
    stage2 = least_squares(
        _joint_gain_q_residuals,
        params_initial,
        args=(
            dense_freqs,
            measured_dense_db,
            target_dense_db,
            center_freqs,
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
    optimal_qs = stage2.x[NUM_EQ_BANDS:]

    _debug_log(
        f"[EQ_CALC] Stage1 gains: {[round(g, 2) for g in gains_stage1]}"
    )
    _debug_log(
        f"[EQ_CALC] Stage2 gains (raw): {[round(g, 2) for g in optimal_gains]}"
    )
    _debug_log(
        f"[EQ_CALC] Stage2 Qs: {[round(q, 3) for q in optimal_qs]}"
    )
    _debug_log(f"[EQ_CALC] Stage2 success: {stage2.success}")
    if hasattr(stage2, "message"):
        _debug_log(f"[EQ_CALC] Stage2 message: {stage2.message}")

    # Apply 70% correction factor (prevents over-compensation)
    optimal_gains = optimal_gains * 0.7

    # Clip to hardware limits
    optimal_gains = np.clip(optimal_gains, GAIN_MIN_DB, GAIN_MAX_DB)

    _debug_log(f"[EQ_CALC] Final gains (after 70% correction): {[round(g, 2) for g in optimal_gains]}")

    return {
        'band_gains': optimal_gains.tolist(),
        'band_qs': optimal_qs.tolist()
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
