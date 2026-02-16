"""
Auto-EQ calculation using least-squares curve fitting.

Accounts for parametric EQ band interaction by optimizing gains
to minimize error between target curve and predicted EQ response.
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


def _debug_log(message: str) -> None:
    if DEBUG:
        print(message)


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


def _eq_error_function(gains, center_freqs, measured_db_at_centers, target_db, qs, all_freqs):
    """
    Error function for least-squares optimization.

    Calculates the error between target curve and (measured + predicted EQ)
    at the EQ band center frequencies only (10 points).

    Args:
        gains: Array of 10 gain values (optimization variable)
        center_freqs: Array of 10 center frequencies
        measured_db_at_centers: Measured spectrum sampled at center frequencies (10 values)
        target_db: Target curve at center frequencies (10 values)
        qs: List of 10 Q values (fixed during optimization)
        all_freqs: Full frequency array (for _predict_eq_response)

    Returns:
        error: Difference between target and (measured + EQ response) at center frequencies
    """
    # Debug: print first call to see what's happening
    if DEBUG and not hasattr(_eq_error_function, '_called'):
        _eq_error_function._called = True
        _debug_log(f"[EQ_ERROR] First call with gains: {gains}")
        _debug_log(f"[EQ_ERROR] measured_db_at_centers: {measured_db_at_centers}")
        _debug_log(f"[EQ_ERROR] target_db: {target_db}")

    # Predict EQ response with current gains (evaluated at all frequencies)
    eq_response = _predict_eq_response(all_freqs, gains, qs, center_freqs)

    # Sample EQ response at the center frequencies
    # Find indices of center frequencies in the frequency array
    eq_response_at_centers = np.interp(center_freqs, all_freqs, eq_response)

    # Calculate combined response (measured + EQ) at center frequencies
    combined_at_centers = measured_db_at_centers + eq_response_at_centers

    # Error = target - combined (at center frequencies only)
    error = target_db - combined_at_centers

    if DEBUG and not hasattr(_eq_error_function, '_printed_error'):
        # Only print once for the initial call (gains = 0)
        if np.allclose(gains, 0):
            _eq_error_function._printed_error = True
            _debug_log(f"[EQ_ERROR] eq_response_at_centers (gains=0): {eq_response_at_centers}")
            _debug_log(f"[EQ_ERROR] combined_at_centers (gains=0): {combined_at_centers}")
            _debug_log(f"[EQ_ERROR] error (gains=0): {error}")
            _debug_log(f"[EQ_ERROR] error magnitude: {np.linalg.norm(error):.2f}")

    return error


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

    # Fixed parameters
    qs = [AUTO_EQ_DEFAULT_Q] * 10
    center_freqs = EQ_FREQUENCIES

    # Sample the measured spectrum at the EQ band center frequencies
    measured_db_at_centers = np.interp(center_freqs, freqs, measured_db)

    # Sample the target curve at the EQ band center frequencies
    target_db_at_centers = np.interp(center_freqs, freqs, target_db)

    _debug_log(f"[EQ_CALC] Measured at centers: {[round(v, 1) for v in measured_db_at_centers]} dB")
    _debug_log(f"[EQ_CALC] Target at centers: {[round(v, 1) for v in target_db_at_centers]} dB")

    # Calculate the desired gain adjustment (target - measured) at each center frequency
    # This is what the optimizer should aim for
    desired_gains = target_db_at_centers - measured_db_at_centers
    _debug_log(f"[EQ_CALC] Desired gains (target - measured): {[round(v, 1) for v in desired_gains]} dB")

    # Initial guess: use desired gains clipped to bounds (better starting point than zeros)
    gains_initial = np.clip(desired_gains, -12.0, 12.0)
    _debug_log(f"[EQ_CALC] Initial guess (clipped desired): {[round(g, 2) for g in gains_initial]} dB")

    # Gain bounds: -12 to +12 dB (EQ hardware limits)
    bounds = (-12.0, 12.0)

    # Optimize gains to minimize error
    # Note: We only evaluate at the 10 center frequencies, not the full spectrum
    # Verbose output: 2 for debugging, 0 for production
    verbose_level = 2 if DEBUG else 0

    result = least_squares(
        _eq_error_function,
        gains_initial,
        args=(center_freqs, measured_db_at_centers, target_db_at_centers, qs, freqs),
        bounds=bounds,
        method='trf',  # Trust Region Reflective (good for bounded problems)
        ftol=1e-4,     # Function tolerance (looser than before)
        xtol=1e-4,     # Parameter tolerance (looser than before)
        gtol=1e-6,     # Gradient tolerance (explicitly set)
        max_nfev=100,  # Max function evaluations (prevent infinite loops)
        verbose=verbose_level
    )

    # Extract optimal gains
    optimal_gains = result.x

    # DEBUG: Log what the optimizer found
    _debug_log(f"[EQ_CALC] Raw optimized gains (before 70%): {[round(g, 2) for g in optimal_gains]}")
    _debug_log(f"[EQ_CALC] Optimization success: {result.success}")
    if hasattr(result, 'message'):
        _debug_log(f"[EQ_CALC] Optimizer message: {result.message}")

    # Apply 70% correction factor (prevents over-compensation)
    optimal_gains = optimal_gains * 0.7

    # Clip to hardware limits
    optimal_gains = np.clip(optimal_gains, -12.0, 12.0)

    _debug_log(f"[EQ_CALC] Final gains (after 70% correction): {[round(g, 2) for g in optimal_gains]}")

    return {
        'band_gains': optimal_gains.tolist(),
        'band_qs': qs
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
