"""
Auto-EQ calculation using least-squares curve fitting.

Accounts for parametric EQ band interaction by optimizing gains
to minimize error between target curve and predicted EQ response.
"""
import numpy as np
from scipy.optimize import least_squares

from mic_eq.config import (
    TARGET_CURVES,
    EQ_FREQUENCIES,
    AUTO_EQ_DEFAULT_Q
)


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
    response_db = np.zeros_like(freqs, dtype=float)

    for gain_db, q, fc in zip(gains, qs, center_freqs):
        # Convert to linear amplitude
        gain_linear = 10 ** (gain_db / 20.0)

        # Calculate bandwidth from Q
        # BW = fc / Q for parametric EQ
        bandwidth = fc / q

        # Calculate frequency response of this peaking band
        # Using standard parametric EQ transfer function
        # H(f) = 1 + (G * j * f / BW) / (1 + j * f / BW) where G = gain_linear - 1
        # Simplified: magnitude response of peaking EQ

        f_ratio = freqs / fc
        # Peaking EQ magnitude response (simplified)
        # At fc: response = gain_linear
        # At fc * Q or fc / Q: response = sqrt(gain_linear)
        denom = 1 + (q * (f_ratio - 1/f_ratio)) ** 2
        band_response = gain_linear / np.sqrt(denom)

        # Convert back to dB and accumulate
        # Combined response in dB = sum of individual responses in linear domain
        response_linear = 10 ** (response_db / 20.0)
        response_linear *= band_response
        response_db = 20 * np.log10(np.maximum(response_linear, 1e-12))

    return response_db


def _eq_error_function(gains, freqs, measured_db, target_db, qs, center_freqs):
    """
    Error function for least-squares optimization.

    Calculates the error between target curve and (measured + predicted EQ).

    Args:
        gains: Array of 10 gain values (optimization variable)
        freqs: Frequency array (Hz)
        measured_db: Measured spectrum (dB)
        target_db: Target curve (dB)
        qs: List of 10 Q values (fixed during optimization)
        center_freqs: List of 10 center frequencies (Hz)

    Returns:
        error: Difference between target and (measured + EQ response)
    """
    # Predict EQ response with current gains
    eq_response = _predict_eq_response(freqs, gains, qs, center_freqs)

    # Calculate combined response (measured + EQ)
    combined = measured_db + eq_response

    # Error = target - combined
    # We want combined to match target, so minimize (target - combined)^2
    error = target_db - combined

    return error


def calculate_eq_bands(freqs, measured_db, target_db):
    """
    Calculate optimal 10-band EQ settings using least-squares optimization.

    Finds gains that minimize error between target curve and
    (measured spectrum + EQ response). Accounts for band interaction.

    Args:
        freqs: Frequency array (Hz)
        measured_db: Measured spectrum in dB
        target_db: Target curve in dB

    Returns:
        eq_settings: Dict with 'band_gains' and 'band_qs' (10-element lists)
    """
    # Fixed parameters
    qs = [AUTO_EQ_DEFAULT_Q] * 10
    center_freqs = EQ_FREQUENCIES

    # Initial guess: flat response (all gains = 0)
    gains_initial = np.zeros(10)

    # Gain bounds: -12 to +12 dB (EQ hardware limits)
    bounds = (-12.0, 12.0)

    # Optimize gains to minimize error
    result = least_squares(
        _eq_error_function,
        gains_initial,
        args=(freqs, measured_db, target_db, qs, center_freqs),
        bounds=bounds,
        method='trf',  # Trust Region Reflective (good for bounded problems)
        ftol=1e-6,     # Function tolerance
        xtol=1e-6,     # Parameter tolerance
        verbose=0
    )

    # Extract optimal gains
    optimal_gains = result.x

    # Apply 70% correction factor (prevents over-compensation)
    optimal_gains = optimal_gains * 0.7

    # Clip to hardware limits
    optimal_gains = np.clip(optimal_gains, -12.0, 12.0)

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
