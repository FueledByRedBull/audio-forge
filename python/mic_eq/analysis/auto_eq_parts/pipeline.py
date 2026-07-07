"""High-level Auto-EQ analysis pipeline."""

from .optimizer import calculate_eq_bands
from .target import get_target_curve

def analyze_auto_eq(
    audio_data,
    sample_rate,
    target_preset='broadcast',
    *,
    target_mode="adaptive",
    smoothing_strength="conservative",
):
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
        target_mode: 'adaptive' for bounded voice-aware targets, 'static' for catalog targets
        smoothing_strength: 'conservative', 'balanced', 'broad', or 'off'

    Returns:
        result: Tuple of (eq_settings, validation_result)
            - eq_settings: Dict with 'band_gains' and 'band_qs' (10-element lists)
            - validation_result: ValidationResult with passed flag

    Raises:
        ValueError: If validation fails (with generic user message)
    """
    from ..spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual
    from ..failure_detection import validate_analysis

    # Step 1: Compute repeatability-aware voiced spectrum.
    spectrum_result = analyze_voice_spectrum(audio_data, sample_rate)
    freqs = spectrum_result.freqs
    spectrum_db = spectrum_result.median_spectrum_db

    # Step 2: Apply perceptual smoothing.
    spectrum_smoothed = smooth_spectrum_perceptual(
        freqs,
        spectrum_db,
        strength=smoothing_strength,
    )

    # Step 3: Get voice-aware bounded target curve.
    target_profile = (
        f"{target_preset}:{target_mode}"
        if not spectrum_result.used_single_spectrum_fallback
        else f"{target_preset}:{target_mode}:fallback"
    )
    target_db = get_target_curve(
        freqs,
        target_preset,
        measured_db=spectrum_smoothed,
        target_mode=target_mode,
    )

    # Step 4: Calculate optimal EQ bands using least-squares
    eq_settings = calculate_eq_bands(
        freqs,
        spectrum_smoothed,
        target_db,
        spectral_repeatability=spectrum_result.spectral_repeatability,
        voiced_window_ratio=spectrum_result.voiced_window_ratio,
        analysis_confidence=spectrum_result.residual_confidence,
        global_snr_db=spectrum_result.snr_db,
        target_profile=target_profile,
        used_spectrum_fallback=spectrum_result.used_single_spectrum_fallback,
        smoothing_strength=smoothing_strength,
    )
    eq_settings["target_mode"] = target_mode

    # Step 5: Validate results
    validation = validate_analysis(eq_settings, spectrum_smoothed, freqs)
    validation.details.update(
        {
            "voiced_window_ratio": spectrum_result.voiced_window_ratio,
            "spectrum_snr_db": spectrum_result.snr_db,
            "spectral_tilt_db_per_octave": spectrum_result.spectral_tilt_db_per_octave,
            "used_single_spectrum_fallback": spectrum_result.used_single_spectrum_fallback,
            "analysis_confidence": spectrum_result.residual_confidence,
            "capture_confidence": eq_settings.get("capture_confidence"),
            "eq_confidence": eq_settings.get("eq_confidence"),
            "validation_confidence": eq_settings.get("validation_confidence"),
            "validation_before_error_db": eq_settings.get("validation_before_error_db"),
            "validation_after_error_db": eq_settings.get("validation_after_error_db"),
            "validation_gain_scale": eq_settings.get("validation_gain_scale"),
            "target_mode": eq_settings.get("target_mode"),
            "smoothing_strength": eq_settings.get("smoothing_strength"),
            "residual_regularization": eq_settings.get("residual_regularization"),
        }
    )

    if not validation.passed:
        # Raise with generic user message (not technical details)
        raise ValueError(validation.reason)

    return eq_settings, validation
