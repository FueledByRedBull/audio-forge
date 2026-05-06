"""High-level Auto-EQ analysis pipeline."""

from .optimizer import calculate_eq_bands
from .target import get_target_curve

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
    from ..spectrum import compute_voice_spectrum, smooth_spectrum_octave
    from ..failure_detection import validate_analysis

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
