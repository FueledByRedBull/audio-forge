"""
Multi-criteria failure detection for auto-EQ analysis.

Combines multiple validation checks to detect invalid recordings that
would produce poor EQ results. Returns generic user-friendly error
messages without technical details.
"""
import numpy as np
from dataclasses import dataclass

from mic_eq.config import (
    ANALYSIS_MIN_PEAK_COUNT,
    ANALYSIS_MIN_DYNAMIC_RANGE,
    ANALYSIS_MIN_SNR,
    ANALYSIS_MAX_SPECTRAL_FLATNESS
)
from .spectrum import find_octave_spaced_peaks


@dataclass
class ValidationResult:
    """Result of analysis validation."""
    passed: bool           # True if analysis is valid
    reason: str | None     # Error message if failed (generic, user-facing)
    details: dict         # Technical details for debugging (optional)


def calculate_spectral_flatness(spectrum_db):
    """
    Calculate spectral flatness (Wiener entropy).

    Flatness = geometric_mean / arithmetic_mean
    - 1.0 = white noise (all frequencies equal)
    - 0.0 = pure tone (single frequency)
    - Voice typically 0.3-0.6

    Args:
        spectrum_db: Spectrum in dB

    Returns:
        flatness: Spectral flatness (0.0 to 1.0)
    """
    # Convert dB to linear power
    linear = 10 ** (spectrum_db / 10)

    # Avoid log(0)
    linear = np.maximum(linear, 1e-12)

    # Geometric mean (exp of mean of log)
    geometric_mean = np.exp(np.mean(np.log(linear)))

    # Arithmetic mean
    arithmetic_mean = np.mean(linear)

    # Flatness ratio
    if arithmetic_mean < 1e-12:
        return 1.0  # Silent

    flatness = geometric_mean / arithmetic_mean
    return min(flatness, 1.0)  # Clip to [0, 1]


def calculate_snr(spectrum_db, noise_floor_freq=200):
    """
    Estimate signal-to-noise ratio.

    SNR = (mean spectrum above noise_floor_freq) - (min spectrum)

    Args:
        spectrum_db: Spectrum in dB
        noise_floor_freq: Frequency below which is considered noise (Hz)

    Returns:
        snr_db: Estimated SNR in dB
    """
    # Find noise floor (minimum in low frequencies)
    # This is a simple approximation
    noise_floor = np.min(spectrum_db)

    # Signal level (mean above noise floor frequency)
    signal_level = np.mean(spectrum_db[spectrum_db > noise_floor])

    snr_db = signal_level - noise_floor
    return snr_db


def validate_analysis(eq_settings, spectrum_db, freqs):
    """
    Multi-criteria validation of analysis results.

    Combines multiple checks to detect invalid recordings:
    - Peak count: Detect voice presence (need formant structure)
    - Dynamic range: Ensure sufficient variation (not silent/flat)
    - SNR: Ensure voice above noise floor
    - Spectral flatness: Ensure tonal (not white noise)

    Args:
        eq_settings: Calculated EQ settings (dict with band_gains)
        spectrum_db: Smoothed spectrum in dB
        freqs: Frequency array in Hz

    Returns:
        result: ValidationResult with passed flag and generic error message
    """
    # Check 1: Peak count (detect voice presence)
    peak_freqs, peak_values = find_octave_spaced_peaks(
        spectrum_db,
        freqs,
        octave_fraction=3
    )
    peak_count = len(peak_freqs)

    # Check 2: Dynamic range (peak - noise floor)
    dynamic_range = np.max(spectrum_db) - np.min(spectrum_db)

    # Check 3: SNR (signal vs noise)
    snr_db = calculate_snr(spectrum_db)

    # Check 4: Spectral flatness (tonal vs noise)
    flatness = calculate_spectral_flatness(spectrum_db)

    # Evaluate all criteria
    failures = []

    if peak_count < ANALYSIS_MIN_PEAK_COUNT:
        failures.append(f"peak_count ({peak_count} < {ANALYSIS_MIN_PEAK_COUNT})")

    if dynamic_range < ANALYSIS_MIN_DYNAMIC_RANGE:
        failures.append(f"dynamic_range ({dynamic_range:.1f} < {ANALYSIS_MIN_DYNAMIC_RANGE} dB)")

    if snr_db < ANALYSIS_MIN_SNR:
        failures.append(f"snr ({snr_db:.1f} < {ANALYSIS_MIN_SNR} dB)")

    if flatness > ANALYSIS_MAX_SPECTRAL_FLATNESS:
        failures.append(f"flatness ({flatness:.2f} > {ANALYSIS_MAX_SPECTRAL_FLATNESS})")

    # DEBUG: Log validation results
    print(f"[VALIDATION] peak_count={peak_count}, dynamic_range={dynamic_range:.1f}dB, snr={snr_db:.1f}dB, flatness={flatness:.2f}")
    if failures:
        print(f"[VALIDATION] FAILED: {', '.join(failures)}")
    else:
        print(f"[VALIDATION] PASSED")

    # Build result
    if failures:
        # Return GENERIC user-facing message (no technical details)
        return ValidationResult(
            passed=False,
            reason="Recording too unclear. Please try again.",
            details={
                'peak_count': peak_count,
                'dynamic_range_db': dynamic_range,
                'snr_db': snr_db,
                'flatness': flatness,
                'failures': failures
            }
        )
    else:
        return ValidationResult(
            passed=True,
            reason=None,
            details={
                'peak_count': peak_count,
                'dynamic_range_db': dynamic_range,
                'snr_db': snr_db,
                'flatness': flatness
            }
        )
