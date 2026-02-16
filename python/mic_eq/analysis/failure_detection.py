"""
Multi-criteria failure detection for auto-EQ analysis.

Combines multiple validation checks to detect invalid recordings that
would produce poor EQ results. Returns generic user-friendly error
messages without technical details.
"""
import os
import numpy as np
from dataclasses import dataclass

from mic_eq.config import (
    ANALYSIS_MIN_PEAK_COUNT,
    ANALYSIS_MIN_DYNAMIC_RANGE,
    ANALYSIS_MIN_SNR,
    ANALYSIS_MAX_SPECTRAL_FLATNESS
)
from .spectrum import find_octave_spaced_peaks

DEBUG = os.environ.get("AUDIOFORGE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str) -> None:
    if DEBUG:
        print(message)


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


def calculate_snr(freqs, spectrum_db, noise_floor_freq=200.0):
    """
    Estimate signal-to-noise ratio.

    SNR is estimated from robust spectral percentiles in the voice band.
    This avoids penalizing low-pitched voices where 80-200 Hz can contain
    legitimate speech fundamentals (not just noise).

    Args:
        freqs: Frequency array in Hz
        spectrum_db: Spectrum in dB
        noise_floor_freq: Frequency below which is considered noise (Hz)

    Returns:
        snr_db: Estimated SNR in dB
    """
    freqs = np.asarray(freqs, dtype=float)
    spectrum_db = np.asarray(spectrum_db, dtype=float)

    if freqs.shape != spectrum_db.shape or spectrum_db.size == 0:
        return 0.0

    # Focus on speech-relevant range.
    voice_mask = (freqs >= 80.0) & (freqs <= 8000.0)
    if np.any(voice_mask):
        spec = spectrum_db[voice_mask]
        f_voice = freqs[voice_mask]
    else:
        spec = spectrum_db
        f_voice = freqs

    # Mid-band (300-3400 Hz) carries most intelligibility.
    mid_mask = (f_voice >= 300.0) & (f_voice <= 3400.0)
    if np.any(mid_mask):
        signal_level_db = float(np.percentile(spec[mid_mask], 80))
    else:
        signal_level_db = float(np.percentile(spec, 80))

    # Floor estimate from lower percentile of whole voice band.
    noise_floor_db = float(np.percentile(spec, 20))

    return signal_level_db - noise_floor_db


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

    # Focus metrics on voice band to avoid out-of-band skew.
    voice_mask = (freqs >= 80.0) & (freqs <= 8000.0)
    if np.any(voice_mask):
        spectrum_voice = spectrum_db[voice_mask]
        freqs_voice = freqs[voice_mask]
    else:
        spectrum_voice = spectrum_db
        freqs_voice = freqs

    # Check 2: Dynamic range in voice band (robust percentile spread)
    dynamic_range = float(
        np.percentile(spectrum_voice, 95) - np.percentile(spectrum_voice, 5)
    )

    # Check 3: SNR (signal vs noise)
    snr_db = calculate_snr(freqs_voice, spectrum_voice)

    # Check 4: Spectral flatness (tonal vs noise)
    flatness = calculate_spectral_flatness(spectrum_voice)

    # Check 5: Excessive correction request suggests unreliable capture.
    band_gains = np.asarray(eq_settings.get("band_gains", []), dtype=float)
    clipped_gains = int(np.sum(np.abs(band_gains) >= 11.5)) if band_gains.size else 0
    gain_rms = float(np.sqrt(np.mean(np.square(band_gains)))) if band_gains.size else 0.0

    # Evaluate all criteria. Use tiered gating to reduce false rejections.
    hard_fail_reasons = []
    soft_failures = []

    if peak_count < max(2, ANALYSIS_MIN_PEAK_COUNT - 1):
        hard_fail_reasons.append(f"peak_count ({peak_count} too low)")

    if flatness > min(0.92, ANALYSIS_MAX_SPECTRAL_FLATNESS + 0.10):
        hard_fail_reasons.append(f"flatness ({flatness:.2f} too noise-like)")

    if clipped_gains >= 6:
        hard_fail_reasons.append(f"clipped_gains ({clipped_gains} >= 6)")
    if gain_rms > 10.0:
        hard_fail_reasons.append(f"gain_rms ({gain_rms:.1f} > 10.0 dB)")

    if peak_count < ANALYSIS_MIN_PEAK_COUNT:
        soft_failures.append(f"peak_count ({peak_count} < {ANALYSIS_MIN_PEAK_COUNT})")
    if dynamic_range < ANALYSIS_MIN_DYNAMIC_RANGE:
        soft_failures.append(
            f"dynamic_range ({dynamic_range:.1f} < {ANALYSIS_MIN_DYNAMIC_RANGE} dB)"
        )
    if snr_db < ANALYSIS_MIN_SNR:
        soft_failures.append(f"snr ({snr_db:.1f} < {ANALYSIS_MIN_SNR} dB)")
    if flatness > ANALYSIS_MAX_SPECTRAL_FLATNESS:
        soft_failures.append(f"flatness ({flatness:.2f} > {ANALYSIS_MAX_SPECTRAL_FLATNESS})")
    if clipped_gains >= 4:
        soft_failures.append(f"clipped_gains ({clipped_gains} >= 4)")
    if gain_rms > 8.0:
        soft_failures.append(f"gain_rms ({gain_rms:.1f} > 8.0 dB)")

    # Fail only on a hard criterion, or when multiple soft criteria fail together.
    failures = list(hard_fail_reasons)
    if not failures and len(soft_failures) >= 2:
        failures = soft_failures

    # DEBUG: Log validation results
    _debug_log(
        f"[VALIDATION] peak_count={peak_count}, dynamic_range={dynamic_range:.1f}dB, "
        f"snr={snr_db:.1f}dB, flatness={flatness:.2f}, clipped={clipped_gains}, "
        f"gain_rms={gain_rms:.1f}, soft_failures={len(soft_failures)}"
    )
    if failures:
        _debug_log(f"[VALIDATION] FAILED: {', '.join(failures)}")
    else:
        _debug_log("[VALIDATION] PASSED")

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
                'clipped_gains': clipped_gains,
                'gain_rms_db': gain_rms,
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
                'flatness': flatness,
                'clipped_gains': clipped_gains,
                'gain_rms_db': gain_rms,
            }
        )
