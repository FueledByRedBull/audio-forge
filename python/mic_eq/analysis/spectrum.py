"""
Audio spectrum analysis for Auto-EQ calibration.

Implements FFT analysis with Hamming window using scipy.signal.welch
for stable spectral estimation of voice recordings.
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks


def compute_voice_spectrum(audio, fs=48000, nperseg=4096):
    """
    Compute voice spectrum with optimal Hamming windowing.

    Uses Welch's method with Hamming window for stable spectral estimation.
    Optimized for voice analysis with 4096-sample FFT (85ms at 48kHz).

    Args:
        audio: Input audio samples (float32 NumPy array)
        fs: Sample rate in Hz (default: 48000)
        nperseg: FFT segment size (default: 4096)

    Returns:
        freqs: Frequency array in Hz (same length as spectrum_db)
        spectrum_db: Power spectrum in dB (relative, full-scale = 0 dB)

    Example:
        >>> audio = np.random.randn(48000)  # 1 second at 48kHz
        >>> freqs, spectrum_db = compute_voice_spectrum(audio)
        >>> print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        >>> print(f"Spectrum range: {spectrum_db.min():.1f} to {spectrum_db.max():.1f} dB")
    """
    # Validate input
    if len(audio) < nperseg:
        raise ValueError(
            f"Audio too short for FFT: need {nperseg} samples, "
            f"got {len(audio)} ({len(audio)/fs:.2f} seconds)"
        )

    # Hamming window for voice analysis
    # Optimal trade-off between frequency resolution and sidelobe suppression
    window = np.hamming(nperseg)

    # Welch's method for stable spectral estimate
    # Averages multiple FFTs with 50% overlap to reduce variance
    freqs, psd = signal.welch(
        audio,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2  # 50% overlap
    )

    # Convert power spectral density to dB
    # Add small noise floor to avoid log(0) = -infinity
    spectrum_db = 10 * np.log10(psd + 1e-12)

    return freqs, spectrum_db


def get_octave_frequencies(fraction=6, limits=(20, 20000), ref_freq=1000.0):
    """
    Calculate IEC 61260-1 compliant center frequencies.

    Implements international standard for fractional octave band filters.
    Uses base-10 octave ratio (G = 10^0.3) per IEC specification.

    Args:
        fraction: Octave fraction (6 = 1/6 octave, 3 = 1/3 octave)
        limits: (min_freq, max_freq) tuple in Hz
        ref_freq: Reference frequency in Hz (1000 Hz per IEC standard)

    Returns:
        f_center: Center frequencies (NumPy array)
        f_lower: Lower band edges (NumPy array)
        f_upper: Upper band edges (NumPy array)

    Reference:
        IEC 61260-1:2019 - Electroacoustics - Octave-band and
        fractional-octave-band filters
    """
    # IEC standard octave ratio (base-10)
    # G = 10^(0.3) ≈ 1.9953
    G = 10 ** 0.3
    b = fraction

    # Calculate band indices
    # x is the band index that centers on ref_freq when x=0
    x_min = int(np.floor(b * np.log10(limits[0] / ref_freq) / np.log10(G)))
    x_max = int(np.ceil(b * np.log10(limits[1] / ref_freq) / np.log10(G)))

    f_center, f_lower, f_upper = [], [], []

    for x in range(x_min, x_max + 1):
        # Odd vs even fraction handling per IEC standard
        # Formula differs slightly for odd vs even fractions
        if b % 2 == 1:
            # Odd fraction: simple power law
            fm = ref_freq * (G ** (x / b))
        else:
            # Even fraction: offset by half step
            fm = ref_freq * (G ** ((2 * x + 1) / (2 * b)))

        # Check if within limits
        if limits[0] <= fm <= limits[1]:
            f_center.append(fm)
            # Calculate band edges (±1/2 band from center)
            f_lower.append(fm / G ** (1 / (2 * b)))
            f_upper.append(fm * G ** (1 / (2 * b)))

    return np.array(f_center), np.array(f_lower), np.array(f_upper)


def smooth_spectrum_octave(freqs, spectrum_db, fraction=6):
    """
    Apply fractional octave smoothing to spectrum.

    Reduces spectral variance while preserving formant detail by
    averaging energy within each fractional octave band. Uses
    IEC 61260-1 compliant band calculation.

    Args:
        freqs: Frequency array from FFT (linear spacing, Hz)
        spectrum_db: Spectrum in dB (same length as freqs)
        fraction: Octave fraction (6 = 1/6 octave, 3 = 1/3 octave)

    Returns:
        smoothed_db: Smoothed spectrum at original frequency resolution

    Note:
        Averages ENERGY in linear domain (NOT arithmetic mean in dB).
        This is critical for correct power averaging.
    """
    # Get octave band frequencies
    f_center, f_lower, f_upper = get_octave_frequencies(fraction)

    smoothed_bands = []
    for fc, fl, fu in zip(f_center, f_lower, f_upper):
        # Find FFT bins within this band
        mask = (freqs >= fl) & (freqs <= fu)
        if np.any(mask):
            # CRITICAL: Energy averaging in LINEAR domain
            # Convert dB to power (10^(dB/10)), average, convert back
            linear_power = 10 ** (spectrum_db[mask] / 10)
            avg_power = np.mean(linear_power)
            smoothed_bands.append(10 * np.log10(avg_power))
        else:
            # No bins in this band (shouldn't happen with proper limits)
            smoothed_bands.append(np.nan)

    # Interpolate back to original frequency resolution
    # This preserves the FFT frequency grid for downstream processing
    valid = ~np.isnan(smoothed_bands)
    if np.sum(valid) > 1:
        # Use nearest value extrapolation for frequencies outside band limits
        smoothed_db = np.interp(
            freqs,
            f_center[valid],
            np.array(smoothed_bands)[valid],
            left=np.array(smoothed_bands)[valid][0],  # Extrapolate with lowest band value
            right=np.array(smoothed_bands)[valid][-1]  # Extrapolate with highest band value
        )
    else:
        # Fallback: return original spectrum if smoothing failed
        smoothed_db = spectrum_db.copy()

    return smoothed_db


def find_octave_spaced_peaks(spectrum_db, freqs, octave_fraction=3):
    """
    Find peaks with TRUE octave spacing using log-frequency transform.

    CRITICAL: Must transform to log-frequency domain first!
    The naive approach (distance=len(freqs)//15) is mathematically incorrect
    because FFT bins are linearly spaced, not logarithmically.

    This implementation:
    1. Transforms to log2(frequency) domain
    2. Resamples to uniform log-frequency grid
    3. Applies find_peaks with constant distance
    4. Maps back to linear frequency

    Args:
        spectrum_db: Spectrum in dB (from compute_voice_spectrum)
        freqs: Frequency array in Hz (linear spacing)
        octave_fraction: Minimum spacing (3 = 1/3 octave, 6 = 1/6 octave)

    Returns:
        peak_freqs: Frequencies of detected peaks (Hz, linear scale)
        peak_values: dB values at peak frequencies

    Example:
        >>> freqs, spectrum_db = compute_voice_spectrum(audio, 48000)
        >>> peaks_freqs, peaks_db = find_octave_spaced_peaks(spectrum_db, freqs)
        >>> print(f"Found {len(peaks_freqs)} peaks")
    """
    # Remove DC bin (can't take log of 0)
    valid = freqs > 0
    log_freqs = np.log2(freqs[valid])
    spectrum_valid = spectrum_db[valid]

    # Resample to UNIFORM log-frequency grid
    # This is critical: constant distance in log-freq = constant octave fraction
    log_freq_uniform = np.linspace(
        log_freqs.min(),
        log_freqs.max(),
        len(log_freqs)
    )
    spectrum_resampled = np.interp(
        log_freq_uniform,
        log_freqs,
        spectrum_valid
    )

    # Calculate total octaves in range
    total_octaves = log_freqs.max() - log_freqs.min()
    bins_per_octave = len(log_freq_uniform) / total_octaves

    # Distance for 1/N octave spacing
    # Example: octave_fraction=3 -> minimum 1/3 octave between peaks
    min_distance = int(bins_per_octave / octave_fraction)

    # Find peaks in log-frequency domain
    peaks, properties = find_peaks(
        spectrum_resampled,
        distance=min_distance,
        prominence=3.0  # 3 dB minimum prominence (avoid noise)
    )

    # Map back to linear frequency
    peak_freqs = 2 ** log_freq_uniform[peaks]
    peak_values = spectrum_resampled[peaks]

    # Filter to voice range (80 Hz - 16 kHz)
    voice_mask = (peak_freqs >= 80) & (peak_freqs <= 16000)
    peak_freqs = peak_freqs[voice_mask]
    peak_values = peak_values[voice_mask]

    return peak_freqs, peak_values
