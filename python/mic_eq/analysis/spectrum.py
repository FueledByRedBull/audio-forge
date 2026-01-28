"""
Audio spectrum analysis for Auto-EQ calibration.

Implements FFT analysis with Hamming window using scipy.signal.welch
for stable spectral estimation of voice recordings.
"""
import numpy as np
from scipy import signal


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
