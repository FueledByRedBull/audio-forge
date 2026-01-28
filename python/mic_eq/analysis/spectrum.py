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
