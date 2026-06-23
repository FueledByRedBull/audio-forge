"""
Audio spectrum analysis for Auto-EQ calibration.

Implements FFT analysis with Hamming window using scipy.signal.welch
for stable spectral estimation of voice recordings.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

VOICE_FRAME_RMS_GATE_DB = -48.0
VOICE_FRAME_FLOOR_PERCENTILE = 20.0
VOICE_FRAME_PEAK_PERCENTILE = 95.0
VOICE_FRAME_GATE_FRACTION = 0.60
VOICE_FRAME_MIN_SPREAD_DB = 6.0
MIN_VOICED_FRAME_RATIO = 0.15
MIN_VOICED_FRAMES = 3


@dataclass(frozen=True, slots=True)
class VoiceSpectrumResult:
    """Rich internal spectrum result used by Auto-EQ solving."""

    freqs: np.ndarray
    median_spectrum_db: np.ndarray
    window_spectra_db: np.ndarray
    voiced_window_ratio: float
    snr_db: float
    spectral_repeatability: np.ndarray
    spectral_tilt_db_per_octave: float
    residual_confidence: float
    used_single_spectrum_fallback: bool


def _select_voiced_samples(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(audio) < frame_size:
        return audio

    starts = np.arange(0, len(audio) - frame_size + 1, hop_size, dtype=int)
    if starts.size == 0:
        return audio

    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop_size]
    frame_power = np.mean(frames * frames, axis=1)
    frame_rms_db = 10.0 * np.log10(frame_power + 1e-12)

    floor_db = float(np.percentile(frame_rms_db, VOICE_FRAME_FLOOR_PERCENTILE))
    peak_db = float(np.percentile(frame_rms_db, VOICE_FRAME_PEAK_PERCENTILE))
    spread_db = peak_db - floor_db
    if spread_db < VOICE_FRAME_MIN_SPREAD_DB:
        return audio

    gate_db = max(
        VOICE_FRAME_RMS_GATE_DB,
        floor_db + VOICE_FRAME_GATE_FRACTION * spread_db,
    )
    voiced_mask = frame_rms_db >= gate_db
    voiced_count = int(np.sum(voiced_mask))

    if voiced_count < MIN_VOICED_FRAMES:
        return audio
    if voiced_count / starts.size < MIN_VOICED_FRAME_RATIO:
        return audio

    sample_mask = np.zeros(len(audio), dtype=bool)
    for start, keep in zip(starts, voiced_mask):
        if keep:
            sample_mask[start:start + frame_size] = True

    voiced = audio[sample_mask]
    if len(voiced) < frame_size:
        return audio
    return voiced


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

    audio = np.asarray(audio, dtype=float)
    hop = max(1, nperseg // 2)
    voiced_audio = _select_voiced_samples(audio, nperseg, hop)
    if len(voiced_audio) >= nperseg:
        audio_for_fft = voiced_audio
    else:
        audio_for_fft = audio

    # Hamming window for voice analysis
    # Optimal trade-off between frequency resolution and sidelobe suppression
    # Welch's method for stable spectral estimate
    # Averages multiple FFTs with 50% overlap to reduce variance
    freqs, psd = signal.welch(
        audio_for_fft,
        fs=fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2  # 50% overlap
    )

    # Convert power spectral density to dB
    # Add small noise floor to avoid log(0) = -infinity
    spectrum_db = 10 * np.log10(psd + 1e-12)

    return freqs, spectrum_db


def _frame_rms_db(frames: np.ndarray) -> np.ndarray:
    frame_power = np.mean(frames * frames, axis=1)
    return 10.0 * np.log10(frame_power + 1e-12)


def _voiced_frame_mask(frame_rms_db: np.ndarray) -> np.ndarray:
    floor_db = float(np.percentile(frame_rms_db, VOICE_FRAME_FLOOR_PERCENTILE))
    peak_db = float(np.percentile(frame_rms_db, VOICE_FRAME_PEAK_PERCENTILE))
    spread_db = peak_db - floor_db
    if spread_db < VOICE_FRAME_MIN_SPREAD_DB:
        return np.ones_like(frame_rms_db, dtype=bool)
    gate_db = max(
        VOICE_FRAME_RMS_GATE_DB,
        floor_db + VOICE_FRAME_GATE_FRACTION * spread_db,
    )
    return frame_rms_db >= gate_db


def _window_spectrum_db(frame: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hamming(frame.size)
    windowed = frame * window
    psd = np.square(np.abs(np.fft.rfft(windowed))) / max(float(np.sum(window * window)), 1e-12)
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / fs)
    return freqs, 10.0 * np.log10(psd + 1e-12)


def _estimate_snr_from_spectrum(freqs: np.ndarray, spectrum_db: np.ndarray) -> float:
    voice_mask = (freqs >= 80.0) & (freqs <= 8000.0)
    spec = spectrum_db[voice_mask] if np.any(voice_mask) else spectrum_db
    if spec.size == 0:
        return 0.0
    signal_db = float(np.percentile(spec, 80.0))
    floor_db = float(np.percentile(spec, 20.0))
    return signal_db - floor_db


def _estimate_tilt_db_per_octave(freqs: np.ndarray, spectrum_db: np.ndarray) -> float:
    mask = (freqs >= 100.0) & (freqs <= 8000.0)
    if np.count_nonzero(mask) < 2:
        return 0.0
    x = np.log2(freqs[mask])
    y = spectrum_db[mask]
    x_center = x - float(np.mean(x))
    denom = float(np.sum(x_center * x_center))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_center, y - float(np.mean(y))) / denom)


def _shape_repeatability(
    freqs: np.ndarray,
    spectra_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = np.asarray(
        [smooth_spectrum_perceptual(freqs, spectrum) for spectrum in spectra_db],
        dtype=float,
    )
    voice_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    if np.any(voice_mask):
        per_window_level = np.mean(smoothed[:, voice_mask], axis=1, keepdims=True)
    else:
        per_window_level = np.mean(smoothed, axis=1, keepdims=True)

    normalized = smoothed - per_window_level
    std_db = np.std(normalized, axis=0)
    repeatability = np.clip(1.0 - std_db / 8.0, 0.0, 1.0)

    if np.any(voice_mask):
        voice_repeatability = float(np.median(repeatability[voice_mask]))
        repeatability[voice_mask] = 0.70 * repeatability[voice_mask] + 0.30 * voice_repeatability
    return np.clip(repeatability, 0.0, 1.0), smoothed


def analyze_voice_spectrum(audio, fs=48000, nperseg=4096) -> VoiceSpectrumResult:
    """Analyze voiced windows and return a repeatability-aware spectrum."""
    if len(audio) < nperseg:
        raise ValueError(
            f"Audio too short for FFT: need {nperseg} samples, "
            f"got {len(audio)} ({len(audio)/fs:.2f} seconds)"
        )

    audio_arr = np.asarray(audio, dtype=float)
    hop = max(1, nperseg // 2)
    frames = np.lib.stride_tricks.sliding_window_view(audio_arr, nperseg)[::hop]
    frame_rms = _frame_rms_db(frames)
    voiced_mask = _voiced_frame_mask(frame_rms)
    voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
    voiced_frames = frames[voiced_mask]

    if voiced_frames.shape[0] < MIN_VOICED_FRAMES or voiced_ratio < MIN_VOICED_FRAME_RATIO:
        freqs, spectrum_db = compute_voice_spectrum(audio_arr, fs, nperseg)
        repeatability = np.full_like(freqs, 0.45, dtype=float)
        return VoiceSpectrumResult(
            freqs=freqs,
            median_spectrum_db=spectrum_db,
            window_spectra_db=np.asarray([spectrum_db], dtype=float),
            voiced_window_ratio=max(voiced_ratio, 1.0 / max(1, frames.shape[0])),
            snr_db=_estimate_snr_from_spectrum(freqs, spectrum_db),
            spectral_repeatability=repeatability,
            spectral_tilt_db_per_octave=_estimate_tilt_db_per_octave(freqs, spectrum_db),
            residual_confidence=0.45,
            used_single_spectrum_fallback=True,
        )

    spectra = []
    freqs = None
    for frame in voiced_frames:
        local_freqs, spectrum_db = _window_spectrum_db(frame, fs)
        if freqs is None:
            freqs = local_freqs
        spectra.append(spectrum_db)

    assert freqs is not None
    spectra_arr = np.asarray(spectra, dtype=float)
    repeatability, smoothed_spectra = _shape_repeatability(freqs, spectra_arr)
    median_spectrum = np.median(smoothed_spectra, axis=0)
    snr_db = _estimate_snr_from_spectrum(freqs, median_spectrum)
    snr_confidence = np.clip((snr_db - 3.0) / 18.0, 0.0, 1.0)
    voice_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    repeatability_score = (
        float(np.median(repeatability[voice_mask]))
        if np.any(voice_mask)
        else float(np.median(repeatability))
    )
    coverage = float(np.clip(voiced_ratio / 0.55, 0.0, 1.0))
    residual_confidence = float(
        np.clip(
            0.50 * repeatability_score
            + 0.30 * coverage
            + 0.20 * snr_confidence,
            0.0,
            1.0,
        )
    )

    return VoiceSpectrumResult(
        freqs=freqs,
        median_spectrum_db=median_spectrum,
        window_spectra_db=spectra_arr,
        voiced_window_ratio=voiced_ratio,
        snr_db=snr_db,
        spectral_repeatability=repeatability,
        spectral_tilt_db_per_octave=_estimate_tilt_db_per_octave(freqs, median_spectrum),
        residual_confidence=residual_confidence,
        used_single_spectrum_fallback=False,
    )


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


def smooth_spectrum_perceptual(freqs, spectrum_db):
    """Apply voice-aware smoothing that varies by frequency region."""
    freqs = np.asarray(freqs, dtype=float)
    spectrum_db = np.asarray(spectrum_db, dtype=float)
    wide = smooth_spectrum_octave(freqs, spectrum_db, fraction=3)
    medium = smooth_spectrum_octave(freqs, spectrum_db, fraction=6)
    fine = smooth_spectrum_octave(freqs, spectrum_db, fraction=12)

    smoothed = medium.copy()
    low_mask = freqs < 180.0
    mid_mask = (freqs >= 180.0) & (freqs < 3500.0)
    sibilance_mask = (freqs >= 3500.0) & (freqs <= 9000.0)
    high_mask = freqs > 9000.0
    smoothed[low_mask] = wide[low_mask]
    smoothed[mid_mask] = medium[mid_mask]
    smoothed[sibilance_mask] = fine[sibilance_mask]
    smoothed[high_mask] = wide[high_mask]
    return smoothed


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
    if len(spectrum_db) != len(freqs):
        raise ValueError("spectrum_db and freqs must have the same length")

    # Remove DC bin (can't take log of 0)
    valid = freqs > 0
    if np.count_nonzero(valid) < 2:
        return np.array([]), np.array([])

    log_freqs = np.log2(freqs[valid])
    spectrum_valid = spectrum_db[valid]
    total_octaves = log_freqs.max() - log_freqs.min()
    if not np.isfinite(total_octaves) or total_octaves <= 0:
        return np.array([]), np.array([])

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

    bins_per_octave = len(log_freq_uniform) / total_octaves

    # Distance for 1/N octave spacing
    # Example: octave_fraction=3 -> minimum 1/3 octave between peaks
    min_distance = max(1, int(bins_per_octave / octave_fraction))

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
