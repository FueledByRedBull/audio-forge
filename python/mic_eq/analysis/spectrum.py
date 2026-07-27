"""Audio spectrum analysis for Auto-EQ and Voice Setup.

Welch/Hamming remains the production estimator. A DPSS multi-taper,
multi-resolution implementation is retained as an explicit fixture experiment;
policy changes only if every perceptual band improves stability by at least the
materiality threshold.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal.windows import dpss

VOICE_FRAME_RMS_GATE_DB = -48.0
VOICE_FRAME_FLOOR_PERCENTILE = 20.0
VOICE_FRAME_PEAK_PERCENTILE = 95.0
VOICE_FRAME_GATE_FRACTION = 0.60
VOICE_FRAME_MIN_SPREAD_DB = 6.0
MIN_VOICED_FRAME_RATIO = 0.15
MIN_VOICED_FRAMES = 3
SILERO_WINDOW_SAMPLES = 512
SILERO_SAMPLE_RATE = 16_000
MULTIRES_MATERIAL_IMPROVEMENT_DB = 0.75
SPECTRUM_ESTIMATOR_POLICY = "welch_hamming"


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
    measurement_coverage: float = 1.0
    outlier_rejection_ratio: float = 0.0
    vad_probability_used: bool = False
    vad_active_window_ratio: float = 0.0
    spectral_snr_db: np.ndarray | None = None
    noise_spectrum_db: np.ndarray | None = None
    noise_reference_source: str = "unavailable"


@dataclass(frozen=True, slots=True)
class SpectrumEstimatorEvaluation:
    """Stability comparison for the optional multi-resolution experiment."""

    current_band_stability_db: dict[str, float]
    multires_band_stability_db: dict[str, float]
    improvement_db: dict[str, float]
    material_improvement: bool
    selected_estimator: str


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
        noverlap=nperseg // 2,  # 50% overlap
        detrend="constant",
    )

    # Convert power spectral density to dB
    # Add small noise floor to avoid log(0) = -infinity
    spectrum_db = 10 * np.log10(psd + 1e-12)

    return freqs, spectrum_db


def _frame_rms_db(frames: np.ndarray) -> np.ndarray:
    frame_power = np.mean(frames * frames, axis=1)
    return 10.0 * np.log10(frame_power + 1e-12)


def _interpolate_vad_probabilities(
    vad_probabilities: np.ndarray | None,
    frame_starts: np.ndarray,
    frame_size: int,
    sample_rate: int,
) -> np.ndarray | None:
    """Map 32 ms Silero posteriors onto arbitrary analysis-frame centres."""
    if vad_probabilities is None:
        return None
    probabilities = np.asarray(vad_probabilities, dtype=float).reshape(-1)
    if probabilities.size == 0 or frame_starts.size == 0 or sample_rate <= 0:
        return None

    vad_window_samples = max(
        1,
        int(np.ceil(sample_rate * SILERO_WINDOW_SAMPLES / SILERO_SAMPLE_RATE)),
    )
    analysis_centres = frame_starts.astype(float) + frame_size * 0.5
    vad_centres = (np.arange(probabilities.size, dtype=float) + 0.5) * vad_window_samples
    return np.interp(
        analysis_centres,
        vad_centres,
        np.clip(probabilities, 0.0, 1.0),
        left=float(np.clip(probabilities[0], 0.0, 1.0)),
        right=float(np.clip(probabilities[-1], 0.0, 1.0)),
    )


def _voiced_frame_mask(
    frame_rms_db: np.ndarray,
    *,
    vad_probabilities: np.ndarray | None = None,
    frame_starts: np.ndarray | None = None,
    frame_size: int | None = None,
    sample_rate: int = 48_000,
) -> np.ndarray:
    floor_db = float(np.percentile(frame_rms_db, VOICE_FRAME_FLOOR_PERCENTILE))
    peak_db = float(np.percentile(frame_rms_db, VOICE_FRAME_PEAK_PERCENTILE))
    spread_db = peak_db - floor_db
    gate_db = max(
        VOICE_FRAME_RMS_GATE_DB,
        floor_db
        + VOICE_FRAME_GATE_FRACTION
        * max(spread_db, VOICE_FRAME_MIN_SPREAD_DB),
    )
    energy_mask = (
        np.ones_like(frame_rms_db, dtype=bool)
        if spread_db < VOICE_FRAME_MIN_SPREAD_DB
        else frame_rms_db >= gate_db
    )
    if vad_probabilities is None or frame_starts is None or frame_size is None:
        return energy_mask

    posterior = _interpolate_vad_probabilities(
        vad_probabilities,
        frame_starts,
        frame_size,
        sample_rate,
    )
    if posterior is None or posterior.shape != frame_rms_db.shape:
        return energy_mask

    # Use the posterior as speech evidence, but retain an energy floor to stop
    # a neural false positive from turning a noise-only window into an EQ
    # measurement. Strong posterior evidence can admit quiet speech below the
    # ordinary energy gate.
    supported_energy = frame_rms_db >= max(
        VOICE_FRAME_RMS_GATE_DB,
        floor_db + 0.25 * max(spread_db, VOICE_FRAME_MIN_SPREAD_DB),
    )
    posterior_mask = (posterior >= 0.35) & supported_energy
    strong_posterior_mask = posterior >= 0.65
    combined = posterior_mask | strong_posterior_mask
    if int(np.count_nonzero(combined)) >= MIN_VOICED_FRAMES:
        return combined
    return energy_mask


def _robust_median_spectrum(
    freqs: np.ndarray,
    spectra_db: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Aggregate window spectra after rejecting shape outliers.

    Level differences between phrases should not make a loud phrase dominate
    the microphone-shape estimate. We therefore normalize each window over
    the voice band, compute a median shape, reject windows whose RMS shape
    error is a robust-MAD outlier, then restore the median
    level of the retained windows.
    """
    if spectra_db.shape[0] < 3:
        return np.median(spectra_db, axis=0), 1.0

    voice_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    if not np.any(voice_mask):
        voice_mask = np.ones(freqs.shape, dtype=bool)
    levels = np.median(spectra_db[:, voice_mask], axis=1)
    normalized = spectra_db - levels[:, np.newaxis]
    centre = np.median(normalized, axis=0)
    shape_error = normalized[:, voice_mask] - centre[voice_mask]
    # A frequency-wise median would hide a narrow but severe resonance because
    # most bins remain unchanged. RMS error preserves that diagnostic signal
    # while the later MAD cutoff remains robust across windows.
    distances = np.sqrt(np.mean(np.square(shape_error), axis=1))
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    cutoff = median_distance + 4.0 * max(mad, 0.25)
    inliers = distances <= cutoff

    minimum_inliers = max(3, int(np.ceil(spectra_db.shape[0] * 0.50)))
    if int(np.count_nonzero(inliers)) < minimum_inliers:
        closest = np.argsort(distances)[:minimum_inliers]
        inliers = np.zeros(spectra_db.shape[0], dtype=bool)
        inliers[closest] = True

    robust_shape = np.median(normalized[inliers], axis=0)
    robust_level = float(np.median(levels[inliers]))
    coverage = float(np.count_nonzero(inliers) / max(1, spectra_db.shape[0]))
    return robust_shape + robust_level, coverage


def _window_spectrum_db(frame: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    frame = np.asarray(frame, dtype=float)
    frame = frame - float(np.mean(frame))
    window = np.hamming(frame.size)
    windowed = frame * window
    psd = np.square(np.abs(np.fft.rfft(windowed))) / max(float(np.sum(window * window)), 1e-12)
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / fs)
    return freqs, 10.0 * np.log10(psd + 1e-12)


def _median_frame_spectrum_db(
    frames: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if frames.ndim != 2 or frames.shape[0] == 0:
        return None
    spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for frame in frames:
        local_freqs, spectrum_db = _window_spectrum_db(frame, fs)
        if freqs is None:
            freqs = local_freqs
        spectra.append(spectrum_db)
    assert freqs is not None
    return freqs, np.median(np.asarray(spectra, dtype=float), axis=0)


def _audio_reference_spectrum_db(
    audio: np.ndarray,
    fs: int,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    signal_arr = np.asarray(audio, dtype=float).reshape(-1)
    if signal_arr.size < nperseg:
        return None
    hop = max(1, nperseg // 2)
    frames = np.lib.stride_tricks.sliding_window_view(signal_arr, nperseg)[::hop]
    return _median_frame_spectrum_db(frames, fs)


def _spectral_snr_db(
    speech_spectrum_db: np.ndarray,
    noise_spectrum_db: np.ndarray,
) -> np.ndarray:
    """Return per-bin signal-to-noise ratio from matched total/noise spectra."""
    total_power = np.power(10.0, np.asarray(speech_spectrum_db, dtype=float) / 10.0)
    noise_power = np.power(10.0, np.asarray(noise_spectrum_db, dtype=float) / 10.0)
    noise_power = np.maximum(noise_power, 1e-18)
    signal_power = np.maximum(total_power - noise_power, noise_power * 1e-6)
    return 10.0 * np.log10(signal_power / noise_power)


def _estimate_snr_from_spectrum(
    freqs: np.ndarray,
    spectrum_db: np.ndarray,
    noise_spectrum_db: np.ndarray | None = None,
) -> float:
    """Return integrated voice-band SNR when a matched noise reference exists."""
    if noise_spectrum_db is None:
        return 0.0
    voice_mask = (freqs >= 80.0) & (freqs <= 8000.0)
    if not np.any(voice_mask):
        voice_mask = np.ones_like(freqs, dtype=bool)
    if not np.any(voice_mask):
        return 0.0
    total_power = np.power(10.0, np.asarray(spectrum_db, dtype=float)[voice_mask] / 10.0)
    noise_power = np.power(
        10.0,
        np.asarray(noise_spectrum_db, dtype=float)[voice_mask] / 10.0,
    )
    noise_sum = max(float(np.sum(noise_power)), 1e-18)
    signal_sum = max(float(np.sum(total_power - noise_power)), noise_sum * 1e-6)
    return float(10.0 * np.log10(signal_sum / noise_sum))


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


def analyze_voice_spectrum(
    audio,
    fs=48000,
    nperseg=4096,
    *,
    vad_probabilities: np.ndarray | None = None,
    noise_audio: np.ndarray | None = None,
    noise_spectrum_override: tuple[np.ndarray, np.ndarray] | None = None,
    noise_reference_source_override: str | None = None,
) -> VoiceSpectrumResult:
    """Analyze speech windows and return a robust repeatability-aware spectrum.

    ``vad_probabilities`` contains one posterior per Silero model window. When
    supplied, the energy mask and neural posterior are fused at analysis-frame
    centres. ``noise_audio`` is an optional room-noise capture used as the
    authoritative frequency-dependent SNR reference. A prevalidated
    ``noise_spectrum_override`` takes precedence so callers can use a
    conservative fusion of explicit room tone and credible in-capture quiet
    frames. Without either reference, sufficiently quiet non-speech frames
    from this capture are used; otherwise SNR remains explicitly unavailable.
    """
    if len(audio) < nperseg:
        raise ValueError(
            f"Audio too short for FFT: need {nperseg} samples, "
            f"got {len(audio)} ({len(audio)/fs:.2f} seconds)"
        )

    audio_arr = np.asarray(audio, dtype=float)
    hop = max(1, nperseg // 2)
    frames = np.lib.stride_tricks.sliding_window_view(audio_arr, nperseg)[::hop]
    frame_rms = _frame_rms_db(frames)
    frame_starts = np.arange(frames.shape[0], dtype=int) * hop
    frame_vad_probabilities = _interpolate_vad_probabilities(
        vad_probabilities,
        frame_starts,
        nperseg,
        fs,
    )
    voiced_mask = _voiced_frame_mask(
        frame_rms,
        vad_probabilities=vad_probabilities,
        frame_starts=frame_starts,
        frame_size=nperseg,
        sample_rate=fs,
    )
    voiced_ratio = float(np.mean(voiced_mask)) if voiced_mask.size else 0.0
    voiced_frames = frames[voiced_mask]
    vad_active_ratio = (
        float(np.mean(frame_vad_probabilities >= 0.35))
        if frame_vad_probabilities is not None
        else 0.0
    )
    vad_used = frame_vad_probabilities is not None
    speech_reference = _median_frame_spectrum_db(voiced_frames, fs)
    noise_reference: tuple[np.ndarray, np.ndarray] | None = None
    noise_reference_source = "unavailable"
    if noise_spectrum_override is not None:
        override_freqs = np.asarray(noise_spectrum_override[0], dtype=float)
        override_spectrum = np.asarray(noise_spectrum_override[1], dtype=float)
        if (
            override_freqs.ndim == 1
            and override_spectrum.shape == override_freqs.shape
            and override_freqs.size >= 2
            and np.all(np.isfinite(override_freqs))
            and np.all(np.isfinite(override_spectrum))
        ):
            noise_reference = (override_freqs, override_spectrum)
            noise_reference_source = (
                str(noise_reference_source_override)
                if noise_reference_source_override
                else "validated_conservative"
            )
    if noise_reference is None and noise_audio is not None:
        noise_reference = _audio_reference_spectrum_db(
            np.asarray(noise_audio, dtype=float),
            fs,
            nperseg,
        )
        if noise_reference is not None:
            noise_reference_source = "explicit_capture"
    if noise_reference is None:
        unvoiced_frames = frames[~voiced_mask]
        if unvoiced_frames.shape[0] >= MIN_VOICED_FRAMES and voiced_frames.shape[0] > 0:
            voiced_level = float(np.median(frame_rms[voiced_mask]))
            unvoiced_level = float(np.median(frame_rms[~voiced_mask]))
            if voiced_level - unvoiced_level >= 3.0:
                noise_reference = _median_frame_spectrum_db(unvoiced_frames, fs)
                if noise_reference is not None:
                    noise_reference_source = "in_capture_non_speech"

    noise_spectrum_db: np.ndarray | None = None
    spectral_snr_db: np.ndarray | None = None
    reference_speech_db: np.ndarray | None = None
    reference_freqs: np.ndarray | None = None
    if speech_reference is not None:
        reference_freqs, reference_speech_db = speech_reference
    if noise_reference is not None and reference_freqs is not None and reference_speech_db is not None:
        noise_freqs, raw_noise_spectrum = noise_reference
        noise_spectrum_db = np.interp(
            reference_freqs,
            noise_freqs,
            raw_noise_spectrum,
            left=float(raw_noise_spectrum[0]),
            right=float(raw_noise_spectrum[-1]),
        )
        spectral_snr_db = _spectral_snr_db(reference_speech_db, noise_spectrum_db)

    if voiced_frames.shape[0] < MIN_VOICED_FRAMES or voiced_ratio < MIN_VOICED_FRAME_RATIO:
        freqs, spectrum_db = compute_voice_spectrum(audio_arr, fs, nperseg)
        aligned_noise_spectrum: np.ndarray | None = None
        aligned_spectral_snr: np.ndarray | None = None
        if (
            reference_freqs is not None
            and reference_speech_db is not None
            and noise_spectrum_db is not None
        ):
            aligned_noise_spectrum = np.interp(freqs, reference_freqs, noise_spectrum_db)
            aligned_speech_spectrum = np.interp(freqs, reference_freqs, reference_speech_db)
            aligned_spectral_snr = _spectral_snr_db(
                aligned_speech_spectrum,
                aligned_noise_spectrum,
            )
        repeatability = np.full_like(freqs, 0.45, dtype=float)
        return VoiceSpectrumResult(
            freqs=freqs,
            median_spectrum_db=spectrum_db,
            window_spectra_db=np.asarray([spectrum_db], dtype=float),
            voiced_window_ratio=max(voiced_ratio, 1.0 / max(1, frames.shape[0])),
            snr_db=_estimate_snr_from_spectrum(
                freqs,
                spectrum_db,
                aligned_noise_spectrum,
            ),
            spectral_repeatability=repeatability,
            spectral_tilt_db_per_octave=_estimate_tilt_db_per_octave(freqs, spectrum_db),
            residual_confidence=0.45,
            used_single_spectrum_fallback=True,
            measurement_coverage=0.45,
            outlier_rejection_ratio=0.0,
            vad_probability_used=vad_used,
            vad_active_window_ratio=vad_active_ratio,
            spectral_snr_db=aligned_spectral_snr,
            noise_spectrum_db=aligned_noise_spectrum,
            noise_reference_source=noise_reference_source,
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
    median_spectrum, inlier_ratio = _robust_median_spectrum(freqs, smoothed_spectra)
    if noise_spectrum_db is not None and reference_freqs is not None:
        noise_spectrum_db = np.interp(freqs, reference_freqs, noise_spectrum_db)
        spectral_snr_db = _spectral_snr_db(median_spectrum, noise_spectrum_db)
    snr_db = _estimate_snr_from_spectrum(freqs, median_spectrum, noise_spectrum_db)
    snr_confidence = (
        float(np.clip((snr_db - 3.0) / 15.0, 0.0, 1.0))
        if noise_spectrum_db is not None
        else 0.25
    )
    voice_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    repeatability_score = (
        float(np.median(repeatability[voice_mask]))
        if np.any(voice_mask)
        else float(np.median(repeatability))
    )
    coverage = float(np.clip(voiced_ratio / 0.55, 0.0, 1.0))
    measurement_coverage = float(np.clip(0.55 * coverage + 0.45 * inlier_ratio, 0.0, 1.0))
    residual_confidence = float(
        np.clip(
            0.45 * repeatability_score
            + 0.25 * coverage
            + 0.20 * snr_confidence,
            0.0,
            1.0,
        )
    )
    residual_confidence = float(
        np.clip(residual_confidence * (0.75 + 0.25 * measurement_coverage), 0.0, 1.0)
    )
    if noise_spectrum_db is None:
        residual_confidence = min(residual_confidence, 0.70)

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
        measurement_coverage=measurement_coverage,
        outlier_rejection_ratio=1.0 - inlier_ratio,
        vad_probability_used=vad_used,
        vad_active_window_ratio=vad_active_ratio,
        spectral_snr_db=spectral_snr_db,
        noise_spectrum_db=noise_spectrum_db,
        noise_reference_source=noise_reference_source,
    )


def compute_multiresolution_spectrum(
    audio: np.ndarray,
    fs: int = 48_000,
    output_nperseg: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Experimental DPSS multi-taper spectrum blended across three resolutions."""
    audio_arr = np.asarray(audio, dtype=float)
    if audio_arr.size < output_nperseg:
        raise ValueError(f"Audio too short for multi-resolution spectrum: {audio_arr.size}")
    voiced = _select_voiced_samples(audio_arr, output_nperseg, output_nperseg // 2)
    source = voiced if voiced.size >= output_nperseg else audio_arr
    target_freqs = np.fft.rfftfreq(output_nperseg, 1.0 / fs)
    resolutions = [size for size in (2048, 4096, 8192) if size <= source.size]
    estimates: dict[int, np.ndarray] = {}

    for size in resolutions:
        hop = size // 2
        frames = np.lib.stride_tricks.sliding_window_view(source, size)[::hop]
        if frames.shape[0] > 24:
            selected = np.linspace(0, frames.shape[0] - 1, 24, dtype=int)
            frames = frames[selected]
        tapers = dpss(size, NW=2.5, Kmax=3, sym=False)
        spectra = []
        for frame in frames:
            for taper in tapers:
                normalization = max(float(np.sum(taper * taper)), 1e-12)
                power = np.square(np.abs(np.fft.rfft(frame * taper))) / normalization
                spectra.append(power)
        median_power = np.median(np.asarray(spectra, dtype=float), axis=0)
        local_freqs = np.fft.rfftfreq(size, 1.0 / fs)
        estimates[size] = np.interp(target_freqs, local_freqs, 10.0 * np.log10(median_power + 1e-12))

    short = estimates[min(resolutions)]
    medium = estimates[min(resolutions, key=lambda size: abs(size - 4096))]
    long = estimates[max(resolutions)]
    low_weight = np.clip((700.0 - target_freqs) / 500.0, 0.0, 1.0)
    high_weight = np.clip((target_freqs - 3500.0) / 2500.0, 0.0, 1.0)
    medium_weight = np.clip(1.0 - low_weight - high_weight, 0.0, 1.0)
    weight_sum = np.maximum(low_weight + medium_weight + high_weight, 1e-12)
    blended = (low_weight * long + medium_weight * medium + high_weight * short) / weight_sum
    return target_freqs, np.asarray(blended, dtype=float)


def evaluate_spectrum_estimators(
    microphone_position_fixtures: list[np.ndarray],
    fs: int = 48_000,
) -> SpectrumEstimatorEvaluation:
    """Compare shape stability and retain Welch unless every voice band improves materially."""
    if len(microphone_position_fixtures) < 3:
        raise ValueError("At least three labelled microphone-position fixtures are required")
    current_rows = []
    multires_rows = []
    common_freqs: np.ndarray | None = None
    for fixture in microphone_position_fixtures:
        current_freqs, current = compute_voice_spectrum(fixture, fs=fs)
        multires_freqs, multires = compute_multiresolution_spectrum(fixture, fs=fs)
        if common_freqs is None:
            common_freqs = current_freqs
        assert common_freqs is not None
        multires = np.interp(common_freqs, multires_freqs, multires)
        voice_mask = (common_freqs >= 100.0) & (common_freqs <= 8000.0)
        current_rows.append(current - float(np.median(current[voice_mask])))
        multires_rows.append(multires - float(np.median(multires[voice_mask])))

    assert common_freqs is not None
    current_std = np.std(np.asarray(current_rows), axis=0)
    multires_std = np.std(np.asarray(multires_rows), axis=0)
    bands = {
        "low_frequency": (80.0, 300.0),
        "formant": (300.0, 3500.0),
        "sibilance": (5000.0, 10_000.0),
    }
    current_band = {}
    multires_band = {}
    improvement = {}
    for name, (low_hz, high_hz) in bands.items():
        mask = (common_freqs >= low_hz) & (common_freqs <= high_hz)
        current_band[name] = float(np.median(current_std[mask]))
        multires_band[name] = float(np.median(multires_std[mask]))
        improvement[name] = current_band[name] - multires_band[name]

    material = bool(
        all(value >= MULTIRES_MATERIAL_IMPROVEMENT_DB for value in improvement.values())
        and float(np.mean(list(improvement.values()))) >= MULTIRES_MATERIAL_IMPROVEMENT_DB
    )
    return SpectrumEstimatorEvaluation(
        current_band_stability_db=current_band,
        multires_band_stability_db=multires_band,
        improvement_db=improvement,
        material_improvement=material,
        selected_estimator="multiresolution_dpss" if material else SPECTRUM_ESTIMATOR_POLICY,
    )


def get_octave_frequencies(fraction=6, limits=(20, 20000), ref_freq=1000.0):
    """
    Calculate fractional-octave centers using IEC 61260-1 spacing equations.

    This helper calculates nominal centers and edges only. It does not implement
    or claim conformance for an IEC filter bank.

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
    averaging energy within fractional-octave bands whose center and edge
    spacing follows the IEC 61260-1 equations. This is offline smoothing, not a
    certified IEC filter-bank implementation.

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


def smooth_spectrum_perceptual(freqs, spectrum_db, strength="balanced"):
    """Apply voice-aware smoothing that varies by frequency region."""
    freqs = np.asarray(freqs, dtype=float)
    spectrum_db = np.asarray(spectrum_db, dtype=float)
    strength = str(strength or "balanced").lower()
    wide = smooth_spectrum_octave(freqs, spectrum_db, fraction=3)
    medium = smooth_spectrum_octave(freqs, spectrum_db, fraction=6)
    fine = smooth_spectrum_octave(freqs, spectrum_db, fraction=12)
    very_wide = smooth_spectrum_octave(freqs, spectrum_db, fraction=2)

    smoothed = medium.copy()
    low_mask = freqs < 180.0
    mid_mask = (freqs >= 180.0) & (freqs < 3500.0)
    sibilance_mask = (freqs >= 3500.0) & (freqs <= 9000.0)
    high_mask = freqs > 9000.0
    smoothed[low_mask] = wide[low_mask]
    smoothed[mid_mask] = medium[mid_mask]
    smoothed[sibilance_mask] = fine[sibilance_mask]
    smoothed[high_mask] = wide[high_mask]
    if strength == "conservative":
        smoothed[mid_mask] = 0.65 * medium[mid_mask] + 0.35 * wide[mid_mask]
        smoothed[sibilance_mask] = 0.60 * fine[sibilance_mask] + 0.40 * medium[sibilance_mask]
        smoothed = 0.85 * smoothed + 0.15 * very_wide
    elif strength == "broad":
        smoothed = 0.50 * smoothed + 0.50 * very_wide
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
