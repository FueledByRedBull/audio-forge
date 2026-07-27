"""Room-noise reference quality and cross-capture consistency analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

MIN_NOISE_DURATION_S = 1.5
QUESTIONABLE_CAPTURE_AGE_S = 120.0
INVALID_CAPTURE_AGE_S = 600.0
OCTAVE_CENTERS_HZ = np.asarray(
    [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    dtype=float,
)


@dataclass(frozen=True)
class CaptureMetadata:
    """Identity and timing attached to one calibration capture."""

    captured_at_unix_s: float | None = None
    input_device: str | None = None
    sample_rate: int | None = None
    channel_mode: str | None = None
    channel_count: int | None = None

    @classmethod
    def coerce(
        cls,
        value: CaptureMetadata | Mapping[str, Any] | None,
    ) -> CaptureMetadata:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("capture metadata must be a mapping or CaptureMetadata")

        timestamp = value.get("captured_at_unix_s")
        sample_rate = value.get("sample_rate")
        channel_count = value.get("channel_count")
        return cls(
            captured_at_unix_s=(
                float(timestamp)
                if timestamp is not None and np.isfinite(float(timestamp))
                else None
            ),
            input_device=_clean_optional_text(value.get("input_device")),
            sample_rate=int(sample_rate) if sample_rate is not None else None,
            channel_mode=_clean_optional_text(value.get("channel_mode")),
            channel_count=int(channel_count) if channel_count is not None else None,
        )


@dataclass
class NoiseReferenceAnalysis:
    """Quality decision plus spectra used by downstream reliability logic."""

    status: str
    quality_score: float
    usable: bool
    conservative: bool
    reasons: list[str]
    guidance: list[str]
    metrics: dict[str, Any]
    frequencies: np.ndarray
    explicit_spectrum_db: np.ndarray
    conservative_spectrum_db: np.ndarray
    in_capture_spectrum_db: np.ndarray | None = None
    conservative_noise_rms_db: float = -120.0

    def diagnostics(self) -> dict[str, Any]:
        """Return a serialization-friendly quality report."""
        return {
            "status": self.status,
            "quality_score": self.quality_score,
            "usable": self.usable,
            "conservative": self.conservative,
            "reasons": list(self.reasons),
            "guidance": list(self.guidance),
            "metrics": dict(self.metrics),
        }


@dataclass
class _FrameAnalysis:
    frequencies: np.ndarray
    spectra_db: np.ndarray
    median_spectrum_db: np.ndarray
    frame_rms_db: np.ndarray
    band_levels_db: np.ndarray
    rms_spread_db: float
    octave_stability_db: float
    spectral_flux_db: float


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _db_rms(audio: np.ndarray) -> float:
    power = float(np.mean(np.square(audio, dtype=np.float64))) if audio.size else 0.0
    return float(10.0 * np.log10(max(power, 1e-18)))


def _db_peak(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return float(20.0 * np.log10(max(peak, 1e-9)))


def _frame_analysis(audio: np.ndarray, sample_rate: int) -> _FrameAnalysis | None:
    frame_size = max(512, int(round(sample_rate * 0.20)))
    if audio.size < frame_size:
        return None
    hop_size = max(1, frame_size // 2)
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop_size]
    if frames.shape[0] == 0:
        return None

    centered = frames - np.mean(frames, axis=1, keepdims=True)
    frame_power = np.mean(np.square(centered, dtype=np.float64), axis=1)
    frame_rms_db = 10.0 * np.log10(np.maximum(frame_power, 1e-18))

    window = np.hanning(frame_size)
    normalization = max(float(np.sum(window * window)), 1e-18)
    spectra_power = np.square(np.abs(np.fft.rfft(centered * window, axis=1)))
    spectra_power /= normalization
    spectra_db = 10.0 * np.log10(np.maximum(spectra_power, 1e-18))
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)

    band_levels: list[np.ndarray] = []
    for center in OCTAVE_CENTERS_HZ:
        low = center / np.sqrt(2.0)
        high = min(center * np.sqrt(2.0), sample_rate * 0.49)
        mask = (frequencies >= low) & (frequencies < high)
        if np.any(mask):
            band_power = np.sum(spectra_power[:, mask], axis=1)
            band_levels.append(10.0 * np.log10(np.maximum(band_power, 1e-18)))
    band_levels_db = (
        np.column_stack(band_levels)
        if band_levels
        else np.empty((frames.shape[0], 0), dtype=float)
    )

    rms_spread_db = float(
        np.percentile(frame_rms_db, 90.0) - np.percentile(frame_rms_db, 10.0)
    )
    if band_levels_db.shape[1] > 0:
        per_band_spread = np.percentile(band_levels_db, 90.0, axis=0) - np.percentile(
            band_levels_db,
            10.0,
            axis=0,
        )
        octave_stability_db = float(np.median(per_band_spread))
        normalized_bands = band_levels_db - np.median(
            band_levels_db,
            axis=1,
            keepdims=True,
        )
        if normalized_bands.shape[0] >= 2:
            flux_rows = np.median(np.abs(np.diff(normalized_bands, axis=0)), axis=1)
            spectral_flux_db = float(np.percentile(flux_rows, 95.0))
        else:
            spectral_flux_db = 0.0
    else:
        octave_stability_db = 0.0
        spectral_flux_db = 0.0

    return _FrameAnalysis(
        frequencies=frequencies,
        spectra_db=spectra_db,
        median_spectrum_db=np.median(spectra_db, axis=0),
        frame_rms_db=frame_rms_db,
        band_levels_db=band_levels_db,
        rms_spread_db=rms_spread_db,
        octave_stability_db=octave_stability_db,
        spectral_flux_db=spectral_flux_db,
    )


def _interpolate_vad(
    probabilities: np.ndarray | None,
    frame_count: int,
) -> np.ndarray | None:
    if probabilities is None or frame_count <= 0:
        return None
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0:
        return None
    source = (np.arange(values.size, dtype=float) + 0.5) / values.size
    target = (np.arange(frame_count, dtype=float) + 0.5) / frame_count
    return np.interp(
        target,
        source,
        np.clip(values, 0.0, 1.0),
        left=float(np.clip(values[0], 0.0, 1.0)),
        right=float(np.clip(values[-1], 0.0, 1.0)),
    )


def _quality_mean(components: list[tuple[float, float]]) -> float:
    values = np.asarray(
        [np.clip(value, 0.0, 1.0) for value, _weight in components],
        dtype=float,
    )
    weights = np.asarray(
        [max(0.0, weight) for _value, weight in components],
        dtype=float,
    )
    if not components or float(np.sum(weights)) <= 0.0:
        return 0.0
    weights /= float(np.sum(weights))
    return float(np.exp(np.sum(weights * np.log(np.maximum(values, 0.02)))))


def _metadata_mismatches(
    noise: CaptureMetadata,
    speech: CaptureMetadata,
    sample_rate: int,
) -> tuple[list[str], float | None]:
    reasons: list[str] = []
    for label, left, right in (
        ("input device", noise.input_device, speech.input_device),
        ("input channel mode", noise.channel_mode, speech.channel_mode),
        ("channel count", noise.channel_count, speech.channel_count),
    ):
        if left is not None and right is not None and left != right:
            reasons.append(f"{label} changed between noise and voice captures")
    for label, metadata in (("noise", noise), ("voice", speech)):
        if metadata.sample_rate is not None and metadata.sample_rate != sample_rate:
            reasons.append(f"{label} capture sample rate does not match analysis")
    if (
        noise.sample_rate is not None
        and speech.sample_rate is not None
        and noise.sample_rate != speech.sample_rate
    ):
        reasons.append("sample rate changed between noise and voice captures")

    age_s: float | None = None
    if noise.captured_at_unix_s is not None and speech.captured_at_unix_s is not None:
        age_s = max(0.0, speech.captured_at_unix_s - noise.captured_at_unix_s)
    return reasons, age_s


def _select_in_capture_noise(
    speech_frames: _FrameAnalysis | None,
    speech_vad_probabilities: np.ndarray | None,
) -> tuple[np.ndarray | None, float | None, int]:
    if speech_frames is None or speech_frames.spectra_db.shape[0] < 4:
        return None, None, 0

    frame_rms = speech_frames.frame_rms_db
    vad = _interpolate_vad(speech_vad_probabilities, frame_rms.size)
    if vad is not None:
        threshold = float(np.percentile(frame_rms, 35.0))
        mask = (vad <= 0.25) & (frame_rms <= threshold)
    else:
        spread = float(np.percentile(frame_rms, 90.0) - np.percentile(frame_rms, 10.0))
        if spread < 6.0:
            return None, None, 0
        mask = frame_rms <= float(np.percentile(frame_rms, 15.0))

    minimum = max(3, int(np.ceil(frame_rms.size * 0.05)))
    if int(np.count_nonzero(mask)) < minimum:
        return None, None, int(np.count_nonzero(mask))
    return (
        np.median(speech_frames.spectra_db[mask], axis=0),
        float(np.median(frame_rms[mask])),
        int(np.count_nonzero(mask)),
    )


def analyze_noise_reference(
    noise_audio: np.ndarray,
    speech_audio: np.ndarray | None,
    sample_rate: int,
    *,
    noise_metadata: CaptureMetadata | Mapping[str, Any] | None = None,
    speech_metadata: CaptureMetadata | Mapping[str, Any] | None = None,
    noise_vad_probabilities: np.ndarray | None = None,
    speech_vad_probabilities: np.ndarray | None = None,
) -> NoiseReferenceAnalysis:
    """Assess a room-noise capture and derive a conservative noise spectrum."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    raw_noise = np.asarray(noise_audio, dtype=float).reshape(-1)
    finite_mask = np.isfinite(raw_noise)
    finite_fraction = float(np.mean(finite_mask)) if raw_noise.size else 0.0
    noise = np.where(finite_mask, raw_noise, 0.0)
    duration_s = float(noise.size / sample_rate)
    noise_rms_db = _db_rms(noise)
    noise_peak_db = _db_peak(noise)
    crest_factor_db = max(0.0, noise_peak_db - noise_rms_db)
    clipped_fraction = (
        float(np.mean(np.abs(noise) >= 0.999)) if noise.size else 0.0
    )
    zero_fraction = (
        float(np.mean(np.abs(noise) <= 1e-12)) if noise.size else 1.0
    )
    noise_frames = _frame_analysis(noise, sample_rate)

    reasons: list[str] = []
    guidance: list[str] = []
    invalid = False
    questionable = False

    if duration_s < MIN_NOISE_DURATION_S:
        invalid = True
        reasons.append("room-noise capture is too short")
        guidance.append(f"Record at least {MIN_NOISE_DURATION_S:.1f} seconds of room tone.")
    if finite_fraction < 1.0:
        invalid = True
        reasons.append("room-noise capture contains non-finite samples")
        guidance.append("Restart the audio stream and record the room tone again.")
    if noise_rms_db <= -95.0 or (zero_fraction >= 0.995 and noise_peak_db <= -90.0):
        invalid = True
        reasons.append("room-noise capture is suspiciously silent")
        guidance.append("Check the selected microphone and record normal room tone again.")
    if clipped_fraction > 0.001:
        invalid = True
        reasons.append("room-noise capture is clipped")
        guidance.append("Lower input gain or remove the transient source, then recapture.")
    elif clipped_fraction > 0.0:
        questionable = True
        reasons.append("room-noise capture contains isolated clipped samples")
        guidance.append("Recapture without taps or handling noise for a cleaner reference.")

    if noise_frames is None:
        invalid = True
        reasons.append("room-noise capture has too few analysis windows")
        frequencies = np.fft.rfftfreq(max(2, noise.size), 1.0 / sample_rate)
        explicit_spectrum = np.full(frequencies.shape, -120.0, dtype=float)
        rms_spread_db = 120.0
        octave_stability_db = 120.0
        spectral_flux_db = 120.0
    else:
        frequencies = noise_frames.frequencies
        explicit_spectrum = noise_frames.median_spectrum_db
        rms_spread_db = noise_frames.rms_spread_db
        octave_stability_db = noise_frames.octave_stability_db
        spectral_flux_db = noise_frames.spectral_flux_db
        if rms_spread_db > 12.0 or octave_stability_db > 14.0:
            invalid = True
            reasons.append("room-noise capture is dominated by changing events")
            guidance.append("Wait for the room to settle and record a new reference.")
        elif rms_spread_db > 6.0 or octave_stability_db > 8.0:
            questionable = True
            reasons.append("room-noise capture is not stationary")
            guidance.append("Avoid movement, speech, and intermittent sounds while recapturing.")
        if spectral_flux_db > 10.0:
            invalid = True
            reasons.append("room-noise capture contains dominant transient events")
            guidance.append("Recapture without keyboard, handling, or impact sounds.")
        elif spectral_flux_db > 6.0 or crest_factor_db > 24.0:
            questionable = True
            reasons.append("room-noise capture contains strong transients")
            guidance.append("Recapture without keyboard, handling, or impact sounds.")

    noise_vad = _interpolate_vad(
        noise_vad_probabilities,
        noise_frames.frame_rms_db.size if noise_frames is not None else 0,
    )
    vad_contamination_ratio = (
        float(np.mean(noise_vad >= 0.35)) if noise_vad is not None else 0.0
    )
    vad_contamination_p90 = (
        float(np.percentile(noise_vad, 90.0)) if noise_vad is not None else 0.0
    )
    if vad_contamination_ratio > 0.30:
        invalid = True
        reasons.append("speech is present in the room-noise capture")
        guidance.append("Remain silent and record the room noise again.")
    elif vad_contamination_ratio > 0.08 or vad_contamination_p90 > 0.55:
        questionable = True
        reasons.append("possible speech contamination in room-noise capture")
        guidance.append("Record another room-noise sample without voices.")

    noise_meta = CaptureMetadata.coerce(noise_metadata)
    speech_meta = CaptureMetadata.coerce(speech_metadata)
    metadata_reasons, capture_age_s = _metadata_mismatches(
        noise_meta,
        speech_meta,
        sample_rate,
    )
    if metadata_reasons:
        invalid = True
        reasons.extend(metadata_reasons)
        guidance.append("Use the same microphone, channel mode, and sample rate for both captures.")
    if capture_age_s is not None:
        if capture_age_s > INVALID_CAPTURE_AGE_S:
            invalid = True
            reasons.append("room-noise reference is stale")
            guidance.append("Record room noise immediately before the voice sample.")
        elif capture_age_s > QUESTIONABLE_CAPTURE_AGE_S:
            questionable = True
            reasons.append("room-noise reference may be stale")
            guidance.append("Recapture room noise under the current conditions.")

    speech = (
        np.asarray(speech_audio, dtype=float).reshape(-1)
        if speech_audio is not None
        else np.empty(0, dtype=float)
    )
    speech = np.where(np.isfinite(speech), speech, 0.0)
    speech_frames = _frame_analysis(speech, sample_rate) if speech.size else None
    in_capture_spectrum, in_capture_rms_db, in_capture_frame_count = (
        _select_in_capture_noise(speech_frames, speech_vad_probabilities)
    )

    level_delta_db: float | None = None
    spectral_shape_distance_db: float | None = None
    conservative_spectrum = explicit_spectrum.copy()
    conservative_rms_db = noise_rms_db
    if (
        in_capture_spectrum is not None
        and speech_frames is not None
        and in_capture_rms_db is not None
    ):
        aligned_in_capture = np.interp(
            frequencies,
            speech_frames.frequencies,
            in_capture_spectrum,
            left=float(in_capture_spectrum[0]),
            right=float(in_capture_spectrum[-1]),
        )
        in_capture_spectrum = aligned_in_capture
        level_delta_db = float(in_capture_rms_db - noise_rms_db)
        voice_mask = (frequencies >= 80.0) & (frequencies <= 8000.0)
        if not np.any(voice_mask):
            voice_mask = np.ones(frequencies.shape, dtype=bool)
        explicit_shape = explicit_spectrum[voice_mask] - float(
            np.median(explicit_spectrum[voice_mask])
        )
        capture_shape = aligned_in_capture[voice_mask] - float(
            np.median(aligned_in_capture[voice_mask])
        )
        spectral_shape_distance_db = float(
            np.median(np.abs(explicit_shape - capture_shape))
        )

        conservative_spectrum = np.maximum(explicit_spectrum, aligned_in_capture)
        conservative_rms_db = max(noise_rms_db, in_capture_rms_db)
        if level_delta_db > 12.0 or spectral_shape_distance_db > 10.0:
            invalid = True
            reasons.append("room noise does not match conditions during the voice capture")
            guidance.append("Recapture room noise and voice without changing the environment.")
        elif level_delta_db > 6.0 or spectral_shape_distance_db > 5.5:
            questionable = True
            reasons.append("room-noise reference only partly matches the voice capture")
            guidance.append("Recapture both samples for a more reliable correction.")
        elif level_delta_db < -20.0:
            invalid = True
            reasons.append("room-noise level changed substantially before the voice capture")
            guidance.append("Record room noise and voice under the same conditions.")
        elif level_delta_db < -12.0:
            questionable = True
            reasons.append("room-noise reference is much louder than in-capture quiet frames")
            guidance.append("Check whether the noise source changed between captures.")

    duration_score = np.clip(duration_s / 3.0, 0.0, 1.0)
    finite_score = np.clip((finite_fraction - 0.995) / 0.005, 0.0, 1.0)
    stationarity_score = np.clip(1.0 - rms_spread_db / 12.0, 0.0, 1.0)
    octave_score = np.clip(1.0 - octave_stability_db / 14.0, 0.0, 1.0)
    transient_score = np.clip(1.0 - max(0.0, crest_factor_db - 12.0) / 18.0, 0.0, 1.0)
    contamination_score = np.clip(1.0 - vad_contamination_ratio / 0.30, 0.0, 1.0)
    consistency_score = 1.0
    if level_delta_db is not None:
        consistency_score *= float(np.clip(1.0 - max(0.0, level_delta_db) / 12.0, 0.0, 1.0))
    if spectral_shape_distance_db is not None:
        consistency_score *= float(
            np.clip(1.0 - spectral_shape_distance_db / 10.0, 0.0, 1.0)
        )
    age_score = (
        1.0
        if capture_age_s is None
        else float(np.clip(1.0 - capture_age_s / INVALID_CAPTURE_AGE_S, 0.0, 1.0))
    )
    quality_score = _quality_mean(
        [
            (float(duration_score), 0.10),
            (float(finite_score), 0.10),
            (float(stationarity_score), 0.18),
            (float(octave_score), 0.15),
            (float(transient_score), 0.10),
            (float(contamination_score), 0.15),
            (float(consistency_score), 0.17),
            (float(age_score), 0.05),
        ]
    )
    if invalid:
        quality_score = min(quality_score, 0.20)
        status = "invalid"
    elif questionable:
        quality_score = min(quality_score, 0.64)
        status = "questionable"
    else:
        status = "usable"

    metrics: dict[str, Any] = {
        "duration_s": duration_s,
        "finite_fraction": finite_fraction,
        "noise_rms_db": noise_rms_db,
        "conservative_noise_rms_db": conservative_rms_db,
        "noise_peak_db": noise_peak_db,
        "crest_factor_db": crest_factor_db,
        "clipped_fraction": clipped_fraction,
        "zero_fraction": zero_fraction,
        "rms_spread_db": rms_spread_db,
        "octave_stability_db": octave_stability_db,
        "spectral_flux_db": spectral_flux_db,
        "vad_contamination_ratio": vad_contamination_ratio,
        "vad_contamination_p90": vad_contamination_p90,
        "capture_age_s": capture_age_s,
        "identity_metadata_available": bool(
            noise_meta.input_device is not None and speech_meta.input_device is not None
        ),
        "in_capture_noise_frame_count": in_capture_frame_count,
        "in_capture_level_delta_db": level_delta_db,
        "spectral_shape_distance_db": spectral_shape_distance_db,
    }
    return NoiseReferenceAnalysis(
        status=status,
        quality_score=float(np.clip(quality_score, 0.0, 1.0)),
        usable=not invalid,
        conservative=bool(questionable or invalid or in_capture_spectrum is not None),
        reasons=list(dict.fromkeys(reasons)),
        guidance=list(dict.fromkeys(guidance)),
        metrics=metrics,
        frequencies=frequencies,
        explicit_spectrum_db=explicit_spectrum,
        conservative_spectrum_db=conservative_spectrum,
        in_capture_spectrum_db=in_capture_spectrum,
        conservative_noise_rms_db=conservative_rms_db,
    )
