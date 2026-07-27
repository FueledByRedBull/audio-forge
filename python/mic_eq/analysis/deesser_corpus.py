"""Deterministic, source-generated labeled corpus for de-esser evaluation.

The waveforms are generated from mathematical signals and seeded noise. No
recorded voice, biometric material, or third-party audio is redistributed.
Corpus specifications and generated samples are dedicated to the public
domain under CC0-1.0; project code remains under the repository license.

This corpus is a reproducible engineering fixture. It does not substitute for
human listening tests or prove perceptual quality on real speakers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORPUS_VERSION = "audioforge-generated-deesser-corpus-v1"
CORPUS_LICENSE = "CC0-1.0"
EVENT_INTERVALS_S = (
    (0.62, 0.84),
    (1.43, 1.67),
    (2.31, 2.55),
    (3.22, 3.47),
    (4.13, 4.38),
)


@dataclass(frozen=True)
class DeEsserCorpusCase:
    """One generated clip and its clip/frame label specification."""

    name: str
    sample_rate: int
    voice_hz: float
    distance_scale: float
    condition: str
    sibilant_kind: str | None
    needs_deesser: bool
    seed: int


@dataclass
class GeneratedDeEsserCase:
    """Generated samples, VAD evidence, and event intervals."""

    specification: DeEsserCorpusCase
    noise_audio: np.ndarray
    speech_audio: np.ndarray
    vad_probabilities: np.ndarray
    event_intervals_s: tuple[tuple[float, float], ...]


def _case_matrix() -> tuple[DeEsserCorpusCase, ...]:
    cases: list[DeEsserCorpusCase] = []
    seed = 5200
    for sample_rate in (44_100, 48_000):
        for voice_label, voice_hz in (("low", 105.0), ("mid", 155.0), ("high", 220.0)):
            for distance_label, distance_scale in (("near", 1.0), ("far", 0.55)):
                for kind in ("s", "sh"):
                    cases.append(
                        DeEsserCorpusCase(
                            name=(
                                f"{voice_label}-{distance_label}-{kind}-{sample_rate}"
                            ),
                            sample_rate=sample_rate,
                            voice_hz=voice_hz,
                            distance_scale=distance_scale,
                            condition="clean",
                            sibilant_kind=kind,
                            needs_deesser=True,
                            seed=seed,
                        )
                    )
                    seed += 1

                negative_conditions = (
                    ("clean", None),
                    ("bright", None),
                    ("hiss", None),
                    ("hvac", None),
                    ("transient", None),
                    ("fricative_f", "f"),
                )
                for condition, kind in negative_conditions:
                    cases.append(
                        DeEsserCorpusCase(
                            name=(
                                f"{voice_label}-{distance_label}-{condition}-"
                                f"{sample_rate}"
                            ),
                            sample_rate=sample_rate,
                            voice_hz=voice_hz,
                            distance_scale=distance_scale,
                            condition=condition,
                            sibilant_kind=kind,
                            needs_deesser=False,
                            seed=seed,
                        )
                    )
                    seed += 1
    return tuple(cases)


CORPUS_CASES = _case_matrix()


def _band_limited_noise(
    rng: np.random.Generator,
    sample_count: int,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    white = rng.normal(size=sample_count)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(sample_count, 1.0 / sample_rate)
    transition_hz = max(150.0, 0.08 * (high_hz - low_hz))
    lower = np.clip((frequencies - low_hz) / transition_hz, 0.0, 1.0)
    upper = np.clip((high_hz - frequencies) / transition_hz, 0.0, 1.0)
    taper = np.sin(0.5 * np.pi * lower) * np.sin(0.5 * np.pi * upper)
    filtered = np.fft.irfft(spectrum * taper, n=sample_count)
    rms = float(np.sqrt(np.mean(filtered * filtered)))
    return filtered / max(rms, 1e-9)


def _event_mask(
    time_s: np.ndarray,
    intervals: tuple[tuple[float, float], ...],
) -> np.ndarray:
    mask = np.zeros(time_s.shape, dtype=float)
    for start_s, end_s in intervals:
        inside = (time_s >= start_s) & (time_s < end_s)
        count = int(np.count_nonzero(inside))
        if count == 0:
            continue
        mask[inside] = np.hanning(max(3, count))[:count]
    return mask


def generate_deesser_case(
    specification: DeEsserCorpusCase,
    *,
    duration_s: float = 5.0,
) -> GeneratedDeEsserCase:
    """Generate one deterministic corpus case."""
    sample_rate = specification.sample_rate
    sample_count = int(round(duration_s * sample_rate))
    time_s = np.arange(sample_count, dtype=float) / sample_rate
    rng = np.random.default_rng(specification.seed)

    syllable_phase = np.mod(time_s, 0.55)
    voice_envelope = np.where(syllable_phase < 0.40, 1.0, 0.025)
    voice_envelope *= 0.72 + 0.28 * np.sin(2.0 * np.pi * 1.7 * time_s) ** 2
    fundamental = specification.voice_hz
    voice = np.zeros(sample_count, dtype=float)
    for harmonic in range(1, 10):
        harmonic_hz = fundamental * harmonic
        if harmonic_hz >= sample_rate * 0.45:
            break
        formant_weight = (
            1.0 / harmonic
            * (
                1.0
                + 1.8 * np.exp(-0.5 * ((harmonic_hz - 700.0) / 260.0) ** 2)
                + 1.2 * np.exp(-0.5 * ((harmonic_hz - 2200.0) / 520.0) ** 2)
            )
        )
        voice += formant_weight * np.sin(
            2.0 * np.pi * harmonic_hz * time_s
            + 0.17 * harmonic
        )
    voice /= max(float(np.max(np.abs(voice))), 1e-9)
    speech = 0.095 * specification.distance_scale * voice_envelope * voice

    event_intervals = EVENT_INTERVALS_S if specification.sibilant_kind in {"s", "sh"} else ()
    event_envelope = _event_mask(time_s, EVENT_INTERVALS_S)
    if specification.sibilant_kind == "s":
        event_noise = _band_limited_noise(
            rng,
            sample_count,
            sample_rate,
            5200.0,
            min(10_500.0, sample_rate * 0.46),
        )
        speech += 0.10 * specification.distance_scale * event_envelope * event_noise
    elif specification.sibilant_kind == "sh":
        event_noise = _band_limited_noise(
            rng,
            sample_count,
            sample_rate,
            3600.0,
            min(8200.0, sample_rate * 0.46),
        )
        speech += 0.085 * specification.distance_scale * event_envelope * event_noise
    elif specification.sibilant_kind == "f":
        event_noise = _band_limited_noise(
            rng,
            sample_count,
            sample_rate,
            1800.0,
            min(6500.0, sample_rate * 0.44),
        )
        speech += 0.028 * specification.distance_scale * event_envelope * event_noise

    if specification.condition == "bright":
        brightness = _band_limited_noise(
            rng,
            sample_count,
            sample_rate,
            4800.0,
            min(10_500.0, sample_rate * 0.46),
        )
        speech += 0.018 * specification.distance_scale * voice_envelope * brightness
    elif specification.condition == "hiss":
        speech += 0.012 * _band_limited_noise(
            rng,
            sample_count,
            sample_rate,
            4300.0,
            min(11_000.0, sample_rate * 0.46),
        )
    elif specification.condition == "hvac":
        speech += 0.018 * np.sin(2.0 * np.pi * 120.0 * time_s)
        speech += 0.010 * _band_limited_noise(
            rng,
            sample_count,
            sample_rate,
            80.0,
            650.0,
        )
    elif specification.condition == "transient":
        for event_s in (0.78, 1.91, 3.04, 4.17):
            start = int(event_s * sample_rate)
            length = min(int(0.018 * sample_rate), sample_count - start)
            if length > 0:
                speech[start : start + length] += (
                    0.13 * np.hanning(length) * rng.normal(size=length)
                )

    room_noise = 0.0018 * rng.normal(size=sample_count)
    speech += room_noise
    noise_audio = (0.0018 * rng.normal(size=int(3.0 * sample_rate))).astype(
        np.float32
    )

    vad_window_samples = max(1, int(np.ceil(sample_rate * 512 / 16_000)))
    vad_count = int(np.ceil(sample_count / vad_window_samples))
    vad_times = (
        np.arange(vad_count, dtype=float) + 0.5
    ) * vad_window_samples / sample_rate
    vad_voice = np.interp(
        vad_times,
        time_s,
        voice_envelope,
        left=0.0,
        right=0.0,
    )
    vad_probabilities = np.where(vad_voice >= 0.20, 0.82, 0.06)
    if event_intervals:
        event_at_vad = _event_mask(vad_times, event_intervals)
        vad_probabilities = np.where(
            event_at_vad > 0.05,
            0.18,
            vad_probabilities,
        )

    return GeneratedDeEsserCase(
        specification=specification,
        noise_audio=noise_audio,
        speech_audio=np.clip(speech, -0.98, 0.98).astype(np.float32),
        vad_probabilities=vad_probabilities.astype(float),
        event_intervals_s=event_intervals,
    )


def labels_for_analysis_frames(
    generated: GeneratedDeEsserCase,
    frame_indices: np.ndarray,
    *,
    hop_ms: float = 20.0,
    frame_ms: float = 40.0,
) -> np.ndarray:
    """Return binary labels at analysis-frame centres."""
    indices = np.asarray(frame_indices, dtype=float)
    centres_s = indices * hop_ms / 1000.0 + frame_ms / 2000.0
    labels = np.zeros(indices.shape, dtype=int)
    for start_s, end_s in generated.event_intervals_s:
        labels[(centres_s >= start_s) & (centres_s < end_s)] = 1
    return labels


__all__ = [
    "CORPUS_CASES",
    "CORPUS_LICENSE",
    "CORPUS_VERSION",
    "DeEsserCorpusCase",
    "GeneratedDeEsserCase",
    "generate_deesser_case",
    "labels_for_analysis_frames",
]
