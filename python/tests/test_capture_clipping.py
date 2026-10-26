"""A raw setup capture is clipped by sustained flat tops, not by a stray full-scale sample."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from mic_eq.analysis import voice_setup
from mic_eq.analysis.voice_setup import raw_capture_is_clipped, validate_voice_setup_verification
from mic_eq.config import EQSettings, Preset

RATES = [44_100, 48_000]
# 0.125 ms of flat top is 6 samples at both common rates; 5 is the longest click-sized run.
LONGEST_UNCLIPPED_RUN = {16_000: 1, 44_100: 5, 48_000: 5, 96_000: 11}
MAX_UNCLIPPED_EVENTS = 20


def _speech(rate: int, seconds: float = 2.0) -> np.ndarray:
    """Seeded voiced-speech stand-in (120 Hz harmonics, syllabic envelope) with a peak of exactly 1.0."""
    rng = np.random.default_rng(1)
    t = np.arange(int(rate * seconds)) / rate
    voice = sum(np.sin(2 * np.pi * 120 * h * t + rng.uniform(0, 2 * np.pi)) / h for h in range(1, 40))
    voice = voice * (0.2 + 0.8 * np.sin(np.pi * 3 * t) ** 2)
    return (voice / np.max(np.abs(voice))).astype(np.float32)


def _spaced(size: int, count: int, length: int = 1, value: float = 1.0) -> np.ndarray:
    """A quiet take with ``count`` separate runs of ``length`` samples at ``value``."""
    audio = np.full(size, 0.01, dtype=np.float32)
    for start in np.linspace(1_000, size - 1_000, count).astype(int):
        audio[start : start + length] = value
    return audio


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stray_full_scale_samples_are_not_clipping(rate, count, sign):
    take = _speech(rate) * 0.5
    take[np.linspace(1_000, take.size - 1_000, count).astype(int)] = sign
    assert not raw_capture_is_clipped(take, rate)


@pytest.mark.parametrize("rate", RATES)
def test_a_peak_normalised_to_exactly_full_scale_is_not_clipping(rate):
    take = _speech(rate)
    assert float(np.max(np.abs(take))) == 1.0
    assert not raw_capture_is_clipped(take, rate)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("gain_db", [3, 6, 12])
def test_hard_clipped_speech_is_clipping(rate, gain_db):
    take = np.clip(_speech(rate) * np.float32(10 ** (gain_db / 20)), -1.0, 1.0)
    assert raw_capture_is_clipped(take, rate)


@pytest.mark.parametrize(("rate", "longest_ok"), LONGEST_UNCLIPPED_RUN.items())
def test_a_flat_top_is_clipping_from_about_an_eighth_of_a_millisecond(rate, longest_ok):
    size = rate * 2
    assert not raw_capture_is_clipped(_spaced(size, 1, longest_ok), rate)
    assert raw_capture_is_clipped(_spaced(size, 1, longest_ok + 1), rate)
    assert raw_capture_is_clipped(_spaced(size, 1, longest_ok + 1, value=-1.0), rate)


def test_the_sample_rate_defaults_to_48_khz():
    take = _spaced(96_000, 1, 6)
    assert raw_capture_is_clipped(take) and not raw_capture_is_clipped(take, 96_000)
    assert not raw_capture_is_clipped(_spaced(96_000, 1, 5))


@pytest.mark.parametrize("rate", RATES)
def test_clicks_recurring_past_a_handful_are_clipping(rate):
    size = rate * 2
    assert not raw_capture_is_clipped(_spaced(size, MAX_UNCLIPPED_EVENTS), rate)
    assert raw_capture_is_clipped(_spaced(size, MAX_UNCLIPPED_EVENTS + 1), rate)
    assert not raw_capture_is_clipped(_spaced(size, MAX_UNCLIPPED_EVENTS, length=5), rate)
    assert raw_capture_is_clipped(_spaced(size, MAX_UNCLIPPED_EVENTS + 1, length=5), rate)


def test_full_scale_starts_at_0_999():
    assert raw_capture_is_clipped(_spaced(48_000, 1, 6, value=0.999), 48_000)
    assert not raw_capture_is_clipped(_spaced(48_000, 1, 6, value=0.9989), 48_000)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_sample_is_clipping_wherever_it_is(bad):
    for index in (0, 500, 999):
        take = np.full(1_000, 0.01, dtype=np.float32)
        take[index] = bad
        assert raw_capture_is_clipped(take)
    assert raw_capture_is_clipped(np.array([bad], dtype=np.float32))


def test_empty_and_very_short_takes_follow_the_same_rule():
    assert not raw_capture_is_clipped(np.zeros(0, dtype=np.float32))
    assert not raw_capture_is_clipped(np.array([1.0], dtype=np.float32))
    assert not raw_capture_is_clipped(np.ones(5, dtype=np.float32))
    assert raw_capture_is_clipped(np.ones(6, dtype=np.float32))
    assert raw_capture_is_clipped(-np.ones(6, dtype=np.float64))


# The same rule has to hold where the analysis and the verification apply it.


def test_the_setup_analysis_refuses_sustained_clipping_but_not_a_stray_sample():
    rate = 48_000
    short = np.full(rate // 10, 0.01, dtype=np.float32)
    stray, sustained = short.copy(), short.copy()
    stray[100] = 1.0
    sustained[100:106] = 1.0

    with pytest.raises(ValueError, match="too short"):
        voice_setup.analyze_voice_setup(
            stray, short, rate, "broadcast", raw_noise_audio=stray, raw_speech_audio=short
        )
    with pytest.raises(ValueError, match="non-finite or clipped"):
        voice_setup.analyze_voice_setup(
            sustained, short, rate, "broadcast", raw_noise_audio=sustained, raw_speech_audio=short
        )


def test_the_setup_analysis_refuses_a_clipped_capture_whichever_raw_capture_it_is_given():
    rate = 48_000
    short = np.full(rate // 10, 0.01, dtype=np.float32)
    sustained = short.copy()
    sustained[100:106] = 1.0

    with pytest.raises(ValueError, match="non-finite or clipped"):
        voice_setup.analyze_voice_setup(short, short, rate, "broadcast", raw_noise_audio=sustained)
    with pytest.raises(ValueError, match="non-finite or clipped"):
        voice_setup.analyze_voice_setup(short, short, rate, "broadcast", raw_speech_audio=sustained)


def test_the_verification_passage_is_refused_for_sustained_clipping_only():
    rate = 48_000
    audio = _speech(rate, 3.5) * 0.5
    stray, sustained = audio.copy(), audio.copy()
    stray[1_000] = 1.0
    sustained[1_000:1_006] = 1.0

    def verify(take):
        return validate_voice_setup_verification(audio, audio, take, rate, {}, "broadcast")

    # An empty setup result ends in a rollback right after the clipping check.
    assert verify(stray)["reason_codes"] == ["candidate_settings_incomplete"]
    refused = verify(sustained)
    assert refused["decision"] == "retry"
    assert refused["reason_codes"] == ["verification_capture_invalid"]
    assert refused["reasons"] == [
        "verification passage was non-finite or clipped; lower input gain and repeat"
    ]


class _Rendered(Exception):
    """Raised by the stub renderer: the take got past the clipping check."""


def test_the_full_chain_renders_refuse_sustained_clipping_but_not_a_stray_sample(monkeypatch):
    def render(*_args, **_kwargs):
        raise _Rendered

    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", render)
    rate = 48_000
    preset = Preset()
    eq = preset.eq.to_dict()
    suppressor = {**asdict(preset.rnnoise), "enabled": True}
    setup = {
        "eq_settings": eq | {
            "band_freqs": EQSettings.from_dict(eq).band_freqs,
            "band_gains": EQSettings.from_dict(eq).band_gains,
            "band_qs": EQSettings.from_dict(eq).band_qs,
        },
        "gate_settings": asdict(preset.gate),
        "suppressor_settings": suppressor,
        "deesser_settings": asdict(preset.deesser),
        "compressor_settings": asdict(preset.compressor),
        "limiter_settings": asdict(preset.limiter),
        "verification_chain_settings": {
            "full_chain": True,
            "input_pre_filtered": False,
            "input_cleanup_mode": "gentle",
            "processing_mode": "normal",
            "gate": asdict(preset.gate),
            "suppressor": suppressor,
        },
    }
    voice = _speech(rate, 3.5) * 0.5
    room = np.full(rate * 2, 0.002, dtype=np.float32)
    stray, sustained = room.copy(), room.copy()
    stray[500] = 1.0
    sustained[500:506] = 1.0

    def verify(raw_room):
        return validate_voice_setup_verification(
            room, voice, voice, rate, setup, "broadcast",
            raw_noise_audio=raw_room, raw_verification_speech_audio=voice,
        )

    with pytest.raises(_Rendered):
        verify(stray)
    refused = verify(sustained)
    assert refused["decision"] == "retry"
    assert refused["reasons"] == ["raw noise capture was non-finite or clipped"]
