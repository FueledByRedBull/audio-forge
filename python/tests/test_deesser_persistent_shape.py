"""Offline persistent-resonance evidence and recommendation ownership."""

from dataclasses import replace

import numpy as np
import pytest

from mic_eq.analysis import deesser_fusion as fusion
from mic_eq.analysis import voice_setup
from mic_eq.analysis.voice_setup import (
    _recommend_deesser_settings,
    _vad_masked_speech_features,
)
from mic_eq.config import DeEsserSettings


def test_fixed_input_prefix_keeps_setup_evidence_repeatable():
    audio = np.random.default_rng(98).normal(0, 0.01, 480_000)
    ordinary = fusion.persistent_shape_evidence(audio, 48_000)
    quiet = fusion.persistent_shape_evidence(audio * 0.1, 48_000)
    extended = fusion.persistent_shape_evidence(np.tile(audio, 2), 48_000)
    assert ordinary["supported"] is True
    assert ordinary["version"] == "deesser-persistent-shape-v1"
    assert quiet["score"] == pytest.approx(ordinary["score"], abs=1e-12)
    assert extended == ordinary


@pytest.mark.parametrize("audio,rate", [
    (np.zeros(480_000), 48_000),
    (np.ones(480_000), 48_000),
    (np.ones(479_999), 48_000),
    (np.ones((480_000, 1)), 48_000),
    (np.full(480_000, np.nan), 48_000),
    (np.full(480_000, np.inf), 48_000),
    (np.ones(480_000), 44_100),
])
def test_invalid_or_unsupported_capture_cannot_add_enabling(audio, rate):
    evidence = fusion.persistent_shape_evidence(audio, rate)
    assert evidence["supported"] is False
    assert evidence["score"] is None
    assert evidence["reason"]


def test_shape_preserves_signed_spectral_changes_between_halves():
    audio = np.random.default_rng(313).normal(0, 0.01, 480_000)
    spectrum = np.fft.rfft(audio[240_000:])
    frequencies = np.fft.rfftfreq(240_000, 1 / 48_000)
    spectrum[(frequencies >= 6333.34) & (frequencies < 8666.66)] *= 4
    audio[240_000:] = np.fft.irfft(spectrum)
    values = fusion._persistent_shape_features(audio)
    assert values is not None
    assert values[4] > values[1] + 8
    assert values[3] < values[0]


def test_speech_feature_extraction_keeps_the_unmasked_setup_shape():
    audio = np.random.default_rng(184).normal(0, 0.01, 480_000)
    expected = fusion.persistent_shape_evidence(audio, 48_000)
    # A posterior with no speech must not replace the fixed input feature with
    # an active-frame subset. The separate recommendation guard owns validity.
    features = _vad_masked_speech_features(
        audio, 48_000, noise_rms_db=-20.0, vad_probabilities=np.zeros(400),
    )
    evidence = features["deesser_frame_evidence"]
    assert evidence["persistent_shape"] == expected
    assert evidence["available"] is False


def _frame_data(transient):
    return {
        "available": True,
        "excess_p90_db": 4.0,
        "peak_hz": 7300.0,
        **{key: float(transient) for key in (
            "frame_probability_p90", "frame_probability_top_mean",
            "candidate_frame_ratio", "temporal_score",
            "absolute_hf_strength_p90", "noise_reliability_p90",
        )},
    }


def _recommend(frame, **kwargs):
    return _recommend_deesser_settings(
        freqs=np.linspace(100, 12000, 120),
        spectrum_db=np.zeros(120),
        capture_confidence=0.8,
        frame_evidence=frame,
        **kwargs,
    )


@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("excess", [-6.0, 4.0, 8.0])
def test_shape_composes_factory_minimums_with_transient_controls(transient, score, excess):
    frame = _frame_data(transient)
    frame["excess_p90_db"] = excess
    baseline, original_diagnostics = _recommend(frame)
    frame["persistent_shape"] = {
        "supported": True, "score": score,
        "version": "deesser-persistent-shape-v1", "reason": "supported",
    }
    settings, diagnostics = _recommend(frame)
    expected = dict(baseline)
    if score >= 0:
        factory = DeEsserSettings()
        expected.update(
            enabled=True,
            low_cut_hz=factory.low_cut_hz,
            high_cut_hz=factory.high_cut_hz,
            auto_amount=max(baseline["auto_amount"], factory.auto_amount),
            max_reduction_db=max(baseline["max_reduction_db"], factory.max_reduction_db),
        )
    assert settings == expected
    assert settings["enabled"] is diagnostics["enabled"]
    assert diagnostics["detection_probability"] == original_diagnostics["detection_probability"]
    assert diagnostics["persistent_shape"]["applied"] is (score >= 0)


def test_persistent_minimums_follow_authoritative_factory_defaults(monkeypatch):
    defaults = DeEsserSettings()
    factory = replace(
        defaults,
        auto_amount=defaults.auto_amount + 0.1,
        max_reduction_db=defaults.max_reduction_db + 1.0,
        low_cut_hz=defaults.low_cut_hz + 250.0,
        high_cut_hz=defaults.high_cut_hz - 250.0,
    )
    monkeypatch.setattr(voice_setup, "DeEsserSettings", lambda: factory)
    frame = _frame_data(False)
    frame["excess_p90_db"] = -6.0
    baseline, _ = _recommend(frame)
    frame["persistent_shape"] = {"supported": True, "score": 0.0}
    settings, _ = _recommend(frame)
    assert settings == baseline | {
        "enabled": True,
        "auto_amount": factory.auto_amount,
        "max_reduction_db": factory.max_reduction_db,
        "low_cut_hz": factory.low_cut_hz,
        "high_cut_hz": factory.high_cut_hz,
    }


@pytest.mark.parametrize("invalid", ["frames", "noise", "features"])
def test_original_validity_guard_preserves_all_transient_settings(invalid):
    frame = _frame_data(True)
    kwargs = {}
    if invalid == "frames":
        frame["available"] = False
    elif invalid == "noise":
        kwargs["noise_reference_status"] = " Invalid "
    else:
        frame["temporal_score"] = np.nan
    baseline, _ = _recommend(frame, **kwargs)
    frame["persistent_shape"] = {"supported": True, "score": 1.0}
    settings, diagnostics = _recommend(frame, **kwargs)
    assert settings["enabled"] is False
    assert settings == pytest.approx(baseline, abs=0, rel=0, nan_ok=True)
    assert diagnostics["invalid_evidence"] is True
    assert diagnostics["persistent_shape"]["applied"] is False


@pytest.mark.parametrize("shape", [
    {"supported": False, "score": 1.0},
    {"supported": True, "score": None},
    {"supported": True, "score": float("nan")},
    {"supported": True, "score": float("inf")},
])
def test_unsupported_shape_retains_transient_recommendation(shape):
    frame = _frame_data(False)
    baseline, _ = _recommend(frame)
    frame["persistent_shape"] = shape
    settings, diagnostics = _recommend(frame)
    assert settings == baseline
    assert diagnostics["persistent_shape"]["applied"] is False
