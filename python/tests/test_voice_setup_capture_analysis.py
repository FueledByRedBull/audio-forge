"""Voice Setup analysis of whole synthesised captures through the native chain.

These are the slowest Voice Setup tests; they live apart from the unit-level
tests in test_voice_setup.py so the two run in separate CI shards.
"""
# pyright: reportMissingImports=false
# (the sibling test module is found by pytest at run time)

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from mic_eq.analysis import voice_setup as voice_setup_module
from mic_eq.analysis.voice_setup import analyze_voice_setup, validate_voice_setup_verification
from test_voice_setup import _FULL_ANALYSIS_VOICE_SECONDS, _make_noise, _make_voice


def test_voice_setup_uses_vad_assisted_when_available(monkeypatch):
    sample_rate = 48_000
    limiter = {
        "enabled": False,
        "ceiling_db": -2.5,
        "release_ms": 175.0,
        "careful_output_enabled": False,
    }

    def fake_offline_vad(audio, _sample_rate, *, threshold=0.48, pre_gain=1.0):
        del threshold, pre_gain
        frame_count = max(1, (np.asarray(audio).size + 511) // 512)
        return np.full(frame_count, 0.8, dtype=np.float32), "silero"

    real_simulate_candidate_chain = voice_setup_module.simulate_candidate_chain

    def simulate_with_supplied_vad(audio, rate, eq_settings, chain_settings=None):
        settings = dict(chain_settings or {})
        if settings.get("full_chain") and settings.get("processing_mode") == "normal":
            frame_count = max(1, (np.asarray(audio).size + 479) // 480)
            settings["vad_probabilities"] = [0.8] * frame_count
            settings["vad_available"] = True
        return real_simulate_candidate_chain(audio, rate, eq_settings, settings)

    monkeypatch.setattr(voice_setup_module, "analyze_offline_vad", fake_offline_vad)
    monkeypatch.setattr(
        voice_setup_module,
        "simulate_candidate_chain",
        simulate_with_supplied_vad,
    )
    result = analyze_voice_setup(
        _make_noise(sample_rate),
        _make_voice(sample_rate, seconds=_FULL_ANALYSIS_VOICE_SECONDS),
        sample_rate,
        "streaming",
        vad_available=True,
        limiter_settings=limiter,
    )

    assert result["gate_settings"]["gate_mode"] == 1
    assert result["gate_settings"]["auto_threshold_enabled"] is True
    assert result["compressor_settings"]["adaptive_release"] is True
    assert result["compressor_settings"]["enabled"] is True
    assert (
        0.0
        <= result["compressor_settings"]["noise_reference_reliability"]
        <= 1.0
    )
    assert result["diagnostics"]["setup_confidence"] > 0.0
    assert result["limiter_settings"] == limiter


LABELLED_FIXTURES = {
    # label: (noise amplitude, voice options, expected apply_recommended)
    "clean": (0.0025, {}, None),
    "sibilant": (0.0025, {"sibilant": True}, None),
    "weak_noisy": (0.012, {"level": 0.04, "noise_amplitude": 0.012}, False),
}


@pytest.fixture(scope="module")
def labelled_analysis():
    """The offline analysis of each labelled capture, run once for the module.

    The tests below only read the results, so each capture is analysed the
    first time a test asks for it.
    """
    sample_rate = 48_000
    results: dict[str, dict] = {}

    def analyse(label: str) -> dict:
        if label not in results:
            noise_amplitude, voice_options, _expected_apply = LABELLED_FIXTURES[label]
            results[label] = analyze_voice_setup(
                _make_noise(sample_rate, amplitude=noise_amplitude),
                _make_voice(sample_rate, seconds=5.0, **voice_options),
                sample_rate,
                "broadcast",
                vad_available=False,
            )
        return results[label]

    return analyse


@pytest.mark.parametrize("label", LABELLED_FIXTURES)
def test_labelled_fixture_recommendations_use_loudness_features_and_offline_dsp(
    labelled_analysis, label
):
    expected_apply = LABELLED_FIXTURES[label][2]
    result = labelled_analysis(label)
    diagnostics = result["diagnostics"]

    assert result["gate_settings"]["gate_mode"] == 0, label
    assert result["gate_settings"]["auto_threshold_enabled"] is False, label
    assert np.isfinite(diagnostics["short_term_lufs"]), label
    assert diagnostics["loudness_range_db"] >= 0.0, label
    assert diagnostics["vad_active_duration_s"] >= 0.0, label
    assert set(diagnostics["robust_band_energy_db"]) == {
        "low",
        "body",
        "presence",
        "sibilance",
    }
    assert diagnostics["offline_validation"] is not None, label
    assert isinstance(diagnostics["offline_validation_passed"], bool), label
    calibration = diagnostics["compressor_calibration"]
    if calibration["backend"] == "rust":
        measured_p95 = calibration["measured_gain_reduction_db"]
        target_p95 = calibration["target_gain_reduction_db"]
        target_met = (
            abs(measured_p95 - target_p95)
            <= calibration["target_p95_tolerance_db"]
        )
        assert calibration["target_p95_met"] is target_met, label
    else:
        assert calibration["target_p95_met"] is False, label
    if not calibration["target_p95_met"]:
        assert any(
            reason.startswith("compressor p95 gain reduction target")
            for reason in diagnostics["uncertainty_reasons"]
        ), label
        assert diagnostics["apply_recommended"] is False, label
    elif label != "weak_noisy":
        assert (
            abs(
                calibration["measured_gain_reduction_db"]
                - calibration["target_gain_reduction_db"]
            )
            <= 0.75
        ), label
    assert result["compressor_settings"]["measured_short_term_lufs"] == diagnostics[
        "short_term_lufs"
    ]
    if expected_apply is not None:
        assert diagnostics["apply_recommended"] is expected_apply, (
            label,
            diagnostics["uncertainty_reasons"],
            diagnostics["setup_confidence"],
        )


def test_the_sibilant_labelled_capture_gets_a_deesser(labelled_analysis):
    sibilant = labelled_analysis("sibilant")

    assert sibilant["deesser_settings"]["enabled"] is True
    assert (
        sibilant["deesser_settings"]["high_cut_hz"]
        > sibilant["deesser_settings"]["low_cut_hz"]
    )
    assert sibilant["diagnostics"]["deesser_temporal_contrast_db"] > 0.75
    assert (
        sibilant["diagnostics"]["offline_validation"]["deesser_gain_reduction_db"]
        > 0.25
    )


def test_the_weak_noisy_labelled_capture_is_a_weak_capture(labelled_analysis):
    assert labelled_analysis("weak_noisy")["diagnostics"]["weak_capture"] is True


def test_the_clean_labelled_capture_is_measured_against_the_validated_noise_reference(
    labelled_analysis,
):
    clean = labelled_analysis("clean")

    if not clean["eq_settings"]["apply_recommended"]:
        assert clean["eq_settings"]["abstention_reasons"]
    assert clean["diagnostics"]["noise_reference_source"] == "validated_conservative"
    assert clean["eq_settings"]["snr_reference_available"] is True


def test_verifying_a_sibilant_setup_against_an_identical_take_rolls_back(labelled_analysis):
    sample_rate = 48_000
    verification = validate_voice_setup_verification(
        _make_noise(sample_rate, amplitude=0.0025),
        _make_voice(sample_rate, seconds=5.0, sibilant=True),
        _make_voice(sample_rate, seconds=5.0, sibilant=True),
        sample_rate,
        deepcopy(labelled_analysis("sibilant")),
        "broadcast",
    )
    assert verification["decision"] == "rollback"
    assert verification["reasons"]
    assert verification["simulation_backend"] == "rust"
    assert verification["perceptual_validation"] is False
    assert set(verification["frequency_dependent_snr_db"]) == {
        "low",
        "body",
        "presence",
        "sibilance",
    }
