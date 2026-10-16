"""Tests for auto voice setup analysis."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import threading
import time
from typing import Any

import numpy as np
import pytest

import mic_eq
from mic_eq.analysis import voice_setup as voice_setup_module
from mic_eq.analysis import vad as vad_analysis
from mic_eq.analysis.auto_eq_parts.headroom import _is_headroom_safe
from mic_eq.analysis.compressor_calibration import _COMPRESSOR_SEARCH_BUDGET
from tools.evaluate_compressor_search import _calibrate_compressor_expanded
from mic_eq.analysis.voice_setup import (
    _calibrate_compressor_threshold,
    _fit_loudness_to_headroom,
    _recommend_compressor_settings,
    _recommend_gate_settings,
    _shape_error_db,
    _vad_masked_speech_features,
    analyze_voice_setup,
    validate_voice_setup_verification,
)
from mic_eq.analysis.auto_eq import get_target_curve, simulate_candidate_chain
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.ui.voice_setup_dialog import (
    VoiceSetupVerificationWorker,
    _candidate_settings_error,
)


def test_offline_vad_applies_live_pre_gain_before_inference(monkeypatch):
    observed: dict[str, Any] = {}

    def fake_analyze(samples, sample_rate, threshold):
        observed["samples"] = np.asarray(samples).copy()
        observed["sample_rate"] = sample_rate
        observed["threshold"] = threshold
        return [0.7]

    monkeypatch.setattr(mic_eq, "CORE_AVAILABLE", True)
    monkeypatch.setattr(mic_eq, "analyze_vad_probabilities", fake_analyze)

    probabilities, backend = vad_analysis.analyze_offline_vad(
        np.asarray([0.5, -0.25], dtype=np.float32),
        48_000,
        threshold=0.43,
        pre_gain=2.0,
    )

    np.testing.assert_allclose(observed["samples"], [1.0, -0.5])
    assert observed["sample_rate"] == 48_000
    assert observed["threshold"] == 0.43
    assert probabilities is not None
    np.testing.assert_allclose(probabilities, [0.7])
    assert backend == "silero"


def test_causal_vad_mapping_keeps_exact_44k1_window_timing():
    sample_rate = 44_100
    frame_ends = np.arange(1, 2_001, dtype=np.int64) * 441
    probabilities = np.arange(700, dtype=np.float32) / 700.0

    mapped = vad_analysis.map_causal_vad_probabilities(
        probabilities,
        frame_ends,
        sample_rate,
    )
    assert mapped is not None

    np.testing.assert_array_equal(mapped[:3], 0.0)
    # 1.6 s contains exactly 50 model windows; 20 s contains 625.
    assert mapped[159] == pytest.approx(49 / 700)
    assert mapped[-1] == pytest.approx(624 / 700)


def test_voice_setup_rejects_unsupported_sample_rate_before_analysis():
    with pytest.raises(ValueError, match="48000 Hz"):
        analyze_voice_setup(
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            44_100,
        )


def test_missing_eq_surfaces_analysis_failure_reason():
    reason = "The proposed processing exceeds safe headroom."
    assert _candidate_settings_error(
        {
            "eq_settings": None,
            "eq_error": reason,
            "limiter_settings": {
                "enabled": True,
                "ceiling_db": -0.5,
                "release_ms": 50.0,
                "careful_output_enabled": True,
            },
        }
    ) == reason


def _make_noise(sample_rate: int, seconds: float = 2.0, amplitude: float = 0.0012) -> np.ndarray:
    rng = np.random.default_rng(1200)
    return (amplitude * rng.normal(size=int(sample_rate * seconds))).astype(np.float32)


def _make_voice(
    sample_rate: int,
    *,
    seconds: float = 10.0,
    sibilant: bool = False,
    level: float = 1.0,
    noise_amplitude: float = 0.0025,
) -> np.ndarray:
    rng = np.random.default_rng(2400 if sibilant else 2401)
    t = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    gate = ((np.floor(t * 3.0) % 4.0) != 3.0).astype(np.float64)
    envelope = gate * (0.55 + 0.25 * np.sin(2.0 * np.pi * 2.1 * t) ** 2)
    sibilance_bursts = (
        ((np.mod(t, 1.25) >= 0.72) & (np.mod(t, 1.25) <= 0.88)).astype(np.float64)
        if sibilant
        else np.zeros_like(t)
    )
    voiced = (
        0.11 * np.sin(2.0 * np.pi * 140.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.004 * np.sin(2.0 * np.pi * 6500.0 * t)
        + 0.30
        * sibilance_bursts
        * np.sin(2.0 * np.pi * 6500.0 * t)
    )
    noise = noise_amplitude * rng.normal(size=t.size)
    return (level * envelope * voiced + noise).astype(np.float32)


# The native analysis costs about 3 s of test time per second of voice. Six
# seconds is the shortest clip that reaches the same decisions as ten: the
# room-noise and compressor-target reasons are unchanged, while a clip under
# about 4.5 s leaves less than the 2 s of speech Voice Setup vetoes on.
_FULL_ANALYSIS_VOICE_SECONDS = 6.0


def _headroom_reading(headroom_db: float) -> dict:
    return {
        "pre_limiter_true_peak_headroom_db": headroom_db,
        "limiter_gain_reduction_db": max(0.0, -headroom_db),
        "true_peak_limiter_gain_reduction_db": 0.0,
    }


def test_loudness_fit_finds_the_loudest_makeup_that_keeps_headroom():
    def fit(compressor, headroom_at):
        key = "target_lufs" if compressor["auto_makeup_enabled"] else "makeup_gain_db"
        renders = []

        def render(settings):
            renders.append(settings[key])
            return _headroom_reading(headroom_at(settings[key]))

        fitted, reading = _fit_loudness_to_headroom(
            compressor, _headroom_reading(headroom_at(compressor[key])), render
        )
        return fitted[key], reading["pre_limiter_true_peak_headroom_db"], renders

    auto = {"auto_makeup_enabled": True, "target_lufs": -14.0, "makeup_gain_db": 0.0}
    manual = {"auto_makeup_enabled": False, "target_lufs": -18.0, "makeup_gain_db": 6.0}

    # One decibel of headroom per decibel of loudness: 7.14 dB short of the 1 dB
    # the check keeps free, so eight whole steps, confirmed by trying seven.
    assert fit(auto, lambda target: -6.14 - (target + 14.0)) == (-22.0, pytest.approx(1.86), [-22.0, -21.0])
    assert fit(manual, lambda makeup: 4.8 - makeup) == (3.0, pytest.approx(1.8), [3.0, 4.0])
    # Makeup pinned at its cap: nothing changes until the target passes -20 LUFS.
    capped = {**auto, "target_lufs": -18.0}
    assert fit(capped, lambda target: -1.5 + max(0.0, -20.0 - target))[:2] == (-23.0, pytest.approx(1.5))
    # Nothing keeps headroom: quieter and still failing would gain nothing.
    assert fit(capped, lambda target: -2.5)[:2] == (-18.0, -2.5)

    # A request that fits, or one already at the floor, is not rendered again.
    assert fit(auto, lambda target: 1.4) == (-14.0, 1.4, [])
    assert fit({**auto, "target_lufs": -24.0}, lambda target: -3.0) == (-24.0, -3.0, [])
    assert fit({**manual, "makeup_gain_db": 0.0}, lambda makeup: -3.0) == (0.0, -3.0, [])
    unknown = {"simulation_backend": "unavailable"}
    assert _fit_loudness_to_headroom(auto, unknown, lambda settings: unknown) == (auto, unknown)
    assert auto["target_lufs"] == -14.0 and manual["makeup_gain_db"] == 6.0


def test_the_limiter_allowance_accepts_the_peaks_the_limiter_may_take_off():
    allowance = voice_setup_module.LIMITER_ALLOWANCE_DB
    excess = voice_setup_module._limiter_excess_db
    headroom_safe = voice_setup_module._headroom_safe

    assert allowance == 3.0
    assert excess(_headroom_reading(-allowance), allowance) == pytest.approx(0.0)
    assert excess(_headroom_reading(-allowance - 0.5), allowance) == pytest.approx(0.5)
    # The final true-peak stage keeps its own limit.
    busy_true_peak = {**_headroom_reading(2.0), "true_peak_limiter_gain_reduction_db": 0.8}
    assert excess(busy_true_peak, allowance) == pytest.approx(0.3)
    assert excess({"simulation_backend": "unavailable"}, allowance) == float("inf")
    nan_limiter = {**_headroom_reading(2.0), "limiter_gain_reduction_db": float("nan")}
    assert excess(nan_limiter, allowance) == float("inf")

    # Without an allowance this is the released check, reading for reading.
    for headroom in (3.0, 1.0, 0.99, 0.0, -2.5):
        for true_peak in (0.0, 0.5, 0.51):
            reading = {**_headroom_reading(headroom), "true_peak_limiter_gain_reduction_db": true_peak}
            assert headroom_safe(reading, 0.0) == _is_headroom_safe(reading)
    leaning = _headroom_reading(-2.5)
    assert not headroom_safe(leaning, 0.0) and headroom_safe(leaning, allowance)
    assert not headroom_safe(busy_true_peak, allowance)


def test_loudness_fit_with_the_limiter_allowance_lowers_only_as_far_as_it_requires():
    def fit(compressor, headroom_at, allowance_db):
        renders = []

        def render(settings):
            renders.append(settings["target_lufs"])
            return _headroom_reading(headroom_at(settings["target_lufs"]))

        fitted, reading = _fit_loudness_to_headroom(
            compressor, _headroom_reading(headroom_at(compressor["target_lufs"])), render, allowance_db
        )
        return fitted["target_lufs"], reading["pre_limiter_true_peak_headroom_db"], renders

    auto = {"auto_makeup_enabled": True, "target_lufs": -14.0, "makeup_gain_db": 0.0}
    def headroom_at(target):
        return -6.14 - (target + 14.0)

    # The released check needs eight steps; letting the limiter take 3 dB needs four.
    assert fit(auto, headroom_at, 0.0) == (-22.0, pytest.approx(1.86), [-22.0, -21.0])
    assert fit(auto, headroom_at, 3.0) == (-18.0, pytest.approx(-2.14), [-18.0, -17.0])
    # A request within the allowance is left alone and not rendered again.
    assert fit(auto, lambda target: -2.5, 3.0) == (-14.0, -2.5, [])
    # Nothing keeps the peaks within it, so the request stays.
    assert fit({**auto, "target_lufs": -18.0}, lambda target: -4.5, 3.0)[:2] == (-18.0, -4.5)


def test_requested_loudness_is_fitted_to_the_headroom_the_chain_reports(monkeypatch):
    # The chain's reading is made up, so the slow native stages are stubbed.
    # A real voice peaks far above its loudness: every decibel of makeup past
    # 3.3 dB takes a decibel of headroom from the limiter.
    sample_rate = 48_000
    result = _analyze_clean_capture(
        monkeypatch,
        noise=_make_noise(sample_rate),
        speech=_make_voice(sample_rate, seconds=_FULL_ANALYSIS_VOICE_SECONDS),
        headroom_at=lambda compressor: 3.3 - float(compressor["makeup_gain_db"]),
        target_lufs=-16.0,
    )
    diagnostics = result["diagnostics"]
    validation = diagnostics["offline_validation"]

    # The 6 dB of makeup the request needs would lean on the limiter by 2.7 dB.
    assert result["compressor_settings"]["auto_makeup_enabled"] is False
    assert diagnostics["compressor_target_lufs"] == -16.0
    assert diagnostics["requested_loudness_headroom_safe"] is False
    assert diagnostics["loudness_lowered_db"] == 4.0
    assert result["compressor_settings"]["makeup_gain_db"] == 2.0
    assert validation["pre_limiter_true_peak_headroom_db"] == pytest.approx(1.3)
    assert diagnostics["offline_headroom_safe"] is True
    assert "final full-chain headroom validation did not pass" not in diagnostics["uncertainty_reasons"]


def test_a_request_that_already_fits_the_headroom_is_left_alone():
    sample_rate = 48_000
    noise, voice = _make_noise(sample_rate), _make_voice(sample_rate, seconds=_FULL_ANALYSIS_VOICE_SECONDS)
    result = analyze_voice_setup(
        noise, voice, sample_rate, "broadcast", vad_available=False, target_lufs=-20.0,
        raw_noise_audio=noise, raw_speech_audio=voice,
    )
    diagnostics = result["diagnostics"]

    assert diagnostics["requested_loudness_headroom_safe"] is True
    assert diagnostics["loudness_lowered_db"] == 0.0
    assert result["compressor_settings"]["target_lufs"] == -20.0


def _full_voice_candidate_metadata(context_key: str | None = None) -> dict:
    return {
        "scope": "full_voice_setup",
        "allowed_scope": ["eq", "gate", "deesser", "compressor", "limiter"],
        "target": {
            "curve": "broadcast",
            "target_lufs": -18.0,
            "dynamics_intensity": "balanced",
        },
        "capture_identity": {
            "generation": 1,
            "sample_rate": 48_000,
            "noise_samples": 16,
            "voice_samples": 16,
            "context_key": context_key,
        },
        "options": {
            "vad_available": False,
            "dynamics_intensity": "balanced",
            "custom_target_p95_db": 3.5,
            "custom_peak_cap_db": 8.0,
        },
        "verified_stages": [],
    }


def test_short_capture_keeps_short_term_unavailable_and_labels_fallback():
    sample_rate = 48_000
    t = np.arange(sample_rate * 2, dtype=float) / sample_rate
    speech = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    features = _vad_masked_speech_features(speech, sample_rate, -80.0)

    assert features["short_term_window_count"] == 0
    assert features["short_term_lufs_source"] == "momentary_400ms_fallback"
    assert np.isfinite(features["short_term_lufs"])


def test_verification_scores_voice_safe_tone_delta_against_setup_capture():
    freqs = np.geomspace(80.0, 12_000.0, 128)
    reference = -60.0 - 5.0 * np.log2(freqs / 1000.0)
    tone_delta = get_target_curve(
        freqs,
        "broadcast",
        measured_db=reference,
        target_mode="adaptive",
    )
    baseline_error = _shape_error_db(
        freqs,
        reference,
        "broadcast",
        reference_db=reference,
    )
    rendered_error = _shape_error_db(
        freqs,
        reference + tone_delta,
        "broadcast",
        reference_db=reference,
    )

    assert baseline_error > 0.0
    assert rendered_error == pytest.approx(0.0, abs=1.0e-9)


def test_gate_vad_threshold_stays_in_calibrated_narrow_snr_range():
    thresholds = [
        _recommend_gate_settings(
            vad_available=True,
            noise_rms_db=-50.0,
            speech_floor_db=-36.0,
            speech_body_db=-22.0,
            speech_snr_db=snr_db,
            speech_dynamic_range_db=8.0,
        )["vad_threshold"]
        for snr_db in (0.0, 5.0, 10.0, 20.0)
    ]

    np.testing.assert_allclose(thresholds, [0.4725, 0.46625, 0.46, 0.4475])
    assert all(0.42 <= threshold <= 0.50 for threshold in thresholds)


def test_static_microphone_brightness_does_not_trigger_deesser(monkeypatch):
    sample_rate = 48_000
    speech = _make_voice(sample_rate, seconds=5.0)
    t = np.arange(speech.size, dtype=float) / sample_rate
    static_brightness = speech + (
        0.025 * np.sin(2.0 * np.pi * 6500.0 * t)
    ).astype(np.float32)

    # The de-esser is chosen from the speech features before any chain is
    # rendered, so the slow native stages are stubbed.
    result = _analyze_clean_capture(
        monkeypatch, noise=_make_noise(sample_rate), speech=static_brightness
    )

    assert result["deesser_settings"]["enabled"] is False
    assert result["diagnostics"]["deesser_frame_evidence_confidence"] < 0.48


def test_dynamics_intensity_is_separate_from_target_loudness():
    gentle, gentle_diag = _recommend_compressor_settings(
        target_preset="broadcast",
        speech_body_db=-22.0,
        speech_loudness_lufs=-20.0,
        loudness_range_db=5.0,
        speech_snr_db=20.0,
        capture_confidence=0.8,
        dynamics_intensity="gentle",
        custom_target_p95_db=3.5,
        custom_peak_cap_db=8.0,
    )
    dense, dense_diag = _recommend_compressor_settings(
        target_preset="broadcast",
        speech_body_db=-22.0,
        speech_loudness_lufs=-20.0,
        loudness_range_db=5.0,
        speech_snr_db=20.0,
        capture_confidence=0.8,
        dynamics_intensity="dense",
        custom_target_p95_db=3.5,
        custom_peak_cap_db=8.0,
    )

    assert gentle["target_lufs"] == dense["target_lufs"] == -18.0
    assert gentle["ratio"] < dense["ratio"]
    assert (
        gentle_diag["target_p95_reduction_db"]
        < dense_diag["target_p95_reduction_db"]
    )
    assert gentle_diag["peak_reduction_cap_db"] < dense_diag["peak_reduction_cap_db"]


def test_explicit_lufs_target_is_independent_of_tone_curve() -> None:
    common = {
        "speech_body_db": -22.0,
        "speech_loudness_lufs": -20.0,
        "loudness_range_db": 5.0,
        "speech_snr_db": 20.0,
        "capture_confidence": 0.8,
        "dynamics_intensity": "balanced",
        "custom_target_p95_db": 3.5,
        "custom_peak_cap_db": 8.0,
        "target_lufs": -19.0,
    }
    broadcast, _ = _recommend_compressor_settings(
        target_preset="broadcast", **common
    )
    podcast, _ = _recommend_compressor_settings(target_preset="podcast", **common)
    assert broadcast["target_lufs"] == podcast["target_lufs"] == -19.0


def test_custom_dynamics_targets_are_bounded():
    settings, diagnostics = _recommend_compressor_settings(
        target_preset="flat",
        speech_body_db=-24.0,
        speech_loudness_lufs=-21.0,
        loudness_range_db=4.0,
        speech_snr_db=18.0,
        capture_confidence=0.8,
        dynamics_intensity="custom",
        custom_target_p95_db=20.0,
        custom_peak_cap_db=2.0,
    )

    assert diagnostics["target_p95_reduction_db"] == 8.0
    assert diagnostics["peak_reduction_cap_db"] == 8.5
    assert settings["dynamics_intensity"] == "custom"


def test_automatic_makeup_needs_a_reliable_capture_and_a_good_score():
    inputs: dict[str, Any] = dict(
        target_preset="broadcast",
        speech_body_db=-22.0,
        speech_loudness_lufs=-20.0,
        loudness_range_db=5.0,
        speech_snr_db=20.0,
        capture_confidence=0.8,
        dynamics_intensity="balanced",
        custom_target_p95_db=3.5,
        custom_peak_cap_db=8.0,
    )
    default, _ = _recommend_compressor_settings(**inputs)
    unreliable, diagnostics = _recommend_compressor_settings(
        **inputs, capture_reliable=False
    )
    low_score, _ = _recommend_compressor_settings(**{**inputs, "capture_confidence": 0.5})

    assert default["auto_makeup_enabled"] is True
    assert unreliable["auto_makeup_enabled"] is False
    assert diagnostics["auto_makeup_enabled"] is False
    assert unreliable["makeup_gain_db"] == 2.0
    assert low_score["auto_makeup_enabled"] is False


def test_expanded_compressor_search_is_bounded_deterministic_and_improves(
    monkeypatch,
):
    def fake_simulation(_audio, _sample_rate, _eq, chain):
        compressor = chain["compressor"]
        ratio = float(compressor["ratio"])
        attack = float(compressor["attack_ms"])
        release = float(compressor["release_ms"])
        pumping = (
            abs(ratio - 4.0)
            + abs(attack - 10.0) / 10.0
            + abs(release - 180.0) / 100.0
        )

        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": pumping,
            "silence_level_delta_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        fake_simulation,
    )
    compressor = {
        "threshold_db": -24.0,
        "ratio": 2.0,
        "attack_ms": 20.0,
        "release_ms": 300.0,
        "auto_makeup_enabled": True,
        "makeup_gain_db": 0.0,
        "target_lufs": -18.0,
        "measured_short_term_lufs": -22.0,
    }
    eq = {
        "band_freqs": list(np.geomspace(60.0, 16000.0, 10)),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
    }

    def run_search():
        return _calibrate_compressor_expanded(
            speech_audio=np.zeros(4800, dtype=np.float32),
            sample_rate=48000,
            eq_settings=eq,
            deesser_settings={"enabled": False},
            compressor_settings=compressor,
            target_p95_db=3.5,
            target_median_db=1.4,
            peak_cap_db=8.0,
        )

    first, first_diag = run_search()
    second, second_diag = run_search()

    assert first_diag["candidate_count"] <= _COMPRESSOR_SEARCH_BUDGET
    assert first_diag["expanded_search_selected"] is True
    assert first_diag["total_objective"] < first_diag["threshold_only_objective"]
    assert first["ratio"] != compressor["ratio"]
    assert {
        key: first[key]
        for key in ("threshold_db", "ratio", "attack_ms", "release_ms")
    } == {
        key: second[key]
        for key in ("threshold_db", "ratio", "attack_ms", "release_ms")
    }
    assert first_diag["candidate_count"] == second_diag["candidate_count"]


def test_compressor_target_miss_alone_blocks_recommendation(monkeypatch):
    measured_p95 = 2.7
    sample_rate = 48_000
    noise = _make_noise(sample_rate, amplitude=0.0025)
    speech = _make_voice(sample_rate, seconds=5.0)
    eq_settings = {
        "band_freqs": [80, 160, 315, 630, 1250, 2500, 4000, 6300, 10000, 16000],
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "analysis_confidence": 1.0,
        "apply_recommended": True,
    }

    monkeypatch.setattr(
        voice_setup_module,
        "analyze_auto_eq",
        lambda *_args, **_kwargs: (dict(eq_settings), {}),
    )
    monkeypatch.setattr(
        voice_setup_module,
        "tune_gate_suppression_dynamics",
        lambda *args, **_kwargs: {
            "status": "accepted",
            "apply_recommended": True,
            "gate_settings": dict(args[3]),
            "suppressor_settings": {"enabled": True, "strength": 0.75, "model": "rnnoise"},
        },
    )

    def fake_calibration(**kwargs):
        return dict(kwargs["compressor_settings"]), {
            "backend": "rust",
            "target_p95_gain_reduction_db": kwargs["target_p95_db"],
            "target_gain_reduction_db": kwargs["target_p95_db"],
            "measured_p95_gain_reduction_db": measured_p95,
            "measured_gain_reduction_db": measured_p95,
            "target_p95_met": False,
        }

    def fake_simulation(audio, _sample_rate, _eq, chain):
        samples = np.asarray(audio, dtype=np.float32)
        if chain.get("return_auto_makeup_activity"):
            return {
                "simulation_backend": "rust",
                "output_audio": samples.tolist(),
                "auto_makeup_activity": [],
            }
        return {
            "simulation_backend": "rust",
            "output_audio": samples.tolist(),
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "compressor_gain_reduction_db": 3.5,
            "compressor_gain_reduction_p95_db": 3.5,
            "deesser_gain_reduction_db": 0.0,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "limiter_gain_reduction_db": 0.0,
            "true_peak_limiter_gain_reduction_db": 0.0,
        }

    monkeypatch.setattr(
        voice_setup_module,
        "_calibrate_compressor_threshold",
        fake_calibration,
    )
    monkeypatch.setattr(
        voice_setup_module,
        "simulate_candidate_chain",
        fake_simulation,
    )

    missed = analyze_voice_setup(
        noise,
        speech,
        sample_rate,
        "broadcast",
        vad_available=False,
        raw_noise_audio=noise,
        raw_speech_audio=speech,
    )
    assert missed["diagnostics"]["weak_capture"] is False
    assert missed["diagnostics"]["joint_tuning"]["apply_recommended"] is True
    assert missed["suppressor_settings"]["strength"] == 0.75
    assert missed["eq_settings"]["apply_recommended"] is True
    assert missed["diagnostics"]["offline_validation_passed"] is True
    assert missed["diagnostics"]["compressor_calibration"]["target_p95_met"] is False
    assert missed["diagnostics"]["apply_recommended"] is False
    assert any(
        reason == (
            "compressor p95 gain reduction target missed: measured 2.70 dB, "
            "target 3.50 dB (±0.75 dB)"
        )
        for reason in missed["diagnostics"]["uncertainty_reasons"]
    )

    measured_p95 = 3.5
    met = analyze_voice_setup(
        noise,
        speech,
        sample_rate,
        "broadcast",
        vad_available=False,
        raw_noise_audio=noise,
        raw_speech_audio=speech,
    )
    assert met["diagnostics"]["weak_capture"] is False
    assert met["eq_settings"]["apply_recommended"] is True
    assert met["diagnostics"]["offline_validation_passed"] is True
    assert met["diagnostics"]["compressor_calibration"]["target_p95_met"] is True
    assert met["diagnostics"]["apply_recommended"] is True


def _analyze_clean_capture(
    monkeypatch, alter=None, *, noise=None, speech=None, headroom_at=None, **options
):
    """Run a clean capture past stubbed slow stages, optionally altering one input.

    ``headroom_at`` maps the compressor a final-check render uses to the
    headroom it reports, or to the whole safety reading as a dict; ``options``
    go to ``analyze_voice_setup``.
    """
    eq_settings = {
        "band_freqs": [80, 160, 315, 630, 1250, 2500, 4000, 6300, 10000, 16000],
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "analysis_confidence": 1.0,
        "apply_recommended": True,
    }

    def calibration(**kwargs):
        target = kwargs["target_p95_db"]
        return dict(kwargs["compressor_settings"]), {
            "backend": "rust",
            "target_p95_gain_reduction_db": target,
            "target_gain_reduction_db": target,
            "measured_p95_gain_reduction_db": target,
            "measured_gain_reduction_db": target,
        }

    def simulation(audio, _sample_rate, _eq, chain):
        samples = np.asarray(audio, dtype=np.float32)
        if chain.get("return_auto_makeup_activity"):
            return {
                "simulation_backend": "rust",
                "output_audio": samples.tolist(),
                "auto_makeup_activity": [],
            }
        reading = 2.0 if headroom_at is None else headroom_at(chain["compressor"])
        return {
            "simulation_backend": "rust",
            "output_audio": samples.tolist(),
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "compressor_gain_reduction_db": 3.5,
            "compressor_gain_reduction_p95_db": 3.5,
            "deesser_gain_reduction_db": 0.0,
            **(reading if isinstance(reading, dict) else _headroom_reading(reading)),
        }

    monkeypatch.setattr(
        voice_setup_module, "analyze_auto_eq", lambda *_args, **_kwargs: (dict(eq_settings), {})
    )
    monkeypatch.setattr(
        voice_setup_module,
        "tune_gate_suppression_dynamics",
        lambda *args, **_kwargs: {
            "status": "accepted",
            "apply_recommended": True,
            "gate_settings": dict(args[3]),
            "suppressor_settings": {"enabled": True, "strength": 0.75, "model": "rnnoise"},
        },
    )
    monkeypatch.setattr(voice_setup_module, "_calibrate_compressor_threshold", calibration)
    monkeypatch.setattr(voice_setup_module, "simulate_candidate_chain", simulation)
    if alter is not None:
        name, change = alter
        real = getattr(voice_setup_module, name)
        monkeypatch.setattr(
            voice_setup_module, name, lambda *args, **kwargs: change(real(*args, **kwargs))
        )

    sample_rate = 48_000
    if noise is None:
        noise = _make_noise(sample_rate, amplitude=0.0025)
    if speech is None:
        speech = _make_voice(sample_rate, seconds=5.0)
    return analyze_voice_setup(
        noise,
        speech,
        sample_rate,
        "broadcast",
        vad_available=False,
        raw_noise_audio=noise,
        raw_speech_audio=speech,
        **options,
    )


def test_a_reliable_capture_gets_automatic_makeup_and_can_be_applied(monkeypatch):
    result = _analyze_clean_capture(monkeypatch)
    diagnostics = result["diagnostics"]

    assert diagnostics["weak_capture"] is False
    assert diagnostics["apply_recommended"] is True
    assert result["compressor_settings"]["auto_makeup_enabled"] is True
    assert diagnostics["uncertainty_reasons"] == []


@pytest.mark.parametrize(
    ("analysis_input", "veto", "reason"),
    [
        (
            "analyze_noise_reference",
            lambda reference: replace(
                reference, status="questionable", reasons=["room noise drifts"]
            ),
            "room noise drifts",
        ),
        (
            "analyze_noise_reference",
            lambda reference: replace(
                reference, status="invalid", reasons=["room noise is clipped"]
            ),
            "room noise is clipped",
        ),
        (
            "analyze_voice_spectrum",
            lambda spectrum: replace(spectrum, used_single_spectrum_fallback=True),
            "too few voiced frames for a stable spectrum",
        ),
        (
            "_vad_masked_speech_features",
            lambda features: {**features, "active_duration_s": 1.5},
            "too little VAD-active speech",
        ),
    ],
    ids=["questionable-noise", "invalid-noise", "single-spectrum", "short-speech"],
)
def test_a_vetoed_capture_is_weak_and_names_its_real_reason(
    monkeypatch, analysis_input, veto, reason
):
    result = _analyze_clean_capture(monkeypatch, (analysis_input, veto))
    diagnostics = result["diagnostics"]
    reasons = diagnostics["uncertainty_reasons"]

    # The veto decides on its own; the score is no longer rewritten to sit under it.
    assert diagnostics["capture_confidence"] >= 0.55
    assert diagnostics["weak_capture"] is True
    assert diagnostics["apply_recommended"] is False
    assert result["compressor_settings"]["auto_makeup_enabled"] is False
    assert reason in reasons
    assert "spectral feature stability is weak" not in reasons
    assert "overall capture quality is low" not in reasons


def test_a_low_score_alone_is_reported_as_low_overall_quality(monkeypatch):
    result = _analyze_clean_capture(
        monkeypatch,
        ("analyze_voice_spectrum", lambda spectrum: replace(spectrum, residual_confidence=0.0)),
    )
    diagnostics = result["diagnostics"]

    assert diagnostics["capture_confidence"] < 0.50
    assert diagnostics["weak_capture"] is True
    assert diagnostics["apply_recommended"] is False
    assert "overall capture quality is low" in diagnostics["uncertainty_reasons"]
    assert "spectral feature stability is weak" not in diagnostics["uncertainty_reasons"]


@pytest.mark.parametrize(
    ("snr_db", "reliable", "weak"),
    [(5.9, False, True), (6.0, True, True), (7.9, True, True), (8.0, True, False)],
    ids=["under-6", "at-6", "under-8", "at-8"],
)
def test_the_noise_referenced_snr_sets_reliability_weakness_and_the_confidence_cap(
    monkeypatch, snr_db, reliable, weak
):
    # Near 6 dB the SNR term drags the geometric-mean capture score under the 0.55
    # makeup floor, so pin that score (the one weighted 0.30 on spectral confidence)
    # and let the SNR alone decide reliability.
    real_score = voice_setup_module._bounded_quality_score
    monkeypatch.setattr(
        voice_setup_module,
        "_bounded_quality_score",
        lambda parts: 0.9 if parts[0][1] == 0.30 else real_score(parts),
    )
    result = _analyze_clean_capture(
        monkeypatch, ("analyze_voice_spectrum", lambda spectrum: replace(spectrum, snr_db=snr_db))
    )
    diagnostics = result["diagnostics"]

    assert result["compressor_settings"]["auto_makeup_enabled"] is reliable
    assert diagnostics["weak_capture"] is weak
    assert ("speech-to-noise ratio is weak" in diagnostics["uncertainty_reasons"]) is weak
    # A weak capture sits exactly on the cap; a strong one is free to score above it.
    if weak:
        assert diagnostics["setup_confidence"] == pytest.approx(0.49)
    else:
        assert diagnostics["setup_confidence"] > 0.49


def _auto_eq_abstains(*reasons):
    return "analyze_auto_eq", lambda out: (
        {**out[0], "apply_recommended": False, "abstention_reasons": list(reasons)},
        out[1],
    )


def _auto_eq_refuses(*, tone_only):
    def refuse(_out):
        from mic_eq.analysis.failure_detection import AnalysisValidationError

        raise AnalysisValidationError(
            "Recording too unclear. Please try again.", ["clipped_gains (7 >= 6)"], tone_only
        )

    return "analyze_auto_eq", refuse


@pytest.mark.parametrize(
    "auto_eq",
    [
        _auto_eq_abstains("no validated correction improved the target safely"),
        _auto_eq_refuses(tone_only=True),
    ],
    ids=["abstention", "validation"],
)
@pytest.mark.usefixtures("partial_results")
def test_a_withheld_tone_leaves_a_clean_setup_recommended(monkeypatch, auto_eq):
    result = _analyze_clean_capture(monkeypatch, auto_eq)
    diagnostics = result["diagnostics"]

    assert diagnostics["apply_recommended"] is True
    assert diagnostics["uncertainty_reasons"] == []
    assert diagnostics["tone_withheld"]["reason_code"] == "tone_withheld"
    assert result["eq_settings"]["apply_recommended"] is False
    assert _candidate_settings_error(result) is None


@pytest.mark.parametrize(
    "auto_eq",
    [_auto_eq_abstains("capture quality score is too low"), _auto_eq_refuses(tone_only=False)],
    ids=["abstention", "validation"],
)
@pytest.mark.usefixtures("partial_results")
def test_a_refusal_the_tone_does_not_explain_still_blocks_the_recommendation(monkeypatch, auto_eq):
    result = _analyze_clean_capture(monkeypatch, auto_eq)
    diagnostics = result["diagnostics"]

    assert diagnostics["apply_recommended"] is False
    assert "Auto-EQ abstained from this capture" in diagnostics["uncertainty_reasons"]
    assert diagnostics["tone_withheld"] is None


@pytest.mark.skipif(not mic_eq.CORE_AVAILABLE, reason="native extension is not built")
@pytest.mark.usefixtures("partial_results")
def test_a_withheld_tone_is_calibrated_and_checked_natively_on_the_users_eq(monkeypatch):
    from mic_eq.analysis.failure_detection import AnalysisValidationError
    from mic_eq.config import EQSettings, Preset

    def refuse(*_args, **_kwargs):
        raise AnalysisValidationError(
            "Recording too unclear. Please try again.", ["clipped_gains (7 >= 6)"], True
        )

    renders, calibrations = [], []
    render, calibrate = (
        voice_setup_module.simulate_candidate_chain,
        voice_setup_module._calibrate_compressor_threshold,
    )

    def recording_render(audio, rate, eq, chain):
        renders.append(eq)
        return render(audio, rate, eq, chain)

    def recording_calibration(**kwargs):
        calibrations.append(kwargs["eq_settings"])
        return calibrate(**kwargs)

    monkeypatch.setattr(voice_setup_module, "analyze_auto_eq", refuse)
    monkeypatch.setattr(voice_setup_module, "simulate_candidate_chain", recording_render)
    monkeypatch.setattr(voice_setup_module, "_calibrate_compressor_threshold", recording_calibration)
    current = Preset().eq.to_dict()
    for band in current["bands"]:
        band["gain_db"] = 9.0
    users_eq = EQSettings.from_dict(current).to_dict()
    sample_rate = 48_000
    noise = _make_noise(sample_rate, amplitude=0.0025)
    speech = _make_voice(sample_rate, seconds=5.0)

    result = analyze_voice_setup(
        noise,
        speech,
        sample_rate,
        "broadcast",
        vad_available=False,
        raw_noise_audio=noise,
        raw_speech_audio=speech,
        incumbent_settings={"eq": current},
    )
    diagnostics = result["diagnostics"]

    assert calibrations == [users_eq]
    assert len(renders) >= 2
    assert all(eq == {**users_eq, "enabled": eq["enabled"]} for eq in renders)
    assert diagnostics["compressor_calibration"]["backend"] == "rust"
    assert diagnostics["offline_validation"]["simulation_backend"] == "rust"
    assert diagnostics["tone_withheld"]["reason_code"] == "tone_withheld"
    assert result["eq_settings"] == {**users_eq, "apply_recommended": False}


def test_partial_results_are_off_by_default():
    assert voice_setup_module.PARTIAL_RESULTS is False


@pytest.mark.parametrize(
    ("auto_eq", "complete"),
    [
        (_auto_eq_abstains("no validated correction improved the target safely"), True),
        (_auto_eq_refuses(tone_only=True), False),
    ],
    ids=["abstention", "validation"],
)
def test_by_default_a_tone_only_refusal_refuses_the_whole_setup(monkeypatch, auto_eq, complete):
    result = _analyze_clean_capture(monkeypatch, auto_eq)
    diagnostics = result["diagnostics"]

    assert diagnostics["tone_withheld"] is None
    assert diagnostics["apply_recommended"] is False
    assert "Auto-EQ abstained from this capture" in diagnostics["uncertainty_reasons"]
    assert (result["eq_settings"] is not None) is complete
    assert (result["eq_settings"] or {}).get("bands") is None


@pytest.mark.usefixtures("partial_results")
def test_input_gain_advice_still_comes_with_a_withheld_tone(monkeypatch):
    result = _analyze_clean_capture(
        monkeypatch,
        _auto_eq_abstains("headroom validation removed the usable correction"),
        speech=_make_voice(48_000, seconds=5.0, level=1.34),
        headroom_at=lambda compressor: -(compressor["target_lufs"] + 24.0),
    )

    assert result["diagnostics"]["apply_recommended"] is False
    assert (
        "the microphone level leaves too little headroom; lower the input gain by about 1 dB"
        in result["diagnostics"]["uncertainty_reasons"]
    )


@pytest.mark.parametrize("auto_makeup", [True, False], ids=["automatic-makeup", "manual-makeup"])
def test_input_gain_advice_is_the_shortfall_left_at_the_loudness_floor(monkeypatch, auto_makeup):
    def headroom_at(compressor):
        # A decibel of headroom per decibel of loudness; the floor is 1 dB short.
        auto = compressor["auto_makeup_enabled"]
        return -(compressor["target_lufs"] + 24.0 if auto else compressor["makeup_gain_db"])

    result = _analyze_clean_capture(
        monkeypatch,
        None if auto_makeup else ("_vad_masked_speech_features", lambda f: {**f, "active_duration_s": 1.5}),
        speech=_make_voice(48_000, seconds=5.0, level=1.34),
        headroom_at=headroom_at,
    )
    diagnostics = result["diagnostics"]

    assert result["compressor_settings"]["auto_makeup_enabled"] is auto_makeup
    assert diagnostics["short_term_lufs"] == pytest.approx(-22.0, abs=0.5)
    # Nothing fits, so the request stays; it is 5 to 7 dB short while the floor is 1 dB short.
    assert diagnostics["loudness_lowered_db"] == 0.0
    assert diagnostics["offline_validation"]["pre_limiter_true_peak_headroom_db"] <= -4.0
    assert diagnostics["apply_recommended"] is False
    assert (
        "the microphone level leaves too little headroom; lower the input gain by about 1 dB"
        in diagnostics["uncertainty_reasons"]
    )


def _safety_reading(headroom_db, limiter_db, true_peak_db):
    return {
        "pre_limiter_true_peak_headroom_db": headroom_db,
        "limiter_gain_reduction_db": limiter_db,
        "true_peak_limiter_gain_reduction_db": true_peak_db,
    }


@pytest.mark.parametrize(
    ("floor_reading", "alter", "reason"),
    [
        # The advice is the term of the released check that is furthest past its limit.
        (
            _safety_reading(-0.5, 0.5, 0.0),
            None,
            "the microphone level leaves too little headroom; lower the input gain by about 2 dB",
        ),
        (
            _safety_reading(3.0, 2.3, 0.0),
            None,
            "the microphone level leaves too little headroom; lower the input gain by about 2 dB",
        ),
        (
            _safety_reading(3.0, 0.5, 1.2),
            None,
            "the microphone level leaves too little headroom; lower the input gain by about 1 dB",
        ),
        # With no finite, positive shortfall there is no amount to give.
        (
            None,
            (
                "_fit_loudness_to_headroom",
                lambda fitted: (fitted[0], {**fitted[1], **_headroom_reading(-3.0)}),
            ),
            "final full-chain headroom validation did not pass",
        ),
        (
            _safety_reading(float("nan"), 0.0, 0.0),
            None,
            "final full-chain headroom validation did not pass",
        ),
        (
            _safety_reading(3.0, float("nan"), 1.2),
            None,
            "final full-chain headroom validation did not pass",
        ),
    ],
    ids=[
        "headroom-term",
        "limiter-term",
        "true-peak-term",
        "nothing-short-at-the-floor",
        "non-finite-headroom",
        "non-finite-limiter-with-true-peak-over",
    ],
)
def test_input_gain_advice_follows_the_term_of_the_released_check_that_fails(
    monkeypatch, floor_reading, alter, reason
):
    result = _analyze_clean_capture(
        monkeypatch,
        alter,
        speech=_make_voice(48_000, seconds=5.0, level=1.34),
        headroom_at=None if floor_reading is None else lambda _compressor: floor_reading,
    )
    diagnostics = result["diagnostics"]

    assert diagnostics["short_term_lufs"] >= -24.0
    assert diagnostics["apply_recommended"] is False
    assert [text for text in diagnostics["uncertainty_reasons"] if "headroom" in text] == [reason]


def test_cancelling_during_the_loudness_fit_cancels_the_analysis(monkeypatch):
    rendered = []

    def headroom_at(compressor):
        rendered.append(compressor)
        return -6.0

    # The request is validated first; the fit renders after it, which is where this cancels.
    with pytest.raises(AnalysisCancelled):
        _analyze_clean_capture(
            monkeypatch, headroom_at=headroom_at, cancel_check=lambda: bool(rendered)
        )
    assert len(rendered) == 1


def _leaning_on_the_limiter(over_db):
    """Headroom of a render whose request presses the limiter ``over_db``; a dB of loudness is a dB of it."""

    def headroom_at(compressor):
        if compressor["auto_makeup_enabled"]:
            loudness = compressor["target_lufs"] + 18.0
        else:
            loudness = compressor["makeup_gain_db"] - 6.0
        return -over_db - loudness

    return headroom_at


@pytest.mark.parametrize(
    ("over_db", "lowered_db"),
    [(2.0, 0.0), (3.0, 0.0), (4.0, 1.0)],
    ids=["within-the-allowance", "at-the-allowance", "past-the-allowance"],
)
def test_the_limiter_may_take_peaks_off_a_proposal_automatic_makeup_raises(
    monkeypatch, over_db, lowered_db
):
    result = _analyze_clean_capture(monkeypatch, headroom_at=_leaning_on_the_limiter(over_db))
    diagnostics = result["diagnostics"]

    assert diagnostics["short_term_lufs"] < -24.0
    assert result["compressor_settings"]["auto_makeup_enabled"] is True
    assert result["limiter_settings"]["enabled"] is True
    assert diagnostics["limiter_allowance_db"] == 3.0
    # The released check would lower the target by over_db + 1 dB; the allowance, by what is left past 3 dB.
    assert diagnostics["requested_loudness_headroom_safe"] is (lowered_db == 0.0)
    assert diagnostics["loudness_lowered_db"] == lowered_db
    assert result["compressor_settings"]["target_lufs"] == -18.0 - lowered_db
    assert diagnostics["offline_headroom_safe"] is True
    assert diagnostics["apply_recommended"] is True
    assert diagnostics["uncertainty_reasons"] == []


@pytest.mark.parametrize(
    "case",
    [
        {
            "limiter_settings": {
                "enabled": False,
                "ceiling_db": -2.5,
                "release_ms": 175.0,
                "careful_output_enabled": False,
            }
        },
        {"alter": ("_vad_masked_speech_features", lambda f: {**f, "active_duration_s": 1.5})},
        {"speech": _make_voice(48_000, seconds=5.0, level=1.34)},
    ],
    ids=["limiter-disabled", "manual-makeup", "voice-too-hot-for-makeup"],
)
def test_the_released_headroom_rule_holds_where_the_limiter_allowance_does_not_apply(
    monkeypatch, case
):
    result = _analyze_clean_capture(monkeypatch, headroom_at=_leaning_on_the_limiter(2.0), **case)
    diagnostics = result["diagnostics"]

    assert diagnostics["limiter_allowance_db"] == 0.0
    # 2 dB of limiter gain reduction is within the allowance but not the released check.
    assert diagnostics["requested_loudness_headroom_safe"] is False
    assert diagnostics["loudness_lowered_db"] == 3.0
    assert diagnostics["offline_validation"]["pre_limiter_true_peak_headroom_db"] == pytest.approx(1.0)
    assert diagnostics["offline_headroom_safe"] is True
    # Each case switches off exactly one of the allowance's three conditions.
    assert sum(
        (
            diagnostics["short_term_lufs"] >= -24.0,
            not result["compressor_settings"]["auto_makeup_enabled"],
            not result["limiter_settings"]["enabled"],
        )
    ) == 1


def test_a_digitally_silent_room_reference_is_weak_but_the_analysis_completes(monkeypatch):
    from mic_eq.analysis.noise_reference import SILENT_REFERENCE_REASON

    result = _analyze_clean_capture(monkeypatch, noise=np.zeros(96_000, dtype=np.float32))
    diagnostics = result["diagnostics"]

    assert diagnostics["weak_capture"] is True
    assert diagnostics["apply_recommended"] is False
    assert result["compressor_settings"]["auto_makeup_enabled"] is False
    assert SILENT_REFERENCE_REASON in diagnostics["uncertainty_reasons"]
    assert diagnostics["noise_reference_quality"]["status"] == "questionable"


def test_production_compressor_search_keeps_nonthreshold_controls(monkeypatch):
    observed_controls: set[tuple[float, float, float]] = set()
    observed_thresholds: set[float] = set()
    simulation_count = 0

    def fake_simulation(_audio, _sample_rate, _eq, chain):
        nonlocal simulation_count
        simulation_count += 1
        compressor = chain["compressor"]
        observed_thresholds.add(float(compressor["threshold_db"]))
        observed_controls.add(
            (
                float(compressor["ratio"]),
                float(compressor["attack_ms"]),
                float(compressor["release_ms"]),
            )
        )
        # An expanded candidate could improve this objective, but the
        # production default must evaluate only threshold variants.
        pumping = (
            abs(float(compressor["ratio"]) - 4.0)
            + abs(float(compressor["attack_ms"]) - 10.0) / 10.0
            + abs(float(compressor["release_ms"]) - 180.0) / 100.0
        )
        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": pumping,
            "silence_level_delta_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        fake_simulation,
    )
    compressor = {
        "threshold_db": -24.0,
        "ratio": 2.0,
        "attack_ms": 20.0,
        "release_ms": 300.0,
        "auto_makeup_enabled": True,
        "makeup_gain_db": 0.0,
        "target_lufs": -18.0,
        "measured_short_term_lufs": -22.0,
    }

    calibrated, diagnostics = _calibrate_compressor_threshold(
        speech_audio=np.zeros(4800, dtype=np.float32),
        sample_rate=48000,
        eq_settings={
            "band_freqs": list(np.geomspace(60.0, 16000.0, 10)),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        deesser_settings={"enabled": False},
        compressor_settings=compressor,
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8.0,
    )

    assert observed_controls == {(2.0, 20.0, 300.0)}
    assert observed_thresholds == {
        -24.0,
        *(float(value) for value in np.linspace(-55.0, -6.0, 33)),
    }
    assert simulation_count == 35  # 34 fixed threshold candidates plus winner recheck.
    assert diagnostics["candidate_count"] == 35
    assert (calibrated["ratio"], calibrated["attack_ms"], calibrated["release_ms"]) == (
        2.0,
        20.0,
        300.0,
    )
    assert diagnostics["expanded_search_selected"] is False
    assert diagnostics["expanded_search_status"] == "disabled_unqualified"
    assert diagnostics["expanded_candidate_objective"] is None


def test_production_compressor_calibration_cannot_enable_expanded_search(monkeypatch):
    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        lambda *_args: {"simulation_backend": "unavailable"},
    )
    unqualified_option: dict[str, Any] = {"allow_expanded_search": True}
    with pytest.raises(TypeError, match="allow_expanded_search"):
        _calibrate_compressor_threshold(
            speech_audio=np.zeros(4800, dtype=np.float32),
            sample_rate=48_000,
            eq_settings={},
            deesser_settings={"enabled": False},
            compressor_settings={
                "threshold_db": -24.0,
                "ratio": 3.0,
                "attack_ms": 10.0,
                "release_ms": 180.0,
            },
            target_p95_db=3.5,
            target_median_db=1.4,
            peak_cap_db=8.0,
            **unqualified_option,
        )


def test_compressor_search_requires_measured_quiet_level(monkeypatch):
    all_quiet_unavailable = False

    def fake_simulation(_audio, _sample_rate, _eq, chain):
        compressor = chain["compressor"]
        threshold = float(compressor["threshold_db"])
        quiet_delta = (
            None if all_quiet_unavailable or threshold == -24.0 else 0.0
        )
        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": 0.0,
            "silence_level_delta_db": quiet_delta,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        fake_simulation,
    )
    compressor = {
        "threshold_db": -24.0,
        "ratio": 2.0,
        "attack_ms": 20.0,
        "release_ms": 300.0,
        "auto_makeup_enabled": True,
        "makeup_gain_db": 0.0,
        "target_lufs": -18.0,
        "measured_short_term_lufs": -22.0,
    }

    def search():
        return _calibrate_compressor_threshold(
            speech_audio=np.zeros(4800, dtype=np.float32),
            sample_rate=48000,
            eq_settings={
                "band_freqs": list(np.geomspace(60.0, 16000.0, 10)),
                "band_gains": [0.0] * 10,
                "band_qs": [1.41] * 10,
            },
            deesser_settings={"enabled": False},
            compressor_settings=compressor,
            target_p95_db=3.5,
            target_median_db=1.4,
            peak_cap_db=8.0,
        )

    measured_winner, measured_diag = search()
    assert measured_winner["threshold_db"] != -24.0
    assert measured_diag["silence_level_delta_db"] == 0.0

    all_quiet_unavailable = True
    unchanged, unavailable_diag = search()
    assert unchanged == compressor
    assert unavailable_diag["backend"] == "unavailable"
    assert unavailable_diag["silence_level_delta_db"] is None


def test_compressor_search_evaluator_keeps_unmeasurable_capture_failed(
    monkeypatch, tmp_path
):
    import json

    from tools import evaluate_compressor_search

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "captures": [
                    {
                        "split": "held_out",
                        "sample_rate": 48_000,
                        "condition": "clean_48k",
                        "path": "capture.wav",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluate_compressor_search,
        "_load_audio",
        lambda _path: (48_000, np.zeros(4800, dtype=np.float32)),
    )
    monkeypatch.setattr(
        evaluate_compressor_search,
        "_calibrate_compressor_expanded",
        lambda **_kwargs: (
            dict(evaluate_compressor_search.INCUMBENT),
            {
                "backend": "unavailable",
                "expanded_search_status": "experimental_opt_in",
                "expanded_search_selected": False,
                "silence_level_delta_db": None,
            },
        ),
    )

    report = evaluate_compressor_search.evaluate(manifest_path, limit=1)

    row = report["rows"][0]
    assert report["capture_count"] == 1
    assert report["measurement_complete"] is False
    assert report["measurable_capture_count"] == 0
    assert report["unmeasurable_capture_count"] == 1
    assert report["median_relative_improvement"] is None
    assert report["improved_fraction"] == 0.0
    assert report["safety_passed"] is False
    assert report["retained"] is False
    assert row["calibration_backend"] == "unavailable"
    assert row["relative_improvement"] is None
    assert row["threshold_only_objective"] is None
    assert row["expanded_objective"] is None
    assert row["silence_level_delta_db"] is None
    assert row["candidate_count"] is None


def test_expanded_compressor_search_keeps_safe_profile_on_effective_tie(
    monkeypatch,
):
    def tied_simulation(_audio, _sample_rate, _eq, _chain):
        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": 0.0,
            "silence_level_delta_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        tied_simulation,
    )
    compressor = {
        "threshold_db": -24.0,
        "ratio": 2.5,
        "attack_ms": 12.0,
        "release_ms": 180.0,
        "auto_makeup_enabled": True,
        "makeup_gain_db": 0.0,
        "target_lufs": -18.0,
        "measured_short_term_lufs": -22.0,
    }
    calibrated, diagnostics = _calibrate_compressor_expanded(
        speech_audio=np.zeros(4800, dtype=np.float32),
        sample_rate=48000,
        eq_settings={
            "band_freqs": list(np.geomspace(60.0, 16000.0, 10)),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        deesser_settings={"enabled": False},
        compressor_settings=compressor,
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8.0,
    )

    assert diagnostics["expanded_search_selected"] is False
    assert calibrated["ratio"] == compressor["ratio"]
    assert calibrated["attack_ms"] == compressor["attack_ms"]
    assert calibrated["release_ms"] == compressor["release_ms"]


def test_compressor_calibration_uses_one_limiter_configuration(monkeypatch):
    observed_limiters: list[dict[str, object]] = []
    observed_compressors: list[dict[str, object]] = []

    def fake_simulation(_audio, _sample_rate, _eq, chain):
        observed_limiters.append(dict(chain["limiter"]))
        observed_compressors.append(dict(chain["compressor"]))
        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": 0.0,
            "silence_level_delta_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        fake_simulation,
    )
    limiter = {
        "enabled": False,
        "ceiling_db": -2.5,
        "release_ms": 175.0,
        "careful_output_enabled": False,
    }
    compressor = {
        "threshold_db": -24.0,
        "ratio": 2.5,
        "attack_ms": 12.0,
        "release_ms": 180.0,
        "auto_makeup_enabled": True,
        "makeup_gain_db": 0.0,
        "target_lufs": -18.0,
        "measured_short_term_lufs": -22.0,
    }

    _calibrate_compressor_threshold(
        speech_audio=np.zeros(4800, dtype=np.float32),
        sample_rate=48000,
        eq_settings={
            "band_freqs": list(np.geomspace(60.0, 16000.0, 10)),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        deesser_settings={"enabled": False},
        compressor_settings=compressor,
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8.0,
        limiter_settings=limiter,
    )

    assert observed_limiters
    assert all(item == limiter for item in observed_limiters)
    assert observed_compressors[-1]["auto_makeup_enabled"] is True


def test_verification_uses_candidate_limiter_and_retries_without_it(monkeypatch):
    observed_limiters: list[dict[str, object]] = []
    limiter_gain_reduction_db = 0.0
    true_peak_limiter_gain_reduction_db = 0.0

    class FakeSpectrum:
        freqs = np.geomspace(80.0, 12_000.0, 16)
        median_spectrum_db = np.zeros(16)
        spectral_snr_db = np.zeros(16)
        snr_db = 0.0

    def fake_simulation(audio, _sample_rate, _eq, chain):
        observed_limiters.append(dict(chain["limiter"]))
        return {
            "simulation_backend": "rust",
            "output_audio": np.zeros_like(audio).tolist(),
            "compressor_gain_reduction_p95_db": 1.0,
            "compressor_gain_reduction_db": 1.0,
            "deesser_gain_reduction_p95_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            # A single event without measured limiter gain is a harmless
            # boundary catch and must not force a reduction loop.
            "true_peak_limited_events": 1,
            "limiter_gain_reduction_db": limiter_gain_reduction_db,
            "true_peak_limiter_gain_reduction_db": (
                true_peak_limiter_gain_reduction_db
            ),
            "output_rms_db": -30.0,
            "input_rms_db": -30.0,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.simulate_candidate_chain",
        fake_simulation,
    )
    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.analyze_voice_spectrum",
        lambda *_args, **_kwargs: FakeSpectrum(),
    )
    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup._shape_error_db",
        lambda *_args, **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup._vad_masked_speech_features",
        lambda *_args, **_kwargs: {"active_loudness_spread_db": 0.0},
    )
    limiter = {
        "enabled": False,
        "ceiling_db": -2.5,
        "release_ms": 175.0,
        "careful_output_enabled": False,
    }
    setup_result = {
        "eq_settings": {
            "band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        "deesser_settings": {"enabled": False},
        "compressor_settings": {
            "target_p95_reduction_db": 3.5,
            "peak_reduction_cap_db": 8.0,
        },
        "limiter_settings": limiter,
    }
    audio = np.zeros(48_000 * 3, dtype=np.float32)

    result = validate_voice_setup_verification(
        audio,
        audio,
        audio,
        48_000,
        setup_result,
        "broadcast",
    )

    assert result["decision"] == "accept"
    assert result["reason_codes"] == ["accepted"]
    assert result["reduction_targets"] == []
    assert result["limiter_activity_events"] == 1

    true_peak_limiter_gain_reduction_db = 1.0
    boundary = validate_voice_setup_verification(
        audio,
        audio,
        audio,
        48_000,
        setup_result,
        "broadcast",
    )
    assert boundary["decision"] == "accept"
    assert boundary["limiter_pressure_db"] == pytest.approx(1.0)
    assert boundary["reduction_targets"] == []

    true_peak_limiter_gain_reduction_db = 1.25
    pressure = validate_voice_setup_verification(
        audio,
        audio,
        audio,
        48_000,
        setup_result,
        "broadcast",
    )
    assert pressure["decision"] == "reduce"
    assert pressure["reason_codes"] == ["limiter_excess"]
    assert pressure["reduction_targets"] == ["limiter"]
    assert pressure["limiter_pressure_db"] == pytest.approx(1.25)

    delivery_retry = validate_voice_setup_verification(
        audio,
        audio,
        np.full_like(audio, 0.1),
        48_000,
        setup_result,
        "broadcast",
    )
    assert delivery_retry["decision"] == "retry"
    assert delivery_retry["reason_codes"] == ["verification_delivery_mismatch"]
    assert "delivery level delta" in delivery_retry["reasons"][0]

    assert observed_limiters
    assert all(item == limiter for item in observed_limiters)
    missing = dict(setup_result)
    missing.pop("limiter_settings")
    retry = validate_voice_setup_verification(
        audio,
        audio,
        audio,
        48_000,
        missing,
        "broadcast",
    )
    assert retry["decision"] == "rollback"
    assert retry["reason_codes"] == ["candidate_settings_incomplete"]
    assert retry["reasons"] == ["candidate limiter settings are missing"]


def _verify_with_limiter_reading(monkeypatch, diagnostics, limiter_db, true_peak_db):
    """Verify a setup against a render whose limiter stages report the given gain reduction."""

    class FakeSpectrum:
        freqs = np.geomspace(80.0, 12_000.0, 16)
        median_spectrum_db = np.zeros(16)
        spectral_snr_db = np.zeros(16)
        snr_db = 0.0

    def fake_simulation(audio, _sample_rate, _eq, _chain):
        return {
            "simulation_backend": "rust",
            "output_audio": np.zeros_like(audio).tolist(),
            "compressor_gain_reduction_p95_db": 1.0,
            "compressor_gain_reduction_db": 1.0,
            "deesser_gain_reduction_p95_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "limiter_gain_reduction_db": limiter_db,
            "true_peak_limiter_gain_reduction_db": true_peak_db,
            "output_rms_db": -30.0,
            "input_rms_db": -30.0,
        }

    monkeypatch.setattr("mic_eq.analysis.voice_setup.simulate_candidate_chain", fake_simulation)
    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.analyze_voice_spectrum", lambda *_args, **_kwargs: FakeSpectrum()
    )
    monkeypatch.setattr("mic_eq.analysis.voice_setup._shape_error_db", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup._vad_masked_speech_features",
        lambda *_args, **_kwargs: {"active_loudness_spread_db": 0.0},
    )
    setup_result = {
        "eq_settings": {
            "band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        "deesser_settings": {"enabled": False},
        "compressor_settings": {"target_p95_reduction_db": 3.5, "peak_reduction_cap_db": 8.0},
        "limiter_settings": {
            "enabled": True,
            "ceiling_db": -2.5,
            "release_ms": 175.0,
            "careful_output_enabled": False,
        },
        **({} if diagnostics is None else {"diagnostics": diagnostics}),
    }
    audio = np.zeros(48_000 * 3, dtype=np.float32)

    return validate_voice_setup_verification(
        audio, audio, audio, 48_000, setup_result, "broadcast"
    )


@pytest.mark.parametrize(
    ("diagnostics", "limiter_db", "true_peak_db", "targets"),
    [
        # A setup that recorded the allowance may press the limiter up to it...
        ({"limiter_allowance_db": 3.0}, 3.0, 0.0, []),
        ({"limiter_allowance_db": 3.0}, 1.5, 0.0, []),
        ({"limiter_allowance_db": 3.0}, 3.25, 0.0, ["limiter"]),
        # ...but its true-peak stage keeps the limit it always had...
        ({"limiter_allowance_db": 3.0}, 0.0, 1.0, []),
        ({"limiter_allowance_db": 3.0}, 0.0, 1.25, ["limiter"]),
        # ...and any other setup result keeps the released rule.
        ({"limiter_allowance_db": 0.0}, 1.5, 0.0, ["limiter"]),
        ({}, 1.5, 0.0, ["limiter"]),
        (None, 1.5, 0.0, ["limiter"]),
        (None, 1.0, 0.0, []),
    ],
    ids=[
        "at-the-allowance",
        "within-the-allowance",
        "past-the-allowance",
        "true-peak-at-its-limit",
        "true-peak-past-its-limit",
        "allowance-not-applied",
        "field-absent",
        "no-diagnostics",
        "released-limit",
    ],
)
def test_verification_leans_on_the_limiter_allowance_only_when_the_setup_recorded_it(
    monkeypatch, diagnostics, limiter_db, true_peak_db, targets
):
    result = _verify_with_limiter_reading(monkeypatch, diagnostics, limiter_db, true_peak_db)

    assert result["reduction_targets"] == targets
    assert result["decision"] == ("reduce" if targets else "accept")


@pytest.mark.parametrize(
    ("diagnostics", "limiter_db", "true_peak_db", "reason"),
    [
        # With the allowance, the stage that fired is named, not the limiter's allowed figure.
        ({"limiter_allowance_db": 3.0}, 2.0, 1.3, "true-peak limiter pressure reached 1.3 dB"),
        ({"limiter_allowance_db": 3.0}, 3.5, 0.0, "limiter pressure reached 3.5 dB"),
        (
            {"limiter_allowance_db": 3.0},
            3.5,
            1.3,
            "limiter pressure reached 3.5 dB and true-peak limiter pressure reached 1.3 dB",
        ),
        # The released rule names its stage the same way.
        (None, 0.5, 1.3, "true-peak limiter pressure reached 1.3 dB"),
        (None, 1.5, 0.0, "limiter pressure reached 1.5 dB"),
        (
            None,
            1.5,
            1.3,
            "limiter pressure reached 1.5 dB and true-peak limiter pressure reached 1.3 dB",
        ),
    ],
    ids=[
        "allowance-true-peak-only",
        "allowance-limiter-only",
        "allowance-both",
        "released-true-peak-only",
        "released-limiter-only",
        "released-both",
    ],
)
def test_verification_names_the_limiter_stage_that_pressed(
    monkeypatch, diagnostics, limiter_db, true_peak_db, reason
):
    result = _verify_with_limiter_reading(monkeypatch, diagnostics, limiter_db, true_peak_db)

    assert result["decision"] == "reduce"
    assert result["reason_codes"] == ["limiter_excess"]
    assert result["reduction_targets"] == ["limiter"]
    assert result["reasons"] == [reason]
    assert result["limiter_pressure_db"] == pytest.approx(max(limiter_db, true_peak_db))


def test_verification_checks_cancellation_between_rendering_stages(monkeypatch):
    calls = 0
    cancelled = False

    def fake_simulation(audio, _sample_rate, _eq, _chain):
        nonlocal calls, cancelled
        calls += 1
        cancelled = calls == 2
        return {
            "simulation_backend": "rust",
            "output_audio": np.zeros_like(audio).tolist(),
        }

    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.simulate_candidate_chain",
        fake_simulation,
    )
    with pytest.raises(AnalysisCancelled):
        validate_voice_setup_verification(
            np.zeros(144_000, dtype=np.float32),
            np.zeros(144_000, dtype=np.float32),
            np.zeros(144_000, dtype=np.float32),
            48_000,
            {
                "eq_settings": {
                    "band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
                    "band_gains": [0.0] * 10,
                    "band_qs": [1.41] * 10,
                },
                "limiter_settings": {
                    "enabled": True,
                    "ceiling_db": -0.5,
                    "release_ms": 50.0,
                    "careful_output_enabled": True,
                },
            },
            "broadcast",
            cancel_check=lambda: cancelled,
        )
    assert calls == 2


def test_verification_worker_cancellation_does_not_emit_result(qapp, monkeypatch):
    entered = threading.Event()
    result_ready = []

    def blocking_validation(*_args, cancel_check=None, **_kwargs):
        if not callable(cancel_check):
            raise AssertionError("worker did not provide a cancellation callback")
        entered.set()
        while not cancel_check():
            time.sleep(0.005)
        raise AnalysisCancelled("sentinel")

    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.validate_voice_setup_verification",
        blocking_validation,
    )
    worker = VoiceSetupVerificationWorker(
        np.zeros(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        48_000,
        {"limiter_settings": {
            "enabled": True,
            "ceiling_db": -0.5,
            "release_ms": 50.0,
            "careful_output_enabled": True,
        }},
        "broadcast",
    )
    worker.result_ready.connect(result_ready.append)
    worker.start()
    assert entered.wait(1.0)
    worker.stop()
    assert worker.wait(2_000)
    qapp.processEvents()
    assert result_ready == []


def test_verification_worker_reports_failure(monkeypatch):
    errors: list[str] = []

    def failing_validation(*_args, **_kwargs):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.validate_voice_setup_verification",
        failing_validation,
    )
    worker = VoiceSetupVerificationWorker(
        np.zeros(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        48_000,
        {"limiter_settings": {
            "enabled": True,
            "ceiling_db": -0.5,
            "release_ms": 50.0,
            "careful_output_enabled": True,
        }},
        "broadcast",
    )
    worker.failed.connect(errors.append)
    worker.run()
    assert errors == ["sentinel"]


def test_candidate_uses_live_controls_and_restores_on_failure_and_close(qapp, monkeypatch):
    from unittest.mock import Mock
    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow
    from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    owner = MainWindow()
    monkeypatch.setattr(owner, "_calibration_context_key", lambda: "test-route")
    owner.meter_timer.stop()
    owner.diagnostics_timer.stop()
    owner.compressor_state.set_settings({"noise_reference_reliability": 0.7})
    owner.eq_state.set_settings({"enabled": False})
    old_compressor = owner.compressor_state.get_settings(include_calibration=True)
    old_limiter = owner.limiter_state.get_settings()
    old_eq = owner.eq_state.get_settings()
    dialog = VoiceSetupDialog(parent=owner)
    limiter = {"enabled": False, "ceiling_db": -2.5, "release_ms": 175.0,
               "careful_output_enabled": False}
    dialog.setup_result = {
        "_candidate": _full_voice_candidate_metadata("test-route"),
        "diagnostics": {"apply_recommended": True},
        "gate_settings": owner.gate_state.get_settings(),
        "deesser_settings": {**owner.deesser_state.get_settings(), "auto_amount": 0.3476},
        "compressor_settings": {**old_compressor, "threshold_db": -21.123,
            "ratio": 2.3476, "noise_reference_reliability": 0.4,
            "target_p95_reduction_db": 3.5, "peak_reduction_cap_db": 8.0},
        "limiter_settings": limiter,
        "eq_settings": {"band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
                        "band_gains": [1.0] * 10, "band_qs": [1.41] * 10},
    }
    dialog._candidate_metadata = deepcopy(dialog.setup_result["_candidate"])
    dialog.noise_audio = dialog.voice_audio = np.zeros(16, dtype=np.float32)
    dialog._analysis_generation = 1
    with monkeypatch.context() as patch:
        patch.setattr(dialog, "_show_summary", lambda _result: None)
        dialog._on_analysis_complete(dialog.setup_result)
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.critical",
        Mock(side_effect=AssertionError("unexpected voice setup error dialog")),
    )
    assert not dialog.curve_combo.isEnabled()
    assert not dialog.dynamics_combo.isEnabled()
    dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData("podcast"))
    dialog.dynamics_combo.setCurrentIndex(dialog.dynamics_combo.findData("dense"))
    dialog.target_lufs_spin.setValue(-14.0)
    dialog._apply_setup()
    assert dialog.setup_state == "verification_ready"
    assert dialog.setup_result["_candidate"]["target"] == {
        "curve": "broadcast",
        "target_lufs": -18.0,
        "dynamics_intensity": "balanced",
    }
    assert dialog.setup_result["_candidate"]["options"]["dynamics_intensity"] == (
        "balanced"
    )
    assert dialog.setup_result["compressor_settings"]["threshold_db"] == -21.123
    assert dialog.setup_result["compressor_settings"]["ratio"] == 2.3476
    assert dialog.setup_result["compressor_settings"]["target_p95_reduction_db"] == 3.5
    assert dialog.setup_result["deesser_settings"]["auto_amount"] == 0.3476
    assert dialog.setup_result["limiter_settings"] == limiter
    assert owner.eq_state.get_settings()["enabled"] is True
    dialog._on_verification_failed("sentinel failure")
    assert owner.compressor_state.get_settings(include_calibration=True) == old_compressor
    assert owner.limiter_state.get_settings() == old_limiter
    assert owner.eq_state.get_settings() == old_eq

    with monkeypatch.context() as patch:
        show_error = Mock()
        patch.setattr("mic_eq.ui.voice_setup_dialog.QMessageBox.critical", show_error)
        patch.setattr(owner.eq_state, "set_settings",
                      Mock(side_effect=RuntimeError("sentinel apply failure")))
        dialog._apply_setup()
        assert show_error.call_count == 1
    assert owner.compressor_state.get_settings(include_calibration=True) == old_compressor
    assert owner.limiter_state.get_settings() == old_limiter
    assert owner.eq_state.get_settings() == old_eq

    entered = threading.Event()
    def blocked_verification(*_args, cancel_check, **_kwargs):
        entered.set()
        while not cancel_check():
            time.sleep(0.005)
        raise AnalysisCancelled("closed")
    monkeypatch.setattr("mic_eq.ui.voice_setup_dialog.validate_voice_setup_verification",
                        blocked_verification)
    dialog._apply_setup()
    dialog.noise_audio = dialog.voice_audio = np.zeros(16, dtype=np.float32)
    dialog._complete_verification(np.zeros(16, dtype=np.float32))
    worker = dialog.analysis_worker
    assert worker is not None
    try:
        assert entered.wait(2.0)
        # Deliver an event while verification is blocked, then close cooperatively.
        events = []
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: events.append("responsive"))
        qapp.processEvents()
        assert events == ["responsive"]
        stale_generation = dialog._analysis_generation
        dialog.reject()
        assert worker.wait(2_000)
        dialog._on_verification_complete({"decision": "accept"}, generation=stale_generation)
        assert "verification" not in dialog.setup_result
        assert owner.compressor_state.get_settings(include_calibration=True) == old_compressor
        assert owner.limiter_state.get_settings() == old_limiter
        assert owner.eq_state.get_settings() == old_eq
    finally:
        worker.stop()
        worker.wait(2_000)
        qapp.processEvents()
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        owner.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("eq_settings", {"band_freqs": [60.0]}, "candidate EQ bands are incomplete"),
        ("limiter_settings", {}, "candidate limiter settings are incomplete"),
    ],
)
def test_incomplete_candidate_offers_retake_instead_of_apply(
    qapp, monkeypatch, field, value, reason
):
    from unittest.mock import Mock

    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow
    from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    owner = MainWindow()
    owner.meter_timer.stop()
    owner.diagnostics_timer.stop()
    dialog = VoiceSetupDialog(parent=owner)
    limiter = {
        "enabled": False,
        "ceiling_db": -2.5,
        "release_ms": 175.0,
        "careful_output_enabled": False,
    }
    result = {
        "_candidate": _full_voice_candidate_metadata(),
        "diagnostics": {
            "apply_recommended": False,
            "uncertainty_reasons": ["capture confidence is weak"],
            "setup_confidence": 0.62,
            "capture_confidence": 0.7,
            "recommendation_uncertainty": 0.4,
            "gate_mode_label": "VAD Assisted",
        },
        "gate_settings": owner.gate_state.get_settings(),
        "deesser_settings": owner.deesser_state.get_settings(),
        "compressor_settings": owner.compressor_state.get_settings(
            include_calibration=True
        ),
        "limiter_settings": limiter,
        "eq_settings": {
            "band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
            "band_gains": [1.0] * 10,
            "band_qs": [1.41] * 10,
        },
    }
    result[field] = value
    dialog._candidate_metadata = deepcopy(result["_candidate"])
    dialog.noise_audio = dialog.voice_audio = np.zeros(16, dtype=np.float32)
    dialog._analysis_generation = 1
    try:
        dialog._on_analysis_complete(result)
        assert dialog.setup_state == "noise_ready"
        assert dialog.start_button.text() == "Record Voice Again"
        assert not dialog.retake_btn.isHidden()
        assert dialog.warning_label.text().startswith("The recommendations are incomplete.")
        assert "Part of the result is missing." in dialog.warning_label.text()
        assert "Apply" not in dialog.start_button.text()

        show_error = Mock()
        monkeypatch.setattr("mic_eq.ui.voice_setup_dialog.QMessageBox.critical", show_error)
        dialog._apply_setup()
        show_error.assert_called_once()
        assert show_error.call_args.args[1] == "Incomplete Voice Setup"
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        owner.deleteLater()
        qapp.processEvents()


def test_complete_advisory_candidate_still_offers_apply(qapp, monkeypatch):
    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow
    from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    owner = MainWindow()
    owner.meter_timer.stop()
    owner.diagnostics_timer.stop()
    dialog = VoiceSetupDialog(parent=owner)
    try:
        result = {
            "_candidate": _full_voice_candidate_metadata(),
            "diagnostics": {
                "apply_recommended": False,
                "uncertainty_reasons": ["capture confidence is weak"],
                "setup_confidence": 0.62,
                "capture_confidence": 0.7,
                "recommendation_uncertainty": 0.4,
                "gate_mode_label": "VAD Assisted",
            },
            "gate_settings": owner.gate_state.get_settings(),
            "deesser_settings": owner.deesser_state.get_settings(),
            "compressor_settings": owner.compressor_state.get_settings(
                include_calibration=True
            ),
            "limiter_settings": {
                "enabled": False,
                "ceiling_db": -2.5,
                "release_ms": 175.0,
                "careful_output_enabled": False,
            },
            "eq_settings": {
                "band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
                "band_gains": [1.0] * 10,
                "band_qs": [1.41] * 10,
            },
        }
        dialog._candidate_metadata = deepcopy(result["_candidate"])
        dialog.noise_audio = dialog.voice_audio = np.zeros(16, dtype=np.float32)
        dialog._analysis_generation = 1
        dialog._on_analysis_complete(result)
        assert dialog.setup_state == "completed"
        assert dialog.start_button.text() == "Apply Suggestions"
        assert not dialog.curve_combo.isEnabled()
        assert not dialog.dynamics_combo.isEnabled()
        assert dialog.warning_label.text().startswith(
            "These settings did not pass every check, so treat them as suggestions."
        )
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        owner.deleteLater()
        qapp.processEvents()


def test_review_shows_settings_before_and_after(qapp, monkeypatch):
    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow
    from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    owner = MainWindow()
    owner.meter_timer.stop()
    owner.diagnostics_timer.stop()
    dialog = VoiceSetupDialog(parent=owner)
    orphan = VoiceSetupDialog()
    try:
        gate = owner.gate_state.get_settings()
        before_db = gate["threshold_db"]
        gate["threshold_db"] = before_db - 6.0
        compressor = owner.compressor_state.get_settings()
        compressor["attack_ms"] += 5.0
        result = {
            "diagnostics": {
                "setup_confidence": 0.9,
                "capture_confidence": 0.9,
                "recommendation_uncertainty": 0.1,
                "gate_mode_label": "VAD Assisted",
            },
            "gate_settings": gate,
            "deesser_settings": owner.deesser_state.get_settings(),
            "compressor_settings": compressor,
            "eq_settings": None,
        }
        dialog._show_summary(result)
        recommended, before = dialog.gate_label.text().split("\nBefore: ")
        assert recommended.endswith(f"opens above {before_db - 6.0:.1f} dB")
        assert before.endswith(f"opens above {before_db:.1f} dB")
        assert dialog.deesser_label.text().endswith("(unchanged)")
        assert dialog.compressor_label.text().endswith("(only finer settings change)")

        orphan._show_summary(result)
        assert orphan.gate_label.text() == f"Noise gate: {recommended.split(': ', 1)[1]}"
        assert "unchanged" not in orphan.deesser_label.text()

        # A loudness target the capture's peaks cannot carry is lowered, and said so.
        result["diagnostics"] |= {"loudness_lowered_db": 4.0, "compressor_target_lufs": -18.0}
        compressor |= {"auto_makeup_enabled": True, "target_lufs": -22.0}
        orphan._show_summary(result)
        assert orphan.compressor_label.text().endswith(
            "aiming for -22 LUFS\nYou asked for -18 LUFS; "
            "it was lowered so peaks do not overload the limiter."
        )
        compressor |= {"auto_makeup_enabled": False, "target_lufs": -18.0}
        orphan._show_summary(result)
        assert orphan.compressor_label.text().endswith(
            "\nMakeup was held 4 dB lower so peaks do not overload the limiter."
        )
        # Makeup does not fall in whole decibels once it reaches zero.
        result["diagnostics"]["loudness_lowered_db"] = 2.5
        orphan._show_summary(result)
        assert orphan.compressor_label.text().endswith(
            "\nMakeup was held 2.5 dB lower so peaks do not overload the limiter."
        )
    finally:
        for widget in (dialog, orphan):
            widget.reject()
            widget.deleteLater()
        owner.close()
        owner.deleteLater()
        qapp.processEvents()


def test_expanded_compressor_search_handles_no_safe_threshold_only_candidate(
    monkeypatch,
):
    incumbent = {
        "ratio": 2.5,
        "attack_ms": 12.0,
        "release_ms": 180.0,
    }

    def simulation_with_rejected_threshold_only(_audio, _sample_rate, _eq, chain):
        compressor = chain["compressor"]
        threshold_only = all(
            abs(float(compressor[key]) - value) <= 1.0e-6
            for key, value in incumbent.items()
        )
        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": 0.0 if threshold_only else -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": 0.0,
            "silence_level_delta_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.compressor_calibration.simulate_candidate_chain",
        simulation_with_rejected_threshold_only,
    )
    calibrated, diagnostics = _calibrate_compressor_expanded(
        speech_audio=np.zeros(4800, dtype=np.float32),
        sample_rate=48000,
        eq_settings={
            "band_freqs": list(np.geomspace(60.0, 16000.0, 10)),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        deesser_settings={"enabled": False},
        compressor_settings={
            "threshold_db": -24.0,
            **incumbent,
            "auto_makeup_enabled": True,
            "makeup_gain_db": 0.0,
            "target_lufs": -18.0,
            "measured_short_term_lufs": -22.0,
        },
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8.0,
    )

    assert diagnostics["expanded_search_selected"] is True
    assert diagnostics["threshold_only_objective"] == float("inf")
    assert any(
        calibrated[key] != value
        for key, value in incumbent.items()
    )


def test_native_chain_reports_robust_reduction_and_can_return_rendered_audio():
    sample_rate = 48_000
    audio = _make_voice(sample_rate, seconds=1.0)
    simulation = simulate_candidate_chain(
        audio,
        sample_rate,
        {
            "band_freqs": [80, 160, 315, 630, 1250, 2500, 4000, 6300, 10000, 16000],
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        {
            "compressor": {
                "enabled": True,
                "threshold_db": -24.0,
                "ratio": 3.0,
            },
            "limiter": {"enabled": False},
            "return_output_audio": True,
        },
    )

    assert simulation["simulation_backend"] == "rust"
    assert len(simulation["output_audio"]) == audio.size
    assert simulation["silence_level_delta_db"] is None or np.isfinite(simulation["silence_level_delta_db"])
    assert simulation["analysis_block_ms"] == 20.0
    assert simulation["active_analysis_block_count"] > 0
    assert (
        simulation["compressor_gain_reduction_median_db"]
        <= simulation["compressor_gain_reduction_p95_db"]
        <= simulation["compressor_gain_reduction_db"] + 1e-6
    )


def test_silent_room_noise_moves_on_and_a_silent_voice_is_the_fault(qapp, monkeypatch):
    # A headset that mutes itself between words records silence as room noise.
    from mic_eq.analysis.noise_reference import SILENT_REFERENCE_GUIDANCE
    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow
    from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    owner = MainWindow()
    owner.meter_timer.stop()
    owner.diagnostics_timer.stop()
    dialog = VoiceSetupDialog(parent=owner)
    analyses = []
    monkeypatch.setattr(dialog, "_start_analysis", lambda: analyses.append(1))
    try:
        dialog.setup_state = "noise_recording"
        dialog._on_recording_complete(np.zeros(96_000, dtype=np.float32))
        assert dialog.setup_state == "noise_ready"
        assert dialog.noise_audio is not None
        assert dialog.phase_label.text() == "Room noise captured"
        assert dialog.start_button.text() == "Record Voice"
        assert dialog.warning_label.text().startswith(SILENT_REFERENCE_GUIDANCE)

        # A leaky gate is unsteady on every take, so it is used as a weak reference.
        leaky_gate = _make_noise(48_000)
        for start in range(0, leaky_gate.size, 19_200):
            leaky_gate[start : start + 9_600] = 0.0
        dialog.setup_state = "noise_recording"
        dialog._on_recording_complete(leaky_gate)
        assert dialog.setup_state == "noise_ready"
        assert "Setup will play safe" in dialog.warning_label.text()

        # A capture the analysis cannot use at all still has to be retaken.
        dialog.setup_state = "noise_recording"
        dialog._on_recording_complete(_make_noise(48_000, seconds=0.5))
        assert dialog.setup_state == "idle" and dialog.noise_audio is None
        assert dialog.phase_label.text() == "Room-noise capture rejected"
        assert dialog.start_button.text() == "Retake Room Noise"

        # Clipping is refused by the analysis, so it is caught at the recording.
        clipped = _make_noise(48_000)
        clipped[::200] = 1.0
        dialog.setup_state = "noise_recording"
        dialog._on_recording_complete(clipped)
        assert dialog.setup_state == "idle" and dialog.noise_audio is None
        assert dialog.warning_label.text().startswith("A loud sound was picked up")

        dialog.setup_state = "noise_recording"
        dialog._on_recording_complete(np.zeros(96_000, dtype=np.float32))
        loud_voice = _make_voice(48_000)
        loud_voice[1000:1006] = 1.0
        dialog.setup_state = "voice_recording"
        dialog._on_recording_complete(loud_voice)
        assert dialog.setup_state == "noise_ready" and not analyses
        assert dialog.warning_label.text().startswith("The recording clipped:")
        dialog.setup_state = "voice_recording"
        dialog._on_recording_complete(np.zeros(480_000, dtype=np.float32))
        assert dialog.setup_state == "noise_ready"
        assert dialog.voice_audio is None and not analyses
        assert dialog.warning_label.text().startswith("No sound came from the microphone.")
        assert dialog.start_button.text() == "Record Voice"

        dialog.setup_state = "voice_recording"
        dialog._on_recording_complete(_make_voice(48_000))
        assert dialog.setup_state == "analyzing" and analyses == [1]
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        owner.deleteLater()
        qapp.processEvents()


@pytest.fixture
def voice_dialog(qapp, monkeypatch):
    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow
    from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    owner = MainWindow()
    owner.meter_timer.stop()
    owner.diagnostics_timer.stop()
    dialog = VoiceSetupDialog(parent=owner)
    monkeypatch.setattr(dialog, "_start_analysis", lambda: None)
    yield dialog
    dialog.reject()
    dialog.deleteLater()
    owner.close()
    owner.deleteLater()
    qapp.processEvents()


def _finish_take(dialog, step, audio):
    """Finish a take as the recording poll does, with the labels it leaves behind."""
    dialog.setup_state = step
    dialog.phase_label.setText("Capturing")
    dialog.progress_bar.setValue(100)
    dialog.time_label.setText("Complete!")
    dialog._on_recording_complete(audio)


def _poked(audio, index, value):
    audio = audio.copy()
    audio[index] = value
    return audio


_CLIPPED = (
    "The recording clipped: its peaks were held at full scale. "
    "Lower the microphone level by about 6 dB (in Windows or on the microphone) or "
)
_LOUD_ROOM = (
    "A loud sound was picked up in the room-noise recording. "
    "Record the room noise again while the room is quiet."
)
_UNREADABLE = (
    "The recording held invalid audio data, which usually means the audio "
    "driver glitched. Restart processing and record again."
)
_TOO_SHORT = "Record at least 1.5 seconds of room tone."


@pytest.mark.parametrize(
    ("take", "message"),
    [
        (
            lambda: _poked(_make_noise(48_000), slice(None, None, 200), 1.0),
            _LOUD_ROOM,
        ),
        (lambda: _poked(_make_noise(48_000), 100, np.nan), _UNREADABLE),
        (lambda: _poked(_make_noise(48_000), 100, np.inf), _UNREADABLE),
        (lambda: _make_noise(48_000, seconds=0.5), _TOO_SHORT),
        (
            lambda: np.zeros(0, dtype=np.float32),
            _TOO_SHORT + " Restart the audio stream and record the room tone again.",
        ),
        # Silence is no reason to read out the headset advice when the take is too short.
        (lambda: np.zeros(24_000, dtype=np.float32), _TOO_SHORT),
    ],
    ids=["clipped", "nan", "infinite", "too-short", "empty", "silent-and-too-short"],
)
def test_a_rejected_room_noise_take_asks_for_a_retake(voice_dialog, take, message):
    _finish_take(voice_dialog, "noise_recording", take())

    assert voice_dialog.warning_label.text() == message
    assert voice_dialog.setup_state == "idle"
    assert voice_dialog.noise_audio is None and voice_dialog.preview_noise_audio is None
    assert voice_dialog.phase_label.text() == "Room-noise capture rejected"
    assert voice_dialog.start_button.text() == "Retake Room Noise"
    assert voice_dialog.time_label.text() == "Ready to record again"
    assert voice_dialog.progress_bar.value() == 0


@pytest.mark.parametrize(
    ("take", "message"),
    [
        (
            lambda: _poked(_make_voice(48_000), slice(1000, 1006), 1.0),
            _CLIPPED + "move back a little, then record again.",
        ),
        (lambda: _poked(_make_voice(48_000), 1000, np.nan), _UNREADABLE),
        (
            lambda: np.zeros(480_000, dtype=np.float32),
            "No sound came from the microphone. Check that it is not muted and "
            "that the right input is selected, then record again.",
        ),
    ],
    ids=["clipped", "nan", "silent"],
)
def test_a_rejected_voice_take_asks_for_another_voice_take(voice_dialog, take, message):
    _finish_take(voice_dialog, "noise_recording", np.zeros(96_000, dtype=np.float32))
    assert voice_dialog.setup_state == "noise_ready"

    _finish_take(voice_dialog, "voice_recording", take())

    assert voice_dialog.warning_label.text() == message
    assert voice_dialog.setup_state == "noise_ready"
    assert voice_dialog.noise_audio is not None
    assert voice_dialog.voice_audio is None and voice_dialog.preview_voice_audio is None
    assert voice_dialog.phase_label.text() == "Voice capture rejected"
    assert voice_dialog.start_button.text() == "Record Voice"
    assert voice_dialog.time_label.text() == "Ready to record again"
    assert voice_dialog.progress_bar.value() == 0
