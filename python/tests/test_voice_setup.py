"""Tests for auto voice setup analysis."""

from __future__ import annotations

from copy import deepcopy
import threading
import time

import numpy as np
import pytest

from mic_eq.analysis.voice_setup import (
    _COMPRESSOR_SEARCH_BUDGET,
    _calibrate_compressor_threshold,
    _recommend_compressor_settings,
    _recommend_gate_settings,
    _vad_masked_speech_features,
    analyze_voice_setup,
    validate_voice_setup_verification,
)
from mic_eq.analysis.auto_eq import simulate_candidate_chain
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.ui.voice_setup_dialog import VoiceSetupVerificationWorker


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


def test_voice_setup_uses_vad_assisted_when_available():
    sample_rate = 48_000
    limiter = {
        "enabled": False,
        "ceiling_db": -2.5,
        "release_ms": 175.0,
        "careful_output_enabled": False,
    }
    result = analyze_voice_setup(
        _make_noise(sample_rate),
        _make_voice(sample_rate),
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


def test_labelled_fixture_recommendations_use_loudness_features_and_offline_dsp():
    sample_rate = 48_000
    fixtures = [
        (
            "clean",
            _make_noise(sample_rate, amplitude=0.0025),
            _make_voice(sample_rate, seconds=5.0),
            None,
        ),
        (
            "sibilant",
            _make_noise(sample_rate, amplitude=0.0025),
            _make_voice(sample_rate, seconds=5.0, sibilant=True),
            None,
        ),
        (
            "weak_noisy",
            _make_noise(sample_rate, amplitude=0.012),
            _make_voice(
                sample_rate,
                seconds=5.0,
                level=0.04,
                noise_amplitude=0.012,
            ),
            False,
        ),
    ]

    fixture_results = {}
    for label, noise, speech, expected_apply in fixtures:
        result = analyze_voice_setup(
            noise,
            speech,
            sample_rate,
            "broadcast",
            vad_available=False,
        )
        fixture_results[label] = result
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
        if label != "weak_noisy":
            assert calibration["backend"] == "rust", label
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

    sibilant = fixture_results["sibilant"]
    assert sibilant["deesser_settings"]["enabled"] is True
    assert (
        sibilant["deesser_settings"]["high_cut_hz"]
        > sibilant["deesser_settings"]["low_cut_hz"]
    )
    assert sibilant["diagnostics"]["deesser_temporal_contrast_db"] > 0.75
    assert (
        fixture_results["sibilant"]["diagnostics"]["offline_validation"][
            "deesser_gain_reduction_db"
        ]
        > 0.25
    )
    assert fixture_results["weak_noisy"]["diagnostics"]["weak_capture"] is True
    if not fixture_results["clean"]["eq_settings"]["apply_recommended"]:
        assert fixture_results["clean"]["eq_settings"]["abstention_reasons"]
    assert fixture_results["clean"]["diagnostics"]["noise_reference_source"] == (
        "validated_conservative"
    )
    assert fixture_results["clean"]["eq_settings"]["snr_reference_available"] is True

    verification = validate_voice_setup_verification(
        _make_noise(sample_rate, amplitude=0.0025),
        _make_voice(sample_rate, seconds=5.0, sibilant=True),
        _make_voice(sample_rate, seconds=5.0, sibilant=True),
        sample_rate,
        fixture_results["sibilant"],
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


def test_static_microphone_brightness_does_not_trigger_deesser():
    sample_rate = 48_000
    speech = _make_voice(sample_rate, seconds=5.0)
    t = np.arange(speech.size, dtype=float) / sample_rate
    static_brightness = speech + (
        0.025 * np.sin(2.0 * np.pi * 6500.0 * t)
    ).astype(np.float32)

    result = analyze_voice_setup(
        _make_noise(sample_rate),
        static_brightness,
        sample_rate,
        "broadcast",
        vad_available=False,
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
            "silence_output_gain_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.simulate_candidate_chain",
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
        return _calibrate_compressor_threshold(
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
            "silence_output_gain_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.simulate_candidate_chain",
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

    assert diagnostics["expanded_search_selected"] is False
    assert calibrated["ratio"] == compressor["ratio"]
    assert calibrated["attack_ms"] == compressor["attack_ms"]
    assert calibrated["release_ms"] == compressor["release_ms"]


def test_compressor_calibration_uses_one_limiter_configuration(monkeypatch):
    observed_limiters: list[dict[str, object]] = []

    def fake_simulation(_audio, _sample_rate, _eq, chain):
        observed_limiters.append(dict(chain["limiter"]))
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
            "silence_output_gain_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.simulate_candidate_chain",
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


def test_verification_uses_candidate_limiter_and_retries_without_it(monkeypatch):
    observed_limiters: list[dict[str, object]] = []

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
            "true_peak_limited_events": 0,
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
    assert retry["decision"] == "retry"
    assert retry["reasons"] == ["candidate limiter settings are missing"]


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
    owner.compressor_panel.set_compressor_settings({"noise_reference_reliability": 0.7})
    owner.eq_panel.set_settings({"enabled": False})
    old_compressor = owner.compressor_panel.get_compressor_settings(include_calibration=True)
    old_limiter = owner.compressor_panel.get_limiter_settings()
    old_eq = owner.eq_panel.get_settings()
    dialog = VoiceSetupDialog(parent=owner)
    limiter = {"enabled": False, "ceiling_db": -2.5, "release_ms": 175.0,
               "careful_output_enabled": False}
    dialog.setup_result = {
        "_candidate": _full_voice_candidate_metadata("test-route"),
        "diagnostics": {"apply_recommended": True},
        "gate_settings": owner.gate_panel.get_settings(),
        "deesser_settings": {**owner.deesser_panel.get_settings(), "auto_amount": 0.3476},
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
    assert dialog.setup_result["compressor_settings"]["threshold_db"] == -21.12
    assert dialog.setup_result["compressor_settings"]["ratio"] == 2.35
    assert dialog.setup_result["compressor_settings"]["target_p95_reduction_db"] == 3.5
    assert dialog.setup_result["deesser_settings"]["auto_amount"] == 0.35
    assert dialog.setup_result["limiter_settings"] == limiter
    assert owner.eq_panel.get_settings()["enabled"] is True
    dialog._on_verification_failed("sentinel failure")
    assert owner.compressor_panel.get_compressor_settings(include_calibration=True) == old_compressor
    assert owner.compressor_panel.get_limiter_settings() == old_limiter
    assert owner.eq_panel.get_settings() == old_eq

    with monkeypatch.context() as patch:
        show_error = Mock()
        patch.setattr("mic_eq.ui.voice_setup_dialog.QMessageBox.critical", show_error)
        patch.setattr(owner.eq_panel, "set_settings",
                      Mock(side_effect=RuntimeError("sentinel apply failure")))
        dialog._apply_setup()
        assert show_error.call_count == 1
    assert owner.compressor_panel.get_compressor_settings(include_calibration=True) == old_compressor
    assert owner.compressor_panel.get_limiter_settings() == old_limiter
    assert owner.eq_panel.get_settings() == old_eq

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
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: events.append("responsive"))
        qapp.processEvents()
        assert events == ["responsive"]
        stale_generation = dialog._analysis_generation
        dialog.reject()
        assert worker.wait(2_000)
        dialog._on_verification_complete({"decision": "accept"}, generation=stale_generation)
        assert "verification" not in dialog.setup_result
        assert owner.compressor_panel.get_compressor_settings(include_calibration=True) == old_compressor
        assert owner.compressor_panel.get_limiter_settings() == old_limiter
        assert owner.eq_panel.get_settings() == old_eq
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
        "gate_settings": owner.gate_panel.get_settings(),
        "deesser_settings": owner.deesser_panel.get_settings(),
        "compressor_settings": owner.compressor_panel.get_compressor_settings(
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
        assert dialog.warning_label.text().startswith("Recommendations are incomplete:")
        assert reason in dialog.warning_label.text()
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
            "gate_settings": owner.gate_panel.get_settings(),
            "deesser_settings": owner.deesser_panel.get_settings(),
            "compressor_settings": owner.compressor_panel.get_compressor_settings(
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
        assert dialog.start_button.text() == "Apply Voice Setup"
        assert not dialog.curve_combo.isEnabled()
        assert not dialog.dynamics_combo.isEnabled()
        assert dialog.warning_label.text().startswith("Advisory recommendations only:")
    finally:
        dialog.reject()
        dialog.deleteLater()
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
            "silence_output_gain_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(
        "mic_eq.analysis.voice_setup.simulate_candidate_chain",
        simulation_with_rejected_threshold_only,
    )
    calibrated, diagnostics = _calibrate_compressor_threshold(
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
    assert simulation["silence_output_gain_db"] <= 0.0
    assert np.isfinite(simulation["silence_level_delta_db"])
    assert simulation["analysis_block_ms"] == 20.0
    assert simulation["active_analysis_block_count"] > 0
    assert (
        simulation["compressor_gain_reduction_median_db"]
        <= simulation["compressor_gain_reduction_p95_db"]
        <= simulation["compressor_gain_reduction_db"] + 1e-6
    )
