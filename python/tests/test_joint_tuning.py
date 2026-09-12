"""Joint selection must keep exact incumbent settings when evidence is weak."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis import joint_tuning
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.config import Preset
from mic_eq.mic_eq_core import simulate_gate_suppressor_order
from tools.evaluate_product_tuning import _selected_settings


def _tuning_args() -> tuple[Any, ...]:
    preset = Preset()
    return (
        np.full(48_000, 0.001, dtype=np.float32),
        np.full(48_000, 0.1, dtype=np.float32),
        48_000,
        asdict(preset.gate),
        asdict(preset.compressor),
        {
            "band_freqs": preset.eq.band_freqs,
            "band_gains": preset.eq.band_gains,
            "band_qs": preset.eq.band_qs,
        },
        asdict(preset.deesser),
        asdict(preset.limiter),
    )


def _downstream_metrics(audio: np.ndarray, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
    del audio
    return {
        "simulation_backend": "rust",
        "compressor_gain_reduction_p95_db": 3.5,
        "compressor_gain_reduction_median_db": 1.47,
        "compressor_gain_reduction_db": 4.0,
        "compressor_pumping_score_db": 0.0,
        "output_true_peak_db": -1.0,
        "limiter_effective_ceiling_db": -0.5,
        "pre_limiter_true_peak_headroom_db": 1.0,
        "non_finite_output": False,
        "candidate_runtime_ms": 0.0,
    }


def test_native_gate_enabled_matches_requested_control():
    audio = np.full(4800, 0.01, dtype=np.float32)
    muted = simulate_gate_suppressor_order(audio, [0.0] * 10, False, 0.0,
                                          {"gate_enabled": True, "gate_threshold_db": -10.0})
    open_gate = simulate_gate_suppressor_order(audio, [0.0] * 10, False, 0.0,
                                             {"gate_enabled": False})
    assert np.max(np.abs(muted["output_audio"])) < 0.001
    assert np.max(np.abs(open_gate["output_audio"])) > 0.005


def test_joint_unavailable_keeps_incumbent_and_cancellation_propagates(monkeypatch):
    preset = Preset()
    incumbent_gate = {**asdict(preset.gate), "threshold_db": -51.0}
    kwargs: dict[str, Any] = dict(incumbent_settings={"gate": incumbent_gate,
                  "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.62}})
    args = (np.ones(48000, dtype=np.float32) * 0.001,
            np.ones(96000, dtype=np.float32) * 0.1, 48000,
            asdict(preset.gate), asdict(preset.compressor),
            {"band_freqs": preset.eq.band_freqs, "band_gains": preset.eq.band_gains,
             "band_qs": preset.eq.band_qs}, asdict(preset.deesser), asdict(preset.limiter))
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: None)
    result = joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs)
    assert not result["apply_recommended"]
    assert result["gate_settings"] == incumbent_gate
    assert result["suppressor_settings"]["strength"] == 0.62
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate_gate_suppressor_order)
    with pytest.raises(AnalysisCancelled):
        joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs, cancel_check=lambda: True)


def test_joint_selection_evaluates_each_model_and_selects_lower_noise_model(
    monkeypatch,
):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        latency = {"rnnoise": 480, "deepfilter-ll": 480, "deepfilter": 1_440}[model]
        is_noise = float(np.mean(np.abs(audio))) < 0.01
        attenuation = {"rnnoise": 0.5, "deepfilter-ll": 0.1, "deepfilter": 0.05}[model]
        return {
            "output_audio": np.asarray(audio, dtype=np.float32)
            * (attenuation if is_noise else 0.98),
            "suppressor_latency_samples": latency,
            "runtime_ms": 0.0,
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _tuning_args()
    frame_count = 100
    incumbent = {
        "gate": dict(args[3]),
        "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
    }
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(frame_count, dtype=np.float32),
        noise_vad_probabilities=np.zeros(frame_count, dtype=np.float32),
        incumbent_settings=incumbent,
        max_suppressor_latency_ms=15.0,
    )

    assert result["apply_recommended"]
    assert result["suppressor_settings"]["model"] == "deepfilter-ll"
    assert result["selected_candidate"]["noise_model"] == "deepfilter-ll"
    assert result["selected_candidate"]["metrics"]["suppressor_latency_ms"] == 10.0
    assert result["max_suppressor_latency_ms"] == 15.0
    assert [entry["model"] for entry in result["model_evaluations"]] == [
        "rnnoise",
        "deepfilter-ll",
        "deepfilter",
    ]
    assert {
        candidate["noise_model"] for candidate in result["candidates"]
    } == {"rnnoise", "deepfilter-ll", "deepfilter"}
    deepfilter_candidates = [
        candidate
        for candidate in result["candidates"]
        if candidate["noise_model"] == "deepfilter"
    ]
    assert deepfilter_candidates
    assert all(not candidate["gates"]["suppressor_latency"] for candidate in deepfilter_candidates)


def test_joint_selection_reports_unavailable_models_and_retains_incumbent(monkeypatch):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        if model != "rnnoise":
            raise RuntimeError(f"{model} backend unavailable")
        return {
            "output_audio": np.asarray(audio, dtype=np.float32),
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _tuning_args()
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.62},
        },
    )

    assert not result["apply_recommended"]
    assert result["suppressor_settings"]["model"] == "rnnoise"
    assert result["suppressor_settings"]["strength"] == 0.62
    statuses = {entry["model"]: entry["status"] for entry in result["model_evaluations"]}
    assert statuses["rnnoise"] == "available"
    assert statuses["deepfilter-ll"] == "unavailable"
    assert statuses["deepfilter"] == "unavailable"
    assert any("backend unavailable" in entry.get("reason", "") for entry in result["model_evaluations"])


def test_deepfilter_incumbent_keeps_selection_inside_deepfilter_family(monkeypatch):
    preset = Preset()
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: None)
    result = joint_tuning.tune_gate_suppression_dynamics(
        np.full(48_000, 0.001, dtype=np.float32),
        np.full(96_000, 0.1, dtype=np.float32),
        48_000,
        asdict(preset.gate),
        asdict(preset.compressor),
        {
            "band_freqs": preset.eq.band_freqs,
            "band_gains": preset.eq.band_gains,
            "band_qs": preset.eq.band_qs,
        },
        asdict(preset.deesser),
        asdict(preset.limiter),
        incumbent_settings={
            "gate": asdict(preset.gate),
            "suppressor": {
                "enabled": True,
                "strength": 0.62,
                "model": "deepfilter-ll",
            },
        },
        noise_model="deepfilter-ll",
    )
    assert [entry["model"] for entry in result["model_evaluations"]] == [
        "deepfilter-ll",
        "deepfilter",
    ]
    policy = result["model_selection_policy"]
    assert policy["id"] == "retain_deepfilter_family_v1"
    assert policy["excluded_models"] == ["rnnoise"]


def test_latency_gate_accepts_the_configured_35_ms_boundary():
    metrics = {
        "speech_retained_ratio": 0.90,
        "false_closure_rate": 0.0,
        "tail_retained_ratio": 0.60,
        "noise_attenuation_db": 6.0,
        "chatter_events": 0.0,
        "compressor_p95_db": 3.5,
        "compressor_target_p95_db": 3.5,
        "compressor_peak_db": 4.0,
        "compressor_peak_cap_db": 8.0,
        "output_true_peak_db": -1.0,
        "limiter_ceiling_db": -0.5,
        "pre_limiter_headroom_db": 1.0,
        "non_finite_output": 0.0,
        "runtime_factor": 0.5,
        "suppressor_latency_ms": 35.0,
        "max_suppressor_latency_ms": 35.0,
    }
    assert joint_tuning._gates(metrics)["suppressor_latency"]


def test_evaluator_uses_actual_selected_model_and_canonicalizes_alias():
    gate = {"threshold_db": -40.0}
    selected_gate, model, strength = _selected_settings(
        {
            "gate_settings": gate,
            "suppressor_settings": {
                "model": "deepfilternet-ll",
                "strength": 0.47,
            },
        },
        incumbent_model="rnnoise",
        incumbent_strength=1.0,
    )
    assert selected_gate == gate
    assert model == "deepfilter-ll"
    assert strength == 0.47


def test_training_noise_floor_does_not_use_heldout_noise(monkeypatch):
    preset = Preset()
    gate = {
        **asdict(preset.gate),
        "gate_mode": 1,
        "auto_threshold_enabled": True,
        "gate_margin_db": 10.0,
    }
    speech = np.full(48_000, 0.1, dtype=np.float32)
    training_noise = np.full(24_000, 0.001, dtype=np.float32)
    noise_a = np.concatenate((training_noise, np.full(24_000, 0.002, dtype=np.float32)))
    noise_b = np.concatenate((training_noise, np.full(24_000, 0.2, dtype=np.float32)))
    observed_thresholds: list[float] = []

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        observed_thresholds.append(float(settings["gate_threshold_db"]))
        return {
            "output_audio": np.asarray(audio, dtype=np.float32),
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)

    def run(noise: np.ndarray) -> list[float]:
        observed_thresholds.clear()
        joint_tuning.tune_gate_suppression_dynamics(
            noise,
            speech,
            48_000,
            gate,
            asdict(preset.compressor),
            {
                "band_freqs": preset.eq.band_freqs,
                "band_gains": preset.eq.band_gains,
                "band_qs": preset.eq.band_qs,
            },
            asdict(preset.deesser),
            asdict(preset.limiter),
            vad_probabilities=np.ones(100, dtype=np.float32),
            noise_vad_probabilities=np.zeros(100, dtype=np.float32),
            incumbent_settings={
                "gate": gate,
                "suppressor": {"enabled": True, "strength": 1.0, "model": "rnnoise"},
            },
        )
        return list(observed_thresholds)

    thresholds_a = run(noise_a)
    thresholds_b = run(noise_b)
    assert thresholds_a
    assert thresholds_a == thresholds_b
    assert {round(threshold, 5) for threshold in thresholds_a} <= {-53.0, -50.0, -47.0}
