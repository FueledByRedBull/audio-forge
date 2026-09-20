"""Focused checks for joint suppression render reuse and progress reporting."""

from dataclasses import asdict
from itertools import count
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis import joint_tuning
from mic_eq.config import Preset


def _args() -> tuple[Any, ...]:
    preset = Preset()
    return (
        np.full(9_600, 0.001, dtype=np.float32),
        np.full(19_200, 0.1, dtype=np.float32),
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


def _downstream_metrics(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
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


class _FakeSuppressor:
    def __init__(self, *, returns_dry_audio: bool) -> None:
        self.returns_dry_audio = returns_dry_audio
        self.calls: list[tuple[float, bool]] = []

    def __call__(
        self,
        audio: np.ndarray,
        _probabilities: list[float],
        _suppressor_before_gate: bool,
        strength: float,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        strength = float(strength)
        request_dry = bool(settings.get("return_dry_audio", False))
        self.calls.append((strength, request_dry))
        is_noise = float(np.mean(np.abs(audio))) < 0.01
        wet_scale = 0.4 if is_noise else 0.98
        dry = np.asarray(audio, dtype=np.float32)
        wet = dry * np.float32(wet_scale)
        result = {
            "output_audio": (
                wet
                if request_dry
                else dry * np.float32(1.0 - strength) + wet * np.float32(strength)
            ),
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
        }
        if request_dry and self.returns_dry_audio:
            result["dry_audio"] = dry
        return result


def test_joint_tuning_reuses_full_wet_render_without_changing_selection_or_scores(
    monkeypatch,
):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _args()
    incumbent = {
        "gate": dict(args[3]),
        "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.5},
    }
    kwargs = {
        "vad_probabilities": np.ones(100, dtype=np.float32),
        "noise_vad_probabilities": np.zeros(100, dtype=np.float32),
        "incumbent_settings": incumbent,
    }

    identity = joint_tuning._array_identity
    unique = count()
    monkeypatch.setattr(joint_tuning, "_array_identity", lambda _: (next(unique), 0, ()))
    baseline_simulator = _FakeSuppressor(returns_dry_audio=True)
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: baseline_simulator)
    baseline = joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs)

    monkeypatch.setattr(joint_tuning, "_array_identity", identity)
    cached_simulator = _FakeSuppressor(returns_dry_audio=True)
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: cached_simulator)
    progress: list[tuple[str, int]] = []
    cached = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        **kwargs,
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
    )

    assert len(cached_simulator.calls) < len(baseline_simulator.calls)
    assert cached["suppressor_settings"] == baseline["suppressor_settings"]
    assert cached["candidate_count"] == baseline["candidate_count"]
    np.testing.assert_allclose(
        [candidate["score"] for candidate in cached["candidates"]],
        [candidate["score"] for candidate in baseline["candidates"]],
        rtol=0.0,
        atol=1.0e-7,
    )
    assert progress[0][1] == 65
    assert progress[-1][1] == 90
    percentages = [item[1] for item in progress]
    assert percentages == sorted(percentages)


def test_joint_tuning_stops_after_unavailable_backend_probe(monkeypatch):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    args = _args()
    calls = 0

    def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: unavailable)
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
        },
    )

    assert calls == 1
    assert not result["apply_recommended"]
    assert result["model_evaluations"][0]["status"] == "unavailable"


def test_requested_dry_audio_cannot_be_silently_omitted():
    with pytest.raises(RuntimeError, match="did not return dry_audio"):
        joint_tuning._run_gate_suppressor(
            _FakeSuppressor(returns_dry_audio=False),
            np.zeros(480, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            {},
            suppressor_strength=1.0,
            noise_model="rnnoise",
            return_dry_audio=True,
        )


def test_runtime_is_a_feasibility_gate_not_a_sound_preference(monkeypatch):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _args()
    simulator = _FakeSuppressor(returns_dry_audio=True)

    def run(runtime_factor):
        def timed(audio, *params):
            result = simulator(audio, *params)
            result["runtime_ms"] = runtime_factor * len(audio) / 48.0
            return result

        monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: timed)
        return joint_tuning.tune_gate_suppression_dynamics(
            *args,
            vad_probabilities=np.ones(100, dtype=np.float32),
            noise_vad_probabilities=np.zeros(100, dtype=np.float32),
            incumbent_settings={
                "gate": dict(args[3]),
                "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.5},
            },
        )

    fast, busy, overloaded = (run(factor) for factor in (0.05, 0.8, 1.1))
    assert fast["candidates"]
    assert [c["score"] for c in fast["candidates"]] == [
        c["score"] for c in busy["candidates"]
    ]
    assert fast["gate_settings"] == busy["gate_settings"]
    assert fast["suppressor_settings"] == busy["suppressor_settings"]
    assert all(c["gates"]["runtime"] for c in busy["candidates"])
    assert all(not c["gates"]["runtime"] for c in overloaded["candidates"])
    assert not overloaded["apply_recommended"]
