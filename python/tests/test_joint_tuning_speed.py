"""Focused checks for joint suppression render reuse and progress reporting."""

from dataclasses import asdict
from itertools import count
import threading
import time
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis import joint_tuning
from mic_eq.config import Preset


def _args() -> tuple[Any, ...]:
    preset = Preset()
    return (
        np.full(96_000, 0.001, dtype=np.float32),
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
        request_activity = bool(settings.get("return_auto_makeup_activity", False))
        self.calls.append((strength, request_dry))
        dry = np.asarray(audio, dtype=np.float32)
        wet_scale = np.where(np.abs(dry) < 0.01, 0.4, 0.98)
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
        if request_activity:
            reliability = float(bool(settings.get("vad_available", True)))
            frame_count = (len(audio) + 479) // 480
            result["auto_makeup_activity"] = [
                (0.8, reliability, -60.0, 0.0)
            ] * frame_count
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
    assert cached["apply_recommended"]
    assert cached["suppressor_settings"]["strength"] != incumbent["suppressor"]["strength"]
    np.testing.assert_allclose(
        [candidate["score"] for candidate in cached["candidates"]],
        [candidate["score"] for candidate in baseline["candidates"]],
        rtol=0.0,
        atol=1.0e-7,
    )
    assert progress[0][1] == 25
    assert progress[-1][1] == 55
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


class _SlowSuppressor(_FakeSuppressor):
    """Fails past a release time and takes long enough for concurrent candidates to overlap."""

    def __init__(self, fail_release_ms: float = float("inf")) -> None:
        super().__init__(returns_dry_audio=True)
        self.fail_release_ms = fail_release_ms
        self.renders: list[tuple[float, float, int]] = []

    def __call__(self, audio, probabilities, before_gate, strength, settings):
        self.renders.append(
            (settings["gate_threshold_db"], settings["gate_release_ms"], len(audio))
        )
        time.sleep(0.02)
        if settings["gate_release_ms"] > self.fail_release_ms:
            raise RuntimeError(f"no render at {settings['gate_release_ms']:g} ms")
        return super().__call__(audio, probabilities, before_gate, strength, settings)


def _tune_with(monkeypatch, simulator, workers):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulator)
    monkeypatch.setattr(joint_tuning, "ANALYSIS_WORKERS", workers)
    args = _args()
    progress: list[tuple[str, int]] = []
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.5},
        },
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
    )
    return result, progress


def test_candidates_evaluated_together_give_the_serial_result_in_the_serial_order(monkeypatch):
    # The gate's release is 100 ms; candidates at +35 and +70 ms fail to render.
    serial, serial_progress = _tune_with(monkeypatch, _SlowSuppressor(fail_release_ms=120.0), 1)
    together, together_progress = _tune_with(monkeypatch, _SlowSuppressor(fail_release_ms=120.0), 8)

    assert len(serial["failure_reasons"]) == 2 and serial["candidates"]
    assert serial["apply_recommended"]
    for key in ("candidates", "failure_reasons", "model_evaluations", "reasons", "decision", "holdout_passed",
                "holdout_gates", "gate_settings", "suppressor_settings", "selected_candidate"):
        assert together[key] == serial[key], key
    assert together_progress == serial_progress


def test_candidates_that_need_the_same_render_wait_for_one_render(monkeypatch):
    simulator = _SlowSuppressor()
    result, _ = _tune_with(monkeypatch, simulator, 8)

    assert result["holdout_evaluated"] and result["candidate_count"] == 10
    # Candidates that differ only in strength share a render, however they overlap: 7 renders for the 10
    # candidates, then one each for the selected candidate and the incumbent on the held-out half.
    assert len(simulator.renders) == 7 + 2


class _ContendedSuppressor(_SlowSuppressor):
    """Reports a render time that grows with the renders running at once, as wall time does."""

    def __init__(self) -> None:
        super().__init__()
        self.guard = threading.Lock()
        self.active = 0
        self.most_active = 0

    def __call__(self, audio, probabilities, before_gate, strength, settings):
        with self.guard:
            self.active += 1
            self.most_active = max(self.most_active, self.active)
            sharing = self.active
        try:
            result = super().__call__(audio, probabilities, before_gate, strength, settings)
        finally:
            with self.guard:
                self.active -= 1
        # 5% of the audio's duration when alone, and longer for each render it shares the processors with.
        result["runtime_ms"] = 0.05 * len(audio) / 48.0 * sharing
        return result


def test_the_runtime_gate_does_not_grow_with_the_candidates_evaluated_beside_it(monkeypatch):
    alone, beside = _ContendedSuppressor(), _ContendedSuppressor()
    serial, _ = _tune_with(monkeypatch, alone, 1)
    together, _ = _tune_with(monkeypatch, beside, 8)

    assert alone.most_active == 1 and beside.most_active > 1  # the second run really did share processors
    factors = {candidate["metrics"]["runtime_factor"] for candidate in together["candidates"]}
    assert factors == {candidate["metrics"]["runtime_factor"] for candidate in serial["candidates"]}
    assert len(factors) == 1 and 0.045 < factors.pop() < 0.055
    assert together["holdout_evaluated"] and together["holdout_gates"]["runtime"]
