"""Native contracts for the retained gated-input suppressor ordering."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mic_eq.mic_eq_core import simulate_gate_suppressor_order

MODELS = [
    "rnnoise",
    *[pytest.param(model, marks=pytest.mark.skipif(
        os.environ.get("AUDIOFORGE_ENABLE_DEEPFILTER", "").lower() not in {"1", "true", "yes"},
        reason="DeepFilter runtime must be explicitly enabled",
    )) for model in ("deepfilter-ll", "deepfilter")],
]


def _capture() -> np.ndarray:
    time = np.arange(28_800) / 48_000.0
    level = np.where((time >= 0.12) & (time < 0.37), 0.08, 0.001)
    return (0.00001 + level * np.sin(2.0 * np.pi * 257.0 * time)).astype(np.float32)


def _render(audio: np.ndarray, *, enabled: bool, strength: float,
            mode: int = 0, available: bool = False, suppressor_first: bool = False,
            model: str = "rnnoise") -> dict:
    return simulate_gate_suppressor_order(
        audio, [0.9 if 12 <= frame < 37 else 0.1 for frame in range((audio.size + 479) // 480)],
        suppressor_first, strength,
        {"noise_model": model, "gate_enabled": enabled, "gate_mode": mode,
         "gate_threshold_db": -40.0, "gate_attack_ms": 10.0, "gate_release_ms": 100.0,
         "vad_available": available, "return_dry_audio": not suppressor_first},
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("strength", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("model", MODELS)
def test_gated_neural_input_preserves_incumbent_order(mode, available, strength, model):
    result = _render(_capture(), enabled=True, strength=strength, mode=mode,
                     available=available, model=model)
    gated_input = np.asarray(result["dry_audio"], dtype=np.float32)
    original_order = _render(gated_input, enabled=False, strength=strength, model=model)
    expected_latency = 1440 if model == "deepfilter" else 480
    assert result["suppressor_latency_samples"] == original_order["suppressor_latency_samples"] == expected_latency
    np.testing.assert_array_equal(result["output_audio"], original_order["output_audio"])


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_strength_zero_retains_exact_original_gate_output(mode, available, model):
    result = _render(_capture(), enabled=True, strength=0.0, mode=mode, available=available, model=model)
    np.testing.assert_array_equal(result["output_audio"], result["dry_audio"])


@pytest.mark.parametrize("strength", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("model", MODELS)
def test_gate_off_keeps_plain_suppressor_identity(strength, model):
    audio = _capture()[:-137]
    reference = _render(audio, enabled=False, strength=strength, suppressor_first=True, model=model)
    result = _render(audio, enabled=False, strength=strength, model=model)
    np.testing.assert_array_equal(result["output_audio"], reference["output_audio"])
    expected_latency = 1440 if model == "deepfilter" else 480
    assert result["suppressor_latency_samples"] == reference["suppressor_latency_samples"] == expected_latency
