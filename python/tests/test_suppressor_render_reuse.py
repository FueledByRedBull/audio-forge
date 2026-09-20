"""Cached wet/dry renders must match native fixed-strength suppression."""
from pathlib import Path

import numpy as np
import pytest

from mic_eq.mic_eq_core import simulate_gate_suppressor_order
from mic_eq.ui.app_bootstrap import configure_deepfilter_env


@pytest.mark.parametrize("model", ["rnnoise", "deepfilter-ll", "deepfilter"])
def test_cached_mix_matches_native_with_partial_frame(model):
    if model != "rnnoise":
        root = Path(__file__).resolve().parents[2]
        archive = "DeepFilterNet3_ll_onnx.tar.gz" if model == "deepfilter-ll" else "DeepFilterNet3_onnx.tar.gz"
        if not (root / "df.dll").is_file() or not (root / "models" / archive).is_file():
            pytest.skip("local DeepFilter runtime assets are unavailable")
    configure_deepfilter_env()
    audio = np.random.default_rng(24).normal(0, .03, 4807).astype(np.float32)
    probabilities = [0.9] * 11
    settings = {"noise_model": model, "gate_enabled": True, "gate_mode": 2}
    wet = simulate_gate_suppressor_order(
        audio, probabilities, False, 1., {**settings, "return_dry_audio": True}
    )
    dry = np.asarray(wet["dry_audio"], dtype=np.float32)
    processed = np.asarray(wet["output_audio"], dtype=np.float32)
    assert dry.shape == processed.shape == audio.shape
    for strength in (0., .25, .75):
        direct = simulate_gate_suppressor_order(audio, probabilities, False, strength, settings)
        mixed = processed * np.float32(strength) + dry * (np.float32(1.) - np.float32(strength))
        np.testing.assert_allclose(mixed, direct["output_audio"], rtol=1e-6, atol=1e-7)
        assert wet["gate_gain"] == direct["gate_gain"]
        assert wet["suppressor_latency_samples"] == direct["suppressor_latency_samples"]


def test_dry_render_rejects_suppressor_first_order():
    with pytest.raises(ValueError, match="requires suppressor_before_gate=false"):
        simulate_gate_suppressor_order(
            np.zeros(480, dtype=np.float32), [0.], True, 1., {"return_dry_audio": True}
        )
