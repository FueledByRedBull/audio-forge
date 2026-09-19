"""Contracts at the Python/native preview boundary."""

import numpy as np
import pytest

from mic_eq.analysis.auto_eq_parts import headroom
from mic_eq.config import EQSettings
from mic_eq.analysis.auto_eq_parts.optimizer import calculate_eq_bands


def test_flat_controls_survive_headroom_adapter():
    controls = {
        "gate_enabled": False,
        "gate_mode": 2,
        "gate_threshold_db": -25.0,
        "gate_vad_pre_gain": 3.0,
        "suppressor_enabled": False,
        "noise_model": "deepfilter",
    }
    flattened = headroom._flatten_chain_settings(controls)
    assert all(flattened[key] == value for key, value in controls.items())


def test_typed_preview_rejects_inaccurate_fallback(monkeypatch):
    monkeypatch.setattr(
        headroom, "_native_simulate", lambda *_: (None, {"message": "test unavailable"})
    )
    with pytest.raises(RuntimeError, match="requires native DSP simulation"):
        headroom.simulate_candidate_chain(
            np.zeros(480, dtype=np.float32), 48_000, EQSettings().to_dict()
        )


def test_headroom_uses_exact_fit_context_and_ignores_recording_level():
    freqs = np.geomspace(80.0, 16_000.0, 128)
    measured = -40.0 - 6.0 * np.log2(freqs / 1000.0)
    target = np.exp(-np.log2(freqs / 2500.0) ** 2)
    context = {}
    result = calculate_eq_bands(freqs, measured, target, fit_context=context)
    before = result["validation_before_error_db"]
    after = result["validation_after_error_db"]
    headroom._refresh_scaled_eq_metadata(result, freqs, measured - 20.0, target, context)
    assert result["validation_before_error_db"] == pytest.approx(before)
    assert result["validation_after_error_db"] == pytest.approx(after)
