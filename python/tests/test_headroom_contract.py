"""Contracts at the Python/native preview boundary."""

import numpy as np
import pytest
from types import SimpleNamespace

from mic_eq.analysis.auto_eq_parts import headroom
from mic_eq.config import EQSettings, Preset
from mic_eq.analysis.auto_eq_parts.optimizer import calculate_eq_bands
from mic_eq import CORE_AVAILABLE


def test_flat_controls_survive_headroom_adapter():
    controls = {
        "gate_enabled": False,
        "gate_mode": 2,
        "gate_threshold_db": -25.0,
        "gate_vad_pre_gain": 3.0,
        "suppressor_enabled": False,
        "noise_model": "deepfilter",
        "vad_available": False,
    }
    flattened = headroom._flatten_chain_settings(controls)
    assert all(flattened[key] == value for key, value in controls.items())


def test_live_calibration_reliability_reaches_native_preview():
    from mic_eq.ui.calibration_dialog import _chain_settings

    preset = Preset()
    owner = SimpleNamespace(
        _get_current_preset=lambda: preset,
        _processing_mode=lambda: "normal",
        processor=SimpleNamespace(get_input_cleanup_mode=lambda: "gentle"),
        compressor_panel=SimpleNamespace(
            get_compressor_settings=lambda *, include_calibration: {
                "noise_reference_reliability": 0.73 if include_calibration else 0.0,
            }
        ),
    )
    for snapshot in (None, preset):
        chain = _chain_settings(owner, full_chain=True, preset=snapshot)
        flattened = headroom._flatten_chain_settings(chain)
        assert flattened["compressor_noise_reference_reliability"] == 0.73
    assert "noise_reference_reliability" not in preset.to_dict()["compressor"]


@pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")
@pytest.mark.parametrize("full_chain,gate_enabled", [(False, False), (True, False), (True, True)])
def test_native_auto_makeup_uses_vad_with_gate_off_or_threshold_only(full_chain, gate_enabled):
    from mic_eq.mic_eq_core import simulate_auto_eq_chain

    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    audio = (0.001 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
    settings = {
        "full_chain": full_chain, "input_pre_filtered": True,
        "gate_enabled": gate_enabled, "gate_mode": 0, "gate_threshold_db": -90.0,
        "gate_auto_threshold_enabled": False, "suppressor_enabled": False,
        "eq_enabled": False, "deesser_enabled": False, "limiter_enabled": False,
        "compressor_enabled": True, "compressor_ratio": 1.0,
        "compressor_threshold_db": 0.0, "compressor_auto_makeup_enabled": True,
    }
    results = [
        simulate_auto_eq_chain(audio, float(sample_rate), [(1000.0, 0.0, 1.41)] * 10,
                               {**settings, "vad_probabilities": [probability] * 200})
        for probability in (0.0, 1.0)
    ]
    assert results[1]["output_rms_db"] > results[0]["output_rms_db"] + 3.0
    assert all(result["analysis_block_ms"] == 10.0 for result in results)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")
def test_full_chain_explicitly_unavailable_vad_does_not_require_model(
    monkeypatch, tmp_path
):
    from mic_eq.mic_eq_core import simulate_auto_eq_chain

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    monkeypatch.delenv("VAD_MODEL_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    sample_rate = 48_000
    audio = np.zeros(sample_rate, dtype=np.float32)
    result = simulate_auto_eq_chain(
        audio,
        float(sample_rate),
        [(1000.0, 0.0, 1.41)] * 10,
        {
            "full_chain": True,
            "processing_mode": "normal",
            "input_pre_filtered": True,
            "vad_available": False,
            "gate_enabled": False,
            "suppressor_enabled": False,
            "eq_enabled": False,
            "deesser_enabled": False,
            "compressor_enabled": True,
            "compressor_auto_makeup_enabled": True,
            "limiter_enabled": False,
            "return_output_audio": True,
        },
    )

    assert result["full_chain"] is True
    assert result["processed_samples"] == audio.size
    assert np.isfinite(np.asarray(result["output_audio"], dtype=np.float32)).all()


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
