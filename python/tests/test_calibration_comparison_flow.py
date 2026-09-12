"""Listening review never applies on reject and always releases its mute owner."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog, QWidget

from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from mic_eq.config import Preset


@pytest.mark.parametrize("dialog_type", [CalibrationDialog, VoiceSetupDialog])
@pytest.mark.parametrize("keep", [False, True])
def test_comparison_uses_one_capture_and_existing_apply_path(qapp, monkeypatch, dialog_type, keep):
    from mic_eq.ui import listening_comparison_dialog

    owner: Any = QWidget()
    owner.processor = SimpleNamespace(sample_rate=lambda: 48000, get_input_cleanup_mode=lambda: "off")
    owner._get_current_preset = Preset
    owner._processing_mode = lambda: "normal"
    owner.eq_panel = SimpleNamespace(get_settings=lambda: {"band_gains": [0.0] * 10})
    owner.set_temporary_output_mute = Mock()
    dialog = dialog_type(owner)
    audio = np.zeros(48000, dtype=np.float32)
    monkeypatch.setattr(dialog, "_candidate_identity_error", lambda *args: None)
    apply = Mock()
    if dialog_type is CalibrationDialog:
        dialog.audio_data = audio
        dialog.preview_audio_data = audio
        dialog.eq_settings = {"band_gains": [1.0] * 10}
        dialog.recording_state = "ready"
        monkeypatch.setattr(dialog, "_apply_eq_settings", apply)
    else:
        dialog.voice_audio = audio
        dialog.preview_voice_audio = audio
        dialog.setup_result = {"eq_settings": {"band_gains": [1.0] * 10},
                               "deesser_settings": {}, "compressor_settings": {}, "limiter_settings": {}}
        dialog.setup_state = "completed"
        monkeypatch.setattr(dialog, "_apply_setup", apply)
    comparison = Mock()
    comparison.exec.return_value = int(QDialog.DialogCode.Accepted if keep else QDialog.DialogCode.Rejected)
    factory = Mock(return_value=comparison)
    monkeypatch.setattr(listening_comparison_dialog, "ListeningComparisonDialog", factory)
    try:
        dialog._compare_recording()
        assert factory.call_args.kwargs["audio_data"] is audio
        assert factory.call_args.kwargs["current_chain_settings"]["full_chain"] is True
        assert factory.call_args.kwargs["current_chain_settings"]["input_pre_filtered"] is False
        assert factory.call_args.kwargs["proposed_chain_settings"]["full_chain"] is True
        assert apply.call_count == int(keep)
        assert owner.set_temporary_output_mute.call_args_list == [
            ((True, "listening_comparison"),), ((False, "listening_comparison"),)]
        comparison.deleteLater.assert_called_once()
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.deleteLater()


def test_suppression_settings_cannot_switch_backend_or_accept_nan(qapp):
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QSlider
    from mic_eq.ui.voice_setup_dialog import _apply_suppressor_settings

    owner = SimpleNamespace(rnnoise_checkbox=QCheckBox(), strength_slider=QSlider(), model_combo=QComboBox())
    owner.model_combo.addItem("RNNoise", "rnnoise")
    owner.strength_slider.setRange(0, 100)
    with pytest.raises(ValueError):
        _apply_suppressor_settings(owner, {"model": "deepfilter", "strength": 0.5, "enabled": True})
    with pytest.raises(ValueError):
        _apply_suppressor_settings(owner, {"model": "rnnoise", "strength": float("nan"), "enabled": True})
    _apply_suppressor_settings(owner, {"model": "rnnoise", "strength": 0.53, "enabled": True})
    assert owner.strength_slider.value() == 53
    assert owner.rnnoise_checkbox.isChecked()


def test_calibration_copy_filters_dc_and_does_not_hide_native_failure(monkeypatch):
    from mic_eq.ui.calibration_dialog import _filtered_capture_for_analysis

    audio = np.full(48000, 0.1, dtype=np.float32)
    filtered = _filtered_capture_for_analysis(audio, 48000)
    assert np.max(np.abs(filtered[24000:])) < 0.001
    assert np.all(audio == np.float32(0.1))
    monkeypatch.setattr("mic_eq.mic_eq_core.simulate_auto_eq_chain",
                        Mock(side_effect=RuntimeError("native failure")))
    with pytest.raises(RuntimeError, match="native failure"):
        _filtered_capture_for_analysis(audio, 48000)
