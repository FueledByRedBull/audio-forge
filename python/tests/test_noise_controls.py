"""Noise controls use state directly and preserve the existing QML contract."""

import pytest
from PySide6.QtCore import QObject

from mic_eq.ui.noise_controls import NoiseControl
from mic_eq.ui.noise_suppression_state import NoiseSuppressionState


class Processor:
    def __init__(self):
        self.writes = []
        self.reject_model = False

    def list_noise_models(self):
        return [("rnnoise", "RNNoise"), ("deepfilter-ll", "DeepFilterNet LL")]

    def set_noise_model(self, model):
        self.writes.append(("model", model))
        return not self.reject_model or model == "rnnoise"

    def set_rnnoise_enabled(self, value):
        self.writes.append(("enabled", value))

    def set_rnnoise_strength(self, value):
        self.writes.append(("strength", value))


def test_controls_preserve_metadata_precision_and_immediate_edits(qapp):
    processor = Processor()
    state = NoiseSuppressionState(processor)
    parent = QObject()
    controls = {key: NoiseControl(state, key, parent) for key in ("enabled", "model", "strength", "latency")}
    enabled, model, strength, latency = (controls[key] for key in controls)
    assert not processor.writes
    assert enabled.name == "Enable noise suppression"
    assert enabled.toolTip == "Enable or disable the selected suppression backend."
    assert model.name == "Noise suppression backend"
    assert model.items == ["RNNoise", "DeepFilterNet LL"]
    assert model.index == 0 and model.text == "RNNoise"
    assert strength.name == "Noise suppression strength"
    assert strength.toolTip == "Processing strength for the selected backend (0% dry, 100% fully processed)."
    assert (strength.minimum, strength.maximum, strength.step) == (0, 100, 1)
    assert all(control.enabled and control.shown for control in controls.values())
    assert latency.text == "Latency: ~10ms (RNNoise)"

    notifications = []
    strength.changed.connect(lambda: notifications.append(strength.value))
    state.set_settings({"strength": 0.123456789})
    assert notifications[-1] == strength.value == 12
    assert strength.text == "12%"
    strength.released()
    assert state.get_settings()["strength"] == 0.123456789
    processor.writes.clear()
    enabled.click()
    assert not enabled.checked
    assert model.enabled and strength.enabled
    assert state.get_settings()["strength"] == 0.123456789
    model.setIndex(1)
    assert model.index == 1 and model.text == "DeepFilterNet LL"
    assert latency.text == "Latency: ~10ms (DeepFilterNet LL)"
    assert state.get_settings()["strength"] == 0.123456789
    strength.setValue(66.5)
    assert state.get_settings()["strength"] == 0.66
    assert strength.value == 66 and strength.text == "66%"
    assert processor.writes == [("enabled", False), ("model", "deepfilter-ll"), ("strength", 0.66)]
    strength.setText("45,6%")
    assert state.get_settings()["strength"] == 0.46


def test_rejected_picker_reads_back_and_reports_error_without_history(qapp):
    processor = Processor()
    state = NoiseSuppressionState(processor)
    parent = QObject()
    model = NoiseControl(state, "model", parent)
    observed, errors, edits = [], [], []
    model.changed.connect(lambda: observed.append(model.index))
    state.writeFailed.connect(errors.append)
    state.configurationEdited.connect(edits.append)
    processor.reject_model = True
    model.setIndex(1)
    assert model.index == 0 and model.text == "RNNoise"
    assert observed and all(index == 0 for index in observed)
    assert errors == ["Noise model 'deepfilter-ll' is unavailable; previous model retained"]
    assert not edits


@pytest.mark.parametrize("value,expected", [(-1, 0), (200, 100), (0.5, 0), (1.5, 2), (49.5, 50)])
def test_strength_clamps_and_rounds_like_the_previous_widget_proxy(qapp, value, expected):
    state = NoiseSuppressionState(Processor())
    parent = QObject()
    strength = NoiseControl(state, "strength", parent)
    strength.setValue(value)
    assert strength.value == expected
    assert state.get_settings()["strength"] == expected / 100


def test_invalid_or_unrelated_gestures_refresh_without_native_writes(qapp):
    processor = Processor()
    state = NoiseSuppressionState(processor)
    parent = QObject()
    strength = NoiseControl(state, "strength", parent)
    model = NoiseControl(state, "model", parent)
    enabled = NoiseControl(state, "enabled", parent)
    latency = NoiseControl(state, "latency", parent)
    notified = []
    strength.changed.connect(lambda: notified.append(True))
    for value in (float("nan"), float("inf"), float("-inf")):
        strength.setValue(value)
    strength.setText("invalid")
    strength.setIndex(0)
    strength.click()
    model.setValue(20)
    model.setIndex(-1)
    model.setIndex(2)
    enabled.setText("20")
    latency.click()
    assert notified
    assert not processor.writes
