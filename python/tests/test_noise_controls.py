"""Noise controls use state directly and preserve the existing QML contract."""

import pytest
from PySide6.QtCore import QObject

from mic_eq.ui.noise_controls import NoiseControl
from mic_eq.ui.noise_suppression_state import NoiseSuppressionState


class Processor:
    def __init__(self):
        self.writes = []

    def list_noise_models(self):
        return [("rnnoise", "RNNoise"), ("deepfilter-ll", "DeepFilterNet LL")]

    def set_noise_model(self, model):
        self.writes.append(("model", model))
        return True

    def set_rnnoise_enabled(self, value):
        self.writes.append(("enabled", value))

    def set_rnnoise_strength(self, value):
        self.writes.append(("strength", value))


def test_controls_preserve_metadata_precision_and_immediate_edits(qapp):
    processor = Processor()
    state = NoiseSuppressionState(processor)
    parent = QObject()

    def pick(index):  # the window's part: report and apply the chosen backend
        state.set_value("model", state.models[index][0])

    controls = {key: NoiseControl(state, key, parent, pick) for key in ("enabled", "model", "strength", "latency")}
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
    picked = []
    model = NoiseControl(state, "model", parent, picked.append)
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
    enabled.setText("20")
    latency.click()
    assert notified
    assert not processor.writes
    assert not picked
