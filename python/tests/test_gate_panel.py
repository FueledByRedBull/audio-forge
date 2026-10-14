"""Classic gate panel widget specifics; shared gate behaviour is in test_gate_controls_state."""

from __future__ import annotations

from mic_eq.ui.gate_panel import GatePanel


class _GateProcessor:
    def __init__(self, *, vad_available: bool):
        self.vad_available = vad_available
        self.gate_mode_calls: list[int] = []

    def set_gate_enabled(self, _value):
        return None

    def set_gate_threshold(self, _value):
        return None

    def set_gate_attack(self, _value):
        return None

    def set_gate_release(self, _value):
        return None

    def is_vad_available(self):
        return self.vad_available

    def set_gate_mode(self, mode: int):
        self.gate_mode_calls.append(int(mode))

    def set_vad_threshold(self, _value):
        return None

    def set_vad_hold_time(self, _value):
        return None

    def set_vad_pre_gain(self, _value):
        return None

    def set_auto_threshold(self, _value):
        return None

    def set_gate_margin(self, _value):
        return None

    def get_noise_floor(self):
        return -60.0


def test_gate_slider_and_spinbox_stay_synchronized(qapp):
    panel = GatePanel(_GateProcessor(vad_available=True))

    panel.threshold_slider.setValue(-32)
    assert panel.threshold_spinbox.value() == -32.0

    panel.threshold_spinbox.setValue(-26.0)
    assert panel.threshold_slider.value() == -26
