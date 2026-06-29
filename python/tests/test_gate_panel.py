"""Tests for gate panel preset application behavior."""

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


def test_gate_panel_preserves_loaded_vad_mode_when_backend_unavailable(qapp):
    processor = _GateProcessor(vad_available=False)
    panel = GatePanel(processor)

    panel.set_settings(
        {
            "gate_mode": 2,
            "vad_threshold": 0.42,
            "vad_hold_time_ms": 180.0,
            "vad_pre_gain": 1.8,
            "auto_threshold_enabled": True,
            "gate_margin_db": 8.0,
        }
    )

    assert panel.gate_mode_combo.currentIndex() == 2
    assert processor.gate_mode_calls[-1] == 2


def test_gate_panel_refreshes_restored_vad_status_when_backend_becomes_available(qapp):
    processor = _GateProcessor(vad_available=False)
    panel = GatePanel(processor)

    panel.set_settings(
        {
            "gate_mode": 2,
            "vad_threshold": 0.42,
            "vad_hold_time_ms": 180.0,
            "vad_pre_gain": 1.8,
            "auto_threshold_enabled": True,
            "gate_margin_db": 8.0,
        }
    )

    assert panel.gate_mode_combo.currentIndex() == 2
    assert panel.vad_info_label.text() == "VAD: Unavailable"

    processor.vad_available = True
    panel.update_vad_confidence(0.75)

    assert panel.gate_mode_combo.currentIndex() == 2
    assert panel.vad_info_label.text() == "VAD: Active | Auto threshold on"
