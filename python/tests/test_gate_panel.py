"""Tests for gate panel preset application behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mic_eq.ui.gate_panel import GatePanel
from mic_eq.ui.gate_state import GateState
from PySide6.QtCore import QSignalBlocker


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


def test_gate_panel_uses_calibrated_vad_default(qapp):
    panel = GatePanel(_GateProcessor(vad_available=True))

    assert panel.vad_threshold_slider.value() == 48
    assert panel.vad_threshold_spinbox.value() == 0.48


def test_vad_threshold_controls_match_native_range_and_endpoint(qapp):
    panel = GatePanel(_GateProcessor(vad_available=True))

    assert panel.vad_threshold_slider.minimum() == 30
    assert panel.vad_threshold_slider.maximum() == 70
    assert panel.vad_threshold_spinbox.minimum() == pytest.approx(0.3)
    assert panel.vad_threshold_spinbox.maximum() == pytest.approx(0.7)
    panel.set_settings({"vad_threshold": 0.7})
    assert panel.get_settings()["vad_threshold"] == pytest.approx(0.7)


def test_bulk_gate_apply_propagates_vad_native_failure(qapp):
    processor = _GateProcessor(vad_available=True)
    processor.set_vad_threshold = Mock(side_effect=RuntimeError("VAD write failed"))
    panel = GatePanel(processor)

    with pytest.raises(RuntimeError, match="VAD write failed"):
        panel.set_settings({"vad_threshold": 0.6})


def test_gate_slider_and_spinbox_stay_synchronized(qapp):
    panel = GatePanel(_GateProcessor(vad_available=True))

    panel.threshold_slider.setValue(-32)
    assert panel.threshold_spinbox.value() == -32.0

    panel.threshold_spinbox.setValue(-26.0)
    assert panel.threshold_slider.value() == -26


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


def test_interactive_unavailable_vad_selection_restores_the_widget(qapp):
    panel = GatePanel(_GateProcessor(vad_available=False))
    panel.gate_mode_combo.setCurrentIndex(1)
    assert panel.gate_mode_combo.currentIndex() == 0
    assert panel.get_settings()["gate_mode"] == 0
    assert panel.threshold_spinbox.isEnabled()


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
    assert panel.vad_info_label.text() == "VAD: Active | Speech confidence only"


def test_vad_only_displays_confidence_threshold_and_moves_marker(qapp):
    panel = GatePanel(_GateProcessor(vad_available=True))
    panel.set_settings({"gate_mode": 2, "vad_threshold": 0.35})
    panel.confidence_meter.setFixedSize(200, 20)
    panel.confidence_meter.set_confidence(0.4)
    before = panel.confidence_meter.grab().toImage()

    assert "VAD Threshold: 0.35" in panel.threshold_status_label.text()
    assert "Level fallback" in panel.threshold_status_label.text()
    assert panel.auto_threshold_checkbox.text() == "Track Noise Floor"
    assert not panel.margin_spinbox.isEnabled()

    panel.vad_threshold_spinbox.setValue(0.7)
    after = panel.confidence_meter.grab().toImage()
    assert before != after
    assert panel.confidence_meter.threshold == 0.7
    assert "VAD Threshold: 0.70" in panel.threshold_status_label.text()

    panel.gate_mode_combo.setCurrentIndex(1)
    assert panel.auto_threshold_checkbox.text() == "Auto Threshold"
    assert panel.margin_spinbox.isEnabled()
    assert "Effective Threshold:" in panel.threshold_status_label.text()


def test_gate_widgets_share_state_without_replacing_unedited_exact_values(qapp):
    native = _GateProcessor(vad_available=True)
    state = GateState(native)
    panel = GatePanel(native, state)
    other = GatePanel(native, state)
    panel.set_settings({"threshold_db": -31.234567, "vad_pre_gain": 1.234567})
    assert other.threshold_spinbox.value() == -31.23
    assert other.vad_pre_gain_spinbox.value() == 1.2
    other.enabled_checkbox.setChecked(False)
    state.flush()
    assert not panel.enabled_checkbox.isChecked()
    assert panel.get_settings()["threshold_db"] == -31.234567
    assert panel.get_settings()["vad_pre_gain"] == 1.234567
    with QSignalBlocker(panel.threshold_spinbox):
        panel.threshold_spinbox.setValue(-75.0)
    assert state.get_settings()["threshold_db"] == -31.234567
    state.changed.emit()
    assert panel.threshold_spinbox.value() == -31.23
    other.threshold_slider.setValue(-35)
    other.threshold_slider.sliderReleased.emit()
    assert panel.threshold_spinbox.value() == -35.0
    assert state.get_settings()["vad_pre_gain"] == 1.234567
