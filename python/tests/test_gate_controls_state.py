"""Gate behaviour through the shared state and the controls the Quick view binds."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from unittest.mock import Mock

import pytest
from PySide6.QtGui import QImage, QPainter

from mic_eq.ui.gate_state import GateState
from mic_eq.ui.processing_controls import GateText, SettingControl
from mic_eq.ui.processing_meters import ProcessingMeters
from test_gate_state import RecordingProcessor

VAD_RESTORED = {
    "gate_mode": 2,
    "vad_threshold": 0.42,
    "vad_hold_time_ms": 180.0,
    "vad_pre_gain": 1.8,
    "auto_threshold_enabled": True,
    "gate_margin_db": 8.0,
}


def _control(state: GateState, key: str) -> SettingControl:
    return SettingControl(state, "gate", key, None)


def window_gate_control(window, label: str) -> SettingControl:
    """The control the scene's gate card binds for ``label``, for example "Threshold"."""
    card = window.quick_bridge.model["stages"][1]
    return next(row["proxy"] for row in card["rows"] + card["advanced"] if row["label"] == label)


def _without_backend() -> RecordingProcessor:
    native = RecordingProcessor()
    native.vad_available = False
    return native


def test_gate_uses_calibrated_vad_default(qapp):
    state = GateState(RecordingProcessor())
    control = _control(state, "vad_threshold")

    assert state.get_settings()["vad_threshold"] == 0.48
    assert control.value == 0.48
    assert control.text == "0.48"


def test_vad_threshold_control_matches_native_range_and_endpoint(qapp):
    native = RecordingProcessor()
    state = GateState(native)
    control = _control(state, "vad_threshold")

    assert control.minimum == pytest.approx(0.3)
    assert control.maximum == pytest.approx(0.7)
    state.set_settings({"vad_threshold": 0.7})
    assert state.get_settings()["vad_threshold"] == pytest.approx(0.7)
    assert native.settings["vad_threshold"] == pytest.approx(0.7)
    assert control.value == pytest.approx(0.7)
    # A drag or typed value past the end stops at the endpoint.
    control.setValue(0.95)
    control.released()
    assert state.get_settings()["vad_threshold"] == pytest.approx(0.7)


def test_bulk_gate_apply_propagates_vad_native_failure(qapp):
    native = RecordingProcessor()
    native.set_vad_threshold = Mock(side_effect=RuntimeError("VAD write failed"))
    state = GateState(native)

    with pytest.raises(RuntimeError, match="VAD write failed"):
        state.set_settings({"vad_threshold": 0.6})


def test_gate_preserves_loaded_vad_mode_when_backend_unavailable(qapp):
    native = _without_backend()
    state = GateState(native)
    mode = _control(state, "gate_mode")

    state.set_settings(VAD_RESTORED)

    assert mode.index == 2
    assert native.settings["gate_mode"] == 2


def test_interactive_unavailable_vad_selection_is_read_back_as_threshold_mode(qapp):
    state = GateState(_without_backend())
    mode = _control(state, "gate_mode")
    threshold = _control(state, "threshold_db")
    reads = []
    mode.changed.connect(lambda: reads.append(mode.index))

    mode.setIndex(1)

    # The view is told to read the mode back, and reads the corrected one.
    assert reads and reads[-1] == 0
    assert mode.index == 0
    assert state.get_settings()["gate_mode"] == 0
    assert threshold.enabled


def test_gate_refreshes_restored_vad_status_when_backend_becomes_available(qapp):
    native = _without_backend()
    state = GateState(native)
    mode = _control(state, "gate_mode")
    info = GateText(state, "vad_info", None)
    seen = []
    info.changed.connect(lambda: seen.append(info.text))

    state.set_settings(VAD_RESTORED)

    assert mode.index == 2
    assert info.text == "VAD: Unavailable"

    native.vad_available = True
    state.update_vad_confidence(0.75)

    assert mode.index == 2
    assert info.text == "VAD: Active | Speech confidence only"
    assert seen[-1] == "VAD: Active | Speech confidence only"


def _render(meter, width: int = 200, height: int = 20) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    meter.paint_meter(painter, width, height)
    painter.end()
    return image


def test_vad_only_displays_confidence_threshold_and_moves_marker(qapp):
    native = RecordingProcessor()
    state = GateState(native)
    meters = ProcessingMeters(native, state)
    status = GateText(state, "threshold_status", None)
    threshold = _control(state, "vad_threshold")
    margin = _control(state, "gate_margin_db")
    auto = _control(state, "auto_threshold_enabled")
    mode = _control(state, "gate_mode")
    state.set_settings({"gate_mode": 2, "vad_threshold": 0.35})
    state.update_vad_confidence(0.4)
    before = _render(meters.confidence)

    assert "VAD Threshold: 0.35" in str(status.text)
    assert "Level fallback" in str(status.text)
    assert state.presentation()["auto_threshold_text"] == "Track Noise Floor"
    assert auto.name == "Track noise floor"
    assert not margin.enabled

    threshold.setValue(0.7)
    threshold.released()
    after = _render(meters.confidence)
    assert before != after
    assert meters.confidence.threshold == 0.7
    assert "VAD Threshold: 0.70" in str(status.text)

    mode.setIndex(1)
    assert state.presentation()["auto_threshold_text"] == "Auto Threshold"
    assert auto.name == "Enable automatic gate threshold"
    assert margin.enabled
    assert "Effective Threshold:" in str(status.text)


def test_gate_controls_share_state_without_replacing_unedited_exact_values(qapp):
    state = GateState(RecordingProcessor())
    threshold, other_threshold = _control(state, "threshold_db"), _control(state, "threshold_db")
    pre_gain = _control(state, "vad_pre_gain")
    enabled, other_enabled = _control(state, "enabled"), _control(state, "enabled")

    state.set_settings({"threshold_db": -31.234567, "vad_pre_gain": 1.234567})
    assert other_threshold.value == -31.23
    assert pre_gain.value == 1.2

    other_enabled.click()
    state.flush()
    assert not enabled.checked
    assert state.get_settings()["threshold_db"] == -31.234567
    assert state.get_settings()["vad_pre_gain"] == 1.234567

    other_threshold.setValue(-35.0)
    other_threshold.released()
    assert threshold.value == -35.0
    assert state.get_settings()["vad_pre_gain"] == 1.234567
