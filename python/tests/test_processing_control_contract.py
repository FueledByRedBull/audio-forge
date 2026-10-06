"""Shared processing presentation metadata and desired/applied setting contract."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QSlider

from mic_eq.ui.compressor_panel import CompressorPanel
from mic_eq.ui.compressor_state import CompressorState
from mic_eq.ui.control_specs import (
    CONTROL_RANGES,
    CONTROL_SPECS,
    GATE_MODES,
    configure_numeric_control,
    control_label,
    control_spec,
    widget_control_label,
)
from mic_eq.ui.deesser_state import DeEsserState
from mic_eq.ui.gate_state import GateState
from mic_eq.ui.gate_panel import GatePanel
from mic_eq.ui.limiter_state import LimiterState
from mic_eq.ui.deesser_panel import DeEsserPanel
from mic_eq.ui.processing_controls import SettingControl


class RecordingProcessor:
    def __init__(self):
        self.values = {"deesser_high_cut_hz": 11000.0}

    def __getattr__(self, name):
        if name == "is_vad_available":
            return lambda: True
        if name in {"is_running", "is_bypass", "is_raw_monitor_enabled", "is_compressor_enabled"}:
            return lambda: False
        if name == "get_deesser_high_cut_hz":
            return lambda: self.values["deesser_high_cut_hz"]
        if name.startswith("set_"):
            key = name.removeprefix("set_")
            return lambda value: self.values.__setitem__(key, value)
        if name.startswith("get_"):
            return lambda: None
        raise AttributeError(name)


@pytest.mark.parametrize(
    ("section", "state_type"),
    [
        ("limiter", LimiterState),
        ("deesser", DeEsserState),
        ("compressor", CompressorState),
        ("gate", GateState),
    ],
)
def test_qml_adapter_reads_shared_control_specs(qapp, section, state_type):
    state = state_type(RecordingProcessor())
    for key, spec in CONTROL_SPECS[section].items():
        control = SettingControl(state, section, key, None)
        minimum, maximum = CONTROL_RANGES.get(section, {}).get(key, (0.0, 1.0))
        assert control.name == spec.accessible_name
        assert control.toolTip == spec.tooltip
        assert control.step == spec.step
        assert control.minimum == minimum
        assert control.maximum == maximum
        assert control.items == list(spec.choices)


def test_numeric_widget_configuration_uses_catalog_ranges_and_rounding(qapp):
    for section, controls in CONTROL_RANGES.items():
        for key, (minimum, maximum) in controls.items():
            spec = control_spec(section, key)
            spinbox = QDoubleSpinBox()
            slider = QSlider(Qt.Orientation.Horizontal)
            configure_numeric_control(
                spinbox, section, key, slider=slider, slider_scale=10,
            )
            assert spinbox.minimum() == minimum
            assert spinbox.maximum() == maximum
            assert spinbox.singleStep() == spec.step
            assert spinbox.suffix() == spec.suffix
            assert spinbox.decimals() == spec.decimals
            assert spinbox.toolTip() == spec.tooltip
            assert spinbox.accessibleName() == spec.accessible_name
            assert slider.minimum() == round(minimum * 10)
            assert slider.maximum() == round(maximum * 10)

    assert GATE_MODES == control_spec("gate", "gate_mode").choices


def test_catalog_keeps_quick_and_classic_row_labels(qapp):
    quick_labels = {
        "gate": {
            "threshold_db": "Threshold", "auto_threshold_enabled": "Auto threshold",
            "attack_ms": "Attack", "release_ms": "Release", "gate_mode": "Gate mode",
            "vad_threshold": "VAD threshold", "vad_hold_time_ms": "Hold time",
            "vad_pre_gain": "VAD pre-gain", "gate_margin_db": "Margin",
        },
        "deesser": {
            "auto_enabled": "Auto", "auto_amount": "Amount", "low_cut_hz": "Low cut",
            "high_cut_hz": "High cut", "threshold_db": "Threshold", "ratio": "Ratio",
            "attack_ms": "Attack", "release_ms": "Release",
            "max_reduction_db": "Max reduction",
        },
        "compressor": {
            "threshold_db": "Threshold", "ratio": "Ratio", "attack_ms": "Attack",
            "release_ms": "Release", "makeup_gain_db": "Makeup gain",
            "adaptive_release": "Adaptive release", "base_release_ms": "Base release",
            "sidechain_highpass_enabled": "Sidechain high-pass",
            "auto_makeup_enabled": "Auto makeup gain", "target_lufs": "Target LUFS",
        },
        "limiter": {
            "ceiling_db": "Ceiling", "careful_output_enabled": "Careful output mode",
            "release_ms": "Release",
        },
    }
    for section, labels in quick_labels.items():
        assert {key: control_label(section, key) for key in labels} == labels

    assert widget_control_label("deesser", "low_cut_hz") == "Low Cut"
    assert widget_control_label("deesser", "high_cut_hz") == "High Cut"
    assert widget_control_label("deesser", "max_reduction_db") == "Max Red"
    assert widget_control_label("compressor", "makeup_gain_db") == "Makeup Gain"
    assert widget_control_label("compressor", "base_release_ms") == "Base Release"
    assert widget_control_label("gate", "gate_mode") == "Gate Mode"
    assert widget_control_label("gate", "vad_threshold") == "VAD Threshold"
    assert widget_control_label("gate", "vad_hold_time_ms") == "Hold Time"
    assert widget_control_label("gate", "vad_pre_gain") == "VAD Pre-Gain"


def test_processing_panels_render_catalog_metadata(qapp):
    processor = RecordingProcessor()
    compressor = CompressorPanel(processor)
    deesser = DeEsserPanel(processor)
    gate = GatePanel(processor)

    for section, panel in (
        ("compressor", compressor),
        ("deesser", deesser),
        ("gate", gate),
    ):
        for key, (spinbox, slider, scale) in panel._numeric_controls.items():
            spec = control_spec(section, key)
            minimum, maximum = CONTROL_RANGES[section][key]
            assert (spinbox.minimum(), spinbox.maximum()) == (minimum, maximum)
            assert spinbox.singleStep() == spec.step
            assert spinbox.suffix() == spec.suffix
            assert spinbox.decimals() == spec.decimals
            assert spinbox.toolTip() == spec.tooltip
            assert spinbox.accessibleName() == spec.accessible_name
            if slider is not None:
                assert slider.minimum() == round(minimum * scale)
                assert slider.maximum() == round(maximum * scale)

    for key, widget in compressor._boolean_controls.items():
        spec = control_spec("compressor", key)
        assert widget.accessibleName() == spec.accessible_name
        assert widget.toolTip() == spec.tooltip
    for key, widget in (
        ("enabled", deesser.enabled_checkbox),
        ("auto_enabled", deesser.auto_checkbox),
    ):
        spec = control_spec("deesser", key)
        assert widget.accessibleName() == spec.accessible_name
        assert widget.toolTip() == spec.tooltip
    for key, widget in (
        ("enabled", gate.enabled_checkbox),
        ("auto_threshold_enabled", gate.auto_threshold_checkbox),
    ):
        spec = control_spec("gate", key)
        assert widget.accessibleName() == spec.accessible_name
        assert widget.toolTip() == spec.tooltip
    for key, widget in (
        ("enabled", compressor.limiter_enabled_checkbox),
        ("careful_output_enabled", compressor.careful_output_checkbox),
    ):
        spec = control_spec("limiter", key)
        assert widget.accessibleName() == spec.accessible_name
        assert widget.toolTip() == spec.tooltip

    for key, spinbox in (
        ("ceiling_db", compressor.ceiling_spinbox),
        ("release_ms", compressor.limiter_release_spinbox),
    ):
        spec = control_spec("limiter", key)
        minimum, maximum = CONTROL_RANGES["limiter"][key]
        assert (spinbox.minimum(), spinbox.maximum()) == (minimum, maximum)
        assert spinbox.singleStep() == spec.step
        assert spinbox.suffix() == spec.suffix
        assert spinbox.decimals() == spec.decimals
        assert spinbox.toolTip() == spec.tooltip
        assert spinbox.accessibleName() == spec.accessible_name

    for panel, section, keys in (
        (compressor, "compressor", (
            "threshold_db", "ratio", "attack_ms", "release_ms", "makeup_gain_db",
            "base_release_ms", "target_lufs",
        )),
        (compressor, "limiter", ("ceiling_db", "release_ms")),
        (deesser, "deesser", (
            "auto_amount", "low_cut_hz", "high_cut_hz", "threshold_db", "ratio",
            "attack_ms", "release_ms", "max_reduction_db",
        )),
        (gate, "gate", (
            "attack_ms", "release_ms", "gate_mode", "vad_threshold", "vad_hold_time_ms",
            "vad_pre_gain", "gate_margin_db",
        )),
    ):
        label_texts = {label.text() for label in panel.findChildren(QLabel)}
        assert {
            f"{widget_control_label(section, key)}:" for key in keys
        } <= label_texts

    for section, key, widget in (
        ("compressor", "adaptive_release", compressor.adaptive_release_checkbox),
        ("compressor", "sidechain_highpass_enabled", compressor.sidechain_highpass_checkbox),
        ("compressor", "auto_makeup_enabled", compressor.auto_makeup_checkbox),
        ("limiter", "careful_output_enabled", compressor.careful_output_checkbox),
        ("deesser", "auto_enabled", deesser.auto_checkbox),
    ):
        assert widget.text() == control_label(section, key)
    assert gate.auto_threshold_checkbox.text() == widget_control_label(
        "gate", "auto_threshold_enabled",
    )

    assert gate.gate_mode_combo.itemText(0) == GATE_MODES[0]
    assert gate.gate_mode_combo.accessibleName() == control_spec("gate", "gate_mode").accessible_name


@pytest.mark.parametrize(
    ("state_type", "key", "edited_value"),
    [
        (LimiterState, "ceiling_db", -3.25),
        (DeEsserState, "threshold_db", -24.5),
        (CompressorState, "threshold_db", -24.5),
        (GateState, "threshold_db", -36.5),
    ],
)
def test_desired_settings_are_distinct_from_processor_accepted_values(
    qapp, state_type, key, edited_value,
):
    state = state_type(RecordingProcessor())
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    accepted = state.get_applied_settings()

    state.set_value(key, edited_value)
    assert state.get_settings()[key] == edited_value
    assert state.get_applied_settings() == accepted

    state.cancel()
    assert state.get_settings() == accepted
    assert state.get_applied_settings() == accepted

    state.set_value(key, edited_value)
    state.flush()
    assert state.get_applied_settings()[key] == edited_value
    assert state.get_settings()[key] == edited_value
