"""Shared processing presentation metadata and desired/applied setting contract."""

from typing import Any

import pytest

from mic_eq.ui.compressor_state import CompressorState
from mic_eq.ui.control_specs import (
    CONTROL_RANGES,
    CONTROL_SPECS,
    control_label,
)
from mic_eq.ui.deesser_state import DeEsserState
from mic_eq.ui.gate_state import GateState
from mic_eq.ui.limiter_state import LimiterState
from mic_eq.ui.processing_controls import SettingControl


class RecordingProcessor:
    def __init__(self):
        self.values: dict[str, Any] = {"deesser_high_cut_hz": 11000.0}

    def __getattr__(self, name):
        if name == "is_vad_available":
            return lambda: True
        if name in {"is_running", "is_bypass", "is_raw_monitor_enabled", "is_compressor_enabled"}:
            return lambda: False
        if name == "get_deesser_high_cut_hz":
            return lambda: self.values["deesser_high_cut_hz"]
        if name.startswith("apply_"):
            return lambda settings: self.values.__setitem__(name, dict(settings))
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


def test_catalog_keeps_the_quick_row_labels(qapp):
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
