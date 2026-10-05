"""Gate ownership, VAD policy and recovery without constructing a widget."""

from dataclasses import asdict

import pytest

from mic_eq.config import GateSettings
from mic_eq.ui.gate_state import GateState


class RecordingProcessor:
    def __init__(self):
        self.settings = asdict(GateSettings())
        self.writes = []
        self.vad_available = True
        self.noise_floor = -60.0
        self.fail_next_margin = False
        self.fail_all_margins = False
        self.stopped = False

    def __getattr__(self, name):
        keys = {
            "set_gate_enabled": "enabled", "set_gate_threshold": "threshold_db",
            "set_gate_attack": "attack_ms", "set_gate_release": "release_ms",
            "set_gate_mode": "gate_mode", "set_vad_threshold": "vad_threshold",
            "set_vad_hold_time": "vad_hold_time_ms", "set_vad_pre_gain": "vad_pre_gain",
            "set_auto_threshold": "auto_threshold_enabled",
        }
        if name in keys:
            return lambda value: self.settings.__setitem__(keys[name], value)
        raise AttributeError(name)

    def set_gate_margin(self, value):
        if self.fail_next_margin or self.fail_all_margins:
            self.fail_next_margin = False
            raise RuntimeError("margin write rejected")
        self.settings["gate_margin_db"] = value
        self.writes.append(self.settings.copy())

    def is_vad_available(self):
        return self.vad_available

    def get_noise_floor(self):
        return self.noise_floor

    def stop(self):
        self.stopped = True


def test_gate_coalesces_exact_values_and_partial_preset_supersedes_pending_edit(qapp):
    native = RecordingProcessor()
    state = GateState(native)
    state.set_settings({"threshold_db": -37.1234567, "vad_pre_gain": 1.234567})
    state._rate_limiter.interval_ms = 5000
    edits = []
    state.configurationEdited.connect(edits.append)
    state.set_value("threshold_db", -31.0)
    state.set_value("threshold_db", -30.0)
    state.set_value("attack_ms", 7.1234567)
    assert len(native.writes) == 1
    assert not edits
    state.flush()
    assert len(native.writes) == 2
    assert native.settings == state.get_settings()
    assert state.get_settings()["vad_pre_gain"] == 1.234567
    assert edits == ["Noise gate edit"]
    state.set_value("threshold_db", -29.0)
    state.set_settings({"threshold_db": -33.456789})
    state.flush()
    assert len(native.writes) == 3
    assert native.settings["threshold_db"] == -33.456789
    assert native.settings["attack_ms"] == 7.1234567
    assert edits == ["Noise gate edit"]


def test_saved_vad_mode_survives_missing_backend_but_interactive_selection_falls_back(qapp):
    native = RecordingProcessor()
    native.vad_available = False
    state = GateState(native)
    state.set_settings({"gate_mode": 2})
    assert native.settings["gate_mode"] == 2
    assert state.presentation()["vad_info"] == "VAD: Unavailable"
    assert not state.control_enabled("vad_threshold")
    state.set_value("gate_mode", 1)
    state.flush()
    assert native.settings["gate_mode"] == 0
    state.set_settings({"gate_mode": 2})
    native.vad_available = True
    state.update_vad_confidence(0.75)
    assert state.get_settings()["gate_mode"] == 2
    assert state.presentation()["vad_info"] == "VAD: Active | Speech confidence only"
    assert state.control_enabled("vad_threshold")
    assert not state.control_enabled("threshold_db")


def test_gate_telemetry_has_no_configuration_side_effects_and_clears_unavailable_values(qapp):
    native = RecordingProcessor()
    state = GateState(native)
    state.set_settings({"gate_mode": 1, "gate_margin_db": 8.0})
    previous = state.get_settings()
    edits = []
    state.configurationEdited.connect(edits.append)
    state.update_vad_confidence(0.75)
    assert state.confidence == 0.75
    assert state.presentation()["threshold_status"] == (
        "Effective Threshold: -52.0 dB (-60.0 dB floor + 8.0 dB margin)"
    )
    assert state.control_enabled("gate_margin_db")
    assert not state.control_enabled("threshold_db")
    state.update_vad_confidence(None)
    assert state.presentation()["noise_floor"] == "Noise Floor: --"
    assert "noise floor unavailable" in state.presentation()["threshold_status"]
    state.update_vad_confidence(0.7)
    native.vad_available = False
    state.refresh_vad_status()
    assert state.confidence is None
    assert state.noise_floor_db is None
    assert state.control_enabled("threshold_db")
    assert state.get_settings() == previous
    assert len(native.writes) == 1
    assert not edits


@pytest.mark.parametrize("interactive", [False, True])
def test_gate_rejected_native_write_restores_all_values_without_history(qapp, interactive):
    native = RecordingProcessor()
    state = GateState(native)
    state.set_settings({"threshold_db": -31.234567, "gate_mode": 1})
    state._rate_limiter.interval_ms = 5000
    previous = state.get_settings()
    edits, errors = [], []
    state.configurationEdited.connect(edits.append)
    state.writeFailed.connect(errors.append)
    native.fail_next_margin = True
    with pytest.raises(RuntimeError, match="margin write rejected"):
        if interactive:
            state.set_value("threshold_db", -25.0)
            state.flush()
        else:
            state.set_settings({"threshold_db": -25.0, "gate_mode": 2})
    assert state.get_settings() == previous
    assert native.settings == previous
    assert not edits
    assert errors == ["margin write rejected"]


def test_gate_failed_rollback_signals_recovery_barrier_before_stopping_audio(qapp):
    native = RecordingProcessor()
    state = GateState(native)
    native.fail_all_margins = True
    recovery = []
    state.recoveryFailed.connect(lambda message: recovery.append((message, native.stopped)))
    with pytest.raises(RuntimeError, match="previous settings also failed"):
        state.set_settings({"threshold_db": -25.0})
    assert len(recovery) == 1
    assert not recovery[0][1]
    assert native.stopped


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("gate_mode", True), ("gate_mode", 1.0), ("gate_mode", 3),
        ("threshold_db", float("nan")), ("threshold_db", -81.0),
        ("auto_threshold_enabled", "false"), ("vad_threshold", 0.71),
        ("vad_pre_gain", 0.9), ("unknown", 1),
    ],
)
def test_gate_rejects_invalid_edits_before_writing_native(qapp, name, value):
    native = RecordingProcessor()
    state = GateState(native)
    previous = state.get_settings()
    with pytest.raises(ValueError):
        state.set_value(name, value)
    with pytest.raises(ValueError):
        state.set_settings({name: value})
    assert state.get_settings() == previous
    assert not native.writes


def test_gate_cancel_restores_the_applied_settings_and_notifies_views(qapp):
    native = RecordingProcessor()
    state = GateState(native)
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    seen = []
    state.changed.connect(lambda: seen.append(state.get_settings()["threshold_db"]))
    state.set_value("threshold_db", -20.0)
    state.cancel()
    state.flush()
    assert seen == [-20.0, -40.0]
    assert len(native.writes) == 1
