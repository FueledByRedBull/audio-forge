"""QML EQ controls read and edit the shared EQ state."""

from unittest.mock import Mock


from mic_eq.config import EQBandSettings
from mic_eq.ui.eq_controls import EQControl
from mic_eq.ui.eq_state import EQState


def test_eq_control_commands_round_only_edited_fields_and_notify_without_polling(qapp):
    state = EQState(Mock())
    state.set_band(0, EQBandSettings("bell", 1234.567, 2.34567, 1.25))
    state.flush()
    frequency = EQControl(state, "frequency", 0)
    gain = EQControl(state, "gain", 0)
    label = EQControl(state, "gainLabel", 0)
    band_type = EQControl(state, "type", 0)
    slope = EQControl(state, "slope", 0)
    q = EQControl(state, "q", 0)
    changed = []
    label.changed.connect(lambda: changed.append(label.text))
    gain.setValue(35.6)
    gain.released()
    assert state.get_band(0).gain_db == 3.6
    assert label.text == "+3.6"
    assert changed == ["+3.6"]
    assert state.get_band(0).frequency_hz == 1234.567
    assert state.get_band(0).q == 1.25
    frequency.setText("1250.5 Hz")
    assert frequency.value == 1251.0
    assert state.get_band(0).frequency_hz == 1251.0
    q.setText("1.25")
    assert state.get_band(0).q == 1.3
    band_type.setIndex(4)
    assert state.get_band(0).filter_type == "high_pass"
    assert not gain.enabled
    assert not q.enabled
    assert slope.enabled
    gain.setValue(100)
    q.setText("5.5")
    assert state.get_band(0).gain_db == 3.6
    assert state.get_band(0).q == 1.3
    slope.setIndex(3)
    assert state.get_band(0).slope_db_per_octave == 48


def test_eq_control_failure_and_invalid_input_read_back_accepted_values(qapp):
    processor = Mock()
    state = EQState(processor)
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    frequency = EQControl(state, "frequency", 0)
    before = state.get_eq_settings().to_dict()
    errors = []
    state.writeFailed.connect(errors.append)
    processor.apply_eq_layers.side_effect = [OSError("EQ rejected"), None]
    frequency.setText("2000")
    assert state.get_eq_settings().to_dict() == before
    assert frequency.value == before["bands"][0]["frequency_hz"]
    assert errors == ["EQ rejected"]
    frequency.setText("nan")
    frequency.setText("invalid")
    assert state.get_eq_settings().to_dict() == before


def test_eq_top_controls_follow_selection_layers_and_diagnostics_signals(qapp):
    state = EQState(Mock())
    selection = EQControl(state, "band")
    enabled = EQControl(state, "enabled")
    layers = EQControl(state, "layers")
    diagnostics = EQControl(state, "diagnostics")
    assert selection.index == 0
    selection.setIndex(7)
    assert state.selected_band == 7
    state.step_band(1)
    assert selection.index == 8
    assert not diagnostics.property("data")["calibrated"]
    state.set_auto_eq_diagnostics({"analysis_confidence": 0.8, "eq_confidence": 0.8})
    assert diagnostics.property("data")["calibrated"]
    assert diagnostics.text != "Auto-EQ: no calibration diagnostics"
    enabled.click()
    assert not enabled.checked
    assert layers.text == state.layer_status()
    assert not diagnostics.property("data")["calibrated"]
