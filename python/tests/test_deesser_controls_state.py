"""De-esser cutoff ranges and preset round trips through the shared state and Quick controls."""

from unittest.mock import Mock

import pytest

from mic_eq.config_parts.presets import Preset
from mic_eq.ui.deesser_state import DeEsserState
from mic_eq.ui.processing_controls import SettingControl


def _control(state: DeEsserState, key: str) -> SettingControl:
    return SettingControl(state, "deesser", key, None)


def _processor() -> Mock:
    processor = Mock()
    processor.get_deesser_high_cut_hz.return_value = 11000.0
    return processor


def test_deesser_controls_preserve_exact_values_and_follow_normalized_state(qapp):
    processor = _processor()
    state = DeEsserState(processor)
    low, other_low = _control(state, "low_cut_hz"), _control(state, "low_cut_hz")
    high = _control(state, "high_cut_hz")
    threshold, ratio = _control(state, "threshold_db"), _control(state, "ratio")
    auto = _control(state, "auto_enabled")
    state.set_settings({
        "low_cut_hz": 4123.456789,
        "high_cut_hz": 9000.123456,
        "auto_amount": 0.512345,
        "threshold_db": -29.123456,
    })
    assert low.value == 4123.46
    assert other_low.value == 4123.46
    assert not threshold.enabled

    auto.click()
    state.flush()
    assert threshold.enabled
    assert ratio.enabled
    assert state.get_settings()["threshold_db"] == -29.123456
    assert state.get_settings()["auto_amount"] == 0.512345
    assert state.get_settings()["low_cut_hz"] == 4123.456789

    other_low.setValue(11000.0)
    other_low.released()
    assert low.value == 11000.0
    assert high.value == 11200.0
    assert state.get_settings()["high_cut_hz"] == 11200.0
    processor.set_deesser_high_cut_hz.assert_called_with(11200.0)


@pytest.mark.parametrize("width_hz", [200.0, 399.0, 400.0, 599.0, 600.0])
def test_deesser_preset_round_trip_preserves_permitted_cutoff_width(width_hz):
    low_cut_hz = 4000.0
    high_cut_hz = low_cut_hz + width_hz
    raw = {
        "deesser": {
            "low_cut_hz": low_cut_hz,
            "high_cut_hz": high_cut_hz,
        }
    }

    preset = Preset.from_dict(raw)
    restored = Preset.from_dict(preset.to_dict())

    assert restored.deesser.low_cut_hz == low_cut_hz
    assert restored.deesser.high_cut_hz == high_cut_hz


def test_deesser_round_trip_preserves_200_hz_cutoff_interval(qapp):
    processor = _processor()
    state = DeEsserState(processor)
    settings = {
        "enabled": True,
        "auto_enabled": False,
        "auto_amount": 0.75,
        "low_cut_hz": 4000.0,
        "high_cut_hz": 4200.0,
        "threshold_db": -32.0,
        "ratio": 6.0,
        "attack_ms": 3.0,
        "release_ms": 100.0,
        "max_reduction_db": 8.0,
    }

    state.set_settings(settings)
    state.flush()

    assert state.get_settings() == settings
    assert state.get_applied_settings() == settings
    processor.set_deesser_low_cut_hz.assert_called_with(4000.0)
    processor.set_deesser_high_cut_hz.assert_called_with(4200.0)


def test_deesser_moves_narrow_range_without_native_clamping(qapp):
    from mic_eq.mic_eq_core import AudioProcessor

    processor = AudioProcessor()
    state = DeEsserState(processor)
    for low, high in [(4000.0, 4200.0), (8000.0, 8200.0), (3000.0, 3200.0)]:
        state.set_settings({"low_cut_hz": low, "high_cut_hz": high})
        state.flush()
        assert processor.get_deesser_low_cut_hz() == low
        assert processor.get_deesser_high_cut_hz() == high


def test_deesser_native_setter_failure_reaches_configuration_caller(qapp):
    processor = _processor()
    state = DeEsserState(processor)
    processor.set_deesser_threshold_db.side_effect = RuntimeError(
        "native de-esser write failed"
    )

    with pytest.raises(RuntimeError, match="native de-esser write failed"):
        state.set_settings({"threshold_db": -11.0})
