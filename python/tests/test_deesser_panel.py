"""Regression tests for de-esser cutoff ranges and preset round trips."""

from unittest.mock import Mock

import pytest

from mic_eq.config_parts.presets import Preset
from mic_eq.ui.deesser_panel import DeEsserPanel
from mic_eq.ui.deesser_state import DeEsserState


def test_deesser_views_preserve_exact_values_and_follow_normalized_state(qapp):
    processor = Mock()
    processor.get_deesser_high_cut_hz.return_value = 11000.0
    state = DeEsserState(processor)
    panel = DeEsserPanel(processor, state)
    other = DeEsserPanel(processor, state)
    panel.set_settings({
        "low_cut_hz": 4123.456789,
        "high_cut_hz": 9000.123456,
        "auto_amount": 0.512345,
        "threshold_db": -29.123456,
    })
    assert panel.low_cut_spinbox.value() == 4123.46
    assert other.low_cut_spinbox.value() == 4123.46
    assert not other.threshold_spinbox.isEnabled()
    panel.auto_checkbox.setChecked(False)
    state.flush()
    assert other.threshold_spinbox.isEnabled()
    assert other.ratio_slider.isEnabled()
    assert panel.get_settings()["threshold_db"] == -29.123456
    assert panel.get_settings()["auto_amount"] == 0.512345
    assert other.get_settings()["low_cut_hz"] == 4123.456789

    other.low_cut_slider.setValue(11000)
    other.low_cut_slider.sliderReleased.emit()
    assert panel.low_cut_spinbox.value() == 11000.0
    assert panel.high_cut_spinbox.value() == 11200.0
    assert panel.get_settings() == other.get_settings()
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


def test_deesser_panel_round_trip_preserves_200_hz_cutoff_interval(qapp):
    processor = Mock()
    processor.get_deesser_high_cut_hz.return_value = 11000.0
    panel = DeEsserPanel(processor)

    panel.set_settings(
        {
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
    )
    panel.deesser_state.flush()

    assert panel.get_settings() == {
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
    processor.set_deesser_low_cut_hz.assert_called_with(4000.0)
    processor.set_deesser_high_cut_hz.assert_called_with(4200.0)


def test_deesser_panel_moves_narrow_range_without_native_clamping(qapp):
    from mic_eq.mic_eq_core import AudioProcessor

    processor = AudioProcessor()
    panel = DeEsserPanel(processor)
    for low, high in [(4000.0, 4200.0), (8000.0, 8200.0), (3000.0, 3200.0)]:
        panel.set_settings({"low_cut_hz": low, "high_cut_hz": high})
        panel.deesser_state.flush()
        assert processor.get_deesser_low_cut_hz() == low
        assert processor.get_deesser_high_cut_hz() == high


def test_deesser_native_setter_failure_reaches_configuration_caller(qapp):
    processor = Mock()
    processor.get_deesser_high_cut_hz.return_value = 11000.0
    panel = DeEsserPanel(processor)
    processor.set_deesser_threshold_db.side_effect = RuntimeError(
        "native de-esser write failed"
    )
    panel.deesser_state._rate_limiter._last_call_time = 0.0

    with pytest.raises(RuntimeError, match="native de-esser write failed"):
        panel.set_settings({"threshold_db": -11.0})
