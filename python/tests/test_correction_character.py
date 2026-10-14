"""Character changes retain the calibrated correction and survive presets."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from copy import deepcopy

from mic_eq.config import EQ_FREQUENCIES, Preset
import test_eq_editing_state as eq_editing
from test_eq_editing_state import SLOPES, TYPES, edit, eq_control, tone_action

quick_window = eq_editing.quick_window


def test_tone_preserves_correction_roundtrip_and_clear(quick_window):
    window = quick_window
    state, processor = window.eq_state, window.processor
    state.apply_auto_eq_results([(freq, -2.0, 1.41) for freq in EQ_FREQUENCIES])
    calibrated = state.get_eq_settings()
    assert calibrated.correction_bands is not None
    correction = deepcopy(calibrated.correction_bands)
    tone_action(window, "Voice").trigger()
    toned = state.get_eq_settings()
    assert toned.correction_bands == correction
    assert toned.band_gains != calibrated.band_gains
    eq_control(window, "type", 0).setIndex(TYPES.index("high_pass"))
    edit(eq_control(window, "frequency", 0), 125.0)
    eq_control(window, "slope", 0).setIndex(SLOPES.index(48))
    toned = state.get_eq_settings()
    assert toned.correction_bands == correction
    assert toned.bands[0].filter_type == "high_pass"
    assert toned.bands[0].frequency_hz == 125.0
    assert toned.bands[0].slope_db_per_octave == 48
    assert processor.get_eq_band_config(0) == toned.bands[0].to_native()
    restored = Preset.from_dict(Preset(eq=toned).to_dict())
    state.set_settings(restored.eq.to_dict())
    assert state.get_eq_settings().to_dict() == restored.eq.to_dict()
    state.clear_correction()
    assert state.get_eq_settings().correction_bands is None
    assert state.get_eq_settings().bands == toned.tone_bands
