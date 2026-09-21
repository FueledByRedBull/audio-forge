"""Character changes retain the calibrated correction and survive presets."""

from copy import deepcopy

from mic_eq.config import EQ_FREQUENCIES, Preset
from mic_eq.mic_eq_core import AudioProcessor
from mic_eq.ui.eq_panel import EQPanel


def test_tone_preserves_correction_roundtrip_and_clear(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    try:
        panel.apply_auto_eq_results([(freq, -2.0, 1.41) for freq in EQ_FREQUENCIES])
        calibrated = panel.get_eq_settings()
        assert calibrated.correction_bands is not None
        correction = deepcopy(calibrated.correction_bands)
        panel._preset_voice()
        toned = panel.get_eq_settings()
        assert toned.correction_bands == correction
        assert toned.band_gains != calibrated.band_gains
        slider = panel.band_sliders[0]
        slider.filter_type_combo.setCurrentIndex(slider.filter_type_combo.findData("high_pass"))
        slider.frequency_spinbox.setValue(125.0)
        slider.slope_combo.setCurrentIndex(slider.slope_combo.findData(48))
        toned = panel.get_eq_settings()
        assert toned.correction_bands == correction
        assert toned.bands[0].filter_type == "high_pass"
        assert toned.bands[0].frequency_hz == 125.0
        assert toned.bands[0].slope_db_per_octave == 48
        assert processor.get_eq_band_config(0) == toned.bands[0].to_native()
        restored = Preset.from_dict(Preset(eq=toned).to_dict())
        panel.set_settings(restored.eq.to_dict())
        assert panel.get_eq_settings().to_dict() == restored.eq.to_dict()
        panel.clear_correction()
        assert panel.get_eq_settings().correction_bands is None
        assert panel.get_eq_settings().bands == toned.tone_bands
    finally:
        panel.close()
        panel.deleteLater()
