# pyright: reportCallIssue=false, reportArgumentType=false
"""Classic EQ panel widget specifics.

EQ editing and the shared EQ presentation are covered by test_eq_editing_state.
"""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QDoubleSpinBox, QSlider, QWidget

from mic_eq.ui.eq_panel import EQBandSlider, EQPanel, EQPresentation
from mic_eq.ui.eq_state import EQState


def test_bass_cut_description_matches_shelf_contour():
    description = EQPanel._preset_bass_cut.__doc__ or ""
    assert "low-shelf" in description
    assert "high-pass" in description


def test_eq_presentation_has_no_numeric_editors_and_fallback_adopts_same_graph(qapp):
    from unittest.mock import Mock

    native = Mock()
    owner = QWidget()
    state = EQState(native, owner)
    settings = state.get_eq_settings()
    payload = settings.to_dict()
    payload["bands"][4].update(filter_type="notch", frequency_hz=2345.678, q=3.4567)
    state.set_settings(payload)
    native.reset_mock()
    presentation = EQPresentation(state, owner)
    assert not owner.findChildren(EQBandSlider)
    assert not owner.findChildren(QDoubleSpinBox)
    assert not owner.findChildren(QSlider)
    assert not native.mock_calls
    panel = EQPanel(native, state, presentation=presentation)
    assert panel.curve_widget is presentation.curve_widget
    assert panel.tone_preset_button is presentation.tone_preset_button
    assert not native.mock_calls
    assert state.get_band(4).filter_type == "notch"
    assert state.get_band(4).frequency_hz == 2345.678
    assert state.get_band(4).q == 3.4567
    with QSignalBlocker(panel.band_sliders[4].frequency_spinbox):
        panel.band_sliders[4].frequency_spinbox.setValue(7000)
    assert state.get_band(4).frequency_hz == 2345.678
    state.changed.emit()
    assert panel.band_sliders[4].frequency_spinbox.value() == 2346
    panel.deleteLater()
    owner.deleteLater()
    qapp.processEvents()


def test_standalone_eq_band_keeps_its_parameter_write_contract(qapp):
    from unittest.mock import Mock

    native = Mock()
    band = EQBandSlider(4, 1280.0, native)
    band.slider.setValue(23)
    band._rate_limiter.flush()
    native.set_eq_band_gain.assert_called_once_with(4, 2.3)
    native.apply_eq_layers.assert_not_called()
    band.set_q(2.345678)
    band.set_frequency(2345.678)
    assert band.settings().q == 2.345678
    assert band.frequency_hz() == 2345.678
    native.set_eq_band_q.assert_not_called()
    band.deleteLater()
    qapp.processEvents()
