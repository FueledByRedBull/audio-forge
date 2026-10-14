# pyright: reportCallIssue=false, reportArgumentType=false
"""Classic EQ panel widget specifics, plus graph-follows-processor and band-reset behaviour.

EQ editing through the shared state is covered by test_eq_editing_state.
"""

from dataclasses import replace

import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDoubleSpinBox, QSlider, QWidget

from mic_eq import AudioProcessor
from mic_eq.config import EQSettings
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


def test_graph_follows_the_processing_mode_the_processor_runs(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    model = presentation.graph_model

    assert model.engine_mode == "normal"
    processor.set_bypass(True)
    QTest.qWait(450)
    assert model.engine_mode == "bypass"
    # Raw monitor wins over bypass, as the DSP loop decides it.
    processor.set_raw_monitor_enabled(True)
    presentation.sync_input_conditioning()
    assert model.engine_mode == "raw"
    processor.set_raw_monitor_enabled(False)
    presentation.sync_input_conditioning()
    assert model.engine_mode == "bypass"
    processor.set_bypass(False)
    presentation.sync_input_conditioning()
    assert model.engine_mode == "normal"

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_graph_ignores_a_processor_that_does_not_report_a_mode(qapp):
    from unittest.mock import Mock

    presentation = EQPresentation(EQState(Mock()))

    presentation.sync_input_conditioning()

    assert presentation.graph_model.engine_mode == "normal"
    presentation.deleteLater()
    qapp.processEvents()


def test_reset_band_restores_one_bands_defaults_and_keeps_it_on_or_off(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    state.apply_auto_eq_results([(100.0 * (i + 1), 1.0, 1.0) for i in range(10)])
    state.set_band_value(2, "gain_db", 4.0, edited=False)
    state.set_band_value(4, "filter_type", "notch", edited=False)
    state.set_band_value(4, "frequency_hz", 777.0, edited=False)
    state.set_band_value(4, "q", 5.0, edited=False)
    state.set_band_value(4, "enabled", False, edited=False)
    events = []
    state.configurationEditStarted.connect(lambda: events.append("start"))
    state.configurationEditFinished.connect(events.append)
    correction = state.get_eq_settings().correction_bands

    state.reset_band(4)

    assert state.get_band(4) == replace(EQSettings().bands[4], enabled=False)
    assert state.get_band(2).gain_db == 4.0
    assert state.get_eq_settings().correction_bands == correction
    assert processor.get_eq_band_config(4) == ("bell", 1280.0, 0.0, 1.41, 12, False)
    assert events == ["start", "Reset EQ band"]
    state.set_band_value(2, "enabled", False, edited=False)
    state.reset_band(2)
    state.set_band_value(5, "gain_db", 2.0, edited=False)
    state.reset_band(5)
    assert not state.get_band(2).enabled and state.get_band(5).enabled
    with pytest.raises(ValueError):
        state.reset_band(10)
    processor.stop()


def test_resetting_a_band_is_one_undo_step_and_a_default_band_records_nothing(
    isolated_main_window,
):
    window = isolated_main_window
    state = window.eq_state
    state.set_band_value(4, "gain_db", 3.0)
    window._commit_pending_configuration_snapshot()
    assert window.current_preset_modified is True
    history = window._configuration_history
    before = history.size
    model = window.eq_presentation.graph_model
    x, y = model.band_handle_position(4, 640, 260)
    state.reset_band(2)
    assert history.size == before

    assert model.request_band_reset(x, y, 640, 260)

    assert state.get_band(4) == EQSettings().bands[4]
    assert history.size == before + 1
    assert history.current is not None and history.current.label == "Reset EQ band"
    # The only difference from the loaded sound is gone.
    assert window.current_preset_modified is False
    window.undo_configuration()
    assert state.get_band(4).gain_db == 3.0
    assert window.current_preset_modified is True
