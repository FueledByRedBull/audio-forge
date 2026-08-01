# pyright: reportCallIssue=false, reportArgumentType=false
"""Test EQ panel UI synchronization with auto-EQ results.

PyQt6's QTest stubs expose QWindow overloads but omit the runtime QWidget
overloads exercised here.
"""

from typing import Any, cast

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QSignalSpy, QTest

from mic_eq import AudioProcessor
from mic_eq.config import EQSettings, Preset, q_from_bandwidth_octaves
from mic_eq.ui.eq_panel import EQPanel


def _close_panel(panel: EQPanel, processor: AudioProcessor, qapp) -> None:
    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_ui_synchronization(qapp):
    # Create processor and panel.
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.resize(1000, 600)
    panel.show()
    qapp.processEvents()

    # Simulate auto-EQ results (typical broadcast curve).
    auto_eq_bands = [
        (72.0, -2.5, 0.7),
        (145.0, -1.2, 1.0),
        (290.0, 0.0, 1.2),
        (580.0, 1.8, 1.4),
        (1150.0, 3.0, 1.6),
        (2300.0, 4.5, 2.0),
        (4600.0, 2.5, 1.8),
        (7600.0, 1.0, 1.2),
        (11100.0, -0.5, 0.9),
        (15100.0, -1.5, 0.7),
    ]

    panel.apply_auto_eq_results(auto_eq_bands)
    qapp.processEvents()

    assert panel.curve_widget.band_markers == [
        freq for freq, _gain, _q in auto_eq_bands
    ]

    # Verify UI sliders updated.
    for i, (expected_freq, expected_gain, expected_q) in enumerate(auto_eq_bands):
        slider = panel.band_sliders[i]
        actual_gain = slider.slider.value() / 10.0
        actual_q = slider.q_spinbox.value()
        actual_freq = slider.frequency_spinbox.value()
        assert abs(actual_gain - expected_gain) <= 0.1
        assert abs(actual_q - expected_q) <= 0.1
        assert abs(actual_freq - expected_freq) <= 0.1
        assert panel.band_freqs_hz[i] == expected_freq
        assert slider.freq_label.text()

    # Verify processor state updated.
    for i in range(10):
        params = processor.get_eq_band_params(i)
        assert params is not None
        freq, gain, q = params
        expected_freq, expected_gain, expected_q = auto_eq_bands[i]
        assert abs(freq - expected_freq) <= 0.1
        assert abs(gain - expected_gain) <= 0.1
        assert abs(q - expected_q) <= 0.1

    # Manual frequency edits should move the active band and marker too.
    panel.band_sliders[5].frequency_spinbox.setValue(2310.0)
    panel.band_sliders[5]._frequency_rate_limiter.flush()
    qapp.processEvents()

    params = processor.get_eq_band_params(5)
    assert params is not None
    freq, gain, q = params
    assert abs(freq - 2310.0) <= 0.1
    assert abs(gain - auto_eq_bands[5][1]) <= 0.1
    assert abs(q - auto_eq_bands[5][2]) <= 0.1
    assert abs(panel.band_freqs_hz[5] - 2310.0) <= 0.1
    assert abs(panel.get_settings()["band_freqs"][5] - 2310.0) <= 0.1
    assert abs(panel.curve_widget.band_markers[5] - 2310.0) <= 0.1

    _close_panel(panel, processor, qapp)


def test_eq_graph_drag_updates_numeric_controls_native_dsp_and_signals(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.resize(1200, 700)
    panel.show()
    qapp.processEvents()
    curve = panel.curve_widget
    qt_target = cast(Any, curve)
    started = QSignalSpy(panel.configurationEditStarted)
    finished = QSignalSpy(panel.configurationEditFinished)
    band_index = 4
    start_x, start_y = curve.band_handle_position(band_index)
    target = QPoint(
        int(round(curve.frequency_to_x(2100.0))),
        int(round(curve.gain_to_y(3.2))),
    )
    expected_frequency = curve.x_to_frequency(target.x())
    expected_gain = curve.y_to_gain(target.y())

    QTest.mousePress(
        qt_target,
        Qt.MouseButton.LeftButton,
        pos=QPoint(int(round(start_x)), int(round(start_y))),
    )
    QTest.mouseMove(qt_target, target, delay=5)
    QTest.mouseRelease(qt_target, Qt.MouseButton.LeftButton, pos=target)
    qapp.processEvents()

    band = panel.band_sliders[band_index]
    assert band.frequency_spinbox.value() == pytest.approx(
        expected_frequency,
        abs=1.0,
    )
    assert band.slider.value() / 10.0 == pytest.approx(expected_gain, abs=0.1)
    native = processor.get_eq_band_config(band_index)
    assert native is not None
    assert native[1] == pytest.approx(expected_frequency, abs=1.0)
    assert native[2] == pytest.approx(expected_gain, abs=0.1)
    assert len(started) == 1
    assert len(finished) == 1
    assert finished[0][0] == "EQ graph edit"

    _close_panel(panel, processor, qapp)


def test_eq_graph_pass_filter_is_horizontal_only_and_clamped(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.resize(900, 600)
    panel.show()
    qapp.processEvents()
    band_index = 3
    band = panel.band_sliders[band_index]
    band.set_filter_type("high_pass")
    band.set_gain(5.0)
    panel._apply_typed_bands(panel.get_eq_settings().bands)
    curve = panel.curve_widget
    qt_target = cast(Any, curve)
    start_x, start_y = curve.band_handle_position(band_index)
    target = QPoint(curve.width() + 100, -100)

    QTest.mousePress(
        qt_target,
        Qt.MouseButton.LeftButton,
        pos=QPoint(int(round(start_x)), int(round(start_y))),
    )
    QTest.mouseMove(qt_target, target, delay=5)
    QTest.mouseRelease(qt_target, Qt.MouseButton.LeftButton, pos=target)
    qapp.processEvents()

    native = processor.get_eq_band_config(band_index)
    assert native is not None
    assert native[0] == "high_pass"
    assert native[1] == pytest.approx(20_000.0)
    assert native[2] == pytest.approx(5.0)
    assert band.slider.value() / 10.0 == pytest.approx(5.0)

    _close_panel(panel, processor, qapp)


def test_eq_graph_keyboard_edit_and_resize_mapping_are_deterministic(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.resize(1000, 600)
    panel.show()
    qapp.processEvents()
    curve = panel.curve_widget
    qt_target = cast(Any, curve)

    for width in (320, 800, 1600):
        curve.resize(width, 100)
        for frequency in (20.0, 80.0, 1000.0, 20_000.0):
            assert curve.x_to_frequency(
                curve.frequency_to_x(frequency)
            ) == pytest.approx(frequency, abs=1.0)
    assert curve.x_to_frequency(-1000.0) == 20.0
    assert curve.x_to_frequency(100_000.0) == 20_000.0
    assert curve.y_to_gain(-1000.0) == 12.0
    assert curve.y_to_gain(100_000.0) == -12.0

    curve.setFocus()
    QTest.keyClick(qt_target, Qt.Key.Key_BracketRight)
    before = panel.band_sliders[0].frequency_spinbox.value()
    QTest.keyClick(qt_target, Qt.Key.Key_Right)
    QTest.keyClick(qt_target, Qt.Key.Key_Up)
    qapp.processEvents()

    assert panel.band_sliders[0].frequency_spinbox.value() > before
    assert panel.band_sliders[0].slider.value() == 1

    _close_panel(panel, processor, qapp)


def test_eq_curve_interaction_warnings(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.show()
    qapp.processEvents()

    panel._apply_preset([0.0] * 10, [1.41] * 10)
    qapp.processEvents()
    assert panel.curve_widget.interaction_warnings == []

    risky_bands = [
        (80.0, 0.0, 1.0),
        (160.0, 0.0, 1.0),
        (300.0, 6.0, 4.5),
        (340.0, 6.0, 4.5),
        (1280.0, 0.0, 1.0),
        (2500.0, 0.0, 1.0),
        (5000.0, 0.0, 1.0),
        (8000.0, 0.0, 1.0),
        (12000.0, 0.0, 1.0),
        (16000.0, 0.0, 1.0),
    ]
    panel.apply_auto_eq_results(risky_bands)
    qapp.processEvents()

    assert panel.curve_widget.interaction_warnings

    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_auto_eq_diagnostics_are_shown_in_eq_panel(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.show()
    qapp.processEvents()

    bands = [
        (80.0, 0.0, 1.0),
        (160.0, 0.0, 1.0),
        (320.0, 0.0, 1.0),
        (640.0, 0.0, 1.0),
        (1280.0, 0.0, 1.0),
        (2500.0, 0.0, 1.0),
        (5000.0, 0.0, 1.0),
        (8000.0, 0.0, 1.0),
        (12000.0, 0.0, 1.0),
        (16000.0, 0.0, 1.0),
    ]
    diagnostics = {
        "analysis_confidence": 0.82,
        "eq_confidence": 0.76,
        "capture_confidence": 0.88,
        "validation_confidence": 0.79,
        "validation_before_error_db": 4.2,
        "validation_after_error_db": 2.1,
        "validation_gain_scale": 0.85,
        "target_profile": "broadcast:adaptive",
        "band_confidences": [0.8] * 10,
        "recommendation_status": "apply",
        "low_confidence_active_bands": 0,
    }

    panel.apply_auto_eq_results(bands, diagnostics=diagnostics)
    qapp.processEvents()

    assert "overall 82%" in panel.auto_eq_diag_label.text()
    assert "Correction ready" in panel.auto_eq_diag_label.text()
    assert "EQ 76%" in panel.auto_eq_diag_label.text()
    assert "4.2 dB -> 2.1 dB" in panel.auto_eq_diag_label.text()

    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_manual_filter_type_slope_and_bypass_sync_to_native_and_curve(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.show()
    qapp.processEvents()
    band = panel.band_sliders[4]

    band.filter_type_combo.setCurrentIndex(
        band.filter_type_combo.findData("notch")
    )
    band.frequency_spinbox.setValue(1000.0)
    band._frequency_rate_limiter.flush()
    band.q_spinbox.setValue(8.0)
    band._rate_limiter.flush()
    qapp.processEvents()

    config = processor.get_eq_band_config(4)
    assert config == ("notch", 1000.0, 0.0, 8.0, 12, True)
    assert not band.slider.isEnabled()
    assert band.q_spinbox.isEnabled()
    assert not band.slope_combo.isEnabled()
    assert min(panel.curve_widget.response_db) < -20.0

    band.filter_type_combo.setCurrentIndex(
        band.filter_type_combo.findData("high_pass")
    )
    band.slope_combo.setCurrentIndex(band.slope_combo.findData(48))
    qapp.processEvents()

    config = processor.get_eq_band_config(4)
    assert config == ("high_pass", 1000.0, 0.0, 8.0, 48, True)
    assert not band.slider.isEnabled()
    assert not band.q_spinbox.isEnabled()
    assert band.slope_combo.isEnabled()

    band.band_enabled_checkbox.setChecked(False)
    qapp.processEvents()
    config = processor.get_eq_band_config(4)
    assert config is not None
    assert config[5] is False
    assert max(abs(value) for value in panel.curve_widget.response_db) < 1e-9

    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_typed_eq_settings_round_trip_through_ui_and_preset(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    typed = EQSettings().to_dict()
    typed["bands"][0].update(
        {
            "filter_type": "high_pass",
            "frequency_hz": 70.0,
            "slope_db_per_octave": 36,
        }
    )
    typed["bands"][4].update(
        {
            "filter_type": "notch",
            "frequency_hz": 2100.0,
            "q": 7.5,
        }
    )
    exact_frequency = 2345.678
    exact_gain = 1.234
    bandwidth_octaves = 1.234
    exact_q = q_from_bandwidth_octaves(
        exact_frequency,
        bandwidth_octaves,
    )
    typed["bands"][5].update(
        {
            "filter_type": "bell",
            "frequency_hz": exact_frequency,
            "gain_db": exact_gain,
            "q": exact_q,
            "bandwidth_mode": "octaves",
            "bandwidth_octaves": bandwidth_octaves,
        }
    )
    typed["bands"][9]["enabled"] = False

    panel.set_settings(typed)
    qapp.processEvents()
    settings = panel.get_eq_settings()
    restored = Preset.from_dict(Preset(name="Typed", eq=settings).to_dict())

    assert restored.eq == settings
    assert settings.bands[0].filter_type == "high_pass"
    assert settings.bands[0].slope_db_per_octave == 36
    assert settings.bands[4].filter_type == "notch"
    assert settings.bands[4].q == 7.5
    assert settings.bands[5].frequency_hz == exact_frequency
    assert settings.bands[5].gain_db == exact_gain
    assert settings.bands[5].q == exact_q
    assert settings.bands[5].bandwidth_mode == "octaves"
    assert settings.bands[5].bandwidth_octaves == bandwidth_octaves
    assert not settings.bands[9].enabled
    low = processor.get_eq_band_config(0)
    middle = processor.get_eq_band_config(4)
    high = processor.get_eq_band_config(9)
    assert low is not None
    assert middle is not None
    assert high is not None
    assert low[0] == "high_pass"
    assert middle[0] == "notch"
    precise = processor.get_eq_band_config(5)
    assert precise is not None
    assert precise[1] == pytest.approx(exact_frequency, abs=1e-9)
    assert precise[2] == pytest.approx(exact_gain, abs=1e-9)
    assert precise[3] == pytest.approx(exact_q, abs=1e-9)
    assert high[5] is False

    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_auto_eq_explicitly_restores_historical_filter_layout(qapp):
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.band_sliders[4].filter_type_combo.setCurrentIndex(
        panel.band_sliders[4].filter_type_combo.findData("notch")
    )
    auto_eq_bands = [
        (frequency, 0.0, 1.41)
        for frequency in (
            80.0,
            160.0,
            320.0,
            640.0,
            1280.0,
            2500.0,
            5000.0,
            8000.0,
            12_000.0,
            16_000.0,
        )
    ]

    panel.apply_auto_eq_results(auto_eq_bands)
    qapp.processEvents()

    low = processor.get_eq_band_config(0)
    middle = processor.get_eq_band_config(4)
    high = processor.get_eq_band_config(9)
    assert low is not None
    assert middle is not None
    assert high is not None
    assert low[0] == "low_shelf"
    assert middle[0] == "bell"
    assert high[0] == "high_shelf"

    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()
