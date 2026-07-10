"""Test EQ panel UI synchronization with auto-EQ results."""

from mic_eq import AudioProcessor
from mic_eq.ui.eq_panel import EQPanel


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

    assert panel.curve_widget.band_markers == [freq for freq, _gain, _q in auto_eq_bands]

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

    # Cleanup to avoid Qt/native teardown crashes across tests.
    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


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
    }

    panel.apply_auto_eq_results(bands, diagnostics=diagnostics)
    qapp.processEvents()

    assert "overall 82%" in panel.auto_eq_diag_label.text()
    assert "EQ 76%" in panel.auto_eq_diag_label.text()
    assert "4.2 dB -> 2.1 dB" in panel.auto_eq_diag_label.text()

    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()
