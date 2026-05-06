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
        freq, gain, q = processor.get_eq_band_params(i)
        expected_freq, expected_gain, expected_q = auto_eq_bands[i]
        assert abs(freq - expected_freq) <= 0.1
        assert abs(gain - expected_gain) <= 0.1
        assert abs(q - expected_q) <= 0.1

    # Manual frequency edits should move the active band and marker too.
    panel.band_sliders[5].frequency_spinbox.setValue(2310.0)
    panel.band_sliders[5]._frequency_rate_limiter.flush()
    qapp.processEvents()

    freq, gain, q = processor.get_eq_band_params(5)
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
