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
        (80.0, -2.5, 0.7),
        (160.0, -1.2, 1.0),
        (320.0, 0.0, 1.2),
        (640.0, 1.8, 1.4),
        (1280.0, 3.0, 1.6),
        (2500.0, 4.5, 2.0),
        (5000.0, 2.5, 1.8),
        (8000.0, 1.0, 1.2),
        (12000.0, -0.5, 0.9),
        (16000.0, -1.5, 0.7),
    ]

    panel.apply_auto_eq_results(auto_eq_bands)
    qapp.processEvents()

    # Verify UI sliders updated.
    for i, (_, expected_gain, expected_q) in enumerate(auto_eq_bands):
        slider = panel.band_sliders[i]
        actual_gain = slider.slider.value() / 10.0
        actual_q = slider.q_spinbox.value()
        assert abs(actual_gain - expected_gain) <= 0.1
        assert abs(actual_q - expected_q) <= 0.1

    # Verify processor state updated.
    for i in range(10):
        _, gain, q = processor.get_eq_band_params(i)
        _, expected_gain, expected_q = auto_eq_bands[i]
        assert abs(gain - expected_gain) <= 0.1
        assert abs(q - expected_q) <= 0.1

    # Cleanup to avoid Qt/native teardown crashes across tests.
    try:
        processor.stop()
    except Exception:
        pass
    panel.close()
    panel.deleteLater()
    qapp.processEvents()
