#!/usr/bin/env python3
"""Test EQ panel UI synchronization with auto-EQ results."""

import sys
from PyQt6.QtWidgets import QApplication
from mic_eq import AudioProcessor
from mic_eq.ui.eq_panel import EQPanel

def test_ui_synchronization():
    app = QApplication(sys.argv)

    # Create processor and panel
    processor = AudioProcessor()
    panel = EQPanel(processor)
    panel.resize(1000, 600)
    panel.show()
    app.processEvents()

    print("EQ Panel displayed with initial flat response.")
    print("Press Enter to apply auto-EQ results...")
    input()

    # Simulate auto-EQ results (typical broadcast curve)
    auto_eq_bands = [
        (80.0, -2.5, 0.7),     # Low shelf cut
        (160.0, -1.2, 1.0),    # Low-mid cut
        (320.0, 0.0, 1.2),     # Mid transition
        (640.0, 1.8, 1.4),     # Lower-mid boost
        (1280.0, 3.0, 1.6),    # Mid boost
        (2500.0, 4.5, 2.0),    # Upper-mid boost (presence)
        (5000.0, 2.5, 1.8),    # Lower-treble boost
        (8000.0, 1.0, 1.2),    # Treble slight boost
        (12000.0, -0.5, 0.9),  # High-treble cut
        (16000.0, -1.5, 0.7),  # High shelf cut
    ]

    # Apply auto-EQ results
    try:
        panel.apply_auto_eq_results(auto_eq_bands)
        print("PASS: apply_auto_eq_results() succeeded")
        app.processEvents()

        # Verify UI sliders updated
        for i, (freq, expected_gain, expected_q) in enumerate(auto_eq_bands):
            slider = panel.band_sliders[i]
            actual_gain = slider.slider.value() / 10.0
            actual_q = slider.q_spinbox.value()

            if abs(actual_gain - expected_gain) > 0.1:
                print(f"FAIL: Band {i} gain mismatch: expected {expected_gain}, got {actual_gain}")
                sys.exit(1)
            if abs(actual_q - expected_q) > 0.1:
                print(f"FAIL: Band {i} Q mismatch: expected {expected_q}, got {actual_q}")
                sys.exit(1)

        print("PASS: All UI sliders updated correctly")

        # Verify processor was updated
        for i in range(10):
            freq, gain, q = processor.get_eq_band_params(i)
            expected_freq, expected_gain, expected_q = auto_eq_bands[i]
            if abs(gain - expected_gain) > 0.1:
                print(f"FAIL: Processor band {i} gain mismatch")
                sys.exit(1)
            if abs(q - expected_q) > 0.1:
                print(f"FAIL: Processor band {i} Q mismatch")
                sys.exit(1)

        print("PASS: Processor updated correctly")
        print("\nAll tests passed! Close window to exit.")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(app.exec())

if __name__ == "__main__":
    test_ui_synchronization()
