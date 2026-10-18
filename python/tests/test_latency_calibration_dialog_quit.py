"""Quitting the application must not destroy a latency analysis that is still running."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import numpy as np
from PySide6.QtWidgets import QWidget

from mic_eq.ui import latency_calibration_dialog as latency_ui


def test_application_quit_waits_for_a_running_latency_analysis(qapp, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def slow_analysis(**_kwargs):
        entered.set()
        release.wait(5.0)
        return None

    monkeypatch.setattr(latency_ui, "analyze_latency", slow_analysis)
    owner = QWidget()
    processor = Mock()
    setattr(owner, "processor", processor)
    processor.get_engine_latency_ms.return_value = 0.0
    processor.is_output_probe_complete.return_value = True
    processor.recording_level_db.return_value = -40.0
    processor.is_recording_complete.return_value = True
    processor.stop_raw_recording.return_value = np.zeros(48_000, dtype=np.float32)
    dialog = latency_ui.LatencyCalibrationDialog(owner)
    dialog._probe = np.zeros(64, dtype=np.float32)
    dialog._probe_started = True
    dialog._played_probe = True
    dialog._capture_sample_rate = 48_000
    dialog._capture_started_at = time.monotonic() - 3.0
    worker = None
    # Emitting the shared app's aboutToQuit would reach every other live object, so only
    # check that the dialog is connected, then run its slot as Qt would.
    try:
        assert qapp.aboutToQuit.disconnect(dialog._wait_for_worker)
        dialog._poll_capture()
        worker = dialog.worker
        assert worker is not None and entered.wait(5.0)
        releaser = threading.Timer(0.3, release.set)
        releaser.start()

        dialog._wait_for_worker()

        assert not worker.isRunning()
        assert dialog.worker is None
    finally:
        release.set()
        if worker is not None:
            worker.wait(2_000)
        dialog.close()
        owner.close()
