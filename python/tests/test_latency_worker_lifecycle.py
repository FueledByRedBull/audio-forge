"""A cancelled latency analysis must finish before its dialog can close."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget

from mic_eq.ui import latency_calibration_dialog as latency_ui


def test_cancel_keeps_running_latency_worker_owned_until_it_stops(qapp, monkeypatch):
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
    dialog.show()
    worker = None
    ui_ticks = 0

    def tick():
        nonlocal ui_ticks
        ui_ticks += 1

    timer = QTimer(dialog)
    timer.setInterval(10)
    timer.timeout.connect(tick)
    try:
        dialog._poll_capture()
        worker = dialog.worker
        assert worker is not None and entered.wait(1.0)

        timer.start()
        started = time.monotonic()
        dialog.reject()
        assert time.monotonic() - started < 0.25

        assert worker.isRunning()
        assert dialog.worker is worker
        assert dialog.isVisible()
        assert dialog.status_label.text() == "Canceling latency analysis..."

        # The old implementation blocked here for 1.5 seconds in reject().
        deadline = time.monotonic() + 1.65
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert ui_ticks >= 50
        assert worker.isRunning()
        assert dialog.worker is worker
        assert dialog.isVisible()
        assert dialog.status_label.text() == "Canceling latency analysis..."

        release.set()
        deadline = time.monotonic() + 2.0
        while dialog.isVisible() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not dialog.isVisible()
    finally:
        timer.stop()
        release.set()
        if worker is not None:
            worker.wait(2_000)
        dialog.close()
        owner.close()


def test_latency_meter_shows_measured_rms_without_inventing_peak(qapp):
    owner = QWidget()
    setattr(owner, "processor", Mock())
    dialog = latency_ui.LatencyCalibrationDialog(owner)
    try:
        dialog._on_level_update(-20.0)

        assert dialog.level_meter.rms_db == -20.0
        assert dialog.level_meter.peak_db == dialog.level_meter.DB_MIN
        assert dialog.level_meter.peak_hold_db == dialog.level_meter.DB_MIN
        assert dialog.level_meter.is_clipping is False
    finally:
        dialog.close()
        owner.close()
