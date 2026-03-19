"""Tests for UI sample-rate propagation and diagnostics formatting."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QWidget

from mic_eq.ui.calibration_dialog import CalibrationDialog, _processor_sample_rate
from mic_eq.ui.latency_calibration_dialog import _capture_sample_rate
from mic_eq.ui.main_window import MainWindow


class _SignalStub:
    def connect(self, _slot):
        return None


class _CaptureWorkerStub:
    last_init: dict | None = None

    def __init__(self, audio_data, sample_rate, target_preset):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.target_preset = target_preset
        _CaptureWorkerStub.last_init = {
            "audio_len": len(audio_data),
            "sample_rate": sample_rate,
            "target_preset": target_preset,
        }
        self.step_progress = _SignalStub()
        self.finished = _SignalStub()
        self.failed = _SignalStub()

    def start(self):
        return None

    def isRunning(self):
        return False

    def stop(self):
        return None

    def wait(self, _timeout=None):
        return True


class _FakeProcessor:
    def __init__(self, sample_rate: int = 44_100, output_sample_rate: int = 48_000):
        self._sample_rate = sample_rate
        self._output_sample_rate = output_sample_rate

    def sample_rate(self) -> int:
        return self._sample_rate

    def output_sample_rate(self) -> int:
        return self._output_sample_rate


class _FakeOwner(QWidget):
    def __init__(self, processor: _FakeProcessor):
        super().__init__()
        self.processor = processor


def test_calibration_analysis_uses_processor_sample_rate(qapp, monkeypatch):
    owner = _FakeOwner(_FakeProcessor(sample_rate=44_100))
    dialog = CalibrationDialog(parent=owner)
    dialog.audio_data = np.ones(256, dtype=np.float32)
    monkeypatch.setattr("mic_eq.ui.calibration_dialog.AnalysisWorker", _CaptureWorkerStub)

    dialog._start_analysis()

    assert _CaptureWorkerStub.last_init is not None
    assert _CaptureWorkerStub.last_init["sample_rate"] == 44_100
    assert dialog.analysis_worker is not None
    assert dialog.analysis_worker.sample_rate == 44_100

    dialog.close()
    owner.close()


def test_calibration_recorded_audio_reports_processor_rate(qapp):
    owner = _FakeOwner(_FakeProcessor(sample_rate=44_100))
    dialog = CalibrationDialog(parent=owner)
    dialog.audio_data = np.ones(32, dtype=np.float32)

    audio, sample_rate = dialog.get_recorded_audio()

    assert audio is not None
    assert sample_rate == 44_100

    dialog.close()
    owner.close()


def test_latency_calibration_uses_processor_output_sample_rate(qapp):
    owner = _FakeOwner(_FakeProcessor(output_sample_rate=44_100))
    assert _capture_sample_rate(owner) == 44_100
    owner.close()


def test_main_window_diagnostics_include_new_metrics():
    diagnostics = {
        "input_dropped_samples": 12,
        "lock_contention_count": 1,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 2,
        "output_underrun_total": 3,
        "output_recovery_count": 4,
        "input_backlog_recovery_count": 5,
        "input_backlog_dropped_samples": 6,
        "clip_event_count": 7,
        "clip_peak_db": -1.25,
        "input_resampler_active": True,
        "output_resampler_active": False,
    }

    dropped_bits: list[str] = []
    MainWindow._extend_diag_tokens(
        dropped_bits,
        diagnostics,
        [
            ("input_backlog_recovery_count", "IBR"),
            ("input_backlog_dropped_samples", "IBD"),
            ("clip_event_count", "CL"),
            ("clip_peak_db", "PK"),
        ],
    )
    backend_bits = ["rnnoise"]
    MainWindow._extend_diag_tokens(
        backend_bits,
        diagnostics,
        [
            ("input_resampler_active", "IR"),
            ("output_resampler_active", "OR"),
        ],
    )

    assert "IBR:5" in dropped_bits
    assert "IBD:6" in dropped_bits
    assert "CL:7" in dropped_bits
    assert "PK:-1.2" in dropped_bits
    assert "IR:Y" in backend_bits
    assert "OR:N" in backend_bits
