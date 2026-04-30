"""
Latency calibration dialog.

Runs probe playback + raw input capture, then estimates round-trip latency using
cross-correlation.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import wave
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)

from ..analysis.latency_calibration import analyze_latency, generate_probe_signal, result_to_profile
from ..config import coerce_device_identity
from .level_meter import LevelMeter


DEBUG = False


def _device_name(device: object) -> str | None:
    identity = coerce_device_identity(device)
    if identity is not None:
        return identity.name
    return device if isinstance(device, str) and device else None


def _capture_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor.")

    sample_rate = int(owner.processor.output_sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Output sample rate is unavailable.")

    return sample_rate


def _play_probe_blocking(probe: np.ndarray, sample_rate: int) -> None:
    """Play probe signal using platform-available APIs."""
    if os.name == "nt":
        import winsound

        pcm = np.clip(probe, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16.tobytes())
            # Synchronous playback is the default when SND_ASYNC is not set.
            # Some Python builds do not expose winsound.SND_SYNC.
            play_flags = winsound.SND_FILENAME
            play_flags |= getattr(winsound, "SND_SYNC", 0)
            winsound.PlaySound(wav_path, play_flags)
        finally:
            try:
                os.remove(wav_path)
            except OSError as e:
                if DEBUG:
                    print(f"[LATENCY_CAL] Failed to remove temporary probe file: {e}")
        return

    # Best-effort fallback for non-Windows hosts.
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        for _ in range(3):
            app.beep()
            time.sleep(0.06)


class LatencyCalibrationWorker(QThread):
    """Background worker for CPU-only latency analysis."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        probe: np.ndarray,
        recording: np.ndarray,
        sample_rate: int,
        expected_playback_start_ms: float | None = None,
        expected_playback_jitter_ms: float | None = None,
    ):
        super().__init__()
        self.probe = probe
        self.recording = recording
        self.sample_rate = sample_rate
        self.expected_playback_start_ms = expected_playback_start_ms
        self.expected_playback_jitter_ms = expected_playback_jitter_ms
        self._stop_event = threading.Event()

    def run(self):
        try:
            if self._stop_event.is_set():
                return

            analysis = analyze_latency(
                reference_probe=self.probe,
                recorded_signal=self.recording,
                sample_rate=self.sample_rate,
                min_search_ms=5.0,
                max_search_ms=500.0,
                expected_playback_start_ms=self.expected_playback_start_ms,
                expected_playback_jitter_ms=self.expected_playback_jitter_ms,
            )

            if self._stop_event.is_set():
                return

            if not analysis.success:
                self.failed.emit(analysis.message or "Low confidence latency estimate.")
                return

            payload = {
                "analysis": analysis,
                "profile": result_to_profile(analysis, sample_rate=self.sample_rate),
            }
            self.finished.emit(payload)
        except Exception as e:
            self.failed.emit(f"Latency calibration failed: {type(e).__name__}: {e}")

    def stop(self):
        self._stop_event.set()


class LatencyCalibrationDialog(QDialog):
    """Dialog that runs and applies latency calibration."""

    calibration_saved = pyqtSignal(dict)
    calibration_reset = pyqtSignal()

    def __init__(self, parent=None, existing_profile: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Latency Calibration")
        self.setModal(True)
        self.setMinimumWidth(560)

        self.worker: LatencyCalibrationWorker | None = None
        self._started_processor = False
        self._latest_profile: dict | None = existing_profile
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(50)
        self._capture_timer.timeout.connect(self._poll_capture)
        self._probe: np.ndarray | None = None
        self._capture_started_at = 0.0
        self._capture_sample_rate = 0
        self._recording_duration_s = 2.5
        self._playback_delay_s = 0.45
        self._played_probe = False
        self._probe_started = False
        self._probe_started_at: float | None = None
        self._probe_finished = threading.Event()

        self._setup_ui(existing_profile)

    def _setup_ui(self, existing_profile: dict | None):
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Run calibration with your current input/output device pair.\n"
            "Best results require a loopback cable or speaker-to-mic route in a quiet room."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        status_group = QGroupBox("Measured Latency")
        status_layout = QGridLayout(status_group)

        status_layout.addWidget(QLabel("Round-Trip:"), 0, 0)
        self.round_trip_label = QLabel("-- ms")
        status_layout.addWidget(self.round_trip_label, 0, 1)

        status_layout.addWidget(QLabel("One-Way Estimate:"), 1, 0)
        self.one_way_label = QLabel("-- ms")
        status_layout.addWidget(self.one_way_label, 1, 1)

        status_layout.addWidget(QLabel("Applied Compensation:"), 2, 0)
        self.comp_label = QLabel("-- ms")
        status_layout.addWidget(self.comp_label, 2, 1)

        status_layout.addWidget(QLabel("Confidence:"), 3, 0)
        self.confidence_label = QLabel("--")
        status_layout.addWidget(self.confidence_label, 3, 1)

        layout.addWidget(status_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.level_meter = LevelMeter("CAP", show_scale=True)
        self.level_meter.setMinimumHeight(120)
        layout.addWidget(self.level_meter)

        button_row = QHBoxLayout()

        self.run_button = QPushButton("Run Calibration")
        self.run_button.clicked.connect(self._on_run_clicked)
        button_row.addWidget(self.run_button)

        self.accept_button = QPushButton("Accept")
        self.accept_button.setEnabled(existing_profile is not None)
        self.accept_button.clicked.connect(self._on_accept_clicked)
        button_row.addWidget(self.accept_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_row.addWidget(self.reset_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self._on_close_clicked)
        button_row.addWidget(self.close_button)

        layout.addLayout(button_row)

        if existing_profile:
            self._apply_profile_to_labels(existing_profile)

    def _get_processor_owner(self) -> Any | None:
        parent: Any = self.parent()
        while parent and not hasattr(parent, "processor"):
            parent = parent.parent()
        return parent

    def _on_run_clicked(self):
        owner = self._get_processor_owner()
        if owner is None:
            QMessageBox.critical(self, "Error", "Could not find audio processor.")
            return

        try:
            if not owner.processor.is_running():
                input_device = getattr(owner, "input_combo", None)
                output_device = getattr(owner, "output_combo", None)
                selected_input = _device_name(input_device.currentData()) if input_device else None
                selected_output = _device_name(output_device.currentData()) if output_device else None
                owner.processor.start(selected_input, selected_output)
                self._started_processor = True
            else:
                self._started_processor = False
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Failed to start processing: {e}")
            return

        self.run_button.setEnabled(False)
        self.accept_button.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Preparing probe...")

        try:
            owner.processor.set_recovery_suppressed(True)
            self._capture_sample_rate = _capture_sample_rate(owner)
            self._probe = generate_probe_signal(
                sample_rate=self._capture_sample_rate,
                duration_ms=80.0,
            )
            owner.processor.start_raw_recording(self._recording_duration_s)
        except Exception as e:
            self._on_worker_failed(f"Latency calibration failed: {type(e).__name__}: {e}")
            return

        self._played_probe = False
        self._probe_started = False
        self._probe_started_at = None
        self._probe_finished.clear()
        self._capture_started_at = time.time()
        self._capture_timer.start()

    def _poll_capture(self):
        owner = self._get_processor_owner()
        if owner is None:
            self._on_worker_failed("Could not find audio processor.")
            return

        try:
            elapsed = time.time() - self._capture_started_at
            if (not self._probe_started) and elapsed >= self._playback_delay_s:
                self.status_label.setText("Playing probe signal...")
                self._probe_started = True
                self._probe_started_at = time.time()

                def _play_probe():
                    probe = self._probe
                    if probe is None:
                        self._on_worker_failed("Probe signal is unavailable.")
                        return
                    try:
                        _play_probe_blocking(probe, self._capture_sample_rate)
                    finally:
                        self._played_probe = True
                        self._probe_finished.set()

                threading.Thread(target=_play_probe, daemon=True).start()

            self._on_level_update(float(owner.processor.recording_level_db()))
            progress = int(min(99.0, (elapsed / self._recording_duration_s) * 100.0))
            self.progress.setValue(progress)

            if elapsed < self._recording_duration_s and not owner.processor.is_recording_complete():
                return

            self._capture_timer.stop()

            if not self._played_probe:
                self._on_worker_failed("Probe signal was not played.")
                return

            self.status_label.setText("Analyzing captured signal...")
            raw = owner.processor.stop_raw_recording()
            if raw is None:
                self._on_worker_failed("Failed to capture recording for calibration.")
                return

            recording = np.asarray(raw, dtype=np.float32)
            if self._probe_started_at is not None:
                expected_start_ms = max(
                    0.0, (self._probe_started_at - self._capture_started_at) * 1000.0
                )
            else:
                expected_start_ms = self._playback_delay_s * 1000.0
            expected_jitter_ms = max(50.0, float(self._capture_timer.interval()))
            if self._probe is None:
                self._on_worker_failed("Probe signal is unavailable.")
                return
            self.worker = LatencyCalibrationWorker(
                probe=self._probe,
                recording=recording,
                sample_rate=self._capture_sample_rate,
                expected_playback_start_ms=expected_start_ms,
                expected_playback_jitter_ms=expected_jitter_ms,
            )
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.failed.connect(self._on_worker_failed)
            self.worker.start()
        except Exception as e:
            self._on_worker_failed(f"Latency calibration failed: {type(e).__name__}: {e}")

    def _on_level_update(self, rms_db: float):
        self.level_meter.set_levels(rms_db, rms_db + 6.0)

    def _on_worker_finished(self, payload: dict):
        analysis = payload.get("analysis")
        profile = payload.get("profile")

        self._latest_profile = profile
        self._apply_profile_to_labels(profile)

        self.progress.setValue(100)
        self.status_label.setText("Calibration successful. Review values and Accept.")
        self.run_button.setEnabled(True)
        self.accept_button.setEnabled(True)

        if analysis is not None and DEBUG:
            print(
                "[LATENCY_CAL] success rt=%.2fms one_way=%.2fms conf=%.2f"
                % (
                    analysis.measured_round_trip_ms,
                    analysis.estimated_one_way_ms,
                    analysis.confidence,
                )
            )

        self._teardown_worker()

    def _on_worker_failed(self, message: str):
        self.status_label.setText(message)
        self.run_button.setEnabled(True)
        self.accept_button.setEnabled(self._latest_profile is not None)

        self._teardown_worker()
        self._stop_owned_processor()

    def _apply_profile_to_labels(self, profile: dict | None):
        if not profile:
            self.round_trip_label.setText("-- ms")
            self.one_way_label.setText("-- ms")
            self.comp_label.setText("-- ms")
            self.confidence_label.setText("--")
            return

        self.round_trip_label.setText(f"{profile.get('measured_round_trip_ms', 0.0):.2f} ms")
        self.one_way_label.setText(f"{profile.get('estimated_one_way_ms', 0.0):.2f} ms")
        self.comp_label.setText(f"{profile.get('applied_compensation_ms', 0.0):.2f} ms")
        self.confidence_label.setText(f"{profile.get('confidence', 0.0):.2f}")

    def _on_accept_clicked(self):
        if not self._latest_profile:
            QMessageBox.information(self, "No Result", "Run calibration first.")
            return

        self.calibration_saved.emit(self._latest_profile)
        self.accept()

    def _on_reset_clicked(self):
        self._latest_profile = None
        self._apply_profile_to_labels(None)
        self.accept_button.setEnabled(False)
        self.status_label.setText("Calibration reset. Using estimated latency.")
        self.calibration_reset.emit()

    def _on_close_clicked(self):
        self.reject()

    def _stop_owned_processor(self):
        if not self._started_processor:
            return

        owner = self._get_processor_owner()
        if owner is None:
            return

        try:
            if owner.processor.is_running():
                owner.processor.stop()
        except Exception as e:
            if DEBUG:
                print(f"[LATENCY_CAL] stop processor failed: {type(e).__name__}: {e}")

        self._started_processor = False

    def _teardown_worker(self):
        self._capture_timer.stop()
        owner = self._get_processor_owner()

        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)

        self.worker = None
        if owner is not None:
            try:
                owner.processor.stop_raw_recording()
            except Exception:
                pass
            try:
                owner.processor.set_output_mute(False)
            except Exception:
                pass
            try:
                owner.processor.set_recovery_suppressed(False)
            except Exception:
                pass
        self._probe_finished.clear()
        self._probe_started = False
        self._played_probe = False
        owner = self._get_processor_owner()
        if owner is not None:
            try:
                owner.processor.set_recovery_suppressed(False)
            except Exception:
                pass

    def closeEvent(self, event):
        self._teardown_worker()
        self._stop_owned_processor()
        super().closeEvent(event)

    def reject(self):
        self._teardown_worker()
        self._stop_owned_processor()
        super().reject()

    def accept(self):
        self._teardown_worker()
        self._stop_owned_processor()
        super().accept()
