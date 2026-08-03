"""Latency calibration dialog for the selected output-to-input route."""

from __future__ import annotations

import json
import logging
import threading
import time
from math import gcd
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)
from scipy.signal import resample_poly

from ..analysis.latency_calibration import (
    analyze_latency,
    generate_probe_signal,
    result_to_profile,
)
from ..config import coerce_device_identity
from .accessibility import set_accessible_group
from .level_meter import LevelMeter
from .device_selection import start_processor_for_route
from .layout_constants import configure_resizable_dialog, create_scrollable_dialog_body


logger = logging.getLogger(__name__)


DEBUG = False


def _device_name(device: object) -> str | None:
    identity = coerce_device_identity(device)
    if identity is not None:
        return identity.name
    return device if isinstance(device, str) and device else None


def _capture_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor.")

    sample_rate = int(owner.processor.sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Processing sample rate is unavailable.")

    return sample_rate


def _output_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor.")
    sample_rate = int(owner.processor.output_sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Output sample rate is unavailable.")
    return sample_rate


def _resample_probe(
    probe: np.ndarray, source_rate: int, output_rate: int
) -> np.ndarray:
    if source_rate == output_rate:
        return np.ascontiguousarray(probe, dtype=np.float32)
    divisor = gcd(source_rate, output_rate)
    resampled = resample_poly(
        probe.astype(np.float64, copy=False),
        output_rate // divisor,
        source_rate // divisor,
    )
    return np.ascontiguousarray(resampled, dtype=np.float32)


def engine_config_signature(processor: Any) -> str:
    diagnostics = dict(processor.get_runtime_diagnostics())
    fields = {
        "limiter_enabled": bool(processor.is_limiter_enabled()),
        "input_fixed_buffer_frames": int(
            diagnostics.get("input_fixed_buffer_frames", 0)
        ),
        "noise_enabled": bool(processor.is_rnnoise_enabled()),
        "noise_model": str(processor.get_noise_model()),
        "output_resampler_active": bool(
            diagnostics.get("output_resampler_active", False)
        ),
        "output_fixed_buffer_frames": int(
            diagnostics.get("output_fixed_buffer_frames", 0)
        ),
        "output_sample_rate": int(processor.output_sample_rate()),
        "processing_sample_rate": int(processor.sample_rate()),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


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
        engine_latency_ms: float = 0.0,
        engine_config_signature: str = "",
    ):
        super().__init__()
        self.probe = probe
        self.recording = recording
        self.sample_rate = sample_rate
        self.expected_playback_start_ms = expected_playback_start_ms
        self.expected_playback_jitter_ms = expected_playback_jitter_ms
        self.engine_latency_ms = engine_latency_ms
        self.engine_config_signature = engine_config_signature
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
                "profile": result_to_profile(
                    analysis,
                    sample_rate=self.sample_rate,
                    engine_latency_ms=self.engine_latency_ms,
                    engine_config_signature=self.engine_config_signature,
                ),
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

        self.worker: LatencyCalibrationWorker | None = None
        self._started_processor = False
        self._latest_profile: dict | None = existing_profile
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(50)
        self._capture_timer.timeout.connect(self._poll_capture)
        self._probe: np.ndarray | None = None
        self._playback_probe: np.ndarray | None = None
        self._capture_started_at = 0.0
        self._capture_sample_rate = 0
        self._recording_duration_s = 2.5
        self._playback_delay_s = 0.45
        self._played_probe = False
        self._probe_started = False
        self._probe_started_at: float | None = None
        self._engine_latency_samples: list[float] = []
        self._engine_signature = ""

        self._setup_ui(existing_profile)
        configure_resizable_dialog(
            self,
            preferred_width=620,
            preferred_height=700,
            minimum_width=460,
            minimum_height=360,
        )

    def _setup_ui(self, existing_profile: dict | None):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.content_scroll_area, layout = create_scrollable_dialog_body(self)
        self.content_scroll_area.setAccessibleName("Latency calibration content")
        outer_layout.addWidget(self.content_scroll_area)

        instructions = QLabel(
            "Run calibration with your current input/output device pair.\n"
            "Best results require a loopback cable or speaker-to-mic route in a quiet room. "
            "Compensation uses the measured route delay directly."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        status_group = QGroupBox("Measured Latency")
        status_layout = QGridLayout(status_group)

        status_layout.addWidget(QLabel("Measured Route:"), 0, 0)
        self.round_trip_label = QLabel("-- ms")
        status_layout.addWidget(self.round_trip_label, 0, 1)

        status_layout.addWidget(QLabel("Directional Estimate:"), 1, 0)
        self.one_way_label = QLabel("-- ms")
        status_layout.addWidget(self.one_way_label, 1, 1)

        status_layout.addWidget(QLabel("Applied Compensation:"), 2, 0)
        self.comp_label = QLabel("-- ms")
        status_layout.addWidget(self.comp_label, 2, 1)

        status_layout.addWidget(QLabel("Confidence:"), 3, 0)
        self.confidence_label = QLabel("--")
        status_layout.addWidget(self.confidence_label, 3, 1)

        status_layout.addWidget(QLabel("Probe Agreement:"), 4, 0)
        self.agreement_label = QLabel("--")
        status_layout.addWidget(self.agreement_label, 4, 1)

        status_layout.addWidget(QLabel("Echo Ambiguity:"), 5, 0)
        self.ambiguity_label = QLabel("--")
        status_layout.addWidget(self.ambiguity_label, 5, 1)

        status_layout.addWidget(QLabel("Engine Latency:"), 6, 0)
        self.engine_label = QLabel("-- ms")
        status_layout.addWidget(self.engine_label, 6, 1)

        status_layout.addWidget(QLabel("Total Latency:"), 7, 0)
        self.total_label = QLabel("-- ms")
        status_layout.addWidget(self.total_label, 7, 1)

        layout.addWidget(status_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAccessibleName("Latency calibration progress")
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setAccessibleName("Latency calibration status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.level_meter = LevelMeter("CAP", show_scale=True)
        self.level_meter.setAccessibleName("Latency calibration capture level")
        self.level_meter.setMinimumHeight(120)
        layout.addWidget(self.level_meter)

        button_row = QGridLayout()

        self.run_button = QPushButton("Run Calibration")
        self.run_button.clicked.connect(self._on_run_clicked)
        button_row.addWidget(self.run_button, 0, 0)

        self.accept_button = QPushButton("Accept")
        self.accept_button.setEnabled(existing_profile is not None)
        self.accept_button.clicked.connect(self._on_accept_clicked)
        button_row.addWidget(self.accept_button, 0, 1)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_row.addWidget(self.reset_button, 1, 0)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self._on_close_clicked)
        button_row.addWidget(self.close_button, 1, 1)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)

        set_accessible_group(
            (
                (self.run_button, "Run latency calibration", None),
                (self.accept_button, "Accept latency calibration", None),
                (self.reset_button, "Reset latency calibration", None),
                (self.close_button, "Close latency calibration", None),
            )
        )
        self.setTabOrder(self.run_button, self.accept_button)
        self.setTabOrder(self.accept_button, self.reset_button)
        self.setTabOrder(self.reset_button, self.close_button)

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
                start_processor_for_route(
                    owner.processor,
                    input_device.currentData() if input_device else None,
                    output_device.currentData() if output_device else None,
                )
                self._started_processor = True
            else:
                self._started_processor = False
        except Exception as e:
            QMessageBox.critical(
                self, "Audio Error", f"Failed to start processing: {e}"
            )
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
            self._playback_probe = _resample_probe(
                self._probe,
                self._capture_sample_rate,
                _output_sample_rate(owner),
            )
            self._engine_signature = engine_config_signature(owner.processor)
            self._engine_latency_samples = []
            owner.processor.start_raw_recording(self._recording_duration_s)
        except Exception as e:
            self._on_worker_failed(
                f"Latency calibration failed: {type(e).__name__}: {e}"
            )
            return

        self._played_probe = False
        self._probe_started = False
        self._probe_started_at = None
        self._capture_started_at = time.monotonic()
        self._capture_timer.start()

    def _poll_capture(self):
        owner = self._get_processor_owner()
        if owner is None:
            self._on_worker_failed("Could not find audio processor.")
            return

        try:
            elapsed = time.monotonic() - self._capture_started_at
            engine_latency = float(owner.processor.get_engine_latency_ms())
            if np.isfinite(engine_latency) and engine_latency >= 0.0:
                self._engine_latency_samples.append(engine_latency)
            if (not self._probe_started) and elapsed >= self._playback_delay_s:
                self.status_label.setText("Playing probe signal...")
                self._probe_started = True
                self._probe_started_at = time.monotonic()
                if self._playback_probe is None:
                    self._on_worker_failed("Probe signal is unavailable.")
                    return
                owner.processor.queue_output_probe(self._playback_probe)

            if self._probe_started and owner.processor.is_output_probe_complete():
                self._played_probe = True

            self._on_level_update(float(owner.processor.recording_level_db()))
            progress = int(min(99.0, (elapsed / self._recording_duration_s) * 100.0))
            self.progress.setValue(progress)

            if (
                elapsed < self._recording_duration_s
                and not owner.processor.is_recording_complete()
            ):
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
                engine_latency_ms=(
                    float(np.median(self._engine_latency_samples))
                    if self._engine_latency_samples
                    else float(owner.processor.get_engine_latency_ms())
                ),
                engine_config_signature=self._engine_signature,
            )
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.failed.connect(self._on_worker_failed)
            self.worker.start()
        except Exception as e:
            self._on_worker_failed(
                f"Latency calibration failed: {type(e).__name__}: {e}"
            )

    def _on_level_update(self, rms_db: float):
        self.level_meter.set_levels(rms_db, rms_db + 6.0)

    def _on_worker_finished(self, payload: dict):
        analysis = payload.get("analysis")
        profile = payload.get("profile")

        self._latest_profile = profile
        self._apply_profile_to_labels(profile)

        self.progress.setValue(100)
        ambiguity = (
            float(profile.get("ambiguity_score", 0.0) or 0.0) if profile else 0.0
        )
        if ambiguity >= 0.75:
            self.status_label.setText(
                "Calibration found the direct path, but echoes are close. Review before accepting."
            )
        else:
            self.status_label.setText(
                "Calibration successful. Review values and Accept."
            )
        self.run_button.setEnabled(True)
        self.accept_button.setEnabled(True)

        if analysis is not None and DEBUG:
            logger.debug(
                "Latency calibration success route=%.2fms conf=%.2f",
                analysis.measured_round_trip_ms,
                analysis.confidence,
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
            self.agreement_label.setText("--")
            self.ambiguity_label.setText("--")
            self.engine_label.setText("-- ms")
            self.total_label.setText("-- ms")
            return

        self.round_trip_label.setText(
            f"{profile.get('measured_round_trip_ms', 0.0):.2f} ms"
        )
        directional = profile.get("directional_latency_ms")
        self.one_way_label.setText(
            f"{float(directional):.2f} ms"
            if directional is not None
            else "Not inferred"
        )
        route_latency = float(profile.get("route_latency_ms", 0.0) or 0.0)
        if route_latency <= 0.0:
            route_latency = float(
                profile.get(
                    "measured_round_trip_ms",
                    profile.get("applied_compensation_ms", 0.0),
                )
                or 0.0
            )
        self.comp_label.setText(f"{route_latency:.2f} ms")
        self.confidence_label.setText(f"{profile.get('confidence', 0.0):.2f}")
        self.agreement_label.setText(
            f"{profile.get('agreement_ms', 0.0):.2f} ms across probes"
        )
        self.ambiguity_label.setText(f"{profile.get('ambiguity_score', 0.0):.2f}")
        self.engine_label.setText(f"{profile.get('engine_latency_ms', 0.0):.2f} ms")
        self.total_label.setText(f"{profile.get('total_latency_ms', 0.0):.2f} ms")

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
        except Exception:
            if DEBUG:
                logger.debug("Stop processor failed", exc_info=True)

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
                owner.processor.cancel_output_probe()
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
        self._probe_started = False
        self._played_probe = False

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
