"""Latency calibration dialog for the selected output-to-input route."""

from __future__ import annotations

import json
import logging
import threading
import time
from math import gcd
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
    QVBoxLayout,
)
from ..analysis.latency_calibration import (
    analyze_latency,
    generate_probe_signal,
    result_to_profile,
)
from ..analysis.signal_processing import resample_poly
from ..runtime_snapshot import RuntimeSnapshot
from .components import session_is_ending
from .latency_calibration_view import NO_READINGS, LatencyCalibrationView
from .capture_session import (
    CaptureSession,
    capture_route_context,
    find_processor_owner,
    known_capture_format_context as _known_capture_format_context,
    processor_output_sample_rate as _output_sample_rate,
    processor_sample_rate as _capture_sample_rate,
)
from .engine_session import AudioLease, AudioUnavailable, EngineBusy, RouteMismatch
from .layout_constants import configure_resizable_dialog
from .quick_scene import QML_DIR, load_scene, quick_theme
from .quick_shell import register_qml_types


logger = logging.getLogger(__name__)


DEBUG = False


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
    snapshot = RuntimeSnapshot.parse(processor.get_runtime_diagnostics())
    fields = {
        "limiter_enabled": bool(processor.is_limiter_enabled()),
        "input_fixed_buffer_frames": int(snapshot.input_fixed_buffer_frames),
        "noise_enabled": bool(processor.is_rnnoise_enabled()),
        "noise_model": str(processor.get_noise_model()),
        "output_resampler_active": snapshot.output_resampler_active,
        "output_fixed_buffer_frames": int(snapshot.output_fixed_buffer_frames),
        "output_sample_rate": int(processor.output_sample_rate()),
        "processing_sample_rate": int(processor.sample_rate()),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


class LatencyCalibrationWorker(QThread):
    """Background worker for CPU-only latency analysis."""

    result_ready = Signal(dict)
    failed = Signal(str)

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
            self.result_ready.emit(payload)
        except Exception as e:
            logger.exception("Latency calibration analysis failed")
            self.failed.emit(f"Latency calibration failed: {type(e).__name__}: {e}")

    def stop(self):
        self._stop_event.set()


class LatencyCalibrationLoadError(RuntimeError):
    """The dialog's Qt Quick scene did not load, so there is nothing to show."""


class LatencyCalibrationDialog(QDialog):
    """Dialog that runs and applies latency calibration."""

    def __init__(self, parent=None, existing_profile: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Latency Calibration")
        self.setModal(True)

        self.worker: LatencyCalibrationWorker | None = None
        self._capture_session: CaptureSession | None = None
        self._close_requested = False
        self._close_result = False
        self._audio_lease: AudioLease | None = None
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
        self._measurement_route_key: str | None = None
        self._measurement_format_context: tuple[int, int, int, int] | None = None
        self._measurement_context_key: str | None = None
        owner = find_processor_owner(self.parent())
        if owner is not None:
            context = capture_route_context(owner)
            self._measurement_route_key = context.route_key
            self._measurement_context_key = context.context_key
        if existing_profile is not None:
            self._measurement_format_context = _known_capture_format_context(
                existing_profile.get("capture_format_context")
            )

        self._setup_ui(existing_profile)
        # The scene's own minimum grows with the text size; the dialog opens at least that wide.
        minimum_width = max(460, int(self.quick_view.rootObject().property("minimumWidth")))
        configure_resizable_dialog(
            self,
            preferred_width=max(620, minimum_width),
            preferred_height=620,
            minimum_width=minimum_width,
            minimum_height=360,
        )

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_worker)

    def _setup_ui(self, existing_profile: dict | None):
        self.view_model = LatencyCalibrationView(
            run=self._on_run_clicked,
            accept=self._on_accept_clicked,
            reset=self._on_reset_clicked,
            close=self._on_close_clicked,
            parent=self,
        )
        self.view_model.set_accept_enabled(existing_profile is not None)
        if existing_profile:
            self._apply_profile_to_labels(existing_profile)

        register_qml_types()
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view,
            QML_DIR / "LatencyCalibration.qml",
            quick_theme(),
            {"vm": self.view_model},
        )
        if reason is not None:
            # Nothing is connected yet; leave the parent without a half-built child.
            self.setParent(None)
            raise LatencyCalibrationLoadError(f"The latency calibration view did not load: {reason}")
        self.quick_view.setAccessibleName("Latency calibration content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)

    def _on_run_clicked(self):
        if self._close_requested or (self.worker is not None and self.worker.isRunning()):
            self.view_model.set_status("Wait for the previous latency analysis to stop.")
            return
        owner = find_processor_owner(self.parent())
        if owner is None:
            QMessageBox.critical(self, "Error", "Could not find audio processor.")
            return

        if owner.engine_session.view.user_muted:
            QMessageBox.information(
                self, "Output muted", "Unmute output before measuring latency."
            )
            return

        self._latest_profile = None
        self._measurement_route_key = None
        self._measurement_format_context = None
        self._measurement_context_key = None
        self._apply_profile_to_labels(None)
        self.view_model.set_accept_enabled(False)

        try:
            lease = owner.engine_session.borrow("latency_calibration")
        except RouteMismatch:
            self._on_worker_failed(
                "Latency calibration requires the selected route to match the active stream."
            )
            return
        except EngineBusy as e:
            QMessageBox.critical(self, "Audio error", str(e))
            return
        except AudioUnavailable as e:
            QMessageBox.critical(
                self, "Audio error", f"Failed to start processing: {e}"
            )
            return
        self._audio_lease = lease

        context = capture_route_context(owner)
        self._measurement_route_key = context.route_key
        self._measurement_format_context = context.format_context
        self._measurement_context_key = context.context_key
        if (
            self._measurement_route_key is None
            or self._measurement_format_context is None
        ):
            self._on_worker_failed(
                "Latency calibration requires known input and output sample rates and channels."
            )
            return

        self.view_model.set_run_enabled(False)
        self.view_model.set_accept_enabled(False)
        self.view_model.set_progress(0)
        self.view_model.set_status("Preparing probe...")

        try:
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
            self._capture_session = CaptureSession(owner.processor, lease)
            self._capture_session.start(self._recording_duration_s)
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
        owner = find_processor_owner(self.parent())
        if owner is None:
            self._on_worker_failed("Could not find audio processor.")
            return

        try:
            elapsed = time.monotonic() - self._capture_started_at
            engine_latency = float(owner.processor.get_engine_latency_ms())
            if np.isfinite(engine_latency) and engine_latency >= 0.0:
                self._engine_latency_samples.append(engine_latency)
            if (not self._probe_started) and elapsed >= self._playback_delay_s:
                self.view_model.set_status("Playing probe signal...")
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
            self.view_model.set_progress(progress)

            if (
                elapsed < self._recording_duration_s
                and not owner.processor.is_recording_complete()
            ):
                return

            self._capture_timer.stop()

            if not self._played_probe:
                self._on_worker_failed("Probe signal was not played.")
                return

            self.view_model.set_status("Analyzing captured signal...")
            session, self._capture_session = self._capture_session, None
            raw = (
                session.stop_recording()
                if session is not None
                else owner.processor.stop_raw_recording()
            )
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
            self.worker.result_ready.connect(self._on_worker_finished)
            self.worker.failed.connect(self._on_worker_failed)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.finished.connect(
                lambda worker=self.worker: self._on_thread_finished(worker)
            )
            self.worker.start()
        except Exception as e:
            self._on_worker_failed(
                f"Latency calibration failed: {type(e).__name__}: {e}"
            )

    def _on_level_update(self, rms_db: float):
        self.view_model.level_meter.set_rms_level(rms_db)

    def _on_worker_finished(self, payload: dict):
        if self._close_requested:
            return
        analysis = payload.get("analysis")
        profile = payload.get("profile")
        if not isinstance(profile, dict) or self._measurement_format_context is None:
            self._on_worker_failed("Latency calibration produced no usable format context.")
            return

        profile = dict(profile)
        profile["capture_format_context"] = list(self._measurement_format_context)

        self._latest_profile = profile
        self._apply_profile_to_labels(profile)

        self.view_model.set_progress(100)
        ambiguity = (
            float(profile.get("ambiguity_score", 0.0) or 0.0) if profile else 0.0
        )
        if ambiguity >= 0.75:
            self.view_model.set_status(
                "Calibration found the direct path, but echoes are close. Review before accepting."
            )
        else:
            self.view_model.set_status(
                "Calibration succeeded. Review the values, then select Accept."
            )
        self.view_model.set_run_enabled(True)
        self.view_model.set_accept_enabled(True)

        if analysis is not None and DEBUG:
            logger.debug(
                "Latency calibration success route=%.2fms conf=%.2f",
                analysis.measured_round_trip_ms,
                analysis.confidence,
            )

        self._teardown_worker()

    def _on_worker_failed(self, message: str):
        if self._close_requested:
            return
        self.view_model.set_status(message)
        self.view_model.set_run_enabled(True)
        self.view_model.set_accept_enabled(self._latest_profile is not None)

        self._teardown_worker()
        self._stop_owned_processor()

    def _apply_profile_to_labels(self, profile: dict | None):
        if not profile:
            self.view_model.set_readings(NO_READINGS)
            return

        directional = profile.get("directional_latency_ms")
        route_latency = float(profile.get("route_latency_ms", 0.0) or 0.0)
        if route_latency <= 0.0:
            route_latency = float(
                profile.get(
                    "measured_round_trip_ms",
                    profile.get("applied_compensation_ms", 0.0),
                )
                or 0.0
            )
        self.view_model.set_readings(
            {
                "total": f"{profile.get('total_latency_ms', 0.0):.2f} ms",
                "confidence": f"{profile.get('confidence', 0.0):.2f}",
                "roundTrip": f"{profile.get('measured_round_trip_ms', 0.0):.2f} ms",
                "oneWay": (
                    f"{float(directional):.2f} ms"
                    if directional is not None
                    else "Not inferred"
                ),
                "comp": f"{route_latency:.2f} ms",
                "agreement": f"{profile.get('agreement_ms', 0.0):.2f} ms across probes",
                "ambiguity": f"{profile.get('ambiguity_score', 0.0):.2f}",
                "engine": f"{profile.get('engine_latency_ms', 0.0):.2f} ms",
            }
        )

    def _on_accept_clicked(self):
        if not self._latest_profile:
            QMessageBox.information(self, "No result", "Run calibration first.")
            return

        owner = find_processor_owner(self.parent())
        context = capture_route_context(owner)
        profile_context = _known_capture_format_context(
            self._latest_profile.get("capture_format_context")
        )
        if (
            context.route_key != self._measurement_route_key
            or context.format_context != self._measurement_format_context
            or context.context_key != self._measurement_context_key
            or profile_context != self._measurement_format_context
        ):
            self.view_model.set_status(
                "Calibration is stale because the route or format changed. Run it again."
            )
            return

        if owner is None or not owner._on_latency_calibration_saved(self._latest_profile):
            self.view_model.set_status(
                "Calibration could not be saved. Check settings-folder access and retry."
            )
            return
        self.accept()

    def _on_reset_clicked(self):
        owner = find_processor_owner(self.parent())
        if owner is None or not owner._on_latency_calibration_reset():
            self.view_model.set_status("Calibration reset could not be saved. Retry when settings can be saved.")
            return
        self._latest_profile = None
        self._apply_profile_to_labels(None)
        self.view_model.set_accept_enabled(False)
        self.view_model.set_status("Calibration reset. Using estimated latency.")

    def _on_close_clicked(self):
        self.reject()

    def _stop_owned_processor(self):
        if self._audio_lease is not None:
            self._audio_lease.release()

    def _on_thread_finished(self, worker: LatencyCalibrationWorker) -> None:
        if self.worker is worker:
            self.worker = None
        if self._close_requested:
            self._finish_close()

    def _teardown_worker(self):
        self._capture_timer.stop()
        owner = find_processor_owner(self.parent())

        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        session, self._capture_session = self._capture_session, None
        if session is not None:
            session.cleanup()
        if owner is not None:
            try:
                owner.processor.cancel_output_probe()
            except Exception:
                pass
        self._probe_started = False
        self._played_probe = False

    def _finish_close(self) -> None:
        if not self._close_requested or self.worker is not None:
            return
        self._close_requested = False
        QDialog.done(
            self,
            int(QDialog.DialogCode.Accepted if self._close_result else QDialog.DialogCode.Rejected),
        )

    # A real method, so Qt drops the aboutToQuit connection when the dialog is destroyed.
    def _wait_for_worker(self) -> None:
        """Keep application shutdown from destroying a running QThread."""
        self._request_close(False)
        if self.worker is not None:
            self.worker.wait()
            self.worker = None

    def _request_close(self, accepted: bool) -> None:
        if self._close_requested:
            return
        self._close_requested = True
        self._close_result = accepted
        self._teardown_worker()
        self._stop_owned_processor()
        if self.worker is not None:
            self.view_model.set_run_enabled(False)
            self.view_model.set_accept_enabled(False)
            self.view_model.set_status("Canceling latency analysis...")
            return
        self._finish_close()

    def closeEvent(self, event):
        if session_is_ending():
            # A veto here would stop Windows from signing out: stop the work, join its thread, close.
            self._wait_for_worker()
            self._finish_close()
            event.accept()
            return
        self._request_close(False)
        event.ignore()

    def reject(self):
        self._request_close(False)

    def accept(self):
        self._request_close(True)
