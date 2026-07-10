"""Wizard dialog for Auto Voice Setup."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ..analysis.voice_setup import analyze_voice_setup
from ..config import EQ_FREQUENCIES, TARGET_CURVES
from .calibration_dialog import (
    RAINBOW_PASSAGE,
    TOO_LOUD_DB,
    TOO_QUIET_DB,
    _device_label,
    _device_name,
    _diagnostic_state,
    _find_eq_panel_owner,
    _find_processor_owner,
    _format_db,
    _format_percent,
    _processor_sample_rate,
    _selected_device_pair,
)
from .layout_constants import SUBDUED_TEXT_STYLE, status_chip_style
from .level_meter import LevelMeter

logger = logging.getLogger(__name__)

NOISE_RECORDING_DURATION = 2.0
VOICE_RECORDING_DURATION = 10.0


class VoiceSetupWorker(QThread):
    """Background worker for multi-stage voice setup analysis."""

    step_progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        noise_audio: np.ndarray,
        voice_audio: np.ndarray,
        sample_rate: int,
        target_preset: str,
        *,
        vad_available: bool,
    ) -> None:
        super().__init__()
        self.noise_audio = noise_audio
        self.voice_audio = voice_audio
        self.sample_rate = sample_rate
        self.target_preset = target_preset
        self.vad_available = vad_available

    def run(self) -> None:
        try:
            self.step_progress.emit("Analyzing room noise and speech...", 20)
            result = analyze_voice_setup(
                self.noise_audio,
                self.voice_audio,
                self.sample_rate,
                self.target_preset,
                vad_available=self.vad_available,
            )
            self.step_progress.emit("Finalizing recommendations...", 95)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceSetupDialog(QDialog):
    """Record room tone and speech, then recommend a voice chain."""

    setup_applied = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Voice Setup")
        self.setModal(True)
        self.setMinimumWidth(640)

        self.setup_state = "idle"
        self.noise_audio: np.ndarray | None = None
        self.voice_audio: np.ndarray | None = None
        self.setup_result: dict[str, Any] | None = None
        self.analysis_worker: VoiceSetupWorker | None = None
        self._started_processor = False
        self._recording_duration = NOISE_RECORDING_DURATION

        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self._poll_recording_progress)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        curve_group = QGroupBox("Step 1: Select Target Curve")
        curve_layout = QVBoxLayout(curve_group)

        curve_input = QHBoxLayout()
        curve_input.addWidget(QLabel("Target Curve:"))
        self.curve_combo = QComboBox()
        for key, curve in TARGET_CURVES.items():
            self.curve_combo.addItem(curve.name, key)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        curve_input.addWidget(self.curve_combo)
        curve_layout.addLayout(curve_input)

        self.curve_description = QLabel()
        self.curve_description.setWordWrap(True)
        self.curve_description.setStyleSheet("color: gray; font-size: 11px; padding: 5px;")
        curve_layout.addWidget(self.curve_description)
        layout.addWidget(curve_group)

        noise_group = QGroupBox("Step 2: Capture Room Noise")
        noise_layout = QVBoxLayout(noise_group)
        noise_hint = QLabel(
            "Stay quiet for 2 seconds so the wizard can measure room noise "
            "and set the gate safely."
        )
        noise_hint.setWordWrap(True)
        noise_layout.addWidget(noise_hint)
        layout.addWidget(noise_group)

        voice_group = QGroupBox("Step 3: Read Passage Aloud")
        voice_layout = QVBoxLayout(voice_group)
        passage_text = QTextEdit()
        passage_text.setPlainText(RAINBOW_PASSAGE)
        passage_text.setReadOnly(True)
        passage_text.setMaximumHeight(150)
        voice_layout.addWidget(passage_text)
        layout.addWidget(voice_group)

        self.recording_group = QGroupBox("Step 4: Record And Analyze")
        recording_layout = QVBoxLayout(self.recording_group)
        self.recording_group.setVisible(False)

        self.phase_label = QLabel("Ready to start setup")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")
        recording_layout.addWidget(self.phase_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            """
        )
        recording_layout.addWidget(self.progress_bar)

        self.time_label = QLabel("Time remaining: 2s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")
        recording_layout.addWidget(self.time_label)

        meter_layout = QHBoxLayout()
        self.level_meter = LevelMeter(label="Level", show_scale=True)
        self.level_meter.setMinimumHeight(150)
        meter_layout.addWidget(self.level_meter)
        recording_layout.addLayout(meter_layout)

        self.warning_label = QLabel("Ready to record")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet("color: gray; font-size: 11pt;")
        recording_layout.addWidget(self.warning_label)

        self.summary_group = QGroupBox("Recommended Settings")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_group.setVisible(False)
        self.overall_label = QLabel("Overall: --")
        self.eq_label = QLabel("EQ: --")
        self.gate_label = QLabel("Gate/VAD: --")
        self.deesser_label = QLabel("De-esser: --")
        self.compressor_label = QLabel("Compressor: --")
        for label in (
            self.overall_label,
            self.eq_label,
            self.gate_label,
            self.deesser_label,
            self.compressor_label,
        ):
            label.setStyleSheet(status_chip_style("idle"))
            summary_layout.addWidget(label)
        hint = QLabel(
            "This wizard tunes EQ, gate/VAD, de-esser, and compressor. "
            "Limiter settings are left unchanged."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(SUBDUED_TEXT_STYLE)
        summary_layout.addWidget(hint)
        recording_layout.addWidget(self.summary_group)

        controls = QHBoxLayout()
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.setVisible(False)
        self.retake_btn.clicked.connect(self._on_retake_clicked)
        controls.addWidget(self.retake_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        controls.addWidget(self.cancel_btn)
        recording_layout.addLayout(controls)
        layout.addWidget(self.recording_group)

        self.start_button = QPushButton("Start Voice Setup")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }"
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)

        self._on_curve_changed(0)

    def _on_curve_changed(self, index: int) -> None:
        if index < 0:
            return
        curve_key = self.curve_combo.currentData()
        curve = TARGET_CURVES[curve_key]
        self.curve_description.setText(curve.description)

    def _on_start_clicked(self) -> None:
        if self.setup_state == "idle":
            self.recording_group.setVisible(True)
            self._start_recording_phase("noise_recording")
        elif self.setup_state == "noise_ready":
            self._start_recording_phase("voice_recording")
        elif self.setup_state == "completed" and self.setup_result is not None:
            self._apply_setup()
        elif self.setup_state in {"noise_recording", "voice_recording"}:
            QMessageBox.information(self, "Recording", "Please let the recording finish.")

    def _start_recording_phase(self, phase: str) -> None:
        self._stop_analysis_worker()
        if not self._ensure_processor_ready():
            return

        self.setup_state = phase
        self.curve_combo.setEnabled(False)
        self.start_button.setEnabled(False)
        self.retake_btn.setVisible(False)
        self.summary_group.setVisible(False)

        if phase == "noise_recording":
            self._recording_duration = NOISE_RECORDING_DURATION
            self.start_button.setText("Recording Noise...")
            self.phase_label.setText("Capturing room noise")
            self.warning_label.setText("Stay quiet and keep the room as it normally is.")
            self.warning_label.setStyleSheet("color: blue; font-weight: bold; font-size: 11pt;")
        else:
            self._recording_duration = VOICE_RECORDING_DURATION
            self.start_button.setText("Recording Voice...")
            self.phase_label.setText("Capturing speech")
            self.warning_label.setText("Speak naturally into the microphone.")
            self.warning_label.setStyleSheet("color: blue; font-weight: bold; font-size: 11pt;")

        self.progress_bar.setValue(0)
        self.time_label.setText(f"Time remaining: {self._recording_duration:.0f}s")
        QTimer.singleShot(100, self._begin_recording_capture)

    def _ensure_processor_ready(self) -> bool:
        parent = _find_processor_owner(self.parent())
        if not parent:
            QMessageBox.critical(self, "Error", "Could not find audio processor")
            return False

        processor_was_running = parent.processor.is_running()
        selected_input, selected_output = _selected_device_pair(parent)

        if processor_was_running:
            get_active_input = getattr(parent.processor, "get_active_input_device", None)
            get_active_output = getattr(parent.processor, "get_active_output_device", None)
            active_pair = (
                _device_name(get_active_input() if callable(get_active_input) else None),
                _device_name(get_active_output() if callable(get_active_output) else None),
            )
            if active_pair != (selected_input, selected_output):
                reply = QMessageBox.question(
                    self,
                    "Switch Devices for Voice Setup?",
                    "Voice setup should record from your selected devices.\n\n"
                    f"Selected input: {_device_label(selected_input, '(Default Input)')}\n"
                    f"Selected output: {_device_label(selected_output, '(Default Output)')}\n\n"
                    f"Active input: {_device_label(active_pair[0], '(Default Input)')}\n"
                    f"Active output: {_device_label(active_pair[1], '(Default Output)')}\n\n"
                    "Switch now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.warning_label.setText("Setup canceled: using current stream devices.")
                    self.warning_label.setStyleSheet("color: orange; font-size: 11pt;")
                    self.start_button.setEnabled(True)
                    self.curve_combo.setEnabled(True)
                    self.setup_state = "idle" if self.noise_audio is None else "noise_ready"
                    self.start_button.setText(
                        "Start Voice Setup" if self.noise_audio is None else "Record Voice"
                    )
                    return False

                try:
                    parent.processor.stop()
                    parent.processor.start(selected_input, selected_output)
                    parent.processor.set_output_mute(False)
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        "Audio Error",
                        f"Failed to switch audio devices for setup:\n{exc}",
                    )
                    return False
            return True

        try:
            parent.processor.start(selected_input, selected_output)
            self._started_processor = True
            return True
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Audio Error",
                f"Failed to start audio processing:\n{exc}\n\n"
                "Check that audio devices are connected and not already in use.",
            )
            return False

    def _begin_recording_capture(self) -> None:
        if self.setup_state not in {"noise_recording", "voice_recording"}:
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            parent.processor.set_recovery_suppressed(True)
            parent.processor.start_raw_recording(self._recording_duration)
        except Exception as exc:
            self._on_recording_failed(f"Recording error: {exc}")
            return

        self.recording_timer.start()

    def _poll_recording_progress(self) -> None:
        if self.setup_state not in {"noise_recording", "voice_recording"}:
            self.recording_timer.stop()
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            progress = float(parent.processor.recording_progress())
            progress_pct = int(progress * 100)
            self.progress_bar.setValue(progress_pct)
            self._update_time_remaining(max(0.0, self._recording_duration * (1.0 - progress)))
            self._update_level(float(parent.processor.recording_level_db()))

            if progress_pct >= 100 or parent.processor.is_recording_complete():
                self.recording_timer.stop()
                audio = parent.processor.stop_raw_recording()
                if audio is None:
                    self._on_recording_failed("Recording failed - no audio data")
                    return
                self._on_recording_complete(np.asarray(audio, dtype=np.float32))
        except Exception as exc:
            self.recording_timer.stop()
            self._on_recording_failed(f"Recording error: {exc}")

    def _update_time_remaining(self, seconds: float) -> None:
        if seconds > 0:
            self.time_label.setText(f"Time remaining: {seconds:.0f}s")
        else:
            self.time_label.setText("Complete!")
            self.time_label.setStyleSheet("font-size: 12pt; color: #4CAF50; font-weight: bold;")

    def _update_level(self, rms_db: float) -> None:
        self.level_meter.set_levels(rms_db, rms_db + 6.0)

        if self.setup_state == "noise_recording":
            if rms_db > -35.0:
                self.warning_label.setText("Room noise is fairly loud. Results may be conservative.")
                self.warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 11pt;")
            else:
                self.warning_label.setText("Noise floor capture looks usable.")
                self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")
            return

        if rms_db < TOO_QUIET_DB:
            self.warning_label.setText("Too quiet. Move closer to the mic.")
            self.warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 11pt;")
        elif rms_db > TOO_LOUD_DB:
            self.warning_label.setText("Too loud. Back off slightly to avoid clipping.")
            self.warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 11pt;")
        else:
            self.warning_label.setText("Voice level looks good.")
            self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")

    def _on_recording_complete(self, audio_data: np.ndarray) -> None:
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(True)
        self.time_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")

        if self.setup_state == "noise_recording":
            self.noise_audio = audio_data
            self.setup_state = "noise_ready"
            self.start_button.setText("Record Voice")
            self.phase_label.setText("Room noise captured")
            self.warning_label.setText("Now read the passage aloud for the full 10 seconds.")
            self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")
            self.progress_bar.setValue(0)
            self.time_label.setText(f"Time remaining: {VOICE_RECORDING_DURATION:.0f}s")
            return

        self.voice_audio = audio_data
        self.setup_state = "analyzing"
        self.start_button.setEnabled(False)
        self.start_button.setText("Analyzing...")
        self.phase_label.setText("Analyzing recordings")
        self.warning_label.setText("Building recommendations for your voice chain.")
        self.warning_label.setStyleSheet("color: blue; font-weight: bold; font-size: 11pt;")
        self._start_analysis()

    def _start_analysis(self) -> None:
        if self.noise_audio is None or self.voice_audio is None:
            self._on_analysis_failed("Missing room-noise or speech capture.")
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_analysis_failed("Could not find audio processor")
            return

        vad_available = False
        try:
            vad_available = bool(parent.processor.is_vad_available())
        except Exception:
            vad_available = False

        self.analysis_worker = VoiceSetupWorker(
            self.noise_audio,
            self.voice_audio,
            _processor_sample_rate(parent),
            self.get_selected_curve(),
            vad_available=vad_available,
        )
        self.analysis_worker.step_progress.connect(self._on_analysis_step)
        self.analysis_worker.finished.connect(self._on_analysis_complete)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()

    def _on_analysis_step(self, step_name: str, percentage: int) -> None:
        self.warning_label.setText(step_name)
        self.progress_bar.setValue(percentage)

    def _on_analysis_complete(self, setup_result: dict[str, Any]) -> None:
        self.analysis_worker = None
        self.setup_result = setup_result
        self.setup_state = "completed"
        self.start_button.setText("Apply Voice Setup")
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        self.phase_label.setText("Recommendations ready")
        diagnostics = setup_result["diagnostics"]
        if diagnostics.get("apply_recommended", False):
            self.warning_label.setText("Review the validated settings and apply them.")
            color = "#4CAF50"
        else:
            reasons = diagnostics.get("uncertainty_reasons") or ["capture confidence is weak"]
            self.warning_label.setText(
                "Advisory recommendations only: " + "; ".join(str(reason) for reason in reasons)
            )
            color = "#FFB74D"
        self.warning_label.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 11pt;"
        )
        self.progress_bar.setValue(100)
        self._show_summary(setup_result)

    def _show_summary(self, setup_result: dict[str, Any]) -> None:
        diagnostics = setup_result["diagnostics"]
        overall_conf = float(diagnostics["setup_confidence"])
        state = _diagnostic_state(overall_conf)
        self.overall_label.setText(
            "Overall: "
            f"{_format_percent(overall_conf)} | "
            f"capture {_format_percent(diagnostics['capture_confidence'])} | "
            f"uncertainty {_format_percent(diagnostics['recommendation_uncertainty'])}"
        )
        self.overall_label.setStyleSheet(status_chip_style(state))

        eq_settings = setup_result.get("eq_settings")
        if eq_settings is not None:
            self.eq_label.setText(
                "EQ: "
                f"{_format_percent(eq_settings.get('analysis_confidence', 0.0))} | "
                f"max correction {max(abs(g) for g in eq_settings['band_gains']):.1f} dB"
            )
            self.eq_label.setStyleSheet(status_chip_style("ok"))
        else:
            eq_error = setup_result.get("eq_error") or "Skipped"
            self.eq_label.setText(f"EQ: skipped | {eq_error}")
            self.eq_label.setStyleSheet(status_chip_style("warn"))

        gate = setup_result["gate_settings"]
        self.gate_label.setText(
            "Gate/VAD: "
            f"{diagnostics['gate_mode_label']} | "
            f"threshold {_format_db(gate['threshold_db'])} | "
            f"VAD {gate['vad_threshold']:.2f}"
        )
        self.gate_label.setStyleSheet(status_chip_style("info"))

        deesser = setup_result["deesser_settings"]
        deesser_state = "ok" if deesser["enabled"] else "info"
        deesser_text = "enabled" if deesser["enabled"] else "left off"
        self.deesser_label.setText(
            "De-esser: "
            f"{deesser_text} | auto {deesser['auto_amount'] * 100.0:.0f}% | "
            f"{deesser['low_cut_hz']:.0f}-{deesser['high_cut_hz']:.0f} Hz"
        )
        self.deesser_label.setStyleSheet(status_chip_style(deesser_state))

        compressor = setup_result["compressor_settings"]
        makeup_text = "auto makeup" if compressor["auto_makeup_enabled"] else f"{compressor['makeup_gain_db']:.1f} dB makeup"
        self.compressor_label.setText(
            "Compressor: "
            f"{compressor['ratio']:.1f}:1 @ {_format_db(compressor['threshold_db'])} | "
            f"{makeup_text} | target {compressor['target_lufs']:.0f} LUFS"
        )
        self.compressor_label.setStyleSheet(status_chip_style("info"))
        self.summary_group.setVisible(True)

    def _apply_setup(self) -> None:
        parent = _find_eq_panel_owner(self.parent())
        if not parent or self.setup_result is None:
            QMessageBox.critical(self, "Error", "Could not apply voice setup.")
            return

        diagnostics = self.setup_result.get("diagnostics") or {}
        if not diagnostics.get("apply_recommended", False):
            reasons = diagnostics.get("uncertainty_reasons") or ["capture confidence is weak"]
            reply = QMessageBox.question(
                self,
                "Apply Advisory Settings?",
                "These settings did not reach validated confidence:\n\n"
                + "\n".join(f"- {reason}" for reason in reasons)
                + "\n\nApply them anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        parent.gate_panel.set_settings(self.setup_result["gate_settings"])
        parent.deesser_panel.set_settings(self.setup_result["deesser_settings"])
        parent.compressor_panel.set_compressor_settings(self.setup_result["compressor_settings"])

        eq_settings = self.setup_result.get("eq_settings")
        if eq_settings is not None:
            freqs_hz = eq_settings.get("band_freqs", EQ_FREQUENCIES)
            if len(freqs_hz) != len(EQ_FREQUENCIES):
                freqs_hz = EQ_FREQUENCIES
            bands = []
            for i, freq in enumerate(freqs_hz):
                bands.append(
                    (
                        freq,
                        eq_settings["band_gains"][i],
                        eq_settings["band_qs"][i],
                    )
                )
            parent.eq_panel.apply_auto_eq_results(bands, diagnostics=eq_settings)

        if hasattr(parent, "status_bar"):
            parent.status_bar.showMessage("Auto voice setup applied", 5000)

        self.setup_applied.emit(self.get_selected_curve())
        self.accept()

    def _on_analysis_failed(self, error: str) -> None:
        self.analysis_worker = None
        self.voice_audio = None
        self.setup_result = None
        self.setup_state = "noise_ready" if self.noise_audio is not None else "idle"
        self.start_button.setEnabled(True)
        self.start_button.setText(
            "Record Voice" if self.noise_audio is not None else "Start Voice Setup"
        )
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 11pt;")
        self.phase_label.setText("Analysis failed")
        self.summary_group.setVisible(False)

    def _on_recording_failed(self, error: str) -> None:
        self.recording_timer.stop()
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 11pt;")
        self.start_button.setText("Start Voice Setup" if self.noise_audio is None else "Record Voice")
        self.setup_state = "idle" if self.noise_audio is None else "noise_ready"
        self._cleanup_recording_tap()

    def _on_retake_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Discard Setup?",
            "Discard the current setup captures and start over?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_setup_ui()

    def _on_cancel_clicked(self) -> None:
        if self.setup_state in {"noise_recording", "voice_recording"}:
            reply = QMessageBox.question(
                self,
                "Cancel Recording?",
                "Discard the current capture and return to the main window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.reject()

    def _reset_setup_ui(self) -> None:
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()

        self.setup_state = "idle"
        self.noise_audio = None
        self.voice_audio = None
        self.setup_result = None
        self.progress_bar.setValue(0)
        self.level_meter.set_levels(-120.0, -120.0)
        self.phase_label.setText("Ready to start setup")
        self.time_label.setText(f"Time remaining: {NOISE_RECORDING_DURATION:.0f}s")
        self.time_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")
        self.warning_label.setText("Ready to record")
        self.warning_label.setStyleSheet("color: gray; font-size: 11pt;")
        self.start_button.setText("Start Voice Setup")
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        self.retake_btn.setVisible(False)
        self.summary_group.setVisible(False)

    def _stop_owned_processor(self) -> None:
        if not self._started_processor:
            return

        parent = _find_processor_owner(self.parent())
        if parent:
            try:
                parent.processor.stop()
            except Exception as exc:
                logger.warning("Failed to stop owned processor: %s", exc)
        self._started_processor = False

    def _cleanup_recording_tap(self) -> None:
        parent = _find_processor_owner(self.parent())
        if not parent:
            return

        try:
            parent.processor.stop_raw_recording()
        except Exception as exc:
            logger.warning("Failed to stop raw recording during cleanup: %s", exc)

        try:
            parent.processor.set_output_mute(False)
        except Exception as exc:
            logger.warning("Failed to unmute output during cleanup: %s", exc)

        try:
            parent.processor.set_recovery_suppressed(False)
        except Exception as exc:
            logger.warning("Failed to re-enable recovery after cleanup: %s", exc)

    def _stop_analysis_worker(self) -> None:
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.wait(1500)
        self.analysis_worker = None

    def get_selected_curve(self) -> str:
        return str(self.curve_combo.currentData() or "broadcast")

    def closeEvent(self, event) -> None:
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        super().closeEvent(event)

    def accept(self) -> None:
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        super().accept()

    def reject(self) -> None:
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        super().reject()
