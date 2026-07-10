"""
Calibration dialog for Auto-EQ feature

DEBUG: Added terminal logging for calibration workflow
"""

import logging
import time
from typing import Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox,
    QPushButton, QTextEdit, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import numpy as np

from ..config import TARGET_CURVES, coerce_device_identity
from .analysis_worker import AnalysisWorker
from .layout_constants import SUBDUED_TEXT_STYLE, status_chip_style
from .level_meter import LevelMeter

# Enable debug logging (set to False for production)
DEBUG = False

logger = logging.getLogger(__name__)

# Rainbow Passage - standard calibration text from audiometry
RAINBOW_PASSAGE = """The Rainbow Passage
The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries, men have been fascinated by this spectacle of the sky. They have tried to find explanations for it. Scientists, however, have found that it is caused by the reflection and refraction of sunlight on drops of water in the air."""

# Recording validation thresholds (dB)
TOO_QUIET_DB = -40.0   # Warn if quieter than this
TOO_LOUD_DB = -3.0      # Warn if louder than this (clipping risk)
RECORDING_DURATION = 10.0  # Seconds


def _find_processor_owner(widget: object) -> Any | None:
    parent: Any = widget
    while parent and not hasattr(parent, "processor"):
        parent = parent.parent()
    return parent


def _find_eq_panel_owner(widget: object) -> Any | None:
    parent: Any = widget
    while parent and not hasattr(parent, "eq_panel"):
        parent = parent.parent()
    return parent


def _processor_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor")

    sample_rate = int(owner.processor.sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Processor sample rate is unavailable")

    return sample_rate


def _chain_settings(owner: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if owner is None:
        return settings
    if hasattr(owner, "deesser_panel"):
        try:
            settings["deesser"] = owner.deesser_panel.get_settings()
        except Exception:
            logger.debug("Failed to collect de-esser settings for Auto-EQ simulation", exc_info=True)
    if hasattr(owner, "compressor_panel"):
        try:
            settings["compressor"] = owner.compressor_panel.get_compressor_settings()
            settings["limiter"] = owner.compressor_panel.get_limiter_settings()
        except Exception:
            logger.debug("Failed to collect dynamics settings for Auto-EQ simulation", exc_info=True)
    return settings


def _selected_device_pair(owner: Any) -> tuple[str | None, str | None]:
    if owner is None:
        return None, None

    input_device = None
    output_device = None

    if hasattr(owner, "input_combo"):
        input_device = owner.input_combo.currentData() or None
    if hasattr(owner, "output_combo"):
        output_device = owner.output_combo.currentData() or None

    return _device_name(input_device), _device_name(output_device)


def _device_name(device: object) -> str | None:
    identity = coerce_device_identity(device)
    if identity is not None:
        return identity.name
    return device if isinstance(device, str) and device else None


def _device_label(device: str | None, default_label: str) -> str:
    return device if device else default_label


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "--"


def _format_db(value: Any) -> str:
    try:
        return f"{float(value):.1f} dB"
    except (TypeError, ValueError):
        return "--"


def _diagnostic_state(confidence: float) -> str:
    if confidence >= 0.72:
        return "ok"
    if confidence >= 0.45:
        return "warn"
    return "bad"


class CalibrationDialog(QDialog):
    """Auto-EQ calibration dialog with target curve selection."""

    # Signal emitted when auto-EQ is applied (emits target curve name)
    auto_eq_applied = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-EQ Calibration")
        self.setModal(True)  # Modal dialog - blocks main window
        self.setMinimumWidth(600)

        # Recording state
        self.recording_state = "idle"  # idle, recording, completed
        self.audio_data: np.ndarray | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self._started_processor = False  # Track if we started processor ourselves
        self._recording_started_at = 0.0
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self._poll_recording_progress)

        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Target curve selector group
        curve_group = QGroupBox("Step 1: Select Target Curve")
        curve_layout = QVBoxLayout(curve_group)

        # Curve dropdown
        curve_input_layout = QHBoxLayout()
        curve_input_layout.addWidget(QLabel("Target Curve:"))

        self.curve_combo = QComboBox()
        for key, curve in TARGET_CURVES.items():
            self.curve_combo.addItem(curve.name, key)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        curve_input_layout.addWidget(self.curve_combo)
        curve_layout.addLayout(curve_input_layout)

        target_mode_layout = QHBoxLayout()
        target_mode_layout.addWidget(QLabel("Target Mode:"))
        self.target_mode_combo = QComboBox()
        self.target_mode_combo.addItem("Adaptive voice-aware", "adaptive")
        self.target_mode_combo.addItem("Static catalog curve", "static")
        self.target_mode_combo.setToolTip(
            "Adaptive mode applies bounded voice-aware target offsets. Static mode uses the selected curve exactly."
        )
        target_mode_layout.addWidget(self.target_mode_combo)
        curve_layout.addLayout(target_mode_layout)

        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItem("Conservative", "conservative")
        self.smoothing_combo.addItem("Balanced", "balanced")
        self.smoothing_combo.addItem("Broad", "broad")
        self.smoothing_combo.setToolTip(
            "Conservative smoothing resists narrow measurement artifacts. Broad is safest but less detailed."
        )
        smoothing_layout.addWidget(self.smoothing_combo)
        curve_layout.addLayout(smoothing_layout)

        # Curve description (updates when selection changes)
        self.curve_description = QLabel()
        self.curve_description.setWordWrap(True)
        self.curve_description.setStyleSheet("color: gray; font-size: 11px; padding: 5px;")
        curve_layout.addWidget(self.curve_description)

        layout.addWidget(curve_group)

        # Instructions group with Rainbow Passage
        instructions_group = QGroupBox("Step 2: Read Passage Aloud")
        instructions_layout = QVBoxLayout(instructions_group)

        # Scrollable text area for Rainbow Passage
        passage_text = QTextEdit()
        passage_text.setPlainText(RAINBOW_PASSAGE)
        passage_text.setReadOnly(True)
        passage_text.setMaximumHeight(150)
        instructions_layout.addWidget(passage_text)

        layout.addWidget(instructions_group)

        # Recording UI group
        self.recording_group = QGroupBox("Step 3: Record Your Voice")
        recording_layout = QVBoxLayout(self.recording_group)
        self.recording_group.setVisible(False)  # Hidden until user clicks Start

        # Progress bar - solid continuous style
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # Hide "0%" text
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        recording_layout.addWidget(self.progress_bar)

        # Time remaining label below progress bar
        self.time_label = QLabel(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")
        recording_layout.addWidget(self.time_label)

        # Level meter (vertical) for real-time validation
        info_layout = QHBoxLayout()
        self.level_meter = LevelMeter(label="Level", show_scale=True)
        self.level_meter.setMinimumHeight(150)
        info_layout.addWidget(self.level_meter)
        recording_layout.addLayout(info_layout)

        # Validation warning label
        self.warning_label = QLabel("Ready to record")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet("color: gray; font-size: 11pt;")
        recording_layout.addWidget(self.warning_label)

        self.diagnostics_group = QGroupBox("Analysis Diagnostics")
        diagnostics_layout = QVBoxLayout(self.diagnostics_group)
        self.confidence_label = QLabel("Confidence: --")
        self.error_label = QLabel("Target error: --")
        self.gain_scale_label = QLabel("Gain scale: --")
        self.target_profile_label = QLabel("Target profile: --")
        for label in (
            self.confidence_label,
            self.error_label,
            self.gain_scale_label,
            self.target_profile_label,
        ):
            label.setStyleSheet(status_chip_style("idle"))
            diagnostics_layout.addWidget(label)
        hint_label = QLabel(
            "Diagnostics are computed from recording clarity, repeatability, "
            "and post-solve validation."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        diagnostics_layout.addWidget(hint_label)
        self.diagnostics_group.setVisible(False)
        recording_layout.addWidget(self.diagnostics_group)

        # Recording controls
        control_layout = QHBoxLayout()

        # Retake button (hidden initially)
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.setVisible(False)
        self.retake_btn.clicked.connect(self._on_retake_clicked)
        control_layout.addWidget(self.retake_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        control_layout.addWidget(self.cancel_btn)

        recording_layout.addLayout(control_layout)
        layout.addWidget(self.recording_group)

        # Start button (opens recording section)
        self.start_button = QPushButton("Start Calibration")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }"
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)

        # Initialize with first curve description
        self._on_curve_changed(0)

    def _on_curve_changed(self, index: int):
        """Update description when curve selection changes."""
        if index < 0:
            return
        curve_key = self.curve_combo.currentData()
        curve = TARGET_CURVES[curve_key]
        self.curve_description.setText(curve.description)

    def _on_start_clicked(self):
        """Handle Start Calibration button click based on recording state."""
        if self.recording_state == "idle":
            # Show recording UI and start recording
            self.recording_group.setVisible(True)
            self._start_recording()
        elif self.recording_state == "recording":
            # User tried to stop early - not allowed
            QMessageBox.information(
                self, "Recording",
                "Please record the full 10 seconds for accurate calibration."
            )
        elif self.recording_state == "completed":
            # Check if analysis was complete and user is applying results
            if hasattr(self, 'eq_settings') and self.eq_settings is not None:
                # Apply EQ settings to main window
                self._apply_eq_settings()
            else:
                # Start new recording
                self._reset_recording_ui()
                self._start_recording()

    def _apply_eq_settings(self):
        """Apply auto-EQ settings to main window and close dialog."""
        if DEBUG:
            logger.debug("Applying EQ settings")

        # Get parent's EQ panel (MainWindow has it)
        parent = _find_eq_panel_owner(self.parent())

        if not parent:
            QMessageBox.critical(self, "Error", "Could not find EQ panel")
            return

        # Build band tuples from eq_settings
        from ..config import EQ_FREQUENCIES as BAND_FREQUENCIES_HZ
        freqs_hz = self.eq_settings.get('band_freqs', BAND_FREQUENCIES_HZ)
        if len(freqs_hz) != len(BAND_FREQUENCIES_HZ):
            freqs_hz = BAND_FREQUENCIES_HZ
        bands = []
        for i, freq in enumerate(freqs_hz):
            gain = self.eq_settings['band_gains'][i]
            q = self.eq_settings['band_qs'][i] if 'band_qs' in self.eq_settings else 1.41
            bands.append((freq, gain, q))

        # Apply settings to EQ panel
        parent.eq_panel.apply_auto_eq_results(bands, diagnostics=self.eq_settings)

        # Emit signal for main window to handle preset save and undo button
        target_curve = self.get_selected_curve()
        self.auto_eq_applied.emit(target_curve)

        if DEBUG:
            logger.debug("EQ settings applied, signal emitted for curve=%s", target_curve)

        # Close dialog
        self.accept()

    def _start_recording(self):
        """Start non-blocking recording."""
        if DEBUG:
            logger.debug("Start recording clicked")

        self._stop_analysis_worker()

        # Get parent's processor (MainWindow has it)
        parent = _find_processor_owner(self.parent())

        if not parent:
            QMessageBox.critical(self, "Error", "Could not find audio processor")
            return

        processor_was_running = parent.processor.is_running()
        selected_input, selected_output = _selected_device_pair(parent)

        if DEBUG:
            logger.debug(
                "Processor state: running=%s, selected_input=%r, selected_output=%r",
                processor_was_running,
                selected_input,
                selected_output,
            )

        if processor_was_running:
            get_active_input = getattr(parent.processor, "get_active_input_device", None)
            get_active_output = getattr(parent.processor, "get_active_output_device", None)
            active_input = _device_name(get_active_input() if callable(get_active_input) else None)
            active_output = _device_name(get_active_output() if callable(get_active_output) else None)
            if DEBUG:
                logger.debug("Active stream devices: input=%r, output=%r", active_input, active_output)

            if active_input != selected_input or active_output != selected_output:
                reply = QMessageBox.question(
                    self,
                    "Switch Devices for Auto-EQ?",
                    "Auto-EQ should record from your selected devices.\n\n"
                    f"Selected input: {_device_label(selected_input, '(Default Input)')}\n"
                    f"Selected output: {_device_label(selected_output, '(Default Output)')}\n\n"
                    f"Active input: {_device_label(active_input, '(Default Input)')}\n"
                    f"Active output: {_device_label(active_output, '(Default Output)')}\n\n"
                    "Switch now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.warning_label.setText("Calibration canceled: using current stream devices")
                    self.warning_label.setStyleSheet("color: orange; font-size: 11pt;")
                    return

                try:
                    if DEBUG:
                        logger.debug("Restarting processor on selected devices")
                    parent.processor.stop()
                    parent.processor.start(selected_input, selected_output)
                    parent.processor.set_output_mute(False)
                    if DEBUG:
                        logger.debug("Processor restarted on selected devices")
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Audio Error",
                        f"Failed to switch audio devices for calibration:\n{e}",
                    )
                    return
            self._started_processor = False
            if DEBUG:
                logger.debug("Reusing running processor session")
        else:
            try:
                if DEBUG:
                    logger.debug("Starting audio processor from main thread")
                parent.processor.start(selected_input, selected_output)
                self._started_processor = True  # Track that we started it
                if DEBUG:
                    logger.debug("Audio processor started successfully")
            except Exception as e:
                QMessageBox.critical(
                    self, "Audio Error",
                    f"Failed to start audio processing:\n{str(e)}\n\n"
                    "Check that audio devices are connected and not in use by another application."
                )
                return

        self.recording_state = "recording"
        self.start_button.setText("Recording...")
        self.start_button.setEnabled(False)  # Prevent early stop

        # Lock Auto-EQ target controls during recording
        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)

        # Let DSP loop settle without blocking the UI thread.
        QTimer.singleShot(100, self._begin_recording_capture)

        self.warning_label.setText("Recording... Speak clearly into your microphone")
        self.warning_label.setStyleSheet("color: blue; font-weight: bold; font-size: 11pt;")

    def _begin_recording_capture(self):
        """Start Rust-side recording and poll progress from the main Qt thread."""
        if self.recording_state != "recording":
            return

        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            parent.processor.set_recovery_suppressed(True)
            parent.processor.start_raw_recording(RECORDING_DURATION)
        except Exception as e:
            self._on_recording_failed(f"Recording error: {e}")
            return

        self._recording_started_at = time.time()
        self.recording_timer.start()
        if DEBUG:
            logger.debug("Started main-thread recording capture")

    def _poll_recording_progress(self):
        """Poll recording state from the main Qt thread."""
        if self.recording_state != "recording":
            self.recording_timer.stop()
            return

        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            progress_float = float(parent.processor.recording_progress())
            progress_pct = int(progress_float * 100)
            self._on_progress_update(progress_pct)
            self._on_time_remaining(max(0.0, RECORDING_DURATION * (1.0 - progress_float)))
            self._on_level_update(float(parent.processor.recording_level_db()))

            if progress_pct >= 100 or parent.processor.is_recording_complete():
                self.recording_timer.stop()
                audio = parent.processor.stop_raw_recording()
                if audio is None:
                    self._on_recording_failed("Recording failed - no audio data")
                    return
                audio_array = np.asarray(audio, dtype=np.float32)
                self._on_recording_complete(audio_array)
        except Exception as e:
            self.recording_timer.stop()
            self._on_recording_failed(f"Recording error: {e}")

    def _on_progress_update(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def _on_time_remaining(self, seconds: float):
        """Update time remaining label."""
        if seconds > 0:
            self.time_label.setText(f"Time remaining: {seconds:.0f}s")
        else:
            self.time_label.setText("✓ Complete!")
            self.time_label.setStyleSheet("font-size: 12pt; color: #4CAF50; font-weight: bold;")

    def _on_level_update(self, rms_db: float):
        """Update level meter with validation warnings."""
        # Update level meter (estimate peak as RMS + 6dB)
        self.level_meter.set_levels(rms_db, rms_db + 6)

        # Show validation warning
        if rms_db < TOO_QUIET_DB:
            self.warning_label.setText("⚠️ Too quiet! Move closer to mic")
            self.warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 11pt;")
        elif rms_db > TOO_LOUD_DB:
            self.warning_label.setText("⚠️ Too loud! Risk of clipping")
            self.warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 11pt;")
        else:
            self.warning_label.setText("✓ Level is good")
            self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")

    def _on_recording_complete(self, audio_data: np.ndarray):
        """Handle recording completion."""
        if DEBUG:
            logger.debug("Recording complete: %d samples", len(audio_data))
            import numpy as np
            rms = np.mean(audio_data**2)**0.5
            peak_db = 20 * np.log10(max(np.abs(audio_data).max(), 1e-6))
            rms_db = 20 * np.log10(max(rms, 1e-6))
            logger.debug("Audio stats - Peak: %.1f dB, RMS: %.1f dB", peak_db, rms_db)

        self.audio_data = audio_data
        self.recording_state = "completed"

        # Update UI
        self.start_button.setText("Record Again")
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(True)

        # Re-enable curve selection
        self.curve_combo.setEnabled(True)
        self.target_mode_combo.setEnabled(True)
        self.smoothing_combo.setEnabled(True)

        # Show completion message
        self.warning_label.setText(f"Recording complete! {len(audio_data)} samples captured")
        self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")

        if DEBUG:
            logger.debug("Audio captured; starting analysis")

        if DEBUG:
            logger.debug("Starting analysis worker")
        self._start_analysis()

    def _on_recording_failed(self, error: str):
        """Handle recording failure."""
        self.recording_timer.stop()
        self.warning_label.setText(f"❌ Recording failed: {error}")
        self.warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 11pt;")
        self._reset_recording_ui()

    def _start_analysis(self):
        """Start analysis worker."""
        if self.audio_data is None:
            if DEBUG:
                logger.debug("No audio data to analyze")
            return

        self._stop_analysis_worker()

        # Get parent's processor for sample rate
        parent = _find_processor_owner(self.parent())

        if not parent:
            if DEBUG:
                logger.debug("Could not find processor")
            return

        sample_rate = _processor_sample_rate(parent)
        target_preset = self.get_selected_curve()
        target_mode = self.get_selected_target_mode()
        smoothing_strength = self.get_selected_smoothing_strength()
        chain_settings = _chain_settings(parent)

        if DEBUG:
            logger.debug(
                "Creating AnalysisWorker: %d samples, %sHz, target=%s, mode=%s, smoothing=%s",
                len(self.audio_data),
                sample_rate,
                target_preset,
                target_mode,
                smoothing_strength,
            )

        # Create and start analysis worker
        self.analysis_worker = AnalysisWorker(
            self.audio_data,
            sample_rate,
            target_preset,
            target_mode=target_mode,
            smoothing_strength=smoothing_strength,
            chain_settings=chain_settings,
        )
        self.analysis_worker.step_progress.connect(self._on_analysis_step)
        self.analysis_worker.finished.connect(self._on_analysis_complete)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()

        if DEBUG:
            logger.debug("AnalysisWorker started")

    def _on_analysis_step(self, step_name: str, percentage: int):
        """Handle analysis step progress."""
        if DEBUG:
            logger.debug("Analysis step %s%%: %s", percentage, step_name)
        self.warning_label.setText(f"Analyzing: {step_name}")
        self.progress_bar.setValue(percentage)

    def _on_analysis_complete(self, eq_settings: dict):
        """Handle analysis completion."""
        if DEBUG:
            logger.debug("Analysis complete")
            logger.debug("Band gains: %s", [round(g, 1) for g in eq_settings['band_gains']])
            max_gain = max(abs(g) for g in eq_settings['band_gains'])
            logger.debug("Max correction: %.1f dB", round(max_gain, 1))

        self.eq_settings = eq_settings
        self.warning_label.setText(f"✓ Analysis complete! Max correction: {round(max(abs(g) for g in eq_settings['band_gains']), 1)} dB")
        self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")
        self.progress_bar.setValue(100)
        self._show_analysis_diagnostics(eq_settings)
        self.analysis_worker = None

        self.start_button.setText("Apply EQ Settings")
        self.start_button.setEnabled(True)
        if DEBUG:
            logger.debug("EQ settings ready to apply")

    def _show_analysis_diagnostics(self, eq_settings: dict) -> None:
        """Show Auto-EQ confidence and validation details before applying."""
        confidence = float(eq_settings.get("analysis_confidence", 0.0) or 0.0)
        eq_confidence = float(eq_settings.get("eq_confidence", confidence) or 0.0)
        capture_confidence = float(eq_settings.get("capture_confidence", confidence) or 0.0)
        validation_confidence = float(eq_settings.get("validation_confidence", 0.0) or 0.0)
        state = _diagnostic_state(confidence)
        before = eq_settings.get("validation_before_error_db")
        after = eq_settings.get("validation_after_error_db")
        scale = eq_settings.get("validation_gain_scale")
        target_profile = eq_settings.get("target_profile", "--")
        residual = eq_settings.get("residual_regularization") or {}
        used_fallback = bool(eq_settings.get("used_spectrum_fallback", False))
        headroom = eq_settings.get("headroom_validation") or {}
        headroom_after = headroom.get("after") if isinstance(headroom, dict) else None
        headroom_safe = bool(headroom.get("safe", True)) if isinstance(headroom, dict) else True
        headroom_advisory = (
            bool(headroom.get("advisory", False)) if isinstance(headroom, dict) else False
        )
        headroom_scale = headroom.get("gain_scale") if isinstance(headroom, dict) else None

        self.confidence_label.setText(
            "Confidence: "
            f"overall {_format_percent(confidence)} | "
            f"EQ {_format_percent(eq_confidence)} | "
            f"capture {_format_percent(capture_confidence)}"
        )
        self.confidence_label.setStyleSheet(status_chip_style(state))
        self.error_label.setText(f"Target error: {_format_db(before)} -> {_format_db(after)}")
        self.error_label.setStyleSheet(status_chip_style("ok" if after is not None else "idle"))
        self.gain_scale_label.setText(
            f"Validation: {_format_percent(validation_confidence)} | gain scale {_format_percent(scale)}"
        )
        self.gain_scale_label.setStyleSheet(status_chip_style("info"))
        if isinstance(residual, dict) and "max_regularized_correction_db" in residual:
            requested = _format_db(residual.get("max_requested_correction_db"))
            regularized = _format_db(residual.get("max_regularized_correction_db"))
            narrow = _format_db(residual.get("max_narrow_residual_db"))
            self.gain_scale_label.setText(
                f"{self.gain_scale_label.text()} | correction {requested}->{regularized} | narrow {narrow}"
            )
        if isinstance(headroom_after, dict):
            pre_tp_headroom = _format_db(headroom_after.get("pre_limiter_true_peak_headroom_db"))
            limiter_gr = _format_db(headroom_after.get("limiter_gain_reduction_db"))
            true_peak_gr = _format_db(headroom_after.get("true_peak_limiter_gain_reduction_db"))
            headroom_status = (
                "advisory only (Rust simulator unavailable)"
                if headroom_advisory
                else "safe correction" if headroom_safe else "headroom risk"
            )
            self.gain_scale_label.setText(
                f"{self.gain_scale_label.text()} | {headroom_status}: "
                f"TP headroom {pre_tp_headroom}, LIM GR {limiter_gr}, TP GR {true_peak_gr}"
            )
            self.gain_scale_label.setStyleSheet(
                status_chip_style("info" if headroom_safe else "warn")
            )
            if headroom_scale is not None and float(headroom_scale) < 1.0:
                self.gain_scale_label.setText(
                    f"{self.gain_scale_label.text()} | headroom scale {_format_percent(headroom_scale)}"
                )
        self.target_profile_label.setText(
            f"Target profile: {target_profile}"
            + (" | fallback spectrum" if used_fallback else "")
        )
        self.target_profile_label.setStyleSheet(status_chip_style("info"))
        self.diagnostics_group.setVisible(True)

    def _on_analysis_failed(self, error: str):
        """Handle analysis failure."""
        if DEBUG:
            logger.debug("Analysis failed: %s", error)
        self.warning_label.setText(f"❌ {error}")
        self.warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 11pt;")
        self.diagnostics_group.setVisible(False)
        self.start_button.setText("Record Again")
        self.start_button.setEnabled(True)
        self.analysis_worker = None

    def _on_retake_clicked(self):
        """Discard and re-record."""
        reply = QMessageBox.question(
            self, "Discard Recording?",
            "Discard current recording and start over?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_recording_ui()

    def _on_cancel_clicked(self):
        """Cancel with confirmation if recording."""
        if self.recording_state == "recording":
            reply = QMessageBox.question(
                self, "Cancel Recording?",
                "Discard recording and return to main window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.recording_timer.stop()
                self._stop_analysis_worker()
                self._cleanup_recording_tap()
                self.reject()
        else:
            self.recording_timer.stop()
            self._stop_analysis_worker()
            self._cleanup_recording_tap()
            self.reject()

    def _reset_recording_ui(self):
        """Reset UI to initial idle state."""
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()

        # Stop processor if we started it ourselves
        self._stop_owned_processor()

        self.recording_state = "idle"
        self.audio_data = None
        self.progress_bar.setValue(0)
        self.time_label.setText(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")
        self.level_meter.set_levels(-120.0, -120.0)
        self.warning_label.setText("Ready to record")
        self.warning_label.setStyleSheet("color: gray; font-size: 11pt;")
        self.diagnostics_group.setVisible(False)
        self.start_button.setText("Start Calibration")
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(False)
        self.curve_combo.setEnabled(True)
        self.target_mode_combo.setEnabled(True)
        self.smoothing_combo.setEnabled(True)

    def _stop_owned_processor(self):
        """Stop the processor only if this dialog started it."""
        if not getattr(self, '_started_processor', False):
            return

        parent = _find_processor_owner(self.parent())
        if parent:
            try:
                if DEBUG:
                    logger.debug("Stopping audio processor owned by calibration dialog")
                parent.processor.stop()
            except Exception:
                if DEBUG:
                    logger.debug("Error stopping processor", exc_info=True)
        self._started_processor = False

    def _cleanup_recording_tap(self):
        """Best-effort cleanup for tap/mute state across cancel/close paths."""
        parent = _find_processor_owner(self.parent())

        if not parent:
            return

        try:
            parent.processor.stop_raw_recording()
        except Exception as e:
            logger.warning("Failed to stop raw recording during cleanup: %s", e)

        try:
            parent.processor.set_output_mute(False)
        except Exception as e:
            logger.warning("Failed to unmute output during cleanup: %s", e)

        try:
            parent.processor.set_recovery_suppressed(False)
        except Exception as e:
            logger.warning("Failed to re-enable recovery after cleanup: %s", e)

    def _stop_analysis_worker(self):
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.analysis_worker.wait(1500)
        self.analysis_worker = None

    def closeEvent(self, event):
        """Ensure recording state is cleaned up if dialog is closed directly."""
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        super().closeEvent(event)

    def accept(self):
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        super().accept()

    def reject(self):
        self.recording_timer.stop()
        self._stop_analysis_worker()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        super().reject()

    def get_recorded_audio(self):
        """
        Return the recorded audio data.

        Returns:
            tuple: (audio_data, sample_rate) where audio_data is NumPy array
                   of samples or None if no recording exists, sample_rate is int (Hz)

        This method returns the recorded voice sample for frequency analysis.
        """
        if self.audio_data is None:
            return None, None

        # Get sample rate from processor (via parent window)
        parent = _find_processor_owner(self.parent())

        if parent and hasattr(parent, 'processor'):
            return self.audio_data, _processor_sample_rate(parent)

        return self.audio_data, 48000

    def get_selected_curve(self):
        """
        Return the selected target curve key.

        Returns:
            str: Target curve key ('broadcast', 'podcast', 'streaming', or 'flat')
        """
        return self.curve_combo.currentData()

    def get_selected_target_mode(self):
        """Return the selected target behavior."""
        return self.target_mode_combo.currentData()

    def get_selected_smoothing_strength(self):
        """Return the selected Auto-EQ smoothing strength."""
        return self.smoothing_combo.currentData()
