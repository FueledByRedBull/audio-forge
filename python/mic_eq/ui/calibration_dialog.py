"""
Calibration dialog for Auto-EQ feature

DEBUG: Added terminal logging for calibration workflow
"""

# Enable debug logging (set to False for production)
DEBUG = False

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox,
    QPushButton, QTextEdit, QScrollArea, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from ..config import TARGET_CURVES
from .recording_worker import RecordingWorker
from .analysis_worker import AnalysisWorker
from .level_meter import LevelMeter
import numpy as np

# Rainbow Passage - standard calibration text from audiometry
RAINBOW_PASSAGE = """The Rainbow Passage
The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries, men have been fascinated by this spectacle of the sky. They have tried to find explanations for it. Scientists, however, have found that it is caused by the reflection and refraction of sunlight on drops of water in the air."""

# Recording validation thresholds (dB)
TOO_QUIET_DB = -40.0   # Warn if quieter than this
TOO_LOUD_DB = -3.0      # Warn if louder than this (clipping risk)
RECORDING_DURATION = 10.0  # Seconds


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
        self.worker: RecordingWorker | None = None
        self._started_processor = False  # Track if we started processor ourselves

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
            print(f"[CALIBRATION_DLG] Applying EQ settings...")

        # Get parent's EQ panel (MainWindow has it)
        parent = self.parent()
        while parent and not hasattr(parent, 'eq_panel'):
            parent = parent.parent()

        if not parent:
            QMessageBox.critical(self, "Error", "Could not find EQ panel")
            return

        # Build band tuples from eq_settings
        from ..config import EQ_FREQUENCIES as BAND_FREQUENCIES_HZ
        bands = []
        for i, freq in enumerate(BAND_FREQUENCIES_HZ):
            gain = self.eq_settings['band_gains'][i]
            q = self.eq_settings['band_qs'][i] if 'band_qs' in self.eq_settings else 1.41
            bands.append((freq, gain, q))

        # Apply settings to EQ panel
        parent.eq_panel.apply_auto_eq_results(bands)

        # Emit signal for main window to handle preset save and undo button
        target_curve = self.get_selected_curve()
        self.auto_eq_applied.emit(target_curve)

        if DEBUG:
            print(f"[CALIBRATION_DLG] EQ settings applied, signal emitted for curve={target_curve}")

        # Close dialog
        self.accept()

    def _start_recording(self):
        """Start non-blocking recording."""
        if DEBUG:
            print("[CALIBRATION_DLG] Start recording clicked")

        # Get parent's processor (MainWindow has it)
        parent = self.parent()
        while parent and not hasattr(parent, 'processor'):
            parent = parent.parent()

        if not parent:
            QMessageBox.critical(self, "Error", "Could not find audio processor")
            return

        # CRITICAL: Start processor from MAIN THREAD before spawning worker
        # The worker must NOT start/stop the processor (threading safety)
        processor_was_running = parent.processor.is_running()

        if DEBUG:
            print(f"[CALIBRATION_DLG] Processor was_running={processor_was_running}")

        if not processor_was_running:
            try:
                if DEBUG:
                    print("[CALIBRATION_DLG] Starting audio processor from main thread...")
                parent.processor.start(None, None)
                self._started_processor = True  # Track that we started it
                if DEBUG:
                    print("[CALIBRATION_DLG] Audio processor started successfully")
            except Exception as e:
                QMessageBox.critical(
                    self, "Audio Error",
                    f"Failed to start audio processing:\n{str(e)}\n\n"
                    "Check that audio devices are connected and not in use by another application."
                )
                return
        else:
            self._started_processor = False
            if DEBUG:
                print("[CALIBRATION_DLG] Processor already running, reusing existing session")

        # Small delay to let DSP loop initialize
        import time
        time.sleep(0.1)

        self.recording_state = "recording"
        self.start_button.setText("Recording...")
        self.start_button.setEnabled(False)  # Prevent early stop

        # Lock target curve selection during recording
        self.curve_combo.setEnabled(False)

        # Create and start recording worker (processor already running)
        if DEBUG:
            print(f"[CALIBRATION_DLG] Creating RecordingWorker with duration={RECORDING_DURATION}s")
        self.worker = RecordingWorker(
            parent.processor,
            duration=RECORDING_DURATION,
            processor_was_started_by_us=self._started_processor  # Pass flag
        )
        self.worker.progress.connect(self._on_progress_update)
        self.worker.time_remaining.connect(self._on_time_remaining)
        self.worker.level_update.connect(self._on_level_update)
        self.worker.finished.connect(self._on_recording_complete)
        self.worker.failed.connect(self._on_recording_failed)
        self.worker.start()
        if DEBUG:
            print("[CALIBRATION_DLG] RecordingWorker started")

        self.warning_label.setText("Recording... Speak clearly into your microphone")
        self.warning_label.setStyleSheet("color: blue; font-weight: bold; font-size: 11pt;")

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
            print(f"[CALIBRATION_DLG] Recording complete: {len(audio_data)} samples")
            import numpy as np
            rms = np.mean(audio_data**2)**0.5
            peak_db = 20 * np.log10(max(np.abs(audio_data).max(), 1e-6))
            rms_db = 20 * np.log10(max(rms, 1e-6))
            print(f"[CALIBRATION_DLG] Audio stats - Peak: {peak_db:.1f} dB, RMS: {rms_db:.1f} dB")

        self.audio_data = audio_data
        self.recording_state = "completed"

        # Update UI
        self.start_button.setText("Record Again")
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(True)

        # Re-enable curve selection
        self.curve_combo.setEnabled(True)

        # Show completion message
        self.warning_label.setText(f"Recording complete! {len(audio_data)} samples captured")
        self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")

        # Phase 20 TODO: Analysis engine exists but not integrated yet
        # Phase 19 built the algorithms, Phase 20 will wire them to this dialog
        if DEBUG:
            print("[CALIBRATION_DLG] NOTE: Analysis engine built (Phase 19) but not integrated")
            print("[CALIBRATION_DLG] TODO: Phase 20 will integrate AnalysisWorker here")
            print("[CALIBRATION_DLG] For now, audio captured but not analyzed")

        # Auto-start analysis (Phase 19-20 bridge integration)
        # This provides immediate value even before Phase 20 full UI
        if DEBUG:
            print("[ANALYSIS] Starting analysis engine (Phase 19 bridge)...")
        self._start_analysis()

    def _on_recording_failed(self, error: str):
        """Handle recording failure."""
        self.warning_label.setText(f"❌ Recording failed: {error}")
        self.warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 11pt;")
        self._reset_recording_ui()

    def _start_analysis(self):
        """Start analysis worker (Phase 19 bridge integration)."""
        if self.audio_data is None:
            if DEBUG:
                print("[ANALYSIS] No audio data to analyze")
            return

        # Get parent's processor for sample rate
        parent = self.parent()
        while parent and not hasattr(parent, 'processor'):
            parent = parent.parent()

        if not parent:
            if DEBUG:
                print("[ANALYSIS] ERROR: Could not find processor")
            return

        sample_rate = 48000  # Fixed for now
        target_preset = self.get_selected_curve()

        if DEBUG:
            print(f"[ANALYSIS] Creating AnalysisWorker: {len(self.audio_data)} samples, {sample_rate}Hz, target={target_preset}")

        # Create and start analysis worker
        self.analysis_worker = AnalysisWorker(self.audio_data, sample_rate, target_preset)
        self.analysis_worker.step_progress.connect(self._on_analysis_step)
        self.analysis_worker.finished.connect(self._on_analysis_complete)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.start()

        if DEBUG:
            print("[ANALYSIS] AnalysisWorker started")

    def _on_analysis_step(self, step_name: str, percentage: int):
        """Handle analysis step progress."""
        if DEBUG:
            print(f"[ANALYSIS] Step {percentage}%: {step_name}")
        self.warning_label.setText(f"Analyzing: {step_name}")
        self.progress_bar.setValue(percentage)

    def _on_analysis_complete(self, eq_settings: dict):
        """Handle analysis completion."""
        if DEBUG:
            print(f"[ANALYSIS] Analysis complete!")
            print(f"[ANALYSIS] Band gains: {[round(g, 1) for g in eq_settings['band_gains']]}")
            max_gain = max(abs(g) for g in eq_settings['band_gains'])
            print(f"[ANALYSIS] Max correction: {round(max_gain, 1)} dB")

        self.eq_settings = eq_settings
        self.warning_label.setText(f"✓ Analysis complete! Max correction: {round(max(abs(g) for g in eq_settings['band_gains']), 1)} dB")
        self.warning_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")
        self.progress_bar.setValue(100)

        # Enable apply button (will be part of Phase 20)
        self.start_button.setText("Apply EQ Settings")
        self.start_button.setEnabled(True)
        if DEBUG:
            print("[ANALYSIS] TODO: Phase 20 will apply these settings to UI")

    def _on_analysis_failed(self, error: str):
        """Handle analysis failure."""
        if DEBUG:
            print(f"[ANALYSIS] Analysis failed: {error}")
        self.warning_label.setText(f"❌ {error}")
        self.warning_label.setStyleSheet("color: orange; font-weight: bold; font-size: 11pt;")
        self.start_button.setText("Record Again")
        self.start_button.setEnabled(True)

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
                if self.worker:
                    self.worker.stop()
                    self.worker.wait(1500)
                self._cleanup_recording_tap()
                self.reject()
        else:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1500)
            self._cleanup_recording_tap()
            self.reject()

    def _reset_recording_ui(self):
        """Reset UI to initial idle state."""
        self._cleanup_recording_tap()

        # Stop processor if we started it ourselves
        if hasattr(self, '_started_processor') and self._started_processor:
            parent = self.parent()
            while parent and not hasattr(parent, 'processor'):
                parent = parent.parent()
            if parent:
                try:
                    if DEBUG:
                        print("[CALIBRATION_DLG] Stopping audio processor (we started it)")
                    parent.processor.stop()
                except Exception as e:
                    if DEBUG:
                        print(f"[CALIBRATION_DLG] Error stopping processor: {e}")
            self._started_processor = False

        self.recording_state = "idle"
        self.audio_data = None
        self.progress_bar.setValue(0)
        self.time_label.setText(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setStyleSheet("font-size: 12pt; color: #4a90d9; font-weight: bold;")
        self.level_meter.set_levels(-120.0, -120.0)
        self.warning_label.setText("Ready to record")
        self.warning_label.setStyleSheet("color: gray; font-size: 11pt;")
        self.start_button.setText("Start Calibration")
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(False)
        self.curve_combo.setEnabled(True)

    def _cleanup_recording_tap(self):
        """Best-effort cleanup for tap/mute state across cancel/close paths."""
        parent = self.parent()
        while parent and not hasattr(parent, 'processor'):
            parent = parent.parent()

        if not parent:
            return

        try:
            parent.processor.stop_raw_recording()
        except Exception:
            pass

        try:
            parent.processor.set_output_mute(False)
        except Exception:
            pass

    def closeEvent(self, event):
        """Ensure recording state is cleaned up if dialog is closed directly."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self._cleanup_recording_tap()
        super().closeEvent(event)

    def get_recorded_audio(self):
        """
        Return the recorded audio data.

        Returns:
            tuple: (audio_data, sample_rate) where audio_data is NumPy array
                   of samples or None if no recording exists, sample_rate is int (Hz)

        This method is called by Phase 19's analysis engine to retrieve
        the recorded voice sample for frequency analysis.
        """
        if self.audio_data is None:
            return None, None

        # Get sample rate from processor (via parent window)
        parent = self.parent()
        while parent and not hasattr(parent, 'processor'):
            parent = parent.parent()

        if parent and hasattr(parent, 'processor'):
            sample_rate = parent.processor.sample_rate()
            return self.audio_data, sample_rate

        return self.audio_data, 48000  # Fallback to 48kHz

    def get_selected_curve(self):
        """
        Return the selected target curve key.

        Returns:
            str: Target curve key ('broadcast', 'podcast', 'streaming', or 'flat')
        """
        return self.curve_combo.currentData()
