"""
Calibration dialog for Auto-EQ feature

DEBUG: Added terminal logging for calibration workflow
"""

# Enable debug logging
DEBUG = True

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox,
    QPushButton, QTextEdit, QScrollArea, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer
from ..config import TARGET_CURVES
from .recording_worker import RecordingWorker
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-EQ Calibration")
        self.setModal(True)  # Modal dialog - blocks main window
        self.setMinimumWidth(600)

        # Recording state
        self.recording_state = "idle"  # idle, recording, completed
        self.audio_data: np.ndarray | None = None
        self.worker: RecordingWorker | None = None

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
            # Start new recording
            self._reset_recording_ui()
            self._start_recording()

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

        self.recording_state = "recording"
        self.start_button.setText("Recording...")
        self.start_button.setEnabled(False)  # Prevent early stop

        # Lock target curve selection during recording
        self.curve_combo.setEnabled(False)

        # Create and start recording worker
        if DEBUG:
            print(f"[CALIBRATION_DLG] Creating RecordingWorker with duration={RECORDING_DURATION}s")
        self.worker = RecordingWorker(parent.processor, duration=RECORDING_DURATION)
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

        # TODO: Phase 19 will pass audio_data to analysis engine
        if DEBUG:
            print("[CALIBRATION_DLG] TODO: Phase 19 will analyze this audio")

    def _on_recording_failed(self, error: str):
        """Handle recording failure."""
        self.warning_label.setText(f"❌ Recording failed: {error}")
        self.warning_label.setStyleSheet("color: red; font-weight: bold; font-size: 11pt;")
        self._reset_recording_ui()

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
                self.reject()
        else:
            self.reject()

    def _reset_recording_ui(self):
        """Reset UI to initial idle state."""
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
