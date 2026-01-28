"""
Calibration dialog for Auto-EQ feature
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox,
    QPushButton, QTextEdit, QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt
from ..config import TARGET_CURVES

# Rainbow Passage - standard calibration text from audiometry
RAINBOW_PASSAGE = """The Rainbow Passage
The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries, men have been fascinated by this spectacle of the sky. They have tried to find explanations for it. Scientists, however, have found that it is caused by the reflection and refraction of sunlight on drops of water in the air."""


class CalibrationDialog(QDialog):
    """Auto-EQ calibration dialog with target curve selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-EQ Calibration")
        self.setModal(True)  # Modal dialog - blocks main window
        self.setMinimumWidth(600)
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

        # Start button (recording will be implemented in Phase 18)
        self.start_button = QPushButton("Start Calibration")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }"
        )
        # For now, just show message - Phase 18 will implement recording
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
        """Handle Start Calibration button click."""
        # Phase 18 will implement actual recording
        # For now, just close and show placeholder
        QMessageBox.information(
            self,
            "Recording Coming Soon",
            "Recording will be implemented in Phase 18.\n\n"
            f"Selected curve: {self.curve_combo.currentText()}\n"
            "This dialog will remain open during recording."
        )

    def get_selected_curve(self) -> str:
        """Return the selected target curve key."""
        return self.curve_combo.currentData()
