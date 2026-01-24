"""
Noise Gate control panel

Adapted from Spectral Workbench project.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QDoubleSpinBox,
    QSlider,
    QLabel,
    QHBoxLayout,
)
from PyQt6.QtCore import Qt
from .rate_limiter import RateLimiter


class GatePanel(QWidget):
    """Noise Gate parameter control panel."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Noise Gate Group
        gate_group = QGroupBox("Noise Gate")
        gate_group.setMinimumWidth(280)
        gate_layout = QGridLayout(gate_group)
        gate_layout.setColumnStretch(0, 0)  # Label column - fixed width
        gate_layout.setColumnStretch(1, 1)  # Control column - stretches
        gate_layout.setColumnMinimumWidth(0, 70)  # Ensure labels fit

        # Enable checkbox
        self.enabled_checkbox = QCheckBox("Enable Noise Gate")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.setToolTip(
            "Reduces gain when signal falls below threshold.\n"
            "Helps eliminate background noise during silence."
        )
        gate_layout.addWidget(QLabel(""), 0, 0)
        gate_layout.addWidget(self.enabled_checkbox, 0, 1, 1, 2)

        # Threshold slider with spinbox
        threshold_layout = QHBoxLayout()

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(-80, -10)
        self.threshold_slider.setValue(-40)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-80.0, -10.0)
        self.threshold_spinbox.setSingleStep(1.0)
        self.threshold_spinbox.setValue(-40.0)
        self.threshold_spinbox.setSuffix(" dB")
        self.threshold_spinbox.setToolTip("Signal level below which gate closes")
        self.threshold_spinbox.setFixedWidth(80)
        threshold_layout.addWidget(self.threshold_spinbox)

        gate_layout.addWidget(QLabel("Threshold:"), 1, 0)
        gate_layout.addLayout(threshold_layout, 1, 1, 1, 2)

        # Attack time
        self.attack_spinbox = QDoubleSpinBox()
        self.attack_spinbox.setRange(0.1, 100.0)
        self.attack_spinbox.setSingleStep(1.0)
        self.attack_spinbox.setValue(10.0)
        self.attack_spinbox.setSuffix(" ms")
        self.attack_spinbox.setToolTip("Time for gate to open when signal exceeds threshold")
        gate_layout.addWidget(QLabel("Attack:"), 2, 0)
        gate_layout.addWidget(self.attack_spinbox, 2, 1, 1, 2)

        # Release time
        self.release_spinbox = QDoubleSpinBox()
        self.release_spinbox.setRange(10.0, 1000.0)
        self.release_spinbox.setSingleStep(10.0)
        self.release_spinbox.setValue(100.0)
        self.release_spinbox.setSuffix(" ms")
        self.release_spinbox.setToolTip("Time for gate to close when signal drops below threshold")
        gate_layout.addWidget(QLabel("Release:"), 3, 0)
        gate_layout.addWidget(self.release_spinbox, 3, 1, 1, 2)

        # Info label
        info_label = QLabel(
            "Gate uses 3dB hysteresis to prevent chattering.\n"
            "IIR envelope follower for smooth transitions."
        )
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        gate_layout.addWidget(QLabel(""), 4, 0)
        gate_layout.addWidget(info_label, 4, 1, 1, 2)

        layout.addWidget(gate_group)

    def _connect_signals(self):
        """Connect signals to slots."""
        self.enabled_checkbox.toggled.connect(self._update_gate)
        self.threshold_slider.valueChanged.connect(self._on_slider_changed)
        self.threshold_slider.sliderReleased.connect(self._rate_limiter.flush)
        self.threshold_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.attack_spinbox.valueChanged.connect(self._update_gate)
        self.release_spinbox.valueChanged.connect(self._update_gate)

        # Initial update
        self._update_gate()

    def _on_slider_changed(self, value):
        """Handle threshold slider change."""
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(float(value))
        self.threshold_spinbox.blockSignals(False)
        self._update_gate()

    def _on_spinbox_changed(self, value):
        """Handle threshold spinbox change."""
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(value))
        self.threshold_slider.blockSignals(False)
        self._update_gate()

    def _update_gate(self):
        """Update noise gate configuration."""
        enabled = self.enabled_checkbox.isChecked()
        threshold = self.threshold_spinbox.value()
        attack = self.attack_spinbox.value()
        release = self.release_spinbox.value()

        def apply():
            self.processor.set_gate_enabled(enabled)
            self.processor.set_gate_threshold(threshold)
            self.processor.set_gate_attack(attack)
            self.processor.set_gate_release(release)

        self._rate_limiter.call(apply)

    def get_settings(self) -> dict:
        """Get current gate settings as a dictionary."""
        return {
            'enabled': self.enabled_checkbox.isChecked(),
            'threshold_db': self.threshold_spinbox.value(),
            'attack_ms': self.attack_spinbox.value(),
            'release_ms': self.release_spinbox.value(),
        }

    def set_settings(self, settings: dict) -> None:
        """Apply settings from a dictionary."""
        if 'enabled' in settings:
            self.enabled_checkbox.setChecked(settings['enabled'])
        if 'threshold_db' in settings:
            self.threshold_spinbox.setValue(settings['threshold_db'])
            self.threshold_slider.setValue(int(settings['threshold_db']))
        if 'attack_ms' in settings:
            self.attack_spinbox.setValue(settings['attack_ms'])
        if 'release_ms' in settings:
            self.release_spinbox.setValue(settings['release_ms'])
        self._update_gate()
