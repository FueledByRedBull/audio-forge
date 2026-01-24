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
    QComboBox,
)
from PyQt6.QtCore import Qt
from .rate_limiter import RateLimiter
from .layout_constants import (
    SPACING_NORMAL, MARGIN_PANEL,
    PRIMARY_LABEL_STYLE, INFO_LABEL_STYLE
)


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
        gate_layout.setSpacing(SPACING_NORMAL)
        gate_layout.setContentsMargins(MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL)

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

        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(threshold_label, 1, 0)
        gate_layout.addLayout(threshold_layout, 1, 1, 1, 2)

        # Attack time
        self.attack_spinbox = QDoubleSpinBox()
        self.attack_spinbox.setRange(0.1, 100.0)
        self.attack_spinbox.setSingleStep(1.0)
        self.attack_spinbox.setValue(10.0)
        self.attack_spinbox.setSuffix(" ms")
        self.attack_spinbox.setToolTip("Time for gate to open when signal exceeds threshold")
        attack_label = QLabel("Attack:")
        attack_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(attack_label, 2, 0)
        gate_layout.addWidget(self.attack_spinbox, 2, 1, 1, 2)

        # Release time
        self.release_spinbox = QDoubleSpinBox()
        self.release_spinbox.setRange(10.0, 1000.0)
        self.release_spinbox.setSingleStep(10.0)
        self.release_spinbox.setValue(100.0)
        self.release_spinbox.setSuffix(" ms")
        self.release_spinbox.setToolTip("Time for gate to close when signal drops below threshold")
        release_label = QLabel("Release:")
        release_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(release_label, 3, 0)
        gate_layout.addWidget(self.release_spinbox, 3, 1, 1, 2)

        # Separator
        gate_layout.addWidget(QLabel(""), 4, 0)
        gate_layout.addWidget(QLabel(""), 4, 1)

        # Gate Mode section
        mode_label = QLabel("Gate Mode:")
        mode_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(mode_label, 5, 0)

        # Mode dropdown
        self.gate_mode_combo = QComboBox()
        self.gate_mode_combo.addItems([
            "Threshold Only",
            "VAD Assisted",
            "VAD Only"
        ])
        self.gate_mode_combo.setCurrentIndex(0)
        self.gate_mode_combo.setToolTip(
            "Threshold Only: Traditional gate using level threshold\n"
            "VAD Assisted: Gate opens when level exceeded OR speech detected\n"
            "VAD Only: Gate opens solely based on speech probability"
        )
        gate_layout.addWidget(QLabel(""), 6, 0)
        gate_layout.addWidget(self.gate_mode_combo, 6, 1, 1, 2)

        # VAD threshold slider
        vad_threshold_layout = QHBoxLayout()
        self.vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_threshold_slider.setRange(30, 80)  # 0.3 to 0.8
        self.vad_threshold_slider.setValue(50)
        self.vad_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.vad_threshold_slider.setTickInterval(10)
        vad_threshold_layout.addWidget(self.vad_threshold_slider)

        self.vad_threshold_spinbox = QDoubleSpinBox()
        self.vad_threshold_spinbox.setRange(0.3, 0.8)
        self.vad_threshold_spinbox.setSingleStep(0.05)
        self.vad_threshold_spinbox.setValue(0.5)
        self.vad_threshold_spinbox.setDecimals(2)
        self.vad_threshold_spinbox.setToolTip("Speech probability threshold (0.3-0.8)")
        self.vad_threshold_spinbox.setFixedWidth(80)
        vad_threshold_layout.addWidget(self.vad_threshold_spinbox)

        vad_threshold_label = QLabel("VAD Threshold:")
        vad_threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(vad_threshold_label, 7, 0)
        gate_layout.addLayout(vad_threshold_layout, 7, 1, 1, 2)

        # Hold time
        self.vad_hold_spinbox = QDoubleSpinBox()
        self.vad_hold_spinbox.setRange(0.0, 500.0)
        self.vad_hold_spinbox.setSingleStep(10.0)
        self.vad_hold_spinbox.setValue(200.0)
        self.vad_hold_spinbox.setSuffix(" ms")
        self.vad_hold_spinbox.setToolTip("Gate hold time after speech ends (prevents chatter)")
        hold_time_label = QLabel("Hold Time:")
        hold_time_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(hold_time_label, 8, 0)
        gate_layout.addWidget(self.vad_hold_spinbox, 8, 1, 1, 2)

        # VAD confidence meter
        from .level_meter import ConfidenceMeter
        self.confidence_meter = ConfidenceMeter()
        self.confidence_meter.setToolTip("Real-time VAD confidence (red=low, green=high)")
        self.vad_info_label = QLabel("VAD: N/A")
        self.vad_info_label.setStyleSheet(INFO_LABEL_STYLE)
        vad_meter_layout = QVBoxLayout()
        vad_meter_layout.setContentsMargins(0, 0, 0, 0)
        vad_meter_layout.setSpacing(2)
        vad_meter_layout.addWidget(self.confidence_meter)
        vad_meter_layout.addWidget(self.vad_info_label)
        confidence_label = QLabel("Confidence:")
        confidence_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addWidget(confidence_label, 9, 0)
        gate_layout.addLayout(vad_meter_layout, 9, 1, 1, 2)

        # Info label
        info_label = QLabel(
            "Gate uses 3dB hysteresis to prevent chattering.\n"
            "IIR envelope follower for smooth transitions."
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        gate_layout.addWidget(QLabel(""), 10, 0)
        gate_layout.addWidget(info_label, 10, 1, 1, 2)

        layout.addWidget(gate_group)

    def _connect_signals(self):
        """Connect signals to slots."""
        self.enabled_checkbox.toggled.connect(self._update_gate)
        self.threshold_slider.valueChanged.connect(self._on_slider_changed)
        self.threshold_slider.sliderReleased.connect(self._rate_limiter.flush)
        self.threshold_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.attack_spinbox.valueChanged.connect(self._update_gate)
        self.release_spinbox.valueChanged.connect(self._update_gate)

        # VAD control signals
        self.gate_mode_combo.currentIndexChanged.connect(self._update_vad_mode)
        self.vad_threshold_slider.valueChanged.connect(self._on_vad_threshold_slider)
        self.vad_threshold_spinbox.valueChanged.connect(self._on_vad_threshold_spinbox)
        self.vad_hold_spinbox.valueChanged.connect(self._update_vad_mode)

        # Initial update
        self._update_gate()
        self._update_vad_controls_enabled()

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

    def _on_vad_threshold_slider(self, value):
        """Handle VAD threshold slider change."""
        threshold = value / 100.0  # Convert 30-80 to 0.3-0.8
        self.vad_threshold_spinbox.blockSignals(True)
        self.vad_threshold_spinbox.setValue(threshold)
        self.vad_threshold_spinbox.blockSignals(False)
        self._update_vad_mode()

    def _on_vad_threshold_spinbox(self, value):
        """Handle VAD threshold spinbox change."""
        self.vad_threshold_slider.blockSignals(True)
        self.vad_threshold_slider.setValue(int(value * 100))
        self.vad_threshold_slider.blockSignals(False)
        self._update_vad_mode()

    def _update_vad_mode(self):
        """Update VAD mode and settings."""
        try:
            mode = self.gate_mode_combo.currentIndex()
            self.processor.set_gate_mode(mode)
            self.processor.set_vad_threshold(self.vad_threshold_spinbox.value())
            self.processor.set_vad_hold_time(self.vad_hold_spinbox.value())
            self._update_vad_controls_enabled()
            self.vad_info_label.setText("VAD: Active")
        except Exception as e:
            self.vad_info_label.setText(f"VAD error: {e}")

    def _update_vad_controls_enabled(self):
        """Enable/disable VAD controls based on gate mode."""
        mode = self.gate_mode_combo.currentIndex()
        # 0 = Threshold Only, 1 = VAD Assisted, 2 = VAD Only
        vad_enabled = mode > 0
        threshold_enabled = mode != 2  # Disabled in VAD Only mode

        # Enable/disable VAD controls
        self.vad_threshold_slider.setEnabled(vad_enabled)
        self.vad_threshold_spinbox.setEnabled(vad_enabled)
        self.vad_hold_spinbox.setEnabled(vad_enabled)
        self.confidence_meter.setEnabled(vad_enabled)

        # Enable/disable level threshold
        self.threshold_slider.setEnabled(threshold_enabled)
        self.threshold_spinbox.setEnabled(threshold_enabled)

    def update_vad_confidence(self, confidence: float):
        """Update VAD confidence meter (called from main window)."""
        self.confidence_meter.set_confidence(confidence)

    def get_settings(self) -> dict:
        """Get current gate settings as a dictionary."""
        settings = {
            'enabled': self.enabled_checkbox.isChecked(),
            'threshold_db': self.threshold_spinbox.value(),
            'attack_ms': self.attack_spinbox.value(),
            'release_ms': self.release_spinbox.value(),
            'gate_mode': self.gate_mode_combo.currentIndex(),
            'vad_threshold': self.vad_threshold_spinbox.value(),
            'vad_hold_time_ms': self.vad_hold_spinbox.value(),
        }
        return settings

    def set_settings(self, settings: dict) -> None:
        """Apply settings from a dictionary with proper signal blocking."""
        if 'enabled' in settings:
            self.enabled_checkbox.blockSignals(True)
            self.enabled_checkbox.setChecked(settings['enabled'])
            self.enabled_checkbox.blockSignals(False)
        if 'threshold_db' in settings:
            self.threshold_spinbox.blockSignals(True)
            self.threshold_slider.blockSignals(True)
            self.threshold_spinbox.setValue(settings['threshold_db'])
            self.threshold_slider.setValue(int(settings['threshold_db']))
            self.threshold_spinbox.blockSignals(False)
            self.threshold_slider.blockSignals(False)
        if 'attack_ms' in settings:
            self.attack_spinbox.blockSignals(True)
            self.attack_spinbox.setValue(settings['attack_ms'])
            self.attack_spinbox.blockSignals(False)
        if 'release_ms' in settings:
            self.release_spinbox.blockSignals(True)
            self.release_spinbox.setValue(settings['release_ms'])
            self.release_spinbox.blockSignals(False)

        # VAD mode settings (v1.2.0+)
        if 'gate_mode' in settings:
            self.gate_mode_combo.blockSignals(True)
            self.gate_mode_combo.setCurrentIndex(settings['gate_mode'])
            self.gate_mode_combo.blockSignals(False)
        if 'vad_threshold' in settings:
            self.vad_threshold_spinbox.blockSignals(True)
            self.vad_threshold_slider.blockSignals(True)
            self.vad_threshold_spinbox.setValue(settings['vad_threshold'])
            self.vad_threshold_slider.setValue(int(settings['vad_threshold'] * 100))
            self.vad_threshold_spinbox.blockSignals(False)
            self.vad_threshold_slider.blockSignals(False)
        if 'vad_hold_time_ms' in settings:
            self.vad_hold_spinbox.blockSignals(True)
            self.vad_hold_spinbox.setValue(settings['vad_hold_time_ms'])
            self.vad_hold_spinbox.blockSignals(False)

        # Update processor and UI state after all settings applied
        self._update_gate()
        self._update_vad_mode()
        self._update_vad_controls_enabled()
