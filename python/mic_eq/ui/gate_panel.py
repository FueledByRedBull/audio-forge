"""
Noise Gate control panel

Adapted from Spectral Workbench project.
"""

import logging

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QCheckBox,
    QDoubleSpinBox,
    QSlider,
    QLabel,
    QHBoxLayout,
    QComboBox,
)
from PyQt6.QtCore import Qt
from .rate_limiter import RateLimiter
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    SPACING_NORMAL,
    MARGIN_PANEL,
    PRIMARY_LABEL_STYLE,
    INFO_LABEL_STYLE,
    fit_spinbox_to_contents,
)


logger = logging.getLogger(__name__)


class GatePanel(QWidget):
    """Noise Gate parameter control panel."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._latest_noise_floor_db = -60.0
        self._preserve_unavailable_vad_mode = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Noise Gate Group
        gate_group = QGroupBox("Noise Gate")
        gate_layout = QFormLayout(gate_group)
        gate_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        gate_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        gate_layout.setSpacing(SPACING_NORMAL)
        gate_layout.setContentsMargins(
            MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL
        )

        # Enable checkbox
        self.enabled_checkbox = QCheckBox("Enable Noise Gate")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.setToolTip(
            "Reduces gain when signal falls below threshold.\n"
            "Helps eliminate background noise during silence."
        )
        gate_layout.addRow(self.enabled_checkbox)

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
        fit_spinbox_to_contents(self.threshold_spinbox)
        threshold_layout.addWidget(self.threshold_spinbox)

        self.threshold_label = QLabel("Manual Threshold:")
        self.threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(self.threshold_label, threshold_layout)

        # Attack time
        self.attack_spinbox = QDoubleSpinBox()
        self.attack_spinbox.setRange(0.1, 100.0)
        self.attack_spinbox.setSingleStep(1.0)
        self.attack_spinbox.setValue(10.0)
        self.attack_spinbox.setSuffix(" ms")
        self.attack_spinbox.setToolTip(
            "Time for gate to open when signal exceeds threshold"
        )
        fit_spinbox_to_contents(self.attack_spinbox)
        attack_label = QLabel("Attack:")
        attack_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(attack_label, self.attack_spinbox)

        # Release time
        self.release_spinbox = QDoubleSpinBox()
        self.release_spinbox.setRange(10.0, 1000.0)
        self.release_spinbox.setSingleStep(10.0)
        self.release_spinbox.setValue(100.0)
        self.release_spinbox.setSuffix(" ms")
        self.release_spinbox.setToolTip(
            "Time for gate to close when signal drops below threshold"
        )
        fit_spinbox_to_contents(self.release_spinbox)
        release_label = QLabel("Release:")
        release_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(release_label, self.release_spinbox)

        # Gate Mode section
        mode_label = QLabel("Gate Mode:")
        mode_label.setStyleSheet(PRIMARY_LABEL_STYLE)

        # Mode dropdown
        self.gate_mode_combo = QComboBox()
        self.gate_mode_combo.addItems(["Threshold Only", "VAD Assisted", "VAD Only"])
        self.gate_mode_combo.setCurrentIndex(0)
        self.gate_mode_combo.setToolTip(
            "Threshold Only: Traditional gate using level threshold\n"
            "VAD Assisted: Gate opens when level exceeded OR speech detected\n"
            "VAD Only: Gate opens solely based on speech probability"
        )
        gate_layout.addRow(mode_label, self.gate_mode_combo)

        # VAD threshold slider
        vad_threshold_layout = QHBoxLayout()
        self.vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_threshold_slider.setRange(30, 80)  # 0.3 to 0.8
        self.vad_threshold_slider.setValue(48)
        self.vad_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.vad_threshold_slider.setTickInterval(10)
        vad_threshold_layout.addWidget(self.vad_threshold_slider)

        self.vad_threshold_spinbox = QDoubleSpinBox()
        self.vad_threshold_spinbox.setRange(0.3, 0.8)
        self.vad_threshold_spinbox.setSingleStep(0.01)
        self.vad_threshold_spinbox.setValue(0.48)
        self.vad_threshold_spinbox.setDecimals(2)
        self.vad_threshold_spinbox.setToolTip("Speech probability threshold (0.3-0.8)")
        fit_spinbox_to_contents(self.vad_threshold_spinbox)
        vad_threshold_layout.addWidget(self.vad_threshold_spinbox)

        vad_threshold_label = QLabel("VAD Threshold:")
        vad_threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(vad_threshold_label, vad_threshold_layout)

        # Hold time
        self.vad_hold_spinbox = QDoubleSpinBox()
        self.vad_hold_spinbox.setRange(0.0, 500.0)
        self.vad_hold_spinbox.setSingleStep(10.0)
        self.vad_hold_spinbox.setValue(200.0)
        self.vad_hold_spinbox.setSuffix(" ms")
        self.vad_hold_spinbox.setToolTip(
            "Gate hold time after speech ends (prevents chatter)"
        )
        fit_spinbox_to_contents(self.vad_hold_spinbox)
        hold_time_label = QLabel("Hold Time:")
        hold_time_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(hold_time_label, self.vad_hold_spinbox)

        # VAD Pre-Gain slider and spinbox (boosts weak signals for better detection)
        vad_pre_gain_layout = QHBoxLayout()
        self.vad_pre_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_pre_gain_slider.setRange(10, 100)  # 1.0 to 10.0
        self.vad_pre_gain_slider.setValue(10)  # Default 1.0
        self.vad_pre_gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.vad_pre_gain_slider.setTickInterval(10)
        vad_pre_gain_layout.addWidget(self.vad_pre_gain_slider)

        self.vad_pre_gain_spinbox = QDoubleSpinBox()
        self.vad_pre_gain_spinbox.setRange(1.0, 10.0)
        self.vad_pre_gain_spinbox.setSingleStep(0.5)
        self.vad_pre_gain_spinbox.setValue(1.0)
        self.vad_pre_gain_spinbox.setDecimals(1)
        self.vad_pre_gain_spinbox.setToolTip(
            "Pre-gain to boost weak signals for better VAD detection"
        )
        fit_spinbox_to_contents(self.vad_pre_gain_spinbox)
        vad_pre_gain_layout.addWidget(self.vad_pre_gain_spinbox)

        vad_pre_gain_label = QLabel("VAD Pre-Gain:")
        vad_pre_gain_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(vad_pre_gain_label, vad_pre_gain_layout)

        # Auto Threshold section
        self.auto_threshold_checkbox = QCheckBox("Auto Threshold")
        self.auto_threshold_checkbox.setChecked(True)
        self.auto_threshold_checkbox.setToolTip(
            "Automatically adjust gate threshold based on estimated noise floor.\n"
            "Recommended for VAD modes.\n"
            "Gate threshold = noise_floor + margin"
        )
        gate_layout.addRow(self.auto_threshold_checkbox)

        # Margin slider and spinbox
        margin_layout = QHBoxLayout()
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(0, 20)  # 0 to 20 dB
        self.margin_slider.setValue(10)  # Default 10 dB
        self.margin_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.margin_slider.setTickInterval(5)
        margin_layout.addWidget(self.margin_slider)

        self.margin_spinbox = QDoubleSpinBox()
        self.margin_spinbox.setRange(0.0, 20.0)
        self.margin_spinbox.setSingleStep(1.0)
        self.margin_spinbox.setValue(10.0)
        self.margin_spinbox.setSuffix(" dB")
        self.margin_spinbox.setToolTip(
            "Margin above noise floor for gate threshold (0-20 dB)"
        )
        fit_spinbox_to_contents(self.margin_spinbox)
        margin_layout.addWidget(self.margin_spinbox)

        margin_label = QLabel("Margin:")
        margin_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(margin_label, margin_layout)

        # Noise floor display (read-only)
        self.noise_floor_label = QLabel("Noise Floor: -60 dB")
        self.noise_floor_label.setStyleSheet(INFO_LABEL_STYLE)
        self.noise_floor_label.setWordWrap(True)
        gate_layout.addRow(self.noise_floor_label)

        self.threshold_status_label = QLabel("Effective Threshold: Manual -40.0 dB")
        self.threshold_status_label.setStyleSheet(INFO_LABEL_STYLE)
        self.threshold_status_label.setWordWrap(True)
        gate_layout.addRow(self.threshold_status_label)

        # VAD confidence meter
        from .level_meter import ConfidenceMeter

        self.confidence_meter = ConfidenceMeter()
        self.confidence_meter.setToolTip(
            "Real-time VAD confidence (red=low, green=high)"
        )
        self.vad_info_label = QLabel("VAD: N/A")
        self.vad_info_label.setStyleSheet(INFO_LABEL_STYLE)
        self.vad_info_label.setWordWrap(True)
        vad_meter_layout = QVBoxLayout()
        vad_meter_layout.setContentsMargins(0, 0, 0, 0)
        vad_meter_layout.setSpacing(2)
        vad_meter_layout.addWidget(self.confidence_meter)
        vad_meter_layout.addWidget(self.vad_info_label)
        confidence_label = QLabel("Confidence:")
        confidence_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        gate_layout.addRow(confidence_label, vad_meter_layout)

        # Info label
        info_label = QLabel(
            "Gate uses 3dB hysteresis to prevent chattering.\n"
            "IIR envelope follower for smooth transitions."
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        info_label.setWordWrap(True)
        gate_layout.addRow(info_label)

        layout.addWidget(gate_group)

        bind_label(
            self.threshold_label,
            self.threshold_spinbox,
            name="Gate manual threshold",
        )
        bind_label(attack_label, self.attack_spinbox, name="Gate attack time")
        bind_label(release_label, self.release_spinbox, name="Gate release time")
        bind_label(mode_label, self.gate_mode_combo, name="Gate operating mode")
        bind_label(
            vad_threshold_label,
            self.vad_threshold_spinbox,
            name="Voice activity threshold",
        )
        bind_label(
            hold_time_label,
            self.vad_hold_spinbox,
            name="Voice activity hold time",
        )
        bind_label(
            vad_pre_gain_label,
            self.vad_pre_gain_spinbox,
            name="Voice activity pre-gain",
        )
        bind_label(margin_label, self.margin_spinbox, name="Automatic gate margin")
        set_accessible_group(
            (
                (self.enabled_checkbox, "Enable noise gate", None),
                (self.threshold_slider, "Gate manual threshold", None),
                (self.vad_threshold_slider, "Voice activity threshold", None),
                (self.vad_pre_gain_slider, "Voice activity pre-gain", None),
                (self.auto_threshold_checkbox, "Enable automatic gate threshold", None),
                (self.margin_slider, "Automatic gate margin", None),
                (self.confidence_meter, "Voice activity confidence", None),
            )
        )

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
        self.vad_pre_gain_slider.valueChanged.connect(self._on_vad_pre_gain_slider)
        self.vad_pre_gain_spinbox.valueChanged.connect(self._on_vad_pre_gain_spinbox)

        # Auto-threshold control signals
        self.auto_threshold_checkbox.toggled.connect(self._update_auto_threshold)
        self.margin_slider.valueChanged.connect(self._on_margin_slider)
        self.margin_spinbox.valueChanged.connect(self._on_margin_spinbox)

        # Initial update
        self._update_gate()
        # Initialize VAD settings (including pre-gain)
        try:
            self._update_vad_mode()
        except (AttributeError, Exception):
            # VAD not available or other error - will be handled when user enables VAD
            logger.debug("Initial VAD setup skipped", exc_info=True)
        self._update_auto_threshold()
        self._refresh_threshold_summary()

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

    def _on_vad_pre_gain_slider(self, value):
        """Handle VAD pre-gain slider change."""
        gain = value / 10.0  # Convert 10-100 to 1.0-10.0
        self.vad_pre_gain_spinbox.blockSignals(True)
        self.vad_pre_gain_spinbox.setValue(gain)
        self.vad_pre_gain_spinbox.blockSignals(False)
        self._update_vad_mode()

    def _on_vad_pre_gain_spinbox(self, value):
        """Handle VAD pre-gain spinbox change."""
        self.vad_pre_gain_slider.blockSignals(True)
        self.vad_pre_gain_slider.setValue(int(value * 10))
        self.vad_pre_gain_slider.blockSignals(False)
        self._update_vad_mode()

    def _is_vad_available(self) -> bool:
        """Return True when Rust VAD backend is available."""
        try:
            return bool(self.processor.is_vad_available())
        except Exception:
            logger.debug("VAD availability check error", exc_info=True)
            return False

    def _is_auto_threshold_active(self) -> bool:
        mode = self.gate_mode_combo.currentIndex()
        return (
            mode > 0
            and self._is_vad_available()
            and self.auto_threshold_checkbox.isChecked()
        )

    def _set_vad_status_text(self, mode: int, vad_available: bool) -> None:
        if mode == 0:
            self.vad_info_label.setText("VAD: Threshold mode")
        elif vad_available:
            if self.auto_threshold_checkbox.isChecked():
                self.vad_info_label.setText("VAD: Active | Auto threshold on")
            else:
                self.vad_info_label.setText("VAD: Active | Manual threshold")
        else:
            self.vad_info_label.setText("VAD: Unavailable")

    def _refresh_threshold_summary(self):
        manual_threshold = self.threshold_spinbox.value()
        self.noise_floor_label.setText(
            f"Noise Floor: {self._latest_noise_floor_db:.1f} dB"
        )
        if self._is_auto_threshold_active():
            margin_db = self.margin_spinbox.value()
            effective_threshold = max(
                -80.0, min(-10.0, self._latest_noise_floor_db + margin_db)
            )
            self.threshold_label.setText("Manual Threshold (fallback):")
            self.threshold_status_label.setText(
                f"Effective Threshold: {effective_threshold:.1f} dB "
                f"({self._latest_noise_floor_db:.1f} dB floor + {margin_db:.1f} dB margin)"
            )
        else:
            self.threshold_label.setText("Manual Threshold:")
            self.threshold_status_label.setText(
                f"Effective Threshold: {manual_threshold:.1f} dB (manual)"
            )

    def refresh_vad_status(self) -> None:
        """Refresh VAD-dependent UI after backend availability changes."""
        mode = self.gate_mode_combo.currentIndex()
        vad_available = self._is_vad_available()
        self._update_vad_controls_enabled()
        self._set_vad_status_text(mode, vad_available)

    def _update_vad_mode(self):
        """Update VAD mode and settings."""
        try:
            mode = self.gate_mode_combo.currentIndex()
            vad_available = self._is_vad_available()

            # Avoid "fake" VAD modes when model/runtime isn't available.
            if (
                mode > 0
                and not vad_available
                and not self._preserve_unavailable_vad_mode
            ):
                self.gate_mode_combo.blockSignals(True)
                self.gate_mode_combo.setCurrentIndex(0)
                self.gate_mode_combo.blockSignals(False)
                mode = 0

            self.processor.set_gate_mode(mode)
            self.processor.set_vad_threshold(self.vad_threshold_spinbox.value())
            self.processor.set_vad_hold_time(self.vad_hold_spinbox.value())
            self.processor.set_vad_pre_gain(self.vad_pre_gain_spinbox.value())
            self.refresh_vad_status()
        except AttributeError:
            # VAD not available - show shorter error message
            self.vad_info_label.setText("VAD: Not available")
        except Exception as e:
            # Truncate long error messages to prevent layout issues
            error_msg = str(e)
            if len(error_msg) > 40:
                error_msg = error_msg[:37] + "..."
            self.vad_info_label.setText(f"VAD: {error_msg}")

    def _update_vad_controls_enabled(self):
        """Enable/disable VAD controls based on gate mode and auto-threshold state."""
        mode = self.gate_mode_combo.currentIndex()
        vad_available = self._is_vad_available()
        # 0 = Threshold Only, 1 = VAD Assisted, 2 = VAD Only
        vad_enabled = mode > 0 and vad_available
        threshold_enabled = mode != 2  # Disabled in VAD Only mode
        auto_threshold_enabled = (
            vad_enabled and self.auto_threshold_checkbox.isChecked()
        )

        # Enable/disable VAD controls
        self.vad_threshold_slider.setEnabled(vad_enabled)
        self.vad_threshold_spinbox.setEnabled(vad_enabled)
        # Hold time remains active in auto-threshold mode (prevents gate chatter)
        self.vad_hold_spinbox.setEnabled(vad_enabled)
        # Pre-gain remains active (boosts signal for VAD detection)
        self.vad_pre_gain_slider.setEnabled(vad_enabled)
        self.vad_pre_gain_spinbox.setEnabled(vad_enabled)
        self.confidence_meter.setEnabled(vad_enabled)

        # Enable/disable level threshold
        self.threshold_slider.setEnabled(
            threshold_enabled and not auto_threshold_enabled
        )
        self.threshold_spinbox.setEnabled(
            threshold_enabled and not auto_threshold_enabled
        )

        # Enable/disable auto-threshold controls (only when VAD is active)
        self.auto_threshold_checkbox.setEnabled(mode > 0 and vad_available)
        self.margin_slider.setEnabled(auto_threshold_enabled)
        self.margin_spinbox.setEnabled(auto_threshold_enabled)
        self._refresh_threshold_summary()

    def _on_margin_slider(self, value):
        """Handle margin slider change."""
        self.margin_spinbox.blockSignals(True)
        self.margin_spinbox.setValue(float(value))
        self.margin_spinbox.blockSignals(False)
        self._update_auto_threshold()

    def _on_margin_spinbox(self, value):
        """Handle margin spinbox change."""
        self.margin_slider.blockSignals(True)
        self.margin_slider.setValue(int(value))
        self.margin_slider.blockSignals(False)
        self._update_auto_threshold()

    def _update_auto_threshold(self):
        """Update auto-threshold configuration."""
        # Always update UI enable/disable states first (before PyO3 calls)
        self._update_vad_controls_enabled()

        try:
            enabled = self.auto_threshold_checkbox.isChecked()
            margin = self.margin_spinbox.value()
            self.processor.set_auto_threshold(enabled)
            self.processor.set_gate_margin(margin)
            self._refresh_threshold_summary()
        except AttributeError:
            logger.debug("Auto-threshold controls unavailable", exc_info=True)

    def update_vad_confidence(self, confidence: float):
        """Update VAD confidence meter (called from main window)."""
        self.confidence_meter.set_confidence(confidence)
        try:
            self._latest_noise_floor_db = float(self.processor.get_noise_floor())
        except Exception:
            self._latest_noise_floor_db = -60.0
        self.refresh_vad_status()

    def get_settings(self) -> dict:
        """Get current gate settings as a dictionary."""
        settings = {
            "enabled": self.enabled_checkbox.isChecked(),
            "threshold_db": self.threshold_spinbox.value(),
            "attack_ms": self.attack_spinbox.value(),
            "release_ms": self.release_spinbox.value(),
            "gate_mode": self.gate_mode_combo.currentIndex(),
            "vad_threshold": self.vad_threshold_spinbox.value(),
            "vad_hold_time_ms": self.vad_hold_spinbox.value(),
            "vad_pre_gain": self.vad_pre_gain_spinbox.value(),
            "auto_threshold_enabled": self.auto_threshold_checkbox.isChecked(),
            "gate_margin_db": self.margin_spinbox.value(),
        }
        return settings

    def set_settings(self, settings: dict) -> None:
        """Apply settings from a dictionary with proper signal blocking."""
        if "enabled" in settings:
            self.enabled_checkbox.blockSignals(True)
            self.enabled_checkbox.setChecked(settings["enabled"])
            self.enabled_checkbox.blockSignals(False)
        if "threshold_db" in settings:
            self.threshold_spinbox.blockSignals(True)
            self.threshold_slider.blockSignals(True)
            self.threshold_spinbox.setValue(settings["threshold_db"])
            self.threshold_slider.setValue(int(settings["threshold_db"]))
            self.threshold_spinbox.blockSignals(False)
            self.threshold_slider.blockSignals(False)
        if "attack_ms" in settings:
            self.attack_spinbox.blockSignals(True)
            self.attack_spinbox.setValue(settings["attack_ms"])
            self.attack_spinbox.blockSignals(False)
        if "release_ms" in settings:
            self.release_spinbox.blockSignals(True)
            self.release_spinbox.setValue(settings["release_ms"])
            self.release_spinbox.blockSignals(False)

        # VAD mode settings (v1.2.0+)
        if "gate_mode" in settings:
            self.gate_mode_combo.blockSignals(True)
            self.gate_mode_combo.setCurrentIndex(settings["gate_mode"])
            self.gate_mode_combo.blockSignals(False)
        if "vad_threshold" in settings:
            self.vad_threshold_spinbox.blockSignals(True)
            self.vad_threshold_slider.blockSignals(True)
            self.vad_threshold_spinbox.setValue(settings["vad_threshold"])
            self.vad_threshold_slider.setValue(int(settings["vad_threshold"] * 100))
            self.vad_threshold_spinbox.blockSignals(False)
            self.vad_threshold_slider.blockSignals(False)
        if "vad_hold_time_ms" in settings:
            self.vad_hold_spinbox.blockSignals(True)
            self.vad_hold_spinbox.setValue(settings["vad_hold_time_ms"])
            self.vad_hold_spinbox.blockSignals(False)
        if "vad_pre_gain" in settings:
            self.vad_pre_gain_spinbox.blockSignals(True)
            self.vad_pre_gain_slider.blockSignals(True)
            self.vad_pre_gain_spinbox.setValue(settings["vad_pre_gain"])
            self.vad_pre_gain_slider.setValue(int(settings["vad_pre_gain"] * 10))
            self.vad_pre_gain_spinbox.blockSignals(False)
            self.vad_pre_gain_slider.blockSignals(False)

        # Auto-threshold settings (v1.2.1+)
        if "auto_threshold_enabled" in settings:
            self.auto_threshold_checkbox.blockSignals(True)
            self.auto_threshold_checkbox.setChecked(settings["auto_threshold_enabled"])
            self.auto_threshold_checkbox.blockSignals(False)
        if "gate_margin_db" in settings:
            self.margin_spinbox.blockSignals(True)
            self.margin_slider.blockSignals(True)
            self.margin_spinbox.setValue(settings["gate_margin_db"])
            self.margin_slider.setValue(int(settings["gate_margin_db"]))
            self.margin_spinbox.blockSignals(False)
            self.margin_slider.blockSignals(False)

        # Preserve stored VAD modes during preset restore even if the backend
        # has not reported availability yet. Interactive changes still fall back.
        self._preserve_unavailable_vad_mode = True
        try:
            self._update_gate()
            self._update_vad_mode()
            self._update_vad_controls_enabled()
            self._update_auto_threshold()
        finally:
            self._preserve_unavailable_vad_mode = False
