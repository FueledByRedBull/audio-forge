"""
Noise Gate control panel

Adapted from Spectral Workbench project.
"""

import logging

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QDoubleSpinBox,
    QSlider,
    QLabel,
    QHBoxLayout,
    QComboBox,
)
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from .gate_state import GateState
from .processing_meters import ProcessingMeters
from .components import Card, ToggleSwitch, form_layout
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    PRIMARY_LABEL_STYLE,
    INFO_LABEL_STYLE,
    fit_spinbox_to_contents,
)


logger = logging.getLogger(__name__)


class GatePanel(QWidget):
    """Noise Gate parameter control panel."""

    configurationEdited = Signal(str)

    def __init__(self, processor, gate_state: GateState | None = None, meters: ProcessingMeters | None = None):
        super().__init__()
        self.processor = processor
        self._meters = meters
        self.gate_state = gate_state or GateState(processor, self)
        self._setup_ui()
        self._connect_signals()
        if gate_state is None:
            self.set_settings({}, propagate_errors=False)

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.enabled_checkbox = ToggleSwitch()
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.setToolTip(
            "Reduces gain when signal falls below threshold.\n"
            "Helps eliminate background noise during silence."
        )
        card = Card(
            "Noise Gate",
            switch=self.enabled_checkbox,
            help_text=(
                "Reduces gain when the signal falls below the threshold, so "
                "background noise drops out during silence. The gate uses 3 dB "
                "of hysteresis and a smoothed envelope to avoid chattering."
            ),
        )
        gate_layout = form_layout()
        card.body.addLayout(gate_layout)
        advanced = QWidget()
        advanced_layout = form_layout(advanced)
        card.add_advanced(advanced)

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
        advanced_layout.addRow(attack_label, self.attack_spinbox)

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
        advanced_layout.addRow(release_label, self.release_spinbox)

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
        advanced_layout.addRow(mode_label, self.gate_mode_combo)

        # VAD threshold slider
        vad_threshold_layout = QHBoxLayout()
        self.vad_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.vad_threshold_slider.setRange(30, 70)  # Native range: 0.3 to 0.7
        self.vad_threshold_slider.setValue(48)
        self.vad_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.vad_threshold_slider.setTickInterval(10)
        vad_threshold_layout.addWidget(self.vad_threshold_slider)

        self.vad_threshold_spinbox = QDoubleSpinBox()
        self.vad_threshold_spinbox.setRange(0.3, 0.7)
        self.vad_threshold_spinbox.setSingleStep(0.01)
        self.vad_threshold_spinbox.setValue(0.48)
        self.vad_threshold_spinbox.setDecimals(2)
        self.vad_threshold_spinbox.setToolTip("Speech probability threshold (0.3-0.7)")
        fit_spinbox_to_contents(self.vad_threshold_spinbox)
        vad_threshold_layout.addWidget(self.vad_threshold_spinbox)

        vad_threshold_label = QLabel("VAD Threshold:")
        vad_threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_layout.addRow(vad_threshold_label, vad_threshold_layout)

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
        advanced_layout.addRow(hold_time_label, self.vad_hold_spinbox)

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
        advanced_layout.addRow(vad_pre_gain_label, vad_pre_gain_layout)

        # Auto Threshold section
        self.auto_threshold_checkbox = ToggleSwitch("Auto threshold")
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
        advanced_layout.addRow(margin_label, margin_layout)

        # Noise floor display (read-only)
        self.noise_floor_label = QLabel("Noise Floor: -60 dB")
        self.noise_floor_label.setStyleSheet(INFO_LABEL_STYLE)
        self.noise_floor_label.setWordWrap(True)
        advanced_layout.addRow(self.noise_floor_label)

        self.threshold_status_label = QLabel("Effective Threshold: Manual -40.0 dB")
        self.threshold_status_label.setStyleSheet(INFO_LABEL_STYLE)
        self.threshold_status_label.setWordWrap(True)
        gate_layout.addRow(self.threshold_status_label)

        # VAD confidence meter
        from .level_meter import ConfidenceMeter

        self.confidence_meter = self._meters.confidence if self._meters else ConfidenceMeter()
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
        advanced_layout.addRow(confidence_label, vad_meter_layout)

        layout.addWidget(card)

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
        self.enabled_checkbox.toggled.connect(
            lambda value: self.gate_state.set_value("enabled", value)
        )
        self.auto_threshold_checkbox.toggled.connect(
            lambda value: self.gate_state.set_value("auto_threshold_enabled", value)
        )
        self.gate_mode_combo.currentIndexChanged.connect(
            lambda value: self.gate_state.set_value("gate_mode", value)
        )
        self._numeric_controls: dict[
            str, tuple[QDoubleSpinBox, QSlider | None, float]
        ] = {
            "threshold_db": (self.threshold_spinbox, self.threshold_slider, 1),
            "attack_ms": (self.attack_spinbox, None, 1),
            "release_ms": (self.release_spinbox, None, 1),
            "vad_threshold": (self.vad_threshold_spinbox, self.vad_threshold_slider, 100),
            "vad_hold_time_ms": (self.vad_hold_spinbox, None, 1),
            "vad_pre_gain": (self.vad_pre_gain_spinbox, self.vad_pre_gain_slider, 10),
            "gate_margin_db": (self.margin_spinbox, self.margin_slider, 1),
        }
        for key, (spinbox, slider, scale) in self._numeric_controls.items():
            spinbox.valueChanged.connect(
                lambda value, key=key: self.gate_state.set_value(key, value)
            )
            if slider is not None:
                slider.valueChanged.connect(
                    lambda value, key=key, scale=scale: self.gate_state.set_value(
                        key, value / scale
                    )
                )
                slider.sliderReleased.connect(self.gate_state.flush)
        self.gate_state.changed.connect(self._render_settings)
        self.gate_state.configurationEdited.connect(self.configurationEdited.emit)
        self._render_settings()

    def _render_settings(self) -> None:
        settings = self.gate_state.get_settings()
        for control, key in (
            (self.enabled_checkbox, "enabled"),
            (self.auto_threshold_checkbox, "auto_threshold_enabled"),
        ):
            with QSignalBlocker(control):
                control.setChecked(bool(settings[key]))
            control.setEnabled(self.gate_state.control_enabled(key))
        with QSignalBlocker(self.gate_mode_combo):
            self.gate_mode_combo.setCurrentIndex(int(settings["gate_mode"]))
        for key, (spinbox, slider, scale) in self._numeric_controls.items():
            with QSignalBlocker(spinbox):
                spinbox.setValue(float(settings[key]))
            spinbox.setEnabled(self.gate_state.control_enabled(key))
            if slider is not None:
                with QSignalBlocker(slider):
                    slider.setValue(int(settings[key] * scale))
                slider.setEnabled(self.gate_state.control_enabled(key))
        self.confidence_meter.set_threshold(self.vad_threshold_spinbox.value())
        self.confidence_meter.set_confidence(self.gate_state.confidence)
        self.confidence_meter.setEnabled(self.gate_state.control_enabled("confidence"))
        text = self.gate_state.presentation()
        self.threshold_label.setText(text["threshold_label"])
        self.threshold_status_label.setText(text["threshold_status"])
        self.noise_floor_label.setText(text["noise_floor"])
        self.vad_info_label.setText(text["vad_info"])
        self.auto_threshold_checkbox.setText(text["auto_threshold_text"])
        self.auto_threshold_checkbox.setAccessibleName(text["auto_threshold_name"])
        self.auto_threshold_checkbox.setToolTip(text["auto_threshold_tooltip"])

    def refresh_vad_status(self) -> None:
        self.gate_state.refresh_vad_status()

    def update_vad_confidence(self, confidence: float | None):
        self.gate_state.update_vad_confidence(confidence)

    def get_settings(self) -> dict:
        return self.gate_state.get_settings()

    def apply_settings_synchronously(self, settings: dict) -> None:
        self.gate_state.set_settings(settings)

    def set_settings(self, settings: dict, *, propagate_errors: bool = True) -> None:
        try:
            self.gate_state.set_settings(settings)
        except RuntimeError:
            if propagate_errors:
                raise
            logger.debug("Initial gate setup failed", exc_info=True)
