"""
Compressor and Limiter control panel

Controls for dynamics processing: threshold, ratio, attack, release, makeup gain.
"""

import logging
import math

from PySide6.QtWidgets import (
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
from PySide6.QtCore import Qt, Signal

from .level_meter import GainReductionMeter
from .rate_limiter import RateLimiter
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    SPACING_NORMAL,
    MARGIN_PANEL,
    PRIMARY_LABEL_STYLE,
    METER_LABEL_STYLE,
    INFO_LABEL_STYLE,
    bind_slider_spinbox,
    fit_spinbox_to_contents,
)


logger = logging.getLogger(__name__)


class CompressorPanel(QWidget):
    """Compressor and Limiter control panel."""

    configurationEdited = Signal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self._noise_reference_reliability = 0.0
        self._dynamics_profile = "balanced"
        self._dynamics_customized = False
        self._calibrated_controls: dict[str, float | bool] | None = None
        self._applying_settings = False
        self._synchronous_updates = False
        self._bulk_applying = False
        self._exact_values: dict[str, float] = {}
        self._exact_display_values: dict[str, float] = {}
        self._comp_rate_limiter = RateLimiter(interval_ms=33)
        self._limiter_rate_limiter = RateLimiter(interval_ms=33)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # === Compressor Group ===
        comp_group = QGroupBox("Compressor")
        comp_layout = QGridLayout(comp_group)
        comp_layout.setSpacing(SPACING_NORMAL)
        comp_layout.setContentsMargins(
            MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL
        )
        comp_layout.setColumnStretch(0, 0)  # Label column - fixed width
        comp_layout.setColumnStretch(1, 1)  # Control column - stretches
        comp_layout.setColumnMinimumWidth(0, 75)  # Ensure "Threshold:" fits

        # Row 0: Enable checkbox (span full width)
        self.comp_enabled_checkbox = QCheckBox("Enable Compressor")
        self.comp_enabled_checkbox.setChecked(True)
        self.comp_enabled_checkbox.setToolTip(
            "Reduces dynamic range by attenuating loud signals.\n"
            "Helps maintain consistent volume levels."
        )
        comp_layout.addWidget(self.comp_enabled_checkbox, 0, 0, 1, 2)

        # Row 1: Threshold (left) and Ratio (right) with PRIMARY_LABEL_STYLE
        # Threshold slider with spinbox
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(-60, 0)
        self.threshold_slider.setValue(-20)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-60.0, 0.0)
        self.threshold_spinbox.setSingleStep(1.0)
        self.threshold_spinbox.setValue(-20.0)
        self.threshold_spinbox.setSuffix(" dB")
        self.threshold_spinbox.setToolTip("Level above which compression begins")
        fit_spinbox_to_contents(self.threshold_spinbox)
        threshold_layout.addWidget(self.threshold_spinbox)

        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(threshold_label, 1, 0)
        comp_layout.addLayout(threshold_layout, 1, 1)

        # Ratio slider with spinbox
        ratio_layout = QHBoxLayout()
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(10, 200)  # 1.0:1 to 20.0:1
        self.ratio_slider.setValue(40)  # 4:1
        self.ratio_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ratio_slider.setTickInterval(20)
        ratio_layout.addWidget(self.ratio_slider)

        self.ratio_spinbox = QDoubleSpinBox()
        self.ratio_spinbox.setRange(1.0, 20.0)
        self.ratio_spinbox.setSingleStep(0.5)
        self.ratio_spinbox.setValue(4.0)
        self.ratio_spinbox.setSuffix(":1")
        self.ratio_spinbox.setToolTip("Compression ratio (higher = more compression)")
        fit_spinbox_to_contents(self.ratio_spinbox)
        ratio_layout.addWidget(self.ratio_spinbox)

        ratio_label = QLabel("Ratio:")
        ratio_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(ratio_label, 2, 0)
        comp_layout.addLayout(ratio_layout, 2, 1)

        # Row 2: Attack (left) and Release (right) with PRIMARY_LABEL_STYLE
        self.attack_spinbox = QDoubleSpinBox()
        self.attack_spinbox.setRange(0.1, 100.0)
        self.attack_spinbox.setSingleStep(1.0)
        self.attack_spinbox.setValue(10.0)
        self.attack_spinbox.setSuffix(" ms")
        self.attack_spinbox.setToolTip(
            "How fast the compressor responds to loud signals"
        )
        fit_spinbox_to_contents(self.attack_spinbox)

        attack_label = QLabel("Attack:")
        attack_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(attack_label, 3, 0)
        comp_layout.addWidget(self.attack_spinbox, 3, 1)

        self.release_spinbox = QDoubleSpinBox()
        self.release_spinbox.setRange(10.0, 1000.0)
        self.release_spinbox.setSingleStep(10.0)
        self.release_spinbox.setValue(200.0)
        self.release_spinbox.setSuffix(" ms")
        self.release_spinbox.setToolTip(
            "How fast the compressor recovers after loud signals"
        )
        fit_spinbox_to_contents(self.release_spinbox)

        release_label = QLabel("Release:")
        release_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(release_label, 4, 0)
        comp_layout.addWidget(self.release_spinbox, 4, 1)

        # Row 3: Makeup Gain (span full width) with PRIMARY_LABEL_STYLE
        makeup_layout = QHBoxLayout()
        self.makeup_slider = QSlider(Qt.Orientation.Horizontal)
        self.makeup_slider.setRange(0, 24)
        self.makeup_slider.setValue(0)
        self.makeup_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.makeup_slider.setTickInterval(6)
        makeup_layout.addWidget(self.makeup_slider)

        self.makeup_spinbox = QDoubleSpinBox()
        self.makeup_spinbox.setRange(0.0, 24.0)
        self.makeup_spinbox.setSingleStep(0.5)
        self.makeup_spinbox.setValue(0.0)
        self.makeup_spinbox.setSuffix(" dB")
        self.makeup_spinbox.setToolTip("Gain added after compression to restore volume")
        fit_spinbox_to_contents(self.makeup_spinbox)
        makeup_layout.addWidget(self.makeup_spinbox)

        makeup_label = QLabel("Makeup Gain:")
        makeup_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(makeup_label, 5, 0)
        comp_layout.addLayout(makeup_layout, 5, 1)

        # Row 4: Separator line
        separator = QLabel("")
        separator.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        comp_layout.addWidget(separator, 6, 0, 1, 2)

        # Row 5: Advanced Settings GroupBox
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(SPACING_NORMAL)
        advanced_layout.setContentsMargins(
            MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL
        )
        advanced_layout.setColumnStretch(0, 0)
        advanced_layout.setColumnStretch(1, 1)

        # Adaptive Release checkbox
        self.adaptive_release_checkbox = QCheckBox("Adaptive Release")
        self.adaptive_release_checkbox.setChecked(False)
        self.adaptive_release_checkbox.setToolTip(
            "Release time adapts based on signal dynamics.\n"
            "Scales from 50ms to 400ms based on sustained overage.\n"
            "Longer release for consistent loud signals, shorter for transients."
        )
        advanced_layout.addWidget(self.adaptive_release_checkbox, 0, 0, 1, 2)

        # Base release time (when adaptive is enabled)
        self.base_release_spinbox = QDoubleSpinBox()
        self.base_release_spinbox.setRange(20.0, 200.0)
        self.base_release_spinbox.setSingleStep(5.0)
        self.base_release_spinbox.setValue(50.0)
        self.base_release_spinbox.setSuffix(" ms")
        self.base_release_spinbox.setToolTip(
            "Base release time when adaptive mode is enabled"
        )
        self.base_release_spinbox.setEnabled(False)
        fit_spinbox_to_contents(self.base_release_spinbox)
        base_release_label = QLabel("Base Release:")
        advanced_layout.addWidget(base_release_label, 1, 0)
        advanced_layout.addWidget(self.base_release_spinbox, 1, 1)

        # Current release time display (with METER_LABEL_STYLE)
        self.current_release_label = QLabel("--")
        self.current_release_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_release_label.setToolTip(
            "Current release time (adaptive or manual)"
        )
        advanced_layout.addWidget(QLabel("Current Release:"), 2, 0)
        advanced_layout.addWidget(self.current_release_label, 2, 1)

        self.sidechain_highpass_checkbox = QCheckBox("Sidechain high-pass")
        self.sidechain_highpass_checkbox.setChecked(True)
        self.sidechain_highpass_checkbox.setToolTip(
            "Ignores low-frequency plosives and rumble in the compressor detector without filtering the audio."
        )
        advanced_layout.addWidget(self.sidechain_highpass_checkbox, 3, 0, 1, 2)

        # Auto Makeup Gain checkbox
        self.auto_makeup_checkbox = QCheckBox("Auto Makeup Gain")
        self.auto_makeup_checkbox.setChecked(False)
        self.auto_makeup_checkbox.setToolTip(
            "Automatically adjust makeup gain from post-compression EBU R128 loudness measurement.\n"
            "Maintains post-compressor output level relative to target LUFS.\n"
            "Uses the selected Target LUFS value."
        )
        advanced_layout.addWidget(self.auto_makeup_checkbox, 4, 0, 1, 2)

        # Target LUFS spinbox
        self.target_lufs_spinbox = QDoubleSpinBox()
        self.target_lufs_spinbox.setRange(-24.0, -12.0)
        self.target_lufs_spinbox.setSingleStep(1.0)
        self.target_lufs_spinbox.setValue(-18.0)
        self.target_lufs_spinbox.setSuffix(" LUFS")
        self.target_lufs_spinbox.setToolTip("Target loudness level (-24 to -12 LUFS)")
        self.target_lufs_spinbox.setEnabled(False)  # Disabled when auto makeup off
        fit_spinbox_to_contents(self.target_lufs_spinbox)
        target_lufs_label = QLabel("Target LUFS:")
        advanced_layout.addWidget(target_lufs_label, 5, 0)
        advanced_layout.addWidget(self.target_lufs_spinbox, 5, 1)

        # Current LUFS display (with METER_LABEL_STYLE)
        self.current_lufs_label = QLabel("--")
        self.current_lufs_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_lufs_label.setToolTip(
            "Current measured loudness (EBU R128 momentary)"
        )
        advanced_layout.addWidget(QLabel("Current LUFS:"), 6, 0)
        advanced_layout.addWidget(self.current_lufs_label, 6, 1)

        # Current makeup gain display (with METER_LABEL_STYLE)
        self.current_makeup_gain_label = QLabel("--")
        self.current_makeup_gain_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_makeup_gain_label.setToolTip("Current auto makeup gain applied")
        advanced_layout.addWidget(QLabel("Auto Gain:"), 7, 0)
        advanced_layout.addWidget(self.current_makeup_gain_label, 7, 1)

        comp_layout.addWidget(advanced_group, 7, 0, 1, 2)

        # Row 6: Gain Reduction Meter
        self.gr_meter = GainReductionMeter()
        comp_layout.addWidget(self.gr_meter, 8, 0, 1, 2)

        layout.addWidget(comp_group)

        # === Limiter Group ===
        limiter_group = QGroupBox("Hard Limiter")
        limiter_layout = QGridLayout(limiter_group)
        limiter_layout.setSpacing(SPACING_NORMAL)
        limiter_layout.setContentsMargins(
            MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL
        )
        limiter_layout.setColumnStretch(0, 0)  # Label column - fixed width
        limiter_layout.setColumnStretch(1, 1)  # Control column - stretches
        limiter_layout.setColumnMinimumWidth(0, 70)  # Ensure labels fit

        # Enable checkbox
        self.limiter_enabled_checkbox = QCheckBox("Enable Limiter")
        self.limiter_enabled_checkbox.setChecked(True)
        self.limiter_enabled_checkbox.setToolTip(
            "Prevents signal from exceeding ceiling level.\n"
            "Acts as a safety net to prevent clipping."
        )
        limiter_layout.addWidget(self.limiter_enabled_checkbox, 0, 0, 1, 3)

        self.careful_output_checkbox = QCheckBox("Careful output mode")
        self.careful_output_checkbox.setChecked(True)
        self.careful_output_checkbox.setToolTip(
            "Adds conservative output headroom by limiting the effective ceiling to -1.5 dB."
        )
        limiter_layout.addWidget(self.careful_output_checkbox, 1, 0, 1, 3)

        # Ceiling slider with spinbox
        ceiling_layout = QHBoxLayout()
        self.ceiling_slider = QSlider(Qt.Orientation.Horizontal)
        self.ceiling_slider.setRange(-120, 0)  # -12.0 to 0.0 dB (x10)
        self.ceiling_slider.setValue(-5)  # -0.5 dB
        self.ceiling_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ceiling_slider.setTickInterval(20)
        ceiling_layout.addWidget(self.ceiling_slider)

        self.ceiling_spinbox = QDoubleSpinBox()
        self.ceiling_spinbox.setRange(-12.0, 0.0)
        self.ceiling_spinbox.setSingleStep(0.1)
        self.ceiling_spinbox.setValue(-0.5)
        self.ceiling_spinbox.setSuffix(" dB")
        self.ceiling_spinbox.setToolTip("Maximum output level (brick-wall ceiling)")
        fit_spinbox_to_contents(self.ceiling_spinbox)
        ceiling_layout.addWidget(self.ceiling_spinbox)

        ceiling_label = QLabel("Ceiling:")
        limiter_layout.addWidget(ceiling_label, 2, 0)
        limiter_layout.addLayout(ceiling_layout, 2, 1, 1, 2)

        # Release time
        self.limiter_release_spinbox = QDoubleSpinBox()
        self.limiter_release_spinbox.setRange(10.0, 500.0)
        self.limiter_release_spinbox.setSingleStep(5.0)
        self.limiter_release_spinbox.setValue(50.0)
        self.limiter_release_spinbox.setSuffix(" ms")
        self.limiter_release_spinbox.setToolTip("How fast the limiter recovers")
        fit_spinbox_to_contents(self.limiter_release_spinbox)
        limiter_release_label = QLabel("Release:")
        limiter_layout.addWidget(limiter_release_label, 3, 0)
        limiter_layout.addWidget(self.limiter_release_spinbox, 3, 1, 1, 2)

        # Info label
        info_label = QLabel(
            "Limiter looks ahead 0.5 ms and ramps its gain down\n"
            "before transients reach the final output."
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        info_label.setWordWrap(True)
        limiter_layout.addWidget(info_label, 4, 0, 1, 3)

        layout.addWidget(limiter_group)

        bind_label(
            threshold_label,
            self.threshold_spinbox,
            name="Compressor threshold",
        )
        bind_label(ratio_label, self.ratio_spinbox, name="Compressor ratio")
        bind_label(attack_label, self.attack_spinbox, name="Compressor attack time")
        bind_label(
            release_label,
            self.release_spinbox,
            name="Compressor release time",
        )
        bind_label(makeup_label, self.makeup_spinbox, name="Compressor makeup gain")
        bind_label(
            base_release_label,
            self.base_release_spinbox,
            name="Adaptive compressor base release",
        )
        bind_label(
            target_lufs_label,
            self.target_lufs_spinbox,
            name="Automatic makeup target loudness",
        )
        bind_label(ceiling_label, self.ceiling_spinbox, name="Limiter ceiling")
        bind_label(
            limiter_release_label,
            self.limiter_release_spinbox,
            name="Limiter release time",
        )
        set_accessible_group(
            (
                (self.comp_enabled_checkbox, "Enable compressor", None),
                (self.threshold_slider, "Compressor threshold", None),
                (self.ratio_slider, "Compressor ratio", None),
                (self.makeup_slider, "Compressor makeup gain", None),
                (self.adaptive_release_checkbox, "Enable adaptive release", None),
                (
                    self.sidechain_highpass_checkbox,
                    "Enable compressor sidechain high-pass",
                    None,
                ),
                (self.auto_makeup_checkbox, "Enable automatic makeup gain", None),
                (self.gr_meter, "Compressor gain reduction", None),
                (self.limiter_enabled_checkbox, "Enable limiter", None),
                (self.careful_output_checkbox, "Enable careful output mode", None),
                (self.ceiling_slider, "Limiter ceiling", None),
            )
        )

    def _connect_signals(self):
        """Connect signals to slots."""
        # Compressor
        self.comp_enabled_checkbox.toggled.connect(self._update_compressor)
        bind_slider_spinbox(
            self.threshold_slider,
            self.threshold_spinbox,
            on_change=self._update_compressor,
        )
        bind_slider_spinbox(
            self.ratio_slider,
            self.ratio_spinbox,
            slider_to_value=lambda value: value / 10.0,
            value_to_slider=lambda value: int(value * 10),
            on_change=self._update_compressor,
        )
        self.attack_spinbox.valueChanged.connect(self._update_compressor)
        self.release_spinbox.valueChanged.connect(self._update_compressor)
        bind_slider_spinbox(
            self.makeup_slider,
            self.makeup_spinbox,
            on_change=self._update_compressor,
        )
        self.adaptive_release_checkbox.toggled.connect(self._update_adaptive_release)
        self.base_release_spinbox.valueChanged.connect(self._update_adaptive_release)
        self.sidechain_highpass_checkbox.toggled.connect(self._update_compressor)
        # Auto makeup gain
        self.auto_makeup_checkbox.toggled.connect(self._update_auto_makeup)
        self.target_lufs_spinbox.valueChanged.connect(self._update_auto_makeup)
        self.threshold_slider.sliderReleased.connect(self._comp_rate_limiter.flush)
        self.ratio_slider.sliderReleased.connect(self._comp_rate_limiter.flush)
        self.makeup_slider.sliderReleased.connect(self._comp_rate_limiter.flush)

        # Limiter
        self.limiter_enabled_checkbox.toggled.connect(self._update_limiter)
        self.careful_output_checkbox.toggled.connect(self._update_limiter)
        bind_slider_spinbox(
            self.ceiling_slider,
            self.ceiling_spinbox,
            slider_to_value=lambda value: value / 10.0,
            value_to_slider=lambda value: int(value * 10),
            on_change=self._update_limiter,
        )
        self.limiter_release_spinbox.valueChanged.connect(self._update_limiter)
        self.ceiling_slider.sliderReleased.connect(self._limiter_rate_limiter.flush)

        # Initial update
        self._update_compressor()
        self._update_limiter()

    def _precise_value(self, key: str, control: QDoubleSpinBox) -> float:
        value = float(control.value())
        if key not in self._exact_values or self._exact_display_values.get(key) != value:
            self._exact_values[key] = value
            self._exact_display_values[key] = value
        return self._exact_values[key]

    def _remember_exact_values(self, settings: dict) -> None:
        controls = {
            "threshold_db": self.threshold_spinbox,
            "ratio": self.ratio_spinbox,
            "attack_ms": self.attack_spinbox,
            "release_ms": self.release_spinbox,
            "makeup_gain_db": self.makeup_spinbox,
            "base_release_ms": self.base_release_spinbox,
            "target_lufs": self.target_lufs_spinbox,
            "ceiling_db": self.ceiling_spinbox,
            "limiter_release_ms": self.limiter_release_spinbox,
        }
        for key, control in controls.items():
            if key in settings:
                self._exact_values[key] = float(settings[key])
                self._exact_display_values[key] = float(control.value())

    def apply_processing_settings_synchronously(
        self, compressor: dict, limiter: dict
    ) -> None:
        """Apply both dynamics stages inline and discard superseded UI writes."""
        self._comp_rate_limiter.cancel()
        self._limiter_rate_limiter.cancel()
        self._synchronous_updates = True
        try:
            self.set_compressor_settings(compressor)
            self.set_limiter_settings(limiter)
        finally:
            self._synchronous_updates = False

    def _update_compressor(self):
        """Update compressor configuration."""
        if self._applying_settings:
            return
        self._mark_dynamics_customized()
        enabled = self.comp_enabled_checkbox.isChecked()
        threshold = self._precise_value("threshold_db", self.threshold_spinbox)
        ratio = self._precise_value("ratio", self.ratio_spinbox)
        attack = self._precise_value("attack_ms", self.attack_spinbox)
        adaptive = self.adaptive_release_checkbox.isChecked()
        release = (
            self._precise_value("base_release_ms", self.base_release_spinbox)
            if adaptive
            else self._precise_value("release_ms", self.release_spinbox)
        )
        makeup = self._precise_value("makeup_gain_db", self.makeup_spinbox)
        sidechain_highpass = self.sidechain_highpass_checkbox.isChecked()

        def apply():
            self.processor.set_compressor_enabled(enabled)
            self.processor.set_compressor_threshold(threshold)
            self.processor.set_compressor_ratio(ratio)
            self.processor.set_compressor_attack(attack)
            if adaptive:
                self.processor.set_compressor_base_release(release)
            else:
                self.processor.set_compressor_release(release)
            self.processor.set_compressor_makeup_gain(makeup)
            if hasattr(self.processor, "set_compressor_sidechain_highpass_enabled"):
                self.processor.set_compressor_sidechain_highpass_enabled(
                    sidechain_highpass
                )

        if self._synchronous_updates:
            self._comp_rate_limiter.call_now(apply)
        else:
            self._comp_rate_limiter.call(apply)
        if not self._applying_settings and not self._bulk_applying:
            self.configurationEdited.emit("Compressor edit")

    def _update_limiter(self):
        """Update limiter configuration."""
        if self._applying_settings:
            return
        enabled = self.limiter_enabled_checkbox.isChecked()
        careful_output = self.careful_output_checkbox.isChecked()
        ceiling = self._precise_value("ceiling_db", self.ceiling_spinbox)
        release = self._precise_value("limiter_release_ms", self.limiter_release_spinbox)

        def apply():
            self.processor.set_limiter_enabled(enabled)
            if hasattr(self.processor, "set_limiter_careful_output_enabled"):
                self.processor.set_limiter_careful_output_enabled(careful_output)
            self.processor.set_limiter_ceiling(ceiling)
            self.processor.set_limiter_release(release)

        if self._synchronous_updates:
            self._limiter_rate_limiter.call_now(apply)
        else:
            self._limiter_rate_limiter.call(apply)
        if not self._applying_settings and not self._bulk_applying:
            self.configurationEdited.emit("Limiter edit")

    def _dynamics_control_values(self) -> dict[str, float | bool]:
        return {
            "enabled": self.comp_enabled_checkbox.isChecked(),
            "threshold_db": self.threshold_spinbox.value(),
            "ratio": self.ratio_spinbox.value(),
            "attack_ms": self.attack_spinbox.value(),
            "release_ms": self.release_spinbox.value(),
            "makeup_gain_db": self.makeup_spinbox.value(),
            "adaptive_release": self.adaptive_release_checkbox.isChecked(),
            "base_release_ms": self.base_release_spinbox.value(),
            "auto_makeup_enabled": self.auto_makeup_checkbox.isChecked(),
            "sidechain_highpass_enabled": self.sidechain_highpass_checkbox.isChecked(),
        }

    def _mark_dynamics_customized(self) -> None:
        if self._applying_settings or self._calibrated_controls is None:
            return
        values = self._dynamics_control_values()
        if any(
            isinstance(value, bool) != isinstance(self._calibrated_controls.get(key), bool)
            or (
                isinstance(value, float)
                and abs(value - float(self._calibrated_controls.get(key, value))) > 1.0e-6
            )
            or value != self._calibrated_controls.get(key)
            for key, value in values.items()
        ):
            self._dynamics_customized = True

    def _update_adaptive_release(self):
        """Update adaptive release configuration."""
        if self._applying_settings:
            return
        self._mark_dynamics_customized()
        try:
            adaptive = self.adaptive_release_checkbox.isChecked()
            release = (
                self._precise_value("base_release_ms", self.base_release_spinbox)
                if adaptive
                else self._precise_value("release_ms", self.release_spinbox)
            )

            self._comp_rate_limiter.flush()
            self.processor.set_compressor_adaptive_release(adaptive)
            if adaptive:
                self.processor.set_compressor_base_release(release)
            else:
                self.processor.set_compressor_release(release)

            # When adaptive is enabled, disable manual release control
            self.release_spinbox.setEnabled(not adaptive)
            self.base_release_spinbox.setEnabled(adaptive)

            # Update current release display
            self._update_current_release()

        except Exception:
            if self._synchronous_updates:
                raise
            logger.debug("Adaptive release update failed", exc_info=True)
        if not self._applying_settings and not self._bulk_applying:
            self.configurationEdited.emit("Adaptive release edit")

    def _update_current_release(self):
        """Update current release time display from processor."""
        try:
            processor = self.processor
            processing_compressor = (
                processor.is_running()
                and processor.is_compressor_enabled()
                and not processor.is_bypass()
                and not processor.is_raw_monitor_enabled()
            )
            current_release = (
                processor.get_compressor_current_release()
                if processing_compressor
                else None
            )
            text = (f"{current_release:.0f} ms"
                    if current_release is not None and math.isfinite(current_release) else "--")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            text = "--"
        if self.current_release_label.text() != text:
            self.current_release_label.setText(text)

    def _update_auto_makeup(self):
        """Update auto makeup gain configuration."""
        if self._applying_settings:
            return
        self._mark_dynamics_customized()
        try:
            auto_makeup = self.auto_makeup_checkbox.isChecked()
            target_lufs = self._precise_value("target_lufs", self.target_lufs_spinbox)

            self.processor.set_compressor_auto_makeup_enabled(auto_makeup)
            self.processor.set_compressor_target_lufs(target_lufs)

            # Enable/disable target LUFS spinbox based on auto makeup state
            self.target_lufs_spinbox.setEnabled(auto_makeup)
            self.makeup_spinbox.setEnabled(not auto_makeup)
            self.makeup_slider.setEnabled(not auto_makeup)

        except Exception:
            if self._synchronous_updates:
                raise
            logger.debug("Auto makeup gain update failed", exc_info=True)
        if not self._applying_settings and not self._bulk_applying:
            self.configurationEdited.emit("Auto makeup edit")

    def update_auto_makeup_meters(self, current_lufs: float | None, makeup_gain: float | None):
        """Update auto makeup readings without substituting plausible defaults."""
        for label, value, unit in (
            (self.current_lufs_label, current_lufs, "LUFS"),
            (self.current_makeup_gain_label, makeup_gain, "dB"),
        ):
            text = f"{value:.1f} {unit}" if value is not None and math.isfinite(value) else "--"
            if label.text() != text:
                label.setText(text)

    def update_gain_reduction(self, gr_db: float | None):
        """Update the gain reduction meter (call from timer)."""
        self.gr_meter.set_gain_reduction(gr_db)

    def get_compressor_settings(self, *, include_calibration: bool = False) -> dict:
        """Get compressor settings, optionally including transient calibration."""
        settings = {
            "enabled": self.comp_enabled_checkbox.isChecked(),
            "threshold_db": self._precise_value("threshold_db", self.threshold_spinbox),
            "ratio": self._precise_value("ratio", self.ratio_spinbox),
            "attack_ms": self._precise_value("attack_ms", self.attack_spinbox),
            "release_ms": self._precise_value("release_ms", self.release_spinbox),
            "makeup_gain_db": self._precise_value("makeup_gain_db", self.makeup_spinbox),
            "adaptive_release": self.adaptive_release_checkbox.isChecked(),
            "base_release_ms": self._precise_value("base_release_ms", self.base_release_spinbox),
            "auto_makeup_enabled": self.auto_makeup_checkbox.isChecked(),
            "target_lufs": self._precise_value("target_lufs", self.target_lufs_spinbox),
            "sidechain_highpass_enabled": self.sidechain_highpass_checkbox.isChecked(),
        }
        if include_calibration:
            settings["noise_reference_reliability"] = self._noise_reference_reliability
            settings["dynamics_intensity"] = (
                "customized" if self._dynamics_customized else self._dynamics_profile
            )
            settings["dynamics_profile"] = self._dynamics_profile
            settings["dynamics_customized"] = self._dynamics_customized
        return settings

    def get_limiter_settings(self) -> dict:
        """Get current limiter settings as a dictionary."""
        return {
            "enabled": self.limiter_enabled_checkbox.isChecked(),
            "ceiling_db": self._precise_value("ceiling_db", self.ceiling_spinbox),
            "release_ms": self._precise_value("limiter_release_ms", self.limiter_release_spinbox),
            "careful_output_enabled": self.careful_output_checkbox.isChecked(),
        }

    def set_compressor_settings(self, settings: dict) -> None:
        """Apply compressor settings from a dictionary."""
        was_synchronous = self._synchronous_updates
        was_applying = self._applying_settings
        was_bulk_applying = self._bulk_applying
        self._comp_rate_limiter.cancel()
        self._synchronous_updates = True
        self._bulk_applying = True
        self._applying_settings = True
        try:
            if "enabled" in settings:
                self.comp_enabled_checkbox.setChecked(settings["enabled"])
            if "threshold_db" in settings:
                self.threshold_spinbox.setValue(settings["threshold_db"])
                self.threshold_slider.setValue(int(settings["threshold_db"]))
            if "ratio" in settings:
                self.ratio_spinbox.setValue(settings["ratio"])
                self.ratio_slider.setValue(int(settings["ratio"] * 10))
            if "attack_ms" in settings:
                self.attack_spinbox.setValue(settings["attack_ms"])
            if "release_ms" in settings:
                self.release_spinbox.setValue(settings["release_ms"])
            if "makeup_gain_db" in settings:
                self.makeup_spinbox.setValue(settings["makeup_gain_db"])
                self.makeup_slider.setValue(int(settings["makeup_gain_db"]))

            # Adaptive release settings (v1.2.0+)
            if "adaptive_release" in settings:
                self.adaptive_release_checkbox.setChecked(settings["adaptive_release"])
            if "base_release_ms" in settings:
                self.base_release_spinbox.setValue(settings["base_release_ms"])

            # Auto makeup gain settings (v1.3.0+)
            if "auto_makeup_enabled" in settings:
                self.auto_makeup_checkbox.setChecked(settings["auto_makeup_enabled"])
            if "target_lufs" in settings:
                self.target_lufs_spinbox.setValue(settings["target_lufs"])
            if "sidechain_highpass_enabled" in settings:
                self.sidechain_highpass_checkbox.setChecked(
                    settings["sidechain_highpass_enabled"]
                )

            self._remember_exact_values(settings)
        finally:
            self._applying_settings = was_applying

        try:
            self._update_compressor()
            self._update_adaptive_release()
            self._update_auto_makeup()
        finally:
            self._bulk_applying = was_bulk_applying
            self._synchronous_updates = was_synchronous

        profile = settings.get("dynamics_profile", settings.get("dynamics_intensity"))
        if profile in {"gentle", "balanced", "dense", "custom"}:
            self._dynamics_profile = str(profile)
            self._dynamics_customized = bool(settings.get("dynamics_customized", False))
            self._calibrated_controls = self._dynamics_control_values()
        elif any(key in settings for key in self._dynamics_control_values()):
            # Numeric preset files have no calibration provenance. A partial
            # control signature cannot distinguish a calibrated intensity from
            # a hand-edited threshold, so keep the loaded label honest.
            self._dynamics_profile = "custom"
            self._dynamics_customized = False
            self._calibrated_controls = self._dynamics_control_values()
        # Room-noise reliability is transient calibration evidence, not a
        # persisted compressor setting. Omission intentionally invalidates it
        # for ordinary preset application; rollback snapshots opt in above.
        if "noise_reference_reliability" not in settings:
            reliability = 0.0
        else:
            try:
                candidate_reliability = float(settings["noise_reference_reliability"])
            except (TypeError, ValueError):
                candidate_reliability = None
            reliability = (
                min(1.0, max(0.0, candidate_reliability))
                if candidate_reliability is not None
                and math.isfinite(candidate_reliability)
                else None
            )
        if reliability is not None:
            self._noise_reference_reliability = reliability
            setter = getattr(
                self.processor,
                "set_compressor_noise_reference_reliability",
                None,
            )
            if callable(setter):
                setter(reliability)

    def set_limiter_settings(self, settings: dict) -> None:
        """Apply limiter settings from a dictionary."""
        was_synchronous = self._synchronous_updates
        was_applying = self._applying_settings
        was_bulk_applying = self._bulk_applying
        self._limiter_rate_limiter.cancel()
        self._synchronous_updates = True
        self._bulk_applying = True
        self._applying_settings = True
        try:
            if "enabled" in settings:
                self.limiter_enabled_checkbox.setChecked(settings["enabled"])
            if "careful_output_enabled" in settings:
                self.careful_output_checkbox.setChecked(settings["careful_output_enabled"])
            if "ceiling_db" in settings:
                self.ceiling_spinbox.setValue(settings["ceiling_db"])
                self.ceiling_slider.setValue(int(settings["ceiling_db"] * 10))
                self._exact_values["ceiling_db"] = float(settings["ceiling_db"])
                self._exact_display_values["ceiling_db"] = float(self.ceiling_spinbox.value())
            if "release_ms" in settings:
                self.limiter_release_spinbox.setValue(settings["release_ms"])
                self._exact_values["limiter_release_ms"] = float(settings["release_ms"])
                self._exact_display_values["limiter_release_ms"] = float(
                    self.limiter_release_spinbox.value()
                )
        finally:
            self._applying_settings = was_applying
        try:
            self._update_limiter()
        finally:
            self._bulk_applying = was_bulk_applying
            self._synchronous_updates = was_synchronous
