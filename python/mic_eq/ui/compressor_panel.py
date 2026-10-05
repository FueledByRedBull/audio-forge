"""
Compressor and Limiter control panel

Controls for dynamics processing: threshold, ratio, attack, release, makeup gain.
"""

import math

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QDoubleSpinBox,
    QSlider,
    QLabel,
    QHBoxLayout,
)
from PySide6.QtCore import QSignalBlocker, Qt, Signal

from .components import Card, ToggleSwitch
from .compressor_state import CompressorState
from .level_meter import GainReductionMeter
from .limiter_state import LimiterState
from .processing_meters import ProcessingMeters
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    SPACING_NORMAL,
    MARGIN_PANEL,
    PRIMARY_LABEL_STYLE,
    METER_LABEL_STYLE,
    bind_slider_spinbox,
    fit_spinbox_to_contents,
)


class CompressorPanel(QWidget):
    """Compressor and Limiter control panel."""

    configurationEdited = Signal(str)

    def __init__(
        self, processor, limiter_state: LimiterState | None = None,
        compressor_state: CompressorState | None = None,
        meters: ProcessingMeters | None = None,
    ):
        super().__init__()
        self.processor = processor
        self._meters = meters
        self.compressor_state = compressor_state or CompressorState(processor, self)
        self.limiter_state = limiter_state or LimiterState(processor, self)
        self._setup_ui()
        self._connect_signals()
        if limiter_state is None:
            self.limiter_state.set_settings({})
        if compressor_state is None:
            self.compressor_state.set_settings({})

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(MARGIN_PANEL)

        # === Compressor Group ===
        self.comp_enabled_checkbox = ToggleSwitch()
        self.comp_enabled_checkbox.setChecked(True)
        self.comp_enabled_checkbox.setToolTip(
            "Reduces dynamic range by attenuating loud signals.\n"
            "Helps maintain consistent volume levels."
        )
        card = Card(
            "Compressor",
            switch=self.comp_enabled_checkbox,
            help_text=(
                "Evens out your level by turning down the loudest moments. "
                "Threshold sets where it starts working; ratio sets how hard."
            ),
        )
        comp_layout = QGridLayout()
        comp_layout.setSpacing(SPACING_NORMAL)
        comp_layout.setColumnStretch(1, 1)
        comp_layout.setColumnMinimumWidth(0, 75)
        card.body.addLayout(comp_layout)
        advanced_group = QWidget()
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(SPACING_NORMAL)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setColumnStretch(1, 1)
        advanced_layout.setColumnMinimumWidth(0, 75)

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
        advanced_layout.addWidget(attack_label, 0, 0)
        advanced_layout.addWidget(self.attack_spinbox, 0, 1)

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
        advanced_layout.addWidget(release_label, 1, 0)
        advanced_layout.addWidget(self.release_spinbox, 1, 1)

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
        advanced_layout.addWidget(makeup_label, 2, 0)
        advanced_layout.addLayout(makeup_layout, 2, 1)


        # Adaptive Release checkbox
        self.adaptive_release_checkbox = ToggleSwitch("Adaptive release")
        self.adaptive_release_checkbox.setChecked(False)
        self.adaptive_release_checkbox.setToolTip(
            "Release time adapts based on signal dynamics.\n"
            "Scales from 50ms to 400ms based on sustained overage.\n"
            "Longer release for consistent loud signals, shorter for transients."
        )
        advanced_layout.addWidget(self.adaptive_release_checkbox, 3, 0, 1, 2)

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
        advanced_layout.addWidget(base_release_label, 4, 0)
        advanced_layout.addWidget(self.base_release_spinbox, 4, 1)

        # Current release time display (with METER_LABEL_STYLE)
        self.current_release_label = self._meters.current_release if self._meters else QLabel("--")
        self.current_release_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_release_label.setToolTip(
            "Current release time (adaptive or manual)"
        )
        advanced_layout.addWidget(QLabel("Current Release:"), 5, 0)
        advanced_layout.addWidget(self.current_release_label, 5, 1)

        self.sidechain_highpass_checkbox = ToggleSwitch("Sidechain high-pass")
        self.sidechain_highpass_checkbox.setChecked(True)
        self.sidechain_highpass_checkbox.setToolTip(
            "Ignores low-frequency plosives and rumble in the compressor detector without filtering the audio."
        )
        advanced_layout.addWidget(self.sidechain_highpass_checkbox, 6, 0, 1, 2)

        # Auto Makeup Gain checkbox
        self.auto_makeup_checkbox = ToggleSwitch("Auto makeup gain")
        self.auto_makeup_checkbox.setChecked(False)
        self.auto_makeup_checkbox.setToolTip(
            "Automatically adjust makeup gain from post-compression EBU R128 loudness measurement.\n"
            "Maintains post-compressor output level relative to target LUFS.\n"
            "Uses the selected Target LUFS value."
        )
        advanced_layout.addWidget(self.auto_makeup_checkbox, 7, 0, 1, 2)

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
        advanced_layout.addWidget(target_lufs_label, 8, 0)
        advanced_layout.addWidget(self.target_lufs_spinbox, 8, 1)

        # Current LUFS display (with METER_LABEL_STYLE)
        self.current_lufs_label = self._meters.current_lufs if self._meters else QLabel("--")
        self.current_lufs_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_lufs_label.setToolTip(
            "Current measured loudness (EBU R128 momentary)"
        )
        advanced_layout.addWidget(QLabel("Current LUFS:"), 9, 0)
        advanced_layout.addWidget(self.current_lufs_label, 9, 1)

        # Current makeup gain display (with METER_LABEL_STYLE)
        self.current_makeup_gain_label = self._meters.current_makeup_gain if self._meters else QLabel("--")
        self.current_makeup_gain_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_makeup_gain_label.setToolTip("Current auto makeup gain applied")
        advanced_layout.addWidget(QLabel("Auto Gain:"), 10, 0)
        advanced_layout.addWidget(self.current_makeup_gain_label, 10, 1)

        self.gr_meter = self._meters.compressor_gr if self._meters else GainReductionMeter()
        comp_layout.addWidget(self.gr_meter, 8, 0, 1, 2)
        card.add_advanced(advanced_group)

        layout.addWidget(card)

        # === Limiter Group ===
        self.limiter_enabled_checkbox = ToggleSwitch()
        self.limiter_enabled_checkbox.setChecked(True)
        self.limiter_enabled_checkbox.setToolTip(
            "Prevents signal from exceeding ceiling level.\n"
            "Acts as a safety net to prevent clipping."
        )
        limiter_card = Card(
            "Limiter",
            switch=self.limiter_enabled_checkbox,
            help_text=(
                "A safety net that stops the output from going above the "
                "ceiling. It looks ahead 0.5 ms and ramps its gain down before "
                "transients reach the output."
            ),
        )
        limiter_layout = QGridLayout()
        limiter_card.body.addLayout(limiter_layout)
        limiter_advanced = QWidget()
        limiter_advanced_layout = QGridLayout(limiter_advanced)
        limiter_advanced_layout.setContentsMargins(0, 0, 0, 0)
        limiter_card.add_advanced(limiter_advanced)
        for section in (limiter_layout, limiter_advanced_layout):
            section.setSpacing(SPACING_NORMAL)
            section.setColumnStretch(1, 1)
            section.setColumnMinimumWidth(0, 75)

        self.careful_output_checkbox = ToggleSwitch("Careful output mode")
        self.careful_output_checkbox.setChecked(True)
        self.careful_output_checkbox.setToolTip(
            "Adds conservative output headroom by limiting the effective ceiling to -1.5 dB."
        )
        limiter_advanced_layout.addWidget(self.careful_output_checkbox, 1, 0, 1, 3)

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
        limiter_advanced_layout.addWidget(limiter_release_label, 3, 0)
        limiter_advanced_layout.addWidget(self.limiter_release_spinbox, 3, 1, 1, 2)

        layout.addWidget(limiter_card)

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
        self._boolean_controls = {
            "enabled": self.comp_enabled_checkbox,
            "adaptive_release": self.adaptive_release_checkbox,
            "sidechain_highpass_enabled": self.sidechain_highpass_checkbox,
            "auto_makeup_enabled": self.auto_makeup_checkbox,
        }
        self._numeric_controls: dict[
            str, tuple[QDoubleSpinBox, QSlider | None, float]
        ] = {
            "threshold_db": (self.threshold_spinbox, self.threshold_slider, 1),
            "ratio": (self.ratio_spinbox, self.ratio_slider, 10),
            "attack_ms": (self.attack_spinbox, None, 1),
            "release_ms": (self.release_spinbox, None, 1),
            "makeup_gain_db": (self.makeup_spinbox, self.makeup_slider, 1),
            "base_release_ms": (self.base_release_spinbox, None, 1),
            "target_lufs": (self.target_lufs_spinbox, None, 1),
        }
        for key, control in self._boolean_controls.items():
            control.toggled.connect(
                lambda value, key=key: self.compressor_state.set_value(key, value)
            )
        for key, (spinbox, slider, scale) in self._numeric_controls.items():
            spinbox.valueChanged.connect(
                lambda value, key=key: self.compressor_state.set_value(key, value)
            )
            if slider is not None:
                slider.valueChanged.connect(
                    lambda value, key=key, scale=scale: self.compressor_state.set_value(
                        key, value / scale
                    )
                )
                slider.sliderReleased.connect(self.compressor_state.flush)
        self.compressor_state.changed.connect(self._render_compressor_settings)
        self.compressor_state.configurationEdited.connect(self.configurationEdited.emit)
        self.compressor_state.configurationEdited.connect(self._update_current_release)

        # Limiter
        self.limiter_enabled_checkbox.toggled.connect(
            lambda value: self.limiter_state.set_value("enabled", value)
        )
        self.careful_output_checkbox.toggled.connect(
            lambda value: self.limiter_state.set_value("careful_output_enabled", value)
        )
        bind_slider_spinbox(
            self.ceiling_slider,
            self.ceiling_spinbox,
            slider_to_value=lambda value: value / 10.0,
            value_to_slider=lambda value: int(value * 10),
            on_change=lambda: self.limiter_state.set_value(
                "ceiling_db", self.ceiling_spinbox.value()
            ),
        )
        self.limiter_release_spinbox.valueChanged.connect(
            lambda value: self.limiter_state.set_value("release_ms", value)
        )
        self.ceiling_slider.sliderReleased.connect(self.limiter_state.flush)
        self.limiter_state.changed.connect(self._render_limiter_settings)
        self.limiter_state.configurationEdited.connect(self.configurationEdited.emit)

        self._render_limiter_settings()
        self._render_compressor_settings()

    def _render_compressor_settings(self) -> None:
        settings = self.compressor_state.get_settings()
        for key, control in self._boolean_controls.items():
            with QSignalBlocker(control):
                control.setChecked(bool(settings[key]))
        for key, (spinbox, slider, scale) in self._numeric_controls.items():
            with QSignalBlocker(spinbox):
                spinbox.setValue(float(settings[key]))
            spinbox.setEnabled(self.compressor_state.control_enabled(key))
            if slider is not None:
                with QSignalBlocker(slider):
                    slider.setValue(int(settings[key] * scale))
                slider.setEnabled(self.compressor_state.control_enabled(key))
        self._update_current_release()

    def apply_processing_settings_synchronously(
        self, compressor: dict, limiter: dict
    ) -> None:
        """Apply both dynamics stages inline and discard superseded UI writes."""
        self.compressor_state.cancel()
        self.limiter_state.cancel()
        self.compressor_state.set_settings(compressor)
        self.limiter_state.set_settings(limiter)

    def _render_limiter_settings(self) -> None:
        """Render authoritative values without turning display rounding into edits."""
        settings = self.limiter_state.get_settings()
        for control, value in (
            (self.limiter_enabled_checkbox, settings["enabled"]),
            (self.careful_output_checkbox, settings["careful_output_enabled"]),
        ):
            with QSignalBlocker(control):
                control.setChecked(bool(value))
        for control, value in (
            (self.ceiling_spinbox, settings["ceiling_db"]),
            (self.limiter_release_spinbox, settings["release_ms"]),
        ):
            with QSignalBlocker(control):
                control.setValue(float(value))
        with QSignalBlocker(self.ceiling_slider):
            self.ceiling_slider.setValue(int(settings["ceiling_db"] * 10))

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
        return self.compressor_state.get_settings(include_calibration=include_calibration)

    def get_limiter_settings(self) -> dict:
        return self.limiter_state.get_settings()

    def set_compressor_settings(self, settings: dict) -> None:
        self.compressor_state.set_settings(settings)

    def set_limiter_settings(self, settings: dict) -> None:
        self.limiter_state.set_settings(settings)
