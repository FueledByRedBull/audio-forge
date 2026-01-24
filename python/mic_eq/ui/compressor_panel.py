"""
Compressor and Limiter control panel

Controls for dynamics processing: threshold, ratio, attack, release, makeup gain.
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

from .level_meter import GainReductionMeter
from .rate_limiter import RateLimiter
from .layout_constants import (
    SPACING_NORMAL, MARGIN_PANEL,
    PRIMARY_LABEL_STYLE, METER_LABEL_STYLE, INFO_LABEL_STYLE
)


class CompressorPanel(QWidget):
    """Compressor and Limiter control panel."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
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
        comp_group.setMinimumWidth(300)
        comp_layout = QGridLayout(comp_group)
        comp_layout.setSpacing(SPACING_NORMAL)
        comp_layout.setContentsMargins(MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL)
        # Two-column layout: columns 0,1 for left; columns 2,3 for right
        comp_layout.setColumnStretch(0, 0)  # Label column - fixed width
        comp_layout.setColumnStretch(1, 1)  # Control column - stretches
        comp_layout.setColumnStretch(2, 0)  # Label column - fixed width
        comp_layout.setColumnStretch(3, 1)  # Control column - stretches
        comp_layout.setColumnMinimumWidth(0, 75)  # Ensure "Threshold:" fits
        comp_layout.setColumnMinimumWidth(2, 75)  # Ensure "Ratio:" fits

        # Row 0: Enable checkbox (span full width)
        self.comp_enabled_checkbox = QCheckBox("Enable Compressor")
        self.comp_enabled_checkbox.setChecked(True)
        self.comp_enabled_checkbox.setToolTip(
            "Reduces dynamic range by attenuating loud signals.\n"
            "Helps maintain consistent volume levels."
        )
        comp_layout.addWidget(self.comp_enabled_checkbox, 0, 0, 1, 4)

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
        self.threshold_spinbox.setFixedWidth(80)
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
        self.ratio_spinbox.setFixedWidth(80)
        ratio_layout.addWidget(self.ratio_spinbox)

        ratio_label = QLabel("Ratio:")
        ratio_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(ratio_label, 1, 2)
        comp_layout.addLayout(ratio_layout, 1, 3)

        # Row 2: Attack (left) and Release (right) with PRIMARY_LABEL_STYLE
        self.attack_spinbox = QDoubleSpinBox()
        self.attack_spinbox.setRange(0.1, 100.0)
        self.attack_spinbox.setSingleStep(1.0)
        self.attack_spinbox.setValue(10.0)
        self.attack_spinbox.setSuffix(" ms")
        self.attack_spinbox.setToolTip("How fast the compressor responds to loud signals")

        attack_label = QLabel("Attack:")
        attack_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(attack_label, 2, 0)
        comp_layout.addWidget(self.attack_spinbox, 2, 1)

        self.release_spinbox = QDoubleSpinBox()
        self.release_spinbox.setRange(10.0, 1000.0)
        self.release_spinbox.setSingleStep(10.0)
        self.release_spinbox.setValue(200.0)
        self.release_spinbox.setSuffix(" ms")
        self.release_spinbox.setToolTip("How fast the compressor recovers after loud signals")

        release_label = QLabel("Release:")
        release_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(release_label, 2, 2)
        comp_layout.addWidget(self.release_spinbox, 2, 3)

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
        self.makeup_spinbox.setFixedWidth(80)
        makeup_layout.addWidget(self.makeup_spinbox)

        makeup_label = QLabel("Makeup Gain:")
        makeup_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        comp_layout.addWidget(makeup_label, 3, 0)
        comp_layout.addLayout(makeup_layout, 3, 1, 1, 3)

        # Row 4: Separator line
        separator = QLabel("")
        separator.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        comp_layout.addWidget(separator, 4, 0, 1, 4)

        # Row 5: Advanced Settings GroupBox
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(SPACING_NORMAL)
        advanced_layout.setContentsMargins(MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL)
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
        self.base_release_spinbox.setToolTip("Base release time when adaptive mode is enabled")
        self.base_release_spinbox.setEnabled(False)
        advanced_layout.addWidget(QLabel("Base Release:"), 1, 0)
        advanced_layout.addWidget(self.base_release_spinbox, 1, 1)

        # Current release time display (with METER_LABEL_STYLE)
        self.current_release_label = QLabel("200 ms")
        self.current_release_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_release_label.setToolTip("Current release time (adaptive or manual)")
        advanced_layout.addWidget(QLabel("Current Release:"), 2, 0)
        advanced_layout.addWidget(self.current_release_label, 2, 1)

        # Auto Makeup Gain checkbox
        self.auto_makeup_checkbox = QCheckBox("Auto Makeup Gain")
        self.auto_makeup_checkbox.setChecked(False)
        self.auto_makeup_checkbox.setToolTip(
            "Automatically adjust makeup gain based on EBU R128 loudness measurement.\n"
            "Maintains consistent output level relative to target LUFS.\n"
            "Target: -18 LUFS (podcast/streaming standard)"
        )
        advanced_layout.addWidget(self.auto_makeup_checkbox, 3, 0, 1, 2)

        # Target LUFS spinbox
        self.target_lufs_spinbox = QDoubleSpinBox()
        self.target_lufs_spinbox.setRange(-24.0, -12.0)
        self.target_lufs_spinbox.setSingleStep(1.0)
        self.target_lufs_spinbox.setValue(-18.0)
        self.target_lufs_spinbox.setSuffix(" LUFS")
        self.target_lufs_spinbox.setToolTip("Target loudness level (-24 to -12 LUFS)")
        self.target_lufs_spinbox.setEnabled(False)  # Disabled when auto makeup off
        advanced_layout.addWidget(QLabel("Target LUFS:"), 4, 0)
        advanced_layout.addWidget(self.target_lufs_spinbox, 4, 1)

        # Current LUFS display (with METER_LABEL_STYLE)
        self.current_lufs_label = QLabel("-18.0 LUFS")
        self.current_lufs_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_lufs_label.setToolTip("Current measured loudness (EBU R128 momentary)")
        advanced_layout.addWidget(QLabel("Current LUFS:"), 5, 0)
        advanced_layout.addWidget(self.current_lufs_label, 5, 1)

        # Current makeup gain display (with METER_LABEL_STYLE)
        self.current_makeup_gain_label = QLabel("0.0 dB")
        self.current_makeup_gain_label.setStyleSheet(METER_LABEL_STYLE)
        self.current_makeup_gain_label.setToolTip("Current auto makeup gain applied")
        advanced_layout.addWidget(QLabel("Auto Gain:"), 6, 0)
        advanced_layout.addWidget(self.current_makeup_gain_label, 6, 1)

        comp_layout.addWidget(advanced_group, 5, 0, 1, 4)

        # Row 6: Gain Reduction Meter
        self.gr_meter = GainReductionMeter()
        comp_layout.addWidget(self.gr_meter, 6, 0, 1, 4)

        layout.addWidget(comp_group)

        # === Limiter Group ===
        limiter_group = QGroupBox("Hard Limiter")
        limiter_group.setMinimumWidth(300)
        limiter_layout = QGridLayout(limiter_group)
        limiter_layout.setSpacing(SPACING_NORMAL)
        limiter_layout.setContentsMargins(MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL)
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
        limiter_layout.addWidget(QLabel(""), 0, 0)
        limiter_layout.addWidget(self.limiter_enabled_checkbox, 0, 1, 1, 2)

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
        self.ceiling_spinbox.setFixedWidth(80)
        ceiling_layout.addWidget(self.ceiling_spinbox)

        limiter_layout.addWidget(QLabel("Ceiling:"), 1, 0)
        limiter_layout.addLayout(ceiling_layout, 1, 1, 1, 2)

        # Release time
        self.limiter_release_spinbox = QDoubleSpinBox()
        self.limiter_release_spinbox.setRange(10.0, 500.0)
        self.limiter_release_spinbox.setSingleStep(5.0)
        self.limiter_release_spinbox.setValue(50.0)
        self.limiter_release_spinbox.setSuffix(" ms")
        self.limiter_release_spinbox.setToolTip("How fast the limiter recovers")
        limiter_layout.addWidget(QLabel("Release:"), 2, 0)
        limiter_layout.addWidget(self.limiter_release_spinbox, 2, 1, 1, 2)

        # Info label
        info_label = QLabel(
            "Limiter uses instant attack (~0.1ms)\n"
            "to catch transients. No lookahead."
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        limiter_layout.addWidget(QLabel(""), 3, 0)
        limiter_layout.addWidget(info_label, 3, 1, 1, 2)

        layout.addWidget(limiter_group)

    def _connect_signals(self):
        """Connect signals to slots."""
        # Compressor
        self.comp_enabled_checkbox.toggled.connect(self._update_compressor)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider)
        self.threshold_spinbox.valueChanged.connect(self._on_threshold_spinbox)
        self.ratio_slider.valueChanged.connect(self._on_ratio_slider)
        self.ratio_spinbox.valueChanged.connect(self._on_ratio_spinbox)
        self.attack_spinbox.valueChanged.connect(self._update_compressor)
        self.release_spinbox.valueChanged.connect(self._update_compressor)
        self.makeup_slider.valueChanged.connect(self._on_makeup_slider)
        self.makeup_spinbox.valueChanged.connect(self._on_makeup_spinbox)
        self.adaptive_release_checkbox.toggled.connect(self._update_adaptive_release)
        self.base_release_spinbox.valueChanged.connect(self._update_adaptive_release)
        # Auto makeup gain
        self.auto_makeup_checkbox.toggled.connect(self._update_auto_makeup)
        self.target_lufs_spinbox.valueChanged.connect(self._update_auto_makeup)
        self.threshold_slider.sliderReleased.connect(self._comp_rate_limiter.flush)
        self.ratio_slider.sliderReleased.connect(self._comp_rate_limiter.flush)
        self.makeup_slider.sliderReleased.connect(self._comp_rate_limiter.flush)

        # Limiter
        self.limiter_enabled_checkbox.toggled.connect(self._update_limiter)
        self.ceiling_slider.valueChanged.connect(self._on_ceiling_slider)
        self.ceiling_spinbox.valueChanged.connect(self._on_ceiling_spinbox)
        self.limiter_release_spinbox.valueChanged.connect(self._update_limiter)
        self.ceiling_slider.sliderReleased.connect(self._limiter_rate_limiter.flush)

        # Initial update
        self._update_compressor()
        self._update_limiter()

    def _on_threshold_slider(self, value):
        """Handle threshold slider change."""
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(float(value))
        self.threshold_spinbox.blockSignals(False)
        self._update_compressor()

    def _on_threshold_spinbox(self, value):
        """Handle threshold spinbox change."""
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(value))
        self.threshold_slider.blockSignals(False)
        self._update_compressor()

    def _on_ratio_slider(self, value):
        """Handle ratio slider change."""
        ratio = value / 10.0
        self.ratio_spinbox.blockSignals(True)
        self.ratio_spinbox.setValue(ratio)
        self.ratio_spinbox.blockSignals(False)
        self._update_compressor()

    def _on_ratio_spinbox(self, value):
        """Handle ratio spinbox change."""
        self.ratio_slider.blockSignals(True)
        self.ratio_slider.setValue(int(value * 10))
        self.ratio_slider.blockSignals(False)
        self._update_compressor()

    def _on_makeup_slider(self, value):
        """Handle makeup gain slider change."""
        self.makeup_spinbox.blockSignals(True)
        self.makeup_spinbox.setValue(float(value))
        self.makeup_spinbox.blockSignals(False)
        self._update_compressor()

    def _on_makeup_spinbox(self, value):
        """Handle makeup gain spinbox change."""
        self.makeup_slider.blockSignals(True)
        self.makeup_slider.setValue(int(value))
        self.makeup_slider.blockSignals(False)
        self._update_compressor()

    def _on_ceiling_slider(self, value):
        """Handle ceiling slider change."""
        ceiling = value / 10.0
        self.ceiling_spinbox.blockSignals(True)
        self.ceiling_spinbox.setValue(ceiling)
        self.ceiling_spinbox.blockSignals(False)
        self._update_limiter()

    def _on_ceiling_spinbox(self, value):
        """Handle ceiling spinbox change."""
        self.ceiling_slider.blockSignals(True)
        self.ceiling_slider.setValue(int(value * 10))
        self.ceiling_slider.blockSignals(False)
        self._update_limiter()

    def _update_compressor(self):
        """Update compressor configuration."""
        enabled = self.comp_enabled_checkbox.isChecked()
        threshold = self.threshold_spinbox.value()
        ratio = self.ratio_spinbox.value()
        attack = self.attack_spinbox.value()
        release = self.release_spinbox.value()
        makeup = self.makeup_spinbox.value()

        def apply():
            self.processor.set_compressor_enabled(enabled)
            self.processor.set_compressor_threshold(threshold)
            self.processor.set_compressor_ratio(ratio)
            self.processor.set_compressor_attack(attack)
            self.processor.set_compressor_release(release)
            self.processor.set_compressor_makeup_gain(makeup)

        self._comp_rate_limiter.call(apply)

    def _update_limiter(self):
        """Update limiter configuration."""
        enabled = self.limiter_enabled_checkbox.isChecked()
        ceiling = self.ceiling_spinbox.value()
        release = self.limiter_release_spinbox.value()

        def apply():
            self.processor.set_limiter_enabled(enabled)
            self.processor.set_limiter_ceiling(ceiling)
            self.processor.set_limiter_release(release)

        self._limiter_rate_limiter.call(apply)

    def _update_adaptive_release(self):
        """Update adaptive release configuration."""
        try:
            adaptive = self.adaptive_release_checkbox.isChecked()
            base_release = self.base_release_spinbox.value()

            self.processor.set_compressor_adaptive_release(adaptive)
            self.processor.set_compressor_base_release(base_release)

            # When adaptive is enabled, disable manual release control
            self.release_spinbox.setEnabled(not adaptive)
            self.base_release_spinbox.setEnabled(adaptive)

            # Update current release display
            self._update_current_release()

        except Exception as e:
            print(f"Adaptive release error: {e}")

    def _update_current_release(self):
        """Update current release time display from processor."""
        try:
            current_release = self.processor.get_compressor_current_release()
            self.current_release_label.setText(f"{current_release:.0f} ms")
        except Exception as e:
            print(f"Current release read error: {e}")

    def _update_auto_makeup(self):
        """Update auto makeup gain configuration."""
        try:
            auto_makeup = self.auto_makeup_checkbox.isChecked()
            target_lufs = self.target_lufs_spinbox.value()

            self.processor.set_compressor_auto_makeup_enabled(auto_makeup)
            self.processor.set_compressor_target_lufs(target_lufs)

            # Enable/disable target LUFS spinbox based on auto makeup state
            self.target_lufs_spinbox.setEnabled(auto_makeup)
            self.makeup_spinbox.setEnabled(not auto_makeup)
            self.makeup_slider.setEnabled(not auto_makeup)

        except Exception as e:
            print(f"Auto makeup gain error: {e}")

    def update_auto_makeup_meters(self, current_lufs: float, makeup_gain: float):
        """Update auto makeup gain metering displays (call from timer)."""
        self.current_lufs_label.setText(f"{current_lufs:.1f} LUFS")
        self.current_makeup_gain_label.setText(f"{makeup_gain:.1f} dB")

    def update_gain_reduction(self, gr_db: float):
        """Update the gain reduction meter (call from timer)."""
        self.gr_meter.set_gain_reduction(gr_db)

    def get_compressor_settings(self) -> dict:
        """Get current compressor settings as a dictionary."""
        return {
            'enabled': self.comp_enabled_checkbox.isChecked(),
            'threshold_db': self.threshold_spinbox.value(),
            'ratio': self.ratio_spinbox.value(),
            'attack_ms': self.attack_spinbox.value(),
            'release_ms': self.release_spinbox.value(),
            'makeup_gain_db': self.makeup_spinbox.value(),
            'adaptive_release': self.adaptive_release_checkbox.isChecked(),
            'base_release_ms': self.base_release_spinbox.value(),
            'auto_makeup_enabled': self.auto_makeup_checkbox.isChecked(),
            'target_lufs': self.target_lufs_spinbox.value(),
        }

    def get_limiter_settings(self) -> dict:
        """Get current limiter settings as a dictionary."""
        return {
            'enabled': self.limiter_enabled_checkbox.isChecked(),
            'ceiling_db': self.ceiling_spinbox.value(),
            'release_ms': self.limiter_release_spinbox.value(),
        }

    def set_compressor_settings(self, settings: dict) -> None:
        """Apply compressor settings from a dictionary."""
        if 'enabled' in settings:
            self.comp_enabled_checkbox.setChecked(settings['enabled'])
        if 'threshold_db' in settings:
            self.threshold_spinbox.setValue(settings['threshold_db'])
            self.threshold_slider.setValue(int(settings['threshold_db']))
        if 'ratio' in settings:
            self.ratio_spinbox.setValue(settings['ratio'])
            self.ratio_slider.setValue(int(settings['ratio'] * 10))
        if 'attack_ms' in settings:
            self.attack_spinbox.setValue(settings['attack_ms'])
        if 'release_ms' in settings:
            self.release_spinbox.setValue(settings['release_ms'])
        if 'makeup_gain_db' in settings:
            self.makeup_spinbox.setValue(settings['makeup_gain_db'])
            self.makeup_slider.setValue(int(settings['makeup_gain_db']))

        # Adaptive release settings (v1.2.0+)
        if 'adaptive_release' in settings:
            self.adaptive_release_checkbox.setChecked(settings['adaptive_release'])
        if 'base_release_ms' in settings:
            self.base_release_spinbox.setValue(settings['base_release_ms'])

        # Auto makeup gain settings (v1.3.0+)
        if 'auto_makeup_enabled' in settings:
            self.auto_makeup_checkbox.setChecked(settings['auto_makeup_enabled'])
        if 'target_lufs' in settings:
            self.target_lufs_spinbox.setValue(settings['target_lufs'])

        self._update_compressor()
        self._update_adaptive_release()
        self._update_auto_makeup()

    def set_limiter_settings(self, settings: dict) -> None:
        """Apply limiter settings from a dictionary."""
        if 'enabled' in settings:
            self.limiter_enabled_checkbox.setChecked(settings['enabled'])
        if 'ceiling_db' in settings:
            self.ceiling_spinbox.setValue(settings['ceiling_db'])
            self.ceiling_slider.setValue(int(settings['ceiling_db'] * 10))
        if 'release_ms' in settings:
            self.limiter_release_spinbox.setValue(settings['release_ms'])
        self._update_limiter()
