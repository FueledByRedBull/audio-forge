"""
De-esser control panel.

Controls sibilance reduction stage placed between noise suppression and EQ.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .level_meter import GainReductionMeter
from .layout_constants import INFO_LABEL_STYLE, MARGIN_PANEL, PRIMARY_LABEL_STYLE, SPACING_NORMAL
from .rate_limiter import RateLimiter


class DeEsserPanel(QWidget):
    """De-esser parameter control panel."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("De-Esser")
        grid = QGridLayout(group)
        grid.setSpacing(SPACING_NORMAL)
        grid.setContentsMargins(MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, 95)

        self.enabled_checkbox = QCheckBox("Enable De-Esser")
        self.enabled_checkbox.setChecked(False)
        self.enabled_checkbox.setToolTip(
            "Reduces harsh sibilance (s, sh, t) using dynamic attenuation."
        )
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(self.enabled_checkbox, 0, 1, 1, 2)

        self.auto_checkbox = QCheckBox("Auto (Smart)")
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.setToolTip(
            "Learns average sibilance and applies dynamic reduction automatically."
        )
        grid.addWidget(QLabel(""), 1, 0)
        grid.addWidget(self.auto_checkbox, 1, 1, 1, 2)

        amount_layout = QHBoxLayout()
        self.auto_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.auto_amount_slider.setRange(0, 100)
        self.auto_amount_slider.setValue(50)
        self.auto_amount_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.auto_amount_slider.setTickInterval(10)
        amount_layout.addWidget(self.auto_amount_slider)

        self.auto_amount_spinbox = QDoubleSpinBox()
        self.auto_amount_spinbox.setRange(0.0, 1.0)
        self.auto_amount_spinbox.setSingleStep(0.05)
        self.auto_amount_spinbox.setValue(0.5)
        self.auto_amount_spinbox.setDecimals(2)
        self.auto_amount_spinbox.setFixedWidth(80)
        amount_layout.addWidget(self.auto_amount_spinbox)

        amount_label = QLabel("Amount:")
        amount_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(amount_label, 2, 0)
        grid.addLayout(amount_layout, 2, 1)

        low_layout = QHBoxLayout()
        self.low_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_cut_slider.setRange(2000, 12000)
        self.low_cut_slider.setValue(4000)
        self.low_cut_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.low_cut_slider.setTickInterval(1000)
        low_layout.addWidget(self.low_cut_slider)

        self.low_cut_spinbox = QDoubleSpinBox()
        self.low_cut_spinbox.setRange(2000.0, 12000.0)
        self.low_cut_spinbox.setSingleStep(100.0)
        self.low_cut_spinbox.setValue(4000.0)
        self.low_cut_spinbox.setSuffix(" Hz")
        self.low_cut_spinbox.setFixedWidth(100)
        low_layout.addWidget(self.low_cut_spinbox)

        low_label = QLabel("Low Cut:")
        low_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(low_label, 3, 0)
        grid.addLayout(low_layout, 3, 1)

        high_layout = QHBoxLayout()
        self.high_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_cut_slider.setRange(2200, 16000)
        self.high_cut_slider.setValue(9000)
        self.high_cut_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.high_cut_slider.setTickInterval(1000)
        high_layout.addWidget(self.high_cut_slider)

        self.high_cut_spinbox = QDoubleSpinBox()
        self.high_cut_spinbox.setRange(2200.0, 16000.0)
        self.high_cut_spinbox.setSingleStep(100.0)
        self.high_cut_spinbox.setValue(9000.0)
        self.high_cut_spinbox.setSuffix(" Hz")
        self.high_cut_spinbox.setFixedWidth(100)
        high_layout.addWidget(self.high_cut_spinbox)

        high_label = QLabel("High Cut:")
        high_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(high_label, 4, 0)
        grid.addLayout(high_layout, 4, 1)

        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(-60, -6)
        self.threshold_slider.setValue(-28)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(6)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-60.0, -6.0)
        self.threshold_spinbox.setSingleStep(1.0)
        self.threshold_spinbox.setValue(-28.0)
        self.threshold_spinbox.setSuffix(" dB")
        self.threshold_spinbox.setFixedWidth(80)
        threshold_layout.addWidget(self.threshold_spinbox)

        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(threshold_label, 5, 0)
        grid.addLayout(threshold_layout, 5, 1)

        ratio_layout = QHBoxLayout()
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(10, 200)
        self.ratio_slider.setValue(40)
        self.ratio_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ratio_slider.setTickInterval(20)
        ratio_layout.addWidget(self.ratio_slider)

        self.ratio_spinbox = QDoubleSpinBox()
        self.ratio_spinbox.setRange(1.0, 20.0)
        self.ratio_spinbox.setSingleStep(0.5)
        self.ratio_spinbox.setValue(4.0)
        self.ratio_spinbox.setSuffix(":1")
        self.ratio_spinbox.setFixedWidth(80)
        ratio_layout.addWidget(self.ratio_spinbox)

        ratio_label = QLabel("Ratio:")
        ratio_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(ratio_label, 6, 0)
        grid.addLayout(ratio_layout, 6, 1)

        self.attack_spinbox = QDoubleSpinBox()
        self.attack_spinbox.setRange(0.1, 50.0)
        self.attack_spinbox.setSingleStep(0.1)
        self.attack_spinbox.setValue(2.0)
        self.attack_spinbox.setSuffix(" ms")
        self.attack_spinbox.setMaximumWidth(100)
        self.attack_spinbox.setMinimumWidth(70)
        attack_label = QLabel("Attack:")
        attack_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(attack_label, 7, 0)
        grid.addWidget(self.attack_spinbox, 7, 1)

        self.release_spinbox = QDoubleSpinBox()
        self.release_spinbox.setRange(5.0, 500.0)
        self.release_spinbox.setSingleStep(5.0)
        self.release_spinbox.setValue(80.0)
        self.release_spinbox.setSuffix(" ms")
        self.release_spinbox.setMaximumWidth(100)
        self.release_spinbox.setMinimumWidth(70)
        release_label = QLabel("Release:")
        release_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(release_label, 8, 0)
        grid.addWidget(self.release_spinbox, 8, 1)

        max_red_layout = QHBoxLayout()
        self.max_reduction_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_reduction_slider.setRange(0, 240)
        self.max_reduction_slider.setValue(60)
        self.max_reduction_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.max_reduction_slider.setTickInterval(20)
        max_red_layout.addWidget(self.max_reduction_slider)

        self.max_reduction_spinbox = QDoubleSpinBox()
        self.max_reduction_spinbox.setRange(0.0, 24.0)
        self.max_reduction_spinbox.setSingleStep(0.5)
        self.max_reduction_spinbox.setValue(6.0)
        self.max_reduction_spinbox.setSuffix(" dB")
        self.max_reduction_spinbox.setFixedWidth(80)
        max_red_layout.addWidget(self.max_reduction_spinbox)

        max_red_label = QLabel("Max Red:")
        max_red_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(max_red_label, 9, 0)
        grid.addLayout(max_red_layout, 9, 1)

        self.gr_meter = GainReductionMeter()
        grid.addWidget(self.gr_meter, 10, 0, 1, 2)

        info = QLabel(
            "Auto mode tracks sibilance-vs-voice balance and adjusts reduction dynamically."
        )
        info.setWordWrap(True)
        info.setStyleSheet(INFO_LABEL_STYLE)
        grid.addWidget(info, 11, 0, 1, 2)

        layout.addWidget(group)

    def _connect_signals(self):
        self.enabled_checkbox.toggled.connect(self._update_deesser)
        self.auto_checkbox.toggled.connect(self._on_auto_toggled)
        self.auto_amount_slider.valueChanged.connect(self._on_auto_amount_slider)
        self.auto_amount_spinbox.valueChanged.connect(self._on_auto_amount_spinbox)

        self.low_cut_slider.valueChanged.connect(self._on_low_cut_slider)
        self.low_cut_spinbox.valueChanged.connect(self._on_low_cut_spinbox)
        self.high_cut_slider.valueChanged.connect(self._on_high_cut_slider)
        self.high_cut_spinbox.valueChanged.connect(self._on_high_cut_spinbox)

        self.threshold_slider.valueChanged.connect(self._on_threshold_slider)
        self.threshold_spinbox.valueChanged.connect(self._on_threshold_spinbox)

        self.ratio_slider.valueChanged.connect(self._on_ratio_slider)
        self.ratio_spinbox.valueChanged.connect(self._on_ratio_spinbox)

        self.attack_spinbox.valueChanged.connect(self._update_deesser)
        self.release_spinbox.valueChanged.connect(self._update_deesser)

        self.max_reduction_slider.valueChanged.connect(self._on_max_reduction_slider)
        self.max_reduction_spinbox.valueChanged.connect(self._on_max_reduction_spinbox)

        self.auto_amount_slider.sliderReleased.connect(self._rate_limiter.flush)
        self.threshold_slider.sliderReleased.connect(self._rate_limiter.flush)
        self.ratio_slider.sliderReleased.connect(self._rate_limiter.flush)
        self.max_reduction_slider.sliderReleased.connect(self._rate_limiter.flush)

        self._update_auto_controls_enabled()
        self._update_deesser()

    def _update_auto_controls_enabled(self):
        auto_enabled = self.auto_checkbox.isChecked()
        self.threshold_slider.setEnabled(not auto_enabled)
        self.threshold_spinbox.setEnabled(not auto_enabled)
        self.ratio_slider.setEnabled(not auto_enabled)
        self.ratio_spinbox.setEnabled(not auto_enabled)

    def _on_auto_toggled(self, _checked):
        self._update_auto_controls_enabled()
        self._update_deesser()

    def _on_auto_amount_slider(self, value):
        amount = value / 100.0
        self.auto_amount_spinbox.blockSignals(True)
        self.auto_amount_spinbox.setValue(amount)
        self.auto_amount_spinbox.blockSignals(False)
        self._update_deesser()

    def _on_auto_amount_spinbox(self, value):
        self.auto_amount_slider.blockSignals(True)
        self.auto_amount_slider.setValue(int(value * 100))
        self.auto_amount_slider.blockSignals(False)
        self._update_deesser()

    def _enforce_band_gap(self, source: str):
        """Maintain minimum 200Hz gap between low/high detector cutoffs."""
        low = self.low_cut_spinbox.value()
        high = self.high_cut_spinbox.value()

        if high <= low + 200.0:
            if source == "low":
                high = min(16000.0, low + 200.0)
            else:
                low = max(2000.0, high - 200.0)

            self.low_cut_slider.blockSignals(True)
            self.low_cut_spinbox.blockSignals(True)
            self.high_cut_slider.blockSignals(True)
            self.high_cut_spinbox.blockSignals(True)

            self.low_cut_spinbox.setValue(low)
            self.low_cut_slider.setValue(int(low))
            self.high_cut_spinbox.setValue(high)
            self.high_cut_slider.setValue(int(high))

            self.low_cut_slider.blockSignals(False)
            self.low_cut_spinbox.blockSignals(False)
            self.high_cut_slider.blockSignals(False)
            self.high_cut_spinbox.blockSignals(False)

    def _on_low_cut_slider(self, value):
        self.low_cut_spinbox.blockSignals(True)
        self.low_cut_spinbox.setValue(float(value))
        self.low_cut_spinbox.blockSignals(False)
        self._enforce_band_gap("low")
        self._update_deesser()

    def _on_low_cut_spinbox(self, value):
        self.low_cut_slider.blockSignals(True)
        self.low_cut_slider.setValue(int(value))
        self.low_cut_slider.blockSignals(False)
        self._enforce_band_gap("low")
        self._update_deesser()

    def _on_high_cut_slider(self, value):
        self.high_cut_spinbox.blockSignals(True)
        self.high_cut_spinbox.setValue(float(value))
        self.high_cut_spinbox.blockSignals(False)
        self._enforce_band_gap("high")
        self._update_deesser()

    def _on_high_cut_spinbox(self, value):
        self.high_cut_slider.blockSignals(True)
        self.high_cut_slider.setValue(int(value))
        self.high_cut_slider.blockSignals(False)
        self._enforce_band_gap("high")
        self._update_deesser()

    def _on_threshold_slider(self, value):
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(float(value))
        self.threshold_spinbox.blockSignals(False)
        self._update_deesser()

    def _on_threshold_spinbox(self, value):
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(value))
        self.threshold_slider.blockSignals(False)
        self._update_deesser()

    def _on_ratio_slider(self, value):
        ratio = value / 10.0
        self.ratio_spinbox.blockSignals(True)
        self.ratio_spinbox.setValue(ratio)
        self.ratio_spinbox.blockSignals(False)
        self._update_deesser()

    def _on_ratio_spinbox(self, value):
        self.ratio_slider.blockSignals(True)
        self.ratio_slider.setValue(int(value * 10))
        self.ratio_slider.blockSignals(False)
        self._update_deesser()

    def _on_max_reduction_slider(self, value):
        db = value / 10.0
        self.max_reduction_spinbox.blockSignals(True)
        self.max_reduction_spinbox.setValue(db)
        self.max_reduction_spinbox.blockSignals(False)
        self._update_deesser()

    def _on_max_reduction_spinbox(self, value):
        self.max_reduction_slider.blockSignals(True)
        self.max_reduction_slider.setValue(int(value * 10))
        self.max_reduction_slider.blockSignals(False)
        self._update_deesser()

    def _update_deesser(self):
        enabled = self.enabled_checkbox.isChecked()
        auto_enabled = self.auto_checkbox.isChecked()
        auto_amount = self.auto_amount_spinbox.value()
        low_cut_hz = self.low_cut_spinbox.value()
        high_cut_hz = self.high_cut_spinbox.value()
        threshold_db = self.threshold_spinbox.value()
        ratio = self.ratio_spinbox.value()
        attack_ms = self.attack_spinbox.value()
        release_ms = self.release_spinbox.value()
        max_reduction_db = self.max_reduction_spinbox.value()

        def apply():
            self.processor.set_deesser_enabled(enabled)
            self.processor.set_deesser_auto_enabled(auto_enabled)
            self.processor.set_deesser_auto_amount(auto_amount)
            self.processor.set_deesser_low_cut_hz(low_cut_hz)
            self.processor.set_deesser_high_cut_hz(high_cut_hz)
            self.processor.set_deesser_threshold_db(threshold_db)
            self.processor.set_deesser_ratio(ratio)
            self.processor.set_deesser_attack_ms(attack_ms)
            self.processor.set_deesser_release_ms(release_ms)
            self.processor.set_deesser_max_reduction_db(max_reduction_db)

        try:
            self._rate_limiter.call(apply)
        except Exception as e:
            print(f"De-esser update error: {type(e).__name__}: {e}")

    def update_gain_reduction(self, gr_db: float):
        self.gr_meter.set_gain_reduction(gr_db)

    def get_settings(self) -> dict:
        return {
            'enabled': self.enabled_checkbox.isChecked(),
            'auto_enabled': self.auto_checkbox.isChecked(),
            'auto_amount': self.auto_amount_spinbox.value(),
            'low_cut_hz': self.low_cut_spinbox.value(),
            'high_cut_hz': self.high_cut_spinbox.value(),
            'threshold_db': self.threshold_spinbox.value(),
            'ratio': self.ratio_spinbox.value(),
            'attack_ms': self.attack_spinbox.value(),
            'release_ms': self.release_spinbox.value(),
            'max_reduction_db': self.max_reduction_spinbox.value(),
        }

    def set_settings(self, settings: dict):
        self.enabled_checkbox.setChecked(settings.get('enabled', False))
        auto_enabled = bool(settings.get('auto_enabled', True))

        low = float(settings.get('low_cut_hz', 4000.0))
        high = float(settings.get('high_cut_hz', 9000.0))
        auto_amount = float(settings.get('auto_amount', 0.5))
        threshold = float(settings.get('threshold_db', -28.0))
        ratio = float(settings.get('ratio', 4.0))
        attack = float(settings.get('attack_ms', 2.0))
        release = float(settings.get('release_ms', 80.0))
        max_red = float(settings.get('max_reduction_db', 6.0))

        self.low_cut_slider.blockSignals(True)
        self.low_cut_spinbox.blockSignals(True)
        self.high_cut_slider.blockSignals(True)
        self.high_cut_spinbox.blockSignals(True)
        self.auto_checkbox.blockSignals(True)
        self.auto_amount_slider.blockSignals(True)
        self.auto_amount_spinbox.blockSignals(True)
        self.threshold_slider.blockSignals(True)
        self.threshold_spinbox.blockSignals(True)
        self.ratio_slider.blockSignals(True)
        self.ratio_spinbox.blockSignals(True)
        self.max_reduction_slider.blockSignals(True)
        self.max_reduction_spinbox.blockSignals(True)

        self.auto_checkbox.setChecked(auto_enabled)
        self.auto_amount_spinbox.setValue(auto_amount)
        self.auto_amount_slider.setValue(int(auto_amount * 100))
        self.low_cut_spinbox.setValue(low)
        self.low_cut_slider.setValue(int(low))
        self.high_cut_spinbox.setValue(high)
        self.high_cut_slider.setValue(int(high))
        self.threshold_spinbox.setValue(threshold)
        self.threshold_slider.setValue(int(threshold))
        self.ratio_spinbox.setValue(ratio)
        self.ratio_slider.setValue(int(ratio * 10))
        self.attack_spinbox.setValue(attack)
        self.release_spinbox.setValue(release)
        self.max_reduction_spinbox.setValue(max_red)
        self.max_reduction_slider.setValue(int(max_red * 10))

        self.low_cut_slider.blockSignals(False)
        self.low_cut_spinbox.blockSignals(False)
        self.high_cut_slider.blockSignals(False)
        self.high_cut_spinbox.blockSignals(False)
        self.auto_checkbox.blockSignals(False)
        self.auto_amount_slider.blockSignals(False)
        self.auto_amount_spinbox.blockSignals(False)
        self.threshold_slider.blockSignals(False)
        self.threshold_spinbox.blockSignals(False)
        self.ratio_slider.blockSignals(False)
        self.ratio_spinbox.blockSignals(False)
        self.max_reduction_slider.blockSignals(False)
        self.max_reduction_spinbox.blockSignals(False)

        self._enforce_band_gap("low")
        self._update_auto_controls_enabled()
        self._update_deesser()
