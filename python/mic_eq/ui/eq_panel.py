"""
10-Band Parametric EQ control panel
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QCheckBox,
    QPushButton,
    QDoubleSpinBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from .eq_curve import EQCurveWidget
from .rate_limiter import RateLimiter


# Default frequencies for each band
BAND_FREQUENCIES = [
    "80",     # Low shelf
    "160",
    "320",
    "640",
    "1.2k",
    "2.5k",
    "5k",
    "8k",
    "12k",
    "16k",    # High shelf
]

# Numeric frequencies in Hz for curve calculation
BAND_FREQUENCIES_HZ = [80, 160, 320, 640, 1280, 2500, 5000, 8000, 12000, 16000]

BAND_LABELS = [
    "LS",   # Low shelf
    "160",
    "320",
    "640",
    "1.2k",
    "2.5k",
    "5k",
    "8k",
    "12k",
    "HS",   # High shelf
]


class EQBandSlider(QWidget):
    """Single EQ band with vertical slider."""

    def __init__(self, band_index: int, label: str, processor, curve_callback=None, parent=None):
        super().__init__(parent)
        self.band_index = band_index
        self.processor = processor
        self.curve_callback = curve_callback
        self._rate_limiter = RateLimiter(interval_ms=33)  # ~30Hz
        self._setup_ui(label)

    def _setup_ui(self, label: str):
        """Setup the band UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Set size policy to allow horizontal expansion
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Gain value label
        self.gain_label = QLabel("0")
        self.gain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gain_label.setMinimumWidth(30)
        self.gain_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.gain_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        layout.addWidget(self.gain_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Vertical slider (-12 to +12 dB)
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(-120, 120)  # Multiply by 10 for 0.1 dB precision
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.slider.setTickInterval(30)  # 3 dB ticks
        self.slider.setMinimumHeight(150)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignCenter)

        # Frequency label
        freq_label = QLabel(label)
        freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(freq_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Q factor spinbox
        q_layout = QHBoxLayout()
        q_layout.setContentsMargins(0, 0, 0, 0)
        q_layout.setSpacing(2)

        q_label = QLabel("Q:")
        q_label.setStyleSheet("font-size: 8px;")
        q_layout.addWidget(q_label)

        self.q_spinbox = QDoubleSpinBox()
        self.q_spinbox.setRange(0.1, 10.0)
        self.q_spinbox.setSingleStep(0.1)
        self.q_spinbox.setDecimals(1)
        self.q_spinbox.setValue(1.41)
        self.q_spinbox.setMinimumWidth(45)
        self.q_spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.q_spinbox.setStyleSheet("font-size: 8px;")
        self.q_spinbox.valueChanged.connect(self._on_q_changed)
        q_layout.addWidget(self.q_spinbox)

        layout.addLayout(q_layout)

    def _on_slider_changed(self, value):
        """Handle slider value change."""
        gain_db = value / 10.0
        self.gain_label.setText(f"{gain_db:+.1f}" if gain_db != 0 else "0")
        # Rate-limit the processor update
        self._rate_limiter.call(
            lambda g=gain_db: self._update_gain(g)
        )

    def _update_gain(self, gain_db):
        """Update processor and curve (rate-limited)."""
        self.processor.set_eq_band_gain(self.band_index, gain_db)
        if self.curve_callback:
            self.curve_callback()

    def _on_slider_released(self):
        """Ensure final value is applied when slider is released."""
        self._rate_limiter.flush()

    def _on_q_changed(self, value):
        """Handle Q spinbox value change."""
        # Rate-limit the processor update
        self._rate_limiter.call(
            lambda q=value: self._update_q(q)
        )

    def _update_q(self, q):
        """Update processor and curve (rate-limited)."""
        self.processor.set_eq_band_q(self.band_index, q)
        if self.curve_callback:
            self.curve_callback()

    def set_gain(self, gain_db: float):
        """Set gain value programmatically."""
        self.slider.blockSignals(True)
        self.slider.setValue(int(gain_db * 10))
        self.slider.blockSignals(False)
        self.gain_label.setText(f"{gain_db:+.1f}" if gain_db != 0 else "0")

    def set_q(self, q: float):
        """Set Q value programmatically."""
        self.q_spinbox.blockSignals(True)
        self.q_spinbox.setValue(q)
        self.q_spinbox.blockSignals(False)

    def reset(self):
        """Reset to 0 dB and default Q."""
        self.set_gain(0.0)
        self.set_q(1.41)
        self.processor.set_eq_band_gain(self.band_index, 0.0)
        self.processor.set_eq_band_q(self.band_index, 1.41)


class EQPanel(QWidget):
    """10-Band Parametric EQ control panel."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.band_sliders = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Allow panel to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # EQ Group
        eq_group = QGroupBox("10-Band Parametric EQ")
        eq_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        eq_layout = QVBoxLayout(eq_group)

        # Enable checkbox and reset button
        controls_layout = QHBoxLayout()

        self.enabled_checkbox = QCheckBox("Enable EQ")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        controls_layout.addWidget(self.enabled_checkbox)

        controls_layout.addStretch()

        reset_btn = QPushButton("Reset All")
        reset_btn.setToolTip("Reset all bands to 0 dB")
        reset_btn.clicked.connect(self._reset_all)
        controls_layout.addWidget(reset_btn)

        eq_layout.addLayout(controls_layout)

        # Frequency response curve (above sliders)
        self.curve_widget = EQCurveWidget()
        self.curve_widget.setFixedHeight(100)
        self.curve_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        eq_layout.addWidget(self.curve_widget)

        # dB scale labels
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("+12 dB"))
        scale_layout.addStretch()
        scale_layout.addWidget(QLabel("0 dB"))
        scale_layout.addStretch()
        scale_layout.addWidget(QLabel("-12 dB"))
        eq_layout.addLayout(scale_layout)

        # Band sliders
        sliders_layout = QHBoxLayout()
        sliders_layout.setSpacing(5)

        for i, label in enumerate(BAND_LABELS):
            band_slider = EQBandSlider(i, label, self.processor, curve_callback=self._update_curve)
            self.band_sliders.append(band_slider)
            sliders_layout.addWidget(band_slider, stretch=1)

        eq_layout.addLayout(sliders_layout, stretch=1)

        # Initial curve update
        self._update_curve()

        # Preset buttons
        presets_layout = QHBoxLayout()

        voice_btn = QPushButton("Voice")
        voice_btn.setToolTip("Preset for voice clarity")
        voice_btn.clicked.connect(self._preset_voice)
        presets_layout.addWidget(voice_btn)

        bass_btn = QPushButton("Bass Cut")
        bass_btn.setToolTip("Reduce low frequencies")
        bass_btn.clicked.connect(self._preset_bass_cut)
        presets_layout.addWidget(bass_btn)

        presence_btn = QPushButton("Presence")
        presence_btn.setToolTip("Boost voice presence frequencies")
        presence_btn.clicked.connect(self._preset_presence)
        presets_layout.addWidget(presence_btn)

        warm_clear_btn = QPushButton("Warm & Clear")
        warm_clear_btn.setToolTip("Bass boost with harshness cut (warm lows, clear mids)")
        warm_clear_btn.clicked.connect(self._preset_warm_clear)
        presets_layout.addWidget(warm_clear_btn)

        flat_btn = QPushButton("Flat")
        flat_btn.setToolTip("Reset to flat response")
        flat_btn.clicked.connect(self._reset_all)
        presets_layout.addWidget(flat_btn)

        eq_layout.addLayout(presets_layout)

        layout.addWidget(eq_group)

    def _on_enabled_toggled(self, checked):
        """Handle EQ enable/disable."""
        self.processor.set_eq_enabled(checked)

    def _reset_all(self):
        """Reset all bands to 0 dB."""
        for slider in self.band_sliders:
            slider.reset()
        self._update_curve()

    def _update_curve(self):
        """Update frequency response curve based on current band parameters."""
        bands = []
        for i, slider in enumerate(self.band_sliders):
            freq = BAND_FREQUENCIES_HZ[i]
            gain = slider.slider.value() / 10.0
            q = slider.q_spinbox.value()
            bands.append((freq, gain, q))
        self.curve_widget.set_all_params(bands)

    def _preset_voice(self):
        """Apply voice clarity preset."""
        # Cut low end, slight boost in presence, cut high end hiss
        gains = [-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 2.0, 0.0, -1.0, -2.0]
        qs = [0.7, 1.0, 1.2, 1.4, 1.6, 2.0, 1.8, 1.2, 0.9, 0.7]  # Wide cuts, focused boosts
        self._apply_preset(gains, qs)

    def _preset_bass_cut(self):
        """Apply bass cut preset (high-pass effect)."""
        gains = [-12.0, -6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        qs = [0.5, 0.7, 0.9, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]  # Wide rolloff
        self._apply_preset(gains, qs)

    def _preset_presence(self):
        """Apply presence boost preset."""
        gains = [0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0]
        qs = [1.41, 1.41, 1.41, 1.41, 2.0, 2.5, 2.0, 1.5, 1.41, 1.41]  # Narrow focus
        self._apply_preset(gains, qs)

    def _preset_warm_clear(self):
        """Apply warm & clear preset - bass boost with harshness cut."""
        # Refined mapping with blended midrange for nasal reduction
        gains = [-12.0, 4.0, 4.0, 3.0, -3.0, -10.0, 0.0, 0.0, 0.0, 0.0]
        qs = [0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707]
        self._apply_preset(gains, qs)

    def _apply_preset(self, gains: list, qs: list = None):
        """Apply a preset with given gain and Q values."""
        if qs is None:
            qs = [1.41] * len(gains)  # Default Q if not provided
        for i, gain in enumerate(gains):
            if i < len(self.band_sliders):
                self.band_sliders[i].set_gain(gain)
                self.processor.set_eq_band_gain(i, gain)
                if i < len(qs):
                    self.band_sliders[i].set_q(qs[i])
                    self.processor.set_eq_band_q(i, qs[i])
        # Update curve after applying preset
        self._update_curve()

    def get_settings(self) -> dict:
        """Get current EQ settings as a dictionary."""
        gains = []
        qs = []
        for slider in self.band_sliders:
            gains.append(slider.slider.value() / 10.0)
            qs.append(slider.q_spinbox.value())
        return {
            'enabled': self.enabled_checkbox.isChecked(),
            'band_gains': gains,
            'band_qs': qs,
        }

    def set_settings(self, settings: dict) -> None:
        """Apply settings from a dictionary."""
        if 'enabled' in settings:
            self.enabled_checkbox.setChecked(settings['enabled'])
        if 'band_gains' in settings:
            gains = settings['band_gains']
            # Default to 1.41 Q for backwards compatibility with old presets
            qs = settings.get('band_qs', [1.41] * len(gains))
            self._apply_preset(gains, qs)
