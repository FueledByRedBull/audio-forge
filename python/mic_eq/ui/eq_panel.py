"""10-band parametric EQ control panel."""

from dataclasses import replace
from typing import Any

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QCheckBox,
    QPushButton,
    QDoubleSpinBox,
    QComboBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from .auto_eq_explanation import explain_auto_eq
from .eq_curve import EQCurveWidget
from .rate_limiter import RateLimiter
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    PRIMARY_LABEL_STYLE,
    SPACING_TIGHT,
    fit_spinbox_to_contents,
    status_chip_style,
)
from .theme import COMPACT_CONTROL_STYLE
from ..config import (
    BUILTIN_PRESETS,
    EQBandSettings,
    EQSettings,
    EQ_FREQUENCIES,
    EQ_SCHEMA_VERSION,
    EQ_SLOPES_DB_PER_OCTAVE,
)


# Numeric frequencies in Hz for curve calculation (single source of truth from config)
BAND_FREQUENCIES_HZ = list(EQ_FREQUENCIES)

EQ_FILTER_OPTIONS = (
    ("Bell", "bell"),
    ("Notch", "notch"),
    ("Low shelf", "low_shelf"),
    ("High shelf", "high_shelf"),
    ("High-pass", "high_pass"),
    ("Low-pass", "low_pass"),
)
PASS_FILTER_TYPES = frozenset({"high_pass", "low_pass"})
GAIN_FILTER_TYPES = frozenset({"bell", "low_shelf", "high_shelf"})
Q_FILTER_TYPES = frozenset({"bell", "notch", "low_shelf", "high_shelf"})


def _default_filter_type(band_index: int) -> str:
    if band_index == 0:
        return "low_shelf"
    if band_index == 9:
        return "high_shelf"
    return "bell"


def _format_frequency_label(freq_hz: float) -> str:
    if freq_hz >= 1000.0:
        value = f"{freq_hz / 1000.0:.1f}".rstrip("0").rstrip(".")
        return f"{value}k"
    return f"{freq_hz:.0f}"


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "--"


def _db_value(value: Any) -> str:
    try:
        return f"{float(value):.1f} dB"
    except (TypeError, ValueError):
        return "--"


def _format_auto_eq_diagnostics(diagnostics: dict | None) -> tuple[str, str, str]:
    if not diagnostics:
        return "Auto-EQ: no calibration diagnostics", "idle", ""

    explanation = explain_auto_eq(diagnostics)
    confidence = float(diagnostics.get("analysis_confidence", 0.0) or 0.0)
    eq_confidence = float(diagnostics.get("eq_confidence", confidence) or 0.0)
    capture_confidence = float(diagnostics.get("capture_confidence", confidence) or 0.0)
    validation_confidence = float(diagnostics.get("validation_confidence", 0.0) or 0.0)
    before = diagnostics.get("validation_before_error_db")
    after = diagnostics.get("validation_after_error_db")
    scale = diagnostics.get("validation_gain_scale")
    used_fallback = bool(diagnostics.get("used_spectrum_fallback", False))
    low_confidence = diagnostics.get("low_confidence_active_bands", 0)
    headroom = diagnostics.get("headroom_validation") or {}
    headroom_after = headroom.get("after") if isinstance(headroom, dict) else None
    headroom_safe = (
        bool(headroom.get("safe", True)) if isinstance(headroom, dict) else True
    )
    headroom_advisory = (
        bool(headroom.get("advisory", False)) if isinstance(headroom, dict) else False
    )
    headroom_scale = headroom.get("gain_scale") if isinstance(headroom, dict) else None
    if not isinstance(low_confidence, int) or isinstance(low_confidence, bool):
        low_confidence = 0

    state = explanation.state
    if not headroom_safe:
        state = "warn" if state == "ok" else state

    text = (
        f"Auto-EQ: {explanation.summary} | "
        f"overall {_percent(confidence)} | "
        f"EQ {_percent(eq_confidence)} | "
        f"capture {_percent(capture_confidence)} | "
        f"validation {_percent(validation_confidence)} | "
        f"target error {_db_value(before)} -> {_db_value(after)} | "
        f"gain scale {_percent(scale)}"
    )
    if used_fallback:
        text += " | fallback spectrum"
    if low_confidence:
        text += f" | active low-confidence bands {low_confidence}"
    if isinstance(headroom_after, dict):
        pre_tp_headroom = headroom_after.get("pre_limiter_true_peak_headroom_db")
        limiter_gr = headroom_after.get("limiter_gain_reduction_db")
        true_peak_gr = headroom_after.get("true_peak_limiter_gain_reduction_db")
        headroom_status = (
            "advisory" if headroom_advisory else "safe" if headroom_safe else "risk"
        )
        text += " | headroom " + headroom_status + f" TP {_db_value(pre_tp_headroom)}"
        if headroom_scale is not None and float(headroom_scale) < 1.0:
            text += f" scale {_percent(headroom_scale)}"
        tooltip_status = (
            "advisory only (Rust simulator unavailable)"
            if headroom_advisory
            else "safe correction"
            if headroom_safe
            else "headroom risk"
        )
        tooltip_extra = (
            f"\nHeadroom status: {tooltip_status}"
            f"\nPre-limiter true-peak headroom: {_db_value(pre_tp_headroom)}"
            f"\nLimiter gain reduction: {_db_value(limiter_gr)}"
            f"\nTrue-peak limiter gain reduction: {_db_value(true_peak_gr)}"
            f"\nHeadroom gain scale: {_percent(headroom_scale)}"
        )
    else:
        tooltip_extra = ""

    explanation_details = "\n".join(f"- {detail}" for detail in explanation.details)
    tooltip = (
        "Auto-EQ calibration diagnostics\n"
        f"Result: {explanation.summary}\n"
        f"Reason code: {explanation.outcome_code}\n"
        f"{explanation_details}\n"
        f"Raw recommendation status: "
        f"{diagnostics.get('recommendation_status', '--')}\n"
        f"Overall confidence: {_percent(confidence)}\n"
        f"EQ confidence: {_percent(eq_confidence)}\n"
        f"Capture confidence: {_percent(capture_confidence)}\n"
        f"Validation confidence: {_percent(validation_confidence)}\n"
        f"Weighted target error before: {_db_value(before)}\n"
        f"Weighted target error after: {_db_value(after)}\n"
        f"Validation gain scale: {_percent(scale)}\n"
        f"Target profile: {diagnostics.get('target_profile', '--')}\n"
        f"Fallback analysis: {'yes' if used_fallback else 'no'}"
        f"{tooltip_extra}"
    )
    return text, state, tooltip


class EQBandSlider(QWidget):
    """Single EQ band with vertical slider."""

    def __init__(
        self,
        band_index: int,
        frequency_hz: float,
        processor,
        curve_callback=None,
        frequency_callback=None,
        parameter_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.band_index = band_index
        self.processor = processor
        self.curve_callback = curve_callback
        self.frequency_callback = frequency_callback
        self.parameter_callback = parameter_callback
        self.bandwidth_mode = "q"
        self.bandwidth_octaves: float | None = None
        self.stage = "combined"
        # Controls intentionally display coarse, useful increments. Keep exact
        # programmatic values so native DSP, graph rendering, and preset
        # round-trips agree until the user edits the corresponding control.
        self._frequency_hz = float(frequency_hz)
        self._gain_db = 0.0
        self._q = 1.41
        self._rate_limiter = RateLimiter(interval_ms=33)  # ~30Hz
        self._frequency_rate_limiter = RateLimiter(interval_ms=33)
        self._setup_ui(frequency_hz)

    def _setup_ui(self, frequency_hz: float):
        """Setup the band UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(SPACING_TIGHT)  # Use tight spacing for band sliders

        # Set size policy to allow horizontal expansion
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.band_enabled_checkbox = QCheckBox("On")
        self.band_enabled_checkbox.setChecked(True)
        self.band_enabled_checkbox.setToolTip("Enable this EQ band")
        self.band_enabled_checkbox.toggled.connect(self._on_band_enabled_changed)
        layout.addWidget(
            self.band_enabled_checkbox,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        self.filter_type_combo = QComboBox()
        for display_name, filter_type in EQ_FILTER_OPTIONS:
            self.filter_type_combo.addItem(display_name, filter_type)
        self.filter_type_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.filter_type_combo.setToolTip("Filter type")
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        layout.addWidget(
            self.filter_type_combo,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        # Gain value label
        self.gain_label = QLabel("0")
        self.gain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gain_label.setMinimumWidth(30)
        self.gain_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.gain_label.setStyleSheet(
            PRIMARY_LABEL_STYLE
        )  # Use consistent primary label style
        layout.addWidget(self.gain_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Vertical slider (-12 to +12 dB)
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(-120, 120)  # Multiply by 10 for 0.1 dB precision
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.slider.setTickInterval(30)  # 3 dB ticks
        self.slider.setMinimumHeight(120)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignCenter)

        # Frequency label
        self.freq_label = QLabel(_format_frequency_label(frequency_hz))
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_label.setStyleSheet(COMPACT_CONTROL_STYLE)
        self.freq_label.setToolTip(f"EQ band {self.band_index + 1} center frequency")
        layout.addWidget(self.freq_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Editable frequency control
        freq_layout = QHBoxLayout()
        freq_layout.setContentsMargins(0, 0, 0, 0)
        freq_layout.setSpacing(2)

        freq_label = QLabel("Hz:")
        freq_label.setStyleSheet(COMPACT_CONTROL_STYLE)
        freq_layout.addWidget(freq_label)

        self.frequency_spinbox = QDoubleSpinBox()
        self.frequency_spinbox.setRange(20.0, 20000.0)
        self.frequency_spinbox.setSingleStep(10.0)
        self.frequency_spinbox.setDecimals(0)
        self.frequency_spinbox.setValue(frequency_hz)
        self.frequency_spinbox.setStyleSheet(COMPACT_CONTROL_STYLE)
        fit_spinbox_to_contents(self.frequency_spinbox)
        self.frequency_spinbox.setToolTip("Center frequency in Hz")
        self.frequency_spinbox.valueChanged.connect(self._on_frequency_changed)
        freq_layout.addWidget(self.frequency_spinbox)

        layout.addLayout(freq_layout)

        # Q factor spinbox
        q_layout = QHBoxLayout()
        q_layout.setContentsMargins(0, 0, 0, 0)
        q_layout.setSpacing(2)

        self.q_label = QLabel("Q:")
        self.q_label.setStyleSheet(COMPACT_CONTROL_STYLE)
        q_layout.addWidget(self.q_label)

        self.q_spinbox = QDoubleSpinBox()
        self.q_spinbox.setRange(0.1, 10.0)
        self.q_spinbox.setSingleStep(0.1)
        self.q_spinbox.setDecimals(1)
        self.q_spinbox.setValue(1.41)
        self.q_spinbox.setStyleSheet(COMPACT_CONTROL_STYLE)
        fit_spinbox_to_contents(self.q_spinbox)
        self.q_spinbox.valueChanged.connect(self._on_q_changed)
        q_layout.addWidget(self.q_spinbox)

        layout.addLayout(q_layout)

        slope_layout = QHBoxLayout()
        slope_layout.setContentsMargins(0, 0, 0, 0)
        slope_layout.setSpacing(2)

        self.slope_label = QLabel("Slope:")
        self.slope_label.setStyleSheet(COMPACT_CONTROL_STYLE)
        slope_layout.addWidget(self.slope_label)

        self.slope_combo = QComboBox()
        for slope in sorted(EQ_SLOPES_DB_PER_OCTAVE):
            self.slope_combo.addItem(f"{slope}", slope)
        self.slope_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.slope_combo.setToolTip("Pass-filter slope in dB per octave")
        self.slope_combo.currentIndexChanged.connect(self._on_slope_changed)
        slope_layout.addWidget(self.slope_combo)

        layout.addLayout(slope_layout)
        band_name = f"EQ band {self.band_index + 1}"
        bind_label(
            freq_label,
            self.frequency_spinbox,
            name=f"{band_name} center frequency",
        )
        bind_label(
            self.q_label,
            self.q_spinbox,
            name=f"{band_name} Q",
        )
        bind_label(
            self.slope_label,
            self.slope_combo,
            name=f"{band_name} pass-filter slope",
        )
        set_accessible_group(
            (
                (self.band_enabled_checkbox, f"Enable {band_name}", None),
                (self.filter_type_combo, f"{band_name} filter type", None),
                (
                    self.slider,
                    f"{band_name} gain",
                    "Gain from minus 12 to plus 12 decibels.",
                ),
            )
        )
        self.set_filter_type(_default_filter_type(self.band_index))
        self._set_parameter_availability()

    def _on_slider_changed(self, value):
        """Handle slider value change."""
        gain_db = value / 10.0
        self._gain_db = gain_db
        self.gain_label.setText(f"{gain_db:+.1f}" if gain_db != 0 else "0")
        # Rate-limit the processor update
        self._rate_limiter.call(lambda g=gain_db: self._update_gain(g))

    def _update_gain(self, gain_db):
        """Update processor and curve (rate-limited)."""
        self.processor.set_eq_band_gain(self.band_index, gain_db)
        if self.parameter_callback:
            self.parameter_callback()
        if self.curve_callback:
            self.curve_callback()

    def _on_slider_released(self):
        """Ensure final value is applied when slider is released."""
        self._rate_limiter.flush()

    def _on_q_changed(self, value):
        """Handle Q spinbox value change."""
        self._q = float(value)
        self.bandwidth_mode = "q"
        self.bandwidth_octaves = None
        # Rate-limit the processor update
        self._rate_limiter.call(lambda q=value: self._update_q(q))

    def _update_q(self, q):
        """Update processor and curve (rate-limited)."""
        self.processor.set_eq_band_q(self.band_index, q)
        if self.parameter_callback:
            self.parameter_callback()
        if self.curve_callback:
            self.curve_callback()

    def _on_filter_type_changed(self) -> None:
        """Apply a manual filter-type change."""
        filter_type = self.filter_type()
        self.bandwidth_mode = "q"
        self.bandwidth_octaves = None
        self._set_parameter_availability()
        self.processor.set_eq_band_filter_type(self.band_index, filter_type)
        if self.parameter_callback:
            self.parameter_callback()
        if self.curve_callback:
            self.curve_callback()

    def _on_slope_changed(self) -> None:
        """Apply an even-order pass-filter slope."""
        slope = self.slope_db_per_octave()
        self.processor.set_eq_band_slope(self.band_index, slope)
        if self.parameter_callback:
            self.parameter_callback()
        if self.curve_callback:
            self.curve_callback()

    def _on_band_enabled_changed(self, enabled: bool) -> None:
        """Apply one band's click-safe bypass state."""
        self.processor.set_eq_band_enabled(self.band_index, enabled)
        if self.parameter_callback:
            self.parameter_callback()
        if self.curve_callback:
            self.curve_callback()

    def _set_parameter_availability(self) -> None:
        filter_type = self.filter_type()
        gain_enabled = filter_type in GAIN_FILTER_TYPES
        q_enabled = filter_type in Q_FILTER_TYPES
        slope_enabled = filter_type in PASS_FILTER_TYPES
        self.slider.setEnabled(gain_enabled)
        self.gain_label.setEnabled(gain_enabled)
        self.q_spinbox.setEnabled(q_enabled)
        self.q_label.setEnabled(q_enabled)
        self.slope_combo.setEnabled(slope_enabled)
        self.slope_label.setEnabled(slope_enabled)
        if filter_type == "notch":
            self.filter_type_combo.setToolTip(
                "Notch: gain is ignored; Q controls rejection width"
            )
        elif slope_enabled:
            self.filter_type_combo.setToolTip(
                "Pass filter: gain and Q are ignored; slope controls order"
            )
        else:
            self.filter_type_combo.setToolTip("Filter type")

    def _on_frequency_changed(self, value):
        """Handle frequency spinbox changes."""
        self._frequency_hz = float(value)
        self.bandwidth_mode = "q"
        self.bandwidth_octaves = None
        self.set_frequency_label(value)
        self._frequency_rate_limiter.call(
            lambda freq=value: self._update_frequency(freq)
        )

    def _update_frequency(self, frequency_hz: float):
        """Update processor, stored band frequency, and curve."""
        self.processor.set_eq_band_frequency(self.band_index, frequency_hz)
        self.set_frequency_label(frequency_hz)
        if self.parameter_callback:
            self.parameter_callback()
        if self.frequency_callback:
            self.frequency_callback(self.band_index, frequency_hz)
        elif self.curve_callback:
            self.curve_callback()

    def set_gain(self, gain_db: float):
        """Set gain value programmatically."""
        self._gain_db = min(12.0, max(-12.0, float(gain_db)))
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(self._gain_db * 10.0)))
        self.slider.blockSignals(False)
        displayed_gain = self.slider.value() / 10.0
        self.gain_label.setText(
            f"{displayed_gain:+.1f}" if displayed_gain != 0 else "0"
        )

    def set_q(self, q: float):
        """Set Q value programmatically."""
        self._q = min(10.0, max(0.1, float(q)))
        self.q_spinbox.blockSignals(True)
        self.q_spinbox.setValue(self._q)
        self.q_spinbox.blockSignals(False)
        self.bandwidth_mode = "q"
        self.bandwidth_octaves = None

    def set_frequency(self, frequency_hz: float):
        """Set center frequency programmatically."""
        self._frequency_hz = min(20_000.0, max(20.0, float(frequency_hz)))
        self.frequency_spinbox.blockSignals(True)
        self.frequency_spinbox.setValue(self._frequency_hz)
        self.frequency_spinbox.blockSignals(False)
        self.set_frequency_label(self._frequency_hz)

    def set_frequency_label(self, frequency_hz: float) -> None:
        """Set displayed center frequency."""
        self.freq_label.setText(_format_frequency_label(frequency_hz))

    def frequency_hz(self) -> float:
        """Return current center frequency."""
        return self._frequency_hz

    def filter_type(self) -> str:
        """Return the stable serialized filter type."""
        value = self.filter_type_combo.currentData()
        return str(value) if value is not None else "bell"

    def slope_db_per_octave(self) -> int:
        """Return the selected pass-filter slope."""
        value = self.slope_combo.currentData()
        return int(value) if value is not None else 12

    def band_enabled(self) -> bool:
        """Return whether this band participates in the response."""
        return self.band_enabled_checkbox.isChecked()

    def set_filter_type(self, filter_type: str) -> None:
        """Set filter type without producing a control callback."""
        index = self.filter_type_combo.findData(filter_type)
        if index < 0:
            raise ValueError(f"Unsupported EQ filter type: {filter_type}")
        self.filter_type_combo.blockSignals(True)
        self.filter_type_combo.setCurrentIndex(index)
        self.filter_type_combo.blockSignals(False)
        self._set_parameter_availability()

    def set_slope(self, slope_db_per_octave: int) -> None:
        """Set pass-filter slope without producing a control callback."""
        index = self.slope_combo.findData(int(slope_db_per_octave))
        if index < 0:
            raise ValueError(f"Unsupported EQ slope: {slope_db_per_octave}")
        self.slope_combo.blockSignals(True)
        self.slope_combo.setCurrentIndex(index)
        self.slope_combo.blockSignals(False)

    def set_band_enabled(self, enabled: bool) -> None:
        """Set per-band bypass without producing a control callback."""
        self.band_enabled_checkbox.blockSignals(True)
        self.band_enabled_checkbox.setChecked(bool(enabled))
        self.band_enabled_checkbox.blockSignals(False)

    def native_config(self) -> tuple[str, float, float, float, int, bool]:
        """Return the exact native v2 tuple for this band."""
        return (
            self.filter_type(),
            self.frequency_hz(),
            self._gain_db,
            self._q,
            self.slope_db_per_octave(),
            self.band_enabled(),
        )

    def settings(self) -> EQBandSettings:
        """Return this band's immutable serialized settings."""
        filter_type, frequency, gain, q, slope, enabled = self.native_config()
        return EQBandSettings(
            filter_type=filter_type,
            frequency_hz=frequency,
            gain_db=gain,
            q=q,
            bandwidth_mode=self.bandwidth_mode,
            bandwidth_octaves=self.bandwidth_octaves,
            slope_db_per_octave=slope,
            stage=self.stage,
            enabled=enabled,
        )

    def set_settings(self, settings: EQBandSettings) -> None:
        """Load an immutable typed band without callback feedback."""
        self.set_filter_type(settings.filter_type)
        self.set_frequency(settings.frequency_hz)
        self.set_gain(settings.gain_db)
        self.set_q(settings.q)
        self.set_slope(settings.slope_db_per_octave)
        self.set_band_enabled(settings.enabled)
        self.bandwidth_mode = settings.bandwidth_mode
        self.bandwidth_octaves = settings.bandwidth_octaves
        self.stage = settings.stage

    def reset(self, frequency_hz: float):
        """Reset to 0 dB, default Q, and default frequency."""
        self.set_filter_type(_default_filter_type(self.band_index))
        self.set_slope(12)
        self.set_band_enabled(True)
        self.set_gain(0.0)
        self.set_q(1.41)
        self.set_frequency(frequency_hz)
        self.processor.set_eq_band_filter_type(
            self.band_index,
            _default_filter_type(self.band_index),
        )
        self.processor.set_eq_band_slope(self.band_index, 12)
        self.processor.set_eq_band_enabled(self.band_index, True)
        self.processor.set_eq_band_gain(self.band_index, 0.0)
        self.processor.set_eq_band_q(self.band_index, 1.41)
        self.processor.set_eq_band_frequency(self.band_index, frequency_hz)


class EQPanel(QWidget):
    """10-Band Parametric EQ control panel."""

    BAND_COLUMN_OPTIONS = (10, 5, 4, 3, 2, 1)
    WIDE_PRESET_LAYOUT_WIDTH = 520

    configurationEditStarted = pyqtSignal()
    configurationEditFinished = pyqtSignal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.band_sliders = []
        self.band_freqs_hz = list(BAND_FREQUENCIES_HZ)
        self._correction_bands: tuple[EQBandSettings, ...] | None = None
        self._tone_bands: tuple[EQBandSettings, ...] = tuple(
            EQSettings().bands
        )
        self._auto_eq_diagnostics: dict | None = None
        self._curve_rate_limiter = RateLimiter(interval_ms=33)
        self._band_layout_columns = 0
        self._preset_layout_columns = 0
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        self._outer_layout = layout

        # Allow panel to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # EQ Group
        eq_group = QGroupBox("10-Band Parametric EQ")
        self._eq_group = eq_group
        eq_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        eq_layout = QVBoxLayout(eq_group)
        self._eq_layout = eq_layout

        # Enable checkbox and reset button
        controls_layout = QHBoxLayout()

        self.enabled_checkbox = QCheckBox("Enable EQ")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        controls_layout.addWidget(self.enabled_checkbox)

        controls_layout.addStretch()

        clear_correction_btn = QPushButton("Clear Mic Correction")
        clear_correction_btn.setToolTip(
            "Remove measured microphone correction while keeping the tone"
        )
        clear_correction_btn.clicked.connect(self.clear_correction)
        controls_layout.addWidget(clear_correction_btn)

        reset_btn = QPushButton("Reset Tone")
        reset_btn.setToolTip("Reset character gains to zero while retaining microphone correction")
        reset_btn.clicked.connect(self._reset_all)
        controls_layout.addWidget(reset_btn)

        eq_layout.addLayout(controls_layout)

        # Frequency response curve (above sliders)
        self.curve_widget = EQCurveWidget()
        self.curve_widget.setFixedHeight(100)
        self.curve_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.curve_widget.setToolTip(
            "The curve shows correction plus tone; handles edit tone only. "
            "Drag a handle to edit frequency and gain. Notch and pass filters "
            "move horizontally only. Use [ and ] plus arrow keys for keyboard editing."
        )
        self.curve_widget.bandDragStarted.connect(self._on_curve_drag_started)
        self.curve_widget.bandDragged.connect(self._on_curve_band_dragged)
        self.curve_widget.bandDragFinished.connect(self._on_curve_drag_finished)
        self.curve_widget.bandDragCancelled.connect(self._on_curve_drag_cancelled)
        eq_layout.addWidget(self.curve_widget)

        self.auto_eq_diag_label = QLabel("Auto-EQ: no calibration diagnostics")
        self.auto_eq_diag_label.setStyleSheet(status_chip_style("idle"))
        self.auto_eq_diag_label.setToolTip(
            "Auto-EQ diagnostics appear after calibration."
        )
        self.auto_eq_diag_label.setWordWrap(True)
        eq_layout.addWidget(self.auto_eq_diag_label)

        # Band sliders
        self.sliders_layout = QGridLayout()
        self.sliders_layout.setSpacing(5)

        for i, frequency_hz in enumerate(BAND_FREQUENCIES_HZ):
            band_slider = EQBandSlider(
                i,
                frequency_hz,
                self.processor,
                curve_callback=self._update_curve,
                frequency_callback=self._on_band_frequency_changed,
                parameter_callback=self._on_band_parameter_changed,
            )
            self.band_sliders.append(band_slider)

        self._reflow_band_sliders(self.width())
        eq_layout.addLayout(self.sliders_layout, stretch=1)

        # Initial curve update
        self._update_curve()

        # Preset buttons
        self.presets_layout = QGridLayout()
        self.presets_layout.setSpacing(SPACING_TIGHT)

        voice_btn = QPushButton("Voice")
        voice_btn.setToolTip("Voice clarity tone; preserves microphone correction and other processing")
        voice_btn.clicked.connect(self._preset_voice)

        bass_btn = QPushButton("Bass Cut")
        bass_btn.setToolTip("EQ-only bass trimming; not a rumble high-pass filter")
        bass_btn.clicked.connect(self._preset_bass_cut)

        presence_btn = QPushButton("Presence")
        presence_btn.setToolTip("Presence tone; preserves microphone correction and other processing")
        presence_btn.clicked.connect(self._preset_presence)

        warm_clear_btn = QPushButton("Warm & Clear")
        warm_clear_btn.setToolTip(
            "Strong EQ-only replacement: trims bass, lifts low mids, and cuts "
            "harsh upper mids; no rumble high-pass filter"
        )
        warm_clear_btn.clicked.connect(self._preset_warm_clear)

        flat_btn = QPushButton("Flat")
        flat_btn.setToolTip("Neutral character gains; preserves microphone correction and other processing")
        flat_btn.clicked.connect(self._reset_all)
        self._preset_buttons = (
            voice_btn,
            bass_btn,
            presence_btn,
            warm_clear_btn,
            flat_btn,
        )
        self._reflow_preset_buttons(self.width())

        eq_layout.addLayout(self.presets_layout)

        layout.addWidget(eq_group)

    def _reflow_band_sliders(self, width: int) -> None:
        outer_margins = self._outer_layout.contentsMargins()
        group_margins = self._eq_layout.contentsMargins()
        available_width = max(
            1,
            width
            - outer_margins.left()
            - outer_margins.right()
            - group_margins.left()
            - group_margins.right()
            - 8,
        )
        band_width = max(
            band_slider.minimumSizeHint().width() for band_slider in self.band_sliders
        )
        spacing = max(0, self.sliders_layout.horizontalSpacing())
        columns = 1
        for candidate in self.BAND_COLUMN_OPTIONS:
            required_width = candidate * band_width + (candidate - 1) * spacing
            if required_width <= available_width:
                columns = candidate
                break
        if columns == self._band_layout_columns:
            return
        self._band_layout_columns = columns
        for band_slider in self.band_sliders:
            self.sliders_layout.removeWidget(band_slider)
        for column in range(10):
            self.sliders_layout.setColumnStretch(column, 0)
        for index, band_slider in enumerate(self.band_sliders):
            row, column = divmod(index, columns)
            self.sliders_layout.addWidget(band_slider, row, column)
            self.sliders_layout.setColumnStretch(column, 1)
        self.sliders_layout.invalidate()
        self.updateGeometry()

    def _reflow_preset_buttons(self, width: int) -> None:
        columns = 5 if width >= self.WIDE_PRESET_LAYOUT_WIDTH else 3
        if columns == self._preset_layout_columns:
            return
        self._preset_layout_columns = columns
        for button in self._preset_buttons:
            self.presets_layout.removeWidget(button)
        for column in range(5):
            self.presets_layout.setColumnStretch(column, 0)
        for index, button in enumerate(self._preset_buttons):
            row, column = divmod(index, columns)
            self.presets_layout.addWidget(button, row, column)
            self.presets_layout.setColumnStretch(column, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        available_width = event.size().width()
        parent = self.parentWidget()
        if parent is not None and parent.width() > 0:
            available_width = min(available_width, parent.width())
        self._reflow_band_sliders(available_width)
        self._reflow_preset_buttons(available_width)

    def _editable_bands(self) -> tuple[EQBandSettings, ...]:
        """Return the independently editable tone stage."""
        return self._tone_bands

    def _native_bands(
        self, bands: tuple[EQBandSettings, ...]
    ) -> list[tuple[str, float, float, float, int, bool]]:
        return [band.to_native() for band in bands]

    def _update_layer_status(self) -> None:
        if self._correction_bands is None:
            text = "Mic correction: none | Tone: editable below"
        else:
            active = sum(
                abs(band.gain_db) >= 0.25
                for band in self._correction_bands
            )
            text = (
                f"Mic correction: {active} measured band(s) | "
                "Tone: independent and preserved"
            )
        self._eq_group.setTitle("Tone EQ — " + text)

    def _apply_layers_to_processor(self) -> None:
        effective = self._editable_bands()
        correction = self._correction_bands or EQSettings().bands
        self.processor.apply_eq_layers(self._native_bands(correction), self._native_bands(effective))
        self.curve_widget.set_all_params(self._native_bands(effective),
                                         correction=self._native_bands(self._correction_bands or ()))
        self.band_freqs_hz = [band.frequency_hz for band in effective]
        if self.curve_widget.band_markers:
            self.curve_widget.set_band_markers(self.band_freqs_hz)
        self._update_layer_status()

    def _sync_tone_layer_from_sliders(self) -> None:
        self._tone_bands = tuple(slider.settings() for slider in self.band_sliders)

    def _on_band_parameter_changed(self) -> None:
        """Apply both stages after a user edits the tone controls."""
        self.set_auto_eq_diagnostics(None)
        self._sync_tone_layer_from_sliders()
        self._apply_layers_to_processor()

    def clear_correction(self) -> None:
        """Remove only microphone correction and retain the selected tone."""
        if self._correction_bands is None:
            return
        self._correction_bands = None
        for slider, band in zip(self.band_sliders, self._tone_bands, strict=True):
            slider.set_settings(band)
        self._apply_layers_to_processor()
        self.set_auto_eq_diagnostics(None)

    def _on_enabled_toggled(self, checked):
        """Handle EQ enable/disable."""
        self.set_auto_eq_diagnostics(None)
        self.processor.set_eq_enabled(checked)

    def _reset_all(self):
        """Reset the editable tone layer to a flat contour."""
        defaults = EQSettings()
        self._apply_typed_bands(defaults.bands)
        self.curve_widget.clear_band_markers()
        self.set_auto_eq_diagnostics(None)

    def _on_band_frequency_changed(self, band_index: int, frequency_hz: float) -> None:
        """Synchronize manual frequency edits with panel state."""
        self._sync_tone_layer_from_sliders()
        self._update_curve()
        if self.curve_widget.band_markers:
            self.curve_widget.set_band_markers(self.band_freqs_hz)

    def _on_curve_drag_started(self, _band_index: int) -> None:
        """Begin one history transaction for the whole drag/key gesture."""
        self.configurationEditStarted.emit()

    def _on_curve_band_dragged(
        self,
        band_index: int,
        frequency_hz: float,
        gain_db: float,
    ) -> None:
        """Rate-limit graph edits while the curve itself tracks immediately."""
        self._curve_rate_limiter.call(
            lambda: self._apply_curve_band_edit(
                band_index,
                frequency_hz,
                gain_db,
            )
        )

    def _apply_curve_band_edit(
        self,
        band_index: int,
        frequency_hz: float,
        gain_db: float,
    ) -> None:
        if not 0 <= band_index < len(self.band_sliders):
            raise ValueError(f"Invalid EQ band index: {band_index}")
        self.set_auto_eq_diagnostics(None)
        slider = self.band_sliders[band_index]
        slider.set_frequency(min(20_000.0, max(20.0, frequency_hz)))
        if slider.filter_type() in GAIN_FILTER_TYPES:
            slider.set_gain(min(12.0, max(-12.0, gain_db)))
        slider.bandwidth_mode = "q"
        slider.bandwidth_octaves = None
        self._sync_tone_layer_from_sliders()
        self._apply_layers_to_processor()
        if self.curve_widget.band_markers:
            self.curve_widget.set_band_markers(self.band_freqs_hz)

    def _on_curve_drag_finished(
        self,
        band_index: int,
        frequency_hz: float,
        gain_db: float,
    ) -> None:
        self._on_curve_band_dragged(band_index, frequency_hz, gain_db)
        self._curve_rate_limiter.flush()
        self.configurationEditFinished.emit("EQ graph edit")

    def _on_curve_drag_cancelled(
        self,
        band_index: int,
        frequency_hz: float,
        gain_db: float,
    ) -> None:
        self._on_curve_band_dragged(band_index, frequency_hz, gain_db)
        self._curve_rate_limiter.flush()
        self.configurationEditFinished.emit("Cancelled EQ graph edit")

    def _update_curve(self):
        """Update frequency response curve based on current band parameters."""
        self.set_auto_eq_diagnostics(None)
        effective = self._editable_bands()
        self.band_freqs_hz = [band.frequency_hz for band in effective]
        self.curve_widget.set_all_params(self._native_bands(effective),
                                         correction=self._native_bands(self._correction_bands or ()))

    def _preset_voice(self):
        """Apply voice clarity preset."""
        self._apply_catalog_preset("voice")

    def _preset_bass_cut(self):
        """Apply bass cut preset (high-pass effect)."""
        self._apply_catalog_preset("bass_cut")

    def _preset_presence(self):
        """Apply presence boost preset."""
        self._apply_catalog_preset("presence")

    def _preset_warm_clear(self):
        """Apply the strong warm/clear EQ contour without a rumble filter."""
        # Refined mapping with blended midrange for nasal reduction
        gains = [-12.0, 4.0, 4.0, 3.0, -3.0, -10.0, 0.0, 0.0, 0.0, 0.0]
        qs = [0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707]
        self._apply_preset(gains, qs)

    def _apply_catalog_preset(self, key: str) -> None:
        """Apply a catalog contour while preserving microphone correction."""
        bands = BUILTIN_PRESETS[key].eq.bands
        self._apply_preset(
            [band.gain_db for band in bands],
            [band.q for band in bands],
        )

    def _apply_preset(
        self,
        gains: list,
        qs: list | None = None,
        freqs: list | None = None,
        show_markers: bool = False,
    ):
        """Apply a preset with given gain and Q values."""
        if qs is None:
            qs = [1.41] * len(gains)  # Default Q if not provided
        if freqs is None:
            freqs = list(BAND_FREQUENCIES_HZ)

        # Build typed historical-layout bands.
        bands = []
        for i in range(min(len(gains), len(self.band_sliders))):
            freq = freqs[i] if i < len(freqs) else BAND_FREQUENCIES_HZ[i]
            gain = gains[i]
            q = qs[i] if i < len(qs) else 1.41
            bands.append(
                EQBandSettings(
                    filter_type=_default_filter_type(i),
                    frequency_hz=float(freq),
                    gain_db=float(gain),
                    q=float(q),
                )
            )

        self._apply_typed_bands(tuple(bands))
        if len(bands) == len(self.band_freqs_hz):
            self.band_freqs_hz = [band.frequency_hz for band in bands]
        if show_markers:
            self.curve_widget.set_band_markers(self.band_freqs_hz)
        else:
            self.curve_widget.clear_band_markers()
        self.set_auto_eq_diagnostics(None)

    def apply_auto_eq_results(
        self,
        bands: list,
        diagnostics: dict | None = None,
        *,
        layer: str = "correction",
    ):
        """
        Apply auto-EQ analysis results to one EQ layer.

        Updates UI sliders, Q spinboxes, and processor atomically.
        Uses blockSignals() to prevent feedback loops during update.

        Args:
            bands: List of 10 (frequency_hz, gain_db, q) tuples
            layer: ``correction`` for measured microphone response or
                ``tone`` for a user character contour.

        Raises:
            ValueError: If bands list does not contain exactly 10 elements
        """
        if len(bands) != 10:
            raise ValueError(f"Expected 10 bands, got {len(bands)}")

        typed_bands = tuple(
            EQBandSettings(
                filter_type=_default_filter_type(index),
                frequency_hz=float(freq),
                gain_db=float(gain),
                q=float(q),
            )
            for index, (freq, gain, q) in enumerate(bands)
        )
        if layer == "correction":
            # The analyzed gains are the complete accepted correction. Start
            # character at zero so an older tone is not applied a second time.
            self._tone_bands = tuple(replace(band, gain_db=0.0) for band in typed_bands)
        self._apply_typed_bands(typed_bands, layer=layer)
        self.curve_widget.set_band_markers(self.band_freqs_hz)
        self.set_auto_eq_diagnostics(diagnostics)

    def _apply_typed_bands(
        self,
        bands: tuple[EQBandSettings, ...],
        *,
        layer: str = "tone",
    ) -> None:
        """Apply one immutable snapshot to a layer and recompose native DSP."""
        if len(bands) != len(self.band_sliders):
            raise ValueError(
                f"Expected {len(self.band_sliders)} bands, got {len(bands)}"
            )
        if layer not in {"correction", "tone"}:
            raise ValueError(f"Unsupported EQ layer: {layer}")
        if layer == "correction":
            self._correction_bands = bands
        else:
            self._tone_bands = bands
        effective = self._editable_bands()
        for slider, band in zip(self.band_sliders, effective, strict=True):
            slider.set_settings(band)
        self._apply_layers_to_processor()

    def set_auto_eq_diagnostics(self, diagnostics: dict | None) -> None:
        """Show the last Auto-EQ confidence and validation diagnostics."""
        self._auto_eq_diagnostics = dict(diagnostics) if diagnostics else None
        text, state, tooltip = _format_auto_eq_diagnostics(self._auto_eq_diagnostics)
        self.auto_eq_diag_label.setText(text)
        self.auto_eq_diag_label.setStyleSheet(status_chip_style(state))
        self.auto_eq_diag_label.setToolTip(
            tooltip or "Auto-EQ diagnostics appear after calibration."
        )

    def get_settings(self) -> dict:
        """Get current EQ settings as a dictionary."""
        settings = self.get_eq_settings()
        payload = settings.to_dict()
        payload.update(
            {
                "band_freqs": settings.band_freqs,
                "band_gains": settings.band_gains,
                "band_qs": settings.band_qs,
            }
        )
        return payload

    def get_eq_settings(self) -> EQSettings:
        """Return tone settings and the independent correction stage."""
        self._sync_tone_layer_from_sliders()
        effective = self._editable_bands()
        if self._correction_bands is None:
            return EQSettings(
                enabled=self.enabled_checkbox.isChecked(),
                bands=effective,
            )
        return EQSettings(
            enabled=self.enabled_checkbox.isChecked(),
            bands=effective,
            correction_bands=self._correction_bands,
            tone_bands=self._tone_bands,
        )

    def set_settings(self, settings: dict) -> None:
        """Apply settings from a dictionary."""
        if "enabled" in settings:
            self.enabled_checkbox.setChecked(bool(settings["enabled"]))
        raw_bands = settings.get("bands")
        if raw_bands is not None:
            payload = {
                "schema_version": settings.get("schema_version", EQ_SCHEMA_VERSION),
                "enabled": bool(settings.get("enabled", True)),
                "bands": list(raw_bands),
            }
            if "layers" in settings:
                payload["layers"] = settings["layers"]
            parsed = EQSettings.from_dict(payload)
            if parsed.correction_bands is None:
                self._correction_bands = None
                self._tone_bands = parsed.bands
            else:
                self._correction_bands = parsed.correction_bands
                self._tone_bands = parsed.tone_bands or parsed.bands
            for slider, band in zip(
                self.band_sliders,
                self._editable_bands(),
                strict=True,
            ):
                slider.set_settings(band)
            self._apply_layers_to_processor()
            self.set_auto_eq_diagnostics(None)
        elif "band_gains" in settings:
            gains = settings["band_gains"]
            # Default to 1.41 Q for backwards compatibility with old presets
            qs = settings.get("band_qs", [1.41] * len(gains))
            freqs = settings.get("band_freqs", BAND_FREQUENCIES_HZ)
            show_markers = any(
                abs(float(freq) - float(default_freq)) > 1e-6
                for freq, default_freq in zip(freqs, BAND_FREQUENCIES_HZ)
            )
            self._apply_preset(gains, qs, freqs, show_markers=show_markers)
        else:
            self.set_auto_eq_diagnostics(None)
