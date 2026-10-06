"""10-band parametric EQ control panel."""

from dataclasses import replace

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QMenu,
    QStackedWidget,
    QPushButton,
    QDoubleSpinBox,
    QComboBox,
    QSizePolicy,
    QToolButton,
)
from PySide6.QtCore import QObject, QSignalBlocker, Qt, Signal
from .eq_state import (
    EQState,
    EQ_FILTER_OPTIONS,
    PASS_FILTER_TYPES,
    GAIN_FILTER_TYPES,
    Q_FILTER_TYPES,
    _default_filter_type,
    _format_frequency_label,
    _format_auto_eq_diagnostics as _format_auto_eq_diagnostics,
)
from .eq_curve import EQCurveWidget
from .eq_graph import EQGraphModel
from .rate_limiter import RateLimiter
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    MARGIN_PANEL,
    PRIMARY_LABEL_STYLE,
    SPACING_NORMAL,
    SPACING_TIGHT,
    SUBDUED_TEXT_STYLE,
    fit_spinbox_to_contents,
    status_chip_style,
)
from .components import Card, Glyph, IconButton, ToggleSwitch
from ..config import (
    EQBandSettings,
    EQSettings,
    EQ_FREQUENCIES,
    EQ_SLOPES_DB_PER_OCTAVE,
)


# Numeric frequencies in Hz for curve calculation (single source of truth from config)
BAND_FREQUENCIES_HZ = list(EQ_FREQUENCIES)


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
        eq_state: EQState | None = None,
    ):
        super().__init__(parent)
        self.band_index = band_index
        self.processor = processor
        self.curve_callback = curve_callback
        self.frequency_callback = frequency_callback
        self.parameter_callback = parameter_callback
        self.eq_state = eq_state
        if eq_state is None:
            self._standalone_band = EQBandSettings(
                _default_filter_type(band_index), frequency_hz, 0.0, 1.41
            )
            self._rate_limiter = RateLimiter(interval_ms=33)
            self._frequency_rate_limiter = RateLimiter(interval_ms=33)
        else:
            self._rate_limiter = eq_state._rate_limiter
            self._frequency_rate_limiter = eq_state._rate_limiter
        self._setup_ui(frequency_hz)
        self._render_settings()

    def _setup_ui(self, frequency_hz: float):
        """Build the editor row for one band."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_NORMAL)

        # Leads the row so it is clear which band the controls edit.
        self.freq_label = QLabel(_format_frequency_label(frequency_hz))
        self.freq_label.setMinimumWidth(48)
        self.freq_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        self.freq_label.setToolTip(f"EQ band {self.band_index + 1} center frequency")

        self.band_enabled_checkbox = ToggleSwitch("On")
        self.band_enabled_checkbox.setChecked(True)
        self.band_enabled_checkbox.setToolTip("Enable this EQ band")
        self.band_enabled_checkbox.toggled.connect(self._on_band_enabled_changed)

        self.filter_type_combo = QComboBox()
        for display_name, filter_type in EQ_FILTER_OPTIONS:
            self.filter_type_combo.addItem(display_name, filter_type)
        self.filter_type_combo.setToolTip("Filter type")
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)

        gain_caption = QLabel("Gain")
        gain_caption.setStyleSheet(SUBDUED_TEXT_STYLE)
        # -12 to +12 dB in 0.1 dB steps.
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(-120, 120)
        self.slider.setValue(0)
        self.slider.setMinimumWidth(120)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.gain_label = QLabel("0")
        self.gain_label.setMinimumWidth(40)

        top_row = QHBoxLayout()
        top_row.setSpacing(MARGIN_PANEL)
        top_row.addWidget(self.freq_label)
        top_row.addWidget(self.band_enabled_checkbox)
        top_row.addWidget(self.filter_type_combo)
        top_row.addSpacing(SPACING_NORMAL)
        top_row.addWidget(gain_caption)
        top_row.addWidget(self.slider, stretch=1)
        top_row.addWidget(self.gain_label)
        layout.addLayout(top_row)

        freq_label = QLabel("Frequency")
        freq_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.frequency_spinbox = QDoubleSpinBox()
        self.frequency_spinbox.setRange(20.0, 20000.0)
        self.frequency_spinbox.setSingleStep(10.0)
        self.frequency_spinbox.setDecimals(0)
        self.frequency_spinbox.setSuffix(" Hz")
        self.frequency_spinbox.setValue(frequency_hz)
        fit_spinbox_to_contents(self.frequency_spinbox)
        self.frequency_spinbox.setToolTip("Center frequency in Hz")
        self.frequency_spinbox.valueChanged.connect(self._on_frequency_changed)

        self.q_label = QLabel("Q")
        self.q_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.q_spinbox = QDoubleSpinBox()
        self.q_spinbox.setRange(0.1, 10.0)
        self.q_spinbox.setSingleStep(0.1)
        self.q_spinbox.setDecimals(1)
        self.q_spinbox.setValue(1.41)
        fit_spinbox_to_contents(self.q_spinbox)
        self.q_spinbox.valueChanged.connect(self._on_q_changed)

        self.slope_label = QLabel("Slope")
        self.slope_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.slope_combo = QComboBox()
        for slope in sorted(EQ_SLOPES_DB_PER_OCTAVE):
            self.slope_combo.addItem(f"{slope} dB/oct", slope)
        self.slope_combo.setToolTip("Pass-filter slope in dB per octave")
        self.slope_combo.currentIndexChanged.connect(self._on_slope_changed)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(SPACING_NORMAL)
        for caption, control in (
            (freq_label, self.frequency_spinbox),
            (self.q_label, self.q_spinbox),
            (self.slope_label, self.slope_combo),
        ):
            bottom_row.addWidget(caption)
            bottom_row.addWidget(control)
            bottom_row.addSpacing(MARGIN_PANEL)
        bottom_row.addStretch(1)
        layout.addLayout(bottom_row)

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
        self._set_parameter_availability()

    def _edit(self, name: str, value) -> None:
        if self.eq_state is not None:
            self.eq_state.set_band_value(self.band_index, name, value)
            return
        changes = {name: value}
        if name in {"frequency_hz", "q", "filter_type"}:
            changes.update(bandwidth_mode="q", bandwidth_octaves=None)
        self._standalone_band = replace(self._standalone_band, **changes)
        self._render_settings()

        def apply():
            self._write_standalone_parameter(name, value)

        if name == "frequency_hz":
            self._frequency_rate_limiter.call(apply)
        elif name in {"gain_db", "q"}:
            self._rate_limiter.call(apply)
        else:
            apply()

    def _write_standalone_parameter(self, name: str, value) -> None:
        if self.parameter_callback:
            self.parameter_callback()
            return
        setters = {
            "gain_db": "set_eq_band_gain",
            "q": "set_eq_band_q",
            "frequency_hz": "set_eq_band_frequency",
            "filter_type": "set_eq_band_filter_type",
            "slope_db_per_octave": "set_eq_band_slope",
            "enabled": "set_eq_band_enabled",
        }
        getattr(self.processor, setters[name])(self.band_index, value)
        if name == "frequency_hz" and self.frequency_callback:
            self.frequency_callback(self.band_index, value)
        elif self.curve_callback:
            self.curve_callback()

    def _on_slider_changed(self, value):
        self._edit("gain_db", value / 10.0)

    def _on_slider_released(self):
        self._rate_limiter.flush()

    def _on_q_changed(self, value):
        self._edit("q", float(value))

    def _on_filter_type_changed(self) -> None:
        self._edit("filter_type", str(self.filter_type_combo.currentData()))

    def _on_slope_changed(self) -> None:
        self._edit("slope_db_per_octave", int(self.slope_combo.currentData()))

    def _on_band_enabled_changed(self, enabled: bool) -> None:
        self._edit("enabled", enabled)

    def _on_frequency_changed(self, value):
        self._edit("frequency_hz", float(value))

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
        self.filter_type_combo.setToolTip(
            "Notch: gain is ignored; Q controls rejection width"
            if filter_type == "notch"
            else "Pass filter: gain and Q are ignored; slope controls order"
            if slope_enabled
            else "Filter type"
        )

    def _render_settings(self) -> None:
        band = self.settings()
        for control, value in (
            (self.slider, int(round(band.gain_db * 10))),
            (self.frequency_spinbox, band.frequency_hz),
            (self.q_spinbox, band.q),
        ):
            with QSignalBlocker(control):
                if isinstance(control, QSlider):
                    control.setValue(int(value))
                else:
                    control.setValue(float(value))
        with QSignalBlocker(self.filter_type_combo):
            self.filter_type_combo.setCurrentIndex(
                self.filter_type_combo.findData(band.filter_type)
            )
        with QSignalBlocker(self.slope_combo):
            self.slope_combo.setCurrentIndex(
                self.slope_combo.findData(band.slope_db_per_octave)
            )
        with QSignalBlocker(self.band_enabled_checkbox):
            self.band_enabled_checkbox.setChecked(band.enabled)
        displayed_gain = self.slider.value() / 10.0
        self.gain_label.setText(f"{displayed_gain:+.1f}" if displayed_gain else "0")
        self.set_frequency_label(band.frequency_hz)
        self._set_parameter_availability()

    def _set_parameter(self, name: str, value) -> None:
        if self.eq_state is not None:
            self.eq_state.set_band_value(self.band_index, name, value, edited=False)
        else:
            changes = {name: value}
            if name in {"frequency_hz", "q", "filter_type"}:
                changes.update(bandwidth_mode="q", bandwidth_octaves=None)
            self._standalone_band = replace(self._standalone_band, **changes)
            self._render_settings()

    def set_gain(self, gain_db: float):
        self._set_parameter("gain_db", min(12.0, max(-12.0, float(gain_db))))

    def set_q(self, q: float):
        self._set_parameter("q", min(10.0, max(0.1, float(q))))

    def set_frequency(self, frequency_hz: float):
        self._set_parameter(
            "frequency_hz", min(20000.0, max(20.0, float(frequency_hz)))
        )

    def set_frequency_label(self, frequency_hz: float) -> None:
        self.freq_label.setText(_format_frequency_label(frequency_hz))

    def frequency_hz(self) -> float:
        return self.settings().frequency_hz

    def filter_type(self) -> str:
        return self.settings().filter_type

    def slope_db_per_octave(self) -> int:
        return self.settings().slope_db_per_octave

    def band_enabled(self) -> bool:
        return self.settings().enabled

    @property
    def bandwidth_mode(self) -> str:
        return self.settings().bandwidth_mode

    @property
    def bandwidth_octaves(self) -> float | None:
        return self.settings().bandwidth_octaves

    @property
    def stage(self) -> str:
        return self.settings().stage

    def set_filter_type(self, filter_type: str) -> None:
        self._set_parameter("filter_type", filter_type)

    def set_slope(self, slope_db_per_octave: int) -> None:
        self._set_parameter("slope_db_per_octave", slope_db_per_octave)

    def set_band_enabled(self, enabled: bool) -> None:
        self._set_parameter("enabled", enabled)

    def native_config(self) -> tuple[str, float, float, float, int, bool]:
        return self.settings().to_native()

    def settings(self) -> EQBandSettings:
        if self.eq_state is not None:
            return self.eq_state.get_band(self.band_index)
        return self._standalone_band

    def set_settings(self, settings: EQBandSettings) -> None:
        if self.eq_state is not None:
            self.eq_state.set_band(self.band_index, settings, edited=False)
        else:
            self._standalone_band = settings
            self._render_settings()

    def reset(self, frequency_hz: float):
        self.set_settings(
            EQBandSettings(
                _default_filter_type(self.band_index), frequency_hz, 0.0, 1.41
            )
        )
        if self.eq_state is not None:
            self.eq_state.flush()
        else:
            for key, value in (
                ("filter_type", self.filter_type()),
                ("slope_db_per_octave", 12),
                ("enabled", True),
                ("gain_db", 0.0),
                ("q", 1.41),
                ("frequency_hz", frequency_hz),
            ):
                self._write_standalone_parameter(key, value)


class EQPresentation(QObject):
    """Shared graph data and menus; the QWidget graph is created on demand."""

    def __init__(self, state: EQState, parent: QObject | None = None):
        super().__init__(parent)
        self.eq_state = state
        self.graph_model = EQGraphModel(self)
        self._curve_widget: EQCurveWidget | None = None
        widget_parent = parent if isinstance(parent, QWidget) else None
        tone_menu = QMenu(widget_parent)
        for name, tooltip, handler in (
            (
                "Voice",
                "Voice clarity tone; preserves the Auto-EQ layer and other processing",
                lambda: state.apply_catalog_preset("voice"),
            ),
            (
                "Bass Cut",
                "EQ-only bass trimming; not a rumble high-pass filter",
                lambda: state.apply_catalog_preset("bass_cut"),
            ),
            (
                "Presence",
                "Presence tone; preserves the Auto-EQ layer and other processing",
                lambda: state.apply_catalog_preset("presence"),
            ),
            (
                "Warm & Clear",
                "Strong EQ-only replacement: trims bass, lifts low mids, and cuts "
                "harsh upper mids; no rumble high-pass filter",
                state.apply_warm_clear_preset,
            ),
            (
                "Flat",
                "Neutral character gains; preserves the Auto-EQ layer and other processing",
                state.reset_tone,
            ),
        ):
            action = tone_menu.addAction(name.replace("&", "&&"))
            action.setToolTip(tooltip)
            action.triggered.connect(handler)
        tone_menu.setToolTipsVisible(True)
        self.tone_menu = tone_menu
        self._tone_preset_button: QPushButton | None = None

        options_menu = QMenu(widget_parent)
        clear_action = options_menu.addAction("Clear Auto-EQ")
        clear_action.setToolTip("Remove the calibrated EQ layer while keeping the tone")
        clear_action.triggered.connect(state.clear_correction)
        reset_action = options_menu.addAction("Reset tone")
        reset_action.setToolTip(
            "Reset character gains to zero while retaining the Auto-EQ layer"
        )
        reset_action.triggered.connect(state.reset_tone)
        options_menu.setToolTipsVisible(True)
        self.options_menu = options_menu
        self._options_button: IconButton | None = None
        self.tone_actions = tuple(self.tone_menu.actions())
        self.options_actions = tuple(self.options_menu.actions())

        self.graph_model.bandSelected.connect(state.select_band)
        self.graph_model.bandDragStarted.connect(state.begin_curve_edit)
        self.graph_model.bandDragged.connect(state.edit_curve_band)
        self.graph_model.bandDragFinished.connect(state.finish_curve_edit)
        self.graph_model.bandDragCancelled.connect(
            lambda index, frequency, gain: state.finish_curve_edit(
                index, frequency, gain, cancelled=True
            )
        )
        state.changed.connect(self._render_curve)
        state.presentationChanged.connect(self._render_presentation)
        for owned in (
            self.tone_menu,
            self.options_menu,
        ):
            self.destroyed.connect(owned.deleteLater)
        self._render_curve()

    @property
    def tone_preset_button(self) -> QPushButton:
        """Create the classic tone menu button only when its panel needs it."""
        if self._tone_preset_button is None:
            parent = self.parent()
            widget_parent = parent if isinstance(parent, QWidget) else None
            self._tone_preset_button = QPushButton("Tone preset", widget_parent)
            self._tone_preset_button.setMenu(self.tone_menu)
            self._tone_preset_button.hide()
            self.destroyed.connect(self._tone_preset_button.deleteLater)
        return self._tone_preset_button

    @property
    def options_button(self) -> IconButton:
        """Create the classic options button only when its panel needs it."""
        if self._options_button is None:
            parent = self.parent()
            widget_parent = parent if isinstance(parent, QWidget) else None
            self._options_button = IconButton(
                Glyph.MORE, "Equalizer options", widget_parent
            )
            self._options_button.setMenu(self.options_menu)
            self._options_button.setPopupMode(
                QToolButton.ToolButtonPopupMode.InstantPopup
            )
            self._options_button.hide()
            self.destroyed.connect(self._options_button.deleteLater)
        return self._options_button

    @property
    def curve_widget(self) -> EQCurveWidget:
        """Create the classic graph only when the QWidget view asks for it."""
        if self._curve_widget is None:
            parent = self.parent()
            widget_parent = parent if isinstance(parent, QWidget) else None
            self._curve_widget = EQCurveWidget(
                widget_parent,
                graph_model=self.graph_model,
            )
            self._curve_widget.setFixedHeight(260)
            self._curve_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._curve_widget.hide()
            self.destroyed.connect(self._curve_widget.deleteLater)
        return self._curve_widget

    def _render_curve(self) -> None:
        settings = self.eq_state.get_eq_settings()
        self.graph_model.set_all_params(
            [band.to_native() for band in settings.bands],
            correction=[band.to_native() for band in settings.correction_bands or ()],
        )
        self._render_presentation()

    def _render_presentation(self) -> None:
        if self.eq_state.show_markers:
            self.graph_model.set_band_markers(
                self.eq_state.get_eq_settings().band_freqs
            )
        else:
            self.graph_model.clear_band_markers()
        if self.eq_state.selected_band is not None:
            self.graph_model.select_band(self.eq_state.selected_band)


class EQPanel(QWidget):
    """10-Band Parametric EQ control panel."""

    configurationEditStarted = Signal()
    configurationEditFinished = Signal(str)
    configurationEdited = Signal(str)

    def __init__(
        self,
        processor,
        eq_state: EQState | None = None,
        *,
        presentation: EQPresentation | None = None,
    ):
        super().__init__()
        self.processor = processor
        self.band_sliders = []
        self.eq_state = eq_state or (
            presentation.eq_state if presentation else EQState(processor, self)
        )
        self._curve_rate_limiter = self.eq_state._curve_rate_limiter
        if presentation is not None and presentation.eq_state is not self.eq_state:
            raise ValueError("EQ presentation and panel must share the same state")
        self.presentation = presentation or EQPresentation(self.eq_state, self)
        self._setup_ui()
        self.eq_state.changed.connect(self._render_settings)
        self.eq_state.presentationChanged.connect(self._render_presentation)
        self.eq_state.configurationEditStarted.connect(
            self.configurationEditStarted.emit
        )
        self.eq_state.configurationEditFinished.connect(
            self.configurationEditFinished.emit
        )
        self.eq_state.configurationEdited.connect(self.configurationEdited.emit)
        self._render_settings()

    def _setup_ui(self):
        """Build the equalizer card: graph first, one band editor below it."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.enabled_checkbox = ToggleSwitch()
        self.enabled_checkbox.setAccessibleName("Enable EQ")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        card = Card(
            "Equalizer",
            switch=self.enabled_checkbox,
            help_text=(
                "The curve shows correction plus tone; handles edit tone only. "
                "Drag a handle to edit frequency and gain. Notch and pass filters "
                "move horizontally only. Use [ and ] plus arrow keys for keyboard editing."
            ),
        )

        self.layer_status_label = QLabel()
        self.layer_status_label.setAccessibleName("Active EQ layers")
        self.layer_status_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.layer_status_label.setWordWrap(True)

        card.add_header_widget(self.presentation.tone_preset_button)
        self.tone_preset_button = self.presentation.tone_preset_button
        self.tone_preset_button.show()
        card.set_menu(self.presentation.options_menu)
        self.curve_widget = self.presentation.curve_widget
        card.body.addWidget(self.curve_widget)
        self.curve_widget.show()
        card.body.addWidget(self.layer_status_label)

        self.auto_eq_diag_label = QLabel("Auto-EQ: no calibration diagnostics")
        self.auto_eq_diag_label.setStyleSheet(status_chip_style("idle"))
        self.auto_eq_diag_label.setProperty("health_state", "idle")
        self.auto_eq_diag_label.setToolTip(
            "Auto-EQ diagnostics appear after calibration."
        )
        self.auto_eq_diag_label.setWordWrap(True)
        card.body.addWidget(self.auto_eq_diag_label)

        # One editor per band; the graph selection decides which one shows.
        self.band_stack = QStackedWidget()
        # Selecting a band must never resize the window, so the stack takes
        # the width it is given instead of reporting its pages' hints.
        self.band_stack.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        for i, frequency_hz in enumerate(BAND_FREQUENCIES_HZ):
            band_slider = EQBandSlider(
                i,
                frequency_hz,
                self.processor,
                eq_state=self.eq_state,
            )
            self.band_sliders.append(band_slider)
            self.band_stack.addWidget(band_slider)

        previous_band = IconButton(Glyph.PREVIOUS, "Previous EQ band")
        previous_band.clicked.connect(lambda: self._step_band(-1))
        next_band = IconButton(Glyph.NEXT, "Next EQ band")
        next_band.clicked.connect(lambda: self._step_band(1))
        editor_row = QHBoxLayout()
        editor_row.setSpacing(SPACING_TIGHT)
        editor_row.addWidget(previous_band, alignment=Qt.AlignmentFlag.AlignTop)
        editor_row.addWidget(next_band, alignment=Qt.AlignmentFlag.AlignTop)
        editor_row.addSpacing(SPACING_NORMAL)
        editor_row.addWidget(self.band_stack, stretch=1)
        card.body.addLayout(editor_row)

        layout.addWidget(card)

    @property
    def band_freqs_hz(self) -> list[float]:
        return self.eq_state.get_eq_settings().band_freqs

    @property
    def _auto_eq_diagnostics(self) -> dict | None:
        return self.eq_state.auto_eq_diagnostics

    def _render_settings(self) -> None:
        settings = self.eq_state.get_eq_settings()
        with QSignalBlocker(self.enabled_checkbox):
            self.enabled_checkbox.setChecked(settings.enabled)
        for slider in self.band_sliders:
            slider._render_settings()
        self._render_presentation()

    def _render_presentation(self) -> None:
        self.layer_status_label.setText(self.eq_state.layer_status())
        selected = self.eq_state.selected_band
        if selected is not None:
            self.band_stack.setCurrentIndex(selected)
        text, state, tooltip = _format_auto_eq_diagnostics(
            self.eq_state.auto_eq_diagnostics
        )
        self.auto_eq_diag_label.setText(text)
        self.auto_eq_diag_label.setStyleSheet(status_chip_style(state))
        self.auto_eq_diag_label.setProperty("health_state", state)
        self.auto_eq_diag_label.setToolTip(
            tooltip or "Auto-EQ diagnostics appear after calibration."
        )

    def _step_band(self, direction: int) -> None:
        self.eq_state.step_band(direction)

    def clear_correction(self) -> None:
        self.eq_state.clear_correction()

    def _on_enabled_toggled(self, checked):
        self.eq_state.set_enabled(checked)

    def _reset_all(self):
        self.eq_state.reset_tone()

    def _on_curve_drag_started(self, band_index: int) -> None:
        self.eq_state.begin_curve_edit(band_index)

    def _on_curve_band_dragged(
        self, band_index: int, frequency_hz: float, gain_db: float
    ) -> None:
        self.eq_state.edit_curve_band(band_index, frequency_hz, gain_db)

    def _on_curve_drag_finished(
        self, band_index: int, frequency_hz: float, gain_db: float
    ) -> None:
        self.eq_state.finish_curve_edit(band_index, frequency_hz, gain_db)

    def _on_curve_drag_cancelled(
        self, band_index: int, frequency_hz: float, gain_db: float
    ) -> None:
        self.eq_state.finish_curve_edit(
            band_index, frequency_hz, gain_db, cancelled=True
        )

    def _preset_voice(self):
        """Apply voice clarity preset."""
        self._apply_catalog_preset("voice")

    def _preset_bass_cut(self):
        """Apply the low-shelf bass-cut contour; no high-pass filter."""
        self._apply_catalog_preset("bass_cut")

    def _preset_presence(self):
        """Apply presence boost preset."""
        self._apply_catalog_preset("presence")

    def _preset_warm_clear(self):
        """Apply the strong warm/clear EQ contour without a rumble filter."""
        self.eq_state.apply_warm_clear_preset()

    def _apply_catalog_preset(self, key: str) -> None:
        self.eq_state.apply_catalog_preset(key)

    def _apply_preset(
        self,
        gains: list,
        qs: list | None = None,
        freqs: list | None = None,
        show_markers: bool = False,
    ):
        self.eq_state.apply_preset(gains, qs, freqs, show_markers=show_markers)

    def apply_auto_eq_results(
        self, bands: list, diagnostics: dict | None = None, *, layer: str = "correction"
    ):
        self.eq_state.apply_auto_eq_results(bands, diagnostics, layer=layer)

    def _apply_typed_bands(
        self, bands: tuple[EQBandSettings, ...], *, layer: str = "tone"
    ) -> None:
        self.eq_state.apply_typed_bands(bands, layer=layer)

    def set_auto_eq_diagnostics(self, diagnostics: dict | None) -> None:
        self.eq_state.set_auto_eq_diagnostics(diagnostics)

    def get_settings(self) -> dict:
        return self.eq_state.get_settings()

    def get_eq_settings(self) -> EQSettings:
        return self.eq_state.get_eq_settings()

    def set_settings(self, settings: dict) -> None:
        self.eq_state.set_settings(settings)
