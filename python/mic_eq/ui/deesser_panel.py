"""
De-esser control panel.

Controls sibilance reduction stage placed between noise suppression and EQ.
"""

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .deesser_state import DeEsserState
from .level_meter import GainReductionMeter
from .processing_meters import ProcessingMeters
from .control_specs import (
    apply_control_presentation,
    configure_numeric_control,
    control_label,
    control_spec,
    widget_control_label,
)
from .components import Card, ToggleSwitch
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    PRIMARY_LABEL_STYLE,
    SPACING_NORMAL,
    fit_spinbox_to_contents,
)


class DeEsserPanel(QWidget):
    """De-esser parameter control panel."""

    configurationEdited = Signal(str)

    def __init__(self, processor, deesser_state: DeEsserState | None = None, meters: ProcessingMeters | None = None):
        super().__init__()
        self.processor = processor
        self._meters = meters
        self.deesser_state = deesser_state or DeEsserState(processor, self)
        self._setup_ui()
        self._connect_signals()
        if deesser_state is None:
            self.deesser_state.set_settings({})

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.enabled_checkbox = ToggleSwitch()
        self.enabled_checkbox.setChecked(False)
        apply_control_presentation(self.enabled_checkbox, "deesser", "enabled")
        card = Card(
            "De-esser",
            switch=self.enabled_checkbox,
            help_text=(
                "Reduces harsh sibilance (s, sh, t). Auto mode tracks the "
                "balance between sibilance and voice and adjusts the reduction "
                "as you speak."
            ),
        )
        grid = QGridLayout()
        card.body.addLayout(grid)
        advanced = QWidget()
        advanced_grid = QGridLayout(advanced)
        advanced_grid.setContentsMargins(0, 0, 0, 0)
        card.add_advanced(advanced)
        for section in (grid, advanced_grid):
            section.setSpacing(SPACING_NORMAL)
            section.setColumnStretch(1, 1)
            section.setColumnMinimumWidth(0, 95)

        self.auto_checkbox = ToggleSwitch(control_label("deesser", "auto_enabled"))
        self.auto_checkbox.setChecked(True)
        apply_control_presentation(self.auto_checkbox, "deesser", "auto_enabled")
        grid.addWidget(self.auto_checkbox, 1, 0, 1, 2)

        amount_layout = QHBoxLayout()
        self.auto_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self.auto_amount_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.auto_amount_slider.setTickInterval(10)
        amount_layout.addWidget(self.auto_amount_slider)

        self.auto_amount_spinbox = QDoubleSpinBox()
        configure_numeric_control(
            self.auto_amount_spinbox, "deesser", "auto_amount",
            slider=self.auto_amount_slider, slider_scale=100,
        )
        self.auto_amount_slider.setValue(50)
        self.auto_amount_spinbox.setValue(0.5)
        fit_spinbox_to_contents(self.auto_amount_spinbox)
        amount_layout.addWidget(self.auto_amount_spinbox)

        amount_label = QLabel(f"{widget_control_label('deesser', 'auto_amount')}:")
        amount_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        grid.addWidget(amount_label, 2, 0)
        grid.addLayout(amount_layout, 2, 1)

        low_layout = QHBoxLayout()
        self.low_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_cut_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.low_cut_slider.setTickInterval(1000)
        low_layout.addWidget(self.low_cut_slider)

        self.low_cut_spinbox = QDoubleSpinBox()
        configure_numeric_control(
            self.low_cut_spinbox, "deesser", "low_cut_hz", slider=self.low_cut_slider,
        )
        self.low_cut_slider.setValue(4000)
        self.low_cut_spinbox.setValue(4000.0)
        fit_spinbox_to_contents(self.low_cut_spinbox)
        low_layout.addWidget(self.low_cut_spinbox)

        low_label = QLabel(f"{widget_control_label('deesser', 'low_cut_hz')}:")
        low_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(low_label, 3, 0)
        advanced_grid.addLayout(low_layout, 3, 1)

        high_layout = QHBoxLayout()
        self.high_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_cut_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.high_cut_slider.setTickInterval(1000)
        high_layout.addWidget(self.high_cut_slider)

        self.high_cut_spinbox = QDoubleSpinBox()
        configure_numeric_control(
            self.high_cut_spinbox, "deesser", "high_cut_hz", slider=self.high_cut_slider,
        )
        self.high_cut_slider.setValue(11000)
        self.high_cut_spinbox.setValue(11000.0)
        fit_spinbox_to_contents(self.high_cut_spinbox)
        high_layout.addWidget(self.high_cut_spinbox)

        high_label = QLabel(f"{widget_control_label('deesser', 'high_cut_hz')}:")
        high_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(high_label, 4, 0)
        advanced_grid.addLayout(high_layout, 4, 1)

        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(6)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_spinbox = QDoubleSpinBox()
        configure_numeric_control(
            self.threshold_spinbox, "deesser", "threshold_db",
            slider=self.threshold_slider,
        )
        self.threshold_slider.setValue(-28)
        self.threshold_spinbox.setValue(-28.0)
        fit_spinbox_to_contents(self.threshold_spinbox)
        threshold_layout.addWidget(self.threshold_spinbox)

        threshold_label = QLabel(f"{widget_control_label('deesser', 'threshold_db')}:")
        threshold_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(threshold_label, 5, 0)
        advanced_grid.addLayout(threshold_layout, 5, 1)

        ratio_layout = QHBoxLayout()
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ratio_slider.setTickInterval(20)
        ratio_layout.addWidget(self.ratio_slider)

        self.ratio_spinbox = QDoubleSpinBox()
        configure_numeric_control(
            self.ratio_spinbox, "deesser", "ratio",
            slider=self.ratio_slider, slider_scale=10,
        )
        self.ratio_slider.setValue(40)
        self.ratio_spinbox.setValue(4.0)
        fit_spinbox_to_contents(self.ratio_spinbox)
        ratio_layout.addWidget(self.ratio_spinbox)

        ratio_label = QLabel(f"{widget_control_label('deesser', 'ratio')}:")
        ratio_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(ratio_label, 6, 0)
        advanced_grid.addLayout(ratio_layout, 6, 1)

        self.attack_spinbox = QDoubleSpinBox()
        configure_numeric_control(self.attack_spinbox, "deesser", "attack_ms")
        self.attack_spinbox.setValue(2.0)
        fit_spinbox_to_contents(self.attack_spinbox)
        attack_label = QLabel(f"{widget_control_label('deesser', 'attack_ms')}:")
        attack_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(attack_label, 7, 0)
        advanced_grid.addWidget(self.attack_spinbox, 7, 1)

        self.release_spinbox = QDoubleSpinBox()
        configure_numeric_control(self.release_spinbox, "deesser", "release_ms")
        self.release_spinbox.setValue(80.0)
        fit_spinbox_to_contents(self.release_spinbox)
        release_label = QLabel(f"{widget_control_label('deesser', 'release_ms')}:")
        release_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(release_label, 8, 0)
        advanced_grid.addWidget(self.release_spinbox, 8, 1)

        max_red_layout = QHBoxLayout()
        self.max_reduction_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_reduction_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.max_reduction_slider.setTickInterval(20)
        max_red_layout.addWidget(self.max_reduction_slider)

        self.max_reduction_spinbox = QDoubleSpinBox()
        configure_numeric_control(
            self.max_reduction_spinbox, "deesser", "max_reduction_db",
            slider=self.max_reduction_slider, slider_scale=10,
        )
        self.max_reduction_slider.setValue(60)
        self.max_reduction_spinbox.setValue(6.0)
        fit_spinbox_to_contents(self.max_reduction_spinbox)
        max_red_layout.addWidget(self.max_reduction_spinbox)

        max_red_label = QLabel(f"{widget_control_label('deesser', 'max_reduction_db')}:")
        max_red_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        advanced_grid.addWidget(max_red_label, 9, 0)
        advanced_grid.addLayout(max_red_layout, 9, 1)

        self.gr_meter = self._meters.deesser_gr if self._meters else GainReductionMeter()
        grid.addWidget(self.gr_meter, 10, 0, 1, 2)

        layout.addWidget(card)

        bind_label(
            amount_label,
            self.auto_amount_spinbox,
            name=control_spec("deesser", "auto_amount").accessible_name,
        )
        bind_label(low_label, self.low_cut_spinbox, name=control_spec("deesser", "low_cut_hz").accessible_name)
        bind_label(high_label, self.high_cut_spinbox, name=control_spec("deesser", "high_cut_hz").accessible_name)
        bind_label(
            threshold_label,
            self.threshold_spinbox,
            name=control_spec("deesser", "threshold_db").accessible_name,
        )
        bind_label(ratio_label, self.ratio_spinbox, name=control_spec("deesser", "ratio").accessible_name)
        bind_label(attack_label, self.attack_spinbox, name=control_spec("deesser", "attack_ms").accessible_name)
        bind_label(
            release_label,
            self.release_spinbox,
            name=control_spec("deesser", "release_ms").accessible_name,
        )
        bind_label(
            max_red_label,
            self.max_reduction_spinbox,
            name=control_spec("deesser", "max_reduction_db").accessible_name,
        )
        set_accessible_group(
            (
                (self.enabled_checkbox, control_spec("deesser", "enabled").accessible_name, None),
                (self.auto_checkbox, control_spec("deesser", "auto_enabled").accessible_name, None),
                (self.auto_amount_slider, control_spec("deesser", "auto_amount").accessible_name, None),
                (self.low_cut_slider, control_spec("deesser", "low_cut_hz").accessible_name, None),
                (self.high_cut_slider, control_spec("deesser", "high_cut_hz").accessible_name, None),
                (self.threshold_slider, control_spec("deesser", "threshold_db").accessible_name, None),
                (self.ratio_slider, control_spec("deesser", "ratio").accessible_name, None),
                (self.max_reduction_slider, control_spec("deesser", "max_reduction_db").accessible_name, None),
                (self.gr_meter, "De-esser gain reduction", None),
            )
        )

    def _connect_signals(self):
        self.enabled_checkbox.toggled.connect(
            lambda value: self.deesser_state.set_value("enabled", value)
        )
        self.auto_checkbox.toggled.connect(
            lambda value: self.deesser_state.set_value("auto_enabled", value)
        )
        self._numeric_controls: dict[
            str, tuple[QDoubleSpinBox, QSlider | None, float]
        ] = {
            "auto_amount": (self.auto_amount_spinbox, self.auto_amount_slider, 100),
            "low_cut_hz": (self.low_cut_spinbox, self.low_cut_slider, 1),
            "high_cut_hz": (self.high_cut_spinbox, self.high_cut_slider, 1),
            "threshold_db": (self.threshold_spinbox, self.threshold_slider, 1),
            "ratio": (self.ratio_spinbox, self.ratio_slider, 10),
            "attack_ms": (self.attack_spinbox, None, 1),
            "release_ms": (self.release_spinbox, None, 1),
            "max_reduction_db": (
                self.max_reduction_spinbox, self.max_reduction_slider, 10,
            ),
        }
        for key, (spinbox, slider, scale) in self._numeric_controls.items():
            spinbox.valueChanged.connect(
                lambda value, key=key: self.deesser_state.set_value(key, value)
            )
            if slider is not None:
                slider.valueChanged.connect(
                    lambda value, key=key, scale=scale: self.deesser_state.set_value(
                        key, value / scale
                    )
                )
                slider.sliderReleased.connect(self.deesser_state.flush)
        self.deesser_state.changed.connect(self._render_settings)
        self.deesser_state.configurationEdited.connect(self.configurationEdited.emit)
        self._render_settings()

    def _render_settings(self) -> None:
        settings = self.deesser_state.get_settings()
        for control, key in (
            (self.enabled_checkbox, "enabled"),
            (self.auto_checkbox, "auto_enabled"),
        ):
            with QSignalBlocker(control):
                control.setChecked(bool(settings[key]))
        for key, (spinbox, slider, scale) in self._numeric_controls.items():
            with QSignalBlocker(spinbox):
                spinbox.setValue(float(settings[key]))
            spinbox.setEnabled(self.deesser_state.control_enabled(key))
            if slider is not None:
                with QSignalBlocker(slider):
                    slider.setValue(int(settings[key] * scale))
                slider.setEnabled(self.deesser_state.control_enabled(key))

    def apply_settings_synchronously(self, settings: dict) -> None:
        """Apply a bulk config inline and cancel any superseded slider edit."""
        self.deesser_state.set_settings(settings)

    def update_gain_reduction(self, gr_db: float | None):
        self.gr_meter.set_gain_reduction(gr_db)

    def get_settings(self) -> dict:
        return self.deesser_state.get_settings()

    def set_settings(self, settings: dict):
        self.deesser_state.set_settings(settings)
