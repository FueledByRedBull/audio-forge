"""Qt Quick view of the main window.

The widget panels stay the controllers: they own every value, range, tooltip
and handler. This module keeps the widget shell alive off screen and exposes
its controls to QML through proxies, so both views drive the same code.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import (
    Property,
    QCoreApplication,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QCursor, QMouseEvent, QPainter, QPen, QRegion
from PySide6.QtQml import qmlRegisterType
from PySide6.QtQuick import QQuickPaintedItem
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QWidget,
)

from .components import Card, Glyph, ToggleSwitch, icon_font, plain_label
from .theme import (
    PALETTE,
    RADIUS_CARD,
    RADIUS_CONTROL,
    prefers_reduced_motion,
    qcolor,
)

QML_DIR = Path(__file__).parent / "qml"
POLL_INTERVAL_MS = 100
# Level history behind the noise suppression trace: 6 seconds at 20 Hz.
TRACE_INTERVAL_MS = 50
TRACE_POINTS = 120
MINIMUM_SCENE_SIZE = (900, 600)
QUICK_VIEW_UNAVAILABLE = "The Qt Quick view could not start; using the classic view."
_LOG = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "text": "",
    "toolTip": "",
    "name": "",
    "enabled": True,
    "shown": True,
    "value": 0.0,
    "minimum": 0.0,
    "maximum": 1.0,
    "step": 0.0,
    "checked": False,
    "items": [],
    "index": 0,
    "state": "",
    "data": {},
}


class WidgetProxy(QObject):
    """What QML sees of one widget or action: its state and its user gestures."""

    changed = Signal()
    itemsChanged = Signal()
    repaintRequested = Signal()

    def __init__(
        self,
        target: QWidget | QAction,
        parent: QObject,
        *,
        companion: QSlider | None = None,
        data: Callable[[], dict] | None = None,
    ):
        super().__init__(parent)
        self.target = target
        self._companion = companion
        # Structured values the scene needs but the widget only shows as text.
        self._data = data
        self._state = self._read()

    def _read(self) -> dict[str, Any]:
        target = self.target
        state = dict(_DEFAULTS)
        if self._data is not None:
            state["data"] = self._data()
        if isinstance(target, QAction):
            state.update(
                text=plain_label(target.text()),
                toolTip=target.toolTip(),
                enabled=target.isEnabled(),
                shown=target.isVisible(),
                checked=target.isChecked(),
            )
            return state
        state.update(
            toolTip=target.toolTip(),
            name=target.accessibleName(),
            enabled=target.isEnabled(),
            shown=not target.isHidden(),
        )
        if isinstance(target, (QSlider, QSpinBox, QDoubleSpinBox)):
            state.update(
                value=float(target.value()),
                minimum=float(target.minimum()),
                maximum=float(target.maximum()),
                step=float(target.singleStep()),
            )
            if isinstance(target, QAbstractSpinBox):
                state["text"] = target.text()
        elif isinstance(target, QComboBox):
            state.update(
                items=[target.itemText(i) for i in range(target.count())],
                index=target.currentIndex(),
                text=target.currentText(),
            )
        elif isinstance(target, QStackedWidget):
            state["index"] = target.currentIndex()
        elif isinstance(target, QAbstractButton):
            state.update(text=plain_label(target.text()), checked=target.isChecked())
        elif isinstance(target, QLabel):
            state.update(
                text=target.text(), state=str(target.property("health_state") or "")
            )
        elif isinstance(target, QStatusBar):
            state["text"] = target.currentMessage()
        return state

    def watch_repaints(self) -> None:
        """Emit ``repaintRequested`` whenever the widget asks to be redrawn.

        An off-screen widget gets no paint events, so the request itself
        is the only signal that its picture changed.
        """

        target = self.target
        if isinstance(target, QWidget) and "update" not in target.__dict__:
            original = target.update
            # WidgetItem pins the widget to the item's size; kept for unwatch.
            self._size_limits = (target.minimumSize(), target.maximumSize())

            def update(*args) -> None:
                original(*args)
                self.repaintRequested.emit()

            target.update = update  # type: ignore[method-assign]

    def unwatch_repaints(self) -> None:
        target = self.target
        if target.__dict__.pop("update", None) is not None and isinstance(target, QWidget):
            target.setMinimumSize(self._size_limits[0])
            target.setMaximumSize(self._size_limits[1])

    def refresh(self) -> None:
        state = self._read()
        if state != self._state:
            items_changed = state["items"] != self._state["items"]
            self._state = state
            if items_changed:
                self.itemsChanged.emit()
            self.changed.emit()

    text = Property(str, lambda self: self._state["text"], notify=changed)
    toolTip = Property(str, lambda self: self._state["toolTip"], notify=changed)
    name = Property(str, lambda self: self._state["name"], notify=changed)
    enabled = Property(bool, lambda self: self._state["enabled"], notify=changed)
    shown = Property(bool, lambda self: self._state["shown"], notify=changed)
    value = Property(float, lambda self: self._state["value"], notify=changed)
    minimum = Property(float, lambda self: self._state["minimum"], notify=changed)
    maximum = Property(float, lambda self: self._state["maximum"], notify=changed)
    step = Property(float, lambda self: self._state["step"], notify=changed)
    checked = Property(bool, lambda self: self._state["checked"], notify=changed)
    items = Property(list, lambda self: self._state["items"], notify=itemsChanged)
    index = Property(int, lambda self: self._state["index"], notify=changed)
    state = Property(str, lambda self: self._state["state"], notify=changed)
    data = Property(dict, lambda self: self._state["data"], notify=changed)

    @Slot(float)
    def setValue(self, value: float) -> None:
        target = self.target
        if isinstance(target, QDoubleSpinBox):
            target.setValue(value)
        elif isinstance(target, (QSlider, QSpinBox)):
            target.setValue(round(value))
        self.refresh()
        # The scene control already moved; make it read back even when the
        # widget clamped or refused the value and nothing changed here.
        self.changed.emit()

    @Slot(str)
    def setText(self, text: str) -> None:
        """Accept a typed number, with or without the unit the field shows."""

        target = self.target
        if not isinstance(target, (QSpinBox, QDoubleSpinBox)):
            return
        number = text.replace(target.suffix().strip(), "").replace(",", ".").strip()
        try:
            self.setValue(float(number))
        except ValueError:
            pass
        # The field shows what was typed until it is told to read again.
        self.changed.emit()
        self.released()

    @Slot()
    def released(self) -> None:
        """End of a slider drag; the panels flush their rate limiters on it."""

        slider = self._companion if self._companion is not None else self.target
        if isinstance(slider, QSlider):
            slider.sliderReleased.emit()

    @Slot(int)
    def setIndex(self, index: int) -> None:
        target = self.target
        if isinstance(target, (QComboBox, QStackedWidget)):
            target.setCurrentIndex(index)
        self.refresh()
        self.changed.emit()

    @Slot()
    def click(self) -> None:
        self._activate(QCursor.pos())

    @Slot(float, float)
    def clickAt(self, x: float, y: float) -> None:
        """Click from the scene; a menu opens at this global position."""

        self._activate(QPoint(round(x), round(y)))

    def _activate(self, menu_position: QPoint) -> None:
        target = self.target
        menu = target.menu() if isinstance(target, (QPushButton, QToolButton)) else None
        if menu is not None:
            if self._state["enabled"]:
                menu.popup(menu_position)
        elif isinstance(target, QAction):
            target.trigger()
        elif isinstance(target, QAbstractButton):
            target.click()
        self.refresh()


class WidgetItem(QQuickPaintedItem):
    """Paints an off-screen widget in the scene and passes input on to it.

    Used for the widgets that draw themselves (EQ graph, meters), so their
    paint and pointer code is shared with the widget view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proxy: WidgetProxy | None = None
        self._widget: QWidget | None = None
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFillColor(qcolor(PALETTE.app_surface, alpha=0))
        self.visibleChanged.connect(self.update)

    def _set_proxy(self, proxy: WidgetProxy | None) -> None:
        self._proxy = proxy
        self._widget = None
        if proxy is not None and isinstance(proxy.target, QWidget):
            self._widget = proxy.target
            proxy.watch_repaints()
            proxy.repaintRequested.connect(self._request_paint)
            proxy.changed.connect(self._request_paint)
        self.update()

    proxy = Property(QObject, lambda self: self._proxy, _set_proxy)

    def _request_paint(self) -> None:
        # Items on another page are brought up to date when they reappear.
        if self.isVisible():
            self.update()

    def paint(self, painter: QPainter) -> None:
        widget = self._widget
        if widget is None:
            return
        size = QSize(max(1, round(self.width())), max(1, round(self.height())))
        if widget.size() != size:
            widget.setFixedSize(size)
        widget.render(painter, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)

    def _send_mouse(self, event, kind: QEvent.Type, button, buttons) -> None:
        if self._widget is not None:
            QCoreApplication.sendEvent(
                self._widget,
                QMouseEvent(
                    kind,
                    event.position(),
                    event.globalPosition(),
                    button,
                    buttons,
                    event.modifiers(),
                ),
            )

    def mousePressEvent(self, event) -> None:
        self.forceActiveFocus()
        self._send_mouse(event, event.type(), event.button(), event.buttons())

    def mouseMoveEvent(self, event) -> None:
        self._send_mouse(event, event.type(), event.button(), event.buttons())

    def mouseReleaseEvent(self, event) -> None:
        self._send_mouse(event, event.type(), event.button(), event.buttons())

    def hoverMoveEvent(self, event) -> None:
        self._send_mouse(
            event,
            QEvent.Type.MouseMove,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
        )

    def keyPressEvent(self, event) -> None:
        if self._widget is not None:
            QCoreApplication.sendEvent(self._widget, event)


class LevelTrace(QQuickPaintedItem):
    """Input (muted) and output (accent) level over the last few seconds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: list[list[float]] = []
        self.setAntialiasing(True)
        self.visibleChanged.connect(self.update)

    def _set_points(self, points: list) -> None:
        self._points = points
        if self.isVisible():
            self.update()

    points = Property(list, lambda self: self._points, _set_points)

    def paint(self, painter: QPainter) -> None:
        width, height = self.width(), self.height()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(PALETTE.data_surface))
        painter.drawRoundedRect(self.boundingRect(), RADIUS_CONTROL, RADIUS_CONTROL)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        first = TRACE_POINTS - len(self._points)
        for line, color in enumerate((PALETTE.text_muted, PALETTE.accent)):
            painter.setPen(QPen(qcolor(color), 1.5))
            painter.drawPolyline(
                [
                    QPointF(
                        width * (first + index) / (TRACE_POINTS - 1),
                        # -60 dB at the bottom, 0 dB at the top.
                        4 + (height - 8) * (1 - min(1.0, max(0.0, (point[line] + 60) / 60))),
                    )
                    for index, point in enumerate(self._points)
                ]
            )


class QuickBridge(QObject):
    """The ``ui`` object in QML: a tree of proxies plus a few window actions."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._proxies: list[WidgetProxy] = []
        self._model = self._build_model(window)
        # ponytail: one timer compares every proxy with its widget. Widgets
        # have no change signal for text, enabled or tooltip; connect per
        # property signals if this ever shows up in a profile.
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self._levels: deque[list[float]] = deque(maxlen=TRACE_POINTS)
        self._trace_timer = QTimer(self)
        self._trace_timer.setInterval(TRACE_INTERVAL_MS)
        self._trace_timer.timeout.connect(self._sample_levels)
        if not prefers_reduced_motion():
            self._trace_timer.start()

    model = Property(dict, lambda self: self._model, constant=True)

    def release(self) -> None:
        """Give the widgets back to the widget view and go away."""

        for proxy in self._proxies:
            proxy.unwatch_repaints()
        self.deleteLater()

    def refresh(self) -> None:
        if self._window.isVisible() and not self._window.isMinimized():
            for proxy in self._proxies:
                proxy.refresh()

    levelsChanged = Signal()
    levels = Property(list, lambda self: list(self._levels), notify=levelsChanged)

    def _sample_levels(self) -> None:
        """Record input and output level from what the meters already show."""

        window = self._window
        meters = (window.input_meter, window.output_meter)
        live = (
            window.isVisible()
            and not window.isMinimized()
            and all(meter.measurement_available for meter in meters)
        )
        if live:
            self._levels.append([float(meter.rms_db) for meter in meters])
        elif self._levels and not any(m.measurement_available for m in meters):
            self._levels.clear()
        else:
            return
        self.levelsChanged.emit()

    @Slot(int)
    def stepBand(self, direction: int) -> None:
        self._window.eq_panel._step_band(direction)

    @Slot()
    def resetDrops(self) -> None:
        self._window._reset_dropped_samples()

    @Slot()
    def exportDiagnostics(self) -> None:
        self._window._export_diagnostics()

    def _proxy(self, target, **kwargs) -> WidgetProxy:
        proxy = WidgetProxy(target, self, **kwargs)
        self._proxies.append(proxy)
        return proxy

    def _field(self, label: str, target, slider=None, display=None) -> dict:
        return {
            "kind": "field" if slider is not None or isinstance(target, QSlider) else "number",
            "label": label,
            "proxy": self._proxy(target, companion=slider),
            "display": self._proxy(display) if display is not None else None,
        }

    def _row(self, kind: str, target, label: str = "") -> dict:
        return {"kind": kind, "label": label, "proxy": self._proxy(target)}

    def _card(self, card: Card, rows: list[dict], advanced: list[dict], **extra) -> dict:
        help_button, menu_button = card.help_button, card.menu_button
        return {
            "trace": False,
            "alert": None,
            "title": card.title_label.text(),
            "help": help_button.toolTip() if help_button is not None else "",
            "toggle": self._proxy(card.switch) if card.switch is not None else None,
            "menu": self._proxy(menu_button) if menu_button is not None else None,
            "rows": rows,
            "advanced": advanced,
            **extra,
        }

    def _build_model(self, w) -> dict:
        field, row, card = self._field, self._row, self._card

        def health(title: str, label: QLabel, *, coded: bool = False) -> dict:
            # `coded` rows carry counter tokens rather than a readable value.
            return {"title": title, "coded": coded, "proxy": self._proxy(label)}
        gate, deesser, comp, eq = (
            w.gate_panel,
            w.deesser_panel,
            w.compressor_panel,
            w.eq_panel,
        )
        (gate_card,) = gate.findChildren(Card)
        (deesser_card,) = deesser.findChildren(Card)
        comp_card, limiter_card = comp.findChildren(Card)
        (eq_card,) = eq.findChildren(Card)

        stages = [
            card(
                w.noise_suppression_group,
                [
                    row("combo", w.model_combo, "Backend"),
                    field("Strength", w.strength_slider, display=w.strength_label),
                    row("text", w.rnnoise_latency_label),
                ],
                [],
                # This card also shows the level history, and the backend
                # state while that is unhealthy.
                trace=True,
                alert=self._proxy(w.backend_diag_label),
            ),
            card(
                gate_card,
                [
                    field("Threshold", gate.threshold_spinbox, gate.threshold_slider),
                    row("toggle", gate.auto_threshold_checkbox, "Auto threshold"),
                    row("text", gate.threshold_status_label),
                ],
                [
                    field("Attack", gate.attack_spinbox),
                    field("Release", gate.release_spinbox),
                    row("combo", gate.gate_mode_combo, "Gate mode"),
                    field("VAD threshold", gate.vad_threshold_spinbox, gate.vad_threshold_slider),
                    field("Hold time", gate.vad_hold_spinbox),
                    field("VAD pre-gain", gate.vad_pre_gain_spinbox, gate.vad_pre_gain_slider),
                    field("Margin", gate.margin_spinbox, gate.margin_slider),
                    row("text", gate.noise_floor_label),
                    row("meter", gate.confidence_meter, "Confidence"),
                    row("text", gate.vad_info_label),
                ],
            ),
            card(
                deesser_card,
                [
                    row("toggle", deesser.auto_checkbox, "Auto"),
                    field("Amount", deesser.auto_amount_spinbox, deesser.auto_amount_slider),
                    row("meter", deesser.gr_meter, "Reduction"),
                ],
                [
                    field("Low cut", deesser.low_cut_spinbox, deesser.low_cut_slider),
                    field("High cut", deesser.high_cut_spinbox, deesser.high_cut_slider),
                    field("Threshold", deesser.threshold_spinbox, deesser.threshold_slider),
                    field("Ratio", deesser.ratio_spinbox, deesser.ratio_slider),
                    field("Attack", deesser.attack_spinbox),
                    field("Release", deesser.release_spinbox),
                    field(
                        "Max reduction",
                        deesser.max_reduction_spinbox,
                        deesser.max_reduction_slider,
                    ),
                ],
            ),
            card(
                comp_card,
                [
                    field("Threshold", comp.threshold_spinbox, comp.threshold_slider),
                    field("Ratio", comp.ratio_spinbox, comp.ratio_slider),
                    row("meter", comp.gr_meter, "Reduction"),
                ],
                [
                    field("Attack", comp.attack_spinbox),
                    field("Release", comp.release_spinbox),
                    field("Makeup gain", comp.makeup_spinbox, comp.makeup_slider),
                    row("toggle", comp.adaptive_release_checkbox, "Adaptive release"),
                    field("Base release", comp.base_release_spinbox),
                    row("text", comp.current_release_label, "Current release"),
                    row("toggle", comp.sidechain_highpass_checkbox, "Sidechain high-pass"),
                    row("toggle", comp.auto_makeup_checkbox, "Auto makeup gain"),
                    field("Target LUFS", comp.target_lufs_spinbox),
                    row("text", comp.current_lufs_label, "Current LUFS"),
                    row("text", comp.current_makeup_gain_label, "Auto gain"),
                ],
            ),
            card(
                limiter_card,
                [field("Ceiling", comp.ceiling_spinbox, comp.ceiling_slider)],
                [
                    row("toggle", comp.careful_output_checkbox, "Careful output mode"),
                    field("Release", comp.limiter_release_spinbox),
                ],
            ),
        ]

        bands = [
            {
                "frequencyLabel": self._proxy(band.freq_label),
                "enabled": self._proxy(band.band_enabled_checkbox),
                "type": self._proxy(band.filter_type_combo),
                "gain": self._proxy(band.slider),
                "gainLabel": self._proxy(band.gain_label),
                "frequency": self._proxy(band.frequency_spinbox),
                "q": self._proxy(band.q_spinbox),
                "slope": self._proxy(band.slope_combo),
            }
            for band in eq.band_sliders
        ]

        settings_cards = []
        for settings_card in w.settings_page.findChildren(Card)[1:]:
            rows = [
                row(
                    "toggle"
                    if isinstance(control, ToggleSwitch)
                    else "menu"
                    if control.menu() is not None
                    else "action",
                    control,
                )
                for control in settings_card.findChildren(QAbstractButton)
                if isinstance(control, (ToggleSwitch, QPushButton))
            ]
            settings_cards.append({"title": settings_card.title_label.text(), "rows": rows})

        font = icon_font()
        return {
            "theme": {
                **{k: v for k, v in asdict(PALETTE).items() if isinstance(v, str)},
                "bandColors": list(PALETTE.eq_band_colors),
                "iconFont": font.family() if font is not None else "",
                "radiusCard": RADIUS_CARD,
                "radiusControl": RADIUS_CONTROL,
                "motion": 0 if prefers_reduced_motion() else 140,
            },
            "glyphs": {
                "mic": Glyph.MICROPHONE,
                "health": Glyph.HEALTH,
                "settings": Glyph.SETTINGS,
                "more": Glyph.MORE,
                "help": Glyph.HELP,
                "previous": Glyph.PREVIOUS,
                "next": Glyph.NEXT,
                "expand": Glyph.EXPAND,
                "refresh": Glyph.REFRESH,
                "undo": Glyph.UNDO,
                "redo": Glyph.REDO,
            },
            "page": self._proxy(w.page_stack),
            "healthPage": w.HEALTH_PAGE_INDEX,
            "banners": [
                self._proxy(w.device_warning_banner),
                self._proxy(w.config_warning_banner),
            ],
            "top": {
                "input": self._proxy(w.input_combo),
                "output": self._proxy(w.output_combo),
                "refresh": self._proxy(w.refresh_btn),
                "mode": self._proxy(w.processing_mode_combo),
                "mute": self._proxy(w.user_mute_checkbox),
                "start": self._proxy(w.start_btn),
                "stop": self._proxy(w.stop_btn),
            },
            "presets": {
                "status": self._proxy(
                    w.preset_status_label,
                    data=lambda: {
                        "name": w.current_preset_name,
                        "modified": bool(w.current_preset_modified),
                    },
                ),
                "menu": self._proxy(w.presets_button),
                "undo": self._proxy(w._undo_auto_eq_button),
                "redo": self._proxy(w.redo_button),
                "testSound": self._proxy(w.test_sound_button),
                "autoEq": self._proxy(w.auto_eq_button),
                "voiceSetup": self._proxy(w.auto_voice_setup_button),
            },
            "eq": {
                **card(eq_card, [], []),
                "tone": self._proxy(eq.tone_preset_button),
                "curve": self._proxy(eq.curve_widget),
                "layers": self._proxy(eq.layer_status_label),
                "diagnostics": self._proxy(
                    eq.auto_eq_diag_label,
                    data=lambda: {"calibrated": eq._auto_eq_diagnostics is not None},
                ),
                "band": self._proxy(eq.band_stack),
                "bands": bands,
            },
            "stages": stages,
            "meters": [self._proxy(w.input_meter), self._proxy(w.output_meter)],
            "status": {
                "message": self._proxy(w.status_bar),
                "transmission": self._proxy(w.transmission_status_label),
                "calibration": self._proxy(w.calibration_status_label),
                "health": self._proxy(w.health_summary_label),
            },
            "health": {
                "advice": self._proxy(w.health_advice_label),
                "route": self._proxy(w.route_status_label),
                "signal": [
                    health("Input", w.input_health_label),
                    health("Output", w.output_health_label),
                    health("Gate", w.gate_health_label),
                ],
                "stream": [
                    health("Backend", w.backend_diag_label, coded=True),
                    health("Callbacks", w.callback_health_label),
                    health("Underruns", w.underrun_health_label),
                    health("Latency", w.latency_label),
                    health("Buffer", w.buffer_label),
                    health("Drops", w.dropped_label, coded=True),
                    health("Recovery", w.recovery_diag_label, coded=True),
                ],
            },
            "settings": {
                "input": [
                    row("combo", w.input_channel_mode_combo, "Input mode"),
                    row("combo", w.input_cleanup_mode_combo, "Cleanup"),
                ],
                "cards": settings_cards,
            },
        }


def install_quick_shell(window) -> bool:
    """Replace the window's widget shell with the QML scene.

    The widget shell moves to an off-screen window. It is shown there (never
    on the desktop) so visibility-driven code such as meter timers keeps
    running. Returns ``False`` and leaves the widget shell in place when the
    scene does not load.
    """

    qmlRegisterType(WidgetItem, "AudioForge", 1, 0, "WidgetItem")  # type: ignore[call-overload]
    qmlRegisterType(LevelTrace, "AudioForge", 1, 0, "LevelTrace")  # type: ignore[call-overload]
    try:
        bridge = QuickBridge(window)
    except Exception:
        _LOG.exception("Qt Quick view could not be prepared")
        _announce_fallback(window)
        return False
    view = QQuickWidget(window)
    view.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    view.setMinimumSize(*MINIMUM_SCENE_SIZE)
    view.setClearColor(qcolor(PALETTE.app_surface))
    view.rootContext().setContextProperty("ui", bridge)
    view.setSource(QUrl.fromLocalFile(str(QML_DIR / "Main.qml")))
    if view.status() != QQuickWidget.Status.Ready:
        _LOG.error(
            "Qt Quick view failed to load: %s",
            "; ".join(error.toString() for error in view.errors()),
        )
        view.deleteLater()
        bridge.release()
        _announce_fallback(window)
        return False

    shell = window.takeCentralWidget()
    shell.setParent(window, Qt.WindowType.Window)
    shell.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    shell.show()
    window.statusBar().hide()
    window.setCentralWidget(view)
    window.quick_bridge = bridge
    window.widget_shell = shell
    # A scene that loads can still fail to render, for example without a
    # usable graphics adapter.
    view.sceneGraphError.connect(
        lambda _error, message: restore_widget_shell(window, message)
    )
    # Left to interpreter shutdown, the scene is torn down in no fixed order
    # and the process can crash on exit; delete it as the event loop ends.
    app = QCoreApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(view.deleteLater)
    return True


def _announce_fallback(window) -> None:
    # A permanent label: transient status messages would replace the notice.
    notice = QLabel(QUICK_VIEW_UNAVAILABLE)
    notice.setAccessibleName("Interface notice")
    window.quick_view_failed = True
    window.statusBar().addPermanentWidget(notice)


def restore_widget_shell(window, reason: str) -> None:
    """Go back to the widget view and say why."""

    bridge = window.__dict__.pop("quick_bridge", None)
    if bridge is None:
        return
    _LOG.error("Qt Quick view stopped rendering: %s", reason)
    shell = window.__dict__.pop("widget_shell")
    view = window.takeCentralWidget()
    shell.hide()
    shell.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
    shell.setParent(window)
    window.setCentralWidget(shell)
    shell.show()
    window.statusBar().show()
    _announce_fallback(window)
    view.deleteLater()
    bridge.release()
