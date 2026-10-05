"""Qt Quick view of the main window.

Migrated settings use shared application state; remaining controls use widget
proxies. The widget shell stays available for rendering failure recovery.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import (
    Property,
    QCoreApplication,
    QEvent,
    QLocale,
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
from ..config_parts.validation import VALIDATION_RANGES
from .limiter_state import LimiterState
from .deesser_state import DeEsserState
from .compressor_state import COMPRESSOR_RANGES, CompressorState
from .gate_state import GateState
from .eq_controls import EQControl
from .noise_controls import NoiseControl
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


class SettingControl(QObject):
    """Present shared processing state to QML without reading a QWidget."""

    changed = Signal()

    def __init__(
        self, state: LimiterState | DeEsserState | CompressorState | GateState,
        section: str, key: str, parent: QObject,
    ):
        super().__init__(parent)
        self._state = state
        self._key = key
        self._section = section
        self._name, self._tip, self._step, self._suffix = {
            "limiter": {
            "enabled": (
                "Enable limiter",
                "Prevents signal from exceeding ceiling level.\n"
                "Acts as a safety net to prevent clipping.", 0.0, "",
            ),
            "careful_output_enabled": (
                "Enable careful output mode",
                "Adds conservative output headroom by limiting the effective ceiling to -1.5 dB.",
                0.0, "",
            ),
            "ceiling_db": (
                "Limiter ceiling", "Maximum output level (brick-wall ceiling)", 0.1, " dB",
            ),
            "release_ms": (
                "Limiter release time", "How fast the limiter recovers", 5.0, " ms",
            ),
            },
            "deesser": {
                "enabled": (
                    "Enable de-esser", "Reduces harsh sibilance (s, sh, t) using dynamic attenuation.", 0.0, "",
                ),
                "auto_enabled": (
                    "Enable automatic de-esser",
                    "Learns average sibilance and applies dynamic reduction automatically.",
                    0.0, "",
                ),
                "auto_amount": ("Automatic de-esser amount", "", 0.05, ""),
                "low_cut_hz": ("De-esser low cutoff", "", 100.0, " Hz"),
                "high_cut_hz": ("De-esser high cutoff", "", 100.0, " Hz"),
                "threshold_db": ("De-esser threshold", "", 1.0, " dB"),
                "ratio": ("De-esser ratio", "", 0.5, ":1"),
                "attack_ms": ("De-esser attack time", "", 0.1, " ms"),
                "release_ms": ("De-esser release time", "", 5.0, " ms"),
                "max_reduction_db": ("De-esser maximum reduction", "", 0.5, " dB"),
            },
            "compressor": {
                "enabled": (
                    "Enable compressor", "Reduces dynamic range by attenuating loud signals.\n"
                    "Helps maintain consistent volume levels.", 0.0, "",
                ),
                "threshold_db": ("Compressor threshold", "Level above which compression begins", 1.0, " dB"),
                "ratio": ("Compressor ratio", "Compression ratio (higher = more compression)", 0.5, ":1"),
                "attack_ms": ("Compressor attack time", "How fast the compressor responds to loud signals", 1.0, " ms"),
                "release_ms": ("Compressor release time", "How fast the compressor recovers after loud signals", 10.0, " ms"),
                "makeup_gain_db": ("Compressor makeup gain", "Gain added after compression to restore volume", 0.5, " dB"),
                "adaptive_release": (
                    "Enable adaptive release", "Release time adapts based on signal dynamics.\n"
                    "Scales from 50ms to 400ms based on sustained overage.\n"
                    "Longer release for consistent loud signals, shorter for transients.", 0.0, "",
                ),
                "base_release_ms": ("Adaptive compressor base release", "Base release time when adaptive mode is enabled", 5.0, " ms"),
                "sidechain_highpass_enabled": (
                    "Enable compressor sidechain high-pass", "Ignores low-frequency plosives and rumble in the compressor detector without filtering the audio.", 0.0, "",
                ),
                "auto_makeup_enabled": (
                    "Enable automatic makeup gain", "Automatically adjust makeup gain from post-compression EBU R128 loudness measurement.\n"
                    "Maintains post-compressor output level relative to target LUFS.\n"
                    "Uses the selected Target LUFS value.", 0.0, "",
                ),
                "target_lufs": ("Automatic makeup target loudness", "Target loudness level (-24 to -12 LUFS)", 1.0, " LUFS"),
            },
            "gate": {
                "enabled": ("Enable noise gate", "Reduces gain when signal falls below threshold.\nHelps eliminate background noise during silence.", 0.0, ""),
                "threshold_db": ("Gate manual threshold", "Signal level below which gate closes", 1.0, " dB"),
                "attack_ms": ("Gate attack time", "Time for gate to open when signal exceeds threshold", 1.0, " ms"),
                "release_ms": ("Gate release time", "Time for gate to close when signal drops below threshold", 10.0, " ms"),
                "gate_mode": ("Gate operating mode", "Threshold Only: Traditional gate using level threshold\nVAD Assisted: Gate opens when level exceeded OR speech detected\nVAD Only: Gate opens solely based on speech probability", 0.0, ""),
                "vad_threshold": ("Voice activity threshold", "Speech probability threshold (0.3-0.7)", 0.01, ""),
                "vad_hold_time_ms": ("Voice activity hold time", "Gate hold time after speech ends (prevents chatter)", 10.0, " ms"),
                "vad_pre_gain": ("Voice activity pre-gain", "Pre-gain to boost weak signals for better VAD detection", 0.5, ""),
                "auto_threshold_enabled": ("Enable automatic gate threshold", "Automatically set the gate level threshold to noise floor + margin.", 0.0, ""),
                "gate_margin_db": ("Automatic gate margin", "Margin above noise floor for gate threshold (0-20 dB)", 1.0, " dB"),
            },
        }[section][key]
        ranges = COMPRESSOR_RANGES if section == "compressor" else {
            name: bounds for name, bounds in VALIDATION_RANGES[section].items()
            if isinstance(bounds[0], (int, float))
        }
        self._choices = ["Threshold Only", "VAD Assisted", "VAD Only"] if key == "gate_mode" else []
        self._numeric = key in ranges and not self._choices
        self._minimum, self._maximum = ranges.get(key, (0.0, 1.0))
        self._decimals = 1 if key == "vad_pre_gain" else 2
        self._locale = QLocale()
        self._locale.setNumberOptions(
            self._locale.numberOptions() | QLocale.NumberOption.OmitGroupSeparator,
        )
        state.changed.connect(self.changed)

    def _presentation(self, key: str, default: str) -> str:
        if isinstance(self._state, GateState) and self._key == "auto_threshold_enabled":
            return self._state.presentation()["auto_threshold_" + key]
        return default

    name = Property(str, lambda self: self._presentation("name", self._name), notify=changed)
    toolTip = Property(str, lambda self: self._presentation("tooltip", self._tip), notify=changed)
    step = Property(float, lambda self: self._step, constant=True)
    minimum = Property(float, lambda self: self._minimum, constant=True)
    maximum = Property(float, lambda self: self._maximum, constant=True)
    enabled = Property(bool, lambda self: self._state.control_enabled(self._key), notify=changed)
    shown = Property(bool, lambda self: True, constant=True)
    checked = Property(
        bool, lambda self: bool(self._state.get_settings()[self._key]), notify=changed,
    )
    value = Property(
        float, lambda self: float(QLocale.c().toString(
            float(self._state.get_settings()[self._key]), "f", self._decimals,
        )),
        notify=changed,
    )
    text = Property(
        str, lambda self: (
            self._choices[self.index] if self._choices else
            self._locale.toString(self.value, "f", self._decimals) + self._suffix
        ),
        notify=changed,
    )
    items = Property(list, lambda self: self._choices, constant=True)
    index = Property(int, lambda self: int(self._state.get_settings()[self._key]) if self._choices else -1, notify=changed)

    @Slot(int)
    def setIndex(self, index: int) -> None:
        try:
            if self._choices and self.enabled and 0 <= index < len(self._choices):
                self._state.set_value(self._key, index)
        except (ValueError, RuntimeError):
            pass
        finally:
            self.changed.emit()

    @Slot(float)
    def setValue(self, value: float) -> None:
        try:
            if self._numeric and self.enabled and math.isfinite(value):
                self._state.set_value(
                    self._key, float(QLocale.c().toString(
                        min(self._maximum, max(self._minimum, value)), "f", self._decimals,
                    )),
                )
        except (ValueError, RuntimeError):
            # Failed writes are reported by the state; restore the displayed value.
            pass
        finally:
            self.changed.emit()

    @Slot(str)
    def setText(self, text: str) -> None:
        try:
            self.setValue(float(text.replace(self._suffix.strip(), "").replace(",", ".").strip()))
        except ValueError:
            self.changed.emit()
        self.released()

    @Slot()
    def click(self) -> None:
        try:
            if self.enabled and not self._numeric and not self._choices:
                self._state.set_value(self._key, not self.checked)
        except (ValueError, RuntimeError):
            pass
        finally:
            self.changed.emit()

    @Slot()
    def released(self) -> None:
        try:
            self._state.flush()
        except RuntimeError:
            pass
        finally:
            self.changed.emit()


class GateText(QObject):
    """Derived gate status, updated with its settings or telemetry."""

    changed = Signal()

    def __init__(self, state: GateState, key: str, parent: QObject):
        super().__init__(parent)
        self._state, self._key = state, key
        state.changed.connect(self.changed)

    text = Property(str, lambda self: self._state.presentation()[self._key], notify=changed)
    toolTip = Property(str, lambda self: "", constant=True)
    name = Property(str, lambda self: "", constant=True)
    enabled = Property(bool, lambda self: True, constant=True)
    shown = Property(bool, lambda self: True, constant=True)


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
        self._window.eq_state.step_band(direction)

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

    def _row(self, kind: str, target, label: str = "") -> dict:
        return {"kind": kind, "label": label, "proxy": self._proxy(target)}

    def _build_model(self, w) -> dict:
        row = self._row

        def control(section: str, key: str) -> SettingControl:
            return SettingControl(getattr(w, section + "_state"), section, key, self)

        def state_field(section: str, label: str, key: str, *, slider: bool = False) -> dict:
            return {
                "kind": "field" if slider else "number", "label": label,
                "proxy": control(section, key), "display": None,
            }

        def state_row(section: str, kind: str, label: str, key: str) -> dict:
            return {"kind": kind, "label": label, "proxy": control(section, key)}

        def gate_text(key: str) -> dict:
            return {"kind": "text", "label": "", "proxy": GateText(w.gate_state, key, self)}

        def health(title: str, label: QLabel, *, coded: bool = False) -> dict:
            # `coded` rows carry counter tokens rather than a readable value.
            return {"title": title, "coded": coded, "proxy": self._proxy(label)}
        meters = w.processing_meters
        eq = w.eq_presentation

        strength = NoiseControl(w.noise_suppression_state, "strength", self)
        stages = [
            {
                "trace": True, "alert": self._proxy(w.backend_diag_label),
                "title": "NOISE SUPPRESSION", "menu": None,
                "help": (
                    "Removes steady background noise such as fans and hum. The "
                    "backend choice affects cleanup quality, CPU use and latency. "
                    "Packaged builds use bundled, integrity-checked model files."
                ),
                "toggle": NoiseControl(w.noise_suppression_state, "enabled", self),
                "rows": [
                    {"kind": "combo", "label": "Backend", "proxy": NoiseControl(w.noise_suppression_state, "model", self)},
                    {"kind": "field", "label": "Strength", "proxy": strength, "display": strength},
                    {"kind": "text", "label": "", "proxy": NoiseControl(w.noise_suppression_state, "latency", self)},
                ],
                "advanced": [],
            },
            {
                "trace": False, "alert": None, "title": "NOISE GATE", "menu": None,
                "help": (
                    "Reduces gain when the signal falls below the threshold, so "
                    "background noise drops out during silence. The gate uses 3 dB "
                    "of hysteresis and a smoothed envelope to avoid chattering."
                ),
                "toggle": control("gate", "enabled"),
                "rows": [
                    state_field("gate", "Threshold", "threshold_db", slider=True),
                    state_row("gate", "toggle", "Auto threshold", "auto_threshold_enabled"),
                    gate_text("threshold_status"),
                ],
                "advanced": [
                    state_field("gate", "Attack", "attack_ms"),
                    state_field("gate", "Release", "release_ms"),
                    state_row("gate", "combo", "Gate mode", "gate_mode"),
                    state_field("gate", "VAD threshold", "vad_threshold", slider=True),
                    state_field("gate", "Hold time", "vad_hold_time_ms"),
                    state_field("gate", "VAD pre-gain", "vad_pre_gain", slider=True),
                    state_field("gate", "Margin", "gate_margin_db", slider=True),
                    gate_text("noise_floor"),
                    row("meter", meters.confidence, "Confidence"),
                    gate_text("vad_info"),
                ],
            },
            {
                "trace": False, "alert": None, "title": "DE-ESSER", "menu": None,
                "help": (
                    "Reduces harsh sibilance (s, sh, t). Auto mode tracks the "
                    "balance between sibilance and voice and adjusts the reduction "
                    "as you speak."
                ),
                "toggle": control("deesser", "enabled"),
                "rows": [
                    state_row("deesser", "toggle", "Auto", "auto_enabled"),
                    state_field("deesser", "Amount", "auto_amount", slider=True),
                    row("meter", meters.deesser_gr, "Reduction"),
                ],
                "advanced": [
                    state_field("deesser", "Low cut", "low_cut_hz", slider=True),
                    state_field("deesser", "High cut", "high_cut_hz", slider=True),
                    state_field("deesser", "Threshold", "threshold_db", slider=True),
                    state_field("deesser", "Ratio", "ratio", slider=True),
                    state_field("deesser", "Attack", "attack_ms"),
                    state_field("deesser", "Release", "release_ms"),
                    state_field("deesser", "Max reduction", "max_reduction_db", slider=True),
                ],
            },
            {
                "trace": False, "alert": None, "title": "COMPRESSOR", "menu": None,
                "help": (
                    "Evens out your level by turning down the loudest moments. "
                    "Threshold sets where it starts working; ratio sets how hard."
                ),
                "toggle": control("compressor", "enabled"),
                "rows": [
                    state_field("compressor", "Threshold", "threshold_db", slider=True),
                    state_field("compressor", "Ratio", "ratio", slider=True),
                    row("meter", meters.compressor_gr, "Reduction"),
                ],
                "advanced": [
                    state_field("compressor", "Attack", "attack_ms"),
                    state_field("compressor", "Release", "release_ms"),
                    state_field("compressor", "Makeup gain", "makeup_gain_db", slider=True),
                    state_row("compressor", "toggle", "Adaptive release", "adaptive_release"),
                    state_field("compressor", "Base release", "base_release_ms"),
                    row("text", meters.current_release, "Current release"),
                    state_row("compressor", "toggle", "Sidechain high-pass", "sidechain_highpass_enabled"),
                    state_row("compressor", "toggle", "Auto makeup gain", "auto_makeup_enabled"),
                    state_field("compressor", "Target LUFS", "target_lufs"),
                    row("text", meters.current_lufs, "Current LUFS"),
                    row("text", meters.current_makeup_gain, "Auto gain"),
                ],
            },
            {
                "trace": False,
                "alert": None,
                "title": "LIMITER",
                "help": (
                    "A safety net that stops the output from going above the "
                    "ceiling. It looks ahead 0.5 ms and ramps its gain down before "
                    "transients reach the output."
                ),
                "toggle": SettingControl(w.limiter_state, "limiter", "enabled", self),
                "menu": None,
                "rows": [{
                    "kind": "field", "label": "Ceiling", "display": None,
                    "proxy": SettingControl(w.limiter_state, "limiter", "ceiling_db", self),
                }],
                "advanced": [
                    {
                        "kind": "toggle", "label": "Careful output mode",
                        "proxy": SettingControl(w.limiter_state, "limiter", "careful_output_enabled", self),
                    },
                    {
                        "kind": "number", "label": "Release", "display": None,
                        "proxy": SettingControl(w.limiter_state, "limiter", "release_ms", self),
                    },
                ],
            },
        ]

        bands = [
            {key: EQControl(w.eq_state, key, index, self) for key in (
                "frequencyLabel", "enabled", "type", "gain", "gainLabel", "frequency", "q", "slope",
            )}
            for index in range(10)
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
                "trace": False, "alert": None, "title": "EQUALIZER",
                "help": (
                    "The curve shows correction plus tone; handles edit tone only. "
                    "Drag a handle to edit frequency and gain. Notch and pass filters "
                    "move horizontally only. Use [ and ] plus arrow keys for keyboard editing."
                ),
                "toggle": EQControl(w.eq_state, "enabled", None, self),
                "menu": self._proxy(eq.options_button), "rows": [], "advanced": [],
                "tone": self._proxy(eq.tone_preset_button),
                "curve": self._proxy(eq.curve_widget),
                "layers": EQControl(w.eq_state, "layers", None, self),
                "diagnostics": EQControl(w.eq_state, "diagnostics", None, self),
                "band": EQControl(w.eq_state, "band", None, self),
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
    window._ensure_processing_panels()
    shell.show()
    window.statusBar().show()
    _announce_fallback(window)
    view.deleteLater()
    bridge.release()
