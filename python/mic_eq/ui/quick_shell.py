"""Qt Quick view of the main window.

Processing and shell controls consume shared state and Qt action notifications.
The classic view stays available for rendering failure recovery.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QCoreApplication,
    QObject,
    QPointF,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QPainter, QPen
from PySide6.QtQml import qmlRegisterType
from PySide6.QtQuick import QQuickPaintedItem
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QLabel,
)

from .components import Glyph, icon_font
from .control_specs import control_label
from .eq_controls import EQControl
from .eq_quick_graph import EQGraphItem
from .quick_meter import QuickMeterItem
from .shell_state import ActionControl
from .level_meter import LevelMeter, GainReductionMeter, ConfidenceMeter
from .noise_controls import NoiseControl
from .processing_controls import SettingControl, GateText
from .theme import (
    PALETTE,
    RADIUS_CARD,
    RADIUS_CONTROL,
    prefers_reduced_motion,
    qcolor,
)

QML_DIR = Path(__file__).parent / "qml"
# Level history behind the noise suppression trace: 6 seconds at 20 Hz.
TRACE_INTERVAL_MS = 50
TRACE_POINTS = 120
MINIMUM_SCENE_SIZE = (900, 600)
QUICK_VIEW_UNAVAILABLE = "The Qt Quick view could not start; using the classic view."
_LOG = logging.getLogger(__name__)

class MeterSource(QObject):
    """Repaint notification for an existing classic meter renderer."""

    repaintRequested = Signal()

    def __init__(self, target: LevelMeter | GainReductionMeter | ConfidenceMeter, parent: QObject):
        super().__init__(parent)
        self.target = target
        target.changed.connect(self.repaintRequested)

    name = Property(str, lambda self: self.target.accessibleName(), constant=True)
    toolTip = Property(str, lambda self: self.target.toolTip(), constant=True)
    enabled = Property(bool, lambda self: self.target.isEnabled(), notify=repaintRequested)
    shown = Property(bool, lambda self: True, constant=True)


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
    """Shared state and commands exposed to the QML scene."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._model = self._build_model(window)
        self._levels: deque[list[float]] = deque(maxlen=TRACE_POINTS)
        self._trace_timer = QTimer(self)
        self._trace_timer.setInterval(TRACE_INTERVAL_MS)
        self._trace_timer.timeout.connect(self._sample_levels)
        if not prefers_reduced_motion():
            self._trace_timer.start()

    model = Property(dict, lambda self: self._model, constant=True)

    def release(self) -> None:
        """Release scene adapters when the classic view takes over."""

        self.deleteLater()

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

    def _row(self, kind: str, target, label: str = "") -> dict:
        return {"kind": kind, "label": label, "proxy": MeterSource(target, self) if kind == "meter" else target}

    def _build_model(self, w) -> dict:
        row = self._row
        def text(name: str):
            return w.shell_texts[name]

        def action(name: str):
            return ActionControl(getattr(w, name + "_action"), self,
                                 name=getattr(w, name).accessibleName())

        def menu(target, label: str):
            # QMenu/QAction are the shared native command owners.
            return ActionControl(target.menuAction(), self, name=label, text=label, menu=target)

        def control(section: str, key: str) -> SettingControl:
            return SettingControl(getattr(w, section + "_state"), section, key, self)

        def state_field(section: str, key: str, *, slider: bool = False) -> dict:
            return {
                "kind": "field" if slider else "number", "label": control_label(section, key),
                "proxy": control(section, key), "display": None,
            }

        def state_row(section: str, kind: str, key: str) -> dict:
            return {"kind": kind, "label": control_label(section, key), "proxy": control(section, key)}

        def gate_text(key: str) -> dict:
            return {"kind": "text", "label": "", "proxy": GateText(w.gate_state, key, self)}

        def health(title: str, name: str, *, coded: bool = False) -> dict:
            # `coded` rows carry counter tokens rather than a readable value.
            return {"title": title, "coded": coded, "proxy": text(name)}
        meters = w.processing_meters
        eq = w.eq_presentation

        strength = NoiseControl(w.noise_suppression_state, "strength", self)
        stages = [
            {
                "trace": True, "alert": text("backend_diag_label"),
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
                    state_field("gate", "threshold_db", slider=True),
                    state_row("gate", "toggle", "auto_threshold_enabled"),
                    gate_text("threshold_status"),
                ],
                "advanced": [
                    state_field("gate", "attack_ms"),
                    state_field("gate", "release_ms"),
                    state_row("gate", "combo", "gate_mode"),
                    state_field("gate", "vad_threshold", slider=True),
                    state_field("gate", "vad_hold_time_ms"),
                    state_field("gate", "vad_pre_gain", slider=True),
                    state_field("gate", "gate_margin_db", slider=True),
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
                    state_row("deesser", "toggle", "auto_enabled"),
                    state_field("deesser", "auto_amount", slider=True),
                    row("meter", meters.deesser_gr, "Reduction"),
                ],
                "advanced": [
                    state_field("deesser", "low_cut_hz", slider=True),
                    state_field("deesser", "high_cut_hz", slider=True),
                    state_field("deesser", "threshold_db", slider=True),
                    state_field("deesser", "ratio", slider=True),
                    state_field("deesser", "attack_ms"),
                    state_field("deesser", "release_ms"),
                    state_field("deesser", "max_reduction_db", slider=True),
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
                    state_field("compressor", "threshold_db", slider=True),
                    state_field("compressor", "ratio", slider=True),
                    row("meter", meters.compressor_gr, "Reduction"),
                ],
                "advanced": [
                    state_field("compressor", "attack_ms"),
                    state_field("compressor", "release_ms"),
                    state_field("compressor", "makeup_gain_db", slider=True),
                    state_row("compressor", "toggle", "adaptive_release"),
                    state_field("compressor", "base_release_ms"),
                    row("text", meters.text_states["current_release"], "Current release"),
                    state_row("compressor", "toggle", "sidechain_highpass_enabled"),
                    state_row("compressor", "toggle", "auto_makeup_enabled"),
                    state_field("compressor", "target_lufs"),
                    row("text", meters.text_states["current_lufs"], "Current LUFS"),
                    row("text", meters.text_states["current_makeup_gain"], "Auto gain"),
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
                "rows": [state_field("limiter", "ceiling_db", slider=True)],
                "advanced": [
                    state_row("limiter", "toggle", "careful_output_enabled"),
                    state_field("limiter", "release_ms"),
                ],
            },
        ]

        bands = [
            {key: EQControl(w.eq_state, key, index, self) for key in (
                "frequencyLabel", "enabled", "type", "gain", "gainLabel", "frequency", "q", "slope",
            )}
            for index in range(10)
        ]

        settings_cards = w.shell_settings_cards

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
            "page": w.shell_page,
            "healthPage": w.HEALTH_PAGE_INDEX,
            "banners": [
                text("device_warning_banner"),
                text("config_warning_banner"),
            ],
            "top": {
                "input": w.input_choice,
                "output": w.output_choice,
                "refresh": action("refresh_btn"),
                "mode": w.processing_mode_choice,
                "mute": action("user_mute_checkbox"),
                "start": action("start_btn"),
                "stop": action("stop_btn"),
            },
            "presets": {
                "status": text("preset_status_label"),
                "menu": menu(w.presets_button.menu(), "Presets"),
                "undo": action("_undo_auto_eq_button"),
                "redo": action("redo_button"),
                "testSound": action("test_sound_button"),
                "autoEq": action("auto_eq_button"),
                "voiceSetup": action("auto_voice_setup_button"),
            },
            "eq": {
                "trace": False, "alert": None, "title": "EQUALIZER",
                "help": (
                    "The curve shows correction plus tone; handles edit tone only. "
                    "Drag a handle to edit frequency and gain. Notch and pass filters "
                    "move horizontally only. Use [ and ] plus arrow keys for keyboard editing."
                ),
                "toggle": EQControl(w.eq_state, "enabled", None, self),
                "menu": menu(eq.options_menu, "Equalizer options"), "rows": [], "advanced": [],
                "tone": menu(eq.tone_menu, "Tone preset"),
                "curve": eq.graph_model,
                "layers": EQControl(w.eq_state, "layers", None, self),
                "diagnostics": EQControl(w.eq_state, "diagnostics", None, self),
                "band": EQControl(w.eq_state, "band", None, self),
                "bands": bands,
            },
            "stages": stages,
            "meters": [MeterSource(w.input_meter, self), MeterSource(w.output_meter, self)],
            "status": {
                "message": w.status_message,
                "transmission": text("transmission_status_label"),
                "calibration": text("calibration_status_label"),
                "health": text("health_summary_label"),
            },
            "health": {
                "advice": text("health_advice_label"),
                "route": text("route_status_label"),
                "signal": [
                    health("Input", "input_health_label"),
                    health("Output", "output_health_label"),
                    health("Gate", "gate_health_label"),
                ],
                "stream": [
                    health("Backend", "backend_diag_label", coded=True),
                    health("Callbacks", "callback_health_label"),
                    health("Underruns", "underrun_health_label"),
                    health("Latency", "latency_label"),
                    health("Buffer", "buffer_label"),
                    health("Drops", "dropped_label", coded=True),
                    health("Recovery", "recovery_diag_label", coded=True),
                ],
            },
            "settings": {
                "input": [
                    row("combo", w.input_channel_mode_choice, "Input mode"),
                    row("combo", w.input_cleanup_mode_choice, "Cleanup"),
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

    qmlRegisterType(EQGraphItem, "AudioForge", 1, 0, "EQGraphItem")  # type: ignore[call-overload]
    qmlRegisterType(QuickMeterItem, "AudioForge", 1, 0, "QuickMeterItem")  # type: ignore[call-overload]
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
