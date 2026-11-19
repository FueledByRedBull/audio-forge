"""Qt Quick view of the main window.

Processing and shell controls consume shared state and Qt action notifications.
When the scene cannot start or stops rendering, the window shows an error screen.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import UTC, datetime
from functools import cache

from PySide6.QtCore import (
    Property,
    QCoreApplication,
    QEvent,
    QObject,
    QPointF,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAccessible,
    QAccessibleAnnouncementEvent,
    QDesktopServices,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtQml import qmlRegisterType, qmlRegisterUncreatableType
from PySide6.QtQuick import QQuickPaintedItem, QSGRendererInterface
from PySide6.QtQuickWidgets import QQuickWidget

from .chain_presentation import PREFILTER_HIGHPASS_HZ, ChainInputs, ChainLine, describe_chain
from . import platform_text
from .command_palette import SHORTCUT as PALETTE_SHORTCUT, CommandPalette, window_commands
from .components import Glyph
from .engine_session import Restarted, RestartFailed, Started, Stopped
from .desktop_integration import RENDERER_RELAUNCH_ENV, relaunch_with_software_rendering
from .control_specs import control_label
from .eq_controls import EQControl
from .eq_graph import EQGraphGeometry
from .eq_hold_compare import EQHoldCompare
from .eq_quick_compare import EQComparisonItem
from .device_selection import VIRTUAL_OUTPUT_MARKERS
from .eq_quick_graph import EQGraphItem, EQGraphPreview
from .health import HEALTH_PEAK_WINDOW_SECONDS, advice_for
from .overview_presentation import (
    NO_MIC_REASON,
    SOUND_SETTINGS_FAILED,
    SOUND_SETTINGS_LINE,
    SOUND_SETTINGS_URI,
    OverviewInputs,
    OverviewView,
    describe_overview,
    split_check,
)
from .quick_meter import QuickMeterItem
from .quick_scene import QML_DIR, load_scene, quick_theme
from .shell_specs import ACTION_SPECS, HEALTH_DECISION_KEYS, HEALTH_LAYOUT_KEYS
from .shell_state import ActionControl
from .noise_controls import NoiseControl
from .quick_error_screen import show_quick_error
from .processing_controls import SettingControl, GateText
from .theme import PALETTE, RADIUS_CONTROL, qcolor

# Level history behind the noise suppression trace: 6 seconds at 20 Hz.
TRACE_INTERVAL_MS = 50
AGE_INTERVAL_MS = 1000
TRACE_POINTS = 120
# The Overview ribbon shows 10 seconds; the history keeps one sample more, the one it scrolls in.
RIBBON_POINTS = 200
LEVEL_HISTORY = RIBBON_POINTS + 1
# A shown window with no scene graph after this many checks has no renderer.
RENDERER_CHECK_MS = 1000
RENDERER_CHECKS = 3
_LOG = logging.getLogger(__name__)

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
        points = self._points[-TRACE_POINTS:]
        first = TRACE_POINTS - len(points)
        for line, color in enumerate((PALETTE.text_muted, PALETTE.accent)):
            painter.setPen(QPen(qcolor(color), 1.5))
            painter.drawPolyline(
                [
                    QPointF(
                        width * (first + index) / (TRACE_POINTS - 1),
                        # -60 dB at the bottom, 0 dB at the top.
                        4 + (height - 8) * (1 - _level_share(point[line])),
                    )
                    for index, point in enumerate(points)
                ]
            )


def _level_share(db: float) -> float:
    """How far up a level sits on the -60 to 0 dB scale, from 0 to 1."""
    return min(1.0, max(0.0, (db + 60) / 60))


class LevelRibbon(QQuickPaintedItem):
    """The last ten seconds of level, mirrored: the microphone above the centre line, what callers hear below.

    The item is one sample wider than the window it shows through; the scene slides
    it by that sample between updates, so it scrolls without a repaint per frame.
    """

    pointsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: list[list[float]] = []
        self.setAntialiasing(True)
        self.visibleChanged.connect(self.update)

    def _set_points(self, points: list) -> None:
        self._points = points
        self.pointsChanged.emit()
        if self.isVisible():
            self.update()

    points = Property(list, lambda self: self._points, _set_points, notify=pointsChanged)

    def paint(self, painter: QPainter) -> None:
        width, middle = self.width(), self.height() / 2
        reach = middle - 2
        painter.setPen(QPen(qcolor(PALETTE.data_grid), 1))
        painter.drawLine(QPointF(0, middle), QPointF(width, middle))
        points = self._points[-LEVEL_HISTORY:]
        if len(points) < 2:
            return
        step = width / RIBBON_POINTS
        first = LEVEL_HISTORY - len(points)
        for line, (stroke, fill, alpha, side) in enumerate((
            (PALETTE.signal_in, PALETTE.signal_in_fill, 0.42, -1),
            (PALETTE.signal_out, PALETTE.signal_out, 0.30, 1),
        )):
            tops = [
                QPointF(step * (first + index), middle + side * reach * _level_share(point[line]))
                for index, point in enumerate(points)
            ]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(fill, alpha=round(255 * alpha)))
            painter.drawPolygon(QPolygonF([QPointF(tops[0].x(), middle), *tops, QPointF(tops[-1].x(), middle)]))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(qcolor(stroke), 1.2))
            painter.drawPolyline(tops)


def _window_running(window) -> bool:
    # The session's own flag: the engine is not asked while it is mid-call.
    return bool(window.__dict__.get("_processing"))


def _backend_name(window, model: str) -> str:
    """The suppressor's short name: its menu label without the note in brackets."""
    labels = dict(window.noise_suppression_state.models)
    return labels.get(model, model).split(" (")[0]


def _last_voice_setup(results) -> datetime | None:
    times = []
    for result in results:
        if result.scope != "full_voice_setup":
            continue
        try:
            stamp = datetime.fromisoformat(result.created_at)
        except ValueError:
            continue
        times.append(stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC))
    return max(times, default=None)


def overview_inputs(window) -> OverviewInputs:
    """What the Overview reads from the window, all of it main-thread state."""
    texts = window.shell_texts
    session = window.engine_session.view
    issues = [
        advice_for(texts[key].text) or texts[key].text
        for level in ("bad", "warn")
        for key in HEALTH_DECISION_KEYS + HEALTH_LAYOUT_KEYS
        if texts[key].state == level
    ]
    noise = window.noise_suppression_state.get_settings()
    input_name = window._device_name_from_identity(window._combo_device_identity(window.input_choice)) or ""
    output_name = window._device_name_from_identity(window._combo_device_identity(window.output_choice)) or ""
    return OverviewInputs(
        now=datetime.now(UTC),
        running=_window_running(window),
        no_mic=window.no_microphone(),
        restart_pending=session.busy is None and window._restart_pending(),
        failed_switch_device=window._failed_switch_device,
        device_bad=texts["callback_health_label"].state == "bad",
        device_advice=advice_for(texts["callback_health_label"].text),
        input_bad=texts["input_health_label"].state == "bad",
        input_advice=advice_for(texts["input_health_label"].text),
        muted=session.output_muted,
        mute_error=session.mute_error,
        mute_requested=session.mute_requested,
        issues=tuple(issues),
        mode=window._processing_mode(),
        backend=_backend_name(window, noise["model"]) if noise["enabled"] else "",
        output_peak_db=window.output_meter.recent_peak_db(HEALTH_PEAK_WINDOW_SECONDS),
        input_name=input_name,
        output_name=output_name,
        output_found=window.output_choice.count() > 0,
        output_virtual=any(marker in output_name.lower() for marker in VIRTUAL_OUTPUT_MARKERS),
        preset_name=window.configuration.identity.name or "Default",
        last_setup=_last_voice_setup(window.config.calibration_results),
    )


class OverviewState(QObject):
    """The Overview's status sentence and card, worked out again whenever what they read changes.

    Assistive technology hears the sentence once each time the state itself changes,
    after the tick that changed it has settled, never for a new reading of the same state.
    """

    changed = Signal()

    def __init__(self, parent: QObject, window, sources) -> None:
        super().__init__(parent)
        self._window = window
        # Replaceable, so a test never opens the real Settings app.
        self.open_url = lambda url: QDesktopServices.openUrl(QUrl(url))
        # "Remind me later" lasts until processing stops or starts again; nothing is saved.
        self._notice_dismissed = False
        self._settings_failed = False
        window.engine_session.changed.connect(self._on_session)
        self._view: OverviewView = describe_overview(overview_inputs(window))
        self._announced = self._view.state
        self._announce = QTimer(self)
        self._announce.setSingleShot(True)
        self._announce.setInterval(0)
        self._announce.timeout.connect(self._announce_state)
        for signal in sources:
            signal.connect(self.refresh)

    state = Property(str, lambda self: self._view.state, notify=changed)
    headline = Property(str, lambda self: self._view.headline, notify=changed)
    detail = Property(str, lambda self: self._view.detail, notify=changed)
    tone = Property(str, lambda self: self._view.tone, notify=changed)
    callersAffected = Property(bool, lambda self: self._view.callers_affected, notify=changed)
    offerRestart = Property(bool, lambda self: self._view.offer_restart, notify=changed)
    showClippingNotice = Property(
        bool, lambda self: self._view.show_clipping_notice and not self._notice_dismissed, notify=changed
    )
    # Only Windows has a Sound settings page to open.
    offerSoundSettings = Property(bool, lambda self: platform_text.IS_WINDOWS, notify=changed)
    soundSettingsLine = Property(str, lambda self: SOUND_SETTINGS_LINE, constant=True)
    soundSettingsFailure = Property(
        str, lambda self: SOUND_SETTINGS_FAILED if self._settings_failed else "", notify=changed
    )
    offerChooseMicrophone = Property(bool, lambda self: self._view.offer_choose_microphone, notify=changed)
    callersLine = Property(str, lambda self: self._view.callers_line, notify=changed)
    navLine = Property(str, lambda self: self._view.nav_line, notify=changed)
    ribbonCaption = Property(str, lambda self: self._view.ribbon_caption, notify=changed)
    recommendSetup = Property(bool, lambda self: self._view.recommend_setup, notify=changed)
    setupLine = Property(str, lambda self: self._view.setup_line, notify=changed)
    checksTitle = Property(str, lambda self: self._view.checks_title, notify=changed)
    # Each line whole, and split into its label and the device it names for the card.
    checks = Property(
        list,
        lambda self: [
            {"text": text, "kind": kind, "label": label, "detail": detail}
            for text, kind in self._view.checks
            for label, detail in (split_check(text),)
        ],
        notify=changed,
    )

    @Slot()
    def openSoundSettings(self) -> None:
        self._settings_failed = not self.open_url(SOUND_SETTINGS_URI)
        self.changed.emit()

    @Slot()
    def dismissNotice(self) -> None:
        self._notice_dismissed = True
        self.changed.emit()

    @Slot(object)
    def _on_session(self, session_view) -> None:
        # A recovery restart ends a stall or a lost device without a Start or Stop; a Voice Setup
        # lease belongs to the same run, so it does not.
        if isinstance(session_view.event, Started | Stopped | Restarted | RestartFailed) and (
            self._notice_dismissed or self._settings_failed
        ):
            self._notice_dismissed = False
            self._settings_failed = False
            self.changed.emit()

    @Slot()
    def refresh(self) -> None:
        view = describe_overview(overview_inputs(self._window))
        if view != self._view:
            self._view = view
            self.changed.emit()
        if view.state != self._announced:
            self._announce.start()

    @Slot()
    def _announce_state(self) -> None:
        view = self._view
        if view.state == self._announced:
            return
        self._announced = view.state
        if QAccessible.isActive():
            QAccessible.updateAccessibility(
                QAccessibleAnnouncementEvent(self._window, f"{view.headline}. {view.detail}")
            )


def chain_inputs(window) -> ChainInputs:
    meters = window.processing_meters

    def reduction(meter) -> float | None:
        return meter.gain_reduction_db if meter.measurement_available else None

    noise = window.noise_suppression_state.get_settings()
    gate = window.gate_state.get_settings()
    eq = window.eq_state.get_eq_settings()
    return ChainInputs(
        running=_window_running(window),
        cleanup_mode=window.input_cleanup_mode_choice.currentText() or "Off",
        gate_enabled=bool(gate["enabled"]),
        gate_threshold_db=float(gate["threshold_db"]),
        suppression_enabled=bool(noise["enabled"]),
        backend=_backend_name(window, noise["model"]),
        strength=float(noise["strength"]),
        deesser_enabled=bool(window.deesser_state.get_settings()["enabled"]),
        deesser_reduction_db=reduction(meters.deesser_gr),
        eq_enabled=eq.enabled,
        eq_bands=sum(1 for band in eq.bands if band.enabled),
        compressor_enabled=bool(window.compressor_state.get_settings()["enabled"]),
        compressor_reduction_db=reduction(meters.compressor_gr),
        limiter_enabled=bool(window.limiter_state.get_settings()["enabled"]),
        limiter_reduction_db=reduction(meters.limiter_gr),
    )


class ChainLines(QObject):
    """The state line and activity of each chain row, keyed as the rows are."""

    changed = Signal()

    def __init__(self, parent: QObject, window, sources) -> None:
        super().__init__(parent)
        self._window = window
        self._lines: dict[str, ChainLine] = describe_chain(chain_inputs(window))
        for signal in sources:
            signal.connect(self.refresh)

    lines = Property(
        dict,
        lambda self: {
            key: {"line": line.line, "activity": line.activity, "metered": line.metered}
            for key, line in self._lines.items()
        },
        notify=changed,
    )

    @Slot()
    def refresh(self) -> None:
        lines = describe_chain(chain_inputs(self._window))
        if lines != self._lines:
            self._lines = lines
            self.changed.emit()


class InspectorState(QObject):
    """Which stage the docked inspector shows, and whether it is open."""

    changed = Signal()

    def __init__(self, parent: QObject, keys: list[str]):
        super().__init__(parent)
        self._keys = frozenset(keys)
        self._key, self._shown = "", False

    # The key stays when the inspector closes, so the panel that fades out is the one that was open.
    stageKey = Property(str, lambda self: self._key, notify=changed)
    shown = Property(bool, lambda self: self._shown, notify=changed)

    @Slot(str)
    def show(self, key: str) -> None:
        if key in self._keys and (key, True) != (self._key, self._shown):
            self._key, self._shown = key, True
            self.changed.emit()

    @Slot()
    def close(self) -> None:
        if self._shown:
            self._shown = False
            self.changed.emit()


class QuickBridge(QObject):
    """Shared state and commands exposed to the QML scene."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        try:
            self._model = self._build_model(window)
        except Exception:
            # Its controls are wired to shared state; none may outlive the failure.
            self.deleteLater()
            raise
        self._levels: deque[list[float]] = deque(maxlen=LEVEL_HISTORY)
        self._trace_timer = QTimer(self)
        self._trace_timer.setInterval(TRACE_INTERVAL_MS)
        self._trace_timer.timeout.connect(self._sample_levels)
        # The age beside Undo counts whole seconds; like the meters, it runs only while the window shows.
        self._age_timer = QTimer(self)
        self._age_timer.setInterval(AGE_INTERVAL_MS)
        self._age_timer.timeout.connect(window.history_chip.refresh)
        # A readout, not an animation: reduced motion keeps it. Like the meters, it
        # runs only while the window shows.
        window.installEventFilter(self)
        self._sync_trace_timer()

    model = Property(dict, lambda self: self._model, constant=True)

    def release(self) -> None:
        """Release scene adapters when the scene is dropped."""

        # Nothing is left to let go of a hold the scene began.
        self._window.eq_state.release_original()
        # The compact window shows this scene's proxies too.
        self._window.close_compact_window()
        self.deleteLater()

    levelsChanged = Signal()
    levels = Property(list, lambda self: list(self._levels), notify=levelsChanged)

    def _sync_trace_timer(self) -> None:
        window = self._window
        if window.isVisible() and not window.isMinimized():
            if not self._trace_timer.isActive():
                self._trace_timer.start()
            if not self._age_timer.isActive():
                self._age_timer.start()
        else:
            self._trace_timer.stop()
            self._age_timer.stop()

    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        if kind == QEvent.Type.Hide:
            # Also sent while the window is destroyed, when it can no longer be asked.
            self._trace_timer.stop()
            self._age_timer.stop()
        elif kind in (QEvent.Type.Show, QEvent.Type.WindowStateChange):
            self._sync_trace_timer()
        return False

    @Slot()
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

    def _row(self, kind: str, proxy, label: str = "") -> dict:
        return {"kind": kind, "label": label, "proxy": proxy}

    @staticmethod
    def keycaps(shortcut: QKeySequence | str) -> list[str]:
        """A shortcut's keys as their caps read; Return is labelled Enter, as on the key."""
        text = QKeySequence(shortcut).toString(QKeySequence.SequenceFormat.PortableText)
        return ["Enter" if key == "Return" else key for key in text.split("+")]

    def _build_model(self, w) -> dict:
        row = self._row
        def text(name: str):
            return w.shell_texts[name]

        def action(name: str):
            spec = ACTION_SPECS[name]
            command = getattr(w, name + "_action")
            if spec.shortcut:
                # No menu holds this action; the window owns its shortcut.
                command.setShortcut(spec.shortcut)
                # Holding the keys must not flap a live mute, as the global hotkey does not.
                command.setAutoRepeat(False)
                w.addAction(command)
            return ActionControl(command, self, name=spec.name, locked_tip=spec.locked_tip, menu=command.menu())

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

        def health(title: str, name: str) -> dict:
            return {"title": title, "proxy": text(name)}
        meters = w.processing_meters
        eq = w.eq_presentation

        strength = NoiseControl(w.noise_suppression_state, "strength", self)
        stages = [
            {
                "trace": True, "alert": text("backend_diag_label"),
                "key": "suppression", "title": "NOISE SUPPRESSION", "name": "Noise suppression", "menu": None,
                "help": (
                    "Removes steady background noise such as fans and hum. The "
                    "backend choice affects cleanup quality, CPU use and latency. "
                    "Packaged builds use bundled, integrity-checked model files."
                ),
                "toggle": NoiseControl(w.noise_suppression_state, "enabled", self),
                "rows": [
                    {
                        "kind": "combo", "label": "Backend",
                        "proxy": NoiseControl(w.noise_suppression_state, "model", self, w._on_model_changed),
                    },
                    {"kind": "field", "label": "Strength", "proxy": strength, "display": strength},
                    {"kind": "text", "label": "", "proxy": NoiseControl(w.noise_suppression_state, "latency", self)},
                ],
                "advanced": [],
            },
            {
                "trace": False, "alert": None, "key": "gate", "title": "NOISE GATE", "name": "Noise gate", "menu": None,
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
                "trace": False, "alert": None, "key": "deesser", "title": "DE-ESSER", "name": "De-esser", "menu": None,
                "help": (
                    "Reduces harsh sibilance (s, sh, t). Auto mode tracks the "
                    "balance between sibilance and voice and adjusts the reduction "
                    "as you speak."
                ),
                "toggle": control("deesser", "enabled"),
                "rows": [
                    state_row("deesser", "toggle", "auto_enabled"),
                    row("text", text("deesser_auto_hint")),
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
                "trace": False, "alert": None, "key": "compressor", "title": "COMPRESSOR", "name": "Compressor", "menu": None,
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
                "key": "limiter",
                "title": "LIMITER",
                "name": "Limiter",
                "help": (
                    "A safety net that stops the output from going above the "
                    "ceiling. It looks ahead 0.5 ms and ramps its gain down before "
                    "transients reach the output."
                ),
                "toggle": SettingControl(w.limiter_state, "limiter", "enabled", self),
                "menu": None,
                "rows": [state_field("limiter", "ceiling_db", slider=True)],
                "readout": text("output_health_label"),
                "advanced": [
                    state_row("limiter", "toggle", "careful_output_enabled"),
                    state_field("limiter", "release_ms"),
                ],
            },
        ]

        bands = [
            {key: EQControl(w.eq_state, key, index, self) for key in (
                "frequencyLabel", "enabled", "type", "gain", "gainLabel", "frequency", "q", "slope", "hint",
            )}
            for index in range(10)
        ]

        settings_cards = w.shell_settings_cards
        eq_toggle = EQControl(w.eq_state, "enabled", None, self)

        # Input cleanup always runs, so its inspector holds the Settings pickers and no switch.
        cleanup = {
            "key": "cleanup", "name": "Input cleanup", "toggle": None, "trace": False, "alert": None,
            "help": "", "readout": None, "advanced": [],
            "rows": [
                row("combo", w.input_cleanup_mode_choice, "Cleanup"),
                row("combo", w.input_channel_mode_choice, "Input mode"),
            ],
        }

        def chain_row(name: str, line: str, toggle=None, *, card=None, page: int = -1) -> dict:
            # A row opens its page, or else the inspector of its card; ``line`` keys its state line.
            return {
                "name": name, "toggle": toggle, "card": card, "key": card["key"] if card else "",
                "page": page, "line": line,
            }

        by_key = {stage["key"]: stage for stage in stages}

        def stage_row(key: str) -> dict:
            return chain_row(by_key[key]["name"], key, by_key[key]["toggle"], card=by_key[key])

        # In signal order, which is not the order of the stages above.
        chain = [
            chain_row("Input cleanup", "cleanup", card=cleanup),
            stage_row("gate"),
            stage_row("suppression"),
            stage_row("deesser"),
            chain_row("Equalizer", "eq", eq_toggle, page=w.EQUALIZER_PAGE_INDEX),
            stage_row("compressor"),
            stage_row("limiter"),
        ]
        meters_changed = [meter.changed for meter in (meters.compressor_gr, meters.deesser_gr, meters.limiter_gr)]
        # The window handles a session event first; connected after it, these read the result.
        session_changed = [
            w.engine_session.changed, w.start_btn_action.changed, w.stop_btn_action.changed,
            text("transmission_status_label").changed,
        ]
        chain_lines = ChainLines(self, w, [
            *meters_changed, *session_changed, w.input_cleanup_mode_choice.changed,
            *(state.changed for state in (
                w.gate_state, w.noise_suppression_state, w.deesser_state, w.eq_state,
                w.compressor_state, w.limiter_state,
            )),
        ])
        overview = OverviewState(self, w, [
            *session_changed, w.input_choice.changed, w.output_choice.changed,
            w.processing_mode_choice.changed, w.noise_suppression_state.changed,
            *(text(key).changed for key in (
                *HEALTH_DECISION_KEYS, *HEALTH_LAYOUT_KEYS, "health_summary_label",
                "route_status_label", "preset_status_label", "calibration_status_label",
            )),
        ])
        # Connected after the window's and the Overview's handlers, so it reads the state they left.
        compare = EQHoldCompare(self, w.eq_state, overview)
        w.engine_session.changed.connect(compare.session_changed)
        w.installEventFilter(compare)
        inspector = InspectorState(self, [row["key"] for row in chain if row["card"]])
        top = {
            "input": w.input_choice,
            "output": w.output_choice,
            "refresh": action("refresh_btn"),
            "mode": w.processing_mode_choice,
            "mute": action("user_mute_checkbox"),
            "start": action("start_btn"),
            "stop": action("stop_btn"),
            "restart": action("restart_audio_btn"),
            "chooseMicrophone": action("choose_microphone_btn"),
            "latency": text("latency_readout_label"),
        }
        presets = {
            "status": text("preset_status_label"),
            "menu": menu(w.presets_menu, "Presets"),
            "undo": action("_undo_auto_eq_button"),
            "redo": action("redo_button"),
            "history": w.history_chip,
            "testSound": action("test_sound_button"),
            "autoEq": action("auto_eq_button"),
            "voiceSetup": action("auto_voice_setup_button"),
        }
        preset_actions = [ActionControl(item, self) for item in w.builtin_presets_menu.actions()]
        commands = window_commands(
            top=top, presets=presets, chain=chain, stages=by_key, preset_actions=preset_actions,
        )
        palette = CommandPalette(self, w, commands, [
            *(top[key].changed for key in ("start", "stop", "mute")),
            *(presets[key].changed for key in ("testSound", "voiceSetup")),
            *(row["toggle"].changed for row in chain if row["toggle"] is not None),
            *(command.setting.changed for command in commands if command.setting is not None),
            *(proxy.changed for proxy in preset_actions),
        ])
        # Another page has no inspector beside it.
        w.shell_page.selected.connect(inspector.close)

        return {
            "theme": quick_theme(),
            "glyphs": {
                "mic": Glyph.MICROPHONE,
                "speaker": Glyph.SPEAKER,
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
            "settingsPage": w.SETTINGS_PAGE_INDEX,
            "equalizerPage": w.EQUALIZER_PAGE_INDEX,
            "inspector": inspector,
            "overview": overview,
            "startBlockedReason": NO_MIC_REASON,
            "ribbonPoints": RIBBON_POINTS,
            "levelIntervalMs": TRACE_INTERVAL_MS,
            "banners": [
                text("device_warning_banner"),
                text("config_warning_banner"),
            ],
            "top": top,
            "sidebar": {
                "chain": chain,
                "lines": chain_lines,
                "measureLatency": ActionControl(
                    w.latency_calibration_action, self, text="Measure latency",
                ),
            },
            "presets": presets,
            "palette": palette,
            "eq": {
                "trace": False, "alert": None, "title": "EQUALIZER",
                "help": (
                    "The curve shows correction (dashed) plus tone and the fixed input filter (dotted); "
                    "adaptive input cleanup follows the signal and is not drawn. Handles edit tone only. "
                    "Drag a handle to edit frequency and gain. Notch and pass filters "
                    "move horizontally only. Use [ and ] plus arrow keys for keyboard editing; "
                    "Delete resets the selected band."
                ),
                "toggle": eq_toggle,
                "menu": menu(eq.options_menu, "Equalizer options"), "rows": [], "advanced": [],
                "tone": menu(eq.tone_menu, "Tone preset"),
                "curve": eq.graph_model,
                "layers": EQControl(w.eq_state, "layers", None, self),
                "diagnostics": EQControl(w.eq_state, "diagnostics", None, self),
                "band": EQControl(w.eq_state, "band", None, self),
                "bands": bands,
                "compare": compare,
                # The height of the named regions drawn across the top of the graph.
                "regionStrip": EQGraphGeometry.REGION_STRIP_HEIGHT,
                # How each line of the graph is drawn, named in words beside its sample.
                "legend": [
                    {"text": "Total (what you hear)", "lineStyle": "solid"},
                    {"text": "Auto-EQ", "lineStyle": "dashed"},
                    {"text": f"High-pass {PREFILTER_HIGHPASS_HZ} Hz", "lineStyle": "dotted"},
                    {"text": "Bands", "lineStyle": "dot"},
                ],
            },
            "stages": stages,
            # The row of window shortcuts on the Overview, read from the actions themselves.
            "shortcuts": [
                {"keys": self.keycaps(w.start_menu_action.shortcut()), "action": "Start"},
                {"keys": self.keycaps(w.stop_menu_action.shortcut()), "action": "Stop"},
                {"keys": self.keycaps(w.user_mute_checkbox_action.shortcut()), "action": "Mute"},
                {"keys": self.keycaps(PALETTE_SHORTCUT), "action": "Search"},
            ],
            "meters": [w.input_meter, w.output_meter],
            "status": {
                "message": w.status_message,
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
                    health("Backend", "backend_diag_label"),
                    health("Device activity", "callback_health_label"),
                    health("Dropouts", "underrun_health_label"),
                    health("Latency", "latency_label"),
                    health("Buffer", "buffer_label"),
                    health("Drops", "dropped_label"),
                    health("Recovery", "recovery_diag_label"),
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


_GRAPHICS_APIS = {
    QSGRendererInterface.GraphicsApi.Software: "software",
    QSGRendererInterface.GraphicsApi.OpenGL: "opengl",
    QSGRendererInterface.GraphicsApi.Direct3D11: "direct3d11",
    QSGRendererInterface.GraphicsApi.Vulkan: "vulkan",
    QSGRendererInterface.GraphicsApi.Null: "null",
}


def graphics_api(window) -> str:
    """The graphics backend of the window's live scene, as a fixed lowercase word."""

    scene = window.centralWidget()
    interface = scene.quickWindow().rendererInterface() if isinstance(scene, QQuickWidget) else None
    api = "unknown" if interface is None else _GRAPHICS_APIS.get(interface.graphicsApi(), "unknown")
    return "software_after_failure" if api == "software" and os.environ.get(RENDERER_RELAUNCH_ENV) else api


@cache
def register_qml_types() -> None:
    """Register the scene's Python types with QML; Qt keeps every registration for the process."""

    qmlRegisterType(EQGraphItem, "AudioForge", 1, 0, "EQGraphItem")  # type: ignore[call-overload]
    qmlRegisterType(EQComparisonItem, "AudioForge", 1, 0, "EQComparisonItem")  # type: ignore[call-overload]
    qmlRegisterType(QuickMeterItem, "AudioForge", 1, 0, "QuickMeterItem")  # type: ignore[call-overload]
    qmlRegisterType(LevelTrace, "AudioForge", 1, 0, "LevelTrace")  # type: ignore[call-overload]
    qmlRegisterType(LevelRibbon, "AudioForge", 1, 0, "LevelRibbon")  # type: ignore[call-overload]
    qmlRegisterType(EQGraphPreview, "AudioForge", 1, 0, "EQGraphPreview")  # type: ignore[call-overload]
    qmlRegisterUncreatableType(QuickBridge, "AudioForge", 1, 0, "QuickBridge", "The window creates it")  # type: ignore[call-overload]


def install_quick_shell(window) -> None:
    """Make the QML scene the window's content, or the error screen if it cannot start."""

    register_qml_types()
    try:
        bridge = QuickBridge(window)
    except Exception as error:
        _LOG.exception("Qt Quick view could not be prepared")
        show_quick_error(window, "bridge_failed", f"{type(error).__name__}: {error}")
        return
    view = QQuickWidget(window)
    reason = load_scene(view, QML_DIR / "Main.qml", bridge._model["theme"], {"ui": bridge})
    if reason is not None:
        _LOG.error("Qt Quick view failed to load: %s", reason)
        view.deleteLater()
        bridge.release()
        show_quick_error(window, "qml_not_ready", reason)
        return

    window.setCentralWidget(view)
    window.quick_bridge = bridge
    # Every text field of the scene is inside the view, so the palette's shortcut is kept here.
    view.installEventFilter(bridge._model["palette"])
    # A scene that loads can still fail to render, for example without a
    # usable graphics adapter.
    view.sceneGraphError.connect(lambda _error, message: _scene_stopped(window, message))
    _watch_for_renderer(window, view)
    # Left to interpreter shutdown, the scene is torn down in no fixed order
    # and the process can crash on exit; delete it as the event loop ends.
    app = QCoreApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(view.deleteLater)


def _scene_stopped(window, reason: str) -> None:
    """Replace a scene that stopped rendering with the error screen."""

    bridge = window.__dict__.pop("quick_bridge", None)
    if bridge is None:
        return
    _LOG.error("Qt Quick view stopped rendering: %s", reason)
    # Read before the scene is replaced; the export names the backend that failed.
    show_quick_error(window, "scene_graph_error", reason, graphics_api(window))
    bridge.release()


def _watch_for_renderer(window, view: QQuickWidget) -> None:
    """Catch a graphics start that failed silently.

    Qt logs "Failed to get a QRhi", leaves the view Ready, emits no error and
    tries no other backend. The scene graph never initialises; the window may
    start hidden or minimized, so only a window on screen counts.
    """

    if os.environ.get("AUDIOFORGE_SMOKE_TEST"):
        return
    timer = QTimer(view)
    timer.setInterval(RENDERER_CHECK_MS)
    shown = 0

    def check() -> None:
        nonlocal shown
        if view.quickWindow().isSceneGraphInitialized():
            timer.stop()
        elif not window.isVisible() or window.isMinimized():
            shown = 0
        else:
            shown += 1
            if shown >= RENDERER_CHECKS:
                timer.stop()
                _renderer_missing(window)

    timer.timeout.connect(check)
    timer.start()


def _renderer_missing(window) -> None:
    """Quit and relaunch once in software rendering, or show the error screen."""

    bridge = window.__dict__.pop("quick_bridge", None)
    if bridge is None:
        return
    # Read before the scene is replaced; the export names the backend that failed.
    api = graphics_api(window)
    _LOG.error("Qt Quick view has no renderer (%s)", api)
    relaunched = os.environ.get(RENDERER_RELAUNCH_ENV)
    # Live audio stays; a login-started copy has it running while the window is hidden.
    spawn_failed = False
    if api != "software" and not relaunched and not window.processor.is_running():
        # Order: close first (closeEvent saves, unregisters the hotkey, stops audio, and can be
        # cancelled), then spawn (it frees the single-instance guard and the D-Bus name so the
        # child can take them, and re-takes them on failure), and quit only if the spawn worked.
        # All of it runs in one event-loop turn, so a failed spawn can show the window again
        # before Qt acts on the last window having closed; the app then stays up with the error
        # screen.
        window._quitting = True
        if window.close():
            if relaunch_with_software_rendering(
                getattr(window, "single_instance", None), getattr(window, "application_bus", None)
            ):
                QCoreApplication.quit()
                bridge.release()
                return
            spawn_failed = True
            window._quitting = False
            if window._tray_icon is not None:
                window._tray_icon.show()
            if window.config.mute_hotkey:
                window._register_mute_hotkey(window.config.mute_hotkey)
            window.show()
    reason = "The graphics renderer did not start"
    if relaunched:
        reason += ", even with software rendering"
    elif spawn_failed:
        reason += ", and AudioForge could not restart itself with software rendering"
    show_quick_error(window, "no_renderer", reason, api)
    bridge.release()
