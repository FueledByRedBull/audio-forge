"""
Main window for AudioForge application

Adapted from Spectral Workbench project.

DEBUG: Added terminal logging for processor state tracking
"""

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QMainWindow,
    QDialog,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QMenu,
    QSystemTrayIcon,
)
from PySide6.QtCore import QEvent, Qt, QTimer, QRect, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon
import os
import sys
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from .components import plain_label
from .eq_presentation import EQPresentation
from .eq_state import EQState
from .processing_meters import ProcessingMeters
from .shell_state import (
    ActionControl,
    ChoiceState,
    PageState,
    StatusState,
    TextState,
)
from .noise_suppression_state import NoiseSuppressionState
from .limiter_state import LimiterState
from .deesser_state import DeEsserState
from .compressor_state import CompressorState
from .gate_state import GateState
from .level_meter import LevelMeter
from .health import (
    DEVICE_ACTIVITY_BAD_MS,
    DEVICE_ACTIVITY_WARN_MS,
    HEALTH_PEAK_WINDOW_SECONDS,
    RecentStreamHealth,
    advice_for,
    plural,
)
from .health import input_health_state as build_input_health_state
from .health import output_health_state as build_output_health_state
from .calibration_dialog import CalibrationDialog
from .latency_calibration_dialog import (
    LatencyCalibrationDialog,
    engine_config_signature,
)
from .voice_setup_dialog import VoiceSetupDialog
from .first_run_setup_dialog import FirstRunSetupDialog
from .app_bootstrap import run_qt_app
from .device_selection import (
    default_device_index,
    find_identity_index,
    identity_is_persistable,
    preferred_output_index,
    start_processor_for_route,
)
from .capture_session import recovery_paused
from .engine_tasks import EngineTasks, attempt
from .theme import contrast_theme_changed, muted_icon
from .config_history import (
    BoundedConfigurationHistory,
    ConfigurationSnapshot,
    explicit_provenance_after_edit,
)
from .shell_specs import (
    ACTION_SPECS,
    CHOICE_SPECS,
    DROPPED_DIAGNOSTICS_TOOLTIP,
    HEALTH_DECISION_KEYS,
    HEALTH_LAYOUT_KEYS,
    INPUT_CHANNEL_MODE_OPTIONS,
    INPUT_CLEANUP_MODE_OPTIONS,
    PROCESSING_MODE_OPTIONS,
    TEXT_SPECS,
)
from .startup_presets import (
    STARTUP_BUILTIN_PREFIX,
    STARTUP_CUSTOM_PREFIX,
    STARTUP_CUSTOM_FILE_PREFIX,
    normalize_startup_preset_id as _normalize_startup_preset_id,
    startup_builtin_id as _startup_builtin_id,
    startup_custom_file_id as _startup_custom_file_id,
    startup_preset_display_name as _startup_preset_display_name,
)
from .desktop_integration import GlobalMuteHotkey, activate_window, build_tray_tooltip
from . import login_startup
from .. import AudioProcessor, __version__, list_input_devices, list_output_devices
from ..diagnostics_export import (
    build_diagnostics_snapshot,
    diagnostics_filename,
    sanitize_setup_result,
    write_diagnostics_snapshot,
)
from ..config import (
    Preset,
    DeviceIdentity,
    GateSettings,
    RNNoiseSettings,
    DeEsserSettings,
    CompressorSettings,
    LimiterSettings,
    PresetValidationError,
    save_preset,
    load_preset,
    import_preset,
    get_presets_dir,
    list_presets,
    BUILTIN_PRESETS,
    save_config,
    load_config,
    build_device_route_key,
    DevicePresetBinding,
    InputDevicePreference,
    build_input_device_preference_key,
    coerce_device_identity,
    legacy_latency_profile_key,
    LatencyCalibrationProfile,
)


def _startup_preset_selection(
    preset_id: str,
    custom_presets: list[tuple[str, Path]],
) -> tuple[str, tuple[str, Path] | None]:
    """Resolve custom startup IDs by filename and migrate older name IDs."""
    normalized = _normalize_startup_preset_id(
        preset_id, tuple(name for name, _filepath in custom_presets)
    )
    if normalized.startswith(STARTUP_CUSTOM_FILE_PREFIX):
        identity = normalized[len(STARTUP_CUSTOM_FILE_PREFIX) :]
        matches = [item for item in custom_presets if item[1].name == identity]
        return normalized, matches[0] if len(matches) == 1 else None
    if not normalized.startswith(STARTUP_CUSTOM_PREFIX):
        return normalized, None

    identity = normalized[len(STARTUP_CUSTOM_PREFIX) :]
    matches = [item for item in custom_presets if item[0] == identity]
    if len(matches) != 1:
        return normalized, None
    name, filepath = matches[0]
    return _startup_custom_file_id(filepath.name), (name, filepath)


def _route_custom_preset_selection(
    preset_id: str,
    custom_presets: list[tuple[str, Path]],
) -> tuple[str, tuple[str, Path]] | None:
    """Resolve route preset IDs, rejecting ambiguous legacy filename/name IDs."""
    if preset_id.startswith(STARTUP_CUSTOM_FILE_PREFIX):
        identity = preset_id[len(STARTUP_CUSTOM_FILE_PREFIX) :]
        matches = [item for item in custom_presets if item[1].name == identity]
    elif preset_id.startswith(STARTUP_CUSTOM_PREFIX):
        identity = preset_id[len(STARTUP_CUSTOM_PREFIX) :]
        matches = [
            item
            for item in custom_presets
            if item[0] == identity or item[1].name == identity
        ]
    else:
        return None

    if len(matches) != 1:
        return None
    name, filepath = matches[0]
    return _startup_custom_file_id(filepath.name), (name, filepath)


# Enable debug logging
DEBUG = False
logger = logging.getLogger(__name__)

INPUT_PHASE_WARNING_CORRELATION = -0.75
PHASE_FIX_LABELS = {
    "delay_left": "delaying the left channel",
    "delay_right": "delaying the right channel",
    "polarity_flip": "flipping one channel's polarity",
}
DEFAULT_MUTE_HOTKEY = "Ctrl+Alt+M"
DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 960
MINIMUM_WINDOW_WIDTH = 900
MINIMUM_WINDOW_HEIGHT = 640
# The pages in navigation order; the shared page state and the page stack follow it.
PAGE_NAMES = ("Mic", "Health", "Settings")
NO_DESTINATION_WARNING = "Warning: No output device is selected. Choose one in the Output list, then press Start."
# Windows allows a program a few seconds to end a session, and the save comes after this wait.
ENGINE_CALL_WAIT_MS = 3_000
# Set on the app while Qt asks to commit data for a sign-out or shutdown; the dialogs read it too.
SESSION_ENDING_PROPERTY = "audioforge_session_ending"
SAVE_PROBLEM = "Settings could not be saved, so recent changes may be lost when AudioForge closes."
NOISE_SELECTION_DROPPED_WARNING = (
    "The noise suppression model you chose could not be applied, and the previous one is still "
    "running. Choose it again. If it keeps failing, check the model files or use Help > Export Diagnostics."
)


# The engine reports this age until a device has made its first callback.
NO_CALLBACK_YET = 2**64 - 1
# A device that has made no callback this long after it was first seen waiting is a
# warning. Any engine restart resets the callback ages, so the wait is timed from the
# moment the not-started state appears, not from when processing began.
DEVICE_STARTUP_GRACE_SECONDS = 5.0
# How the engine's supervisor words a stall restart's reason (callback ages, in ms).
STALL_REASON_PREFIX = "input_cb_age_ms="
# The native getters the meters and Health read; an older extension module lacks them.
METER_GETTERS = (
    "take_input_peak_since_read",
    "take_output_pre_limiter_peak_since_read",
    "take_output_peak_since_read",
    "get_input_full_scale_count",
    "get_output_full_scale_count",
)


def _ago_text(age_ms: int | None, *, output: bool = False) -> str:
    if age_ms is None:
        return "--"
    if age_ms == NO_CALLBACK_YET:
        return "has not started playing sound yet" if output else "has not started sending sound yet"
    return f"{age_ms} ms ago"


def _engine_in_use(window) -> bool:
    """Whether a call is out on the engine worker; the meters and Health wait for it."""
    return bool(window.__dict__.get("_engine_busy") or window.__dict__.get("_recovery_polling"))


def _session_is_ending() -> bool:
    app = QGuiApplication.instance()
    return app is not None and bool(app.property(SESSION_ENDING_PROPERTY))


def _end_session_query() -> None:
    app = QGuiApplication.instance()
    if app is not None:
        app.setProperty(SESSION_ENDING_PROPERTY, False)


def _device_label(identity: DeviceIdentity, devices: list) -> str:
    """A picker label; the pickers cut it off at the end, so the markers come before the name."""
    ordinal = identity.name_ordinal
    twin = f"[#{ordinal + 1}] " if ordinal is not None and sum(
        item.name == identity.name for item in devices
    ) > 1 else ""
    return twin + ("(Default) " if identity.is_default else "") + identity.name


def _list_devices() -> tuple[Any, Any]:
    """Both device lists, or for each the exception that stopped it; the worker runs this."""
    return attempt(list_input_devices), attempt(list_output_devices)


def _output_muted(window) -> bool:
    """Whether a mute the engine accepted is silencing the output (not one still pending).

    The Output row, the OUT meter and the status dot all read this, so they agree.
    """
    return not window.__dict__.get("_output_mute_error") and bool(
        window.__dict__.get("user_muted", False) or window.__dict__.get("_temporary_mute_reasons", set())
    )


def _level_reading(processor, name: str) -> float | None:
    """One finite number from the audio engine, or ``None`` when it cannot be read."""
    try:
        value = float(getattr(processor, name)())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _exec_and_release(dialog: QDialog) -> int:
    """Run a modal dialog, then release it: parented to the window, it would otherwise live as long as the window."""
    try:
        return dialog.exec()
    finally:
        dialog.deleteLater()


def _fit_window_geometry_to_screens(
    geometry: dict[str, int] | None,
    available_geometries: list[QRect],
) -> QRect | None:
    """Fit restored geometry entirely inside one available screen."""
    screens = [QRect(rect) for rect in available_geometries if not rect.isEmpty()]
    if not screens:
        return None

    if geometry is None:
        target = screens[0]
        width = min(DEFAULT_WINDOW_WIDTH, target.width())
        height = min(DEFAULT_WINDOW_HEIGHT, target.height())
        return QRect(
            target.x() + (target.width() - width) // 2,
            target.y() + (target.height() - height) // 2,
            width,
            height,
        )

    requested = QRect(
        int(geometry["x"]),
        int(geometry["y"]),
        int(geometry["width"]),
        int(geometry["height"]),
    )
    intersection_areas = []
    for screen in screens:
        intersection = requested.intersected(screen)
        intersection_areas.append(intersection.width() * intersection.height())
    target_index = max(range(len(screens)), key=intersection_areas.__getitem__)
    has_visible_area = intersection_areas[target_index] > 0
    target = screens[target_index] if has_visible_area else screens[0]

    minimum_width = min(MINIMUM_WINDOW_WIDTH, target.width())
    minimum_height = min(MINIMUM_WINDOW_HEIGHT, target.height())
    width = min(max(requested.width(), minimum_width), target.width())
    height = min(max(requested.height(), minimum_height), target.height())

    if not has_visible_area:
        x = target.x() + (target.width() - width) // 2
        y = target.y() + (target.height() - height) // 2
    else:
        x = min(max(requested.x(), target.x()), target.x() + target.width() - width)
        y = min(max(requested.y(), target.y()), target.y() + target.height() - height)
    return QRect(x, y, width, height)


class MainWindow(QMainWindow):
    """Main application window for AudioForge."""

    # TextState fields are Qt properties, which type checkers cannot read.
    shell_texts: dict[str, Any]
    input_choice: ChoiceState
    output_choice: ChoiceState
    processing_mode_choice: ChoiceState
    input_channel_mode_choice: ChoiceState
    input_cleanup_mode_choice: ChoiceState
    refresh_btn_action: QAction
    start_btn_action: QAction
    stop_btn_action: QAction
    user_mute_checkbox_action: QAction
    _undo_auto_eq_button_action: QAction
    redo_button_action: QAction
    test_sound_button_action: QAction
    auto_eq_button_action: QAction
    auto_voice_setup_button_action: QAction
    HEALTH_PAGE_INDEX = 1
    SETTINGS_PAGE_INDEX = 2

    def __init__(self, *, login_startup: bool = False):
        super().__init__()
        self.setWindowTitle("AudioForge - Microphone Audio Processor")

        # Create audio processor
        self.processor = AudioProcessor()
        stop_audio = self._stop_audio_on_worker
        self.limiter_state = LimiterState(self.processor, self, stop_audio=stop_audio)
        self.deesser_state = DeEsserState(self.processor, self, stop_audio=stop_audio)
        self.compressor_state = CompressorState(self.processor, self, stop_audio=stop_audio)
        self.gate_state = GateState(self.processor, self, stop_audio=stop_audio)
        self.eq_state = EQState(self.processor, self, stop_audio=stop_audio)
        self.noise_suppression_state = NoiseSuppressionState(self.processor, self, stop_audio=stop_audio)
        self.processing_meters = ProcessingMeters(self.processor, self.gate_state, self)
        self.input_meter = LevelMeter("IN", show_scale=True, name="Input level meter", parent=self)
        self.output_meter = LevelMeter("OUT", show_scale=True, name="Output level meter", parent=self)
        # Load configuration
        self.config = load_config()
        self._login_startup = login_startup
        self._login_startup_deadline: float | None = None
        self._login_startup_next_retry = 0.0
        self._login_startup_message = ""
        # Capture the persisted IDs before normal UI restoration can enrich a
        # name-only selection from today's enumeration.
        self._login_startup_route = (
            self.config.last_input_device_identity, self.config.last_output_device_identity
        )
        self._startup_restore_ready = False
        self.current_preset_path = None
        self.current_preset_name = "Default"
        self.preset_modified = False
        self._saved_preset_payload: str | None = None
        self._last_preset_identity_persisted = True
        self._temporary_mute_reasons: set[str] = set()
        self._output_mute_error: str | None = None
        self.user_muted = bool(getattr(self.config, "user_muted", False))

        # Bounded settings history with transient calibration evidence;
        # live audio buffers and realtime processor state stay out.
        self._configuration_history = BoundedConfigurationHistory(limit=50)
        self._history_ready = False
        self._history_replaying = False
        self._history_transaction_depth = 0
        self._current_value_provenance: dict[str, str] = {}
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(250)
        self._history_timer.timeout.connect(self._commit_pending_configuration_snapshot)
        self._undo_action = None
        self._redo_action = None
        self._calibration_dialog_open = False
        self._processing = False
        self._engine = EngineTasks(self)
        self._engine_busy: str | None = None  # the call the Start, Stop and Refresh controls wait on
        self._recovery_polling = False
        self._device_wait_since: float | None = None
        self._stream_health = RecentStreamHealth()
        self._last_backend_warning = None
        # The engine's dropped-selection count when last read; a new engine counts from zero.
        self._noise_drops_seen = 0
        # The latest Voice Setup or Auto-EQ outcome, kept for the diagnostics export.
        self.last_setup_result: dict | None = None
        self._last_output_underrun_total = 0
        self._last_output_true_peak_event_count = 0
        self._last_input_phase_warning_count = 0
        # Why the Qt Quick view could not run, while the error screen shows; the
        # packaged smoke test and the diagnostics export read it.
        self.quick_view_failure: str | None = None
        self.quick_view_graphics_api = "not_applicable"
        self.single_instance = None  # set by the app bootstrap; the relaunch releases it
        self._problems: dict[str, str] = {}  # on the warning banner until resolved
        self._quitting = False
        self._shut_down = False
        self._quit_dont_ask = False
        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_icons: tuple[QIcon, QIcon] = (QIcon(), QIcon())  # heard, muted
        self._tray_menu: QMenu | None = None
        self._tray_mute_action: QAction | None = None
        self._mute_hotkey: GlobalMuteHotkey | None = None
        self._close_to_tray_action: QAction | None = None
        self._mute_hotkey_action: QAction | None = None

        # Shared state first: the view only renders it.
        self._setup_shell_models()
        self._setup_menubar()
        self._setup_options_menu()
        for state in (self.limiter_state, self.deesser_state, self.compressor_state, self.gate_state, self.eq_state, self.noise_suppression_state):
            state.writeFailed.connect(
                lambda message: self.status_message.showMessage(message, 7000)
            )
            state.recoveryFailed.connect(
                lambda _message: self.set_temporary_output_mute(True, "configuration")
            )
        for state in (self.limiter_state, self.deesser_state, self.compressor_state, self.eq_state):
            state.set_settings({})
        try:
            self.gate_state.set_settings({})
        except RuntimeError:
            logger.exception("Initial gate settings could not be applied")

        self._finish_shell()
        self._setup_desktop_integration()
        self._apply_output_mute()

        # Populate device lists
        self._refresh_devices()

        # Connect device change signals for persistence
        self.input_choice.currentIndexChanged.connect(
            lambda _index: self._on_device_changed(self.input_choice)
        )
        self.output_choice.currentIndexChanged.connect(
            lambda _index: self._on_device_changed(self.output_choice)
        )
        self.input_channel_mode_choice.currentIndexChanged.connect(
            self._on_input_channel_mode_changed
        )
        self.input_cleanup_mode_choice.currentIndexChanged.connect(
            self._on_input_cleanup_mode_changed
        )

        self._apply_initial_window_geometry()

        # Restore settings from config
        self._restore_from_config()
        self._initialize_configuration_history()
        self._connect_configuration_history_inputs()
        self._show_config_banner()
        if self.config.load_warning:
            warning = self.config.load_warning
            QTimer.singleShot(
                0,
                lambda: self.status_message.showMessage(warning),
            )
        if not self._login_startup:
            QTimer.singleShot(0, self._maybe_show_first_run_setup)

        # Meter update timer (60 FPS)
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._update_meters)
        self.meter_timer.setInterval(16)
        self._sync_meter_timer()

        # Slower diagnostics/recovery service timer.
        self.diagnostics_timer = QTimer(self)
        self.diagnostics_timer.timeout.connect(self._update_diagnostics)
        self.diagnostics_timer.start(250)

    def _apply_initial_window_geometry(self) -> None:
        primary_screen = QGuiApplication.primaryScreen()
        ordered_screens = []
        if primary_screen is not None:
            ordered_screens.append(primary_screen)
        ordered_screens.extend(
            screen
            for screen in QGuiApplication.screens()
            if screen is not primary_screen
        )
        # setGeometry() places the client area; leave room for the title bar.
        # The frame margins are known once the native window exists.
        self.winId()
        frame = self.windowHandle().frameMargins()
        fitted = _fit_window_geometry_to_screens(
            self.config.window_geometry,
            [screen.availableGeometry().marginsRemoved(frame) for screen in ordered_screens],
        )
        if fitted is None:
            self.setMinimumSize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)
            self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
            return

        self.setMinimumSize(
            min(MINIMUM_WINDOW_WIDTH, fitted.width()),
            min(MINIMUM_WINDOW_HEIGHT, fitted.height()),
        )
        self.setGeometry(fitted)
        saved = self.config.window_geometry
        if saved is not None:
            geometry = {
                "x": fitted.x(),
                "y": fitted.y(),
                "width": fitted.width(),
                "height": fitted.height(),
            }
            if saved.get("maximized"):
                geometry["maximized"] = True
                self.setWindowState(Qt.WindowState.WindowMaximized)
            self.config.window_geometry = geometry

    def _on_page_changed(self, index: int) -> None:
        # The shortcut can be removed or disabled outside the app.
        if index == self.SETTINGS_PAGE_INDEX and hasattr(self, "_login_startup_action"):
            self._refresh_login_startup_action()
            for row in self._settings_value_rows:
                row.changed.emit()

    def _settings_card(self, title: str, menu: QMenu, *, skip: QMenu | None = None) -> dict:
        """List a menu's entries as the rows the settings page shows: switches, actions and submenus."""

        rows = []
        # QAction.menu() hands PySide a wrapper that can delete the menu, so
        # submenus are looked up from the parent instead.
        submenus = menu.findChildren(
            QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly
        )
        for action in menu.actions():
            submenu = next((m for m in submenus if m.menuAction() is action), None)
            if action.isSeparator() or (skip is not None and submenu is skip):
                continue
            value = self._settings_choice_values.get(submenu) if submenu is not None else None
            proxy = ActionControl(action, self, menu=submenu, value=value)
            if value is not None and submenu is not None:
                submenu.triggered.connect(proxy.changed)
                self._settings_value_rows.append(proxy)
            rows.append({"kind": "menu" if submenu is not None else "toggle" if action.isCheckable() else "action",
                         "label": "", "proxy": proxy})
        return {"title": title.upper(), "rows": rows}

    def _setup_shell_models(self) -> None:
        """Create the state the view renders; workflows own it outside the view."""
        self.shell_texts = {
            key: TextState(self, text=spec.text, name=spec.name, tip=spec.tip, shown=spec.shown)
            for key, spec in TEXT_SPECS.items()
        }
        self.shell_texts["preset_status_label"].update(
            data={"name": self.current_preset_name, "modified": bool(self.preset_modified)},
        )
        self.status_message = StatusState(
            self, f"Sample Rate: {self.processor.sample_rate()} Hz | Status: Ready"
        )
        self._reset_health_labels()
        self.eq_presentation = EQPresentation(self.eq_state, self)

        for key, choice_spec in CHOICE_SPECS.items():
            state = ChoiceState(self, name=choice_spec.name, tip=choice_spec.tip, locked_tip=choice_spec.locked_tip)
            for label, data in choice_spec.options:
                state.addItem(label, data)
            setattr(self, key + "_choice", state)
        self.processing_mode_choice.currentIndexChanged.connect(self._on_processing_mode_changed)

        commands = {
            "refresh_btn": self._on_refresh_clicked, "start_btn": self._on_start_clicked,
            "stop_btn": self._on_stop_clicked, "_undo_auto_eq_button": self.undo_configuration,
            "redo_button": self.redo_configuration, "test_sound_button": self._on_test_my_sound_clicked,
            "auto_eq_button": self._on_auto_eq_clicked,
            "auto_voice_setup_button": self._on_auto_voice_setup_clicked,
            "user_mute_checkbox": self._on_user_mute_toggled,
        }
        for key, action_spec in ACTION_SPECS.items():
            action = QAction(action_spec.text, self)
            action.setToolTip(action_spec.tip)
            action.setEnabled(action_spec.enabled)
            action.setCheckable(action_spec.checkable)
            if key == "user_mute_checkbox":
                action.setChecked(self.user_muted)
            setattr(self, key + "_action", action)
            (action.toggled if action_spec.checkable else action.triggered).connect(commands[key])
        self.shell_page = PageState(self, len(PAGE_NAMES))
        self.shell_page.selected.connect(self._on_page_changed)

    def _finish_shell(self) -> None:
        """Own the menu actions, then show the Qt Quick view or the error screen."""

        menubar = self.menuBar()
        for menu in menubar.findChildren(
            QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            # A hidden menu bar disables its shortcuts unless the window owns
            # the actions too.
            self.addActions(menu.actions())
        menubar.hide()

        # Rows that open a menu of choices show the choice that is current.
        def checked(menu: QMenu, *, skip: QAction | None = None) -> Callable[[], str]:
            return lambda: next(
                (plain_label(a.text()) for a in menu.actions() if a.isChecked() and a is not skip), ""
            )

        self._settings_choice_values = {
            self._startup_preset_menu: checked(self._startup_preset_menu),
            self._device_preset_menu: checked(
                self._device_preset_menu, skip=self.auto_apply_device_presets_action
            ),
        }
        self._settings_value_rows: list[ActionControl] = []
        # The tray submenu is mostly switches, so it gets a card of its own.
        self.shell_settings_cards = [
            self._settings_card("Options", self._options_menu, skip=self._tray_background_menu),
            self._settings_card("Tray and background", self._tray_background_menu),
            self._settings_card("Help", self._help_menu),
        ]

        from .quick_shell import install_quick_shell

        install_quick_shell(self)

    def _set_health_chip(self, key: str, text: str, state: str) -> None:
        self.shell_texts[key].update(text=text, status=state)

    def _reset_health_labels(self) -> None:
        startup_message = self.__dict__.get("_login_startup_message", "")
        self._set_health_chip(
            "health_summary_label",
            startup_message or TEXT_SPECS["health_summary_label"].text,
            "warn" if startup_message else "idle",
        )
        for key in HEALTH_DECISION_KEYS + HEALTH_LAYOUT_KEYS:
            self._set_health_chip(key, TEXT_SPECS[key].text, "idle")
        # What the last reading explained must not outlive it.
        self.shell_texts["health_advice_label"].update(text=TEXT_SPECS["health_advice_label"].text)
        for key in ("health_summary_label", "recovery_diag_label", "dropped_label"):
            self.shell_texts[key].update(tip=TEXT_SPECS[key].tip)

    @staticmethod
    def _diag_token(label: str, value) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return f"{label}:{'Y' if value else 'N'}"
        if isinstance(value, float):
            return f"{label}:{value:.1f}"
        return f"{label}:{value}"

    @classmethod
    def _extend_diag_tokens(
        cls, tokens: list[str], diagnostics: dict, keys: list[tuple[str, str]]
    ) -> None:
        for key, label in keys:
            token = cls._diag_token(label, diagnostics.get(key))
            if token is not None:
                tokens.append(token)

    @staticmethod
    def _is_valid_input_channel_mode(mode: object) -> bool:
        return isinstance(mode, str) and any(
            mode == option_mode for _label, option_mode in INPUT_CHANNEL_MODE_OPTIONS
        )

    def _select_input_channel_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_channel_mode(mode) else "phase_safe_mono"
        index = self.input_channel_mode_choice.findData(target)
        self.input_channel_mode_choice.setCurrentIndex(index if index >= 0 else 0)

    def _apply_input_channel_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_channel_mode(mode) else "phase_safe_mono"
        try:
            if hasattr(self.processor, "set_input_channel_mode"):
                self.processor.set_input_channel_mode(target)
        except Exception:
            logger.warning("Failed to apply input channel mode", exc_info=True)

    @staticmethod
    def _is_valid_input_cleanup_mode(mode: object) -> bool:
        return isinstance(mode, str) and any(
            mode == option_mode for _label, option_mode in INPUT_CLEANUP_MODE_OPTIONS
        )

    def _select_input_cleanup_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_cleanup_mode(mode) else "off"
        index = self.input_cleanup_mode_choice.findData(target)
        self.input_cleanup_mode_choice.setCurrentIndex(index if index >= 0 else 0)

    def _apply_input_cleanup_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_cleanup_mode(mode) else "off"
        try:
            if hasattr(self.processor, "set_input_cleanup_mode"):
                self.processor.set_input_cleanup_mode(target)
        except Exception:
            logger.warning("Failed to apply input cleanup mode", exc_info=True)

    @staticmethod
    def _is_valid_processing_mode(mode: object) -> bool:
        return isinstance(mode, str) and any(
            mode == option_mode for _label, option_mode in PROCESSING_MODE_OPTIONS
        )

    def _processing_mode(self) -> str:
        """Return the selected mode, or its last applied value during initialization."""
        combo = self.__dict__.get("processing_mode_choice")
        if combo is not None:
            mode = combo.currentData()
            if self._is_valid_processing_mode(mode):
                return str(mode)
        return self.__dict__.get("_applied_processing_mode", "normal")

    def _set_processing_mode(self, mode: str, *, notify: bool = False) -> None:
        target = mode if self._is_valid_processing_mode(mode) else "normal"
        combo = self.__dict__.get("processing_mode_choice")
        if combo is not None:
            index = combo.findData(target)
            if index >= 0 and combo.currentIndex() != index:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

        try:
            if hasattr(self.processor, "set_raw_monitor_enabled"):
                self.processor.set_raw_monitor_enabled(target == "raw")
            if hasattr(self.processor, "set_bypass"):
                self.processor.set_bypass(target == "bypass")
        except Exception as error:
            raise RuntimeError(f"Processing mode could not be applied: {error}") from error
        self._applied_processing_mode = target

        if notify:
            message = {
                "normal": "Processing active",
                "bypass": (
                    "Voice effects bypassed; input conditioning and configured "
                    "output protection remain"
                ),
                "raw": "Raw monitor enabled for this session; it is not stored in presets",
            }[target]
            self.status_message.showMessage(message)
        update_summary = getattr(self, "_update_session_summary", None)
        if callable(update_summary):
            update_summary()

    def _on_processing_mode_changed(self, _index: int) -> None:
        previous = self.__dict__.get("_applied_processing_mode", "normal")
        try:
            self._set_processing_mode(self._processing_mode(), notify=True)
        except RuntimeError as error:
            try:
                self._set_processing_mode(previous)
            except RuntimeError:
                self.set_temporary_output_mute(True, "processing_mode_error")
                self.status_message.showMessage(f"{error}; output muted because restoration failed")
                return
            self.status_message.showMessage(f"{error}; previous mode restored", 6000)
        else:
            self.set_temporary_output_mute(False, "processing_mode_error")
            self._queue_configuration_snapshot()

    def _apply_output_mute(self) -> None:
        """Apply user and temporary mute state without either one clearing the other."""
        muted = bool(
            self.__dict__.get("user_muted", False)
            or self.__dict__.get("_temporary_mute_reasons", set())
        )
        try:
            self.processor.set_output_mute(muted)
            self._output_mute_error = None
        except Exception:
            self._output_mute_error = "unavailable"
            logger.warning("Failed to apply output mute state", exc_info=True)
        tray_mute_action = self.__dict__.get("_tray_mute_action")
        if tray_mute_action is not None:
            tray_mute_action.blockSignals(True)
            tray_mute_action.setChecked(bool(self.user_muted))
            tray_mute_action.blockSignals(False)
        self._update_session_summary()

    def set_temporary_output_mute(
        self, muted: bool, reason: str = "calibration"
    ) -> None:
        """Set one temporary mute owner; releasing it never changes user mute."""
        reason = str(reason or "temporary")
        reasons = self.__dict__.setdefault("_temporary_mute_reasons", set())
        if muted:
            reasons.add(reason)
        else:
            reasons.discard(reason)
        self._apply_output_mute()

    def _on_user_mute_toggled(self, checked: bool) -> None:
        self.user_muted = bool(checked)
        self._apply_output_mute()
        saved = False
        if self.__dict__.get("config") is not None:
            self.config.user_muted = self.user_muted
            saved = self._save_config_safely()
        if self.__dict__.get("_output_mute_error"):
            message = "Output mute change could not be applied"
            if saved:
                message += "; preference saved for the next start"
        elif self.user_muted or self.__dict__.get("_temporary_mute_reasons", set()):
            message = "Output muted"
        else:
            message = "Output unmuted"
        if not saved:
            message += "; preference could not be saved"
        self.status_message.showMessage(message, 6000 if not saved else 3000)

    def _update_session_summary(self) -> None:
        """Keep the compact route/preset/transmission summary current."""
        # The shared state this reads exists once _setup_shell_models has built the route choices.
        if "output_choice" not in self.__dict__:
            return
        input_name = self._device_name_from_identity(
            self._combo_device_identity(self.input_choice)
        ) or "Default input"
        output_name = self._device_name_from_identity(
            self._combo_device_identity(self.output_choice)
        ) or "Select a destination"
        self.shell_texts["route_status_label"].update(text=f"Route: {input_name} -> {output_name}")
        state = "unsaved changes" if self.preset_modified else "saved"
        self.shell_texts["preset_status_label"].update(
            text=f"Preset: {self.current_preset_name or 'Default'} ({state})",
            data={"name": self.current_preset_name, "modified": bool(self.preset_modified)},
        )
        running = bool(getattr(self, "processor", None) and self.processor.is_running())
        mode = self._processing_mode()
        mode_label = next(label for label, value in PROCESSING_MODE_OPTIONS if value == mode)
        transmission = f"{'Running' if running else 'Stopped'} / {mode_label}"
        muted = _output_muted(self)
        pending = bool(self.__dict__.get("_output_mute_error"))
        if pending:
            transmission += " / Mute pending"
        elif muted:
            transmission += " / Muted"
        # The status dot follows this state: green while heard, amber while muted.
        self.shell_texts["transmission_status_label"].update(
            text=f"Transmission: {transmission}",
            status="idle" if not running else "warn" if muted or pending else "ok",
        )
        self.output_meter.set_muted(muted)
        tray = self.__dict__.get("_tray_icon")
        if tray is not None:
            tray_muted = bool(self.user_muted or self._temporary_mute_reasons)
            tooltip = build_tray_tooltip(
                processing=running,
                muted=tray_muted,
                health=self.shell_texts["health_summary_label"].text.removeprefix("Health: "),
                route=(input_name, output_name),
            )
            # This runs every diagnostics tick; each assignment is a shell notification.
            if tooltip != self.__dict__.get("_tray_tooltip"):
                self._tray_tooltip = tooltip
                tray.setToolTip(tooltip)
            if tray_muted != self.__dict__.get("_tray_icon_muted", False):
                self._tray_icon_muted = tray_muted
                tray.setIcon(self._tray_icons[tray_muted])
        if self.__dict__.get("_history_ready") and not self.__dict__.get("_calibration_dialog_open"):
            self._refresh_calibration_status()

    def _refresh_calibration_status(self) -> None:
        from .calibration_history import calibration_summary

        self.shell_texts["calibration_status_label"].update(text=calibration_summary(self))

    def _sync_calibration_evidence(self, *, force_reset: bool = False) -> None:
        """Apply calibration-derived DSP state after an explicit state change.

        Rendering calibration status is deliberately read-only. This method is
        called by configuration and route commands so a matching capture keeps
        its noise reference after a tone edit, while a changed route or
        correction clears it explicitly.
        """
        if self.__dict__.get("_calibration_dialog_open") or self.__dict__.get(
            "_history_replaying"
        ):
            return
        if not hasattr(self, "config") or not hasattr(self, "compressor_state"):
            return
        from .calibration_history import current_calibration

        records = getattr(self.config, "calibration_results", ())
        if not records:
            if not force_reset and not self.__dict__.get(
                "_persisted_calibration_active", False
            ):
                return
            result = None
        else:
            result = current_calibration(self, "full_voice_setup")
        reliability = result.noise_reference_reliability if result is not None else 0.0
        current = self.compressor_state.get_settings(
            include_calibration=True
        )
        current_reliability = current["noise_reference_reliability"]
        if current_reliability != reliability:
            self.compressor_state.set_settings(
                {"noise_reference_reliability": reliability}
            )
        self._persisted_calibration_active = result is not None

    def _set_preset_identity(
        self,
        preset: Preset,
        *,
        path: Path | None = None,
        preset_id: str | None = None,
        persist_last_used: bool = True,
    ) -> None:
        self.current_preset_name = str(preset.name or "Default")
        self.current_preset_description = preset.description
        self.current_preset_path = path
        self._saved_preset_payload = self._preset_payload(preset)
        self._saved_processing_mode = "bypass" if preset.bypass else "normal"
        if persist_last_used:
            if preset_id is not None:
                self.config.last_preset = preset_id
            elif path is not None:
                self.config.last_preset = str(path)
        self._mirror_preset_record()
        self._set_preset_modified()

    def _mirror_preset_record(self) -> dict[str, Any]:
        """Copy the preset the window shows into the record that history snapshots share."""
        record = self.__dict__.setdefault("_preset_record", {})
        record.update(
            name=self.current_preset_name,
            description=self.__dict__.get("current_preset_description", ""),
            path=self.current_preset_path,
            saved_payload=self._saved_preset_payload,
            saved_mode=self.__dict__.get("_saved_processing_mode", "normal"),
            last_preset=self.config.last_preset,
        )
        return record

    def _restore_preset_record(self, record: dict[str, Any]) -> None:
        """Show the preset that went with a history snapshot, and remember it for the next launch."""
        self._preset_record = record
        self.current_preset_name = record["name"]
        self.current_preset_description = record["description"]
        self.current_preset_path = record["path"]
        self._saved_preset_payload = record["saved_payload"]
        self._saved_processing_mode = record["saved_mode"]
        if self.config.last_preset != record["last_preset"]:
            self.config.last_preset = record["last_preset"]
            self._save_config_safely()

    @staticmethod
    def _preset_payload(preset: Preset) -> str:
        payload = preset.to_dict()
        for key in ("name", "description", "version", "value_provenance"):
            payload.pop(key, None)
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _set_preset_modified(self, modified: bool | None = None) -> None:
        if modified is None:
            saved_payload = self.__dict__.get("_saved_preset_payload")
            if saved_payload is None:
                modified = False
            else:
                try:
                    modified = (
                        self._preset_payload(self._get_current_preset())
                        != saved_payload
                        or self._processing_mode()
                        != self.__dict__.get("_saved_processing_mode", "normal")
                    )
                except (PresetValidationError, TypeError, ValueError):
                    modified = True
        self.preset_modified = bool(modified)
        self._update_session_summary()

    def _save_config_safely(self) -> bool:
        """Persist config without letting a write failure disrupt the UI."""
        try:
            saved = bool(save_config(self.config))
        except Exception:
            logger.warning("Could not save AudioForge configuration", exc_info=True)
            saved = False
        self._set_problem("save", None if saved else SAVE_PROBLEM)
        return saved

    def _set_problem(self, key: str, message: str | None) -> None:
        """Keep a settings or startup problem on the warning banner until it is resolved."""
        if "shell_texts" not in self.__dict__:  # the banner exists once _setup_shell_models has run
            return
        if message:
            self._problems[key] = message
        else:
            self._problems.pop(key, None)
        self._show_config_banner()

    def _report_problem(self, key: str, message: str, timeout: int = 0) -> None:
        self.status_message.showMessage(message, timeout)
        self._set_problem(key, message)

    def _report_preset_problem(
        self, key: str, preset_id: str, message: str, timeout: int = 0
    ) -> None:
        """A problem that ends when preset ``preset_id`` loads, not when any preset does."""
        self.__dict__.setdefault("_failed_presets", {})[key] = preset_id
        self._report_problem(key, message, timeout)

    def _show_config_banner(self) -> None:
        text = "\n".join(filter(None, (self.config.load_warning, *self._problems.values())))
        self.shell_texts["config_warning_banner"].update(text=text, shown=bool(text))

    def _save_ui_state(self) -> bool:
        saved = self._save_config_safely()
        if not saved:
            self.status_message.showMessage("Could not save window settings", 5000)
        return saved

    @staticmethod
    def _combo_device_identity(combo: ChoiceState) -> DeviceIdentity | None:
        return coerce_device_identity(combo.currentData())

    @staticmethod
    def _device_name_from_identity(identity: DeviceIdentity | None) -> str:
        return identity.name if identity is not None else ""

    @staticmethod
    def _identity_from_device_info(device: object, direction: str) -> DeviceIdentity:
        """Copy the native enumeration record into the persisted schema."""
        return DeviceIdentity(
            name=str(getattr(device, "name", "")),
            is_default=bool(getattr(device, "is_default", False)),
            endpoint_id=str(getattr(device, "endpoint_id", "") or ""),
            host_api=str(getattr(device, "host_api", "") or ""),
            direction=direction,
            sample_rate=getattr(device, "sample_rate", None),
            channels=getattr(device, "channels", None),
            name_ordinal=getattr(device, "name_ordinal", None),
        )

    @staticmethod
    def _find_combo_index_by_identity(
        combo: ChoiceState, identity: DeviceIdentity | None
    ) -> int:
        return find_identity_index(MainWindow._combo_identities(combo), identity)

    def _select_combo_identity(
        self,
        combo: ChoiceState,
        identity: DeviceIdentity | None,
    ) -> bool:
        index = self._find_combo_index_by_identity(combo, identity)
        if index >= 0:
            combo.setCurrentIndex(index)
            return True
        return False

    @staticmethod
    def _default_combo_index(combo: ChoiceState) -> int:
        return default_device_index(MainWindow._combo_identities(combo))

    @staticmethod
    def _preferred_output_combo_index(combo: ChoiceState) -> int:
        return preferred_output_index(MainWindow._combo_identities(combo))

    @staticmethod
    def _combo_identities(combo: ChoiceState) -> list[DeviceIdentity | None]:
        return [coerce_device_identity(combo.itemData(i)) for i in range(combo.count())]

    def _device_selection_to_name(self, combo: ChoiceState) -> str:
        identity = self._combo_device_identity(combo)
        if identity is not None:
            return identity.name
        value = combo.currentData()
        if isinstance(value, str):
            return value
        return ""

    def _current_input_preference_key(self) -> str | None:
        identity = self._combo_device_identity(self.input_choice)
        if identity is None or not identity_is_persistable(
            self._combo_identities(self.input_choice), identity
        ):
            return None
        return build_input_device_preference_key(identity)

    def _input_preference_for_current_route(self) -> InputDevicePreference:
        route_key = self._current_device_route_key()
        route_preferences = getattr(self.config, "route_input_preferences", {})
        if route_key is not None and route_key in route_preferences:
            return route_preferences[route_key]

        device_key = self._current_input_preference_key()
        device_preferences = getattr(self.config, "input_device_preferences", {})
        if device_key is not None and device_key in device_preferences:
            return device_preferences[device_key]

        # An unmatched route or device is unbound; global settings never
        # transfer to a different endpoint.
        return InputDevicePreference()

    def _apply_input_preferences_for_current_route(self) -> None:
        preference = self._input_preference_for_current_route()
        self._select_input_channel_mode(preference.channel_mode)
        self._select_input_cleanup_mode(preference.cleanup_mode)
        self._apply_input_channel_mode(preference.channel_mode)
        self._apply_input_cleanup_mode(preference.cleanup_mode)
        if hasattr(self, "config"):
            self.config.input_channel_mode = preference.channel_mode
            self.config.input_cleanup_mode = preference.cleanup_mode

    def _store_input_preference(self, *, channel_mode: str, cleanup_mode: str) -> None:
        preference = InputDevicePreference(
            channel_mode=channel_mode,
            cleanup_mode=cleanup_mode,
            provenance="explicit_user",
        )
        route_preferences = getattr(self.config, "route_input_preferences", None)
        device_preferences = getattr(self.config, "input_device_preferences", None)
        if route_preferences is not None and device_preferences is not None:
            route_key = self._current_device_route_key()
            if route_key in route_preferences:
                route_preferences[route_key] = preference
            else:
                device_key = self._current_input_preference_key()
                if device_key is not None:
                    device_preferences[device_key] = preference
        self.config.input_channel_mode = channel_mode
        self.config.input_cleanup_mode = cleanup_mode

    def _set_route_input_preference(
        self, *, channel_mode: str, cleanup_mode: str
    ) -> bool:
        """Persist an exact-route override; return false when stable IDs are absent."""
        route_key = self._current_device_route_key()
        route_preferences = getattr(self.config, "route_input_preferences", None)
        if route_key is None or route_preferences is None:
            return False
        if not (
            self._is_valid_input_channel_mode(channel_mode)
            and self._is_valid_input_cleanup_mode(cleanup_mode)
        ):
            return False
        previous = route_preferences.get(route_key)
        route_preferences[route_key] = InputDevicePreference(
            channel_mode=channel_mode,
            cleanup_mode=cleanup_mode,
            provenance="explicit_user",
        )
        saved = self._save_config_safely()
        if not saved:
            if previous is None:
                del route_preferences[route_key]
            else:
                route_preferences[route_key] = previous
            self.status_message.showMessage("Route input settings could not be saved", 6000)
            return False
        self._apply_input_preferences_for_current_route()
        return True

    def _save_current_route_input_preference(self, _checked: bool = False) -> bool:
        """Save the visible input controls as an exact-route override."""
        channel_mode = self.input_channel_mode_choice.currentData()
        cleanup_mode = self.input_cleanup_mode_choice.currentData()
        if not (
            isinstance(channel_mode, str) and isinstance(cleanup_mode, str)
        ):
            return False
        if self._current_device_route_key() is None:
            self.status_message.showMessage(
                "Connect devices with stable endpoint IDs before saving route input settings",
                5000,
            )
            return False
        if self._set_route_input_preference(
            channel_mode=channel_mode,
            cleanup_mode=cleanup_mode,
        ):
            self.status_message.showMessage(
                "Saved input settings for the current device route", 4000
            )
            return True
        return False

    def _setup_menubar(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        assert menubar is not None

        # File menu
        file_menu = menubar.addMenu("&File")
        assert file_menu is not None

        start_action = QAction("&Start Processing", self)
        start_action.setShortcut("Ctrl+Return")
        start_action.triggered.connect(self._on_start_clicked)
        file_menu.addAction(start_action)

        stop_action = QAction("S&top Processing", self)
        stop_action.setShortcut("Ctrl+.")
        stop_action.triggered.connect(self._on_stop_clicked)
        file_menu.addAction(stop_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self._quit_from_tray)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        assert edit_menu is not None

        self._undo_action = QAction("&Undo", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.setEnabled(False)
        self._undo_action.triggered.connect(self.undo_configuration)
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("&Redo", self)
        self._redo_action.setShortcut("Ctrl+Shift+Z")
        self._redo_action.setEnabled(False)
        self._redo_action.triggered.connect(self.redo_configuration)
        edit_menu.addAction(self._redo_action)

        # Presets menu
        presets_menu = menubar.addMenu("&Presets")
        assert presets_menu is not None
        self.presets_menu = presets_menu

        save_preset_action = QAction("&Save Preset", self)
        save_preset_action.setShortcut("Ctrl+S")
        save_preset_action.triggered.connect(self._save_preset)
        presets_menu.addAction(save_preset_action)
        save_as_action = QAction("Save Preset &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_preset_as)
        presets_menu.addAction(save_as_action)

        load_preset_action = QAction("&Load Complete Preset...", self)
        load_preset_action.setShortcut("Ctrl+O")
        load_preset_action.triggered.connect(self._load_preset)
        presets_menu.addAction(load_preset_action)

        presets_menu.addSeparator()

        # Built-in presets submenu
        builtin_menu = presets_menu.addMenu("&Complete Sound Presets")
        assert builtin_menu is not None
        for key, preset in BUILTIN_PRESETS.items():
            action = QAction(preset.name, self)
            action.setToolTip(preset.description)
            action.triggered.connect(
                lambda checked, p=preset, k=key: self._apply_preset(p, preset_key=k)
            )
            builtin_menu.addAction(action)

        presets_menu.addSeparator()

        # Open presets folder
        open_folder_action = QAction("Open Presets &Folder", self)
        open_folder_action.triggered.connect(self._open_presets_folder)
        presets_menu.addAction(open_folder_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        assert help_menu is not None
        self._help_menu = help_menu

        diagnostics_action = QAction("Export &Diagnostics...", self)
        diagnostics_action.setShortcut("Ctrl+Shift+D")
        diagnostics_action.setToolTip(
            "Save a privacy-safe support snapshot without audio or device names."
        )
        diagnostics_action.triggered.connect(self._export_diagnostics)
        help_menu.addAction(diagnostics_action)

        help_menu.addSeparator()

        licenses_action = QAction("&Licenses...", self)
        licenses_action.triggered.connect(self._show_licenses)
        help_menu.addAction(licenses_action)

        qt_action = QAction("About &Qt", self)
        qt_action.triggered.connect(lambda: QMessageBox.aboutQt(self))
        help_menu.addAction(qt_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_options_menu(self):
        """Setup Options menu with startup preset selector."""
        menubar = self.menuBar()
        assert menubar is not None

        # Options menu
        options_menu = menubar.addMenu("&Options")
        assert options_menu is not None
        self._options_menu = options_menu

        setup_action = QAction("Run Guided &Setup...", self)
        setup_action.setToolTip(
            "Select a route, verify native streams, measure latency, and run Auto Voice Setup."
        )
        setup_action.triggered.connect(self._show_first_run_setup)
        options_menu.addAction(setup_action)
        options_menu.addSeparator()

        # Startup Preset submenu
        startup_menu = options_menu.addMenu("Startup &Preset...")
        assert startup_menu is not None
        self._startup_preset_menu = startup_menu
        self._startup_custom_actions: list[QAction] = []

        # "Last Used" option (default, checked if startup_preset is empty)
        last_used_action = QAction("Last Used", self)
        last_used_action.setCheckable(True)
        last_used_action.setData("")
        last_used_action.triggered.connect(lambda: self._set_startup_preset(""))
        startup_menu.addAction(last_used_action)

        # Separator
        startup_menu.addSeparator()

        # Built-in presets
        for key, preset in BUILTIN_PRESETS.items():
            action = QAction(preset.name, self)
            action.setCheckable(True)
            preset_id = _startup_builtin_id(key)
            action.setData(preset_id)
            action.triggered.connect(
                lambda checked, item_id=preset_id: self._set_startup_preset(item_id)
            )
            startup_menu.addAction(action)

        # Separator
        startup_menu.addSeparator()
        startup_menu.aboutToShow.connect(self._refresh_startup_preset_menu)
        self._refresh_startup_preset_menu()

        options_menu.addSeparator()

        device_preset_menu = options_menu.addMenu("Preset for Current &Route")
        assert device_preset_menu is not None
        self._device_preset_menu = device_preset_menu
        self._device_preset_actions: dict[str, QAction] = {}
        self._device_custom_actions: list[QAction] = []
        self._device_custom_separator: QAction | None = None

        self.auto_apply_device_presets_action = QAction(
            "Automatically Apply Route Presets", self
        )
        self.auto_apply_device_presets_action.setCheckable(True)
        self.auto_apply_device_presets_action.setChecked(
            self.config.auto_apply_device_presets
        )
        self.auto_apply_device_presets_action.toggled.connect(
            self._on_auto_apply_device_presets_toggled
        )
        device_preset_menu.addAction(self.auto_apply_device_presets_action)

        clear_route_preset = QAction("No Route Preset", self)
        clear_route_preset.setCheckable(True)
        clear_route_preset.triggered.connect(self._clear_current_route_preset)
        device_preset_menu.addAction(clear_route_preset)
        self._clear_route_preset_action = clear_route_preset
        device_preset_menu.addSeparator()

        for key, preset in BUILTIN_PRESETS.items():
            preset_id = _startup_builtin_id(key)
            action = QAction(preset.name, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked, item_id=preset_id: self._bind_current_route_preset(
                    item_id
                )
            )
            device_preset_menu.addAction(action)
            self._device_preset_actions[preset_id] = action

        device_preset_menu.aboutToShow.connect(self._refresh_device_preset_menu)
        self._refresh_device_preset_menu()

        options_menu.addSeparator()

        self.save_route_input_preference_action = QAction(
            "Save Input Settings for Current Route", self
        )
        self.save_route_input_preference_action.setToolTip(
            "Save the selected channel and cleanup modes for this exact input/output route."
        )
        self.save_route_input_preference_action.triggered.connect(
            self._save_current_route_input_preference
        )
        options_menu.addAction(self.save_route_input_preference_action)

        self.use_measured_latency_action = QAction(
            "Include Measured Route Delay in Latency Estimate", self
        )
        self.use_measured_latency_action.setCheckable(True)
        self.use_measured_latency_action.setChecked(self.config.use_measured_latency)
        self.use_measured_latency_action.toggled.connect(
            self._on_use_measured_latency_toggled
        )
        options_menu.addAction(self.use_measured_latency_action)

        latency_calibration_action = QAction("Measure Route Latency (Advanced)...", self)
        latency_calibration_action.triggered.connect(
            self._on_latency_calibration_clicked
        )
        options_menu.addAction(latency_calibration_action)

        options_menu.addSeparator()

        desktop_menu = options_menu.addMenu("Tray &Background")
        assert desktop_menu is not None
        self._tray_background_menu = desktop_menu
        self._close_to_tray_action = QAction(
            "Keep running in tray when window is closed", self
        )
        self._close_to_tray_action.setCheckable(True)
        self._close_to_tray_action.setChecked(
            bool(getattr(self.config, "close_to_tray", False))
        )
        self._close_to_tray_action.setToolTip(
            "Hide the window while audio continues; use the tray menu to show or quit."
        )
        self._close_to_tray_action.toggled.connect(self._on_close_to_tray_toggled)
        desktop_menu.addAction(self._close_to_tray_action)

        confirm_quit_action = QAction("Ask before quitting while audio is running", self)
        confirm_quit_action.setCheckable(True)
        confirm_quit_action.setChecked(self.config.confirm_quit_when_live)
        confirm_quit_action.toggled.connect(self._on_confirm_quit_toggled)
        desktop_menu.addAction(confirm_quit_action)

        self._mute_hotkey_action = QAction(
            f"Enable global mute shortcut ({DEFAULT_MUTE_HOTKEY})", self
        )
        self._mute_hotkey_action.setCheckable(True)
        self._mute_hotkey_action.setChecked(bool(getattr(self.config, "mute_hotkey", "")))
        self._mute_hotkey_action.setToolTip(
            "Toggle Output Mute from any application while AudioForge is running."
        )
        self._mute_hotkey_action.toggled.connect(self._on_mute_hotkey_toggled)
        desktop_menu.addAction(self._mute_hotkey_action)

        desktop_menu.addSeparator()
        self._login_startup_action = QAction("Configure login shortcut for this copy", self)
        self._login_startup_action.setCheckable(True)
        self._login_startup_action.toggled.connect(self._on_login_startup_toggled)
        desktop_menu.addAction(self._login_startup_action)
        self._login_startup_status_action = QAction("", self)
        desktop_menu.addAction(self._login_startup_status_action)
        self._login_startup_status_action.setEnabled(False)
        desktop_menu.aboutToShow.connect(self._refresh_login_startup_action)
        self._refresh_login_startup_action()

    def _refresh_login_startup_action(self) -> None:
        packaged = sys.platform == "win32" and bool(getattr(sys, "frozen", False))
        state = login_startup.registration_state(Path(sys.executable)) if packaged else "absent"
        self._login_startup_action.blockSignals(True)
        self._login_startup_action.setChecked(state == "configured")
        self._login_startup_action.setEnabled(packaged and state in {"absent", "configured"})
        self._login_startup_action.blockSignals(False)
        # The Settings row follows changed(), which the block suppressed.
        self._login_startup_action.changed.emit()
        messages = {
            "absent": "No login shortcut configured",
            "configured": "Shortcut configured; Windows may disable it",
            "other": "Login shortcut belongs to another copy or is unrecognized",
            "unreadable": "Login shortcut could not be read; left unchanged",
        }
        self._login_startup_status_action.setText(
            messages[state] if packaged else "Login startup requires the packaged executable"
        )
        self._login_startup_action.setToolTip(
            "Windows Settings > Apps > Startup controls whether this shortcut runs. "
            "Moving a portable copy requires removing its shortcut before moving it."
        )

    def _on_login_startup_toggled(self, checked: bool) -> None:
        if not checked:
            self._cancel_login_startup()
        try:
            login_startup.set_login_startup(Path(sys.executable), checked)
        except Exception as error:
            logger.exception("Could not change the login shortcut")
            self.status_message.showMessage(str(error), 8000)
        self._refresh_login_startup_action()

    def begin_login_startup(self) -> None:
        """Wait up to a minute for the tray and the exact saved route, without focus."""
        self._hidden_to_tray = True
        self._login_startup_deadline = time.monotonic() + 60.0
        self._login_startup_next_retry = 0.0
        self._service_login_startup()

    def _cancel_login_startup(self, message: str = "") -> None:
        pending = self.__dict__.get("_login_startup_deadline") is not None
        self._login_startup_deadline = None
        self._login_startup = False
        if pending or message:
            self._login_startup_message = message
            if message:
                logger.warning(message)
                self.status_message.showMessage(message)
            self._update_session_summary()

    def _service_login_startup(self) -> None:
        deadline = self.__dict__.get("_login_startup_deadline")
        if deadline is None or self._quitting:
            return
        now = time.monotonic()
        if now < self._login_startup_next_retry:
            return
        self._login_startup_next_retry = now + 1.0
        if self._tray_icon is None:
            self._setup_desktop_integration(tray_only=True)
        if self._tray_icon is None or not self._tray_icon.isVisible():
            if now >= deadline:
                self._cancel_login_startup("Login stopped: Windows tray unavailable; exiting")
                app = QGuiApplication.instance()
                if app is not None:
                    app.exit(1)
            return
        if self.quick_view_failure is not None:
            # Nobody can see or stop audio from a window that did not open.
            self._cancel_login_startup("Login stopped: the main window could not open")
            self._tray_icon.showMessage(
                "AudioForge could not open its main window",
                "Audio was not started. Choose Show AudioForge in the tray menu to see why.",
                QSystemTrayIcon.MessageIcon.Critical,
                10000,
            )
            return
        if login_startup.registration_state(Path(sys.executable)) != "configured":
            self._cancel_login_startup("Login stopped: this copy's shortcut is no longer configured")
            return
        if (
            not self._startup_restore_ready or self.config.load_warning
            or self.config.save_blocked_reason
        ):
            self._cancel_login_startup("Login stopped: saved settings could not be restored")
            return
        saved_input, saved_output = self._login_startup_route
        if any(identity is None or not identity.endpoint_id for identity in self._login_startup_route):
            self._cancel_login_startup("Login stopped: select and save both exact audio endpoints")
            return
        if now >= deadline:
            self._cancel_login_startup("Login stopped: saved endpoints did not become available")
            return
        self._refresh_devices()
        indices = (
            find_identity_index(self._combo_identities(self.input_choice), saved_input),
            find_identity_index(self._combo_identities(self.output_choice), saved_output),
        )
        if min(indices) < 0:
            self._login_startup_message = "Login waiting for saved audio endpoints"
            self.status_message.showMessage(self._login_startup_message)
            self.stop_btn_action.setEnabled(True)
            return
        for combo, index in zip((self.input_choice, self.output_choice), indices):
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(blocked)
        self._apply_input_preferences_for_current_route()
        self._apply_latency_compensation_for_current_devices()
        route_key = self._current_device_route_key()
        if (
            self.config.auto_apply_device_presets
            and route_key in self.config.device_preset_bindings
            and not self._apply_bound_preset_for_current_route()
        ):
            self._cancel_login_startup("Login stopped: saved route preset could not be restored")
            return
        self._cancel_login_startup()
        if not self._start_processing(interactive=False):
            message = (
                "Login needs attention: audio may still be running; use Stop"
                if self.processor.is_running()
                else "Login stopped: audio could not start safely; open AudioForge"
            )
            self._cancel_login_startup(message)

    def _setup_desktop_integration(self, *, tray_only: bool = False) -> None:
        """Create the optional tray icon and register the configured hotkey."""
        app = QGuiApplication.instance()
        if app is not None and not tray_only:
            app.aboutToQuit.connect(self._on_about_to_quit)
            if isinstance(app, QGuiApplication):
                # Direct, so the flags are set before Qt closes the windows for a Windows
                # sign-out or shutdown; that close asks nothing, hides nothing and never refuses.
                app.commitDataRequest.connect(
                    self._on_commit_data, Qt.ConnectionType.DirectConnection
                )

        configured_hotkey = str(getattr(self.config, "mute_hotkey", "") or "")
        if configured_hotkey and not tray_only:
            self._register_mute_hotkey(configured_hotkey)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            if self._close_to_tray_action is not None:
                self._close_to_tray_action.setEnabled(False)
                self._close_to_tray_action.blockSignals(True)
                self._close_to_tray_action.setChecked(False)
                self._close_to_tray_action.blockSignals(False)
                # The Settings row follows changed(), which the block suppressed.
                self._close_to_tray_action.changed.emit()
            return

        if self._close_to_tray_action is not None:
            self._close_to_tray_action.setEnabled(True)
            self._close_to_tray_action.blockSignals(True)
            self._close_to_tray_action.setChecked(bool(self.config.close_to_tray))
            self._close_to_tray_action.blockSignals(False)
            # The Settings row follows changed(), which the block suppressed.
            self._close_to_tray_action.changed.emit()

        icon = self.windowIcon()
        if icon.isNull() and isinstance(app, QGuiApplication):
            icon = app.windowIcon()
        if icon.isNull():
            icon = QIcon.fromTheme("audio-card")
        self._tray_icons = (icon, muted_icon(icon))
        self._tray_icon = QSystemTrayIcon(icon, self)
        self._tray_icon.setToolTip("AudioForge microphone processor")
        self._tray_menu = QMenu(self)

        show_action = QAction("Show AudioForge", self)
        show_action.triggered.connect(self._show_from_tray)
        self._tray_menu.addAction(show_action)

        self._tray_mute_action = QAction("Mute Output", self)
        self._tray_mute_action.setCheckable(True)
        self._tray_mute_action.setChecked(self.user_muted)
        self._tray_mute_action.toggled.connect(self._on_tray_mute_toggled)
        self._tray_menu.addAction(self._tray_mute_action)
        self._tray_menu.addSeparator()

        if self._close_to_tray_action is not None:
            self._tray_menu.addAction(self._close_to_tray_action)
        self._tray_menu.addSeparator()
        quit_action = QAction("Quit AudioForge", self)
        quit_action.triggered.connect(self._quit_from_tray)
        self._tray_menu.addAction(quit_action)
        self._tray_icon.setContextMenu(self._tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()
        self._update_session_summary()

    def _on_commit_data(self) -> None:
        app = QGuiApplication.instance()
        if app is not None:
            app.setProperty(SESSION_ENDING_PROPERTY, True)
        # Qt does not say when another program refuses the sign-out; the query is over when this
        # call returns to the event loop, and a closeEvent in it has run by then.
        QTimer.singleShot(0, _end_session_query)

    def _on_about_to_quit(self) -> None:
        """Shut down a window Qt never closed: hidden in the tray, or the app quit without it."""
        self._quitting = True
        if not self._shut_down:
            self._shutdown()

    def _show_from_tray(self) -> None:
        activate_window(self)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_from_tray()

    def _on_tray_mute_toggled(self, checked: bool) -> None:
        if self.user_mute_checkbox_action.isChecked() != bool(checked):
            self.user_mute_checkbox_action.setChecked(bool(checked))

    def _on_close_to_tray_toggled(self, checked: bool) -> None:
        setattr(self.config, "close_to_tray", bool(checked))
        saved = self._save_config_safely()
        message = "Close-to-tray enabled" if checked else "Close-to-tray disabled"
        if not saved:
            message += "; preference could not be saved"
        self.status_message.showMessage(message, 5000 if not saved else 3000)

    def _on_confirm_quit_toggled(self, checked: bool) -> None:
        self.config.confirm_quit_when_live = bool(checked)
        if not self._save_config_safely():
            self.status_message.showMessage("Preference could not be saved", 5000)

    def _on_mute_hotkey_toggled(self, checked: bool) -> None:
        configured_hotkey = DEFAULT_MUTE_HOTKEY if checked else ""
        setattr(self.config, "mute_hotkey", configured_hotkey)
        if checked:
            registered = self._register_mute_hotkey(configured_hotkey)
        else:
            self._unregister_mute_hotkey()
            registered = True
        saved = self._save_config_safely()
        if checked and not registered:
            return
        message = (
            f"Global mute shortcut {'enabled' if checked else 'disabled'}"
        )
        if not saved:
            message += "; preference could not be saved"
        self.status_message.showMessage(message, 5000 if not saved else 3000)

    def _register_mute_hotkey(self, shortcut: str) -> bool:
        self._unregister_mute_hotkey()
        hotkey = GlobalMuteHotkey(shortcut, self._toggle_mute_from_hotkey)
        registered, error = hotkey.register()
        if not registered:
            self.config.mute_hotkey = ""
            if self._mute_hotkey_action is not None:
                self._mute_hotkey_action.blockSignals(True)
                self._mute_hotkey_action.setChecked(False)
                self._mute_hotkey_action.blockSignals(False)
                # The Settings row follows changed(), which the block suppressed.
                self._mute_hotkey_action.changed.emit()
            self.status_message.showMessage(
                f"Global mute shortcut unavailable: {error}",
                8000,
            )
            return False
        self._mute_hotkey = hotkey
        return True

    def _unregister_mute_hotkey(self) -> None:
        if self._mute_hotkey is None:
            return
        self._mute_hotkey.unregister()
        self._mute_hotkey = None

    def _toggle_mute_from_hotkey(self) -> None:
        self.user_mute_checkbox_action.setChecked(not self.user_mute_checkbox_action.isChecked())

    def _quit_from_tray(self) -> None:
        # An open setup dialog holds provisional settings and puts the old ones back when it
        # is rejected; do that before the unsaved-changes prompt can offer to save them.
        # A message box on top of the setup dialog is rejected first, then the dialog under it.
        for _ in range(8):
            modal = QApplication.activeModalWidget()
            if not isinstance(modal, QDialog) or modal is self:
                break
            modal.reject()
        choice = self._quit_choice()
        if choice == "tray":
            self._hide_to_tray()
            return
        if choice == "cancel":
            return
        self._quitting = True
        if self.close():
            app = QGuiApplication.instance()
            if app is not None:
                app.quit()

    def _quit_choice(self) -> str:
        """Ask before a quit stops live audio: "quit", "tray" (hide, keep running) or "cancel"."""
        close_to_tray = (
            self._close_to_tray_action is not None and self._close_to_tray_action.isChecked()
        )
        if (
            close_to_tray
            or not self.processor.is_running()
            or not self.config.confirm_quit_when_live
        ):
            return "quit"
        box = QMessageBox(
            QMessageBox.Icon.Question,
            "Stop Audio and Quit?",
            "AudioForge is processing audio. Quitting stops it.",
            parent=self,
        )
        quit_button = box.addButton("Quit", QMessageBox.ButtonRole.AcceptRole)
        tray_button = None
        if self._tray_icon is not None and self._tray_icon.isVisible():
            tray_button = box.addButton("Keep Running in Tray", QMessageBox.ButtonRole.AcceptRole)
        cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_button)
        box.setEscapeButton(cancel_button)
        dont_ask = QCheckBox("Don't ask again")
        box.setCheckBox(dont_ask)
        box.exec()
        clicked = box.clickedButton()
        if clicked is quit_button:
            # closeEvent applies it once the quit goes ahead, and saves the config on the way out.
            self._quit_dont_ask = dont_ask.isChecked()
            return "quit"
        return "tray" if clicked is tray_button else "cancel"

    def _hide_to_tray(self) -> None:
        assert self._tray_icon is not None
        if not self.__dict__.get("_hidden_to_tray"):
            self._tray_icon.showMessage(
                "AudioForge is running in the background",
                "Audio continues while the window is closed. "
                "Use the tray icon to show the window or Quit AudioForge.",
                QSystemTrayIcon.MessageIcon.Information,
                8000,
            )
        self._hidden_to_tray = True
        self.hide()
        self.status_message.showMessage(
            "AudioForge is still running in the tray; use Quit AudioForge to exit",
            5000,
        )

    def _set_startup_preset(self, preset_id: str):
        """Set the startup preset and update checked states.

        Args:
            preset_id: Stable preset ID to load on startup (empty string = Last Used)
        """
        previous = self.config.startup_preset
        self.config.startup_preset = preset_id

        saved = self._save_config_safely()
        if not saved:
            self.config.startup_preset = previous
            self.status_message.showMessage(
                "Startup preset could not be saved; previous selection kept", 6000
            )
            self._update_startup_preset_menu(previous)
            return
        self._set_problem("startup_preset", None)

        if preset_id:
            _normalized_id, custom_preset = _startup_preset_selection(
                preset_id, list_presets()
            )
            preset_name = (
                custom_preset[0]
                if custom_preset is not None
                else _startup_preset_display_name(preset_id)
            )
            message = f"Startup preset set to {preset_name}"
        else:
            message = "Startup preset set to Last Used"
        self.status_message.showMessage(message, 5000)

        self._update_startup_preset_menu(preset_id)

    def _update_startup_preset_menu(self, preset_id: str | None = None) -> None:
        selected_id = (
            self.config.startup_preset if preset_id is None else preset_id
        )
        for action in self._startup_preset_menu.actions():
            if action.isCheckable():
                action.setChecked(str(action.data() or "") == selected_id)

    def _refresh_startup_preset_menu(self) -> None:
        custom_presets = list_presets()
        for action in self._startup_custom_actions:
            self._startup_preset_menu.removeAction(action)
            action.deleteLater()
        self._startup_custom_actions.clear()

        startup_preset_id, _custom_preset = _startup_preset_selection(
            self.config.startup_preset, custom_presets
        )
        for name, filepath in custom_presets:
            action = QAction(name, self)
            action.setCheckable(True)
            preset_id = _startup_custom_file_id(filepath.name)
            action.setData(preset_id)
            action.triggered.connect(
                lambda _checked, item_id=preset_id: self._set_startup_preset(item_id)
            )
            self._startup_preset_menu.addAction(action)
            self._startup_custom_actions.append(action)

        self._update_startup_preset_menu(startup_preset_id)

    def _maybe_show_first_run_setup(self) -> None:
        if self.__dict__.get("_login_startup", False) or os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get(
            "AUDIOFORGE_SMOKE_TEST"
        ):
            return
        if self.__dict__.get("quick_view_failure") is not None:
            return  # the error screen is what to read now
        if self.config.first_run_setup_state in {"not_started", "in_progress"}:
            self._show_first_run_setup(restart_completed=False)

    def _show_first_run_setup(
        self, _checked: bool = False, *, restart_completed: bool = True
    ) -> None:
        _exec_and_release(FirstRunSetupDialog(self, restart_completed=restart_completed))

    def _current_device_route_key(self) -> str | None:
        input_identity = self._combo_device_identity(self.input_choice)
        output_identity = self._combo_device_identity(self.output_choice)
        if input_identity is None or output_identity is None:
            return None
        if not identity_is_persistable(
            self._combo_identities(self.input_choice), input_identity
        ) or not identity_is_persistable(
            self._combo_identities(self.output_choice), output_identity
        ):
            return None
        return build_device_route_key(input_identity, output_identity)

    def _current_capture_format_context(self) -> tuple[int | None, ...]:
        """Return mutable endpoint format fields used by calibration evidence."""
        input_identity = self._combo_device_identity(self.input_choice)
        output_identity = self._combo_device_identity(self.output_choice)
        context = [
            getattr(input_identity, "sample_rate", None),
            getattr(input_identity, "channels", None),
            getattr(output_identity, "sample_rate", None),
            getattr(output_identity, "channels", None),
        ]
        processor = self.__dict__.get("processor")
        try:
            if processor is not None and processor.is_running():
                diagnostics = dict(processor.get_runtime_diagnostics())
                for index, value in (
                    (0, diagnostics.get("input_sample_rate")),
                    (2, diagnostics.get("output_sample_rate")),
                ):
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        rate = int(value)
                        if rate > 0:
                            context[index] = rate
        except Exception:
            logger.debug("Could not read active audio format", exc_info=True)
        return tuple(context)

    def _latency_profile_matches_capture_format(
        self, profile: LatencyCalibrationProfile
    ) -> bool:
        context = profile.capture_format_context
        current = self._current_capture_format_context()
        return (
            context is not None
            and len(context) == len(current)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in context
            )
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in current
            )
            and context == current
        )

    def _on_auto_apply_device_presets_toggled(self, checked: bool) -> None:
        self.config.auto_apply_device_presets = bool(checked)
        saved = self._save_config_safely()
        if checked:
            self._apply_bound_preset_for_current_route()
        if not saved:
            state = "enabled" if checked else "disabled"
            self.status_message.showMessage(
                f"Automatic route presets {state} for this session, "
                "but the preference could not be saved",
                6000,
            )

    def _bind_current_route_preset(self, preset_id: str) -> None:
        route_key = self._current_device_route_key()
        if route_key is None:
            self.status_message.showMessage(
                "Connect and select both route devices before binding a preset", 5000
            )
            return
        previous = self.config.device_preset_bindings.get(route_key)
        self.config.device_preset_bindings[route_key] = DevicePresetBinding(
            preset_id=preset_id,
            provenance="explicit_user",
        )
        if not self._save_config_safely():
            if previous is None:
                self.config.device_preset_bindings.pop(route_key, None)
            else:
                self.config.device_preset_bindings[route_key] = previous
            self._update_device_preset_menu()
            self.status_message.showMessage(
                "Preset binding could not be saved; previous binding kept",
                6000,
            )
            return
        self._update_device_preset_menu()
        _normalized_id, custom_preset = _startup_preset_selection(
            preset_id, list_presets()
        )
        preset_name = (
            custom_preset[0]
            if custom_preset is not None
            else _startup_preset_display_name(preset_id)
        )
        self.status_message.showMessage(
            f"Bound {preset_name} to this device route",
            5000,
        )

    def _clear_current_route_preset(self) -> None:
        route_key = self._current_device_route_key()
        if route_key is None:
            return
        removed = self.config.device_preset_bindings.pop(route_key, None)
        if removed is not None:
            if not self._save_config_safely():
                self.config.device_preset_bindings[route_key] = removed
                self._update_device_preset_menu()
                self.status_message.showMessage(
                    "Preset binding could not be cleared because the change could not be saved",
                    6000,
                )
                return
        self._update_device_preset_menu()
        self.status_message.showMessage("Cleared the preset binding for this route", 4000)

    def _refresh_device_preset_menu(self) -> None:
        custom_presets = list_presets()
        for action in self._device_custom_actions:
            self._device_preset_menu.removeAction(action)
            action.deleteLater()
            self._device_preset_actions.pop(str(action.data() or ""), None)
        self._device_custom_actions.clear()
        if self._device_custom_separator is not None:
            self._device_preset_menu.removeAction(self._device_custom_separator)
            self._device_custom_separator.deleteLater()
            self._device_custom_separator = None

        if custom_presets:
            self._device_custom_separator = self._device_preset_menu.addSeparator()
        for name, filepath in custom_presets:
            preset_id = _startup_custom_file_id(filepath.name)
            action = QAction(name, self)
            action.setCheckable(True)
            action.setData(preset_id)
            action.triggered.connect(
                lambda _checked, item_id=preset_id: self._bind_current_route_preset(
                    item_id
                )
            )
            self._device_preset_menu.addAction(action)
            self._device_custom_actions.append(action)
            self._device_preset_actions[preset_id] = action

        self._update_device_preset_menu()

    def _update_device_preset_menu(self) -> None:
        route_key = self._current_device_route_key()
        binding = (
            self.config.device_preset_bindings.get(route_key)
            if route_key is not None
            else None
        )
        selected_id = binding.preset_id if binding is not None else ""
        if selected_id.startswith(
            (STARTUP_CUSTOM_PREFIX, STARTUP_CUSTOM_FILE_PREFIX)
        ):
            selection = _route_custom_preset_selection(selected_id, list_presets())
            selected_id = selection[0] if selection is not None else ""
        self._clear_route_preset_action.setChecked(binding is None)
        self._clear_route_preset_action.setEnabled(route_key is not None)
        for preset_id, action in self._device_preset_actions.items():
            action.setChecked(preset_id == selected_id)
            action.setEnabled(route_key is not None)

    def _load_device_preset_id(self, preset_id: str) -> bool:
        if preset_id.startswith(STARTUP_BUILTIN_PREFIX):
            key = preset_id[len(STARTUP_BUILTIN_PREFIX) :]
            preset = BUILTIN_PRESETS.get(key)
            if preset is None:
                return False
            return self._apply_preset(preset, preset_key=key, persist_last_used=False)
        if not preset_id.startswith(
            (STARTUP_CUSTOM_PREFIX, STARTUP_CUSTOM_FILE_PREFIX)
        ):
            return False
        selection = _route_custom_preset_selection(preset_id, list_presets())
        if selection is None:
            return False
        _normalized_id, (_name, filepath) = selection
        preset = load_preset(filepath)
        return self._apply_preset(
            preset, preset_path=filepath, persist_last_used=False
        )

    def _apply_bound_preset_for_current_route(self) -> bool:
        if not self.config.auto_apply_device_presets:
            return False
        route_key = self._current_device_route_key()
        if route_key is None:
            return False
        binding = self.config.device_preset_bindings.get(route_key)
        if binding is None:
            return False
        try:
            loaded = self._load_device_preset_id(binding.preset_id)
        except (OSError, PresetValidationError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to apply route preset", exc_info=True)
            loaded = False
        if loaded:
            selection = _route_custom_preset_selection(
                binding.preset_id, list_presets()
            )
            preset_name = (
                selection[1][0]
                if selection is not None
                else _startup_preset_display_name(binding.preset_id)
            )
            if selection is not None and selection[0] != binding.preset_id:
                self.config.device_preset_bindings[route_key] = DevicePresetBinding(
                    preset_id=selection[0], provenance=binding.provenance
                )
                if not self._save_config_safely():
                    self.config.device_preset_bindings[route_key] = binding
            self.status_message.showMessage(
                f"Route preset: {preset_name}",
                5000,
            )
        else:
            self.status_message.showMessage(
                "The preset bound to this route is unavailable; existing settings were kept",
                6000,
            )
        return loaded

    def _latency_profile_key(self) -> str | None:
        return self._current_device_route_key()

    def _legacy_latency_profile_key(self) -> str:
        input_name = self._device_name_from_identity(
            self._combo_device_identity(self.input_choice)
        )
        output_name = self._device_name_from_identity(
            self._combo_device_identity(self.output_choice)
        )
        return legacy_latency_profile_key(
            input_name or "default-input",
            output_name or "default-output",
        )

    def _current_latency_profile(self) -> LatencyCalibrationProfile | None:
        key = self._latency_profile_key()
        if key is None:
            return None
        profiles = self.config.latency_calibration_profiles
        profile = profiles.get(key)
        if profile is not None:
            return (
                profile
                if self._latency_profile_matches_capture_format(profile)
                else None
            )

        legacy_key = self._legacy_latency_profile_key()
        profile = profiles.get(legacy_key)
        if profile is not None:
            profiles[key] = profile
            if legacy_key != key and legacy_key in profiles:
                del profiles[legacy_key]
            saved = self._save_config_safely()
            if not saved:
                self.status_message.showMessage("Updated latency profile could not be saved", 6000)
        return (
            profile
            if profile is not None
            and self._latency_profile_matches_capture_format(profile)
            else None
        )

    def _sync_latency_profile_for_current_devices(
        self, profile: LatencyCalibrationProfile
    ) -> str:
        key = self._latency_profile_key()
        if key is None:
            raise ValueError(
                "Stable endpoint identity is unavailable for this duplicate-name route"
            )
        if not self._latency_profile_matches_capture_format(profile):
            raise ValueError(
                "Latency calibration result is stale for the current route or format"
            )
        legacy_key = self._legacy_latency_profile_key()
        self.config.latency_calibration_profiles[key] = profile
        if legacy_key != key and legacy_key in self.config.latency_calibration_profiles:
            del self.config.latency_calibration_profiles[legacy_key]
        return key

    def _refresh_latency_profile_engine(
        self, profile: LatencyCalibrationProfile
    ) -> bool:
        try:
            if not self.processor.is_running():
                return False
            engine_latency_ms = max(0.0, float(self.processor.get_engine_latency_ms()))
            signature = engine_config_signature(self.processor)
        except Exception:
            logger.exception("Failed to refresh engine latency profile")
            return False

        route_latency_ms = max(
            0.0,
            float(profile.route_latency_ms or profile.applied_compensation_ms),
        )
        total_latency_ms = route_latency_ms + engine_latency_ms
        changed = (
            abs(profile.engine_latency_ms - engine_latency_ms) > 0.01
            or abs(profile.total_latency_ms - total_latency_ms) > 0.01
            or profile.engine_config_signature != signature
        )
        if changed:
            profile.engine_latency_ms = engine_latency_ms
            profile.total_latency_ms = total_latency_ms
            profile.engine_config_signature = signature
        return changed

    def _apply_latency_compensation_for_current_devices(self):
        compensation_ms = 0.0
        profile = self._current_latency_profile()

        if self.config.use_measured_latency and profile is not None:
            self._refresh_latency_profile_engine(profile)
            route_latency_ms = float(profile.route_latency_ms)
            if route_latency_ms <= 0.0:
                # Compatibility for profiles constructed in-memory by older
                # callers; persisted profiles are migrated by from_dict().
                route_latency_ms = float(profile.applied_compensation_ms)
            compensation_ms = max(0.0, route_latency_ms)

        try:
            self.processor.set_latency_compensation_ms(compensation_ms)
        except Exception:
            logger.exception("Failed to apply latency compensation")

    def _on_use_measured_latency_toggled(self, enabled: bool):
        self.config.use_measured_latency = bool(enabled)
        self._apply_latency_compensation_for_current_devices()
        saved = self._save_config_safely()
        mode = "enabled" if enabled else "disabled"
        message = f"Measured route delay in latency estimate {mode}"
        if not saved:
            message += "; preference could not be saved"
        self.status_message.showMessage(message, 4000 if saved else 6000)

    def _on_latency_calibration_clicked(self) -> bool:
        if self._latency_profile_key() is None:
            self.status_message.showMessage(
                "Cannot persist calibration: duplicate device names lack stable endpoint IDs",
                6000,
            )
            return False
        profile = self._current_latency_profile()
        if profile is not None:
            self._refresh_latency_profile_engine(profile)
        existing_profile = profile.to_dict() if profile is not None else None

        dialog = LatencyCalibrationDialog(self, existing_profile=existing_profile)
        self._calibration_dialog_open = True
        try:
            with recovery_paused(self.processor):
                return _exec_and_release(dialog) == QDialog.DialogCode.Accepted
        finally:
            self._calibration_dialog_open = False

    def _on_latency_calibration_saved(self, profile_data: dict) -> bool:
        previous = dict(self.config.latency_calibration_profiles)
        try:
            profile = LatencyCalibrationProfile.from_dict(profile_data)
            self._sync_latency_profile_for_current_devices(profile)
        except ValueError as error:
            self.status_message.showMessage(str(error), 6000)
            return False
        saved = self._save_config_safely()
        if not saved:
            self.config.latency_calibration_profiles = previous
            self.status_message.showMessage("Latency calibration could not be saved; retry saving", 6000)
            return False
        self._apply_latency_compensation_for_current_devices()
        route_latency_ms = float(profile.route_latency_ms)
        if route_latency_ms <= 0.0:
            route_latency_ms = float(profile.measured_round_trip_ms)
        self.status_message.showMessage(
            f"Measured route latency saved for current device pair ({route_latency_ms:.1f} ms)",
            5000,
        )
        return True

    def _on_latency_calibration_reset(self) -> bool:
        key = self._latency_profile_key()
        if key is None:
            self._apply_latency_compensation_for_current_devices()
            return False
        legacy_key = self._legacy_latency_profile_key()
        previous = dict(self.config.latency_calibration_profiles)
        removed = False
        for candidate in {key, legacy_key}:
            if candidate in self.config.latency_calibration_profiles:
                del self.config.latency_calibration_profiles[candidate]
                removed = True
        if removed:
            saved = self._save_config_safely()
            if not saved:
                self.config.latency_calibration_profiles = previous
                self.status_message.showMessage("Latency calibration reset could not be saved", 6000)
                return False
        self._apply_latency_compensation_for_current_devices()
        self.status_message.showMessage(
            "Latency calibration reset for current device pair", 4000
        )
        return True

    def _on_refresh_clicked(self) -> None:
        """List the devices on the worker: the enumeration can take seconds."""
        if self.__dict__.get("_engine_busy"):
            return
        if self.processor.is_running():
            self._refresh_devices()  # says why it cannot refresh
            return
        self._run_engine("refresh", _list_devices, self._refresh_devices)

    def _refresh_devices(self, listing: tuple[Any, Any] | None = None):
        """Refresh the device lists; ``listing`` is the worker's, else they are listed here."""
        if self.processor.is_running():
            self.refresh_btn_action.setEnabled(False)
            self.status_message.showMessage("Stop processing before refreshing audio devices", 5000)
            return
        input_devices, output_devices = _list_devices() if listing is None else listing
        previous_route = self._current_device_route_key()
        previous_capture_format = self._current_capture_format_context()
        previous_input = (
            self.config.last_input_device_identity
            or self._combo_device_identity(self.input_choice)
        )
        previous_output = (
            self.config.last_output_device_identity
            or self._combo_device_identity(self.output_choice)
        )

        # Block signals to prevent spurious config saves during refresh
        signal_widgets = [self.input_choice, self.output_choice]
        if "input_channel_mode_choice" in self.__dict__:
            signal_widgets.append(self.input_channel_mode_choice)
        if "input_cleanup_mode_choice" in self.__dict__:
            signal_widgets.append(self.input_cleanup_mode_choice)
        signal_states = [widget.blockSignals(True) for widget in signal_widgets]

        try:
            self.input_choice.clear()
            self.output_choice.clear()

            input_found = False
            output_found = False
            config_dirty = False
            listing_errors: list[str] = []

            # Get input devices
            try:
                if isinstance(input_devices, Exception):
                    raise input_devices
                input_found = len(input_devices) > 0
                for device in input_devices:
                    identity = self._identity_from_device_info(device, "input")
                    self.input_choice.addItem(
                        _device_label(identity, input_devices), identity
                    )
                if previous_input is not None:
                    if not self._select_combo_identity(self.input_choice, previous_input):
                        fallback_index = (
                            -1 if self.__dict__.get("_login_startup", False)
                            else self._default_combo_index(self.input_choice)
                        )
                        self.input_choice.setCurrentIndex(fallback_index)
                        if fallback_index >= 0:
                            self._report_problem(
                                "input_device",
                                f"Previous input device '{previous_input.name}' is disconnected; "
                                "using the default until it returns",
                            )
                        elif previous_input.name:
                            self._report_problem(
                                "input_device",
                                f"Saved input device '{previous_input.name}' is disconnected",
                            )
                    else:
                        self._set_problem("input_device", None)
                        resolved = self._combo_device_identity(self.input_choice)
                        if (
                            resolved is not None
                            and resolved.to_dict() != previous_input.to_dict()
                        ):
                            self.config.last_input_device = resolved.name
                            self.config.last_input_device_identity = resolved
                            config_dirty = True
                elif self.input_choice.count() > 0:
                    fallback_index = self._default_combo_index(self.input_choice)
                    if fallback_index >= 0:
                        self.input_choice.setCurrentIndex(fallback_index)
            except (RuntimeError, OSError) as e:
                listing_errors.append(f"input devices ({e})")
                logger.warning("Input device enumeration failed", exc_info=True)

            # Get output devices
            try:
                if isinstance(output_devices, Exception):
                    raise output_devices
                output_found = len(output_devices) > 0
                for device in output_devices:
                    identity = self._identity_from_device_info(device, "output")
                    self.output_choice.addItem(
                        _device_label(identity, output_devices), identity
                    )
                if previous_output is not None:
                    if not self._select_combo_identity(self.output_choice, previous_output):
                        self.output_choice.setCurrentIndex(-1)
                        self.output_choice.setPlaceholderText("Select a replacement destination")
                        self._report_problem(
                            "output_device",
                            f"Destination '{previous_output.name}' is disconnected; "
                            "select a replacement or reconnect it before starting",
                        )
                    else:
                        self._set_problem("output_device", None)
                        resolved = self._combo_device_identity(self.output_choice)
                        if (
                            resolved is not None
                            and resolved.to_dict() != previous_output.to_dict()
                        ):
                            self.config.last_output_device = resolved.name
                            self.config.last_output_device_identity = resolved
                            config_dirty = True
                elif self.output_choice.count() > 0:
                    preferred_index = self._preferred_output_combo_index(self.output_choice)
                    if preferred_index >= 0:
                        self.output_choice.setCurrentIndex(preferred_index)

            except (RuntimeError, OSError) as e:
                listing_errors.append(f"output devices ({e})")
                logger.warning("Output device enumeration failed", exc_info=True)

            # Update warning banner visibility and text
            if listing_errors:
                self.shell_texts["device_warning_banner"].update(
                    text="Warning: Could not list " + " or ".join(listing_errors) + ".", shown=True
                )
            elif not input_found and not output_found:
                self.shell_texts["device_warning_banner"].update(text=
                    "Warning: No audio devices detected. Check your audio drivers and connections."
                )
                self.shell_texts["device_warning_banner"].update(shown=True)
            elif not input_found:
                self.shell_texts["device_warning_banner"].update(text=
                    "Warning: No input devices detected. Check your microphone connections."
                )
                self.shell_texts["device_warning_banner"].update(shown=True)
            elif not output_found:
                self.shell_texts["device_warning_banner"].update(text=
                    "Warning: No output devices detected. Check your audio output connections."
                )
                self.shell_texts["device_warning_banner"].update(shown=True)
            else:
                self.shell_texts["device_warning_banner"].update(shown=False)

        finally:
            for widget, blocked in zip(signal_widgets, signal_states):
                widget.blockSignals(blocked)
            current_route = self._current_device_route_key()
            capture_format_changed = (
                previous_capture_format != self._current_capture_format_context()
            )
            route_changed = previous_route != current_route
            if route_changed or capture_format_changed:
                self._apply_input_preferences_for_current_route()
                self._apply_latency_compensation_for_current_devices()
            if route_changed and not self.__dict__.get("_login_startup", False):
                self._apply_bound_preset_for_current_route()
            if route_changed or capture_format_changed:
                self._sync_calibration_evidence(force_reset=True)

        if config_dirty:
            if not self._save_config_safely():
                self.status_message.showMessage(
                    "Device selections applied for this session, but could not be saved",
                    6000,
                )
        self._update_session_summary()

    def _restore_from_config(self):
        """Restore settings from loaded config."""
        restored_count = 0
        config_dirty = False
        requested_preset = bool(self.config.startup_preset or self.config.last_preset)

        self.input_choice.blockSignals(True)
        self.output_choice.blockSignals(True)

        # Restore input device
        input_identity = self.config.last_input_device_identity
        if input_identity is None and self.config.last_input_device:
            input_identity = coerce_device_identity(self.config.last_input_device)
        if input_identity is not None:
            index = self._find_combo_index_by_identity(self.input_choice, input_identity)
            if index >= 0:
                self.input_choice.setCurrentIndex(index)
                resolved = self._combo_device_identity(self.input_choice)
                if (
                    resolved is not None
                    and resolved.to_dict() != input_identity.to_dict()
                ):
                    self.config.last_input_device = resolved.name
                    self.config.last_input_device_identity = resolved
                    config_dirty = True
                restored_count += 1
            else:
                if self.__dict__.get("_login_startup", False):
                    self.input_choice.setCurrentIndex(-1)
                self.status_message.showMessage(
                    f"Previous input device '{input_identity.name}' is disconnected; "
                    + ("waiting for the saved endpoint" if self.__dict__.get("_login_startup", False)
                       else "using the default until it returns")
                )

        # Restore output device
        output_identity = self.config.last_output_device_identity
        if output_identity is None and self.config.last_output_device:
            output_identity = coerce_device_identity(self.config.last_output_device)
        if output_identity is not None:
            index = self._find_combo_index_by_identity(
                self.output_choice, output_identity
            )
            if index >= 0:
                self.output_choice.setCurrentIndex(index)
                resolved = self._combo_device_identity(self.output_choice)
                if (
                    resolved is not None
                    and resolved.to_dict() != output_identity.to_dict()
                ):
                    self.config.last_output_device = resolved.name
                    self.config.last_output_device_identity = resolved
                    config_dirty = True
                restored_count += 1
            else:
                self.output_choice.setCurrentIndex(-1)
                self.output_choice.setPlaceholderText("Select a replacement destination")
                self.status_message.showMessage(
                    f"Destination '{output_identity.name}' is disconnected; select a replacement"
                )

        # Restore preset (startup preset takes priority over last used)
        preset_loaded = False

        # Check startup preset first
        if self.config.startup_preset:
            custom_presets = list_presets()
            preset_id, custom_startup_preset = _startup_preset_selection(
                self.config.startup_preset, custom_presets
            )
            if preset_id != self.config.startup_preset:
                self.config.startup_preset = preset_id
                config_dirty = True
            preset_name = (
                custom_startup_preset[0]
                if custom_startup_preset is not None
                else _startup_preset_display_name(preset_id)
            )
            # Try built-in presets
            if preset_id.startswith(STARTUP_BUILTIN_PREFIX):
                preset_key = preset_id[len(STARTUP_BUILTIN_PREFIX) :]
                if preset_key in BUILTIN_PRESETS:
                    preset = BUILTIN_PRESETS[preset_key]
                    preset_loaded = self._apply_preset(
                        preset,
                        preset_key=preset_key,
                        persist_last_used=False,
                    )
                    if preset_loaded:
                        self.status_message.showMessage(f"Startup preset: {preset_name}", 5000)
            # Try custom presets
            elif preset_id.startswith(
                (STARTUP_CUSTOM_PREFIX, STARTUP_CUSTOM_FILE_PREFIX)
            ):
                if custom_startup_preset is not None:
                    _custom_name, filepath = custom_startup_preset
                    try:
                        preset = load_preset(filepath)
                        preset_loaded = self._apply_preset(
                            preset,
                            preset_path=filepath,
                            persist_last_used=False,
                        )
                        if preset_loaded:
                            self.status_message.showMessage(
                                f"Startup preset: {preset_name}", 5000
                            )
                    except Exception:
                        logger.warning(
                            "Failed to load startup preset %s",
                            preset_name,
                            exc_info=True,
                        )
                        self.status_message.showMessage(
                            f"Failed to load startup preset: {preset_name}", 5000
                        )
            # Try legacy unresolved custom display name.
            else:
                for name, filepath in custom_presets:
                    if name == preset_id:
                        try:
                            preset = load_preset(filepath)
                            preset_loaded = self._apply_preset(
                                preset,
                                preset_path=filepath,
                                persist_last_used=False,
                            )
                            if preset_loaded:
                                self.status_message.showMessage(
                                    f"Startup preset: {preset_name}", 5000
                                )
                        except Exception:
                            logger.warning(
                                "Failed to load startup preset %s",
                                preset_name,
                                exc_info=True,
                            )
                            self.status_message.showMessage(
                                f"Failed to load startup preset: {preset_name}", 5000
                            )
                        break
            if not preset_loaded:
                logger.warning(
                    "Startup preset %r not found; falling back to last used",
                    preset_name,
                )
                self._report_preset_problem(
                    "startup_preset",
                    self.config.startup_preset,
                    f"Startup preset '{preset_name}' could not be loaded",
                    5000,
                )

        startup_preset_failed = bool(self.config.startup_preset) and not preset_loaded
        # Fall back to last_preset if startup_preset not set or not found
        if not preset_loaded and self.config.last_preset:
            try:
                # Check if it's a built-in preset
                if self.config.last_preset.startswith("builtin:"):
                    preset_key = self.config.last_preset[8:]  # Remove "builtin:" prefix
                    if preset_key in BUILTIN_PRESETS:
                        preset = BUILTIN_PRESETS[preset_key]
                        preset_loaded = self._apply_preset(preset, preset_key=preset_key)
                        if not preset_loaded:
                            self.config.last_preset = ""
                            config_dirty = True
                        restored_count += int(preset_loaded)
                    else:
                        self._report_preset_problem(
                            "last_preset",
                            self.config.last_preset,
                            f"Previous preset '{preset_key}' not found, starting with defaults",
                        )
                        self.config.last_preset = ""
                        config_dirty = True
                else:
                    # It's a file path
                    preset_path = Path(self.config.last_preset)
                    if preset_path.exists():
                        preset = load_preset(preset_path)
                        preset_loaded = self._apply_preset(preset, preset_path=preset_path)
                        if not preset_loaded:
                            self.config.last_preset = ""
                            config_dirty = True
                        restored_count += int(preset_loaded)
                    else:
                        self._report_preset_problem(
                            "last_preset",
                            self.config.last_preset,
                            "Previous preset file not found, starting with defaults",
                        )
                        self.config.last_preset = ""
                        config_dirty = True
            except (OSError, ValueError, PresetValidationError) as e:
                logger.warning("Preset restore failed", exc_info=True)
                warning = f"Could not restore the previous preset; starting with defaults. {e}"
                self.config.load_warning = "\n".join(
                    text for text in (self.config.load_warning, warning) if text
                )
                # A file that cannot be read right now may load next time (load_preset wraps OS errors in
                # PresetValidationError); only a bad one loses its reference.
                if not (isinstance(e, OSError) or isinstance(e.__cause__, OSError)):
                    self.config.last_preset = ""
                    cleared = self._save_config_safely()
                    if not cleared:
                        self.config.load_warning += "\nThe cleared preset reference could not be saved."

        self._startup_restore_ready = (
            not startup_preset_failed and (preset_loaded or not requested_preset)
        )
        # Show appropriate status message
        if self.config.load_warning:
            self.status_message.showMessage(self.config.load_warning)
        elif self.__dict__.get("_last_preset_identity_persisted") is False:
            self.status_message.showMessage(
                "Preset applied for this session, but could not be remembered for the next launch",
                6000,
            )
        elif restored_count == 0:
            self.status_message.showMessage("Ready")
        elif restored_count < 3:
            self.status_message.showMessage(
                "Restored partial settings (some devices/presets unavailable)"
            )
        else:
            self.status_message.showMessage("Restored settings from previous session")

        self._apply_latency_compensation_for_current_devices()
        if (
            "input_channel_mode_choice" in self.__dict__
            and "input_cleanup_mode_choice" in self.__dict__
        ):
            self._apply_input_preferences_for_current_route()

        # Route-specific DSP is more specific than the generic startup/last-used
        # preset and is intentionally applied only after both endpoints resolve.
        if not self.__dict__.get("_login_startup", False):
            self._apply_bound_preset_for_current_route()

        self.input_choice.blockSignals(False)
        self.output_choice.blockSignals(False)
        if "input_channel_mode_choice" in self.__dict__:
            self.input_channel_mode_choice.blockSignals(False)
        if "input_cleanup_mode_choice" in self.__dict__:
            self.input_cleanup_mode_choice.blockSignals(False)

        if config_dirty:
            if not self._save_config_safely():
                self.status_message.showMessage(
                    "Restored settings applied for this session, but could not be saved",
                    6000,
                )

    def _on_device_changed(self, changed: ChoiceState | None = None):
        """Save the route; only the picker the user changed, as the other may show a stand-in.

        A saved device that is unplugged leaves its picker on the default device or on nothing,
        and the saved one must still be there when it returns. No ``changed`` saves both.
        """
        if self.__dict__.get("_login_startup_deadline") is not None:
            self._cancel_login_startup()
        self._login_startup = False
        banner = self.shell_texts["device_warning_banner"]
        if banner.text == NO_DESTINATION_WARNING:
            banner.update(shown=False)
        if hasattr(self, "config"):  # Check config is initialized
            if changed is not self.output_choice:
                self._set_problem("input_device", None)
                input_identity = self._combo_device_identity(self.input_choice)
                self.config.last_input_device_identity = input_identity
                self.config.last_input_device = self._device_name_from_identity(input_identity)
            if changed is not self.input_choice:
                self._set_problem("output_device", None)
                output_identity = self._combo_device_identity(self.output_choice)
                self.config.last_output_device_identity = output_identity
                self.config.last_output_device = self._device_name_from_identity(output_identity)
            self._apply_input_preferences_for_current_route()
            saved = self._save_config_safely()
            self._apply_latency_compensation_for_current_devices()
            self._apply_bound_preset_for_current_route()
            self._sync_calibration_evidence(force_reset=True)
            if not saved:
                self.status_message.showMessage(
                    "Device route applied for this session, but could not be saved",
                    6000,
                )
            self._update_session_summary()

    def _on_input_channel_mode_changed(self):
        """Persist and apply the selected input channel mixdown mode."""
        if not hasattr(self, "config"):
            return
        mode = self.input_channel_mode_choice.currentData()
        if not self._is_valid_input_channel_mode(mode):
            mode = "phase_safe_mono"
        cleanup_mode = getattr(self.config, "input_cleanup_mode", "off")
        self._store_input_preference(channel_mode=mode, cleanup_mode=cleanup_mode)
        self._apply_input_channel_mode(mode)
        self._save_input_preferences()
        self._sync_calibration_evidence(force_reset=True)

    def _on_input_cleanup_mode_changed(self):
        """Persist and apply the selected adaptive input cleanup mode."""
        if not hasattr(self, "config"):
            return
        mode = self.input_cleanup_mode_choice.currentData()
        if not self._is_valid_input_cleanup_mode(mode):
            mode = "off"
        channel_mode = getattr(self.config, "input_channel_mode", "phase_safe_mono")
        self._store_input_preference(channel_mode=channel_mode, cleanup_mode=mode)
        self._apply_input_cleanup_mode(mode)
        self._save_input_preferences()
        self._sync_calibration_evidence(force_reset=True)

    def _save_input_preferences(self) -> None:
        saved = self._save_config_safely()
        if not saved:
            self.status_message.showMessage(
                "Input settings applied for this session, but could not be saved", 6000
            )

    def _run_engine(
        self, name: str, call: Callable[[], Any], done: Callable[[Any], object]
    ) -> None:
        """Run a blocking engine call on the worker, with the engine controls locked until ``done`` ran."""
        if self._shut_down:
            return
        if self._engine.threaded:  # a call that returns at once would only cost the focus of a button
            self._engine_busy = name
            self._sync_processing_controls()

        def finish(outcome: Any) -> None:
            self._engine_busy = None
            if self._shut_down:
                return
            try:
                done(outcome)
            finally:
                self._sync_processing_controls()

        self._engine.submit(call, finish)

    def _start_processing(self, *, interactive: bool = True) -> bool:
        """Start audio processing and wait for the engine; login and setup need the answer."""
        route = self._prepare_start(interactive)
        if isinstance(route, bool):
            return route
        return self._finish_start(
            attempt(partial(start_processor_for_route, self.processor, *route)), interactive
        )

    def _on_start_clicked(self) -> None:
        """Start from the button or menu; the engine starts on the worker, so the window stays alive."""
        if self.__dict__.get("_engine_busy"):
            return
        if self.__dict__.get("_recovery_polling"):
            # Queued behind that call, the start would meet the streams a retry reopened.
            self._start_after_recovery = True
            return
        route = self._prepare_start(True)
        if not isinstance(route, bool):
            self._run_engine(
                "start",
                partial(start_processor_for_route, self.processor, *route),
                partial(self._finish_start, interactive=True),
            )

    def _prepare_start(
        self, interactive: bool
    ) -> bool | tuple[DeviceIdentity | None, DeviceIdentity | None]:
        """The route to start, or the final answer when there is nothing to start."""
        if interactive:
            if self.__dict__.get("_login_startup_deadline") is not None:
                self._cancel_login_startup()
            self._login_startup = False
            self._login_startup_message = ""
        if self.processor.is_running():
            if DEBUG:
                logger.debug("Start processing clicked, but processor already running")
            return True

        if "configuration" in self.__dict__.get("_temporary_mute_reasons", set()):
            self.status_message.showMessage(
                "Audio remains stopped and muted because configuration recovery failed; "
                "reload the preset or restart AudioForge before starting",
                7000,
            )
            self._sync_processing_controls()
            return False

        if self._combo_device_identity(self.output_choice) is None:
            self.status_message.showMessage("Select an available destination before starting")
            banner = self.shell_texts["device_warning_banner"]
            if not banner.shown:
                banner.update(text=NO_DESTINATION_WARNING, shown=True)
            return False

        input_device = self._device_selection_to_name(self.input_choice) or None
        output_device = self._device_selection_to_name(self.output_choice) or None
        self._apply_input_preferences_for_current_route()
        # Set the persisted mute before native startup so the first buffer is safe.
        self._apply_output_mute()
        if not interactive and self._output_mute_error is not None:
            return False

        if DEBUG:
            logger.debug(
                "Starting processing - Input: %s, Output: %s",
                input_device or "(default)",
                output_device or "(default)",
            )
        return (
            self._combo_device_identity(self.input_choice),
            self._combo_device_identity(self.output_choice),
        )

    def _finish_start(self, outcome: Any, interactive: bool) -> bool:
        """Show how the engine start went: ``outcome`` is its result or the exception it raised."""
        try:
            if isinstance(outcome, Exception):
                raise outcome
            result = outcome
            self._apply_latency_compensation_for_current_devices()
            # Native start clears temporary mute state; restore both owners.
            if DEBUG:
                logger.debug("Restoring output mute state after processing start")
            self._apply_output_mute()
            if not interactive and self._output_mute_error is not None:
                self.processor.stop()
                raise RuntimeError("Output mute state could not be restored")
            self.status_message.showMessage(f"Processing: {result}")
            self.start_btn_action.setEnabled(False)
            self.stop_btn_action.setEnabled(True)
            self.input_choice.setEnabled(False)
            self.output_choice.setEnabled(False)
            self.refresh_btn_action.setEnabled(False)
            self._processing = True
            self._update_session_summary()
            self._sync_meter_timer()
            if DEBUG:
                logger.debug("Processing started: %s", result)
            return True
        except Exception as e:
            logger.exception("Start processing failed")
            if not interactive and self.processor.is_running():
                try:
                    self.processor.stop()
                except Exception:
                    logger.exception("Could not stop audio after failed login startup")
            error_msg = str(e)
            # Provide actionable guidance based on error type
            if "device" in error_msg.lower() or "audio" in error_msg.lower():
                guidance = (
                    "Try these steps:\n"
                    "1. Press the refresh button (the circular arrow) next to the Output list\n"
                    "2. Ensure your microphone is connected\n"
                    "3. Check Windows audio settings\n"
                    "4. Try selecting a different device"
                )
            else:
                guidance = (
                    "Try these steps:\n"
                    "1. Stop and restart the application\n"
                    "2. Check that no other app is using the audio device"
                )
            if interactive:
                QMessageBox.critical(
                    self,
                    "Error Starting Processing",
                    f"Failed to start audio processing:\n\n{e}\n\n{guidance}",
                )
            self.status_message.showMessage(f"Could not start processing: {e}")
            self._sync_processing_controls()
            return False

    def _sync_processing_controls(self) -> None:
        """Reflect the native processor state after recovery or a failed start."""
        # The engine is mid-call: nothing may be pressed, and the processor is not asked.
        busy = self.__dict__.get("_engine_busy") is not None
        running = not busy and bool(self.processor.is_running())
        idle = not (busy or running)
        self.start_btn_action.setEnabled(idle)
        self.stop_btn_action.setEnabled(
            not busy
            and (running or self._restart_pending() or self.__dict__.get("_login_startup_deadline") is not None)
        )
        self.input_choice.setEnabled(idle)
        self.output_choice.setEnabled(idle)
        self.refresh_btn_action.setEnabled(idle)
        if busy:
            return
        update_summary = getattr(self, "_update_session_summary", None)
        if callable(update_summary):
            update_summary()
        sync_meters = getattr(self, "_sync_meter_timer", None)
        if callable(sync_meters):
            sync_meters()

    def _restart_pending(self) -> bool:
        """Whether the engine, streams stopped, is waiting to retry a failed restart; Stop cancels it."""
        getter = getattr(self.processor, "is_recovery_requested", None)
        return callable(getter) and bool(getter())

    def _stop_processing(self):
        """Stop audio processing and wait for the engine."""
        if self._prepare_stop():
            self._finish_stop(attempt(self.processor.stop))

    def _on_stop_clicked(self) -> None:
        """Stop from the button or menu; the engine stops on the worker, so the window stays alive."""
        if not self.__dict__.get("_engine_busy") and self._prepare_stop():
            self._run_engine("stop", self.processor.stop, self._finish_stop)

    def _prepare_stop(self) -> bool:
        """Whether there is a running engine, or a restart retry, to stop."""
        if self.__dict__.get("_login_startup_deadline") is not None:
            self._cancel_login_startup()
        if not (self.processor.is_running() or self._restart_pending()):
            self._sync_processing_controls()
            if DEBUG:
                logger.debug("Stop processing clicked, but processor is not running")
            return False

        if DEBUG:
            logger.debug("Stopping processing")
        return True

    def _finish_stop(self, outcome: Any) -> None:
        """Show how the engine stop went: ``outcome`` is None or the exception it raised."""
        try:
            if isinstance(outcome, Exception):
                raise outcome
            self.status_message.showMessage("Processing stopped")
            self.start_btn_action.setEnabled(True)
            self.stop_btn_action.setEnabled(False)
            self.input_choice.setEnabled(True)
            self.output_choice.setEnabled(True)
            self.refresh_btn_action.setEnabled(True)
            self._processing = False
            self._device_wait_since = None
            self._sync_meter_timer()
            self._update_session_summary()
            if DEBUG:
                logger.debug("Processing stopped")
        except Exception as e:
            logger.exception("Stop processing failed")
            QMessageBox.critical(self, "Error", f"Failed to stop processing:\n{e}")

    def _stop_audio_on_worker(self) -> None:
        """Stop the engine after a failed configuration; a start or recovery on the worker may hold the lock."""
        self._engine.submit(self.processor.stop, self._audio_stop_failed)

    def _audio_stop_failed(self, outcome: Any) -> None:
        if isinstance(outcome, Exception) and not self._shut_down:
            logger.error("Stopping audio after a failed configuration failed", exc_info=outcome)
            self.status_message.showMessage(f"Stopping audio failed: {outcome}; restart AudioForge")

    def _on_auto_eq_clicked(self):
        """Open Auto-EQ calibration dialog."""
        if DEBUG:
            logger.debug("Auto-EQ button clicked; opening calibration dialog")

        # Commit any pending manual edit before the modal operation.
        self.capture_pre_auto_eq_state()

        dialog = CalibrationDialog(self)
        # Connect signal to handle auto-EQ completion (preset save, undo button enable)
        dialog.auto_eq_applied.connect(self.on_auto_eq_applied)
        self._calibration_dialog_open = True
        self._history_transaction_depth += 1
        try:
            with recovery_paused(self.processor):
                _exec_and_release(dialog)  # Modal dialog - blocks until user closes
        finally:
            self._history_transaction_depth -= 1
            self._calibration_dialog_open = False
        self._sync_calibration_evidence()
        if DEBUG:
            logger.debug("Calibration dialog closed, result=%s", dialog.result())
            is_running = self.processor.is_running()
            logger.debug("After calibration - processor running=%s", is_running)

    def _on_test_my_sound_clicked(self) -> None:
        from .listening_comparison_dialog import open_test_my_sound

        open_test_my_sound(self)

    def _on_auto_voice_setup_clicked(self) -> bool:
        """Open the multi-stage voice setup wizard."""
        self._commit_pending_configuration_snapshot()
        dialog = VoiceSetupDialog(self)
        applied = False

        def on_applied(target_curve: str) -> None:
            nonlocal applied
            applied = True
            self.on_voice_setup_applied(target_curve)

        dialog.setup_applied.connect(on_applied)
        self._calibration_dialog_open = True
        self._history_transaction_depth += 1
        try:
            with recovery_paused(self.processor):
                _exec_and_release(dialog)
        finally:
            self._history_transaction_depth -= 1
            self._calibration_dialog_open = False
        self._sync_calibration_evidence()
        return applied

    def record_setup_result(self, result: dict) -> None:
        """Keep the latest setup outcome for the diagnostics export; the setup dialogs call this."""
        try:
            clean = sanitize_setup_result(result)
        except Exception:
            logger.exception("Setup result could not be recorded")
            return
        if clean is not None:
            self.last_setup_result = clean

    def capture_pre_auto_eq_state(self):
        """Commit pending edits before Auto-EQ starts its transaction."""
        self._commit_pending_configuration_snapshot()

    def _initialize_configuration_history(self) -> None:
        """Create the immutable baseline after startup restoration."""
        preset = self._get_current_preset()
        if self._saved_preset_payload is None:
            self.current_preset_name = "Default"
            self.current_preset_path = None
            self._saved_preset_payload = self._preset_payload(preset)
            self._saved_processing_mode = self._processing_mode()
            self.preset_modified = False
        snapshot = ConfigurationSnapshot.from_preset(
            preset,
            label="Startup configuration",
            source="startup",
            noise_reference_reliability=self.compressor_state.get_settings(
                include_calibration=True
            )["noise_reference_reliability"],
            calibration_context_key=self._calibration_context_key(),
            processing_mode=self._processing_mode(),
            preset_record=self._mirror_preset_record(),
        )
        self._configuration_history.initialize(snapshot)
        self._current_value_provenance = dict(snapshot.to_preset().value_provenance)
        self._history_ready = True
        self._sync_calibration_evidence()
        self._update_history_actions()
        self._update_session_summary()

    def _calibration_context_key(self) -> str | None:
        route = self._current_device_route_key()
        if route is None:
            return None
        return json.dumps((
            route,
            self.config.input_channel_mode,
            self.config.input_cleanup_mode,
            self._current_capture_format_context(),
            self._processing_mode(),
        ))

    def _connect_configuration_history_inputs(self) -> None:
        """Observe processing controls and coalesce one user gesture."""
        self.eq_state.configurationEditStarted.connect(
            self._begin_configuration_transaction
        )
        self.eq_state.configurationEditFinished.connect(
            self._end_configuration_transaction
        )
        for state in (
            self.eq_state, self.gate_state, self.deesser_state,
            self.compressor_state, self.limiter_state, self.noise_suppression_state,
        ):
            state.configurationEdited.connect(self._queue_configuration_snapshot)

    def _begin_configuration_transaction(self) -> None:
        """Suppress intermediate history entries for a compound gesture."""
        if self._history_transaction_depth == 0:
            self._commit_pending_configuration_snapshot()
        self._history_transaction_depth += 1

    def _end_configuration_transaction(self, label: str) -> None:
        """Commit one final entry after a compound gesture."""
        if self._history_transaction_depth <= 0:
            logger.warning("Unbalanced configuration-history transaction")
            return
        self._history_transaction_depth -= 1
        if self._history_transaction_depth == 0:
            self._commit_pending_configuration_snapshot(
                label=label,
                source="eq_graph",
            )

    def _queue_configuration_snapshot(self, *_args) -> None:
        """Debounce UI signals into one immutable history entry."""
        if (
            not self._history_ready
            or self._history_replaying
            or self._history_transaction_depth > 0
        ):
            return
        self.preset_modified = True
        self._sync_calibration_evidence()
        self._update_session_summary()
        self._history_timer.start()

    def _commit_pending_configuration_snapshot(
        self,
        *,
        label: str = "Manual processing edit",
        source: str = "ui",
        provenance: dict[str, str] | None = None,
    ) -> bool:
        """Validate and record the current processing configuration."""
        if not self._history_ready or self._history_replaying:
            return False
        try:
            for name in ("limiter_state", "deesser_state", "compressor_state", "gate_state", "eq_state"):
                state = getattr(self, name, None)
                if state is not None:
                    state.flush()
        except RuntimeError:
            return False
        finally:
            self._history_timer.stop()
        if "configuration" in self.__dict__.get("_temporary_mute_reasons", set()):
            return False
        preset = self._get_current_preset()
        current = self._configuration_history.current
        if provenance is not None:
            preset.value_provenance = dict(provenance)
        elif current is not None:
            preset.value_provenance = explicit_provenance_after_edit(
                current,
                preset,
            )
        try:
            snapshot = ConfigurationSnapshot.from_preset(
                preset,
                label=label,
                source=source,
                noise_reference_reliability=self.compressor_state.get_settings(
                    include_calibration=True
                )["noise_reference_reliability"],
                calibration_context_key=self._calibration_context_key(),
                processing_mode=self._processing_mode(),
                preset_record=self.__dict__.get("_preset_record"),
            )
            recorded = self._configuration_history.record(snapshot)
        except (PresetValidationError, TypeError, ValueError) as error:
            logger.warning(
                "Configuration history snapshot rejected: %s",
                error,
            )
            self.status_message.showMessage(
                "Could not record this configuration edit",
                5000,
            )
            return False
        self._sync_calibration_evidence()
        if recorded:
            self._current_value_provenance = dict(snapshot.to_preset().value_provenance)
            if source == "ui":
                self.eq_state.set_auto_eq_diagnostics(None)
        self._set_preset_modified()
        self._update_history_actions()
        return recorded

    def _restore_configuration_snapshot(
        self,
        snapshot: ConfigurationSnapshot,
    ) -> None:
        """Restore one validated snapshot without creating a new entry."""
        preset = snapshot.to_preset()
        self._history_replaying = True
        try:
            self.apply_processing_configuration(
                preset,
                noise_reference_reliability=(
                    snapshot.noise_reference_reliability
                    if snapshot.calibration_context_key is not None
                    and snapshot.calibration_context_key == self._calibration_context_key()
                    else 0.0
                ),
                processing_mode=snapshot.processing_mode,
            )
            if snapshot.preset_record is not None:
                self._restore_preset_record(snapshot.preset_record)
            self._set_preset_modified()
        finally:
            self._history_replaying = False

    def _update_history_actions(self) -> None:
        history = self._configuration_history
        undo_label = history.undo_label
        redo_label = history.redo_label
        if self._undo_action is not None:
            self._undo_action.setEnabled(history.can_undo)
            self._undo_action.setText(f"&Undo {undo_label}" if undo_label else "&Undo")
        if self._redo_action is not None:
            self._redo_action.setEnabled(history.can_redo)
            self._redo_action.setText(f"&Redo {redo_label}" if redo_label else "&Redo")
        self.redo_button_action.setEnabled(history.can_redo)
        if self._undo_auto_eq_button_action is not None:
            self._undo_auto_eq_button_action.setEnabled(history.can_undo)
            self._undo_auto_eq_button_action.setToolTip(
                f"Undo {undo_label} (Ctrl+Z)"
                if undo_label
                else "No processing-configuration edit to undo"
            )

    def undo_configuration(self) -> None:
        """Undo one validated processing-configuration snapshot."""
        self._commit_pending_configuration_snapshot()
        undone_label = self._configuration_history.undo_label
        try:
            restored = self._configuration_history.undo(
                self._restore_configuration_snapshot
            )
        except Exception as error:
            logger.warning("Configuration undo failed", exc_info=True)
            QMessageBox.warning(
                self,
                "Undo Failed",
                f"The previous configuration could not be restored:\n{error}",
            )
            return
        if restored is None:
            self.status_message.showMessage("No configuration change to undo", 3000)
        else:
            self.status_message.showMessage(
                f"Undid: {undone_label or restored.label}",
                3000,
            )
        self._update_history_actions()

    def redo_configuration(self) -> None:
        """Redo one validated processing-configuration snapshot."""
        # A fresh edit invalidates the redo branch even if its debounce timer
        # has not fired yet. Commit it before querying history so redo cannot
        # overwrite an unrecorded user change.
        self._commit_pending_configuration_snapshot()
        try:
            restored = self._configuration_history.redo(
                self._restore_configuration_snapshot
            )
        except Exception as error:
            logger.warning("Configuration redo failed", exc_info=True)
            QMessageBox.warning(
                self,
                "Redo Failed",
                f"The next configuration could not be restored:\n{error}",
            )
            return
        if restored is None:
            self.status_message.showMessage("No configuration change to redo", 3000)
        else:
            self.status_message.showMessage(f"Redid: {restored.label}", 3000)
        self._update_history_actions()

    def _prompt_save_current_preset(
        self,
        *,
        title: str,
        question: str,
        preset_name: str,
        description: str,
    ) -> None:
        reply = QMessageBox.question(
            self,
            title,
            question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        preset = self._get_saveable_preset()
        if preset is None:
            return
        preset.name = preset_name
        preset.description = description
        preset.version = __version__
        self._save_preset_file(preset)

    def _get_saveable_preset(self) -> Preset | None:
        """Save accepted sound, never an interactive write still awaiting the DSP."""
        try:
            self._flush_processing_configuration_writes()
        except RuntimeError:
            return None
        if "configuration" in self.__dict__.get("_temporary_mute_reasons", set()):
            self.status_message.showMessage("Restore a complete configuration before saving", 7000)
            return None
        return self._get_current_preset()

    def _save_preset_file(self, preset: Preset, *, filepath: Path | None = None) -> Path | None:
        self._last_preset_identity_persisted = True
        try:
            try:
                filepath = save_preset(preset, filepath=filepath, overwrite=filepath is not None)
            except FileExistsError:
                confirm_reply = QMessageBox.question(
                    self,
                    "Overwrite Preset?",
                    f"A preset file for '{preset.name}' already exists. Overwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if confirm_reply != QMessageBox.StandardButton.Yes:
                    return None
                filepath = save_preset(preset, overwrite=True)

            previous_last_preset = self.config.last_preset
            self.config.last_preset = str(filepath)
            persisted = self._save_config_safely()
            if not persisted:
                self.config.last_preset = previous_last_preset
                self._last_preset_identity_persisted = False
            self._set_preset_identity(preset, path=filepath, persist_last_used=persisted)
            self._set_problem("last_preset", None)
            message = f"Preset saved: {filepath.name}"
            if self._processing_mode() == "raw":
                message += "; Raw Monitor is session-only and was not stored in the preset"
            if not persisted:
                message += "; could not remember it for the next launch"
            self.status_message.showMessage(message, 6000 if not persisted else 3000)
            return filepath
        except (IOError, OSError, TypeError, ValueError) as exc:
            logger.warning("Preset save failed", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save preset:\n{exc}\n\n"
                "Check you have write permission to the presets folder.",
            )
            return None

    def on_auto_eq_applied(self, target_curve: str):
        """
        Handle auto-EQ application completion.

        Shows undo button, prompts for preset save.

        Args:
            target_curve: The target curve used ('broadcast', 'podcast', etc.)
        """
        from ..config import generate_auto_eq_preset_name

        self._commit_pending_configuration_snapshot(
            label=f"Auto-EQ ({target_curve.title()})",
            source="auto_eq",
        )

        preset_name = generate_auto_eq_preset_name(target_curve)
        self._prompt_save_current_preset(
            title="Save Auto-EQ as Preset?",
            question=(
                f"Save these full processing-chain settings from Auto-EQ as preset "
                f"'{preset_name}'?"
            ),
            preset_name=preset_name,
            description=(
                "Complete processing preset with Auto-EQ using "
                f"the {target_curve.title()} tone preset"
            ),
        )

    def on_voice_setup_applied(self, target_curve: str):
        """Offer to save the applied voice-setup chain as a preset."""
        self._commit_pending_configuration_snapshot(
            label=f"Auto Voice Setup ({target_curve.title()})",
            source="voice_setup",
        )
        preset_name = f"Voice Setup {target_curve.title()}"
        self._prompt_save_current_preset(
            title="Save Voice Setup as Preset?",
            question=f"Save these calibrated voice-chain settings as preset '{preset_name}'?",
            preset_name=preset_name,
            description=(
                "Auto-generated voice chain with EQ, gate/VAD, de-esser, and compressor "
                f"tuned for the {target_curve.title()} target"
            ),
        )

    def _on_model_changed(self, index: int):
        state = self.noise_suppression_state
        models = state.models
        # The Quick picker activates the model that is already selected.
        if not 0 <= index < len(models) or models[index][0] == state.get_settings()["model"]:
            return
        try:
            state.set_value("model", models[index][0])
        except RuntimeError as error:
            QMessageBox.warning(self, "Model Switch Failed", str(error))
        else:
            self.status_message.showMessage(f"Switched to {models[index][1]}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._shut_down = False  # a window shown again after a refused quit must shut down again
        self._sync_meter_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._sync_meter_timer()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_meter_timer()
        elif (
            event.type() == QEvent.Type.ThemeChange
            and "status_message" in self.__dict__
            and contrast_theme_changed()
        ):
            self.status_message.showMessage(
                "Restart AudioForge to match the new Windows contrast colors", 8000
            )

    def _sync_meter_timer(self) -> None:
        timer = self.__dict__.get("meter_timer")
        if timer is None or _engine_in_use(self):
            return
        try:
            running = self.processor.is_running()
        except (AttributeError, OSError, RuntimeError):
            running = False
        if running and self.isVisible() and not self.isMinimized():
            if not timer.isActive():
                # The getters kept collecting while the timer was stopped; the
                # first tick must not report that whole gap as new clipping.
                self._note_meter_peaks(partial(_level_reading, self.processor), resumed=True)
                timer.start()
        else:
            timer.stop()
            if not running:
                self._invalidate_live_meters()

    def _invalidate_live_meters(self) -> None:
        if self.__dict__.get("_live_meters_invalidated"):
            return
        self.input_meter.set_unavailable()
        self.output_meter.set_unavailable()
        self.processing_meters.compressor_gr.set_gain_reduction(None)
        self.processing_meters.update_auto_makeup(None, None)
        self.processing_meters.text_states["current_release"].update(text="--")
        self.processing_meters.deesser_gr.set_gain_reduction(None)
        self.gate_state.update_vad_confidence(None)
        self._live_meters_invalidated = True

    def _update_meters(self):
        """Refresh available live measurements; keep recovery on its own timer."""
        if _engine_in_use(self):
            return
        try:
            running = self.processor.is_running()
        except (AttributeError, OSError, RuntimeError):
            running = False
        if not running:
            self._invalidate_live_meters()
            self._last_backend_warning = None
            self._reset_health_labels()
            return
        if self.__dict__.get("_hidden_to_tray") and self.isHidden():
            return
        self._live_meters_invalidated = False

        reading = partial(_level_reading, self.processor)
        self.input_meter.set_levels(reading("get_input_rms_db"), reading("get_input_peak_db"))
        self.output_meter.set_levels(reading("get_output_rms_db"), reading("get_output_peak_db"))
        self._note_meter_peaks(reading)
        self.processing_meters.compressor_gr.set_gain_reduction(reading("get_compressor_gain_reduction_db"))
        self.processing_meters.deesser_gr.set_gain_reduction(reading("get_deesser_gain_reduction_db"))
        self.processing_meters.update_current_release()
        try:
            auto_makeup_enabled = self.processor.get_compressor_auto_makeup_enabled()
        except (AttributeError, OSError, RuntimeError):
            auto_makeup_enabled = False
        self.processing_meters.update_auto_makeup(
            reading("get_compressor_current_lufs") if auto_makeup_enabled else None,
            reading("get_compressor_current_makeup_gain") if auto_makeup_enabled else None,
        )
        self.gate_state.update_vad_confidence(reading("get_vad_probability"))

    def _note_meter_peaks(self, reading, *, resumed: bool = False) -> None:
        """Take the held peaks and full-scale counts into the meter models.

        The ``take_*`` getters reset what they return, so this is the only
        caller in the app; dialogs and Health read the meter models. With
        ``resumed`` the peaks are taken and dropped and the counts only set a
        new baseline.
        """

        def count(name: str) -> int | None:
            value = reading(name)
            return None if value is None else int(value)

        input_peak = reading("take_input_peak_since_read")
        pre_limiter_peak = reading("take_output_pre_limiter_peak_since_read")
        output_peak = reading("take_output_peak_since_read")
        if resumed:
            input_peak = pre_limiter_peak = output_peak = None
            self.input_meter.forget_full_scale_count()
            self.output_meter.forget_full_scale_count()
        self.input_meter.note_peak(input_peak, input_peak, count("get_input_full_scale_count"))
        self.output_meter.note_peak(pre_limiter_peak, output_peak, count("get_output_full_scale_count"))
        if self.limiter_state.get_applied_settings()["enabled"]:
            # Health calls a limiter holding the output at its ceiling normal, so the
            # marker means a full-scale output only while the limiter is off. Bypass
            # and Raw monitor skip the lookahead limiter, but the final true-peak
            # limiter and the ceiling still apply there.
            self.output_meter.clear_clip_marker()

    def _update_diagnostics(self):
        """Update slower diagnostics and service recovery."""
        if _engine_in_use(self):
            return
        if self.__dict__.get("_login_startup_deadline") is not None:
            try:
                self._service_login_startup()
            except Exception:
                logger.exception("Login route restoration failed")
                self._cancel_login_startup("Login stopped: route restoration failed; open AudioForge")
                if self._tray_icon is None or not self._tray_icon.isVisible():
                    app = QGuiApplication.instance()
                    if app is not None:
                        app.exit(1)
        sync_meters = getattr(self, "_sync_meter_timer", None)
        if callable(sync_meters):
            sync_meters()
        if not self.processor.is_running():
            self._processing = False
            self._device_wait_since = None
            self._last_backend_warning = None
            self._reset_health_labels()
            # A failed native restart intentionally leaves recovery pending
            # while the stream is stopped. Keep servicing that request here.
            self._service_stream_recovery()
            self._sync_processing_controls()
            return

        try:
            for name in METER_GETTERS:
                getattr(self.processor, name)
            diagnostics = self.processor.get_runtime_diagnostics()
            input_rms = self.processor.get_input_rms_db()
            output_rms = self.processor.get_output_rms_db()
            output_buf = self.processor.get_output_buffer_samples()
            latency_ms = self.processor.get_latency_ms()
            dsp_time_ms = self.processor.get_dsp_time_smoothed_ms()
            input_buf = self.processor.get_input_buffer_smoothed_samples()
            rnnoise_buf = self.processor.get_buffer_smoothed_samples()
            input_callback_age_ms = self.processor.get_input_callback_age_ms()
            output_callback_age_ms = self.processor.get_output_callback_age_ms()
            if not isinstance(diagnostics, dict) or not all(math.isfinite(value) for value in (
                input_rms, output_rms, output_buf, latency_ms, dsp_time_ms,
                input_buf, rnnoise_buf, input_callback_age_ms, output_callback_age_ms,
            )):
                raise ValueError("Invalid live telemetry")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
            self._reset_health_labels()
            self._set_health_chip("health_summary_label", "Health: readings unavailable", "warn")
            self.shell_texts["health_advice_label"].update(text=(
                "AudioForge could not read the audio engine's status, so Health cannot be checked. "
                "Stop and start processing; if this keeps happening, restart AudioForge."
            ))
            # Missing presentation data must not suppress native recovery service.
            self._service_stream_recovery()
            return

        self._update_diagnostic_labels(
            diagnostics=diagnostics,
            latency_ms=latency_ms,
            dsp_time_ms=dsp_time_ms,
            input_buf=input_buf,
            output_buf=output_buf,
            rnnoise_buf=rnnoise_buf,
            input_rms_db=input_rms,
            output_rms_db=output_rms,
            input_callback_age_ms=input_callback_age_ms,
            output_callback_age_ms=output_callback_age_ms,
            input_peak_db=self.input_meter.recent_peak_db(HEALTH_PEAK_WINDOW_SECONDS),
            input_full_scale_samples=self.input_meter.full_scale_samples_within(HEALTH_PEAK_WINDOW_SECONDS),
            output_peak_db=self.output_meter.recent_peak_db(HEALTH_PEAK_WINDOW_SECONDS),
            output_at_full_scale=self.output_meter.clipped_within(HEALTH_PEAK_WINDOW_SECONDS),
            limiter_enabled=bool(self.limiter_state.get_settings()["enabled"]),
        )
        self._service_stream_recovery()

    def _update_diagnostic_labels(
        self,
        *,
        diagnostics: dict,
        latency_ms: float,
        dsp_time_ms: float,
        input_buf: int,
        output_buf: int,
        rnnoise_buf: int,
        input_rms_db: float | None = None,
        output_rms_db: float | None = None,
        input_callback_age_ms: int | None = None,
        output_callback_age_ms: int | None = None,
        input_peak_db: float | None = None,
        input_full_scale_samples: int = 0,
        output_peak_db: float | None = None,
        output_at_full_scale: bool = False,
        limiter_enabled: bool = True,
    ) -> None:
        """Update diagnostic status labels from a runtime diagnostic snapshot.

        The peak and clip arguments come from the meters' held peaks, so a clip
        stays visible for several seconds and the old per-tick clip counters
        no longer decide the Input and Output lines.
        """
        if not hasattr(self, "_stream_health"):
            self._stream_health = RecentStreamHealth()
        try:
            recent_events = self._stream_health.observe(diagnostics)
        except ValueError:
            self._reset_health_labels()
            self._set_health_chip("recovery_diag_label", "Recovery: counters could not be read", "warn")
            return
        recovery_pending = False
        for name in ("is_recovering", "is_recovery_requested"):
            getter = getattr(getattr(self, "processor", None), name, None)
            if callable(getter):
                try:
                    recovery_pending = recovery_pending or bool(getter())
                except (OSError, RuntimeError, TypeError, ValueError):
                    recovery_pending = True
        self._set_health_chip(
            "latency_label",
            f"Latency: about {latency_ms:.0f} ms (processing {dsp_time_ms:.1f} ms)",
            "info",
        )

        pipeline_buf = input_buf + rnnoise_buf
        if pipeline_buf < 960:
            buf_status = "OK"
            buf_state = "ok"
        elif pipeline_buf < 1920:
            buf_status = "FILLING UP"
            buf_state = "warn"
        else:
            buf_status = "BACKED UP"
            buf_state = "bad"
        self._set_health_chip(
            "buffer_label",
            f"Buffer: {buf_status} ({plural(pipeline_buf, 'sample')})",
            buf_state,
        )

        dropped = diagnostics.get("input_dropped_samples", 0)
        lock_contention = diagnostics.get("lock_contention_count", 0)
        non_finite = diagnostics.get("suppressor_non_finite_count", 0)
        restart_count = diagnostics.get("stream_restart_count", 0)
        underruns = int(diagnostics.get("output_underrun_total", 0) or 0)
        underrun_streak = int(diagnostics.get("output_underrun_streak", 0) or 0)
        previous_underruns = int(
            getattr(self, "_last_output_underrun_total", underruns) or 0
        )
        new_underruns_observed = underruns > previous_underruns
        self._last_output_underrun_total = underruns
        phase_warning_count = int(diagnostics.get("input_phase_warning_count", 0) or 0)
        previous_phase_warnings = int(
            getattr(self, "_last_input_phase_warning_count", phase_warning_count) or 0
        )
        new_phase_warning_observed = phase_warning_count > previous_phase_warnings
        self._last_input_phase_warning_count = phase_warning_count
        raw_input_stereo_correlation = diagnostics.get("input_stereo_correlation")
        if raw_input_stereo_correlation is None:
            input_stereo_correlation = None
        else:
            try:
                input_stereo_correlation = float(raw_input_stereo_correlation)
            except (TypeError, ValueError):
                input_stereo_correlation = None
        current_phase_warning = (
            input_stereo_correlation is not None
            and input_stereo_correlation < INPUT_PHASE_WARNING_CORRELATION
        )
        phase_rescue_strategy = str(
            diagnostics.get("input_phase_rescue_strategy", "none") or "none"
        )
        phase_rescue_active = phase_rescue_strategy not in {"", "none"}
        cleanup_mode = str(diagnostics.get("input_cleanup_mode", "off") or "off")
        cleanup_hum_detected = bool(
            diagnostics.get("input_cleanup_hum_detected", False)
        )
        cleanup_rumble_detected = bool(
            diagnostics.get("input_cleanup_rumble_detected", False)
        )
        output_recovery_events = int(
            diagnostics.get(
                "output_recovery_event_count",
                diagnostics.get("output_recovery_count", 0),
            )
            or 0
        )
        output_true_peak_count = int(
            diagnostics.get("output_true_peak_event_count", 0) or 0
        )
        previous_output_true_peak_count = int(
            getattr(self, "_last_output_true_peak_event_count", output_true_peak_count)
            or 0
        )
        new_output_true_peak_observed = (
            output_true_peak_count > previous_output_true_peak_count
        )
        self._last_output_true_peak_event_count = output_true_peak_count
        gate_chatter_count = int(diagnostics.get("gate_chatter_event_count", 0) or 0)
        new_gate_chatter_observed = "gate_chatter_event_count" in recent_events
        gate_auto_relax_active = bool(diagnostics.get("gate_auto_relax_active", False))
        rt_error_name = diagnostics.get("rt_error_name")
        rt_error_active = bool(rt_error_name and rt_error_name != "none")
        input_crest_db = diagnostics.get("input_crest_factor_db")
        output_lufs = diagnostics.get("output_short_term_lufs")
        output_true_peak_db = diagnostics.get("output_true_peak_db")
        output_true_peak_headroom_db = diagnostics.get("output_true_peak_headroom_db")
        limiter_history_db = float(
            diagnostics.get("limiter_gain_reduction_history_db", 0.0) or 0.0
        )
        true_peak_limiter_history_db = float(
            diagnostics.get("output_true_peak_gain_reduction_history_db", 0.0) or 0.0
        )
        try:
            output_true_peak_headroom = (
                None
                if output_true_peak_headroom_db is None
                else float(output_true_peak_headroom_db)
            )
        except (TypeError, ValueError):
            output_true_peak_headroom = None

        input_health_text, input_health_state = build_input_health_state(
            rms_db=input_rms_db,
            peak_db=input_peak_db,
            full_scale_samples=input_full_scale_samples,
            phase_rescue_active=phase_rescue_active,
            cleanup_rumble_detected=cleanup_rumble_detected,
            cleanup_hum_detected=cleanup_hum_detected,
            cleanup_mode=cleanup_mode,
            crest_factor_db=input_crest_db
            if isinstance(input_crest_db, (int, float))
            else None,
        )
        if phase_rescue_active:
            fix = PHASE_FIX_LABELS.get(phase_rescue_strategy, phase_rescue_strategy.replace("_", " "))
            input_health_text = f"Input: PHASE FIX ACTIVE ({fix})"
        self._set_health_chip(
            "input_health_label",
            input_health_text,
            input_health_state,
        )

        output_health_text, output_health_state = build_output_health_state(
            muted=_output_muted(self),
            rms_db=output_rms_db,
            at_full_scale=output_at_full_scale,
            limiter_enabled=limiter_enabled,
            peak_db=output_peak_db,
            true_peak_delta=new_output_true_peak_observed,
            true_peak_db=output_true_peak_db,
            true_peak_headroom_db=output_true_peak_headroom,
            short_term_lufs=output_lufs,
            limiter_history_db=limiter_history_db,
            true_peak_limiter_history_db=true_peak_limiter_history_db,
        )
        self._set_health_chip(
            "output_health_label",
            output_health_text,
            output_health_state,
        )

        if gate_auto_relax_active:
            gate_health_text = f"Gate: EASED ({plural(gate_chatter_count, 'time')})"
            gate_health_state = "warn"
        elif new_gate_chatter_observed:
            gate_health_text = f"Gate: CHATTERING ({plural(gate_chatter_count, 'time')})"
            gate_health_state = "warn"
        else:
            gate_health_text = "Gate: OK"
            gate_health_state = "ok"
        self._set_health_chip(
            "gate_health_label",
            gate_health_text,
            gate_health_state,
        )

        callback_ages = [
            age
            for age in (input_callback_age_ms, output_callback_age_ms)
            if age is not None and 0 <= age < NO_CALLBACK_YET
        ]
        not_started = NO_CALLBACK_YET in (input_callback_age_ms, output_callback_age_ms)
        if not_started and self._processing:
            now = time.monotonic()
            if self._device_wait_since is None:
                self._device_wait_since = now
            start_overdue = now - self._device_wait_since >= DEVICE_STARTUP_GRACE_SECONDS
        else:
            self._device_wait_since = None
            start_overdue = False
        if callback_ages or not_started:
            max_callback_age = max(callback_ages, default=0)
            callback_state = (
                "bad"
                if max_callback_age > DEVICE_ACTIVITY_BAD_MS
                else "warn"
                if max_callback_age > DEVICE_ACTIVITY_WARN_MS or start_overdue
                else "ok"
                if callback_ages
                else "idle"
            )
            callback_health_text = (
                f"Device activity: input {_ago_text(input_callback_age_ms)}, "
                f"output {_ago_text(output_callback_age_ms, output=True)}"
            )
        else:
            callback_health_text = "Device activity: --"
            callback_state = "idle"
        self._set_health_chip(
            "callback_health_label",
            callback_health_text,
            callback_state,
        )

        underrun_state = "warn" if underrun_streak or new_underruns_observed else "ok"
        underrun_text = f"Dropouts: {underruns}"
        if underrun_streak:
            underrun_text += f" ({underrun_streak} in a row)"
        self._set_health_chip(
            "underrun_health_label",
            underrun_text,
            underrun_state,
        )

        dropped_bits = [
            f"Drops: {dropped}",
            f"U:{underruns}",
            f"L:{lock_contention}",
            f"NF:{non_finite}",
            f"RS:{restart_count}",
        ]
        if underrun_streak:
            dropped_bits.append(f"US:{underrun_streak}")
        if phase_warning_count:
            dropped_bits.append(f"PH:{phase_warning_count}")
        if input_stereo_correlation is not None:
            dropped_bits.append(f"COR:{input_stereo_correlation:.2f}")
        self._extend_diag_tokens(
            dropped_bits,
            diagnostics,
            [
                ("input_cleanup_mode", "CLN"),
                ("input_cleanup_hum_detected", "HUM"),
                ("input_cleanup_rumble_detected", "RMB"),
                ("input_cleanup_high_pass_hz", "HPF"),
                ("input_phase_rescue_strategy", "PRS"),
                ("input_phase_estimated_delay_samples", "PDL"),
                ("input_phase_polarity_flipped", "PFL"),
            ],
        )
        self._extend_diag_tokens(
            dropped_bits,
            diagnostics,
            [
                ("input_backlog_recovery_count", "IBR"),
                ("input_backlog_dropped_samples", "IBD"),
                ("output_short_write_dropped_samples", "OSW"),
                ("rt_buffer_overflow_count", "RTO"),
                ("input_callback_error_count", "ICE"),
                ("output_callback_error_count", "OCE"),
                ("clip_event_count", "CL"),
                ("clip_peak_db", "PK"),
                ("input_crest_factor_db", "ICF"),
                ("output_clip_event_count", "OCL"),
                ("output_clip_peak_db", "OPK"),
                ("output_crest_factor_db", "OCF"),
                ("output_short_term_lufs", "LU"),
                ("output_true_peak_event_count", "OTP"),
                ("output_true_peak_db", "TPK"),
                ("output_true_peak_headroom_db", "TPH"),
                ("limiter_gain_reduction_history_db", "LGR"),
                ("output_true_peak_gain_reduction_history_db", "TPGR"),
                ("limiter_effective_ceiling_db", "LIM"),
                ("gate_chatter_event_count", "GCH"),
                ("gate_auto_relax_active", "GAR"),
                ("deesser_detector_confidence", "DSC"),
                ("noise_selection_dropped_count", "NSD"),
            ],
        )
        if rt_error_active:
            dropped_bits.append(f"RT:{rt_error_name}")
        # Most of these also have a row of their own; this row adds them all up.
        attention = [
            # Gate chatter has its own entry below; it is a setting, not an engine fault.
            (bool(recent_events - {"gate_chatter_event_count"}), "the audio engine reported a fault"),
            (recovery_pending, "a restart is pending"),
            (underrun_streak > 0 or new_underruns_observed, "the output ran out of audio"),
            (rt_error_active, "the audio engine reported an error"),
            (new_output_true_peak_observed, "the output went above its ceiling"),
            (new_gate_chatter_observed or gate_auto_relax_active, "the gate is chattering"),
            (
                phase_rescue_active or current_phase_warning or new_phase_warning_observed,
                "the microphone channels cancel each other",
            ),
            (cleanup_rumble_detected, "low rumble was detected"),
            (
                limiter_history_db >= 6.0 or true_peak_limiter_history_db >= 3.0,
                "the limiter is working hard",
            ),
            (
                output_true_peak_headroom is not None and output_true_peak_headroom < 0.75,
                "the output has little headroom",
            ),
        ]
        causes = [cause for active, cause in attention if active]
        dropped_state = "warn" if causes else "ok"
        dropped_detail = " | ".join(dropped_bits)
        if causes and not (dropped or underruns or restart_count):
            dropped_summary = f"Drops: nothing dropped, but {causes[0]}"
        else:
            dropped_summary = (
                f"Drops: {plural(dropped, 'sample')} dropped, {plural(underruns, 'dropout')}, "
                f"{plural(restart_count, 'restart')}"
            )
            if causes:
                dropped_summary += f" (needs attention: {causes[0]})"
        self._set_health_chip("dropped_label", dropped_summary, dropped_state)
        self.shell_texts["dropped_label"].update(tip=
            f"{DROPPED_DIAGNOSTICS_TOOLTIP}\n\nTechnical counters (for support reports):\n{dropped_detail}"
        )

        # A start that cannot load a model chosen while it ran has no caller to tell. Warn once
        # per new drop, before the backend warning so that one stays on the line.
        noise_drops = int(diagnostics.get("noise_selection_dropped_count", 0) or 0)
        if noise_drops > self._noise_drops_seen:
            self.status_message.showMessage(NOISE_SELECTION_DROPPED_WARNING, 8000)
        self._noise_drops_seen = noise_drops

        backend_available = diagnostics.get("noise_backend_available", True)
        backend_failed = diagnostics.get("noise_backend_failed", False)
        backend_error = diagnostics.get("noise_backend_error")
        noise_model = diagnostics.get("noise_model", "rnnoise")
        if noise_model != "rnnoise" and (backend_failed or not backend_available):
            warning = (
                "Noise suppression stopped working, so your voice is passing through unprocessed. "
                "Choose another model on the Noise Suppression card."
            )
            if backend_error:
                warning += f" Details: {backend_error}"
            if warning != self._last_backend_warning:
                self.status_message.showMessage(warning, 6000)
                self._last_backend_warning = warning
        elif backend_available:
            self._last_backend_warning = None

        try:
            noise_model = diagnostics.get("noise_model", "rnnoise")
            backend_ok = diagnostics.get("noise_backend_available", True)
            backend_failed = diagnostics.get("noise_backend_failed", False)
            backend_error = diagnostics.get("noise_backend_error")
            restart_count = diagnostics.get("stream_restart_count", 0)
            output_recovery_count = output_recovery_events
            non_finite = diagnostics.get("suppressor_non_finite_count", 0)
            suppressed = diagnostics.get("recovery_suppressed", False)

            backend_bits = [
                str(noise_model),
                "working" if backend_ok else "failed" if backend_failed else "not available",
            ]
            if non_finite:
                backend_bits.append(f"{plural(non_finite, 'invalid sample')} replaced")
            if backend_error:
                backend_bits.append("reported an error")
            for key, direction in (
                ("input_resampler_active", "input"), ("output_resampler_active", "output"),
            ):
                if diagnostics.get(key):
                    backend_bits.append(f"resampling {direction}")
            self._set_health_chip(
                "backend_diag_label",
                f"Backend: {', '.join(backend_bits)}",
                "ok" if backend_ok else "warn",
            )

            recovery_bits = [
                plural(restart_count, "restart"),
                plural(output_recovery_count, "output recovery", "output recoveries"),
            ]
            if suppressed:
                recovery_bits.append("paused during setup")
            reason = diagnostics.get("last_restart_reason")
            recent_recovery = bool(recent_events & {
                "stream_restart_count", "output_recovery_count", "output_recovery_event_count",
                "input_backlog_recovery_count",
            })
            if recovery_pending:
                recovery_bits.append("restart pending")
            elif recent_recovery:
                recovery_bits.append("restarted just now")
            self._set_health_chip(
                "recovery_diag_label",
                f"Recovery: {', '.join(recovery_bits)}",
                "warn"
                if recovery_pending or recent_recovery
                else ("info" if suppressed else "ok"),
            )
            self.shell_texts["recovery_diag_label"].update(tip=
                f"Last recovery: {reason}" if reason else "No recovery recorded."
            )
        except Exception:
            logger.debug("Diagnostic label update failed", exc_info=True)
        self._update_health_summary()

    def _update_health_summary(self) -> None:
        """Summarize the worst health chip and say what to do about it."""
        chips = [self.shell_texts[key] for key in HEALTH_DECISION_KEYS + HEALTH_LAYOUT_KEYS]
        levels = {chip.state for chip in chips}
        state = next((level for level in ("bad", "warn", "ok") if level in levels), "idle")
        if self.__dict__.get("_output_mute_error") and state in {"ok", "idle"}:
            state = "warn"
        # The chip's colour is its worst row, so its text names that row's problem.
        issues = [chip.text for chip in chips if chip.state == "bad"] or [
            chip.text for chip in chips if chip.state == "warn"
        ]
        text = issues[0] if issues else f"Health: {state.upper()}"
        self._set_health_chip("health_summary_label", text, state)
        advice = advice_for(issues[0]) if issues else ""
        if not advice:
            advice = {
                "ok": "Everything looks fine.",
                "idle": TEXT_SPECS["health_advice_label"].text,
            }.get(state, "Check the items marked below.")
        self.shell_texts["health_advice_label"].update(text=advice)
        self.shell_texts["health_summary_label"].update(tip=advice)
        self._update_session_summary()

    _recovery_service_failing = False

    def _service_stream_recovery(self) -> None:
        """Poll the engine's own stream recovery; it detects stalls, backs off and restarts."""
        if _engine_in_use(self) or self._shut_down:
            return
        # Not locked like a click: it asks the engine every tick and mostly has nothing to do.
        self._recovery_polling = True
        self._engine.submit(lambda: self.processor.service_recovery(), self._recovery_serviced)

    def _recovery_serviced(self, outcome: Any) -> None:
        """Show what the engine's own recovery did: ``outcome`` is its answer or the exception raised."""
        self._recovery_polling = False
        start = self.__dict__.pop("_start_after_recovery", False)
        if self._shut_down:
            return
        try:
            if isinstance(outcome, Exception):
                raise outcome
            recovery_result = outcome
            if recovery_result is not None:
                if recovery_result:
                    self._apply_latency_compensation_for_current_devices()
                    reason = ""
                    try:
                        reason = self.processor.get_last_restart_reason() or ""
                    except Exception:
                        reason = ""
                    if reason.startswith(STALL_REASON_PREFIX):
                        reason = "audio stalled"
                    suffix = f" after: {reason}" if reason else ""
                    self.status_message.showMessage(
                        f"Audio restarted automatically{suffix}; you may have heard a short gap",
                        4000,
                    )
                else:
                    err_msg = ""
                    try:
                        err_msg = self.processor.get_last_stream_error() or ""
                    except Exception:
                        err_msg = ""
                    message = "The selected devices are not available, so AudioForge keeps retrying"
                    self.status_message.showMessage(f"{message}: {err_msg}" if err_msg else message, 6000)
                self._sync_processing_controls()
            self._recovery_service_failing = False
        except Exception:
            # Runs every tick: log the first failure of a streak, not each repeat.
            if not self._recovery_service_failing:
                logger.warning("Rust recovery service failed", exc_info=True)
            self._recovery_service_failing = True
        if start:
            self._on_start_clicked()

    def _export_diagnostics(self) -> None:
        """Export an allowlisted support snapshot off the realtime path."""
        filepath, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export AudioForge Diagnostics",
            diagnostics_filename(__version__),
            "JSON Files (*.json);;All Files (*)",
        )
        if not filepath:
            return

        try:
            runtime = dict(self.processor.get_runtime_diagnostics())
            # Without levels a microphone that delivers silence looks healthy.
            for key in ("input_rms_db", "input_peak_db"):
                try:
                    runtime[key] = float(getattr(self.processor, f"get_{key}")())
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
                    runtime[key] = None  # one failed read must not lose the report
            preset = self._get_current_preset().to_dict()
            snapshot = build_diagnostics_snapshot(
                app_version=__version__,
                runtime_diagnostics=runtime,
                config=self.config,
                processing_settings=preset,
                input_device=self._combo_device_identity(self.input_choice),
                output_device=self._combo_device_identity(self.output_choice),
                processing_sample_rate_hz=int(self.processor.sample_rate()),
                output_sample_rate_hz=int(self.processor.output_sample_rate()),
                running=bool(self.processor.is_running()),
                setup_result=self.last_setup_result,
                interface=self._interface_snapshot(),
            )
            write_diagnostics_snapshot(filepath, snapshot)
        except Exception:
            logger.exception("Diagnostics export failed")
            QMessageBox.critical(
                self,
                "Diagnostics Export Failed",
                "AudioForge could not create the diagnostics file. Check that the folder "
                "is writable and has free space, then try again.",
            )
            return

        self.status_message.showMessage("Privacy-safe diagnostics exported", 4000)
        text = (
            "The snapshot was saved without raw audio, device names, "
            "environment variables, secrets, or arbitrary paths."
        )
        if snapshot["setup_result"] is not None:
            text += (
                " It lists the outcome, reason codes and numbers of your last Voice Setup or "
                "Auto-EQ run, but not the messages shown to you."
            )
        QMessageBox.information(self, "Diagnostics Exported", text)

    def _interface_snapshot(self) -> dict[str, str | None]:
        """Which view is showing and why, as fixed words for the diagnostics export."""
        if "quick_bridge" in self.__dict__:
            from .quick_shell import graphics_api

            return {"view": "quick", "classic_reason": "none", "graphics_api": graphics_api(self)}
        return {
            "view": "error",
            "classic_reason": self.quick_view_failure,
            "graphics_api": self.quick_view_graphics_api,
        }

    def _show_licenses(self):
        """Open the notices shipped with this copy of AudioForge."""
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
        notices = root / "licenses"
        if not (notices / "THIRD_PARTY_NOTICES.md").is_file() or not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(notices))
        ):
            QMessageBox.warning(
                self,
                "License Notices",
                f"Unable to open the bundled license notices. Check: {notices}",
            )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AudioForge",
            f"<h2>AudioForge v{__version__}</h2>"
            "<p>Low-latency microphone audio processor</p>"
            "<p>AudioForge source: MIT. Uses Qt and PySide6 under LGPLv3, "
            "with separately licensed dependencies. See Settings &gt; Help &gt; Licenses "
            "for the complete notices and library replacement instructions.</p>"
            "<p>Inspired by SteelSeries GG Sonar ClearCast AI</p>"
            "<h3>Processing Chain:</h3>"
            "<p>Mic -&gt; Input Cleanup -&gt; Gate -&gt; AI Noise -&gt; De-Esser -&gt; EQ -&gt; Comp -&gt; True-Peak Limiter -&gt; Output</p>"
            "<h3>Features:</h3>"
            "<ul>"
            "<li>Threshold and Silero VAD-assisted gating</li>"
            "<li>RNNoise plus opt-in DeepFilterNet suppression</li>"
            "<li>Phase-safe mono and tracked hum cleanup</li>"
            "<li>Auto-EQ and uncertainty-aware Auto Voice Setup</li>"
            "<li>10-band EQ, dynamic de-esser, and compressor</li>"
            "<li>Band-limited true-peak output protection</li>"
            "<li>Runtime health, recovery, and calibration diagnostics</li>"
            "</ul>"
            "<p><b>Target neural-suppression latency:</b> up to about 35ms</p>",
        )

    def _reset_dropped_samples(self):
        """Reset the dropped samples counter."""
        if self.processor:
            self.processor.reset_dropped_samples()
            self.status_message.showMessage("Dropped-sample count reset to zero", 3000)

    def _get_current_preset(self) -> Preset:
        """Get current settings as a Preset object."""
        gate_settings = self.gate_state.get_settings()
        eq_settings = self.eq_state.get_eq_settings()
        deesser_settings = self.deesser_state.get_settings()
        compressor_settings = self.compressor_state.get_settings()
        limiter_settings = self.limiter_state.get_settings()

        return Preset(
            name="Custom",
            description="User-defined preset",
            gate=GateSettings(**gate_settings),
            eq=eq_settings,
            rnnoise=RNNoiseSettings(**self.noise_suppression_state.get_settings()),
            deesser=DeEsserSettings(**deesser_settings),
            compressor=CompressorSettings(**compressor_settings),
            limiter=LimiterSettings(**limiter_settings),
            bypass=self._processing_mode() == "bypass",
            value_provenance=dict(self._current_value_provenance),
        )

    def _write_processing_configuration(self, preset: Preset) -> None:
        """Write a validated chain while the caller owns history and output mute."""
        self.noise_suppression_state.set_settings(asdict(preset.rnnoise))
        self.gate_state.set_settings(asdict(preset.gate))
        self.eq_state.set_settings(preset.eq.to_dict())
        self.deesser_state.set_settings(asdict(preset.deesser))
        self.compressor_state.set_settings(asdict(preset.compressor))
        self.limiter_state.set_settings(asdict(preset.limiter))
        self._set_processing_mode("bypass" if preset.bypass else "normal")
        self._current_value_provenance = dict(preset.value_provenance)

    def _cancel_processing_configuration_writes(self) -> None:
        """Discard interactive writes superseded by the bulk configuration."""
        for state in (self.eq_state, self.gate_state, self.deesser_state,
                      self.compressor_state, self.limiter_state):
            state.cancel()

    def _flush_processing_configuration_writes(self) -> None:
        """Finish queued state writes before releasing configuration mute."""
        for state in (self.eq_state, self.gate_state, self.deesser_state,
                      self.compressor_state, self.limiter_state):
            state.flush()

    def apply_processing_configuration(
        self,
        preset: Preset,
        *,
        noise_reference_reliability: float | None = None,
        processing_mode: str | None = None,
        compressor_metadata: dict[str, object] | None = None,
    ) -> None:
        """Apply all settings or restore them, without changing preset identity/history."""
        if processing_mode is not None and not self._is_valid_processing_mode(
            processing_mode
        ):
            raise ValueError(f"Unknown processing mode: {processing_mode!r}")
        if compressor_metadata is not None and compressor_metadata.keys() - {
            "dynamics_intensity", "dynamics_profile", "dynamics_customized",
        }:
            raise ValueError("Unknown compressor calibration metadata")
        candidate = ConfigurationSnapshot.from_preset(
            preset, label="Apply", source="configuration",
            noise_reference_reliability=noise_reference_reliability or 0.0,
        ).to_preset()
        # A bulk replacement supersedes queued gestures. Capture the accepted
        # state after cancellation so a failed replacement cannot apply them.
        self._cancel_processing_configuration_writes()
        previous = self._get_current_preset()
        previous_mode = self._processing_mode()
        previous_compressor = self.compressor_state.get_settings(
            include_calibration=True
        )
        replaying = self.__dict__.get("_history_replaying", False)
        self._history_replaying = True
        timer = self.__dict__.get("_history_timer")
        if timer is not None:
            timer.stop()
        release_mute = True
        try:
            self.set_temporary_output_mute(True, "configuration")
            if self.__dict__.get("_output_mute_error") and self.processor.is_running():
                self._stop_audio_on_worker()
                raise RuntimeError("Audio is muted and stopping because configuration mute failed")
            try:
                self._write_processing_configuration(candidate)
                if processing_mode is not None:
                    self._set_processing_mode(processing_mode)
                if noise_reference_reliability is not None or compressor_metadata:
                    self.compressor_state.set_settings({
                        **(compressor_metadata or {}),
                        "noise_reference_reliability": noise_reference_reliability or 0.0,
                    })
                if noise_reference_reliability is None and not self.__dict__.get("_calibration_dialog_open"):
                    self._history_replaying = False
                    try:
                        self._sync_calibration_evidence(force_reset=True)
                    finally:
                        self._history_replaying = True
                self._flush_processing_configuration_writes()
            except Exception as error:
                try:
                    self._cancel_processing_configuration_writes()
                    self._write_processing_configuration(previous)
                    self._set_processing_mode(previous_mode)
                    self.compressor_state.set_settings(previous_compressor)
                    self._flush_processing_configuration_writes()
                except Exception as restore_error:
                    release_mute = False
                    # Before the stop is asked for: its failure message must be the one left showing.
                    self.status_message.showMessage(
                        "Audio muted and stopping: configuration restoration failed",
                        7000,
                    )
                    self._stop_audio_on_worker()
                    raise RuntimeError(
                        f"Configuration failed ({error}); restoration failed ({restore_error}). "
                        "Audio is muted and stopping."
                    ) from restore_error
                raise RuntimeError(f"Configuration was not applied; previous sound restored: {error}") from error
        finally:
            self._history_replaying = replaying
            if release_mute:
                self.set_temporary_output_mute(False, "configuration")

    def _apply_preset(
        self,
        preset: Preset,
        preset_key: str | None = None,
        *,
        preset_path: Path | None = None,
        persist_last_used: bool = True,
    ) -> bool:
        """Apply a preset to the UI and processor.

        Args:
            preset: Preset object to apply
            preset_key: Optional key for built-in presets (e.g., "voice", "bass_cut")
        """
        if (
            self.__dict__.get("_history_ready", False)
            and not self.__dict__.get("_history_replaying", False)
            and not self._confirm_discard_changes()
        ):
            return False
        try:
            self.apply_processing_configuration(preset)
        except Exception as error:
            if self.__dict__.get("_history_ready", False) and self.__dict__.get("_login_startup_deadline") is None:
                QMessageBox.critical(self, "Preset Not Applied", str(error))
            else:
                self.config.load_warning = "\n".join(filter(None, (
                    self.config.load_warning, f"Could not restore preset: {error}"
                )))
            return False
        history_ready = bool(self.__dict__.get("_history_ready", False))
        history_replaying = bool(self.__dict__.get("_history_replaying", False))
        preset_id = f"builtin:{preset_key}" if preset_key else None
        persist_identity = (
            persist_last_used
            and not history_replaying
            and (preset_id is not None or preset_path is not None)
        )
        previous_last_preset = ""
        if persist_identity:
            previous_last_preset = self.config.last_preset
        if not history_replaying:
            self._preset_record = {}  # a load starts a record of its own; a save keeps the loaded one
            self._set_preset_identity(
                preset,
                path=preset_path,
                preset_id=preset_id,
                persist_last_used=persist_last_used,
            )
        if history_ready and not history_replaying:
            loaded = {preset_id} if preset_id is not None else set()
            if preset_path is not None:
                loaded |= {
                    str(preset_path),
                    _startup_custom_file_id(preset_path.name),
                    f"{STARTUP_CUSTOM_PREFIX}{preset.name}",
                }
            for key, failed_id in self.__dict__.get("_failed_presets", {}).items():
                if failed_id in loaded:
                    self._set_problem(key, None)
            self._commit_pending_configuration_snapshot(
                label=f"Loaded preset ({preset.name})",
                source="preset",
                provenance=self._current_value_provenance,
            )

        persisted = True
        if persist_identity:
            persisted = self._save_config_safely()
            if not persisted:
                logger.warning("Could not remember loaded preset %s", preset.name)
            if not persisted:
                self.config.last_preset = previous_last_preset
                self._mirror_preset_record()
            self._last_preset_identity_persisted = bool(persisted)

        message = f"Loaded preset: {preset.name}"
        if persist_identity and not persisted:
            message += "; applied for this session, but could not be remembered for the next launch"
        self.status_message.showMessage(message, 6000 if persist_identity and not persisted else 3000)
        return True

    def _confirm_discard_changes(self) -> bool:
        if not self.__dict__.get("_history_ready", False):
            return True
        self._set_preset_modified()
        if not self.preset_modified:
            return True
        question = "Save your sound changes before continuing?"
        if self._processing_mode() == "raw":
            question += "\nRaw Monitor is session-only and will not be restored from a preset."
        reply = QMessageBox.question(
            self, "Unsaved Sound Changes",
            question,
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_preset()
        return reply == QMessageBox.StandardButton.Discard

    def _save_preset(self) -> bool:
        """Update an owned preset; imported and built-in sounds require Save As."""
        path = self.__dict__.get("current_preset_path")
        if path is None or Path(path).resolve().parent != get_presets_dir().resolve():
            return self._save_preset_as()
        preset = self._get_saveable_preset()
        if preset is None:
            return False
        preset.name = self.current_preset_name
        preset.description = self.__dict__.get("current_preset_description", "")
        return self._save_preset_file(preset, filepath=Path(path)) is not None

    def _save_preset_as(self) -> bool:
        name, ok = QInputDialog.getText(
            self, "Save Preset As", "Preset name:",
            text=self.__dict__.get("current_preset_name", "My Preset"),
        )
        if not ok or not name.strip():
            return False
        preset = self._get_saveable_preset()
        if preset is None:
            return False
        preset.name = name.strip()
        preset.description = self.__dict__.get("current_preset_description", "")
        return self._save_preset_file(preset) is not None

    def _load_preset(self):
        """Load a preset from file."""
        presets_dir = get_presets_dir()

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            str(presets_dir),
            "JSON Files (*.json);;All Files (*.*)",
        )

        if not filepath:
            return

        try:
            requested_path = Path(filepath)
            try:
                preset = load_preset(requested_path)
                preset_path = requested_path
            except PresetValidationError:
                preset, preset_path = import_preset(requested_path)
            if not self._apply_preset(preset, preset_path=preset_path):
                return
        except PresetValidationError as e:
            # Actionable error for validation failures
            logger.warning("Preset validation failed", exc_info=True)
            QMessageBox.warning(
                self,
                "Invalid Preset",
                f"Could not load preset:\n\n{e}\n\n"
                "Please check the preset file and correct the invalid values.",
            )
        except json.JSONDecodeError as e:
            # Actionable error for malformed JSON
            logger.warning("Preset JSON decode failed", exc_info=True)
            QMessageBox.warning(
                self,
                "Invalid Preset File",
                f"The preset file is not valid JSON:\n\n"
                f"Error at line {e.lineno}: {e.msg}\n\n"
                "Please check the file format or try a different preset.",
            )
        except Exception as e:
            # Fallback for unexpected errors with actionable guidance
            logger.exception("Preset load failed")
            QMessageBox.critical(
                self,
                "Error Loading Preset",
                f"Failed to load preset:\n\n{type(e).__name__}: {e}\n\n"
                "If this problem persists, try:\n"
                "1. Check that the file exists and is readable\n"
                "2. Verify the file is a valid AudioForge preset\n"
                "3. Try loading a different preset",
            )

    def _open_presets_folder(self):
        """Open the presets folder in the file explorer."""
        presets_dir = get_presets_dir()
        if not presets_dir.is_dir() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(presets_dir))):
            QMessageBox.warning(
                self,
                "Presets Folder",
                f"Unable to open the presets folder. Check: {presets_dir}",
            )

    def closeEvent(self, event):
        """Handle window close."""
        # A Windows sign-out or shutdown waits on this close, so it never asks or refuses.
        session_ending = _session_is_ending()
        if not self._quitting and not session_ending:
            if (
                self._tray_icon is not None
                and self._tray_icon.isVisible()
                and self._close_to_tray_action is not None
                and self._close_to_tray_action.isChecked()
            ):
                choice = "tray"
            else:
                choice = self._quit_choice()
            if choice == "tray":
                self._hide_to_tray()
            if choice != "quit":
                event.ignore()
                return

        if not session_ending and not self._confirm_discard_changes():
            self._quitting = False
            self._quit_dont_ask = False
            event.ignore()
            return
        self._quitting = True
        saved = self._shutdown()
        if not saved and not session_ending:
            QMessageBox.warning(
                self,
                "Settings Not Saved",
                "Audio processing has stopped, but window settings could not be saved.",
            )
        event.accept()

    def _shutdown(self) -> bool:
        """Save the settings and stop audio; closeEvent and an app quit without a close share this."""
        if self._quit_dont_ask:
            self.config.confirm_quit_when_live = False
        self._shut_down = True  # from here the answers of engine calls still out are dropped
        engine_idle = self._engine.wait(ENGINE_CALL_WAIT_MS)
        if not engine_idle:
            logger.warning("An engine call is still out at shutdown; the exit ends the audio")
        self._unregister_mute_hotkey()
        if self._tray_icon is not None:
            self._tray_icon.hide()
        # The client area, as setGeometry() restores it, and unaffected by maximizing.
        normal = self.normalGeometry()
        geometry = {
            "x": normal.x(),
            "y": normal.y(),
            "width": normal.width(),
            "height": normal.height(),
        }
        if self.isMaximized():
            geometry["maximized"] = True
        self.config.window_geometry = geometry
        try:
            saved = self._save_ui_state()
        finally:
            # Not beside a call that is still out: this stop would wait behind it on the GUI thread.
            if engine_idle and self.processor.is_running():
                self.processor.stop()
        return saved


def run_app():
    """Run the AudioForge application."""
    return run_qt_app(MainWindow)


if __name__ == "__main__":
    sys.exit(run_app())
