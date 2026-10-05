"""
Main window for AudioForge application

Adapted from Spectral Workbench project.

DEBUG: Added terminal logging for processor state tracking
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QStatusBar,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QMenu,
    QSlider,
    QScrollArea,
    QFrame,
    QStackedWidget,
    QButtonGroup,
    QFormLayout,
    QSizePolicy,
    QSystemTrayIcon,
)
from PySide6.QtCore import QEvent, QSignalBlocker, Qt, QTimer, QRect, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon
import os
import sys
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

from .gate_panel import GatePanel
from .eq_panel import EQPanel
from .compressor_panel import CompressorPanel
from .deesser_panel import DeEsserPanel
from .level_meter import LevelMeter
from .health import RecentStreamHealth, advice_for
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
from .stream_recovery import StreamRecoveryManager
from .config_history import (
    BoundedConfigurationHistory,
    ConfigurationSnapshot,
    explicit_provenance_after_edit,
)
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    SPACING_SECTION,
    SPACING_NORMAL,
    MARGIN_PANEL,
    PRIMARY_ACTION_BUTTON_STYLE,
    DESTRUCTIVE_ACTION_BUTTON_STYLE,
    PRIMARY_LABEL_STYLE,
    SPACING_TIGHT,
    SUBDUED_TEXT_STYLE,
    WARNING_BANNER_STYLE,
    configure_responsive_combo,
    status_chip_style,
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
from .components import Card, Glyph, IconButton, NavButton, ToggleSwitch, plain_label
from .theme import CARD_STYLE, PALETTE, prefers_reduced_motion
from .desktop_integration import GlobalMuteHotkey, activate_window, build_tray_tooltip
from . import login_startup
from .. import AudioProcessor, __version__, list_input_devices, list_output_devices
from ..diagnostics_export import (
    build_diagnostics_snapshot,
    diagnostics_filename,
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

INPUT_CHANNEL_MODE_OPTIONS = (
    ("Average", "average"),
    ("Left", "left"),
    ("Right", "right"),
    ("Max RMS", "max_rms"),
    ("Phase-safe mono", "phase_safe_mono"),
)
INPUT_CLEANUP_MODE_OPTIONS = (
    ("Off", "off"),
    ("Gentle", "gentle"),
    ("Strong", "strong"),
)
INPUT_PHASE_WARNING_CORRELATION = -0.75
PROCESSING_MODE_OPTIONS = (
    ("Normal", "normal"),
    ("Bypass", "bypass"),
    ("Raw Monitor", "raw"),
)
DEFAULT_MUTE_HOTKEY = "Ctrl+Alt+M"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 850
MINIMUM_WINDOW_WIDTH = 900
MINIMUM_WINDOW_HEIGHT = 640
DROPPED_DIAGNOSTICS_TOOLTIP = (
    "Dropped samples and warning-signaling runtime counters.\n"
    "Right-click to reset dropped samples."
)


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

    COMPACT_LAYOUT_BREAKPOINT = 1200
    HEALTH_PAGE_INDEX = 1

    def __init__(self, *, login_startup: bool = False):
        super().__init__()
        self.setWindowTitle("AudioForge - Microphone Audio Processor")

        # Create audio processor
        self.processor = AudioProcessor()

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
        self._rnnoise_strength_exact = 1.0
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
        self._undo_auto_eq_button = None
        self._calibration_dialog_open = False
        self._stream_recovery = StreamRecoveryManager()
        self._stream_health = RecentStreamHealth()
        self._last_backend_warning = None
        self._last_output_underrun_total = 0
        self._last_input_clip_event_count = 0
        self._last_output_clip_event_count = 0
        self._last_output_true_peak_event_count = 0
        self._last_input_phase_warning_count = 0
        self._last_gate_chatter_event_count = 0
        self._responsive_layout_compact: bool | None = None
        self._quitting = False
        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_menu: QMenu | None = None
        self._tray_mute_action: QAction | None = None
        self._mute_hotkey: GlobalMuteHotkey | None = None
        self._close_to_tray_action: QAction | None = None
        self._mute_hotkey_action: QAction | None = None

        # Set up UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_options_menu()
        self._setup_statusbar()
        self._finish_shell()
        self._setup_desktop_integration()
        self.user_mute_checkbox.blockSignals(True)
        self.user_mute_checkbox.setChecked(self.user_muted)
        self.user_mute_checkbox.blockSignals(False)
        self._apply_output_mute()

        # Populate device lists
        self._refresh_devices()

        # Connect device change signals for persistence
        self.input_combo.currentIndexChanged.connect(self._on_device_changed)
        self.output_combo.currentIndexChanged.connect(self._on_device_changed)
        self.input_channel_mode_combo.currentIndexChanged.connect(
            self._on_input_channel_mode_changed
        )
        self.input_cleanup_mode_combo.currentIndexChanged.connect(
            self._on_input_cleanup_mode_changed
        )

        self._apply_initial_window_geometry()
        self._update_responsive_layouts(self.width())

        # Restore settings from config
        self._restore_from_config()
        self._initialize_configuration_history()
        self._connect_configuration_history_inputs()
        if self.config.load_warning:
            warning = self.config.load_warning
            self.config_warning_banner.setText(warning)
            self.config_warning_banner.setVisible(True)
            QTimer.singleShot(
                0,
                lambda: self.status_bar.showMessage(warning),
            )
        if not self._login_startup:
            QTimer.singleShot(0, self._maybe_show_first_run_setup)

        # Meter update timer (60 FPS)
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._update_meters)
        self.meter_timer.setInterval(100 if prefers_reduced_motion() else 16)
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
        fitted = _fit_window_geometry_to_screens(
            self.config.window_geometry,
            [screen.availableGeometry() for screen in ordered_screens],
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
        if self.config.window_geometry is not None:
            self.config.window_geometry = {
                "x": fitted.x(),
                "y": fitted.y(),
                "width": fitted.width(),
                "height": fitted.height(),
            }

    def _setup_ui(self):
        """Build the shell: navigation rail, top bar, pages and level meters."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root = QHBoxLayout(central_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.page_stack = QStackedWidget()
        root.addWidget(self._build_nav_rail())
        column = QVBoxLayout()
        column.setContentsMargins(
            SPACING_SECTION, MARGIN_PANEL, SPACING_SECTION, MARGIN_PANEL
        )
        column.setSpacing(MARGIN_PANEL)
        root.addLayout(column, stretch=1)

        # Warning banner for missing audio devices (hidden by default)
        self.device_warning_banner = QLabel(
            "Warning: No audio devices detected. Check your audio drivers and connections."
        )
        self.device_warning_banner.setStyleSheet(WARNING_BANNER_STYLE)
        self.device_warning_banner.setAccessibleName("Audio device warning")
        self.device_warning_banner.setWordWrap(True)
        self.device_warning_banner.setVisible(False)
        column.addWidget(self.device_warning_banner)

        self.config_warning_banner = QLabel()
        self.config_warning_banner.setStyleSheet(WARNING_BANNER_STYLE)
        self.config_warning_banner.setAccessibleName("Configuration warning")
        self.config_warning_banner.setWordWrap(True)
        self.config_warning_banner.setVisible(False)
        column.addWidget(self.config_warning_banner)

        column.addWidget(self._build_top_bar())

        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(SPACING_NORMAL)
        middle_layout.addWidget(self.page_stack, stretch=1)
        self.input_meter = LevelMeter("IN", show_scale=True)
        self.input_meter.setAccessibleName("Input level meter")
        self.input_meter.setFixedWidth(50)
        middle_layout.addWidget(self.input_meter)
        self.output_meter = LevelMeter("OUT", show_scale=True)
        self.output_meter.setAccessibleName("Output level meter")
        self.output_meter.setFixedWidth(50)
        middle_layout.addWidget(self.output_meter)
        column.addLayout(middle_layout, stretch=1)

        self.page_stack.addWidget(self._build_mic_page())
        self.health_details = self._build_health_page()
        self.page_stack.addWidget(self.health_details)
        self.settings_page = self._build_settings_page()
        self.page_stack.addWidget(self.settings_page)
        self.page_stack.currentChanged.connect(self._on_page_changed)

        self.setTabOrder(self.input_combo, self.output_combo)
        self.setTabOrder(self.output_combo, self.refresh_btn)
        self.setTabOrder(self.refresh_btn, self.processing_mode_combo)
        self.setTabOrder(self.processing_mode_combo, self.user_mute_checkbox)
        self.setTabOrder(self.user_mute_checkbox, self.start_btn)
        self.setTabOrder(self.start_btn, self.stop_btn)

    def _build_nav_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(76)
        rail.setStyleSheet(f"QFrame#rail {{ background-color: {PALETTE.rail_surface}; }}")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, MARGIN_PANEL, 0, MARGIN_PANEL)
        layout.setSpacing(SPACING_TIGHT)
        self.nav_group = QButtonGroup(self)
        for index, (glyph, label) in enumerate(
            (
                (Glyph.MICROPHONE, "Mic"),
                (Glyph.HEALTH, "Health"),
                (Glyph.SETTINGS, "Settings"),
            )
        ):
            button = NavButton(glyph, label)
            button.setChecked(index == 0)
            self.nav_group.addButton(button, index)
            layout.addWidget(button)
        self.nav_group.idClicked.connect(self.page_stack.setCurrentIndex)
        layout.addStretch(1)
        return rail

    def _on_page_changed(self, index: int) -> None:
        self.nav_group.button(index).setChecked(True)
        with QSignalBlocker(self.health_details_button):
            self.health_details_button.setChecked(index == self.HEALTH_PAGE_INDEX)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("card")
        bar.setStyleSheet(CARD_STYLE)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(MARGIN_PANEL, SPACING_NORMAL, MARGIN_PANEL, SPACING_NORMAL)
        layout.setSpacing(SPACING_NORMAL)

        input_label = QLabel("Input")
        self.input_combo = QComboBox()
        configure_responsive_combo(self.input_combo, minimum_chars=6)
        bind_label(input_label, self.input_combo, name="Input audio device")

        output_label = QLabel("Output")
        self.output_combo = QComboBox()
        configure_responsive_combo(self.output_combo, minimum_chars=6)
        bind_label(output_label, self.output_combo, name="Output audio device")

        self.refresh_btn = IconButton(Glyph.REFRESH, "Refresh audio devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)

        self.processing_mode_combo = QComboBox()
        for label, mode in PROCESSING_MODE_OPTIONS:
            self.processing_mode_combo.addItem(label, mode)
        self.processing_mode_combo.setToolTip(
            "Normal applies the voice chain. Bypass keeps input conditioning and output protection. "
            "Raw Monitor skips input filtering and the voice chain for diagnostics; it is session-only "
            "and is not stored in presets."
        )
        self.processing_mode_combo.setAccessibleName("Processing mode")
        self.processing_mode_combo.currentIndexChanged.connect(
            self._on_processing_mode_changed
        )

        self.user_mute_checkbox = ToggleSwitch("Mute")
        self.user_mute_checkbox.setAccessibleName("Mute output")
        self.user_mute_checkbox.setToolTip(
            "Keep transmission muted until you turn this off. "
            "Calibration and stream recovery use a separate temporary mute."
        )
        self.user_mute_checkbox.toggled.connect(self._on_user_mute_toggled)

        # One transport control is shown at a time; _sync_transport_buttons
        # follows the enabled state the processing code already maintains.
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet(DESTRUCTIVE_ACTION_BUTTON_STYLE)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.installEventFilter(self)

        self._route_labels = (input_label, output_label)
        layout.addWidget(input_label)
        layout.addWidget(self.input_combo, stretch=1)
        layout.addWidget(output_label)
        layout.addWidget(self.output_combo, stretch=1)
        layout.addWidget(self.refresh_btn)
        layout.addSpacing(SPACING_SECTION)
        layout.addWidget(self.processing_mode_combo)
        layout.addWidget(self.user_mute_checkbox)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self._sync_transport_buttons()
        return bar

    def _sync_transport_buttons(self) -> None:
        running = self.stop_btn.isEnabled()
        self.stop_btn.setVisible(running)
        self.start_btn.setVisible(not running)

    def eventFilter(self, watched, event):
        if (
            watched is self.__dict__.get("stop_btn")
            and event.type() == QEvent.Type.EnabledChange
        ):
            self._sync_transport_buttons()
        return super().eventFilter(watched, event)

    @staticmethod
    def _create_page() -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        scroll.setWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, SPACING_NORMAL, 0)
        layout.setSpacing(MARGIN_PANEL)
        return scroll, layout

    def _build_mic_page(self) -> QScrollArea:
        self.content_scroll_area, layout = self._create_page()

        self.preset_status_label = QLabel("Preset: Default (saved)")
        self.preset_status_label.setAccessibleName("Active preset and saved state")
        self.preset_status_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        self.presets_button = QPushButton("Presets")
        self.presets_button.setAccessibleName("Preset actions")

        self._undo_auto_eq_button = QPushButton("Undo")
        self._undo_auto_eq_button.setEnabled(False)
        self._undo_auto_eq_button.setToolTip(
            "Undo the most recent processing-configuration edit (Ctrl+Z)"
        )
        self._undo_auto_eq_button.clicked.connect(self.undo_configuration)

        self.auto_eq_button = QPushButton("Auto-EQ")
        self.auto_eq_button.setToolTip(
            "Automatically calibrate EQ to your voice and microphone\n"
            "Choose a tone preset, read a short passage, then review the result"
        )
        self.auto_eq_button.clicked.connect(self._on_auto_eq_clicked)

        self.test_sound_button = QPushButton("Test my sound")
        self.test_sound_button.setToolTip(
            "Record five seconds, then play back the raw recording and the "
            "result of the current settings. Output is muted while you listen."
        )
        self.test_sound_button.clicked.connect(self._on_test_my_sound_clicked)

        self.auto_voice_setup_button = QPushButton("Auto Voice Setup")
        self.auto_voice_setup_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.auto_voice_setup_button.setToolTip(
            "Record room noise and speech, then calibrate EQ, gate/VAD,\n"
            "de-esser, and compressor in one pass"
        )
        self.auto_voice_setup_button.clicked.connect(self._on_auto_voice_setup_clicked)

        actions = QHBoxLayout()
        actions.setSpacing(SPACING_NORMAL)
        # The label gives way first when the window is narrow.
        self.preset_status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        actions.addWidget(self.preset_status_label, stretch=1)
        actions.addWidget(self.presets_button)
        actions.addWidget(self._undo_auto_eq_button)
        actions.addWidget(self.test_sound_button)
        actions.addWidget(self.auto_eq_button)
        actions.addWidget(self.auto_voice_setup_button)
        layout.addLayout(actions)

        self.gate_panel = GatePanel(self.processor)
        self.deesser_panel = DeEsserPanel(self.processor)
        self.compressor_panel = CompressorPanel(self.processor)
        self.noise_suppression_group = self._create_noise_suppression_group()
        self.eq_panel = EQPanel(self.processor)
        layout.addWidget(self.eq_panel)

        self.cards_layout = QGridLayout()
        self.cards_layout.setSpacing(MARGIN_PANEL)
        self._card_widgets = (
            self.noise_suppression_group,
            self.gate_panel,
            self.deesser_panel,
            self.compressor_panel,
        )
        layout.addLayout(self.cards_layout)
        layout.addStretch(1)
        return self.content_scroll_area

    def _build_health_page(self) -> QScrollArea:
        scroll, details_layout = self._create_page()

        self.route_status_label = QLabel("Route: --")
        self.transmission_status_label = QLabel("Transmission: Stopped")
        self.health_summary_label = QLabel("Health: --")
        self.calibration_status_label = QLabel("No saved calibration")
        for label, name in (
            (self.route_status_label, "Current audio route"),
            (self.transmission_status_label, "Audio transmission state"),
            (self.health_summary_label, "Compact audio health summary"),
            (self.calibration_status_label, "Saved calibration validity"),
        ):
            label.setAccessibleName(name)
            label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.route_status_label.setWordWrap(True)
        self.health_advice_label = QLabel("Start processing to see audio health.")
        self.health_advice_label.setAccessibleName("Audio health advice")
        self.health_advice_label.setStyleSheet(PRIMARY_LABEL_STYLE)
        self.health_advice_label.setWordWrap(True)
        details_layout.addWidget(self.health_advice_label)
        details_layout.addWidget(self.route_status_label)

        # Lives in the status bar; checked while this page is shown.
        self.health_details_button = QPushButton("Details")
        self.health_details_button.setCheckable(True)
        self.health_details_button.setAccessibleName("Show audio health details")
        self.health_details_button.toggled.connect(
            lambda shown: self.page_stack.setCurrentIndex(
                self.HEALTH_PAGE_INDEX if shown else 0
            )
        )

        self.health_decision_layout = QGridLayout()
        self.health_decision_layout.setSpacing(SPACING_NORMAL)

        self.input_health_label = QLabel("Input: --")
        self.input_health_label.setToolTip(
            "Input level decision from the current meter and clipping counter."
        )

        self.output_health_label = QLabel("Output: --")
        self.output_health_label.setToolTip(
            "Final output protection state. Warns on recent output clipping."
        )

        self.gate_health_label = QLabel("Gate: --")
        self.gate_health_label.setToolTip(
            "Gate stability state. Warns when rapid open/close chatter is detected."
        )

        self.backend_diag_label = QLabel("Backend: --")
        self.backend_diag_label.setToolTip(
            "Active suppression backend state and fallback health."
        )

        self.callback_health_label = QLabel("Callbacks: --")
        self.callback_health_label.setToolTip(
            "Input/output callback heartbeat age. Warns when callbacks look stale."
        )

        self.underrun_health_label = QLabel("Underruns: --")
        self.underrun_health_label.setToolTip(
            "Output underrun health. Warns on recent or consecutive underruns."
        )
        self._health_decision_widgets = (
            self.input_health_label,
            self.output_health_label,
            self.gate_health_label,
            self.backend_diag_label,
            self.callback_health_label,
            self.underrun_health_label,
        )
        details_layout.addLayout(self.health_decision_layout)

        self.health_layout = QGridLayout()
        self.health_layout.setSpacing(SPACING_NORMAL)

        self.latency_label = QLabel("Latency: --")
        self.latency_label.setToolTip(
            "Total processing latency and smoothed DSP time per processing chunk."
        )

        self.buffer_label = QLabel("Buffer: --")
        self.buffer_label.setToolTip(
            "Input plus suppression buffer health.\nOK is healthy, WARN indicates buildup, BAD indicates heavy backlog."
        )

        self.dropped_label = QLabel("Drops: --")
        self.dropped_label.setToolTip(DROPPED_DIAGNOSTICS_TOOLTIP)
        self.dropped_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dropped_label.customContextMenuRequested.connect(
            self._on_dropped_context_menu
        )

        self.recovery_diag_label = QLabel("Recovery: --")
        self.recovery_diag_label.setToolTip(
            "Stream restarts and true output recovery events.\n"
            "Normal drift-retime adjustments are informational and do not warn."
        )
        self._health_layout_widgets = (
            self.latency_label,
            self.buffer_label,
            self.dropped_label,
            self.recovery_diag_label,
        )
        details_layout.addLayout(self.health_layout)
        details_layout.addStretch(1)

        self._reset_health_labels()
        for label, name in (
            (self.input_health_label, "Input health"),
            (self.output_health_label, "Output health"),
            (self.gate_health_label, "Gate health"),
            (self.backend_diag_label, "Noise suppression backend health"),
            (self.callback_health_label, "Audio callback health"),
            (self.underrun_health_label, "Output underrun health"),
            (self.latency_label, "Processing latency"),
            (self.buffer_label, "Audio buffer health"),
            (self.dropped_label, "Dropped audio samples"),
            (self.recovery_diag_label, "Stream recovery health"),
        ):
            label.setAccessibleName(name)
            label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum,
            )
        return scroll

    def _build_settings_page(self) -> QScrollArea:
        scroll, self.settings_layout = self._create_page()

        input_card = Card("Audio input")
        form = QFormLayout()
        form.setSpacing(SPACING_NORMAL)
        self.input_channel_mode_combo = QComboBox()
        for label, mode in INPUT_CHANNEL_MODE_OPTIONS:
            self.input_channel_mode_combo.addItem(label, mode)
        self.input_channel_mode_combo.setToolTip(
            "How multichannel input is converted to mono. Use Left/Right or Phase-safe mono if stereo channels cancel."
        )
        input_mode_label = QLabel("Input mode")
        bind_label(
            input_mode_label,
            self.input_channel_mode_combo,
            name="Input channel mode",
        )
        form.addRow(input_mode_label, self.input_channel_mode_combo)

        self.input_cleanup_mode_combo = QComboBox()
        for label, mode in INPUT_CLEANUP_MODE_OPTIONS:
            self.input_cleanup_mode_combo.addItem(label, mode)
        self.input_cleanup_mode_combo.setToolTip(
            "Optional adaptive input cleanup after the fixed safe pre-filter. Off preserves the existing DC/80 Hz path."
        )
        cleanup_label = QLabel("Cleanup")
        bind_label(
            cleanup_label,
            self.input_cleanup_mode_combo,
            name="Input cleanup mode",
        )
        form.addRow(cleanup_label, self.input_cleanup_mode_combo)
        input_card.body.addLayout(form)
        self.settings_layout.addWidget(input_card)
        self.settings_layout.addStretch(1)
        return scroll

    def _menu_card(self, title: str, menu: QMenu, *, skip: QMenu | None = None) -> Card:
        """Show a menu's entries as rows: switches, buttons and submenu buttons."""

        card = Card(title)
        # QAction.menu() hands PySide a wrapper that can delete the menu, so
        # submenus are looked up from the parent instead.
        submenus = menu.findChildren(
            QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly
        )
        for action in menu.actions():
            submenu = next((m for m in submenus if m.menuAction() is action), None)
            if action.isSeparator() or (skip is not None and submenu is skip):
                continue
            row: QPushButton | ToggleSwitch
            if isinstance(submenu, QMenu):
                row = QPushButton()
                row.setMenu(submenu)
            elif action.isCheckable():
                row = ToggleSwitch()
                row.clicked.connect(action.trigger)
            else:
                row = QPushButton()
                row.clicked.connect(action.trigger)
            if isinstance(row, QPushButton):
                # No wider than its label, but allowed to shrink with the page.
                row.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

            def sync(row=row, action=action) -> None:
                row.setText(plain_label(action.text()))
                row.setToolTip(action.toolTip())
                row.setEnabled(action.isEnabled())
                row.setVisible(action.isVisible())
                if isinstance(row, ToggleSwitch):
                    row.setChecked(action.isChecked())
                else:
                    row.setMaximumWidth(row.sizeHint().width())

            action.changed.connect(sync)
            sync()
            card.body.addWidget(row)
        return card

    def _finish_shell(self) -> None:
        """Wire what needs the menus and status bar, which are built after the pages."""

        menubar = self.menuBar()
        menus: dict[str, QMenu] = {}
        for menu in menubar.findChildren(
            QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            menus[menu.title().replace("&", "")] = menu
            # A hidden menu bar disables its shortcuts unless the window owns
            # the actions too.
            self.addActions(menu.actions())
        menubar.hide()

        self.presets_button.setMenu(menus["Presets"])
        end = self.settings_layout.count() - 1
        # The tray submenu is mostly switches, so it gets a card of its own.
        tray_menu = next(
            menu
            for menu in menus["Options"].findChildren(
                QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly
            )
            if menu.title().replace("&", "") == "Tray Background"
        )
        for offset, card in enumerate(
            (
                self._menu_card("Options", menus["Options"], skip=tray_menu),
                self._menu_card("Tray and background", tray_menu),
                self._menu_card("Help", menus["Help"]),
            )
        ):
            self.settings_layout.insertWidget(end + offset, card)

        for widget in (
            self.transmission_status_label,
            self.calibration_status_label,
            self.health_summary_label,
            self.health_details_button,
        ):
            self.status_bar.addPermanentWidget(widget)

        # The Qt Quick view of the same controls. AUDIOFORGE_QML=0 keeps the
        # widget view, which is also what remains if the scene cannot load.
        if os.environ.get("AUDIOFORGE_QML", "1") != "0":
            from .quick_shell import install_quick_shell

            install_quick_shell(self)

    @staticmethod
    def _remove_grid_widgets(layout: QGridLayout, widgets: tuple[QWidget, ...]) -> None:
        for widget in widgets:
            layout.removeWidget(widget)
        for column in range(10):
            layout.setColumnStretch(column, 0)
        for row in range(10):
            layout.setRowStretch(row, 0)

    def _update_responsive_layouts(self, width: int) -> None:
        if not hasattr(self, "_card_widgets"):
            return
        compact = width < self.COMPACT_LAYOUT_BREAKPOINT
        if compact == self._responsive_layout_compact:
            return
        self._responsive_layout_compact = compact
        # The pickers keep their accessible names when the captions go.
        for label in self._route_labels:
            label.setVisible(not compact)
        self._remove_grid_widgets(self.cards_layout, self._card_widgets)
        columns = 1 if compact else 2
        for index, widget in enumerate(self._card_widgets):
            row, column = divmod(index, columns)
            self.cards_layout.addWidget(widget, row, column, Qt.AlignmentFlag.AlignTop)
            self.cards_layout.setColumnStretch(column, 1)
        self._layout_health_chips(compact)

    def _layout_health_chips(self, compact: bool) -> None:
        decision_widgets = self._health_decision_widgets
        self._remove_grid_widgets(self.health_decision_layout, decision_widgets)
        decision_columns = 3 if compact else len(decision_widgets)
        for index, label in enumerate(decision_widgets):
            label.setWordWrap(compact)
            label.setSizePolicy(
                QSizePolicy.Policy.Expanding
                if compact
                else QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Minimum,
            )
            row, column = divmod(index, decision_columns)
            self.health_decision_layout.addWidget(label, row, column)
            if compact:
                self.health_decision_layout.setColumnStretch(column, 1)
        if not compact:
            self.health_decision_layout.setColumnStretch(len(decision_widgets), 1)

        health_widgets = self._health_layout_widgets
        self._remove_grid_widgets(self.health_layout, health_widgets)
        latency_label, buffer_label, dropped_label, recovery_label = health_widgets
        for label in health_widgets:
            label.setWordWrap(compact)
            label.setSizePolicy(
                QSizePolicy.Policy.Expanding
                if compact
                else QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Minimum,
            )
        if compact:
            self.health_layout.addWidget(latency_label, 0, 0)
            self.health_layout.addWidget(buffer_label, 0, 1)
            self.health_layout.addWidget(recovery_label, 0, 2)
            self.health_layout.addWidget(dropped_label, 1, 0, 1, 3)
            for column in range(3):
                self.health_layout.setColumnStretch(column, 1)
            return

        self.health_layout.addWidget(latency_label, 0, 0)
        self.health_layout.addWidget(buffer_label, 0, 1)
        self.health_layout.addWidget(dropped_label, 0, 2)
        self.health_layout.addWidget(recovery_label, 0, 3)
        self.health_layout.setColumnStretch(4, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layouts(event.size().width())

    def _create_noise_suppression_group(self) -> Card:
        self.rnnoise_checkbox = ToggleSwitch()
        self.rnnoise_checkbox.setChecked(True)
        self.rnnoise_checkbox.setToolTip(
            "Enable or disable the selected suppression backend."
        )
        self.rnnoise_checkbox.toggled.connect(self._on_rnnoise_toggled)
        group = Card(
            "Noise Suppression",
            switch=self.rnnoise_checkbox,
            help_text=(
                "Removes steady background noise such as fans and hum. The "
                "backend choice affects cleanup quality, CPU use and latency. "
                "Packaged builds use bundled, integrity-checked model files."
            ),
        )
        layout = group.body

        model_layout = QHBoxLayout()
        backend_label = QLabel("Backend:")
        model_layout.addWidget(backend_label)
        self.model_combo = QComboBox()
        for model_id, display_name in self.processor.list_noise_models():
            self.model_combo.addItem(display_name, model_id)
        self.model_combo.setToolTip(
            "Choose the suppression backend.\n"
            "RNNoise: low latency baseline.\n"
            "DeepFilterNet LL: low latency with stronger cleanup.\n"
            "DeepFilterNet: stronger cleanup at about 30 ms."
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        configure_responsive_combo(self.model_combo)
        model_layout.addWidget(self.model_combo, stretch=1)
        bind_label(
            backend_label,
            self.model_combo,
            name="Noise suppression backend",
        )
        layout.addLayout(model_layout)

        strength_layout = QHBoxLayout()
        strength_label = QLabel("Strength:")
        strength_layout.addWidget(strength_label)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.setToolTip(
            "Processing strength for the selected backend (0% dry, 100% fully processed)."
        )
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        strength_layout.addWidget(self.strength_slider)
        bind_label(
            strength_label,
            self.strength_slider,
            name="Noise suppression strength",
        )

        self.strength_label = QLabel("100%")
        self.strength_label.setMinimumWidth(48)
        strength_layout.addWidget(self.strength_label)
        layout.addLayout(strength_layout)

        self.rnnoise_latency_label = QLabel("Latency: ~10ms (RNNoise)")
        self.rnnoise_latency_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.rnnoise_latency_label.setWordWrap(True)
        layout.addWidget(self.rnnoise_latency_label)

        set_accessible_group(
            (
                (
                    self.rnnoise_checkbox,
                    "Enable noise suppression",
                    self.rnnoise_checkbox.toolTip(),
                ),
                (
                    self.strength_slider,
                    "Noise suppression strength",
                    self.strength_slider.toolTip(),
                ),
            )
        )
        return group

    def _set_health_chip(self, label: QLabel, text: str, state: str) -> None:
        if label.text() != text:
            label.setText(text)
        if label.property("health_state") != state:
            label.setStyleSheet(status_chip_style(state))
            label.setProperty("health_state", state)

    def _reset_health_labels(self) -> None:
        startup_message = self.__dict__.get("_login_startup_message", "")
        self._set_health_chip(
            self.health_summary_label, startup_message or "Health: --",
            "warning" if startup_message else "idle",
        )
        self._set_health_chip(self.input_health_label, "Input: --", "idle")
        self._set_health_chip(self.output_health_label, "Output: --", "idle")
        self._set_health_chip(self.gate_health_label, "Gate: --", "idle")
        self._set_health_chip(self.callback_health_label, "Callbacks: --", "idle")
        self._set_health_chip(self.underrun_health_label, "Underruns: --", "idle")
        self._set_health_chip(self.latency_label, "Latency: --", "idle")
        self._set_health_chip(self.buffer_label, "Buffer: --", "idle")
        self._set_health_chip(self.dropped_label, "Drops: --", "idle")
        self._set_health_chip(self.backend_diag_label, "Backend: --", "idle")
        self._set_health_chip(self.recovery_diag_label, "Recovery: --", "idle")
        if self.dropped_label.toolTip() != DROPPED_DIAGNOSTICS_TOOLTIP:
            self.dropped_label.setToolTip(DROPPED_DIAGNOSTICS_TOOLTIP)
        if self.dropped_label.accessibleDescription():
            self.dropped_label.setAccessibleDescription("")

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
        index = self.input_channel_mode_combo.findData(target)
        self.input_channel_mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def _apply_input_channel_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_channel_mode(mode) else "phase_safe_mono"
        try:
            if hasattr(self.processor, "set_input_channel_mode"):
                self.processor.set_input_channel_mode(target)
        except Exception:
            logger.debug("Failed to apply input channel mode", exc_info=True)

    @staticmethod
    def _is_valid_input_cleanup_mode(mode: object) -> bool:
        return isinstance(mode, str) and any(
            mode == option_mode for _label, option_mode in INPUT_CLEANUP_MODE_OPTIONS
        )

    def _select_input_cleanup_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_cleanup_mode(mode) else "off"
        index = self.input_cleanup_mode_combo.findData(target)
        self.input_cleanup_mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def _apply_input_cleanup_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_cleanup_mode(mode) else "off"
        try:
            if hasattr(self.processor, "set_input_cleanup_mode"):
                self.processor.set_input_cleanup_mode(target)
        except Exception:
            logger.debug("Failed to apply input cleanup mode", exc_info=True)

    @staticmethod
    def _is_valid_processing_mode(mode: object) -> bool:
        return isinstance(mode, str) and any(
            mode == option_mode for _label, option_mode in PROCESSING_MODE_OPTIONS
        )

    def _processing_mode(self) -> str:
        """Return the selected mode, or its last applied value during initialization."""
        combo = self.__dict__.get("processing_mode_combo")
        if combo is not None:
            mode = combo.currentData()
            if self._is_valid_processing_mode(mode):
                return str(mode)
        return self.__dict__.get("_applied_processing_mode", "normal")

    def _set_processing_mode(self, mode: str, *, notify: bool = False) -> None:
        target = mode if self._is_valid_processing_mode(mode) else "normal"
        combo = self.__dict__.get("processing_mode_combo")
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
            self.status_bar.showMessage(message)
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
                self.status_bar.showMessage(f"{error}; output muted because restoration failed")
                return
            self.status_bar.showMessage(f"{error}; previous mode restored", 6000)
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
            logger.debug("Failed to apply output mute state", exc_info=True)
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
        self.status_bar.showMessage(message, 6000 if not saved else 3000)

    def _update_session_summary(self) -> None:
        """Keep the compact route/preset/transmission summary current."""
        if self.__dict__.get("route_status_label") is None:
            return
        input_name = self._device_name_from_identity(
            self._combo_device_identity(self.input_combo)
        ) or "Default input"
        output_name = self._device_name_from_identity(
            self._combo_device_identity(self.output_combo)
        ) or "Select a destination"
        self.route_status_label.setText(f"Route: {input_name} -> {output_name}")
        state = "unsaved changes" if self.preset_modified else "saved"
        self.preset_status_label.setText(
            f"Preset: {self.current_preset_name or 'Default'} ({state})"
        )
        running = bool(getattr(self, "processor", None) and self.processor.is_running())
        mode = self._processing_mode()
        mode_label = next(label for label, value in PROCESSING_MODE_OPTIONS if value == mode)
        transmission = f"{'Running' if running else 'Stopped'} / {mode_label}"
        if self.__dict__.get("_output_mute_error"):
            transmission += " / Mute pending"
        elif self.__dict__.get("user_muted", False) or self.__dict__.get(
            "_temporary_mute_reasons", set()
        ):
            transmission += " / Muted"
        self.transmission_status_label.setText(f"Transmission: {transmission}")
        tray = self.__dict__.get("_tray_icon")
        if tray is not None:
            tray.setToolTip(build_tray_tooltip(
                processing=running,
                muted=bool(self.user_muted or self._temporary_mute_reasons),
                health=self.health_summary_label.text().removeprefix("Health: "),
                route=f"{input_name} -> {output_name}",
            ))
        if self.__dict__.get("_history_ready") and not self.__dict__.get("_calibration_dialog_open"):
            self._refresh_calibration_status()

    def _refresh_calibration_status(self) -> None:
        from .calibration_history import calibration_summary

        self.calibration_status_label.setText(calibration_summary(self))

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
        if not hasattr(self, "config") or not hasattr(self, "compressor_panel"):
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
        current = self.compressor_panel.get_compressor_settings(
            include_calibration=True
        )
        current_reliability = current["noise_reference_reliability"]
        if current_reliability != reliability:
            self.compressor_panel.set_compressor_settings(
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
        self._set_preset_modified()

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

    @property
    def current_preset_modified(self) -> bool:
        return bool(self.__dict__.get("preset_modified", False))

    def _save_config_safely(self) -> bool:
        """Persist config without letting a write failure disrupt the UI."""
        try:
            return bool(save_config(self.config))
        except Exception:
            logger.warning("Could not save AudioForge configuration", exc_info=True)
            return False

    def _save_ui_state(self) -> bool:
        saved = self._save_config_safely()
        if not saved:
            self.status_bar.showMessage("Could not save window settings", 5000)
        return saved

    def _set_noise_suppression_latency_label(self, model_id: str) -> None:
        if model_id == "deepfilter":
            self.rnnoise_latency_label.setText("Latency: ~30ms (DeepFilterNet)")
        elif model_id == "deepfilter-ll":
            self.rnnoise_latency_label.setText("Latency: ~10ms (DeepFilterNet LL)")
        else:
            self.rnnoise_latency_label.setText("Latency: ~10ms (RNNoise)")

    @staticmethod
    def _combo_device_identity(combo: QComboBox) -> DeviceIdentity | None:
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
        combo: QComboBox, identity: DeviceIdentity | None
    ) -> int:
        return find_identity_index(MainWindow._combo_identities(combo), identity)

    def _select_combo_identity(
        self,
        combo: QComboBox,
        identity: DeviceIdentity | None,
    ) -> bool:
        index = self._find_combo_index_by_identity(combo, identity)
        if index >= 0:
            combo.setCurrentIndex(index)
            return True
        return False

    @staticmethod
    def _default_combo_index(combo: QComboBox) -> int:
        return default_device_index(MainWindow._combo_identities(combo))

    @staticmethod
    def _preferred_output_combo_index(combo: QComboBox) -> int:
        return preferred_output_index(MainWindow._combo_identities(combo))

    @staticmethod
    def _combo_identities(combo: QComboBox) -> list[DeviceIdentity | None]:
        return [coerce_device_identity(combo.itemData(i)) for i in range(combo.count())]

    def _device_selection_to_name(self, combo: QComboBox) -> str:
        identity = self._combo_device_identity(combo)
        if identity is not None:
            return identity.name
        value = combo.currentData()
        if isinstance(value, str):
            return value
        return ""

    def _current_input_preference_key(self) -> str | None:
        identity = self._combo_device_identity(self.input_combo)
        if identity is None or not identity_is_persistable(
            self._combo_identities(self.input_combo), identity
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
            self.status_bar.showMessage("Route input settings could not be saved", 6000)
            return False
        self._apply_input_preferences_for_current_route()
        return True

    def _save_current_route_input_preference(self, _checked: bool = False) -> bool:
        """Save the visible input controls as an exact-route override."""
        channel_mode = self.input_channel_mode_combo.currentData()
        cleanup_mode = self.input_cleanup_mode_combo.currentData()
        if not (
            isinstance(channel_mode, str) and isinstance(cleanup_mode, str)
        ):
            return False
        if self._current_device_route_key() is None:
            self.status_bar.showMessage(
                "Connect devices with stable endpoint IDs before saving route input settings",
                5000,
            )
            return False
        if self._set_route_input_preference(
            channel_mode=channel_mode,
            cleanup_mode=cleanup_mode,
        ):
            self.status_bar.showMessage(
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
        start_action.triggered.connect(self._start_processing)
        file_menu.addAction(start_action)

        stop_action = QAction("S&top Processing", self)
        stop_action.setShortcut("Ctrl+.")
        stop_action.triggered.connect(self._stop_processing)
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

    def _setup_statusbar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            f"Sample Rate: {self.processor.sample_rate()} Hz | Status: Ready"
        )

    def _setup_options_menu(self):
        """Setup Options menu with startup preset selector."""
        menubar = self.menuBar()
        assert menubar is not None

        # Options menu
        options_menu = menubar.addMenu("&Options")
        assert options_menu is not None

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
        self._last_used_action = last_used_action  # Store for updating checked state

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
            self.status_bar.showMessage(str(error), 8000)
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
                self.status_bar.showMessage(message)
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
            find_identity_index(self._combo_identities(self.input_combo), saved_input),
            find_identity_index(self._combo_identities(self.output_combo), saved_output),
        )
        if min(indices) < 0:
            self._login_startup_message = "Login waiting for saved audio endpoints"
            self.status_bar.showMessage(self._login_startup_message)
            self.stop_btn.setEnabled(True)
            return
        for combo, index in zip((self.input_combo, self.output_combo), indices):
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
            app.aboutToQuit.connect(self._mark_quitting)

        configured_hotkey = str(getattr(self.config, "mute_hotkey", "") or "")
        if configured_hotkey and not tray_only:
            self._register_mute_hotkey(configured_hotkey)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            if self._close_to_tray_action is not None:
                self._close_to_tray_action.setEnabled(False)
                self._close_to_tray_action.blockSignals(True)
                self._close_to_tray_action.setChecked(False)
                self._close_to_tray_action.blockSignals(False)
            return

        if self._close_to_tray_action is not None:
            self._close_to_tray_action.setEnabled(True)
            self._close_to_tray_action.blockSignals(True)
            self._close_to_tray_action.setChecked(bool(self.config.close_to_tray))
            self._close_to_tray_action.blockSignals(False)

        icon = self.windowIcon()
        if icon.isNull() and isinstance(app, QGuiApplication):
            icon = app.windowIcon()
        if icon.isNull():
            icon = QIcon.fromTheme("audio-card")
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

    def _mark_quitting(self) -> None:
        self._quitting = True

    def _show_from_tray(self) -> None:
        activate_window(self)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_from_tray()

    def _on_tray_mute_toggled(self, checked: bool) -> None:
        if self.user_mute_checkbox.isChecked() != bool(checked):
            self.user_mute_checkbox.setChecked(bool(checked))

    def _on_close_to_tray_toggled(self, checked: bool) -> None:
        setattr(self.config, "close_to_tray", bool(checked))
        saved = self._save_config_safely()
        message = "Close-to-tray enabled" if checked else "Close-to-tray disabled"
        if not saved:
            message += "; preference could not be saved"
        self.status_bar.showMessage(message, 5000 if not saved else 3000)

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
        self.status_bar.showMessage(message, 5000 if not saved else 3000)

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
            self.status_bar.showMessage(
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
        self.user_mute_checkbox.setChecked(not self.user_mute_checkbox.isChecked())

    def _quit_from_tray(self) -> None:
        self._quitting = True
        if self.close():
            app = QGuiApplication.instance()
            if app is not None:
                app.quit()

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
            self.status_bar.showMessage(
                "Startup preset could not be saved; previous selection kept", 6000
            )
            self._update_startup_preset_menu(previous)
            return

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
        self.status_bar.showMessage(message, 5000)

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
        if self.config.first_run_setup_state in {"not_started", "in_progress"}:
            self._show_first_run_setup(restart_completed=False)

    def _show_first_run_setup(
        self, _checked: bool = False, *, restart_completed: bool = True
    ) -> None:
        dialog = FirstRunSetupDialog(self, restart_completed=restart_completed)
        dialog.exec()

    def _current_device_route_key(self) -> str | None:
        input_identity = self._combo_device_identity(self.input_combo)
        output_identity = self._combo_device_identity(self.output_combo)
        if input_identity is None or output_identity is None:
            return None
        if not identity_is_persistable(
            self._combo_identities(self.input_combo), input_identity
        ) or not identity_is_persistable(
            self._combo_identities(self.output_combo), output_identity
        ):
            return None
        return build_device_route_key(input_identity, output_identity)

    def _current_capture_format_context(self) -> tuple[int | None, ...]:
        """Return mutable endpoint format fields used by calibration evidence."""
        input_identity = self._combo_device_identity(self.input_combo)
        output_identity = self._combo_device_identity(self.output_combo)
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
            self.status_bar.showMessage(
                f"Automatic route presets {state} for this session, "
                "but the preference could not be saved",
                6000,
            )

    def _bind_current_route_preset(self, preset_id: str) -> None:
        route_key = self._current_device_route_key()
        if route_key is None:
            self.status_bar.showMessage(
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
            self.status_bar.showMessage(
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
        self.status_bar.showMessage(
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
                self.status_bar.showMessage(
                    "Preset binding could not be cleared because the change could not be saved",
                    6000,
                )
                return
        self._update_device_preset_menu()
        self.status_bar.showMessage("Cleared the preset binding for this route", 4000)

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
            self.status_bar.showMessage(
                f"Route preset: {preset_name}",
                5000,
            )
        else:
            self.status_bar.showMessage(
                "The preset bound to this route is unavailable; existing settings were kept",
                6000,
            )
        return loaded

    def _latency_profile_key(self) -> str | None:
        return self._current_device_route_key()

    def _legacy_latency_profile_key(self) -> str:
        input_name = self._device_name_from_identity(
            self._combo_device_identity(self.input_combo)
        )
        output_name = self._device_name_from_identity(
            self._combo_device_identity(self.output_combo)
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
                self.status_bar.showMessage("Updated latency profile could not be saved", 6000)
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
        self.status_bar.showMessage(message, 4000 if saved else 6000)

    def _on_latency_calibration_clicked(self) -> bool:
        if self._latency_profile_key() is None:
            self.status_bar.showMessage(
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
            return dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            self._calibration_dialog_open = False

    def _on_latency_calibration_saved(self, profile_data: dict) -> bool:
        previous = dict(self.config.latency_calibration_profiles)
        try:
            profile = LatencyCalibrationProfile.from_dict(profile_data)
            self._sync_latency_profile_for_current_devices(profile)
        except ValueError as error:
            self.status_bar.showMessage(str(error), 6000)
            return False
        saved = self._save_config_safely()
        if not saved:
            self.config.latency_calibration_profiles = previous
            self.status_bar.showMessage("Latency calibration could not be saved; retry saving", 6000)
            return False
        self._apply_latency_compensation_for_current_devices()
        route_latency_ms = float(profile.route_latency_ms)
        if route_latency_ms <= 0.0:
            route_latency_ms = float(profile.measured_round_trip_ms)
        self.status_bar.showMessage(
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
                self.status_bar.showMessage("Latency calibration reset could not be saved", 6000)
                return False
        self._apply_latency_compensation_for_current_devices()
        self.status_bar.showMessage(
            "Latency calibration reset for current device pair", 4000
        )
        return True

    def _refresh_devices(self):
        """Refresh the device lists."""
        if self.processor.is_running():
            self.refresh_btn.setEnabled(False)
            self.status_bar.showMessage("Stop processing before refreshing audio devices", 5000)
            return
        previous_route = self._current_device_route_key()
        previous_capture_format = self._current_capture_format_context()
        previous_input = (
            self.config.last_input_device_identity
            or self._combo_device_identity(self.input_combo)
        )
        previous_output = (
            self.config.last_output_device_identity
            or self._combo_device_identity(self.output_combo)
        )

        # Block signals to prevent spurious config saves during refresh
        signal_widgets = [self.input_combo, self.output_combo]
        if "input_channel_mode_combo" in self.__dict__:
            signal_widgets.append(self.input_channel_mode_combo)
        if "input_cleanup_mode_combo" in self.__dict__:
            signal_widgets.append(self.input_cleanup_mode_combo)
        signal_states = [widget.blockSignals(True) for widget in signal_widgets]

        try:
            self.input_combo.clear()
            self.output_combo.clear()

            input_found = False
            output_found = False
            config_dirty = False

            # Get input devices
            try:
                input_devices = list_input_devices()
                input_found = len(input_devices) > 0
                for device in input_devices:
                    identity = self._identity_from_device_info(device, "input")
                    duplicate_suffix = (
                        f" [#{identity.name_ordinal + 1}]"
                        if sum(item.name == identity.name for item in input_devices) > 1
                        and identity.name_ordinal is not None
                        else ""
                    )
                    label = f"{device.name}{duplicate_suffix}" + (
                        " (Default)" if device.is_default else ""
                    )
                    self.input_combo.addItem(
                        label,
                        identity,
                    )
                if previous_input is not None:
                    if not self._select_combo_identity(self.input_combo, previous_input):
                        fallback_index = (
                            -1 if self.__dict__.get("_login_startup", False)
                            else self._default_combo_index(self.input_combo)
                        )
                        self.input_combo.setCurrentIndex(fallback_index)
                        if fallback_index >= 0:
                            self.status_bar.showMessage(
                                f"Previous input device '{previous_input.name}' is disconnected; "
                                "using the default until it returns"
                            )
                        elif previous_input.name:
                            self.status_bar.showMessage(
                                f"Saved input device '{previous_input.name}' is disconnected"
                            )
                    else:
                        resolved = self._combo_device_identity(self.input_combo)
                        if (
                            resolved is not None
                            and resolved.to_dict() != previous_input.to_dict()
                        ):
                            self.config.last_input_device = resolved.name
                            self.config.last_input_device_identity = resolved
                            config_dirty = True
                elif self.input_combo.count() > 0:
                    fallback_index = self._default_combo_index(self.input_combo)
                    if fallback_index >= 0:
                        self.input_combo.setCurrentIndex(fallback_index)
            except (RuntimeError, OSError) as e:
                self.input_combo.addItem(f"Error: {e}")
                logger.warning("Input device enumeration failed", exc_info=True)

            # Get output devices
            try:
                output_devices = list_output_devices()
                output_found = len(output_devices) > 0
                for device in output_devices:
                    identity = self._identity_from_device_info(device, "output")
                    duplicate_suffix = (
                        f" [#{identity.name_ordinal + 1}]"
                        if sum(item.name == identity.name for item in output_devices) > 1
                        and identity.name_ordinal is not None
                        else ""
                    )
                    label = f"{device.name}{duplicate_suffix}" + (
                        " (Default)" if device.is_default else ""
                    )
                    self.output_combo.addItem(
                        label,
                        identity,
                    )
                if previous_output is not None:
                    if not self._select_combo_identity(self.output_combo, previous_output):
                        self.output_combo.setCurrentIndex(-1)
                        self.output_combo.setPlaceholderText("Select a replacement destination")
                        self.status_bar.showMessage(
                            f"Destination '{previous_output.name}' is disconnected; "
                            "select a replacement or reconnect it before starting"
                        )
                    else:
                        resolved = self._combo_device_identity(self.output_combo)
                        if (
                            resolved is not None
                            and resolved.to_dict() != previous_output.to_dict()
                        ):
                            self.config.last_output_device = resolved.name
                            self.config.last_output_device_identity = resolved
                            config_dirty = True
                elif self.output_combo.count() > 0:
                    preferred_index = self._preferred_output_combo_index(self.output_combo)
                    if preferred_index >= 0:
                        self.output_combo.setCurrentIndex(preferred_index)

            except (RuntimeError, OSError) as e:
                self.output_combo.addItem(f"Error: {e}")
                logger.warning("Output device enumeration failed", exc_info=True)

            # Update warning banner visibility and text
            if not input_found and not output_found:
                self.device_warning_banner.setText(
                    "Warning: No audio devices detected. Check your audio drivers and connections."
                )
                self.device_warning_banner.setVisible(True)
            elif not input_found:
                self.device_warning_banner.setText(
                    "Warning: No input devices detected. Check your microphone connections."
                )
                self.device_warning_banner.setVisible(True)
            elif not output_found:
                self.device_warning_banner.setText(
                    "Warning: No output devices detected. Check your audio output connections."
                )
                self.device_warning_banner.setVisible(True)
            else:
                self.device_warning_banner.setVisible(False)

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
                self.status_bar.showMessage(
                    "Device selections applied for this session, but could not be saved",
                    6000,
                )
        self._update_session_summary()

    def _restore_from_config(self):
        """Restore settings from loaded config."""
        restored_count = 0
        config_dirty = False
        requested_preset = bool(self.config.startup_preset or self.config.last_preset)

        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)

        # Restore input device
        input_identity = self.config.last_input_device_identity
        if input_identity is None and self.config.last_input_device:
            input_identity = coerce_device_identity(self.config.last_input_device)
        if input_identity is not None:
            index = self._find_combo_index_by_identity(self.input_combo, input_identity)
            if index >= 0:
                self.input_combo.setCurrentIndex(index)
                resolved = self._combo_device_identity(self.input_combo)
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
                    self.input_combo.setCurrentIndex(-1)
                self.status_bar.showMessage(
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
                self.output_combo, output_identity
            )
            if index >= 0:
                self.output_combo.setCurrentIndex(index)
                resolved = self._combo_device_identity(self.output_combo)
                if (
                    resolved is not None
                    and resolved.to_dict() != output_identity.to_dict()
                ):
                    self.config.last_output_device = resolved.name
                    self.config.last_output_device_identity = resolved
                    config_dirty = True
                restored_count += 1
            else:
                self.output_combo.setCurrentIndex(-1)
                self.output_combo.setPlaceholderText("Select a replacement destination")
                self.status_bar.showMessage(
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
                        self.status_bar.showMessage(f"Startup preset: {preset_name}", 5000)
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
                            self.status_bar.showMessage(
                                f"Startup preset: {preset_name}", 5000
                            )
                    except Exception:
                        logger.warning(
                            "Failed to load startup preset %s",
                            preset_name,
                            exc_info=True,
                        )
                        self.status_bar.showMessage(
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
                                self.status_bar.showMessage(
                                    f"Startup preset: {preset_name}", 5000
                                )
                        except Exception:
                            logger.warning(
                                "Failed to load startup preset %s",
                                preset_name,
                                exc_info=True,
                            )
                            self.status_bar.showMessage(
                                f"Failed to load startup preset: {preset_name}", 5000
                            )
                        break
            if not preset_loaded:
                logger.warning(
                    "Startup preset %r not found; falling back to last used",
                    preset_name,
                )
                self.status_bar.showMessage(
                    f"Startup preset '{preset_name}' not found", 5000
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
                        self.status_bar.showMessage(
                            f"Previous preset '{preset_key}' not found, starting with defaults"
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
                        self.status_bar.showMessage(
                            "Previous preset file not found, starting with defaults"
                        )
                        self.config.last_preset = ""
                        config_dirty = True
            except (OSError, ValueError, PresetValidationError) as e:
                logger.warning("Preset restore failed", exc_info=True)
                warning = f"Could not restore the previous preset; starting with defaults. {e}"
                self.config.load_warning = "\n".join(
                    text for text in (self.config.load_warning, warning) if text
                )
                self.config.last_preset = ""
                cleared = self._save_config_safely()
                if not cleared:
                    self.config.load_warning += "\nThe cleared preset reference could not be saved."

        self._startup_restore_ready = (
            not startup_preset_failed and (preset_loaded or not requested_preset)
        )
        # Show appropriate status message
        if self.config.load_warning:
            self.status_bar.showMessage(self.config.load_warning)
        elif self.__dict__.get("_last_preset_identity_persisted") is False:
            self.status_bar.showMessage(
                "Preset applied for this session, but could not be remembered for the next launch",
                6000,
            )
        elif restored_count == 0:
            self.status_bar.showMessage("Ready")
        elif restored_count < 3:
            self.status_bar.showMessage(
                "Restored partial settings (some devices/presets unavailable)"
            )
        else:
            self.status_bar.showMessage("Restored settings from previous session")

        self._apply_latency_compensation_for_current_devices()
        if (
            "input_channel_mode_combo" in self.__dict__
            and "input_cleanup_mode_combo" in self.__dict__
        ):
            self._apply_input_preferences_for_current_route()

        # Route-specific DSP is more specific than the generic startup/last-used
        # preset and is intentionally applied only after both endpoints resolve.
        if not self.__dict__.get("_login_startup", False):
            self._apply_bound_preset_for_current_route()

        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)
        if "input_channel_mode_combo" in self.__dict__:
            self.input_channel_mode_combo.blockSignals(False)
        if "input_cleanup_mode_combo" in self.__dict__:
            self.input_cleanup_mode_combo.blockSignals(False)

        if config_dirty:
            if not self._save_config_safely():
                self.status_bar.showMessage(
                    "Restored settings applied for this session, but could not be saved",
                    6000,
                )

    def _on_device_changed(self):
        """Handle device selection change - save to config."""
        if self.__dict__.get("_login_startup_deadline") is not None:
            self._cancel_login_startup()
        self._login_startup = False
        if hasattr(self, "config"):  # Check config is initialized
            input_identity = self._combo_device_identity(self.input_combo)
            output_identity = self._combo_device_identity(self.output_combo)
            self.config.last_input_device_identity = input_identity
            self.config.last_output_device_identity = output_identity
            self.config.last_input_device = self._device_name_from_identity(
                input_identity
            )
            self.config.last_output_device = self._device_name_from_identity(
                output_identity
            )
            self._apply_input_preferences_for_current_route()
            saved = self._save_config_safely()
            self._apply_latency_compensation_for_current_devices()
            self._apply_bound_preset_for_current_route()
            self._sync_calibration_evidence(force_reset=True)
            if not saved:
                self.status_bar.showMessage(
                    "Device route applied for this session, but could not be saved",
                    6000,
                )
            self._update_session_summary()

    def _on_input_channel_mode_changed(self):
        """Persist and apply the selected input channel mixdown mode."""
        if not hasattr(self, "config"):
            return
        mode = self.input_channel_mode_combo.currentData()
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
        mode = self.input_cleanup_mode_combo.currentData()
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
            self.status_bar.showMessage(
                "Input settings applied for this session, but could not be saved", 6000
            )

    def _start_processing(self, *, interactive: bool = True) -> bool:
        """Start audio processing."""
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
            self.status_bar.showMessage(
                "Audio remains stopped and muted because configuration recovery failed; "
                "reload the preset or restart AudioForge before starting",
                7000,
            )
            self._sync_processing_controls()
            return False

        if self._combo_device_identity(self.output_combo) is None:
            self.status_bar.showMessage("Select an available destination before starting")
            return False

        input_device = self._device_selection_to_name(self.input_combo) or None
        output_device = self._device_selection_to_name(self.output_combo) or None
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

        try:
            result = start_processor_for_route(
                self.processor,
                self._combo_device_identity(self.input_combo),
                self._combo_device_identity(self.output_combo),
            )
            self._apply_latency_compensation_for_current_devices()
            # Native start clears temporary mute state; restore both owners.
            if DEBUG:
                logger.debug("Restoring output mute state after processing start")
            self._apply_output_mute()
            if not interactive and self._output_mute_error is not None:
                self.processor.stop()
                raise RuntimeError("Output mute state could not be restored")
            self.status_bar.showMessage(f"Processing: {result}")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
            self._stream_recovery.mark_processing_started()
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
                    "1. Click 'Refresh' to update device list\n"
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
            self.status_bar.showMessage(f"Error: {e}")
            self._sync_processing_controls()
            return False

    def _sync_processing_controls(self) -> None:
        """Reflect the native processor state after recovery or a failed start."""
        running = bool(self.processor.is_running())
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running or self.__dict__.get("_login_startup_deadline") is not None)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.refresh_btn.setEnabled(not running)
        update_summary = getattr(self, "_update_session_summary", None)
        if callable(update_summary):
            update_summary()
        sync_meters = getattr(self, "_sync_meter_timer", None)
        if callable(sync_meters):
            sync_meters()

    def _stop_processing(self):
        """Stop audio processing."""
        if self.__dict__.get("_login_startup_deadline") is not None:
            self._cancel_login_startup()
        if not self.processor.is_running():
            self._sync_processing_controls()
            if DEBUG:
                logger.debug("Stop processing clicked, but processor is not running")
            return

        if DEBUG:
            logger.debug("Stopping processing")

        try:
            self.processor.stop()
            self.status_bar.showMessage("Processing stopped")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self._stream_recovery.mark_processing_stopped()
            self._sync_meter_timer()
            self._update_session_summary()
            if DEBUG:
                logger.debug("Processing stopped")
        except RuntimeError as e:
            logger.exception("Stop processing failed")
            QMessageBox.critical(self, "Error", f"Failed to stop processing:\n{e}")

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
            dialog.exec()  # Modal dialog - blocks until user closes
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
            dialog.exec()
        finally:
            self._history_transaction_depth -= 1
            self._calibration_dialog_open = False
        self._sync_calibration_evidence()
        return applied

    def capture_pre_auto_eq_state(self):
        """Commit pending edits before Auto-EQ starts its transaction."""
        self._commit_pending_configuration_snapshot()

    def _initialize_configuration_history(self) -> None:
        """Create the immutable baseline after startup restoration."""
        preset = self._get_current_preset()
        snapshot = ConfigurationSnapshot.from_preset(
            preset,
            label="Startup configuration",
            source="startup",
            noise_reference_reliability=self.compressor_panel.get_compressor_settings(
                include_calibration=True
            )["noise_reference_reliability"],
            calibration_context_key=self._calibration_context_key(),
            processing_mode=self._processing_mode(),
        )
        self._configuration_history.initialize(snapshot)
        self._current_value_provenance = dict(snapshot.to_preset().value_provenance)
        if self._saved_preset_payload is None:
            self.current_preset_name = "Default"
            self.current_preset_path = None
            self._saved_preset_payload = self._preset_payload(preset)
            self._saved_processing_mode = self._processing_mode()
            self.preset_modified = False
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
        self.eq_panel.configurationEditStarted.connect(
            self._begin_configuration_transaction
        )
        self.eq_panel.configurationEditFinished.connect(
            self._end_configuration_transaction
        )
        for panel in (
            self.eq_panel,
            self.gate_panel,
            self.deesser_panel,
            self.compressor_panel,
        ):
            panel.configurationEdited.connect(self._queue_configuration_snapshot)

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
        self._history_timer.stop()
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
                noise_reference_reliability=self.compressor_panel.get_compressor_settings(
                    include_calibration=True
                )["noise_reference_reliability"],
                calibration_context_key=self._calibration_context_key(),
                processing_mode=self._processing_mode(),
            )
            recorded = self._configuration_history.record(snapshot)
        except (PresetValidationError, TypeError, ValueError) as error:
            logger.warning(
                "Configuration history snapshot rejected: %s",
                error,
            )
            self.status_bar.showMessage(
                "Could not record this configuration edit",
                5000,
            )
            return False
        self._sync_calibration_evidence()
        if recorded:
            self._current_value_provenance = dict(snapshot.to_preset().value_provenance)
            if source == "ui":
                self.eq_panel.set_auto_eq_diagnostics(None)
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
        if self._undo_auto_eq_button is not None:
            self._undo_auto_eq_button.setEnabled(history.can_undo)
            self._undo_auto_eq_button.setToolTip(
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
            self.status_bar.showMessage("No configuration change to undo", 3000)
        else:
            self.status_bar.showMessage(
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
            self.status_bar.showMessage("No configuration change to redo", 3000)
        else:
            self.status_bar.showMessage(f"Redid: {restored.label}", 3000)
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

        preset = self._get_current_preset()
        preset.name = preset_name
        preset.description = description
        preset.version = __version__
        self._save_preset_file(preset)

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
            message = f"Preset saved: {filepath.name}"
            if self._processing_mode() == "raw":
                message += "; Raw Monitor is session-only and was not stored in the preset"
            if not persisted:
                message += "; could not remember it for the next launch"
            self.status_bar.showMessage(message, 6000 if not persisted else 3000)
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
                "Auto-generated voice chain with EQ, gate/VAD, de-esser, "
                f"and compressor tuned for the {target_curve.title()} target"
            ),
        )

    def _on_rnnoise_toggled(self, checked):
        """Handle RNNoise toggle."""
        self.processor.set_rnnoise_enabled(checked)
        self._queue_configuration_snapshot()

    def _on_strength_changed(self, value: int):
        """Handle RNNoise strength slider change."""
        strength = value / 100.0  # Convert 0-100 to 0.0-1.0
        self._rnnoise_strength_exact = strength
        self.strength_label.setText(f"{value}%")
        self.processor.set_rnnoise_strength(strength)
        self._queue_configuration_snapshot()

    def _apply_noise_model(self, model_id: str) -> None:
        """Apply manual or calibrated selection without disguising a failed switch."""
        index = self.model_combo.findData(model_id)
        if index < 0 or not self.processor.set_noise_model(model_id):
            raise ValueError(f"Noise model {model_id!r} is unavailable; previous model retained")
        blocked = self.model_combo.blockSignals(True)
        try:
            self.model_combo.setCurrentIndex(index)
        finally:
            self.model_combo.blockSignals(blocked)
        self._set_noise_suppression_latency_label(model_id)

    def _on_model_changed(self, index: int):
        """Apply a manual selection, restoring the actual model on failure."""
        model_id = self.model_combo.itemData(index)
        if not model_id:
            return
        previous_model = self.processor.get_noise_model()
        try:
            self._apply_noise_model(model_id)
        except Exception as error:
            blocked = self.model_combo.blockSignals(True)
            try:
                self.model_combo.setCurrentIndex(self.model_combo.findData(previous_model))
            finally:
                self.model_combo.blockSignals(blocked)
            QMessageBox.warning(self, "Model Switch Failed", str(error))
        else:
            self.status_bar.showMessage(f"Switched to {self.model_combo.currentText()}")
            self._queue_configuration_snapshot()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_meter_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._sync_meter_timer()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_meter_timer()

    def _sync_meter_timer(self) -> None:
        timer = self.__dict__.get("meter_timer")
        if timer is None:
            return
        try:
            running = self.processor.is_running()
        except (AttributeError, OSError, RuntimeError):
            running = False
        if running and self.isVisible() and not self.isMinimized():
            if not timer.isActive():
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
        self.compressor_panel.update_gain_reduction(None)
        self.compressor_panel.update_auto_makeup_meters(None, None)
        self.compressor_panel.current_release_label.setText("--")
        self.deesser_panel.update_gain_reduction(None)
        self.gate_panel.update_vad_confidence(None)
        self._live_meters_invalidated = True

    def _update_meters(self):
        """Refresh available live measurements; keep recovery on its own timer."""
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

        def reading(name: str) -> float | None:
            try:
                value = float(getattr(self.processor, name)())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
                return None
            return value if math.isfinite(value) else None

        self.input_meter.set_levels(reading("get_input_rms_db"), reading("get_input_peak_db"))
        self.output_meter.set_levels(reading("get_output_rms_db"), reading("get_output_peak_db"))
        self.compressor_panel.update_gain_reduction(reading("get_compressor_gain_reduction_db"))
        self.deesser_panel.update_gain_reduction(reading("get_deesser_gain_reduction_db"))
        self.compressor_panel._update_current_release()
        try:
            auto_makeup_enabled = self.processor.get_compressor_auto_makeup_enabled()
        except (AttributeError, OSError, RuntimeError):
            auto_makeup_enabled = False
        self.compressor_panel.update_auto_makeup_meters(
            reading("get_compressor_current_lufs") if auto_makeup_enabled else None,
            reading("get_compressor_current_makeup_gain") if auto_makeup_enabled else None,
        )
        self.gate_panel.update_vad_confidence(reading("get_vad_probability"))

    def _update_diagnostics(self):
        """Update slower diagnostics and service recovery."""
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
            self._stream_recovery.mark_processing_stopped()
            self._last_backend_warning = None
            self._reset_health_labels()
            # A failed native restart intentionally leaves recovery pending
            # while the stream is stopped. Keep servicing that request here.
            self._service_stream_recovery(
                diagnostics={},
                input_rms=-120.0,
                output_rms=-120.0,
                output_buf=0,
            )
            self._sync_processing_controls()
            return

        try:
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
            self._set_health_chip(self.health_summary_label, "Health: telemetry unavailable", "warn")
            # Missing presentation data must not suppress native recovery service.
            self._service_stream_recovery(
                diagnostics={}, input_rms=float("nan"), output_rms=float("nan"), output_buf=0,
            )
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
        )
        self._service_stream_recovery(
            diagnostics=diagnostics,
            input_rms=input_rms,
            output_rms=output_rms,
            output_buf=output_buf,
        )

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
    ) -> None:
        """Update diagnostic status labels from a runtime diagnostic snapshot."""
        if not hasattr(self, "_stream_health"):
            self._stream_health = RecentStreamHealth()
        try:
            recent_events = self._stream_health.observe(diagnostics)
        except ValueError:
            self._reset_health_labels()
            self._set_health_chip(self.recovery_diag_label, "Health data unavailable", "warn")
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
            self.latency_label,
            f"Latency: ~{latency_ms:.0f}ms | DSP {dsp_time_ms:.1f}ms",
            "info",
        )

        pipeline_buf = input_buf + rnnoise_buf
        if pipeline_buf < 960:
            buf_status = "OK"
            buf_state = "ok"
        elif pipeline_buf < 1920:
            buf_status = "WARN"
            buf_state = "warn"
        else:
            buf_status = "BAD"
            buf_state = "bad"
        self._set_health_chip(
            self.buffer_label,
            f"Buffer: {buf_status} ({pipeline_buf})",
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
        input_clip_count = int(diagnostics.get("clip_event_count", 0) or 0)
        previous_input_clip_count = int(
            getattr(self, "_last_input_clip_event_count", input_clip_count) or 0
        )
        new_input_clip_observed = input_clip_count > previous_input_clip_count
        self._last_input_clip_event_count = input_clip_count
        output_clip_count = int(diagnostics.get("output_clip_event_count", 0) or 0)
        previous_output_clip_count = int(
            getattr(self, "_last_output_clip_event_count", output_clip_count) or 0
        )
        new_output_clip_observed = output_clip_count > previous_output_clip_count
        self._last_output_clip_event_count = output_clip_count
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
            clip_delta=new_input_clip_observed,
            phase_rescue_active=phase_rescue_active,
            cleanup_rumble_detected=cleanup_rumble_detected,
            cleanup_hum_detected=cleanup_hum_detected,
            cleanup_mode=cleanup_mode,
            crest_factor_db=input_crest_db
            if isinstance(input_crest_db, (int, float))
            else None,
        )
        if phase_rescue_active:
            strategy_label = phase_rescue_strategy.replace("_", " ").upper()
            input_health_text = f"Input: PHASE {strategy_label}"
        self._set_health_chip(
            self.input_health_label,
            input_health_text,
            input_health_state,
        )

        output_health_text, output_health_state = build_output_health_state(
            rms_db=output_rms_db,
            clip_delta=new_output_clip_observed,
            true_peak_delta=new_output_true_peak_observed,
            output_clip_count=output_clip_count,
            true_peak_count=output_true_peak_count,
            true_peak_db=output_true_peak_db,
            true_peak_headroom_db=output_true_peak_headroom,
            short_term_lufs=output_lufs,
            limiter_history_db=limiter_history_db,
            true_peak_limiter_history_db=true_peak_limiter_history_db,
        )
        self._set_health_chip(
            self.output_health_label,
            output_health_text,
            output_health_state,
        )

        if gate_auto_relax_active:
            gate_health_text = f"Gate: RELAX (GCH:{gate_chatter_count})"
            gate_health_state = "warn"
        elif new_gate_chatter_observed:
            gate_health_text = f"Gate: CHATTER (GCH:{gate_chatter_count})"
            gate_health_state = "warn"
        else:
            gate_health_text = "Gate: OK"
            gate_health_state = "ok"
        self._set_health_chip(
            self.gate_health_label,
            gate_health_text,
            gate_health_state,
        )

        callback_ages = [
            age
            for age in (input_callback_age_ms, output_callback_age_ms)
            if age is not None and age >= 0
        ]
        if callback_ages:
            max_callback_age = max(callback_ages)
            callback_state = (
                "bad"
                if max_callback_age > 1000
                else "warn"
                if max_callback_age > 250
                else "ok"
            )
            callback_health_text = (
                f"Callbacks: I:{input_callback_age_ms}ms O:{output_callback_age_ms}ms"
            )
        else:
            callback_health_text = "Callbacks: --"
            callback_state = "idle"
        self._set_health_chip(
            self.callback_health_label,
            callback_health_text,
            callback_state,
        )

        underrun_state = "warn" if underrun_streak or new_underruns_observed else "ok"
        underrun_text = f"Underruns: {underruns}"
        if underrun_streak:
            underrun_text += f" streak:{underrun_streak}"
        self._set_health_chip(
            self.underrun_health_label,
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
            ],
        )
        if rt_error_active:
            dropped_bits.append(f"RT:{rt_error_name}")
        dropped_state = (
            "ok"
            if (
                not recent_events
                and not recovery_pending
                and underrun_streak == 0
                and not new_underruns_observed
                and not rt_error_active
                and not new_input_clip_observed
                and not new_output_clip_observed
                and not new_output_true_peak_observed
                and not new_gate_chatter_observed
                and not gate_auto_relax_active
                and not phase_rescue_active
                and not current_phase_warning
                and not new_phase_warning_observed
                and not cleanup_rumble_detected
                and limiter_history_db < 6.0
                and true_peak_limiter_history_db < 3.0
                and (
                    output_true_peak_headroom is None
                    or output_true_peak_headroom >= 0.75
                )
            )
            else "warn"
        )
        dropped_detail = " | ".join(dropped_bits)
        dropped_summary = dropped_bits[:5]
        if dropped_state != "ok":
            dropped_summary.append("WARN")
        self._set_health_chip(
            self.dropped_label,
            " | ".join(dropped_summary),
            dropped_state,
        )
        self.dropped_label.setToolTip(
            f"{DROPPED_DIAGNOSTICS_TOOLTIP}\n\nCurrent counters:\n{dropped_detail}"
        )
        self.dropped_label.setAccessibleDescription(dropped_detail)

        backend_available = diagnostics.get("noise_backend_available", True)
        backend_failed = diagnostics.get("noise_backend_failed", False)
        backend_error = diagnostics.get("noise_backend_error")
        noise_model = diagnostics.get("noise_model", "rnnoise")
        if noise_model != "rnnoise" and (backend_failed or not backend_available):
            warning = (
                backend_error or "Selected neural backend fell back to dry passthrough."
            )
            if warning != self._last_backend_warning:
                self.status_bar.showMessage(warning, 6000)
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

            backend_bits = [noise_model]
            if backend_ok:
                backend_bits.append("OK")
            elif backend_failed:
                backend_bits.append("FAILED")
            else:
                backend_bits.append("UNAVAILABLE")
            if non_finite:
                backend_bits.append(f"NF:{non_finite}")
            if backend_error:
                backend_bits.append("ERR")
            self._extend_diag_tokens(
                backend_bits,
                diagnostics,
                [
                    ("input_resampler_active", "IR"),
                    ("output_resampler_active", "OR"),
                ],
            )
            self._set_health_chip(
                self.backend_diag_label,
                f"Backend: {' '.join(str(bit) for bit in backend_bits)}",
                "ok" if backend_ok else "warn",
            )

            recovery_bits = [f"R:{restart_count}"]
            recovery_bits.append(f"ORE:{output_recovery_count}")
            if suppressed:
                recovery_bits.append("SUPP")
            reason = diagnostics.get("last_restart_reason")
            recent_recovery = bool(recent_events & {
                "stream_restart_count", "output_recovery_count", "output_recovery_event_count",
                "input_backlog_recovery_count",
            })
            if recovery_pending:
                recovery_bits.append("PENDING")
            elif recent_recovery:
                recovery_bits.append("RECENT")
            self._set_health_chip(
                self.recovery_diag_label,
                f"Recovery: {' '.join(recovery_bits)}",
                "warn"
                if recovery_pending or recent_recovery
                else ("info" if suppressed else "ok"),
            )
            self.recovery_diag_label.setToolTip(
                f"Last recovery: {reason}" if reason else "No recovery recorded."
            )
        except Exception:
            logger.debug("Diagnostic label update failed", exc_info=True)
        if self.__dict__.get("health_summary_label") is not None:
            states = []
            for label in self._health_decision_widgets + self._health_layout_widgets:
                state = label.property("health_state")
                states.append(state if isinstance(state, str) else "idle")
            state = next((level for level in ("bad", "warn", "ok") if level in states), "idle")
            if self.__dict__.get("_output_mute_error") and state in {"ok", "idle"}:
                state = "warn"
            issues = [
                label.text() for label in self._health_decision_widgets + self._health_layout_widgets
                if label.property("health_state") in {"bad", "warn"}
            ]
            text = issues[0] if issues else f"Health: {state.upper()}"
            self._set_health_chip(self.health_summary_label, text, state)
            advice_label = self.__dict__.get("health_advice_label")
            if advice_label is not None:
                advice = advice_for(issues[0]) if issues else ""
                if not advice:
                    advice = {
                        "ok": "Everything looks fine.",
                        "idle": "Start processing to see audio health.",
                    }.get(state, "Check the items marked below.")
                advice_label.setText(advice)
                self.health_summary_label.setToolTip(advice)
            self._update_session_summary()

    def _service_stream_recovery(
        self,
        *,
        diagnostics: dict,
        input_rms: float,
        output_rms: float,
        output_buf: int,
    ) -> None:
        """Service UI-side and Rust-side stream recovery."""
        gate_mode_combo = getattr(getattr(self, "gate_panel", None), "gate_mode_combo", None)
        gate_mode = gate_mode_combo.currentIndex() if gate_mode_combo is not None else 0
        if self._stream_recovery.maybe_recover_output_stall(
            input_rms=input_rms,
            output_rms=output_rms,
            output_buf=output_buf,
            calibration_dialog_open=self._calibration_dialog_open,
            recovery_suppressed=bool(diagnostics.get("recovery_suppressed", False)),
            intentional_mute=bool(diagnostics.get("output_muted", False)),
            expected_gate_closed=(
                gate_mode == 2
                and float(diagnostics.get("gate_fused_score", 1.0)) < 0.12
            ),
        ):
            self._recover_output_path()

        input_cb_age_ms = None
        output_cb_age_ms = None
        try:
            if hasattr(self.processor, "get_input_callback_age_ms"):
                input_cb_age_ms = self.processor.get_input_callback_age_ms()
            if hasattr(self.processor, "get_output_callback_age_ms"):
                output_cb_age_ms = self.processor.get_output_callback_age_ms()
        except Exception:
            input_cb_age_ms = None
            output_cb_age_ms = None
        if input_cb_age_ms is not None and output_cb_age_ms is not None:
            if self._stream_recovery.maybe_recover_callback_stall(
                input_cb_age_ms=input_cb_age_ms,
                output_cb_age_ms=output_cb_age_ms,
                calibration_dialog_open=self._calibration_dialog_open,
            ):
                self._recover_output_path()

        try:
            recovery_result = self.processor.service_recovery()
            if recovery_result is not None:
                if recovery_result:
                    self._apply_latency_compensation_for_current_devices()
                    reason = ""
                    try:
                        reason = self.processor.get_last_restart_reason() or ""
                    except Exception:
                        reason = ""
                    suffix = f" ({reason})" if reason else ""
                    self.status_bar.showMessage(
                        f"Recovered audio stream{suffix}",
                        4000,
                    )
                else:
                    err_msg = ""
                    try:
                        err_msg = self.processor.get_last_stream_error() or ""
                    except Exception:
                        err_msg = ""
                    if err_msg:
                        self.status_bar.showMessage(
                            f"Selected route unavailable; retrying: {err_msg}",
                            6000,
                        )
                    else:
                        self.status_bar.showMessage(
                            "Selected route unavailable; retrying recovery",
                            6000,
                        )
                self._sync_processing_controls()
        except Exception:
            logger.debug("Rust recovery service failed", exc_info=True)

    def _recover_output_path(self):
        """Best-effort output recovery while preserving both mute owners."""
        try:
            self.processor.stop()
            self._apply_output_mute()
            result = start_processor_for_route(
                self.processor,
                self._combo_device_identity(self.input_combo),
                self._combo_device_identity(self.output_combo),
            )
            self._apply_latency_compensation_for_current_devices()
            self._apply_output_mute()
            self.status_bar.showMessage(
                f"Recovered output path automatically: {result}",
                4000,
            )
            self._sync_processing_controls()
        except Exception as e:
            logger.exception("Auto-recovery failed")
            self.status_bar.showMessage(
                f"Auto-recovery failed: {e}",
                5000,
            )
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
            self._update_session_summary()

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
            preset = self._get_current_preset().to_dict()
            snapshot = build_diagnostics_snapshot(
                app_version=__version__,
                runtime_diagnostics=runtime,
                config=self.config,
                processing_settings=preset,
                input_device=self._combo_device_identity(self.input_combo),
                output_device=self._combo_device_identity(self.output_combo),
                processing_sample_rate_hz=int(self.processor.sample_rate()),
                output_sample_rate_hz=int(self.processor.output_sample_rate()),
                running=bool(self.processor.is_running()),
            )
            write_diagnostics_snapshot(filepath, snapshot)
        except Exception:
            logger.exception("Diagnostics export failed")
            QMessageBox.critical(
                self,
                "Diagnostics Export Failed",
                "AudioForge could not create the diagnostics snapshot.",
            )
            return

        self.status_bar.showMessage("Privacy-safe diagnostics exported", 4000)
        QMessageBox.information(
            self,
            "Diagnostics Exported",
            "The snapshot was saved without raw audio, device names, "
            "environment variables, secrets, or arbitrary paths.",
        )

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
            "<p><b>Target neural-suppression latency:</b> up to about 30ms</p>",
        )

    def _on_dropped_context_menu(self, pos):
        """Handle right-click context menu on dropped samples label."""
        menu = QMenu(self)
        reset_action = QAction("Reset Counter", self)
        reset_action.triggered.connect(self._reset_dropped_samples)
        menu.addAction(reset_action)
        menu.exec(self.dropped_label.mapToGlobal(pos))

    def _reset_dropped_samples(self):
        """Reset the dropped samples counter."""
        if self.processor:
            self.processor.reset_dropped_samples()
            self.status_bar.showMessage("Dropped samples counter reset", 3000)

    def _get_current_preset(self) -> Preset:
        """Get current settings as a Preset object."""
        gate_settings = self.gate_panel.get_settings()
        eq_settings = self.eq_panel.get_eq_settings()
        deesser_settings = self.deesser_panel.get_settings()
        compressor_settings = self.compressor_panel.get_compressor_settings()
        limiter_settings = self.compressor_panel.get_limiter_settings()

        return Preset(
            name="Custom",
            description="User-defined preset",
            gate=GateSettings(**gate_settings),
            eq=eq_settings,
            rnnoise=RNNoiseSettings(
                enabled=self.rnnoise_checkbox.isChecked(),
                strength=float(getattr(self, "_rnnoise_strength_exact", 1.0)),
                model=self.model_combo.currentData() or "rnnoise",
            ),
            deesser=DeEsserSettings(**deesser_settings),
            compressor=CompressorSettings(**compressor_settings),
            limiter=LimiterSettings(**limiter_settings),
            bypass=self._processing_mode() == "bypass",
            value_provenance=dict(self._current_value_provenance),
        )

    def _write_processing_configuration(self, preset: Preset) -> None:
        """Write a validated chain while the caller owns history and output mute."""
        model = preset.rnnoise.model
        index = self.model_combo.findData(model)
        if index < 0 or not self.processor.set_noise_model(model):
            raise RuntimeError(f"Noise model {model!r} is unavailable")
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._set_noise_suppression_latency_label(model)
        gate_settings = asdict(preset.gate)
        gate_apply = getattr(self.gate_panel, "apply_settings_synchronously", None)
        if callable(gate_apply):
            gate_apply(gate_settings)
        else:
            self.gate_panel.set_settings(gate_settings)
        self.eq_panel.set_settings(preset.eq.to_dict())
        block_signals = getattr(self.rnnoise_checkbox, "blockSignals", None)
        blocked = block_signals(True) if callable(block_signals) else None
        try:
            self.rnnoise_checkbox.setChecked(preset.rnnoise.enabled)
        finally:
            if callable(block_signals):
                block_signals(blocked)
        self.processor.set_rnnoise_enabled(preset.rnnoise.enabled)
        self._rnnoise_strength_exact = float(preset.rnnoise.strength)
        blocked = self.strength_slider.blockSignals(True)
        try:
            self.strength_slider.setValue(round(self._rnnoise_strength_exact * 100))
        finally:
            self.strength_slider.blockSignals(blocked)
        self.strength_label.setText(f"{self.strength_slider.value()}%")
        self.processor.set_rnnoise_strength(self._rnnoise_strength_exact)
        deesser_settings = asdict(preset.deesser)
        deesser_apply = getattr(self.deesser_panel, "apply_settings_synchronously", None)
        if callable(deesser_apply):
            deesser_apply(deesser_settings)
        else:
            self.deesser_panel.set_settings(deesser_settings)
        compressor_settings = asdict(preset.compressor)
        limiter_settings = asdict(preset.limiter)
        dynamics_apply = getattr(
            self.compressor_panel,
            "apply_processing_settings_synchronously",
            None,
        )
        if callable(dynamics_apply):
            dynamics_apply(compressor_settings, limiter_settings)
        else:
            self.compressor_panel.set_compressor_settings(compressor_settings)
            self.compressor_panel.set_limiter_settings(limiter_settings)
        self._set_processing_mode("bypass" if preset.bypass else "normal")
        self._current_value_provenance = dict(preset.value_provenance)

    def _cancel_processing_configuration_writes(self) -> None:
        """Discard interactive writes superseded by the bulk configuration."""
        limiters = []
        for band in getattr(self.eq_panel, "band_sliders", ()):
            for name in ("_rate_limiter", "_frequency_rate_limiter"):
                limiter = getattr(band, name, None)
                if limiter is not None:
                    limiters.append(limiter)
        panel_limiters = (
            (self.eq_panel, ("_curve_rate_limiter",)),
            (self.gate_panel, ("_rate_limiter",)),
            (self.deesser_panel, ("_rate_limiter",)),
            (self.compressor_panel, ("_comp_rate_limiter", "_limiter_rate_limiter")),
        )
        for panel, names in panel_limiters:
            limiters.extend(
                limiter
                for name in names
                if (limiter := getattr(panel, name, None)) is not None
            )
        for limiter in limiters:
            limiter.cancel()

    def _flush_eq_configuration_writes(self) -> None:
        """Drain compatible queued writes while configuration mute is held."""
        for band in getattr(self.eq_panel, "band_sliders", ()):
            for name in ("_rate_limiter", "_frequency_rate_limiter"):
                limiter = getattr(band, name, None)
                if limiter is not None:
                    limiter.flush()
        curve_limiter = getattr(self.eq_panel, "_curve_rate_limiter", None)
        if curve_limiter is not None:
            curve_limiter.flush()

    def _flush_processing_configuration_writes(self) -> None:
        """Finish any compatibility-path panel writes before unmuting."""
        self._flush_eq_configuration_writes()
        for panel, names in (
            (self.gate_panel, ("_rate_limiter",)),
            (self.deesser_panel, ("_rate_limiter",)),
            (self.compressor_panel, ("_comp_rate_limiter", "_limiter_rate_limiter")),
        ):
            for name in names:
                limiter = getattr(panel, name, None)
                if limiter is not None:
                    limiter.flush()

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
        previous = self._get_current_preset()
        previous_mode = self._processing_mode()
        previous_compressor = self.compressor_panel.get_compressor_settings(
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
                self.processor.stop()
                raise RuntimeError("Audio stopped because configuration mute failed")
            try:
                self._cancel_processing_configuration_writes()
                self._write_processing_configuration(candidate)
                if processing_mode is not None:
                    self._set_processing_mode(processing_mode)
                if noise_reference_reliability is not None or compressor_metadata:
                    self.compressor_panel.set_compressor_settings({
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
                    self.compressor_panel.set_compressor_settings(previous_compressor)
                    self._flush_processing_configuration_writes()
                except Exception as restore_error:
                    release_mute = False
                    try:
                        self.processor.stop()
                    except Exception as stop_error:
                        self.status_bar.showMessage(
                            "Audio remains muted, but configuration restoration and "
                            "processor stop both failed; restart AudioForge",
                            7000,
                        )
                        raise RuntimeError(
                            f"Configuration failed ({error}); restoration failed ({restore_error}); "
                            f"processor stop failed ({stop_error}). Audio remains muted."
                        ) from stop_error
                    self.status_bar.showMessage(
                        "Audio stopped and muted: configuration restoration failed",
                        7000,
                    )
                    raise RuntimeError(
                        f"Configuration failed ({error}); restoration failed ({restore_error}). "
                        "Audio is stopped and muted."
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
        require_exact: bool = False,
        scope: str = "complete",
        preset_path: Path | None = None,
        persist_last_used: bool = True,
    ) -> bool:
        """Apply a preset to the UI and processor.

        Args:
            preset: Preset object to apply
            preset_key: Optional key for built-in presets (e.g., "voice", "bass_cut")
            scope: "complete" replaces the processing chain; "eq" replaces EQ only.
        """
        if scope not in {"complete", "eq"}:
            raise ValueError(f"Unknown preset scope: {scope}")
        if scope == "eq":
            self.eq_panel.enabled_checkbox.setChecked(preset.eq.enabled)
            self.eq_panel._apply_typed_bands(preset.eq.bands, layer="tone")
            self.status_bar.showMessage(f"Applied EQ-only template: {preset.name}")
            if self.__dict__.get("_history_ready", False) and not self.__dict__.get(
                "_history_replaying", False
            ):
                self._commit_pending_configuration_snapshot(
                    label=f"EQ-only template ({preset.name})",
                    source="preset",
                )
            else:
                self._set_preset_modified(True)
            return True

        if (
            self.__dict__.get("_history_ready", False)
            and not self.__dict__.get("_history_replaying", False)
            and not self._confirm_discard_changes()
        ):
            return False
        try:
            self.apply_processing_configuration(preset)
        except Exception as error:
            if require_exact:
                raise
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
            self._set_preset_identity(
                preset,
                path=preset_path,
                preset_id=preset_id,
                persist_last_used=persist_last_used,
            )
        if history_ready and not history_replaying:
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
            self._last_preset_identity_persisted = bool(persisted)

        message = f"Loaded preset: {preset.name}"
        if persist_identity and not persisted:
            message += "; applied for this session, but could not be remembered for the next launch"
        self.status_bar.showMessage(message, 6000 if persist_identity and not persisted else 3000)
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
        preset = self._get_current_preset()
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
        preset = self._get_current_preset()
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
        import subprocess

        presets_dir = get_presets_dir()

        if os.name == "nt":  # Windows
            subprocess.run(["explorer", str(presets_dir)])
        elif os.name == "posix":  # Linux/Mac
            subprocess.run(["xdg-open", str(presets_dir)])

    def closeEvent(self, event):
        """Handle window close."""
        if (
            not self._quitting
            and self._tray_icon is not None
            and self._tray_icon.isVisible()
            and self._close_to_tray_action is not None
            and self._close_to_tray_action.isChecked()
        ):
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
            event.ignore()
            self.status_bar.showMessage(
                "AudioForge is still running in the tray; use Quit AudioForge to exit",
                5000,
            )
            return

        if not self._confirm_discard_changes():
            self._quitting = False
            event.ignore()
            return
        self._quitting = True
        self._unregister_mute_hotkey()
        if self._tray_icon is not None:
            self._tray_icon.hide()
        self.config.window_geometry = {
            "x": self.x(),
            "y": self.y(),
            "width": self.width(),
            "height": self.height(),
        }
        try:
            saved = self._save_ui_state()
        finally:
            if self.processor.is_running():
                self.processor.stop()
        if not saved:
            QMessageBox.warning(
                self,
                "Settings Not Saved",
                "Audio processing has stopped, but window settings could not be saved.",
            )
        event.accept()


def run_app():
    """Run the AudioForge application."""
    return run_qt_app(MainWindow)


if __name__ == "__main__":
    sys.exit(run_app())
