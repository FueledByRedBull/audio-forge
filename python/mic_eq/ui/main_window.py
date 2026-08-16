"""
Main window for AudioForge application

Adapted from Spectral Workbench project.

DEBUG: Added terminal logging for processor state tracking
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QStatusBar,
    QMessageBox,
    QSplitter,
    QFileDialog,
    QInputDialog,
    QMenu,
    QSlider,
    QScrollArea,
    QFrame,
    QTabWidget,
    QAbstractButton,
    QSpinBox,
    QDoubleSpinBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QAction, QGuiApplication
import os
import sys
import json
import logging
import shutil
from pathlib import Path

from .gate_panel import GatePanel
from .eq_panel import EQPanel
from .compressor_panel import CompressorPanel
from .deesser_panel import DeEsserPanel
from .level_meter import LevelMeter
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
    SECONDARY_ACTION_BUTTON_STYLE,
    SUBDUED_TEXT_STYLE,
    WARNING_BANNER_STYLE,
    configure_responsive_combo,
    status_chip_style,
)
from .startup_presets import (
    STARTUP_BUILTIN_PREFIX,
    STARTUP_CUSTOM_PREFIX,
    normalize_startup_preset_id as _normalize_startup_preset_id,
    startup_builtin_id as _startup_builtin_id,
    startup_custom_id as _startup_custom_id,
    startup_preset_display_name as _startup_preset_display_name,
)
from .theme import prefers_reduced_motion
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
    get_presets_dir,
    get_preset_imports_dir,
    list_presets,
    BUILTIN_PRESETS,
    save_config,
    load_config,
    build_device_route_key,
    DevicePresetBinding,
    coerce_device_identity,
    legacy_latency_profile_key,
    LatencyCalibrationProfile,
)
from ..config_parts.presets import validate_preset_file_size

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

    LEFT_PANE_MIN_WIDTH = 290
    RIGHT_PANE_MIN_WIDTH = 340
    COMPACT_LAYOUT_BREAKPOINT = 1200
    VERTICAL_SPLITTER_BREAKPOINT = 1160

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioForge - Microphone Audio Processor")

        # Create audio processor
        self.processor = AudioProcessor()

        # Load configuration
        self.config = load_config()
        self.current_preset_path = None

        # Bounded processing-configuration history. It stores immutable preset
        # data only; live audio buffers and realtime processor state stay out.
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
        self._last_backend_warning = None
        self._last_output_underrun_total = 0
        self._last_input_clip_event_count = 0
        self._last_output_clip_event_count = 0
        self._last_output_true_peak_event_count = 0
        self._last_input_phase_warning_count = 0
        self._last_gate_chatter_event_count = 0
        self._responsive_layout_compact: bool | None = None
        self._splitter_is_vertical: bool | None = None
        self._ui_state_timer = QTimer(self)
        self._ui_state_timer.setSingleShot(True)
        self._ui_state_timer.timeout.connect(self._save_ui_state)

        # Set up UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_options_menu()
        self._setup_statusbar()

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
        QTimer.singleShot(0, self._maybe_show_first_run_setup)

        # Meter update timer (60 FPS)
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._update_meters)
        self.meter_timer.start(100 if prefers_reduced_motion() else 16)

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
        """Set up the user interface."""
        self.content_scroll_area = QScrollArea()
        self.content_scroll_area.setWidgetResizable(True)
        self.content_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.content_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        central_widget = QWidget()
        self.content_scroll_area.setWidget(central_widget)
        self.setCentralWidget(self.content_scroll_area)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(
            MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL
        )
        main_layout.setSpacing(SPACING_SECTION)

        # Warning banner for missing audio devices (hidden by default)
        self.device_warning_banner = QLabel(
            "Warning: No audio devices detected. Check your audio drivers and connections."
        )
        self.device_warning_banner.setStyleSheet(WARNING_BANNER_STYLE)
        self.device_warning_banner.setAccessibleName("Audio device warning")
        self.device_warning_banner.setWordWrap(True)
        self.device_warning_banner.setVisible(False)
        main_layout.addWidget(self.device_warning_banner)

        # Top: Device selection
        device_group = QGroupBox("Audio Devices")
        self.device_layout = QGridLayout(device_group)
        self.device_layout.setSpacing(
            SPACING_NORMAL
        )  # Consistent spacing for device controls

        # Input device
        input_label = QLabel("Input:")
        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(150)
        bind_label(input_label, self.input_combo, name="Input audio device")

        # Output device
        output_label = QLabel("Output:")
        self.output_combo = QComboBox()
        self.output_combo.setMinimumWidth(150)
        bind_label(output_label, self.output_combo, name="Output audio device")

        input_mode_label = QLabel("Input Mode:")
        self.input_channel_mode_combo = QComboBox()
        for label, mode in INPUT_CHANNEL_MODE_OPTIONS:
            self.input_channel_mode_combo.addItem(label, mode)
        self.input_channel_mode_combo.setMinimumWidth(130)
        self.input_channel_mode_combo.setToolTip(
            "How multichannel input is converted to mono. Use Left/Right or Phase-safe mono if stereo channels cancel."
        )
        bind_label(
            input_mode_label,
            self.input_channel_mode_combo,
            name="Input channel mode",
        )

        cleanup_label = QLabel("Cleanup:")
        self.input_cleanup_mode_combo = QComboBox()
        for label, mode in INPUT_CLEANUP_MODE_OPTIONS:
            self.input_cleanup_mode_combo.addItem(label, mode)
        self.input_cleanup_mode_combo.setMinimumWidth(96)
        self.input_cleanup_mode_combo.setToolTip(
            "Optional adaptive input cleanup after the fixed safe pre-filter. Off preserves the existing DC/80 Hz path."
        )
        bind_label(
            cleanup_label,
            self.input_cleanup_mode_combo,
            name="Input cleanup mode",
        )

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.refresh_btn.setAccessibleName("Refresh audio devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        self._device_layout_widgets = (
            input_label,
            self.input_combo,
            output_label,
            self.output_combo,
            input_mode_label,
            self.input_channel_mode_combo,
            cleanup_label,
            self.input_cleanup_mode_combo,
            self.refresh_btn,
        )

        main_layout.addWidget(device_group)

        # Middle: meters plus tabbed controls/EQ splitter
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(SPACING_NORMAL)

        input_meter_layout = QVBoxLayout()
        self.input_meter = LevelMeter("IN", show_scale=True)
        self.input_meter.setAccessibleName("Input level meter")
        self.input_meter.setFixedWidth(50)
        input_meter_layout.addWidget(self.input_meter)
        middle_layout.addLayout(input_meter_layout)

        self.gate_panel = GatePanel(self.processor)
        self.deesser_panel = DeEsserPanel(self.processor)
        self.compressor_panel = CompressorPanel(self.processor)
        self.noise_suppression_group = self._create_noise_suppression_group()

        self.control_tabs = QTabWidget()
        self.control_tabs.setAccessibleName("Processing controls")
        self.control_tabs.setDocumentMode(True)
        self.control_tabs.setMinimumWidth(self.LEFT_PANE_MIN_WIDTH)
        self.control_tabs.addTab(
            self._create_tab_page([self.gate_panel, self.noise_suppression_group]),
            "Cleanup",
        )
        self.control_tabs.addTab(
            self._create_tab_page([self.deesser_panel, self.compressor_panel]),
            "Dynamics",
        )
        self.control_tabs.currentChanged.connect(self._on_main_control_tab_changed)

        self.eq_panel = EQPanel(self.processor)
        self.eq_panel.setMinimumWidth(0)

        self.eq_scroll_area = QScrollArea()
        self.eq_scroll_area.setWidget(self.eq_panel)
        self.eq_scroll_area.setWidgetResizable(True)
        self.eq_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.eq_scroll_area.setMinimumWidth(self.RIGHT_PANE_MIN_WIDTH)
        self.eq_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.eq_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.control_tabs)
        self.main_splitter.addWidget(self.eq_scroll_area)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        middle_layout.addWidget(self.main_splitter, stretch=1)

        output_meter_layout = QVBoxLayout()
        self.output_meter = LevelMeter("OUT", show_scale=True)
        self.output_meter.setAccessibleName("Output level meter")
        self.output_meter.setFixedWidth(50)
        output_meter_layout.addWidget(self.output_meter)
        middle_layout.addLayout(output_meter_layout)

        main_layout.addLayout(middle_layout, stretch=1)

        control_group = QGroupBox("Processing")
        control_stack = QVBoxLayout(control_group)
        control_stack.setSpacing(SPACING_NORMAL)
        control_stack.setContentsMargins(
            MARGIN_PANEL, SPACING_NORMAL, MARGIN_PANEL, MARGIN_PANEL
        )

        self.action_layout = QGridLayout()
        self.action_layout.setSpacing(SPACING_NORMAL)
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.start_btn.setMinimumWidth(132)
        self.start_btn.clicked.connect(self._start_processing)

        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet(DESTRUCTIVE_ACTION_BUTTON_STYLE)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(132)
        self.stop_btn.clicked.connect(self._stop_processing)

        self.auto_eq_button = QPushButton("Auto-EQ")
        self.auto_eq_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.auto_eq_button.setMinimumWidth(108)
        self.auto_eq_button.setToolTip(
            "Automatically calibrate EQ to your voice and microphone\n"
            "Select target curve, read passage, and get professional tuning"
        )
        self.auto_eq_button.clicked.connect(self._on_auto_eq_clicked)

        self.auto_voice_setup_button = QPushButton("Auto Voice Setup")
        self.auto_voice_setup_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.auto_voice_setup_button.setMinimumWidth(148)
        self.auto_voice_setup_button.setToolTip(
            "Record room noise and speech, then calibrate EQ, gate/VAD,\n"
            "de-esser, and compressor in one pass"
        )
        self.auto_voice_setup_button.clicked.connect(self._on_auto_voice_setup_clicked)

        self._undo_auto_eq_button = QPushButton("Undo")
        self._undo_auto_eq_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self._undo_auto_eq_button.setEnabled(False)
        self._undo_auto_eq_button.setMinimumWidth(108)
        self._undo_auto_eq_button.setToolTip(
            "Undo the most recent processing-configuration edit (Ctrl+Z)"
        )
        self._undo_auto_eq_button.clicked.connect(self.undo_configuration)

        self.bypass_checkbox = QCheckBox("Master Bypass")
        self.bypass_checkbox.setToolTip(
            "Bypass all processing (pass audio through unchanged)"
        )
        self.bypass_checkbox.toggled.connect(self._on_bypass_toggled)

        self.raw_monitor_checkbox = QCheckBox("Raw Monitor")
        self.raw_monitor_checkbox.setToolTip(
            "Diagnostic path: bypass pre-filter + DSP chain and use clean output write path"
        )
        self.raw_monitor_checkbox.toggled.connect(self._on_raw_monitor_toggled)
        self._action_layout_widgets = (
            self.start_btn,
            self.stop_btn,
            self.auto_eq_button,
            self.auto_voice_setup_button,
            self._undo_auto_eq_button,
            self.bypass_checkbox,
            self.raw_monitor_checkbox,
        )
        control_stack.addLayout(self.action_layout)

        self.health_decision_layout = QGridLayout()
        self.health_decision_layout.setSpacing(SPACING_NORMAL)
        self.health_decision_layout.setContentsMargins(0, 2, 0, 0)

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
        control_stack.addLayout(self.health_decision_layout)

        self.health_layout = QGridLayout()
        self.health_layout.setSpacing(SPACING_NORMAL)
        self.health_layout.setContentsMargins(0, 2, 0, 0)

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
        control_stack.addLayout(self.health_layout)
        main_layout.addWidget(control_group)

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
        self.setTabOrder(self.input_combo, self.output_combo)
        self.setTabOrder(self.output_combo, self.input_channel_mode_combo)
        self.setTabOrder(
            self.input_channel_mode_combo,
            self.input_cleanup_mode_combo,
        )
        self.setTabOrder(self.input_cleanup_mode_combo, self.refresh_btn)
        self.setTabOrder(self.start_btn, self.stop_btn)
        self.setTabOrder(self.stop_btn, self.auto_eq_button)
        self.setTabOrder(self.auto_eq_button, self.auto_voice_setup_button)
        self.setTabOrder(self.auto_voice_setup_button, self._undo_auto_eq_button)
        self.setTabOrder(self._undo_auto_eq_button, self.bypass_checkbox)
        self.setTabOrder(self.bypass_checkbox, self.raw_monitor_checkbox)

    @staticmethod
    def _remove_grid_widgets(layout: QGridLayout, widgets: tuple[QWidget, ...]) -> None:
        for widget in widgets:
            layout.removeWidget(widget)
        for column in range(10):
            layout.setColumnStretch(column, 0)
        for row in range(10):
            layout.setRowStretch(row, 0)

    def _update_responsive_layouts(self, width: int) -> None:
        if not hasattr(self, "_health_layout_widgets"):
            return
        self._layout_main_splitter(width < self.VERTICAL_SPLITTER_BREAKPOINT)
        compact = width < self.COMPACT_LAYOUT_BREAKPOINT
        if compact == self._responsive_layout_compact:
            return
        self._responsive_layout_compact = compact
        self._layout_device_controls(compact)
        self._layout_processing_actions(compact)
        self._layout_health_chips(compact)

    def _layout_main_splitter(self, vertical: bool) -> None:
        if vertical == self._splitter_is_vertical:
            return
        self._splitter_is_vertical = vertical
        if vertical:
            self.main_splitter.setOrientation(Qt.Orientation.Vertical)
            available = max(520, self.height() - 260)
            self.main_splitter.setSizes([available // 2, available - available // 2])
            return
        self.main_splitter.setOrientation(Qt.Orientation.Horizontal)
        self.main_splitter.setSizes(
            self._clamp_splitter_sizes(self.config.main_splitter_sizes or [])
        )

    def _layout_device_controls(self, compact: bool) -> None:
        widgets = self._device_layout_widgets
        self._remove_grid_widgets(self.device_layout, widgets)
        (
            input_label,
            input_combo,
            output_label,
            output_combo,
            mode_label,
            mode_combo,
            cleanup_label,
            cleanup_combo,
            refresh_button,
        ) = widgets
        if compact:
            self.device_layout.addWidget(input_label, 0, 0)
            self.device_layout.addWidget(input_combo, 0, 1, 1, 3)
            self.device_layout.addWidget(output_label, 0, 4)
            self.device_layout.addWidget(output_combo, 0, 5, 1, 3)
            self.device_layout.addWidget(mode_label, 1, 0)
            self.device_layout.addWidget(mode_combo, 1, 1, 1, 2)
            self.device_layout.addWidget(cleanup_label, 1, 3)
            self.device_layout.addWidget(cleanup_combo, 1, 4, 1, 2)
            self.device_layout.addWidget(refresh_button, 1, 7)
            self.device_layout.setColumnStretch(1, 1)
            self.device_layout.setColumnStretch(5, 1)
            return

        for column, widget in enumerate(widgets):
            self.device_layout.addWidget(widget, 0, column)
        self.device_layout.setColumnStretch(1, 1)
        self.device_layout.setColumnStretch(3, 1)

    def _layout_processing_actions(self, compact: bool) -> None:
        widgets = self._action_layout_widgets
        self._remove_grid_widgets(self.action_layout, widgets)
        action_buttons = widgets[:5]
        bypass_checkbox, raw_monitor_checkbox = widgets[5:]
        if compact:
            for index, button in enumerate(action_buttons):
                row, column = divmod(index, 3)
                self.action_layout.addWidget(button, row, column)
            for column in range(3):
                self.action_layout.setColumnStretch(column, 1)
            self.action_layout.addWidget(
                bypass_checkbox,
                1,
                3,
                Qt.AlignmentFlag.AlignVCenter,
            )
            self.action_layout.addWidget(
                raw_monitor_checkbox,
                1,
                4,
                Qt.AlignmentFlag.AlignVCenter,
            )
            return

        for column, button in enumerate(action_buttons):
            self.action_layout.addWidget(button, 0, column)
        self.action_layout.setColumnStretch(5, 1)
        self.action_layout.addWidget(
            bypass_checkbox,
            0,
            6,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.action_layout.addWidget(
            raw_monitor_checkbox,
            0,
            7,
            Qt.AlignmentFlag.AlignVCenter,
        )

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

    def _create_tab_page(self, widgets: list[QWidget]) -> QScrollArea:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 8, 8)
        layout.setSpacing(SPACING_SECTION)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def _create_noise_suppression_group(self) -> QGroupBox:
        group = QGroupBox("Noise Suppression")
        layout = QVBoxLayout(group)
        layout.setSpacing(SPACING_NORMAL)

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

        self.rnnoise_checkbox = QCheckBox("Enable Noise Suppression")
        self.rnnoise_checkbox.setChecked(True)
        self.rnnoise_checkbox.setToolTip(
            "Enable or disable the selected suppression backend."
        )
        self.rnnoise_checkbox.toggled.connect(self._on_rnnoise_toggled)
        layout.addWidget(self.rnnoise_checkbox)

        strength_layout = QHBoxLayout()
        strength_label = QLabel("Strength:")
        strength_layout.addWidget(strength_label)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.setTickInterval(25)
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

        info_label = QLabel(
            "Backend choice affects cleanup quality, CPU use, and latency.\n"
            "Packaged builds use bundled, integrity-checked neural model assets."
        )
        info_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
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
        label.setText(text)
        label.setStyleSheet(status_chip_style(state))

    def _reset_health_labels(self) -> None:
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
        self.dropped_label.setToolTip(DROPPED_DIAGNOSTICS_TOOLTIP)
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
        target = mode if self._is_valid_input_channel_mode(mode) else "average"
        for index in range(self.input_channel_mode_combo.count()):
            if self.input_channel_mode_combo.itemData(index) == target:
                self.input_channel_mode_combo.setCurrentIndex(index)
                return
        self.input_channel_mode_combo.setCurrentIndex(0)

    def _apply_input_channel_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_channel_mode(mode) else "average"
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
        for index in range(self.input_cleanup_mode_combo.count()):
            if self.input_cleanup_mode_combo.itemData(index) == target:
                self.input_cleanup_mode_combo.setCurrentIndex(index)
                return
        self.input_cleanup_mode_combo.setCurrentIndex(0)

    def _apply_input_cleanup_mode(self, mode: str) -> None:
        target = mode if self._is_valid_input_cleanup_mode(mode) else "off"
        try:
            if hasattr(self.processor, "set_input_cleanup_mode"):
                self.processor.set_input_cleanup_mode(target)
        except Exception:
            logger.debug("Failed to apply input cleanup mode", exc_info=True)

    def _schedule_ui_state_save(self) -> None:
        self._ui_state_timer.start(200)

    def _save_ui_state(self) -> None:
        if (
            hasattr(self, "main_splitter")
            and self.main_splitter.orientation() == Qt.Orientation.Horizontal
        ):
            self.config.main_splitter_sizes = self._clamp_splitter_sizes(
                self.main_splitter.sizes()
            )
        if hasattr(self, "control_tabs"):
            self.config.main_control_tab_index = int(self.control_tabs.currentIndex())
        save_config(self.config)

    def _clamp_splitter_sizes(self, sizes: list[int]) -> list[int]:
        total = max(sum(int(size) for size in sizes), self.width() - 150, 760)
        default_left = min(
            max(self.LEFT_PANE_MIN_WIDTH, total // 3), total - self.RIGHT_PANE_MIN_WIDTH
        )
        default_right = max(total - default_left, self.RIGHT_PANE_MIN_WIDTH)
        if len(sizes) != 2:
            return [default_left, default_right]

        left = int(sizes[0])
        min_left = self.LEFT_PANE_MIN_WIDTH
        max_left = max(min_left, total - self.RIGHT_PANE_MIN_WIDTH)
        left = max(min_left, min(left, max_left))
        right = max(total - left, self.RIGHT_PANE_MIN_WIDTH)
        return [left, right]

    def _restore_ui_state(self) -> None:
        if hasattr(self, "control_tabs"):
            index = self.config.main_control_tab_index
            if 0 <= index < self.control_tabs.count():
                self.control_tabs.setCurrentIndex(index)
        if hasattr(self, "main_splitter"):
            if self.main_splitter.orientation() == Qt.Orientation.Horizontal:
                self.main_splitter.setSizes(
                    self._clamp_splitter_sizes(self.config.main_splitter_sizes or [])
                )

    def _on_main_control_tab_changed(self, _: int) -> None:
        self._schedule_ui_state_save()

    def _on_splitter_moved(self, _: int, __: int) -> None:
        self._schedule_ui_state_save()

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
        exit_action.triggered.connect(self.close)
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

        save_preset_action = QAction("&Save Preset...", self)
        save_preset_action.setShortcut("Ctrl+S")
        save_preset_action.triggered.connect(self._save_preset)
        presets_menu.addAction(save_preset_action)

        load_preset_action = QAction("&Load Preset...", self)
        load_preset_action.setShortcut("Ctrl+O")
        load_preset_action.triggered.connect(self._load_preset)
        presets_menu.addAction(load_preset_action)

        presets_menu.addSeparator()

        # Built-in presets submenu
        builtin_menu = presets_menu.addMenu("&Built-in Presets")
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
        custom_presets = list_presets()
        custom_names = tuple(name for name, _filepath in custom_presets)
        startup_preset_id = _normalize_startup_preset_id(
            self.config.startup_preset, custom_names
        )

        # "Last Used" option (default, checked if startup_preset is empty)
        last_used_action = QAction("Last Used", self)
        last_used_action.setCheckable(True)
        last_used_action.setData("")
        last_used_action.setChecked(startup_preset_id == "")
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
            action.setChecked(startup_preset_id == preset_id)
            action.triggered.connect(
                lambda checked, item_id=preset_id: self._set_startup_preset(item_id)
            )
            startup_menu.addAction(action)

        # Separator
        startup_menu.addSeparator()

        # Custom presets
        for name, filepath in custom_presets:
            action = QAction(name, self)
            action.setCheckable(True)
            preset_id = _startup_custom_id(name)
            action.setData(preset_id)
            action.setChecked(startup_preset_id == preset_id)
            action.triggered.connect(
                lambda checked, item_id=preset_id: self._set_startup_preset(item_id)
            )
            startup_menu.addAction(action)

        options_menu.addSeparator()

        device_preset_menu = options_menu.addMenu("Preset for Current &Route")
        assert device_preset_menu is not None
        self._device_preset_actions: dict[str, QAction] = {}

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

        if custom_presets:
            device_preset_menu.addSeparator()
        for name, filepath in custom_presets:
            preset_id = _startup_custom_id(filepath.name)
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked, item_id=preset_id: self._bind_current_route_preset(
                    item_id
                )
            )
            device_preset_menu.addAction(action)
            self._device_preset_actions[preset_id] = action

        device_preset_menu.aboutToShow.connect(self._update_device_preset_menu)

        options_menu.addSeparator()

        self.use_measured_latency_action = QAction(
            "Use Measured Latency Compensation", self
        )
        self.use_measured_latency_action.setCheckable(True)
        self.use_measured_latency_action.setChecked(self.config.use_measured_latency)
        self.use_measured_latency_action.toggled.connect(
            self._on_use_measured_latency_toggled
        )
        options_menu.addAction(self.use_measured_latency_action)

        latency_calibration_action = QAction("Run Latency Calibration...", self)
        latency_calibration_action.triggered.connect(
            self._on_latency_calibration_clicked
        )
        options_menu.addAction(latency_calibration_action)

    def _set_startup_preset(self, preset_id: str):
        """Set the startup preset and update checked states.

        Args:
            preset_id: Stable preset ID to load on startup (empty string = Last Used)
        """
        # Update config
        self.config.startup_preset = preset_id

        # Save config
        save_config(self.config)

        # Show status message
        if preset_id:
            preset_name = _startup_preset_display_name(preset_id)
            self.status_bar.showMessage(f"Startup preset set to {preset_name}", 5000)
        else:
            self.status_bar.showMessage("Startup preset set to Last Used", 5000)

        # Update checked states of all startup preset actions
        # Get the Options menu
        menubar = self.menuBar()
        assert menubar is not None
        for action in menubar.actions():
            options_menu = action.menu()
            if options_menu is not None and options_menu.title() == "&Options":
                for menu_action in options_menu.actions():
                    startup_menu = menu_action.menu()
                    if (
                        startup_menu is not None
                        and startup_menu.title() == "Startup &Preset..."
                    ):
                        # Update checked state for all actions in the startup menu
                        for preset_action in startup_menu.actions():
                            if preset_action.isCheckable():
                                preset_action.setChecked(
                                    str(preset_action.data() or "") == preset_id
                                )
                        break
                break

    def _maybe_show_first_run_setup(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
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

    def _on_auto_apply_device_presets_toggled(self, checked: bool) -> None:
        self.config.auto_apply_device_presets = bool(checked)
        save_config(self.config)
        if checked:
            self._apply_bound_preset_for_current_route()

    def _bind_current_route_preset(self, preset_id: str) -> None:
        route_key = self._current_device_route_key()
        if route_key is None:
            self.status_bar.showMessage(
                "Connect and select both route devices before binding a preset", 5000
            )
            return
        self.config.device_preset_bindings[route_key] = DevicePresetBinding(
            preset_id=preset_id,
            provenance="explicit_user",
        )
        save_config(self.config)
        self._update_device_preset_menu()
        self.status_bar.showMessage(
            f"Bound {_startup_preset_display_name(preset_id)} to this device route",
            5000,
        )

    def _clear_current_route_preset(self) -> None:
        route_key = self._current_device_route_key()
        if route_key is None:
            return
        removed = self.config.device_preset_bindings.pop(route_key, None)
        if removed is not None:
            save_config(self.config)
        self._update_device_preset_menu()
        self.status_bar.showMessage("Cleared the preset binding for this route", 4000)

    def _update_device_preset_menu(self) -> None:
        route_key = self._current_device_route_key()
        binding = (
            self.config.device_preset_bindings.get(route_key)
            if route_key is not None
            else None
        )
        selected_id = binding.preset_id if binding is not None else ""
        self._clear_route_preset_action.setChecked(not selected_id)
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
            self._apply_preset(preset, preset_key=key)
            return True
        if not preset_id.startswith(STARTUP_CUSTOM_PREFIX):
            return False
        custom_id = preset_id[len(STARTUP_CUSTOM_PREFIX) :]
        candidates = [
            (name, filepath)
            for name, filepath in list_presets()
            if filepath.name == custom_id or name == custom_id
        ]
        if len(candidates) != 1:
            return False
        _name, filepath = candidates[0]
        preset = load_preset(filepath)
        self._apply_preset(preset)
        self.current_preset_path = filepath
        self.config.last_preset = str(filepath)
        save_config(self.config)
        return True

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
            self.status_bar.showMessage(
                f"Route preset: {_startup_preset_display_name(binding.preset_id)}",
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
        profile = self.config.latency_calibration_profiles.get(key)
        if profile is not None:
            return profile

        legacy_key = self._legacy_latency_profile_key()
        profile = self.config.latency_calibration_profiles.get(legacy_key)
        if profile is not None:
            self.config.latency_calibration_profiles[key] = profile
            if (
                legacy_key != key
                and legacy_key in self.config.latency_calibration_profiles
            ):
                del self.config.latency_calibration_profiles[legacy_key]
            save_config(self.config)
        return profile

    def _sync_latency_profile_for_current_devices(
        self, profile: LatencyCalibrationProfile
    ) -> str:
        key = self._latency_profile_key()
        if key is None:
            raise ValueError(
                "Stable endpoint identity is unavailable for this duplicate-name route"
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
            if self._refresh_latency_profile_engine(profile):
                save_config(self.config)
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
        save_config(self.config)
        self._apply_latency_compensation_for_current_devices()
        mode = "enabled" if enabled else "disabled"
        self.status_bar.showMessage(f"Measured latency compensation {mode}", 4000)

    def _on_latency_calibration_clicked(self) -> bool:
        if self._latency_profile_key() is None:
            self.status_bar.showMessage(
                "Cannot persist calibration: duplicate device names lack stable endpoint IDs",
                6000,
            )
            return False
        profile = self._current_latency_profile()
        if profile is not None and self._refresh_latency_profile_engine(profile):
            save_config(self.config)
        existing_profile = profile.to_dict() if profile is not None else None

        dialog = LatencyCalibrationDialog(self, existing_profile=existing_profile)
        dialog.calibration_saved.connect(self._on_latency_calibration_saved)
        dialog.calibration_reset.connect(self._on_latency_calibration_reset)
        self._calibration_dialog_open = True
        try:
            dialog.exec()
        finally:
            self._calibration_dialog_open = False
        return self._current_latency_profile() is not None

    def _on_latency_calibration_saved(self, profile_data: dict):
        profile = LatencyCalibrationProfile.from_dict(profile_data)
        try:
            self._sync_latency_profile_for_current_devices(profile)
        except ValueError as error:
            self.status_bar.showMessage(str(error), 6000)
            return
        save_config(self.config)
        self._apply_latency_compensation_for_current_devices()
        route_latency_ms = float(profile.route_latency_ms)
        if route_latency_ms <= 0.0:
            route_latency_ms = float(profile.measured_round_trip_ms)
        self.status_bar.showMessage(
            f"Measured route latency saved for current device pair ({route_latency_ms:.1f} ms)",
            5000,
        )

    def _on_latency_calibration_reset(self):
        key = self._latency_profile_key()
        if key is None:
            self._apply_latency_compensation_for_current_devices()
            return
        legacy_key = self._legacy_latency_profile_key()
        removed = False
        for candidate in {key, legacy_key}:
            if candidate in self.config.latency_calibration_profiles:
                del self.config.latency_calibration_profiles[candidate]
                removed = True
        if removed:
            save_config(self.config)
        self._apply_latency_compensation_for_current_devices()
        self.status_bar.showMessage(
            "Latency calibration reset for current device pair", 4000
        )

    def _refresh_devices(self):
        """Refresh the device lists."""
        previous_input = (
            self.config.last_input_device_identity
            or self._combo_device_identity(self.input_combo)
        )
        previous_output = (
            self.config.last_output_device_identity
            or self._combo_device_identity(self.output_combo)
        )

        # Block signals to prevent spurious config saves during refresh
        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)
        if "input_channel_mode_combo" in self.__dict__:
            self.input_channel_mode_combo.blockSignals(True)
        if "input_cleanup_mode_combo" in self.__dict__:
            self.input_cleanup_mode_combo.blockSignals(True)

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
                    fallback_index = self._default_combo_index(self.input_combo)
                    if fallback_index >= 0:
                        self.input_combo.setCurrentIndex(fallback_index)
                    if previous_input.name:
                        self.status_bar.showMessage(
                            f"Previous input device '{previous_input.name}' is disconnected; "
                            "using the default until it returns"
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
                    fallback_index = self._default_combo_index(self.output_combo)
                    if fallback_index >= 0:
                        self.output_combo.setCurrentIndex(fallback_index)
                    if previous_output.name:
                        self.status_bar.showMessage(
                            f"Previous output device '{previous_output.name}' is disconnected; "
                            "using the default until it returns"
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

        # Restore signals
        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)

        if config_dirty:
            save_config(self.config)

    def _restore_from_config(self):
        """Restore settings from loaded config."""
        restored_count = 0
        config_dirty = False

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
                self.status_bar.showMessage(
                    f"Previous input device '{input_identity.name}' is disconnected; "
                    "using the default until it returns"
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
                self.status_bar.showMessage(
                    f"Previous output device '{output_identity.name}' is disconnected; "
                    "using the default until it returns"
                )

        # Restore preset (startup preset takes priority over last used)
        preset_loaded = False

        # Check startup preset first
        if self.config.startup_preset:
            custom_presets = list_presets()
            preset_id = _normalize_startup_preset_id(
                self.config.startup_preset,
                tuple(name for name, _filepath in custom_presets),
            )
            if preset_id != self.config.startup_preset:
                self.config.startup_preset = preset_id
                config_dirty = True
            preset_name = _startup_preset_display_name(preset_id)
            # Try built-in presets
            if preset_id.startswith(STARTUP_BUILTIN_PREFIX):
                preset_key = preset_id[len(STARTUP_BUILTIN_PREFIX) :]
                if preset_key in BUILTIN_PRESETS:
                    preset = BUILTIN_PRESETS[preset_key]
                    self._apply_preset(preset, preset_key=preset_key)
                    self.status_bar.showMessage(f"Startup preset: {preset_name}", 5000)
                    preset_loaded = True
            # Try custom presets
            elif preset_id.startswith(STARTUP_CUSTOM_PREFIX):
                custom_name = preset_id[len(STARTUP_CUSTOM_PREFIX) :]
                for name, filepath in custom_presets:
                    if name == custom_name:
                        try:
                            preset = load_preset(filepath)
                            self._apply_preset(preset)
                            self.status_bar.showMessage(
                                f"Startup preset: {preset_name}", 5000
                            )
                            preset_loaded = True
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
            # Try legacy unresolved custom display name.
            else:
                for name, filepath in custom_presets:
                    if name == preset_id:
                        try:
                            preset = load_preset(filepath)
                            self._apply_preset(preset)
                            self.status_bar.showMessage(
                                f"Startup preset: {preset_name}", 5000
                            )
                            preset_loaded = True
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

        # Fall back to last_preset if startup_preset not set or not found
        if not preset_loaded and self.config.last_preset:
            try:
                # Check if it's a built-in preset
                if self.config.last_preset.startswith("builtin:"):
                    preset_key = self.config.last_preset[8:]  # Remove "builtin:" prefix
                    if preset_key in BUILTIN_PRESETS:
                        preset = BUILTIN_PRESETS[preset_key]
                        self._apply_preset(preset)
                        # Re-save config to persist preset for next session
                        save_config(self.config)
                        restored_count += 1
                    else:
                        self.status_bar.showMessage(
                            f"Previous preset '{preset_key}' not found, starting with defaults"
                        )
                        self.config.last_preset = ""
                        save_config(self.config)
                else:
                    # It's a file path
                    preset_path = Path(self.config.last_preset)
                    if preset_path.exists():
                        preset = load_preset(preset_path)
                        self._apply_preset(preset)
                        self.current_preset_path = preset_path
                        # Re-save config to persist preset for next session
                        save_config(self.config)
                        restored_count += 1
                    else:
                        self.status_bar.showMessage(
                            "Previous preset file not found, starting with defaults"
                        )
                        self.config.last_preset = ""
                        save_config(self.config)
            except (IOError, OSError, ValueError, json.JSONDecodeError) as e:
                logger.warning("Preset restore failed", exc_info=True)
                self.status_bar.showMessage(f"Failed to restore preset: {e}")
                self.config.last_preset = ""
                save_config(self.config)

        # Show appropriate status message
        if restored_count == 0:
            self.status_bar.showMessage("Ready")
        elif restored_count < 3:
            self.status_bar.showMessage(
                "Restored partial settings (some devices/presets unavailable)"
            )
        else:
            self.status_bar.showMessage("Restored settings from previous session")

        self._restore_ui_state()
        self._apply_latency_compensation_for_current_devices()
        if "input_channel_mode_combo" in self.__dict__:
            input_channel_mode = getattr(self.config, "input_channel_mode", "average")
            self._select_input_channel_mode(input_channel_mode)
            self._apply_input_channel_mode(input_channel_mode)
        if "input_cleanup_mode_combo" in self.__dict__:
            input_cleanup_mode = getattr(self.config, "input_cleanup_mode", "off")
            self._select_input_cleanup_mode(input_cleanup_mode)
            self._apply_input_cleanup_mode(input_cleanup_mode)

        # Route-specific DSP is more specific than the generic startup/last-used
        # preset and is intentionally applied only after both endpoints resolve.
        self._apply_bound_preset_for_current_route()

        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)
        if "input_channel_mode_combo" in self.__dict__:
            self.input_channel_mode_combo.blockSignals(False)
        if "input_cleanup_mode_combo" in self.__dict__:
            self.input_cleanup_mode_combo.blockSignals(False)

        if config_dirty:
            save_config(self.config)

    def _on_device_changed(self):
        """Handle device selection change - save to config."""
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
            save_config(self.config)
            self._apply_latency_compensation_for_current_devices()
            self._apply_bound_preset_for_current_route()

    def _on_input_channel_mode_changed(self):
        """Persist and apply the selected input channel mixdown mode."""
        if not hasattr(self, "config"):
            return
        mode = self.input_channel_mode_combo.currentData()
        if not self._is_valid_input_channel_mode(mode):
            mode = "average"
        self.config.input_channel_mode = mode
        self._apply_input_channel_mode(mode)
        save_config(self.config)

    def _on_input_cleanup_mode_changed(self):
        """Persist and apply the selected adaptive input cleanup mode."""
        if not hasattr(self, "config"):
            return
        mode = self.input_cleanup_mode_combo.currentData()
        if not self._is_valid_input_cleanup_mode(mode):
            mode = "off"
        self.config.input_cleanup_mode = mode
        self._apply_input_cleanup_mode(mode)
        save_config(self.config)

    def _start_processing(self):
        """Start audio processing."""
        if self.processor.is_running():
            if DEBUG:
                logger.debug("Start processing clicked, but processor already running")
            return

        input_device = self._device_selection_to_name(self.input_combo) or None
        output_device = self._device_selection_to_name(self.output_combo) or None
        self._apply_input_channel_mode(
            getattr(self.config, "input_channel_mode", "average")
        )
        self._apply_input_cleanup_mode(
            getattr(self.config, "input_cleanup_mode", "off")
        )

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
            # Unmute output after starting processing (in case it was muted by calibration)
            if DEBUG:
                logger.debug("Unmuting output after processing start")
            self.processor.set_output_mute(False)
            self.status_bar.showMessage(f"Processing: {result}")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)
            self._stream_recovery.mark_processing_started()
            if DEBUG:
                logger.debug("Processing started: %s", result)
        except Exception as e:
            logger.exception("Start processing failed")
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
            QMessageBox.critical(
                self,
                "Error Starting Processing",
                f"Failed to start audio processing:\n\n{e}\n\n{guidance}",
            )
            self.status_bar.showMessage(f"Error: {e}")

    def _stop_processing(self):
        """Stop audio processing."""
        if not self.processor.is_running():
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
            self._stream_recovery.mark_processing_stopped()
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
        if DEBUG:
            logger.debug("Calibration dialog closed, result=%s", dialog.result())
            is_running = self.processor.is_running()
            logger.debug("After calibration - processor running=%s", is_running)

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
        )
        self._configuration_history.initialize(snapshot)
        self._current_value_provenance = dict(snapshot.to_preset().value_provenance)
        self._history_ready = True
        self._update_history_actions()

    def _connect_configuration_history_inputs(self) -> None:
        """Observe processing controls and coalesce one user gesture."""
        self.eq_panel.configurationEditStarted.connect(
            self._begin_configuration_transaction
        )
        self.eq_panel.configurationEditFinished.connect(
            self._end_configuration_transaction
        )
        for slider in self.findChildren(QSlider):
            slider.valueChanged.connect(self._queue_configuration_snapshot)
        for spinbox in self.findChildren(QSpinBox):
            spinbox.valueChanged.connect(self._queue_configuration_snapshot)
        for spinbox in self.findChildren(QDoubleSpinBox):
            spinbox.valueChanged.connect(self._queue_configuration_snapshot)
        for combo in self.findChildren(QComboBox):
            combo.currentIndexChanged.connect(self._queue_configuration_snapshot)
        for button in self.findChildren(QAbstractButton):
            button.toggled.connect(self._queue_configuration_snapshot)

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
        if recorded:
            self._current_value_provenance = dict(snapshot.to_preset().value_provenance)
        self._update_history_actions()
        return recorded

    def _restore_configuration_snapshot(
        self,
        snapshot: ConfigurationSnapshot,
    ) -> None:
        """Restore one validated snapshot without creating a new entry."""
        preset = snapshot.to_preset()
        previous_preset = self._get_current_preset()
        previous_provenance = dict(self._current_value_provenance)
        self._history_replaying = True
        self._history_timer.stop()
        try:
            self._apply_preset(preset, require_exact=True)
            self._current_value_provenance = dict(preset.value_provenance)
        except Exception:
            try:
                self._apply_preset(previous_preset, require_exact=True)
                self._current_value_provenance = previous_provenance
            except Exception:
                logger.exception(
                    "Configuration-history rollback failed after restore error"
                )
            raise
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

        existing_presets = list_presets()
        existing_names = [name.lower() for name, _ in existing_presets]
        if preset_name.lower() in existing_names:
            confirm_reply = QMessageBox.question(
                self,
                "Overwrite Preset?",
                f"Preset '{preset_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm_reply != QMessageBox.StandardButton.Yes:
                return

        preset = self._get_current_preset()
        preset.name = preset_name
        preset.description = description
        preset.version = __version__
        save_preset(preset)
        QMessageBox.information(
            self,
            "Preset Saved",
            f"Preset '{preset_name}' saved successfully.",
        )

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
            question=f"Save these auto-EQ settings as preset '{preset_name}'?",
            preset_name=preset_name,
            description=f"Auto-generated EQ settings using {target_curve.title()} target curve",
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

    def undo_auto_eq(self):
        """Compatibility alias for the former one-slot Auto-EQ undo."""
        self.undo_configuration()

    def _on_bypass_toggled(self, checked):
        """Handle bypass toggle."""
        self.processor.set_bypass(checked)
        if checked:
            self.status_bar.showMessage(
                "Master bypass enabled - audio passing through unchanged"
            )
        else:
            self.status_bar.showMessage("Processing active")

    def _on_raw_monitor_toggled(self, checked):
        """Handle raw monitor toggle."""
        self.processor.set_raw_monitor_enabled(checked)
        if checked:
            self.status_bar.showMessage(
                "Raw monitor enabled - skipping pre-filter and DSP chain"
            )
        else:
            self.status_bar.showMessage("Raw monitor disabled")

    def _on_rnnoise_toggled(self, checked):
        """Handle RNNoise toggle."""
        self.processor.set_rnnoise_enabled(checked)

    def _on_strength_changed(self, value: int):
        """Handle RNNoise strength slider change."""
        strength = value / 100.0  # Convert 0-100 to 0.0-1.0
        self.strength_label.setText(f"{value}%")
        self.processor.set_rnnoise_strength(strength)

    def _on_model_changed(self, index: int):
        """Handle noise model selection change."""
        model_id = self.model_combo.itemData(index)
        if not model_id:
            return

        try:
            success = self.processor.set_noise_model(model_id)
            if not success:
                # Model switch failed - show error and revert
                QMessageBox.warning(
                    self,
                    "Model Switch Failed",
                    f"Could not switch to {self.model_combo.currentText()}.\n\n"
                    f"The model may not be available in this build.\n"
                    f"Reverting to previous model.",
                )
                # Revert to RNNoise using find-by-ID loop (not hardcoded index)
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i) == "rnnoise":
                        self.model_combo.setCurrentIndex(i)
                        return
            else:
                self._set_noise_suppression_latency_label(model_id)
                self.status_bar.showMessage(
                    f"Switched to {self.model_combo.currentText()}"
                )
        except Exception as e:
            # Unexpected error - show detailed dialog with guidance
            logger.exception("Model switch error")
            QMessageBox.critical(
                self,
                "Error Switching Model",
                f"An unexpected error occurred while switching noise models:\n\n"
                f"{type(e).__name__}: {e}\n\n"
                f"This may indicate a problem with the selected neural backend.\n"
                f"Please try:\n"
                f"1. Restarting the application\n"
                f"2. Using RNNoise model as fallback\n"
                f"3. Verifying the bundled model/runtime assets",
            )
            # Revert to RNNoise using find-by-ID loop (NOT hardcoded index)
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == "rnnoise":
                    self.model_combo.setCurrentIndex(i)
                    return

    def _update_meters(self):
        """Update level meters from processor (called by timer)."""
        if self.processor.is_running():
            input_rms = self.processor.get_input_rms_db()
            input_peak = self.processor.get_input_peak_db()
            output_rms = self.processor.get_output_rms_db()
            output_peak = self.processor.get_output_peak_db()
            gr_db = self.processor.get_compressor_gain_reduction_db()
            deesser_gr_db = self.processor.get_deesser_gain_reduction_db()

            # Update meters
            self.input_meter.set_levels(input_rms, input_peak)
            self.output_meter.set_levels(output_rms, output_peak)
            self.compressor_panel.update_gain_reduction(gr_db)
            if hasattr(self, "deesser_panel"):
                self.deesser_panel.update_gain_reduction(deesser_gr_db)

            # Update compressor current release time
            try:
                self.compressor_panel._update_current_release()
            except Exception:
                pass

            # Update auto makeup gain meters (if enabled)
            try:
                if hasattr(self, "compressor_panel"):
                    auto_makeup_enabled = (
                        self.processor.get_compressor_auto_makeup_enabled()
                    )
                    if auto_makeup_enabled and hasattr(
                        self.compressor_panel, "update_auto_makeup_meters"
                    ):
                        current_lufs = self.processor.get_compressor_current_lufs()
                        makeup_gain = (
                            self.processor.get_compressor_current_makeup_gain()
                        )
                        self.compressor_panel.update_auto_makeup_meters(
                            current_lufs, makeup_gain
                        )
            except Exception:
                logger.debug("Auto makeup meter update error", exc_info=True)

            # Update VAD confidence meter (if VAD is available)
            try:
                vad_prob = self.processor.get_vad_probability()
                self.gate_panel.update_vad_confidence(vad_prob)
            except (AttributeError, Exception):
                # VAD not available in this build
                pass

        else:
            self._last_backend_warning = None
            self._reset_health_labels()

    def _update_diagnostics(self):
        """Update slower diagnostics and service recovery."""
        if not self.processor.is_running():
            self._stream_recovery.mark_processing_stopped()
            self._last_backend_warning = None
            self._reset_health_labels()
            return

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
        output_short_write_dropped = int(
            diagnostics.get("output_short_write_dropped_samples", 0) or 0
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
        previous_gate_chatter_count = int(
            getattr(self, "_last_gate_chatter_event_count", gate_chatter_count) or 0
        )
        new_gate_chatter_observed = gate_chatter_count > previous_gate_chatter_count
        self._last_gate_chatter_event_count = gate_chatter_count
        gate_auto_relax_active = bool(diagnostics.get("gate_auto_relax_active", False))
        rt_overflows = diagnostics.get("rt_buffer_overflow_count", 0)
        input_callback_errors = diagnostics.get("input_callback_error_count", 0)
        output_callback_errors = diagnostics.get("output_callback_error_count", 0)
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
                dropped == 0
                and underrun_streak == 0
                and not new_underruns_observed
                and lock_contention == 0
                and non_finite == 0
                and output_short_write_dropped == 0
                and rt_overflows == 0
                and input_callback_errors == 0
                and output_callback_errors == 0
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
            if reason:
                recovery_bits.append("RECENT")
            self._set_health_chip(
                self.recovery_diag_label,
                f"Recovery: {' '.join(recovery_bits)}",
                "warn"
                if restart_count or output_recovery_count or reason
                else ("info" if suppressed else "ok"),
            )
        except Exception:
            logger.debug("Diagnostic label update failed", exc_info=True)

    def _service_stream_recovery(
        self,
        *,
        diagnostics: dict,
        input_rms: float,
        output_rms: float,
        output_buf: int,
    ) -> None:
        """Service UI-side and Rust-side stream recovery."""
        if self._stream_recovery.maybe_recover_output_stall(
            input_rms=input_rms,
            output_rms=output_rms,
            output_buf=output_buf,
            calibration_dialog_open=self._calibration_dialog_open,
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
                            f"Auto-recovery failed: {err_msg}",
                            6000,
                        )
                    else:
                        self.status_bar.showMessage(
                            "Auto-recovery failed",
                            6000,
                        )
        except Exception:
            logger.debug("Rust recovery service failed", exc_info=True)

    def _recover_output_path(self):
        """Best-effort output recovery: unmute + restart with selected devices."""
        try:
            self.processor.set_output_mute(False)
            self.processor.stop()
            result = start_processor_for_route(
                self.processor,
                self._combo_device_identity(self.input_combo),
                self._combo_device_identity(self.output_combo),
            )
            self.processor.set_output_mute(False)
            self.status_bar.showMessage(
                f"Recovered output path automatically: {result}",
                4000,
            )
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

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AudioForge",
            f"<h2>AudioForge v{__version__}</h2>"
            "<p>Low-latency microphone audio processor</p>"
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
                strength=self.strength_slider.value() / 100.0,
                model=self.model_combo.currentData() or "rnnoise",
            ),
            deesser=DeEsserSettings(**deesser_settings),
            compressor=CompressorSettings(**compressor_settings),
            limiter=LimiterSettings(**limiter_settings),
            bypass=self.bypass_checkbox.isChecked(),
            value_provenance=dict(self._current_value_provenance),
        )

    def _apply_preset(
        self,
        preset: Preset,
        preset_key: str | None = None,
        *,
        require_exact: bool = False,
    ):
        """Apply a preset to the UI and processor.

        Args:
            preset: Preset object to apply
            preset_key: Optional key for built-in presets (e.g., "voice", "bass_cut")
        """
        requested_model = getattr(preset.rnnoise, "model", "rnnoise")
        requested_model_index = next(
            (
                index
                for index in range(self.model_combo.count())
                if self.model_combo.itemData(index) == requested_model
            ),
            -1,
        )
        model_fallback_warning: str | None = None
        if require_exact and requested_model_index < 0:
            raise RuntimeError(
                f"Noise model {requested_model!r} is not present in this runtime"
            )

        # Apply gate settings (including VAD mode and auto-threshold)
        self.gate_panel.set_settings(
            {
                "enabled": preset.gate.enabled,
                "threshold_db": preset.gate.threshold_db,
                "attack_ms": preset.gate.attack_ms,
                "release_ms": preset.gate.release_ms,
                "gate_mode": preset.gate.gate_mode,
                "vad_threshold": preset.gate.vad_threshold,
                "vad_hold_time_ms": preset.gate.vad_hold_time_ms,
                "vad_pre_gain": preset.gate.vad_pre_gain,
                "auto_threshold_enabled": preset.gate.auto_threshold_enabled,  # v1.6.0+
                "gate_margin_db": preset.gate.gate_margin_db,  # v1.6.0+
            }
        )

        # Apply EQ settings
        self.eq_panel.set_settings(
            {
                "enabled": preset.eq.enabled,
                "schema_version": preset.eq.schema_version,
                "bands": [band.to_dict() for band in preset.eq.bands],
            }
        )

        # Apply RNNoise settings
        self.rnnoise_checkbox.setChecked(preset.rnnoise.enabled)
        self.processor.set_rnnoise_enabled(preset.rnnoise.enabled)

        # Apply strength
        strength_percent = int(preset.rnnoise.strength * 100)
        self.strength_slider.setValue(strength_percent)
        self.processor.set_rnnoise_strength(preset.rnnoise.strength)

        # Apply model selection
        model = requested_model
        model_found = False
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model:
                # Block signals to prevent duplicate model initialization
                # setCurrentIndex triggers currentIndexChanged which calls set_noise_model,
                # so we need to block it here since we'll call it directly below
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(i)
                self.model_combo.blockSignals(False)
                model_found = True
                # Try to set model, handle errors gracefully
                try:
                    success = self.processor.set_noise_model(model)
                    if success:
                        self._set_noise_suppression_latency_label(model)
                    else:
                        if require_exact:
                            raise RuntimeError(f"Noise model {model!r} is unavailable")
                        # Model switch failed - show warning and use RNNoise
                        logger.warning(
                            "Failed to switch to %s from preset; using RNNoise", model
                        )
                        self.status_bar.showMessage(
                            f"Note: Preset specifies {model} but not available, using RNNoise",
                            5000,
                        )
                        model_fallback_warning = (
                            f"{model} was unavailable; using RNNoise"
                        )
                        # Fall back to RNNoise using find-by-ID loop (NOT hardcoded index)
                        for j in range(self.model_combo.count()):
                            if self.model_combo.itemData(j) == "rnnoise":
                                self.model_combo.blockSignals(True)
                                self.model_combo.setCurrentIndex(j)
                                self.model_combo.blockSignals(False)
                                self.processor.set_noise_model("rnnoise")
                                self._set_noise_suppression_latency_label("rnnoise")
                                break
                except Exception:
                    if require_exact:
                        raise
                    # Unexpected error - log and fall back
                    logger.exception("Error switching model in preset")
                    self.status_bar.showMessage(
                        "Error loading preset model, using RNNoise", 5000
                    )
                    model_fallback_warning = f"{model} failed to load; using RNNoise"
                    # Fall back to RNNoise using find-by-ID loop (NOT hardcoded index)
                    for j in range(self.model_combo.count()):
                        if self.model_combo.itemData(j) == "rnnoise":
                            self.model_combo.blockSignals(True)
                            self.model_combo.setCurrentIndex(j)
                            self.model_combo.blockSignals(False)
                            self.processor.set_noise_model("rnnoise")
                            self._set_noise_suppression_latency_label("rnnoise")
                            break
                break

        if not model_found:
            logger.warning("Preset model %r not found in available models", model)
            rnnoise_index = next(
                (
                    index
                    for index in range(self.model_combo.count())
                    if self.model_combo.itemData(index) == "rnnoise"
                ),
                -1,
            )
            if rnnoise_index < 0:
                raise RuntimeError("RNNoise fallback is not present in this runtime")
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(rnnoise_index)
            self.model_combo.blockSignals(False)
            if not self.processor.set_noise_model("rnnoise"):
                raise RuntimeError("RNNoise fallback could not be activated")
            self._set_noise_suppression_latency_label("rnnoise")
            self.status_bar.showMessage(
                f"Note: Preset model {model!r} is unavailable; using RNNoise",
                5000,
            )
            model_fallback_warning = f"{model} was unavailable; using RNNoise"

        # Apply de-esser settings
        self.deesser_panel.set_settings(
            {
                "enabled": preset.deesser.enabled,
                "auto_enabled": preset.deesser.auto_enabled,
                "auto_amount": preset.deesser.auto_amount,
                "low_cut_hz": preset.deesser.low_cut_hz,
                "high_cut_hz": preset.deesser.high_cut_hz,
                "threshold_db": preset.deesser.threshold_db,
                "ratio": preset.deesser.ratio,
                "attack_ms": preset.deesser.attack_ms,
                "release_ms": preset.deesser.release_ms,
                "max_reduction_db": preset.deesser.max_reduction_db,
            }
        )

        # Apply compressor settings
        self.compressor_panel.set_compressor_settings(
            {
                "enabled": preset.compressor.enabled,
                "threshold_db": preset.compressor.threshold_db,
                "ratio": preset.compressor.ratio,
                "attack_ms": preset.compressor.attack_ms,
                "release_ms": preset.compressor.release_ms,
                "makeup_gain_db": preset.compressor.makeup_gain_db,
                "adaptive_release": preset.compressor.adaptive_release,
                "base_release_ms": preset.compressor.base_release_ms,
                "auto_makeup_enabled": preset.compressor.auto_makeup_enabled,
                "target_lufs": preset.compressor.target_lufs,
                "sidechain_highpass_enabled": preset.compressor.sidechain_highpass_enabled,
            }
        )

        # Apply limiter settings
        self.compressor_panel.set_limiter_settings(
            {
                "enabled": preset.limiter.enabled,
                "ceiling_db": preset.limiter.ceiling_db,
                "release_ms": preset.limiter.release_ms,
                "careful_output_enabled": preset.limiter.careful_output_enabled,
            }
        )

        # Apply bypass
        self.bypass_checkbox.setChecked(preset.bypass)
        self.processor.set_bypass(preset.bypass)

        # Preserve migration provenance across apply, undo, and redo. A later
        # manual edit marks only changed paths explicit when it is committed.
        normalized_preset = Preset.from_dict(preset.to_dict())
        self._current_value_provenance = dict(normalized_preset.value_provenance)

        if model_fallback_warning is None:
            self.status_bar.showMessage(f"Loaded preset: {preset.name}")
        else:
            self.status_bar.showMessage(
                f"Loaded preset: {preset.name} ({model_fallback_warning})",
                6000,
            )

        history_ready = bool(self.__dict__.get("_history_ready", False))
        history_replaying = bool(self.__dict__.get("_history_replaying", False))
        if history_ready and not history_replaying:
            self._commit_pending_configuration_snapshot(
                label=f"Loaded preset ({preset.name})",
                source="preset",
                provenance=self._current_value_provenance,
            )

        # Save to config if preset_key provided (built-in preset)
        if preset_key:
            self.config.last_preset = f"builtin:{preset_key}"
            save_config(self.config)
            self.current_preset_path = None  # Built-in, not a file

    def _save_preset(self):
        """Save current settings as a preset."""
        # Get preset name from user
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Enter preset name:", text="My Preset"
        )
        if not ok or not name.strip():
            return

        # Get current settings
        preset = self._get_current_preset()
        preset.name = name.strip()

        # Get description (optional)
        description, ok = QInputDialog.getText(
            self,
            "Save Preset",
            "Enter description (optional):",
        )
        if ok:
            preset.description = description.strip()

        # Save to file
        try:
            filepath = save_preset(preset)
            self.status_bar.showMessage(f"Preset saved: {filepath}")
            QMessageBox.information(
                self, "Preset Saved", f"Preset '{name}' saved to:\n{filepath}"
            )
        except (IOError, OSError, ValueError) as e:
            logger.warning("Preset save failed", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save preset:\n{e}\n\nCheck you have write permission to the presets folder.",
            )

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
                imports_dir = get_preset_imports_dir()
                imported_path = imports_dir / requested_path.name
                if requested_path.resolve(strict=True) != imported_path.resolve(
                    strict=False
                ):
                    validate_preset_file_size(requested_path)
                    shutil.copy2(requested_path, imported_path)
                preset = load_preset(imported_path)
                preset_path = imported_path
            self._apply_preset(preset)
            # Save to config for persistence
            self.current_preset_path = preset_path
            self.config.last_preset = str(preset_path)
            save_config(self.config)
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
        self.config.window_geometry = {
            "x": self.x(),
            "y": self.y(),
            "width": self.width(),
            "height": self.height(),
        }
        self._save_ui_state()

        if self.processor.is_running():
            self.processor.stop()
        event.accept()


def run_app():
    """Run the AudioForge application."""
    return run_qt_app(MainWindow)


if __name__ == "__main__":
    sys.exit(run_app())
