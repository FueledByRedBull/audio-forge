"""
Main window for AudioForge application

Adapted from Spectral Workbench project.

DEBUG: Added terminal logging for processor state tracking
"""

# Enable debug logging
DEBUG = True

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
    QSizePolicy,
    QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
import sys
import json
from pathlib import Path

from .gate_panel import GatePanel
from .eq_panel import EQPanel
from .compressor_panel import CompressorPanel
from .level_meter import LevelMeter
from .calibration_dialog import CalibrationDialog
from .layout_constants import SPACING_SECTION, SPACING_NORMAL
from .. import AudioProcessor, list_input_devices, list_output_devices
from ..config import (
    Preset,
    GateSettings,
    EQSettings,
    RNNoiseSettings,
    CompressorSettings,
    LimiterSettings,
    PresetValidationError,
    save_preset,
    load_preset,
    get_presets_dir,
    BUILTIN_PRESETS,
    save_config,
    load_config,
    AppConfig,
)


class MainWindow(QMainWindow):
    """Main application window for AudioForge."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioForge - Microphone Audio Processor")

        # Create audio processor
        self.processor = AudioProcessor()

        # Load configuration
        self.config = load_config()
        self.current_preset_path = None

        # Auto-EQ undo state (single-level undo)
        self._pre_auto_eq_state = None
        self._undo_auto_eq_button = None

        # Set up UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()

        # Populate device lists
        self._refresh_devices()

        # Connect device change signals for persistence
        self.input_combo.currentIndexChanged.connect(self._on_device_changed)
        self.output_combo.currentIndexChanged.connect(self._on_device_changed)

        # Restore settings from config
        self._restore_from_config()

        # Meter update timer (60 FPS)
        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._update_meters)
        self.meter_timer.start(16)  # ~60 FPS

        # Set initial window size to fit all content
        self.resize(1100, 850)
        self.setMinimumSize(1000, 850)

    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(SPACING_SECTION)  # Consistent spacing between major sections

        # Warning banner for missing audio devices (hidden by default)
        self.device_warning_banner = QLabel(
            "⚠ Warning: No audio devices detected. Check your audio drivers and connections."
        )
        self.device_warning_banner.setStyleSheet(
            "QLabel { background-color: #FFA500; color: black; padding: 10px; "
            "font-weight: bold; border-radius: 5px; }"
        )
        self.device_warning_banner.setVisible(False)
        main_layout.addWidget(self.device_warning_banner)

        # Top: Device selection
        device_group = QGroupBox("Audio Devices")
        device_layout = QHBoxLayout(device_group)
        device_layout.setSpacing(SPACING_NORMAL)  # Consistent spacing for device controls

        # Input device
        device_layout.addWidget(QLabel("Input:"))
        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(150)
        device_layout.addWidget(self.input_combo, stretch=1)

        # Output device
        device_layout.addWidget(QLabel("Output:"))
        self.output_combo = QComboBox()
        self.output_combo.setMinimumWidth(150)
        device_layout.addWidget(self.output_combo, stretch=1)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_devices)
        device_layout.addWidget(refresh_btn)

        main_layout.addWidget(device_group)

        # Middle: Control panels in horizontal layout with meters
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(SPACING_NORMAL)  # Consistent spacing for control panels

        # Input meter (far left)
        input_meter_layout = QVBoxLayout()
        self.input_meter = LevelMeter("IN", show_scale=True)
        self.input_meter.setFixedWidth(55)
        input_meter_layout.addWidget(self.input_meter)
        middle_layout.addLayout(input_meter_layout)

        # Control panels
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(SPACING_NORMAL)  # Consistent spacing between panel groups

        # ============================================================
        # LEFT SIDE: All panels in a single scrollable container
        # This prevents overlap issues between Gate, RNNoise, and Compressor
        # ============================================================
        
        # Create a container widget for all left panels
        left_container = QWidget()
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 12, 0)  # 12px right margin for scrollbar
        left_container_layout.setSpacing(SPACING_NORMAL)

        # 1. Noise Gate panel (direct widget, no nested scroll area)
        self.gate_panel = GatePanel(self.processor)
        left_container_layout.addWidget(self.gate_panel)

        # 2. RNNoise panel
        rnnoise_group = QGroupBox("RNNoise (ML Noise Suppression)")
        rnnoise_layout = QVBoxLayout(rnnoise_group)
        rnnoise_layout.setSpacing(SPACING_NORMAL)

        self.rnnoise_checkbox = QCheckBox("Enable RNNoise")
        self.rnnoise_checkbox.setChecked(True)
        self.rnnoise_checkbox.setToolTip(
            "ML-based noise suppression.\n"
            "Removes background noise while preserving voice."
        )
        self.rnnoise_checkbox.toggled.connect(self._on_rnnoise_toggled)
        rnnoise_layout.addWidget(self.rnnoise_checkbox)

        rnnoise_info = QLabel(
            "ML-based noise suppression trained on\n"
            "voice and noise samples."
        )
        rnnoise_info.setStyleSheet("color: gray; font-size: 11px;")
        rnnoise_info.setWordWrap(True)
        rnnoise_layout.addWidget(rnnoise_info)

        # Strength slider
        strength_layout = QHBoxLayout()

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)  # 0-100 integer steps
        self.strength_slider.setValue(100)     # Default: 100% (full processing)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.setTickInterval(25)  # 0, 25, 50, 75, 100
        self.strength_slider.setToolTip("RNNoise processing strength (0% = dry, 100% = fully processed)")
        strength_layout.addWidget(self.strength_slider)

        self.strength_label = QLabel("100%")
        self.strength_label.setFixedWidth(50)
        strength_layout.addWidget(self.strength_label)

        rnnoise_layout.addLayout(strength_layout)

        # Connect slider to update handler
        self.strength_slider.valueChanged.connect(self._on_strength_changed)

        # Model selection dropdown
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI Model:"))

        self.model_combo = QComboBox()
        # Populate from processor
        for model_id, display_name in self.processor.list_noise_models():
            self.model_combo.addItem(display_name, model_id)
        self.model_combo.setToolTip(
            "RNNoise: Low latency (~10ms), good quality\n"
            "DeepFilterNet: Low Latency (~10ms), better quality than RNNoise"
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()

        rnnoise_layout.addLayout(model_layout)

        # Latency info label (updates based on model)
        self.rnnoise_latency_label = QLabel("Latency: ~10ms (RNNoise)")
        self.rnnoise_latency_label.setStyleSheet("color: gray; font-size: 11px;")
        rnnoise_layout.addWidget(self.rnnoise_latency_label)

        left_container_layout.addWidget(rnnoise_group)

        # 3. Compressor panel (direct widget, no nested scroll area)
        self.compressor_panel = CompressorPanel(self.processor)
        left_container_layout.addWidget(self.compressor_panel)

        # Add stretch at bottom to push content up when there's extra space
        left_container_layout.addStretch()

        # Wrap everything in ONE scroll area - this is the key fix!
        # Using a single scroll area prevents the overlap issues
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidget(left_container)
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        # Fixed width for left panel - doesn't stretch with window
        # Expanding height - fills available vertical space, scrollbar appears when overflow
        left_scroll_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_scroll_area.setFixedWidth(450)  # Account for ~20px scrollbar width

        panels_layout.addWidget(left_scroll_area)

        # ============================================================
        # RIGHT SIDE: EQ panel (stretches to fill remaining space)
        # ============================================================
        self.eq_panel = EQPanel(self.processor)
        panels_layout.addWidget(self.eq_panel, stretch=1)

        middle_layout.addLayout(panels_layout, stretch=1)

        # Output meter (far right)
        output_meter_layout = QVBoxLayout()
        self.output_meter = LevelMeter("OUT", show_scale=True)
        self.output_meter.setFixedWidth(55)
        output_meter_layout.addWidget(self.output_meter)
        middle_layout.addLayout(output_meter_layout)

        main_layout.addLayout(middle_layout, stretch=1)

        # Bottom: Control buttons
        control_group = QGroupBox("Processing Control")
        control_layout = QHBoxLayout(control_group)

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }"
        )
        self.start_btn.clicked.connect(self._start_processing)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }"
        )
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_processing)
        control_layout.addWidget(self.stop_btn)

        # Auto-EQ button
        self.auto_eq_button = QPushButton("Auto-EQ")
        self.auto_eq_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }"
        )
        self.auto_eq_button.setToolTip(
            "Automatically calibrate EQ to your voice and microphone\n"
            "Select target curve, read passage, and get professional tuning"
        )
        self.auto_eq_button.clicked.connect(self._on_auto_eq_clicked)
        control_layout.addWidget(self.auto_eq_button)

        # Undo Auto-EQ button (disabled initially)
        self._undo_auto_eq_button = QPushButton("Undo Auto-EQ")
        self._undo_auto_eq_button.setEnabled(False)
        self._undo_auto_eq_button.setToolTip("Restore EQ settings from before last auto-EQ")
        self._undo_auto_eq_button.clicked.connect(self.undo_auto_eq)
        control_layout.addWidget(self._undo_auto_eq_button)

        control_layout.addStretch()

        # Latency display
        self.latency_label = QLabel("Latency: -- ms")
        self.latency_label.setStyleSheet(
            "QLabel { background-color: #333; color: #0f0; padding: 5px 10px; "
            "border-radius: 3px; font-family: monospace; font-size: 12px; }"
        )
        self.latency_label.setToolTip(
            "Total processing latency (RNNoise buffering + output buffer)\n"
            "DSP time shows smoothed actual processing time per 10ms chunk"
        )
        control_layout.addWidget(self.latency_label)

        # DSP Time display
        # NOTE: DSP label removed - now combined with latency display

        # Buffer Health display
        self.buffer_label = QLabel("Buffer: --")
        self.buffer_label.setStyleSheet(
            "QLabel { background-color: #333; color: #0f0; padding: 5px 10px; "
            "border-radius: 3px; font-family: monospace; font-size: 12px; }"
        )
        self.buffer_label.setToolTip(
            "Buffer health indicator\n"
            "OK: Input + RNNoise buffers are healthy\n"
            "WARN: Buffers accumulating (may cause drift)\n"
            "BAD: Significant buffer buildup"
        )
        control_layout.addWidget(self.buffer_label)

        # Dropped Samples display
        self.dropped_label = QLabel("Dropped: 0")
        self.dropped_label.setStyleSheet(
            "QLabel { background-color: #333; color: #0f0; padding: 5px 10px; "
            "border-radius: 3px; font-family: monospace; font-size: 12px; }"
        )
        self.dropped_label.setToolTip(
            "Dropped samples counter\n"
            "Shows total audio samples dropped due to buffer overflow\n"
            "Green: 0 dropped (healthy)\n"
            "Yellow: > 0 dropped (buffer underrun detected)\n"
            "Right-click to reset counter"
        )
        self.dropped_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dropped_label.customContextMenuRequested.connect(self._on_dropped_context_menu)
        control_layout.addWidget(self.dropped_label)

        control_layout.addSpacing(20)

        self.bypass_checkbox = QCheckBox("Master Bypass")
        self.bypass_checkbox.setToolTip("Bypass all processing (pass audio through unchanged)")
        self.bypass_checkbox.toggled.connect(self._on_bypass_toggled)
        control_layout.addWidget(self.bypass_checkbox)

        main_layout.addWidget(control_group)

    def _setup_menubar(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

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

        # Presets menu
        presets_menu = menubar.addMenu("&Presets")

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
        for key, preset in BUILTIN_PRESETS.items():
            action = QAction(preset.name, self)
            action.setToolTip(preset.description)
            action.triggered.connect(lambda checked, p=preset, k=key: self._apply_preset(p, preset_key=k))
            builtin_menu.addAction(action)

        presets_menu.addSeparator()

        # Open presets folder
        open_folder_action = QAction("Open Presets &Folder", self)
        open_folder_action.triggered.connect(self._open_presets_folder)
        presets_menu.addAction(open_folder_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

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

    def _refresh_devices(self):
        """Refresh the device lists."""
        # Block signals to prevent spurious config saves during refresh
        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)

        self.input_combo.clear()
        self.output_combo.clear()

        input_found = False
        output_found = False

        # Get input devices
        try:
            input_devices = list_input_devices()
            input_found = len(input_devices) > 0
            for device in input_devices:
                label = f"{device.name}" + (" (Default)" if device.is_default else "")
                self.input_combo.addItem(label, device.name)
                if device.is_default:
                    self.input_combo.setCurrentIndex(self.input_combo.count() - 1)
        except (RuntimeError, OSError) as e:
            self.input_combo.addItem(f"Error: {e}")
            print(f"Device enumeration failed: {type(e).__name__}: {e}")

        # Get output devices
        try:
            output_devices = list_output_devices()
            output_found = len(output_devices) > 0
            vb_cable_index = -1
            default_index = -1

            for i, device in enumerate(output_devices):
                label = f"{device.name}" + (" (Default)" if device.is_default else "")
                self.output_combo.addItem(label, device.name)

                # Highlight VB Audio Cable if found
                name_lower = device.name.lower()
                if "cable" in name_lower or "vb-audio" in name_lower or "virtual" in name_lower:
                    vb_cable_index = i

                if device.is_default:
                    default_index = i

            # Prefer VB Audio Cable, then default
            if vb_cable_index >= 0:
                self.output_combo.setCurrentIndex(vb_cable_index)
            elif default_index >= 0:
                self.output_combo.setCurrentIndex(default_index)

        except (RuntimeError, OSError) as e:
            self.output_combo.addItem(f"Error: {e}")
            print(f"Device enumeration failed: {type(e).__name__}: {e}")

        # Update warning banner visibility and text
        if not input_found and not output_found:
            self.device_warning_banner.setText(
                "⚠ Warning: No audio devices detected. Check your audio drivers and connections."
            )
            self.device_warning_banner.setVisible(True)
        elif not input_found:
            self.device_warning_banner.setText(
                "⚠ Warning: No input devices detected. Check your microphone connections."
            )
            self.device_warning_banner.setVisible(True)
        elif not output_found:
            self.device_warning_banner.setText(
                "⚠ Warning: No output devices detected. Check your audio output connections."
            )
            self.device_warning_banner.setVisible(True)
        else:
            self.device_warning_banner.setVisible(False)

        # Restore signals
        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)

    def _restore_from_config(self):
        """Restore settings from loaded config."""
        restored_count = 0

        # Restore input device
        if self.config.last_input_device:
            found = False
            for i in range(self.input_combo.count()):
                if self.input_combo.itemData(i) == self.config.last_input_device:
                    self.input_combo.setCurrentIndex(i)
                    found = True
                    restored_count += 1
                    break
            if not found:
                self.status_bar.showMessage(
                    f"Previous input device '{self.config.last_input_device}' not found, using default"
                )
                # Clear missing device from config
                self.config.last_input_device = ""
                save_config(self.config)

        # Restore output device
        if self.config.last_output_device:
            found = False
            for i in range(self.output_combo.count()):
                if self.output_combo.itemData(i) == self.config.last_output_device:
                    self.output_combo.setCurrentIndex(i)
                    found = True
                    restored_count += 1
                    break
            if not found:
                self.status_bar.showMessage(
                    f"Previous output device '{self.config.last_output_device}' not found, using default"
                )
                # Clear missing device from config
                self.config.last_output_device = ""
                save_config(self.config)

        # Restore preset
        if self.config.last_preset:
            try:
                # Check if it's a built-in preset
                if self.config.last_preset.startswith("builtin:"):
                    preset_key = self.config.last_preset[8:]  # Remove "builtin:" prefix
                    if preset_key in BUILTIN_PRESETS:
                        preset = BUILTIN_PRESETS[preset_key]
                        self._apply_preset(preset)
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
                        restored_count += 1
                    else:
                        self.status_bar.showMessage(
                            "Previous preset file not found, starting with defaults"
                        )
                        self.config.last_preset = ""
                        save_config(self.config)
            except (IOError, OSError, ValueError, json.JSONDecodeError) as e:
                print(f"Preset restore failed: {type(e).__name__}: {e}")
                self.status_bar.showMessage(f"Failed to restore preset: {e}")
                self.config.last_preset = ""
                save_config(self.config)

        # Show appropriate status message
        if restored_count == 0:
            self.status_bar.showMessage("Ready")
        elif restored_count < 3:
            self.status_bar.showMessage("Restored partial settings (some devices/presets unavailable)")
        else:
            self.status_bar.showMessage("Restored settings from previous session")

        # Restore window geometry if saved
        if self.config.window_geometry:
            geom = self.config.window_geometry
            width = geom.get('width', 1000)
            height = geom.get('height', 750)
            x = geom.get('x', 100)
            y = geom.get('y', 100)
            self.restoreGeometry(
                bytes(f"{width}x{height}+{x}+{y}", 'utf-8')
            )

    def _on_device_changed(self):
        """Handle device selection change - save to config."""
        if hasattr(self, 'config'):  # Check config is initialized
            self.config.last_input_device = self.input_combo.currentData() or ""
            self.config.last_output_device = self.output_combo.currentData() or ""
            save_config(self.config)

    def _start_processing(self):
        """Start audio processing."""
        if self.processor.is_running():
            if DEBUG:
                print("[MAIN] Start processing clicked, but processor already running")
            return

        input_device = self.input_combo.currentData()
        output_device = self.output_combo.currentData()

        if DEBUG:
            print(f"[MAIN] Starting processing - Input: {input_device or '(default)'}, Output: {output_device or '(default)'}")

        try:
            result = self.processor.start(input_device, output_device)
            # Unmute output after starting processing (in case it was muted by calibration)
            if DEBUG:
                print("[MAIN] Unmuting output (set_output_mute=False)")
            self.processor.set_output_mute(False)
            self.status_bar.showMessage(f"Processing: {result}")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)
            if DEBUG:
                print(f"[MAIN] Processing started: {result}")
        except Exception as e:
            print(f"Start processing failed: {type(e).__name__}: {e}")
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
                f"Failed to start audio processing:\n\n{e}\n\n{guidance}"
            )
            self.status_bar.showMessage(f"Error: {e}")

    def _stop_processing(self):
        """Stop audio processing."""
        if not self.processor.is_running():
            if DEBUG:
                print("[MAIN] Stop processing clicked, but processor not running")
            return

        if DEBUG:
            print("[MAIN] Stopping processing...")

        try:
            self.processor.stop()
            self.status_bar.showMessage("Processing stopped")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
            if DEBUG:
                print("[MAIN] Processing stopped")
        except RuntimeError as e:
            print(f"Stop processing failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to stop processing:\n{e}")

    def _on_auto_eq_clicked(self):
        """Open Auto-EQ calibration dialog."""
        if DEBUG:
            print("[MAIN] Auto-EQ button clicked - opening calibration dialog")

        # Capture state before showing dialog
        self.capture_pre_auto_eq_state()

        dialog = CalibrationDialog(self)
        # Connect signal to handle auto-EQ completion (preset save, undo button enable)
        dialog.auto_eq_applied.connect(self.on_auto_eq_applied)
        dialog.exec()  # Modal dialog - blocks until user closes
        if DEBUG:
            print(f"[MAIN] Calibration dialog closed, result={dialog.result()}")
            is_running = self.processor.is_running()
            print(f"[MAIN] After calibration - processor running={is_running}")

    def capture_pre_auto_eq_state(self):
        """Capture current EQ state before auto-EQ application."""
        self._pre_auto_eq_state = self.eq_panel.capture_state()

    def on_auto_eq_applied(self, target_curve: str):
        """
        Handle auto-EQ application completion.

        Shows undo button, prompts for preset save.

        Args:
            target_curve: The target curve used ('broadcast', 'podcast', etc.)
        """
        from ..config import generate_auto_eq_preset_name, save_preset, list_presets

        # Reset curve overlay mode (hide "Current vs New" comparison)
        self.eq_panel.reset_curve_overlay()

        # Show undo button
        if self._undo_auto_eq_button:
            self._undo_auto_eq_button.setEnabled(True)

        # Prompt to save as preset
        preset_name = generate_auto_eq_preset_name(target_curve)

        reply = QMessageBox.question(
            self,
            "Save Auto-EQ as Preset?",
            f"Save these auto-EQ settings as preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Check if preset already exists
            existing_presets = list_presets()
            existing_names = [name.lower() for name, _ in existing_presets]

            if preset_name.lower() in existing_names:
                # Confirm overwrite
                confirm_reply = QMessageBox.question(
                    self,
                    "Overwrite Preset?",
                    f"Preset '{preset_name}' already exists. Overwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if confirm_reply != QMessageBox.StandardButton.Yes:
                    return  # User cancelled

            # Create preset from current settings
            from ..config import Preset, CompressorSettings, LimiterSettings, GateSettings

            preset = Preset(
                name=preset_name,
                description=f"Auto-generated EQ settings using {target_curve.title()} target curve",
                version="1.6.0",
                gate=GateSettings(**self.gate_panel.get_settings()),
                eq=EQSettings(**self.eq_panel.get_settings()),
                rnnoise=RNNoiseSettings(
                    enabled=self.rnnoise_checkbox.isChecked(),
                    strength=self.strength_slider.value() / 100.0,
                    model=self.model_combo.currentData() or 'rnnoise',
                ),
                compressor=CompressorSettings(**self.compressor_panel.get_compressor_settings()),
                limiter=LimiterSettings(**self.compressor_panel.get_limiter_settings()),
                bypass=self.bypass_checkbox.isChecked(),
            )

            filepath = save_preset(preset)
            QMessageBox.information(
                self,
                "Preset Saved",
                f"Preset '{preset_name}' saved successfully."
            )

    def undo_auto_eq(self):
        """Undo last auto-EQ application and restore previous state."""
        if self._pre_auto_eq_state is None:
            QMessageBox.information(self, "No Undo", "No auto-EQ to undo.")
            return

        # Restore previous state
        self.eq_panel.restore_state(self._pre_auto_eq_state)

        # Clear saved state
        self._pre_auto_eq_state = None

        # Disable undo button
        if self._undo_auto_eq_button:
            self._undo_auto_eq_button.setEnabled(False)

        # Show toast message
        toast = QLabel("Auto-EQ undone")
        toast.setStyleSheet("""
            background-color: #333;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
        """)
        toast.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        toast.show()

        # Position toast near center of window
        toast.move(self.mapToGlobal(self.rect().center()) - toast.rect().center())

        # Auto-hide toast after 2 seconds
        QTimer.singleShot(2000, toast.deleteLater)

    def _on_bypass_toggled(self, checked):
        """Handle bypass toggle."""
        self.processor.set_bypass(checked)
        if checked:
            self.status_bar.showMessage("Master bypass enabled - audio passing through unchanged")
        else:
            self.status_bar.showMessage("Processing active")

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
                    f"Reverting to previous model."
                )
                # Revert to RNNoise using find-by-ID loop (not hardcoded index)
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i) == "rnnoise":
                        self.model_combo.setCurrentIndex(i)
                        return
            else:
                # Update latency display based on model
                if model_id == "deepfilter":
                    self.rnnoise_latency_label.setText("Latency: ~10ms (DeepFilterNet LL)")
                else:
                    self.rnnoise_latency_label.setText("Latency: ~10ms (RNNoise)")
                self.status_bar.showMessage(f"Switched to {self.model_combo.currentText()}")
        except Exception as e:
            # Unexpected error - show detailed dialog with guidance
            print(f"Model switch error: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self,
                "Error Switching Model",
                f"An unexpected error occurred while switching noise models:\n\n"
                f"{type(e).__name__}: {e}\n\n"
                f"This may indicate a problem with the DeepFilterNet integration.\n"
                f"Please try:\n"
                f"1. Restarting the application\n"
                f"2. Using RNNoise model as fallback\n"
                f"3. Ensuring df.dll is available (falls back to passthrough if not found)"
            )
            # Revert to RNNoise using find-by-ID loop (NOT hardcoded index)
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == "rnnoise":
                    self.model_combo.setCurrentIndex(i)
                    return

    def _update_meters(self):
        """Update level meters from processor (called by timer)."""
        if self.processor.is_running():
            # Get levels from processor
            input_rms = self.processor.get_input_rms_db()
            input_peak = self.processor.get_input_peak_db()
            output_rms = self.processor.get_output_rms_db()
            output_peak = self.processor.get_output_peak_db()
            gr_db = self.processor.get_compressor_gain_reduction_db()

            latency_ms = self.processor.get_latency_ms()

            # Get DSP performance metrics
            dsp_time_ms = self.processor.get_dsp_time_smoothed_ms()
            input_buf = self.processor.get_input_buffer_smoothed_samples()
            output_buf = self.processor.get_output_buffer_samples()
            rnnoise_buf = self.processor.get_buffer_smoothed_samples()

            # Update meters
            self.input_meter.set_levels(input_rms, input_peak)
            self.output_meter.set_levels(output_rms, output_peak)
            self.compressor_panel.update_gain_reduction(gr_db)

            # Update compressor current release time
            try:
                self.compressor_panel._update_current_release()
            except Exception:
                pass

            # Update auto makeup gain meters (if enabled)
            try:
                if hasattr(self, 'compressor_panel'):
                    auto_makeup_enabled = self.processor.get_compressor_auto_makeup_enabled()
                    if auto_makeup_enabled and hasattr(self.compressor_panel, 'update_auto_makeup_meters'):
                        current_lufs = self.processor.get_compressor_current_lufs()
                        makeup_gain = self.processor.get_compressor_current_makeup_gain()
                        self.compressor_panel.update_auto_makeup_meters(current_lufs, makeup_gain)
            except Exception as e:
                print(f"Auto makeup meter update error: {e}")

            # Update VAD confidence meter (if VAD is available)
            try:
                vad_prob = self.processor.get_vad_probability()
                self.gate_panel.update_vad_confidence(vad_prob)
            except (AttributeError, Exception):
                # VAD not available in this build
                pass

            # Update combined latency display
            self.latency_label.setText(f"Latency: ~{latency_ms:.0f}ms (DSP: {dsp_time_ms:.1f}ms)")

            # Update buffer health display (Input + RNNoise only)
            # Healthy: < 960 samples (2 frames), Warning: < 1920 (4 frames), Bad: >= 1920
            pipeline_buf = input_buf + rnnoise_buf
            if pipeline_buf < 960:
                buf_status = "OK"
                buf_color = "#0f0"  # Green
            elif pipeline_buf < 1920:
                buf_status = "WARN"
                buf_color = "#ff0"  # Yellow
            else:
                buf_status = "BAD"
                buf_color = "#f00"  # Red

            self.buffer_label.setStyleSheet(
                f"QLabel {{ background-color: #333; color: {buf_color}; padding: 5px 10px; "
                "border-radius: 3px; font-family: monospace; font-size: 12px; }"
            )
            self.buffer_label.setText(f"Buffer: {buf_status} ({pipeline_buf})")

            # Update dropped samples display
            dropped = self.processor.get_dropped_samples()
            if dropped == 0:
                dropped_color = "#0f0"  # Green - no drops
            else:
                dropped_color = "#ff0"  # Yellow - drops detected
            self.dropped_label.setStyleSheet(
                f"QLabel {{ background-color: #333; color: {dropped_color}; padding: 5px 10px; "
                "border-radius: 3px; font-family: monospace; font-size: 12px; }"
            )
            self.dropped_label.setText(f"Dropped: {dropped}")
        else:
            self.latency_label.setText("Latency: -- ms")
            self.buffer_label.setText("Buffer: --")
            self.buffer_label.setStyleSheet(
                "QLabel { background-color: #333; color: #0f0; padding: 5px 10px; "
                "border-radius: 3px; font-family: monospace; font-size: 12px; }"
            )
            self.dropped_label.setText("Dropped: --")
            self.dropped_label.setStyleSheet(
                "QLabel { background-color: #333; color: #0f0; padding: 5px 10px; "
                "border-radius: 3px; font-family: monospace; font-size: 12px; }"
            )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AudioForge",
            "<h2>AudioForge v1.5.0</h2>"
            "<p>Low-latency microphone audio processor</p>"
            "<p>Inspired by SteelSeries GG Sonar ClearCast AI</p>"
            "<h3>Processing Chain:</h3>"
            "<p>Mic → Pre-Filter → Gate → AI Noise (RNNoise/DeepFilter) → EQ → Comp → Limiter → Output</p>"
            "<h3>Features:</h3>"
            "<ul>"
            "<li>Noise Gate with IIR envelope follower</li>"
            "<li>Silero VAD-assisted gate mode</li>"
            "<li>AI Noise Suppression: RNNoise or DeepFilterNet (experimental)</li>"
            "<li>10-band parametric EQ</li>"
            "<li>Compressor with soft-knee gain reduction</li>"
            "<li>Hard limiter for clipping prevention</li>"
            "<li>OBS-style visual level meters</li>"
            "<li>VB Audio Cable routing for Discord/etc</li>"
            "</ul>"
            "<p><b>Target latency:</b> &lt;30ms</p>"
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
        eq_settings = self.eq_panel.get_settings()
        compressor_settings = self.compressor_panel.get_compressor_settings()
        limiter_settings = self.compressor_panel.get_limiter_settings()

        return Preset(
            name="Custom",
            description="User-defined preset",
            gate=GateSettings(**gate_settings),
            eq=EQSettings(**eq_settings),
            rnnoise=RNNoiseSettings(
                enabled=self.rnnoise_checkbox.isChecked(),
                strength=self.strength_slider.value() / 100.0,
                model=self.model_combo.currentData() or 'rnnoise',
            ),
            compressor=CompressorSettings(**compressor_settings),
            limiter=LimiterSettings(**limiter_settings),
            bypass=self.bypass_checkbox.isChecked(),
        )

    def _apply_preset(self, preset: Preset, preset_key: str = None):
        """Apply a preset to the UI and processor.

        Args:
            preset: Preset object to apply
            preset_key: Optional key for built-in presets (e.g., "voice", "bass_cut")
        """
        # Apply gate settings
        self.gate_panel.set_settings({
            'enabled': preset.gate.enabled,
            'threshold_db': preset.gate.threshold_db,
            'attack_ms': preset.gate.attack_ms,
            'release_ms': preset.gate.release_ms,
        })

        # Apply EQ settings
        self.eq_panel.set_settings({
            'enabled': preset.eq.enabled,
            'band_gains': preset.eq.band_gains,
            'band_qs': preset.eq.band_qs,
        })

        # Apply RNNoise settings
        self.rnnoise_checkbox.setChecked(preset.rnnoise.enabled)
        self.processor.set_rnnoise_enabled(preset.rnnoise.enabled)

        # Apply strength
        strength_percent = int(preset.rnnoise.strength * 100)
        self.strength_slider.setValue(strength_percent)
        self.processor.set_rnnoise_strength(preset.rnnoise.strength)

        # Apply model selection
        model = getattr(preset.rnnoise, 'model', 'rnnoise')
        model_found = False
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model:
                self.model_combo.setCurrentIndex(i)
                model_found = True
                # Try to set model, handle errors gracefully
                try:
                    success = self.processor.set_noise_model(model)
                    if success:
                        # Update latency label
                        if model == "deepfilter":
                            self.rnnoise_latency_label.setText("Latency: ~10ms (DeepFilterNet LL)")
                        else:
                            self.rnnoise_latency_label.setText("Latency: ~10ms (RNNoise)")
                    else:
                        # Model switch failed - show warning and use RNNoise
                        print(f"Warning: Failed to switch to {model} in preset, using RNNoise")
                        self.status_bar.showMessage(
                            f"Note: Preset specifies {model} but not available, using RNNoise",
                            5000
                        )
                        # Fall back to RNNoise using find-by-ID loop (NOT hardcoded index)
                        for j in range(self.model_combo.count()):
                            if self.model_combo.itemData(j) == "rnnoise":
                                self.model_combo.setCurrentIndex(j)
                                self.processor.set_noise_model("rnnoise")
                                self.rnnoise_latency_label.setText("Latency: ~10ms (RNNoise)")
                                break
                except Exception as e:
                    # Unexpected error - log and fall back
                    print(f"Error switching model in preset: {type(e).__name__}: {e}")
                    self.status_bar.showMessage(
                        f"Error loading preset model, using RNNoise",
                        5000
                    )
                    # Fall back to RNNoise using find-by-ID loop (NOT hardcoded index)
                    for j in range(self.model_combo.count()):
                        if self.model_combo.itemData(j) == "rnnoise":
                            self.model_combo.setCurrentIndex(j)
                            self.processor.set_noise_model("rnnoise")
                            self.rnnoise_latency_label.setText("Latency: ~10ms (RNNoise)")
                            break
                break

        if not model_found:
            print(f"Warning: Preset model '{model}' not found in available models")

        # Apply compressor settings
        self.compressor_panel.set_compressor_settings({
            'enabled': preset.compressor.enabled,
            'threshold_db': preset.compressor.threshold_db,
            'ratio': preset.compressor.ratio,
            'attack_ms': preset.compressor.attack_ms,
            'release_ms': preset.compressor.release_ms,
            'makeup_gain_db': preset.compressor.makeup_gain_db,
        })

        # Apply limiter settings
        self.compressor_panel.set_limiter_settings({
            'enabled': preset.limiter.enabled,
            'ceiling_db': preset.limiter.ceiling_db,
            'release_ms': preset.limiter.release_ms,
        })

        # Apply bypass
        self.bypass_checkbox.setChecked(preset.bypass)
        self.processor.set_bypass(preset.bypass)

        self.status_bar.showMessage(f"Loaded preset: {preset.name}")

        # Save to config if preset_key provided (built-in preset)
        if preset_key:
            self.config.last_preset = f"builtin:{preset_key}"
            save_config(self.config)
            self.current_preset_path = None  # Built-in, not a file

    def _save_preset(self):
        """Save current settings as a preset."""
        # Get preset name from user
        name, ok = QInputDialog.getText(
            self,
            "Save Preset",
            "Enter preset name:",
            text="My Preset"
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
                self,
                "Preset Saved",
                f"Preset '{name}' saved to:\n{filepath}"
            )
        except (IOError, OSError, ValueError) as e:
            print(f"Preset save failed: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save preset:\n{e}\n\nCheck you have write permission to the presets folder."
            )

    def _load_preset(self):
        """Load a preset from file."""
        presets_dir = get_presets_dir()

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            str(presets_dir),
            "JSON Files (*.json);;All Files (*.*)"
        )

        if not filepath:
            return

        try:
            preset = load_preset(Path(filepath))
            self._apply_preset(preset)
            # Save to config for persistence
            self.current_preset_path = Path(filepath)
            self.config.last_preset = str(filepath)
            save_config(self.config)
        except PresetValidationError as e:
            # Actionable error for validation failures
            print(f"Preset load failed: {type(e).__name__}: {e}")
            QMessageBox.warning(
                self,
                "Invalid Preset",
                f"Could not load preset:\n\n{e}\n\n"
                "Please check the preset file and correct the invalid values."
            )
        except json.JSONDecodeError as e:
            # Actionable error for malformed JSON
            print(f"Preset load failed: {type(e).__name__}: {e}")
            QMessageBox.warning(
                self,
                "Invalid Preset File",
                f"The preset file is not valid JSON:\n\n"
                f"Error at line {e.lineno}: {e.msg}\n\n"
                "Please check the file format or try a different preset."
            )
        except Exception as e:
            # Fallback for unexpected errors with actionable guidance
            print(f"Preset load failed: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self,
                "Error Loading Preset",
                f"Failed to load preset:\n\n{type(e).__name__}: {e}\n\n"
                "If this problem persists, try:\n"
                "1. Check that the file exists and is readable\n"
                "2. Verify the file is a valid MicEq preset\n"
                "3. Try loading a different preset"
            )

    def _open_presets_folder(self):
        """Open the presets folder in the file explorer."""
        import subprocess
        import os

        presets_dir = get_presets_dir()

        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', str(presets_dir)])
        elif os.name == 'posix':  # Linux/Mac
            subprocess.run(['xdg-open', str(presets_dir)])

    def closeEvent(self, event):
        """Handle window close."""
        # Save window geometry to config
        geometry = {
            'x': self.x(),
            'y': self.y(),
            'width': self.width(),
            'height': self.height()
        }
        self.config.window_geometry = geometry

        # Save config before closing
        save_config(self.config)

        if self.processor.is_running():
            self.processor.stop()
        event.accept()


def run_app():
    """Run the MicEq application."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    import sys
    import os

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set application icon
    # In bundled exe, icon is in _internal; in dev, it's in project root
    icon_path = "mic_eq.ico"
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        icon_path = os.path.join(sys._MEIPASS, "_internal", "mic_eq.ico")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        # No icon file, skip silently
        pass

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
