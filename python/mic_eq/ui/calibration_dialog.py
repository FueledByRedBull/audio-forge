"""
Calibration dialog for Auto-EQ feature

DEBUG: Added terminal logging for calibration workflow
"""

import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import numpy as np

from .. import CORE_AVAILABLE
from ..config import DeviceIdentity, TARGET_CURVES, coerce_device_identity
from .analysis_worker import AnalysisWorker
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    SUBDUED_TEXT_STYLE,
    configure_resizable_dialog,
    configure_responsive_combo,
    create_scrollable_dialog_body,
    status_chip_style,
)
from .level_meter import LevelMeter
from .device_selection import start_processor_for_route
from .theme import (
    DESCRIPTION_LABEL_STYLE,
    PRIMARY_ACTION_BUTTON_STYLE,
    PROGRESS_BAR_STYLE,
    PROGRESS_LABEL_STYLE,
    message_text_style,
)

# Enable debug logging (set to False for production)
DEBUG = False

logger = logging.getLogger(__name__)

_TEMPORARY_MUTE_REASON = "auto_eq_calibration"

# Rainbow Passage - standard calibration text from audiometry
RAINBOW_PASSAGE = """The Rainbow Passage
The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries, men have been fascinated by this spectacle of the sky. They have tried to find explanations for it. Scientists, however, have found that it is caused by the reflection and refraction of sunlight on drops of water in the air."""

# Recording validation thresholds (dB)
TOO_QUIET_DB = -40.0  # Warn if quieter than this
TOO_LOUD_DB = -3.0  # Warn if louder than this (clipping risk)
RECORDING_DURATION = 10.0  # Seconds


def _find_processor_owner(widget: object) -> Any | None:
    parent: Any = widget
    while parent and not hasattr(parent, "processor"):
        parent = parent.parent()
    return parent


def _find_eq_panel_owner(widget: object) -> Any | None:
    parent: Any = widget
    while parent and not hasattr(parent, "eq_panel"):
        parent = parent.parent()
    return parent


def _processor_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor")

    sample_rate = int(owner.processor.sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Processor sample rate is unavailable")

    return sample_rate


def _chain_settings(
    owner: Any,
    *,
    full_chain: bool = False,
    input_pre_filtered: bool = False,
) -> dict[str, Any]:
    if full_chain:
        preset = owner._get_current_preset().to_dict()
        return {
            **{name: preset[name] for name in ("gate", "rnnoise", "deesser", "compressor", "limiter")},
            "full_chain": True,
            "input_pre_filtered": bool(input_pre_filtered),
            "input_cleanup_mode": owner.processor.get_input_cleanup_mode(),
            "processing_mode": owner._processing_mode(),
        }
    settings: dict[str, Any] = {}
    if owner is None:
        return settings
    if hasattr(owner, "deesser_panel"):
        try:
            settings["deesser"] = owner.deesser_panel.get_settings()
        except Exception:
            logger.debug(
                "Failed to collect de-esser settings for Auto-EQ simulation",
                exc_info=True,
            )
    if hasattr(owner, "compressor_panel"):
        try:
            settings["compressor"] = owner.compressor_panel.get_compressor_settings()
            settings["limiter"] = owner.compressor_panel.get_limiter_settings()
        except Exception:
            logger.debug(
                "Failed to collect dynamics settings for Auto-EQ simulation",
                exc_info=True,
            )
    return settings


def _filtered_capture_for_analysis(
    audio_data: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Apply only the live fixed pre-filter for existing analysis paths."""
    audio = np.ascontiguousarray(np.asarray(audio_data, dtype=np.float32).reshape(-1))
    if audio.size == 0:
        return audio.copy()
    if not CORE_AVAILABLE:
        raise RuntimeError("Native input conditioning is unavailable")
    from ..mic_eq_core import simulate_auto_eq_chain

    result = simulate_auto_eq_chain(
        audio,
        float(sample_rate),
        [(1000.0, 0.0, 1.41)] * 10,
        {
            "full_chain": True,
            "processing_mode": "bypass",
            "limiter_enabled": False,
            "return_output_audio": True,
        },
    )
    output = np.asarray(result.get("output_audio"), dtype=np.float32).reshape(-1)
    if output.size != audio.size or not np.isfinite(output).all():
        raise ValueError("fixed pre-filter returned invalid audio")
    return np.ascontiguousarray(output, dtype=np.float32)


def _selected_device_pair(owner: Any) -> tuple[str | None, str | None]:
    selected_input, selected_output = _selected_device_identities(owner)
    return _device_name(selected_input), _device_name(selected_output)


def _selected_device_identities(
    owner: Any,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None]:
    if owner is None:
        return None, None

    input_device = None
    output_device = None

    if hasattr(owner, "input_combo"):
        input_device = owner.input_combo.currentData() or None
    if hasattr(owner, "output_combo"):
        output_device = owner.output_combo.currentData() or None

    return coerce_device_identity(input_device), coerce_device_identity(output_device)


def _active_device_identities(
    processor: Any,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None]:
    """Read the exact route used by the running native stream."""

    def read_identity(direction: str) -> DeviceIdentity | None:
        get_name = getattr(processor, f"get_active_{direction}_device", None)
        get_endpoint_id = getattr(
            processor, f"get_active_{direction}_device_endpoint_id", None
        )
        get_name_ordinal = getattr(
            processor, f"get_active_{direction}_device_name_ordinal", None
        )
        return coerce_device_identity({
            "name": get_name() if callable(get_name) else None,
            "endpoint_id": get_endpoint_id() if callable(get_endpoint_id) else None,
            "name_ordinal": get_name_ordinal() if callable(get_name_ordinal) else None,
            "direction": direction,
        })

    return read_identity("input"), read_identity("output")


def _route_identity_matches(
    selected: DeviceIdentity | None,
    active: DeviceIdentity | None,
) -> bool:
    """Match endpoint IDs first, with ordinal-safe legacy fallback."""
    if selected is None or active is None:
        return selected is None and active is None

    if selected.endpoint_id:
        return bool(active.endpoint_id) and (
            selected.endpoint_id.casefold() == active.endpoint_id.casefold()
        )

    return (
        " ".join(selected.name.casefold().split())
        == " ".join(active.name.casefold().split())
        and (selected.name_ordinal or 0) == (active.name_ordinal or 0)
    )


def _route_identities_match(
    selected: tuple[DeviceIdentity | None, DeviceIdentity | None],
    active: tuple[DeviceIdentity | None, DeviceIdentity | None],
) -> bool:
    return _route_identity_matches(selected[0], active[0]) and _route_identity_matches(
        selected[1], active[1]
    )


def _start_selected_route(owner: Any) -> object:
    """Start the processor on the exact identities selected by the owner UI."""
    input_device = (
        owner.input_combo.currentData() if hasattr(owner, "input_combo") else None
    )
    output_device = (
        owner.output_combo.currentData() if hasattr(owner, "output_combo") else None
    )
    return start_processor_for_route(owner.processor, input_device, output_device)


def _restart_processor_for_route(
    processor: Any,
    selected: tuple[DeviceIdentity | None, DeviceIdentity | None],
    previous: tuple[DeviceIdentity | None, DeviceIdentity | None],
) -> object:
    """Restart on a selected route, restoring the old route when startup fails."""
    if any(device is None for device in previous):
        raise RuntimeError("Cannot identify the current route; stop processing before switching devices")
    processor.stop()
    try:
        return start_processor_for_route(processor, selected[0], selected[1])
    except Exception as switch_error:
        try:
            start_processor_for_route(processor, previous[0], previous[1])
        except Exception as restore_error:
            raise RuntimeError(
                f"{switch_error}; failed to restore previous route: {restore_error}"
            ) from switch_error
        raise RuntimeError(f"{switch_error}; previous route restored") from switch_error


def _sync_owner_processing_controls(owner: Any) -> None:
    """Let an owner refresh start/stop controls after a native handoff error."""
    sync = getattr(owner, "_sync_processing_controls", None)
    if callable(sync):
        sync()


def _device_name(device: object) -> str | None:
    identity = coerce_device_identity(device)
    if identity is not None:
        return identity.name
    return device if isinstance(device, str) and device else None


def _device_label(device: str | None, default_label: str) -> str:
    return device if device else default_label


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "--"


def _format_db(value: Any) -> str:
    try:
        return f"{float(value):.1f} dB"
    except (TypeError, ValueError):
        return "--"


def _diagnostic_state(confidence: float) -> str:
    if confidence >= 0.72:
        return "ok"
    if confidence >= 0.45:
        return "warn"
    return "bad"


def _set_temporary_mute(owner: Any, reason: str, enabled: bool) -> None:
    """Mute only this dialog's temporary capture reason when supported."""
    setter = getattr(owner, "set_temporary_output_mute", None)
    if callable(setter):
        setter(bool(enabled), reason)
        return
    processor = getattr(owner, "processor", None)
    setter = getattr(processor, "set_output_mute", None)
    if callable(setter):
        # Standalone dialog owners have no reason registry. Preserve their
        # user mute flag when using the processor-only test/runtime fallback.
        setter(bool(enabled or getattr(owner, "user_muted", False)))


def _candidate_metadata(
    scope: str,
    *,
    target: dict[str, Any],
    capture: dict[str, Any],
    options: dict[str, Any],
    allowed_scope: tuple[str, ...],
    verified_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe the captured candidate without retaining audio buffers."""
    return {
        "scope": scope,
        "allowed_scope": list(allowed_scope),
        "target": deepcopy(target),
        "capture_identity": deepcopy(capture),
        "options": deepcopy(options),
        "verified_stages": list(verified_stages),
    }


def _owner_calibration_context_key(owner: Any) -> str | None:
    """Read the owner's route/format/channel/cleanup identity when available."""
    getter = getattr(owner, "_calibration_context_key", None)
    if not callable(getter):
        return None
    value = getter()
    return value if isinstance(value, str) and value else None


class CalibrationDialog(QDialog):
    """Auto-EQ calibration dialog with target curve selection."""

    # Signal emitted when auto-EQ is applied (emits target curve name)
    auto_eq_applied = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-EQ Calibration")
        self.setModal(True)  # Modal dialog - blocks main window

        # Recording state
        self.recording_state = "idle"  # idle, recording, analyzing, ready
        self.audio_data: np.ndarray | None = None
        self.preview_audio_data: np.ndarray | None = None
        self.eq_settings: dict | None = None
        self._candidate_target_metadata: tuple[str, str, str] | None = None
        self._candidate_metadata: dict[str, Any] | None = None
        self._capture_context_key: str | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self._analysis_workers: list[AnalysisWorker] = []
        self._analysis_generation = 0
        self._close_requested = False
        self._close_result = False
        self._started_processor = False  # Track if we started processor ourselves
        self._recording_started_at = 0.0
        self._capture_start_timer = QTimer(self)
        self._capture_start_timer.setSingleShot(True)
        self._capture_start_timer.timeout.connect(self._begin_recording_capture)
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self._poll_recording_progress)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_analysis_workers)

        self._setup_ui()
        configure_resizable_dialog(
            self,
            preferred_width=720,
            preferred_height=800,
            minimum_width=480,
            minimum_height=360,
        )

    def _setup_ui(self):
        """Setup dialog UI."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.content_scroll_area, layout = create_scrollable_dialog_body(self)
        self.content_scroll_area.setAccessibleName("Auto-EQ calibration content")
        outer_layout.addWidget(self.content_scroll_area)

        # Target curve selector group
        curve_group = QGroupBox("Step 1: Select Target Curve")
        curve_layout = QVBoxLayout(curve_group)

        # Curve dropdown
        curve_input_layout = QFormLayout()
        curve_input_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        curve_input_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        curve_label = QLabel("Target Curve:")

        self.curve_combo = QComboBox()
        for key, curve in TARGET_CURVES.items():
            self.curve_combo.addItem(curve.name, key)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        configure_responsive_combo(self.curve_combo)
        curve_input_layout.addRow(curve_label, self.curve_combo)
        bind_label(curve_label, self.curve_combo)

        target_mode_label = QLabel("Target Mode:")
        self.target_mode_combo = QComboBox()
        self.target_mode_combo.addItem("Adaptive voice-aware", "adaptive")
        self.target_mode_combo.addItem("Static catalog curve", "static")
        self.target_mode_combo.setToolTip(
            "Adaptive mode applies bounded voice-aware target offsets. Static mode uses the selected curve exactly."
        )
        configure_responsive_combo(self.target_mode_combo)
        curve_input_layout.addRow(target_mode_label, self.target_mode_combo)
        bind_label(target_mode_label, self.target_mode_combo)

        smoothing_label = QLabel("Smoothing:")
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItem("Conservative", "conservative")
        self.smoothing_combo.addItem("Balanced", "balanced")
        self.smoothing_combo.addItem("Broad", "broad")
        self.smoothing_combo.setToolTip(
            "Conservative smoothing resists narrow measurement artifacts. Broad is safest but less detailed."
        )
        configure_responsive_combo(self.smoothing_combo)
        curve_input_layout.addRow(smoothing_label, self.smoothing_combo)
        bind_label(smoothing_label, self.smoothing_combo)
        curve_layout.addLayout(curve_input_layout)

        # Curve description (updates when selection changes)
        self.curve_description = QLabel()
        self.curve_description.setWordWrap(True)
        self.curve_description.setStyleSheet(DESCRIPTION_LABEL_STYLE)
        curve_layout.addWidget(self.curve_description)

        layout.addWidget(curve_group)

        # Instructions group with Rainbow Passage
        instructions_group = QGroupBox("Step 2: Read Passage Aloud")
        instructions_layout = QVBoxLayout(instructions_group)

        # Scrollable text area for Rainbow Passage
        passage_text = QTextEdit()
        passage_text.setPlainText(RAINBOW_PASSAGE)
        passage_text.setReadOnly(True)
        passage_text.setMaximumHeight(130)
        passage_text.setAccessibleName("Calibration passage")
        passage_text.setAccessibleDescription(
            "Read-only passage to speak during Auto-EQ calibration."
        )
        instructions_layout.addWidget(passage_text)

        layout.addWidget(instructions_group)

        # Recording UI group
        self.recording_group = QGroupBox("Step 3: Record Your Voice")
        recording_layout = QVBoxLayout(self.recording_group)
        self.recording_group.setVisible(False)  # Hidden until user clicks Start

        # Progress bar - solid continuous style
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # Hide "0%" text
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setAccessibleName("Calibration progress")
        recording_layout.addWidget(self.progress_bar)

        # Time remaining label below progress bar
        self.time_label = QLabel(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.time_label.setAccessibleName("Calibration time remaining")
        recording_layout.addWidget(self.time_label)

        # Level meter (vertical) for real-time validation
        info_layout = QHBoxLayout()
        self.level_meter = LevelMeter(label="Level", show_scale=True)
        self.level_meter.setMinimumHeight(120)
        info_layout.addWidget(self.level_meter)
        recording_layout.addLayout(info_layout)

        # Validation warning label
        self.warning_label = QLabel("Ready to record")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.warning_label.setAccessibleName("Calibration status")
        self.warning_label.setWordWrap(True)
        recording_layout.addWidget(self.warning_label)

        self.diagnostics_group = QGroupBox("Analysis Diagnostics")
        diagnostics_layout = QVBoxLayout(self.diagnostics_group)
        self.confidence_label = QLabel("Confidence: --")
        self.error_label = QLabel("Target error: --")
        self.gain_scale_label = QLabel("Gain scale: --")
        self.target_profile_label = QLabel("Target profile: --")
        for label in (
            self.confidence_label,
            self.error_label,
            self.gain_scale_label,
            self.target_profile_label,
        ):
            label.setStyleSheet(status_chip_style("idle"))
            label.setWordWrap(True)
            diagnostics_layout.addWidget(label)
        hint_label = QLabel(
            "Diagnostics are computed from recording clarity, repeatability, "
            "and post-solve validation."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        diagnostics_layout.addWidget(hint_label)
        self.diagnostics_group.setVisible(False)
        recording_layout.addWidget(self.diagnostics_group)

        # Recording controls
        control_layout = QHBoxLayout()

        # Retake button (hidden initially)
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.setVisible(False)
        self.retake_btn.clicked.connect(self._on_retake_clicked)
        control_layout.addWidget(self.retake_btn)

        self.compare_button = QPushButton("Compare Recording")
        self.compare_button.setEnabled(False)
        self.compare_button.setToolTip("Listen to this same passage with current and proposed processing")
        self.compare_button.clicked.connect(self._compare_recording)
        control_layout.addWidget(self.compare_button)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        control_layout.addWidget(self.cancel_btn)

        recording_layout.addLayout(control_layout)
        layout.addWidget(self.recording_group)

        # Start button (opens recording section)
        self.start_button = QPushButton("Start Calibration")
        self.start_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)

        set_accessible_group(
            (
                (self.start_button, "Start Auto-EQ calibration", None),
                (self.retake_btn, "Retake calibration recording", None),
                (self.cancel_btn, "Cancel Auto-EQ calibration", None),
                (self.level_meter, "Calibration input level", None),
            )
        )
        self.setTabOrder(self.curve_combo, self.target_mode_combo)
        self.setTabOrder(self.target_mode_combo, self.smoothing_combo)
        self.setTabOrder(self.smoothing_combo, self.start_button)
        self.setTabOrder(self.start_button, self.retake_btn)
        self.setTabOrder(self.retake_btn, self.cancel_btn)

        # Initialize with first curve description
        self._on_curve_changed(0)

    def _on_curve_changed(self, index: int):
        """Update description when curve selection changes."""
        if index < 0:
            return
        curve_key = self.curve_combo.currentData()
        curve = TARGET_CURVES[curve_key]
        self.curve_description.setText(curve.description)

    def _on_start_clicked(self):
        """Handle Start Calibration button click based on recording state."""
        if self.recording_state == "idle":
            # Show recording UI and start recording
            self.recording_group.setVisible(True)
            self._start_recording()
        elif self.recording_state == "recording":
            # User tried to stop early - not allowed
            QMessageBox.information(
                self,
                "Recording",
                "Please record the full 10 seconds for accurate calibration.",
            )
        elif self.recording_state == "analyzing":
            return
        elif self.recording_state == "ready":
            if self.eq_settings is not None:
                self._apply_eq_settings()
            else:
                self._reset_recording_ui()
                self._start_recording()

    def _apply_eq_settings(self):
        """Apply auto-EQ settings to main window and close dialog."""
        if self.recording_state != "ready" or self.eq_settings is None:
            return
        if self._candidate_target_metadata is None:
            return
        eq_settings = deepcopy(self.eq_settings)
        if DEBUG:
            logger.debug("Applying EQ settings")

        # Get parent's EQ panel (MainWindow has it)
        parent = _find_eq_panel_owner(self.parent())

        if not parent:
            QMessageBox.critical(self, "Error", "Could not find EQ panel")
            return

        identity_error = self._candidate_identity_error(eq_settings, parent)
        if identity_error is not None:
            QMessageBox.critical(
                self,
                "Stale Auto-EQ Candidate",
                f"{identity_error}; no changes were applied.",
            )
            return

        # Build all band tuples before touching the live EQ state.
        from ..config import EQ_FREQUENCIES as BAND_FREQUENCIES_HZ

        try:
            freqs_hz = eq_settings.get("band_freqs", BAND_FREQUENCIES_HZ)
            gains = eq_settings["band_gains"]
            qs = eq_settings.get("band_qs", [1.41] * len(BAND_FREQUENCIES_HZ))
            if any(
                len(values) != len(BAND_FREQUENCIES_HZ)
                for values in (freqs_hz, gains, qs)
            ):
                raise ValueError("Auto-EQ candidate does not contain 10 complete bands")
            bands = [(freq, gains[i], qs[i]) for i, freq in enumerate(freqs_hz)]
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.critical(self, "Error", f"Invalid Auto-EQ candidate: {error}")
            return

        get_eq_settings = getattr(parent.eq_panel, "get_settings", None)
        if not callable(get_eq_settings):
            QMessageBox.critical(
                self,
                "Error",
                "Could not snapshot current EQ settings; no changes were applied.",
            )
            return
        try:
            eq_snapshot = deepcopy(get_eq_settings())
        except Exception as error:
            logger.warning("Failed to snapshot EQ before candidate apply", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Could not snapshot current EQ settings; no changes were applied: {error}",
            )
            return

        try:
            parent.eq_panel.apply_auto_eq_results(bands, diagnostics=eq_settings)
            parent.eq_panel.set_settings({"enabled": True})
            parent.eq_panel.set_auto_eq_diagnostics(eq_settings)
            accepted_eq = deepcopy(get_eq_settings())
            if not isinstance(accepted_eq, Mapping):
                raise TypeError("EQ panel returned invalid settings")
            if "enabled" in accepted_eq and not bool(accepted_eq["enabled"]):
                raise ValueError("EQ panel did not accept the enabled state")
            for key in ("schema_version", "bands", "layers"):
                if key in accepted_eq:
                    eq_settings[key] = deepcopy(accepted_eq[key])
            for key in ("enabled", "band_freqs", "band_gains", "band_qs"):
                if key in accepted_eq:
                    value = accepted_eq[key]
                    eq_settings[key] = (
                        bool(value) if key == "enabled" else list(value)
                    )
        except Exception as error:
            restore_error = None
            try:
                current_eq = parent.eq_panel.get_settings()
            except Exception:
                current_eq = None
            if current_eq != eq_snapshot:
                try:
                    parent.eq_panel.set_settings(eq_snapshot)
                except Exception as restore_exc:
                    restore_error = restore_exc
                    logger.warning(
                        "Failed to restore EQ after candidate apply failure",
                        exc_info=True,
                    )
            logger.warning("Failed to apply Auto-EQ candidate", exc_info=True)
            message = f"Could not apply Auto-EQ settings: {error}"
            if restore_error is not None:
                message += f"; EQ restoration failed: {restore_error}"
            QMessageBox.critical(
                self, "Error", message
            )
            return

        self.eq_settings = eq_settings
        candidate = eq_settings.get("_candidate")
        if isinstance(candidate, dict):
            candidate["verified_stages"] = ["eq"]

        # Emit signal for main window to handle preset save and undo button
        target_curve = (
            self._candidate_metadata["target"]["curve"]
            if self._candidate_metadata is not None
            else self._candidate_target_metadata[0]
        )
        self.auto_eq_applied.emit(target_curve)
        from .calibration_history import persist_calibration

        persist_calibration(self, parent, "eq_only", target_curve, ("eq",))

        if DEBUG:
            logger.debug(
                "EQ settings applied, signal emitted for curve=%s", target_curve
            )

        # Close dialog
        self.accept()

    def _compare_recording(self) -> None:
        if self.recording_state != "ready" or self.audio_data is None or self.eq_settings is None:
            return
        owner = _find_eq_panel_owner(self.parent())
        if owner is None:
            return
        error = self._candidate_identity_error(self.eq_settings, owner)
        if error:
            QMessageBox.warning(self, "Stale recording", error)
            return
        from .listening_comparison_dialog import ListeningComparisonDialog

        try:
            chain = _chain_settings(
                owner,
                full_chain=True,
                input_pre_filtered=self.preview_audio_data is None,
            )
        except Exception as error:
            QMessageBox.warning(self, "Comparison unavailable", str(error))
            return
        dialog = ListeningComparisonDialog(
            audio_data=self.preview_audio_data
            if self.preview_audio_data is not None
            else self.audio_data,
            sample_rate=_processor_sample_rate(owner),
            current_settings=owner.eq_panel.get_settings(),
            proposed_settings=self.eq_settings,
            current_chain_settings=chain,
            proposed_chain_settings=chain,
            parent=self,
        )
        _set_temporary_mute(owner, "listening_comparison", True)
        try:
            keep = dialog.exec() == int(QDialog.DialogCode.Accepted)
        finally:
            _set_temporary_mute(owner, "listening_comparison", False)
            dialog.deleteLater()
        if keep:
            self._apply_eq_settings()

    def _candidate_identity_error(
        self, eq_settings: Mapping[str, Any], parent: Any
    ) -> str | None:
        candidate = eq_settings.get("_candidate")
        expected = self._candidate_metadata
        if not isinstance(candidate, Mapping) or not isinstance(expected, Mapping):
            return "Auto-EQ candidate identity is missing"
        if candidate.get("scope") != "eq_only":
            return "Auto-EQ candidate scope is invalid"
        if tuple(candidate.get("allowed_scope") or ()) != ("eq",):
            return "Auto-EQ candidate scope is incomplete"
        for key in ("scope", "allowed_scope", "target", "options", "capture_identity"):
            if candidate.get(key) != expected.get(key):
                return "Auto-EQ candidate options or capture identity changed"

        target = candidate.get("target")
        if not isinstance(target, Mapping) or self._candidate_target_metadata is None:
            return "Auto-EQ candidate target is missing"
        if tuple(target.get(key) for key in ("curve", "mode", "smoothing")) != self._candidate_target_metadata:
            return "Auto-EQ candidate target does not match the captured target"

        capture = candidate.get("capture_identity")
        if not isinstance(capture, Mapping):
            return "Auto-EQ capture identity is missing"
        if self.audio_data is None:
            return "Auto-EQ recording is missing"
        if capture.get("sample_count") != int(self.audio_data.size):
            return "Auto-EQ recording changed after analysis"
        try:
            if capture.get("sample_rate") != _processor_sample_rate(parent):
                return "Audio sample rate changed after analysis"
        except RuntimeError:
            return "Auto-EQ capture sample rate is unavailable"
        if capture.get("context_key") != _owner_calibration_context_key(parent):
            return "Audio route or input cleanup context changed after capture"
        generation = capture.get("generation")
        if generation != self._analysis_generation:
            return "Auto-EQ candidate is stale"
        return None

    def _start_recording(self):
        """Start non-blocking recording."""
        if DEBUG:
            logger.debug("Start recording clicked")

        self._clear_eq_candidate()
        self._cancel_analysis_workers()

        # Get parent's processor (MainWindow has it)
        parent = _find_processor_owner(self.parent())

        if not parent:
            QMessageBox.critical(self, "Error", "Could not find audio processor")
            return

        processor_was_running = parent.processor.is_running()
        selected_identities = _selected_device_identities(parent)
        selected_input = _device_name(selected_identities[0])
        selected_output = _device_name(selected_identities[1])

        if DEBUG:
            logger.debug(
                "Processor state: running=%s, selected_input=%r, selected_output=%r",
                processor_was_running,
                selected_input,
                selected_output,
            )

        if processor_was_running:
            active_identities = _active_device_identities(parent.processor)
            active_input = _device_name(active_identities[0])
            active_output = _device_name(active_identities[1])
            if DEBUG:
                logger.debug(
                    "Active stream devices: input=%r, output=%r",
                    active_input,
                    active_output,
                )

            if not _route_identities_match(selected_identities, active_identities):
                reply = QMessageBox.question(
                    self,
                    "Switch Devices for Auto-EQ?",
                    "Auto-EQ should record from your selected devices.\n\n"
                    f"Selected input: {_device_label(selected_input, '(Default Input)')}\n"
                    f"Selected output: {_device_label(selected_output, '(Default Output)')}\n\n"
                    f"Active input: {_device_label(active_input, '(Default Input)')}\n"
                    f"Active output: {_device_label(active_output, '(Default Output)')}\n\n"
                    "Switch now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.warning_label.setText(
                        "Calibration canceled: using current stream devices"
                    )
                    self.warning_label.setStyleSheet(message_text_style("warn"))
                    return

                try:
                    if DEBUG:
                        logger.debug("Restarting processor on selected devices")
                    _restart_processor_for_route(
                        parent.processor, selected_identities, active_identities
                    )
                    _set_temporary_mute(parent, _TEMPORARY_MUTE_REASON, False)
                    if DEBUG:
                        logger.debug("Processor restarted on selected devices")
                except Exception as e:
                    _sync_owner_processing_controls(parent)
                    QMessageBox.critical(
                        self,
                        "Audio Error",
                        f"Failed to switch audio devices for calibration:\n{e}",
                    )
                    return
            self._started_processor = False
            if DEBUG:
                logger.debug("Reusing running processor session")
        else:
            try:
                if DEBUG:
                    logger.debug("Starting audio processor from main thread")
                _start_selected_route(parent)
                self._started_processor = True  # Track that we started it
                if DEBUG:
                    logger.debug("Audio processor started successfully")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Audio Error",
                    f"Failed to start audio processing:\n{str(e)}\n\n"
                    "Check that audio devices are connected and not in use by another application.",
                )
                return

        self._capture_context_key = _owner_calibration_context_key(parent)
        self.recording_state = "recording"
        self.start_button.setText("Recording...")
        self.start_button.setEnabled(False)  # Prevent early stop

        # Lock Auto-EQ target controls during recording
        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)

        # Let DSP loop settle without blocking the UI thread.
        self._capture_start_timer.start(100)

        self.warning_label.setText("Recording... Speak clearly into your microphone")
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))

    def _begin_recording_capture(self):
        """Start Rust-side recording and poll progress from the main Qt thread."""
        self._capture_start_timer.stop()
        if self.recording_state != "recording":
            return

        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            _set_temporary_mute(parent, _TEMPORARY_MUTE_REASON, True)
            parent.processor.set_recovery_suppressed(True)
            parent.processor.start_raw_recording(
                RECORDING_DURATION,
                before_cleanup=True,
            )
        except Exception as e:
            self._on_recording_failed(f"Recording error: {e}")
            return

        self._recording_started_at = time.time()
        self.recording_timer.start()
        if DEBUG:
            logger.debug("Started main-thread recording capture")

    def _poll_recording_progress(self):
        """Poll recording state from the main Qt thread."""
        if self.recording_state != "recording":
            self.recording_timer.stop()
            return

        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            progress_float = float(parent.processor.recording_progress())
            progress_pct = int(progress_float * 100)
            self._on_progress_update(progress_pct)
            self._on_time_remaining(
                max(0.0, RECORDING_DURATION * (1.0 - progress_float))
            )
            self._on_level_update(
                float(parent.processor.recording_level_db()),
                float(parent.processor.get_input_peak_db()),
            )

            if progress_pct >= 100 or parent.processor.is_recording_complete():
                self.recording_timer.stop()
                audio = parent.processor.stop_raw_recording()
                _set_temporary_mute(parent, _TEMPORARY_MUTE_REASON, False)
                if audio is None:
                    self._on_recording_failed("Recording failed - no audio data")
                    return
                audio_array = np.asarray(audio, dtype=np.float32)
                self._on_recording_complete(audio_array)
        except Exception as e:
            self.recording_timer.stop()
            self._on_recording_failed(f"Recording error: {e}")

    def _on_progress_update(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def _on_time_remaining(self, seconds: float):
        """Update time remaining label."""
        if seconds > 0:
            self.time_label.setText(f"Time remaining: {seconds:.0f}s")
        else:
            self.time_label.setText("✓ Complete!")
            self.time_label.setStyleSheet(message_text_style("ok", strong=True))

    def _on_level_update(self, rms_db: float, peak_db: float):
        """Update level meter with validation warnings."""
        self.level_meter.set_levels(rms_db, peak_db)

        # Show validation warning
        if rms_db < TOO_QUIET_DB:
            self.warning_label.setText("⚠️ Too quiet! Move closer to mic")
            self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        elif peak_db > TOO_LOUD_DB:
            self.warning_label.setText("⚠️ Too loud! Risk of clipping")
            self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        else:
            self.warning_label.setText("✓ Level is good")
            self.warning_label.setStyleSheet(message_text_style("ok", strong=True))

    def _on_recording_complete(self, audio_data: np.ndarray):
        """Handle recording completion."""
        if DEBUG:
            logger.debug("Recording complete: %d samples", len(audio_data))

            rms = np.mean(audio_data**2) ** 0.5
            peak_db = 20 * np.log10(max(np.abs(audio_data).max(), 1e-6))
            rms_db = 20 * np.log10(max(rms, 1e-6))
            logger.debug("Audio stats - Peak: %.1f dB, RMS: %.1f dB", peak_db, rms_db)

        self.preview_audio_data = np.ascontiguousarray(
            np.asarray(audio_data, dtype=np.float32).reshape(-1).copy()
        )
        try:
            sample_rate = _processor_sample_rate(_find_processor_owner(self.parent()))
        except Exception:
            sample_rate = 48_000
        self.audio_data = _filtered_capture_for_analysis(
            self.preview_audio_data,
            sample_rate,
        )
        self._clear_eq_candidate()
        self.recording_state = "analyzing"

        # Update UI
        self.start_button.setText("Analyzing...")
        self.start_button.setEnabled(False)
        self.retake_btn.setVisible(True)

        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)

        # Show completion message
        self.warning_label.setText(
            f"Recording complete! {len(audio_data)} samples captured"
        )
        self.warning_label.setStyleSheet(message_text_style("ok", strong=True))

        if DEBUG:
            logger.debug("Audio captured; starting analysis")

        if DEBUG:
            logger.debug("Starting analysis worker")
        self._start_analysis()

    def _on_recording_failed(self, error: str):
        """Handle recording failure."""
        self.recording_timer.stop()
        self.warning_label.setText(f"❌ Recording failed: {error}")
        self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        self._reset_recording_ui()

    def _start_analysis(self):
        """Start analysis worker."""
        if self.audio_data is None:
            self._on_analysis_failed("No audio data to analyze")
            return

        self._cancel_analysis_workers()

        # Get parent's processor for sample rate
        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_analysis_failed("Could not find processor")
            return

        try:
            sample_rate = _processor_sample_rate(parent)
            target_preset = self.get_selected_curve()
            target_mode = self.get_selected_target_mode()
            smoothing_strength = self.get_selected_smoothing_strength()
        except Exception as error:
            self._on_analysis_failed(f"Analysis setup failed: {error}")
            return
        self._candidate_target_metadata = (
            target_preset,
            target_mode,
            smoothing_strength,
        )
        self._candidate_metadata = _candidate_metadata(
            "eq_only",
            target={
                "curve": target_preset,
                "mode": target_mode,
                "smoothing": smoothing_strength,
            },
            capture={
                "sample_rate": sample_rate,
                "sample_count": int(self.audio_data.size),
                "context_key": self._capture_context_key,
            },
            options={
                "target_mode": target_mode,
                "smoothing_strength": smoothing_strength,
            },
            allowed_scope=("eq",),
        )
        self.eq_settings = None
        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)
        chain_settings = _chain_settings(parent)

        if DEBUG:
            logger.debug(
                "Creating AnalysisWorker: %d samples, %sHz, target=%s, mode=%s, smoothing=%s",
                len(self.audio_data),
                sample_rate,
                target_preset,
                target_mode,
                smoothing_strength,
            )

        # Create and start analysis worker
        generation = self._analysis_generation + 1
        self._analysis_generation = generation
        self._candidate_metadata["capture_identity"]["generation"] = generation
        try:
            worker = AnalysisWorker(
                self.audio_data,
                sample_rate,
                target_preset,
                target_mode=target_mode,
                smoothing_strength=smoothing_strength,
                chain_settings=chain_settings,
            )
        except Exception as error:
            self._on_analysis_failed(f"Analysis setup failed: {error}")
            return
        self.analysis_worker = worker
        self._analysis_workers.append(worker)
        worker.step_progress.connect(
            lambda step_name, percentage, token=generation: self._on_analysis_step(
                step_name, percentage, token
            )
        )
        worker.result_ready.connect(
            lambda settings, token=generation: self._on_analysis_complete(
                settings, token
            )
        )
        worker.failed.connect(
            lambda error, token=generation: self._on_analysis_failed(error, token)
        )
        worker.finished.connect(
            lambda finished_worker=worker: self._on_analysis_thread_finished(
                finished_worker
            )
        )
        worker.start()

        if DEBUG:
            logger.debug("AnalysisWorker started")

    def _on_analysis_step(
        self,
        step_name: str,
        percentage: int,
        generation: int | None = None,
    ):
        """Handle analysis step progress."""
        if generation is not None and generation != self._analysis_generation:
            return
        if self.recording_state != "analyzing":
            return
        if self._close_requested:
            return
        if DEBUG:
            logger.debug("Analysis step %s%%: %s", percentage, step_name)
        self.warning_label.setText(f"Analyzing: {step_name}")
        self.progress_bar.setValue(percentage)

    def _on_analysis_complete(
        self, eq_settings: dict, generation: int | None = None
    ):
        """Handle analysis completion."""
        if generation is not None and generation != self._analysis_generation:
            return
        if self.recording_state != "analyzing":
            return
        if self._close_requested:
            return
        if self._candidate_target_metadata is None:
            self._on_analysis_failed("Analysis target metadata is unavailable")
            return
        if DEBUG:
            logger.debug("Analysis complete")
            logger.debug(
                "Band gains: %s", [round(g, 1) for g in eq_settings["band_gains"]]
            )
            max_gain = max(abs(g) for g in eq_settings["band_gains"])
            logger.debug("Max correction: %.1f dB", round(max_gain, 1))

        eq_settings = deepcopy(eq_settings)
        candidate = deepcopy(self._candidate_metadata or {})
        if isinstance(candidate, dict):
            candidate["verified_stages"] = []
        eq_settings["_candidate"] = candidate
        apply_recommended = bool(eq_settings.get("apply_recommended", True))
        if apply_recommended:
            self.eq_settings = eq_settings
        else:
            self._clear_eq_candidate()
        self.recording_state = "ready"
        self.compare_button.setEnabled(apply_recommended)
        if apply_recommended:
            self.warning_label.setText(
                "Analysis complete! Max correction: "
                f"{round(max(abs(g) for g in eq_settings['band_gains']), 1)} dB"
            )
            self.warning_label.setStyleSheet(message_text_style("ok", strong=True))
        else:
            reasons = eq_settings.get("abstention_reasons") or [
                "the recording did not support a safe correction"
            ]
            self.warning_label.setText(
                "No EQ applied: " + "; ".join(str(reason) for reason in reasons)
            )
            self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        self.progress_bar.setValue(100)
        self._show_analysis_diagnostics(eq_settings)
        self.start_button.setText(
            "Apply EQ Settings" if apply_recommended else "Record Again"
        )
        self.start_button.setEnabled(True)
        if DEBUG:
            logger.debug("EQ settings ready to apply")

    def _show_analysis_diagnostics(self, eq_settings: dict) -> None:
        """Show Auto-EQ confidence and validation details before applying."""
        confidence = float(eq_settings.get("analysis_confidence", 0.0) or 0.0)
        eq_confidence = float(eq_settings.get("eq_confidence", confidence) or 0.0)
        capture_confidence = float(
            eq_settings.get("capture_confidence", confidence) or 0.0
        )
        validation_confidence = float(
            eq_settings.get("validation_confidence", 0.0) or 0.0
        )
        state = _diagnostic_state(confidence)
        before = eq_settings.get("validation_before_error_db")
        after = eq_settings.get("validation_after_error_db")
        scale = eq_settings.get("validation_gain_scale")
        target_profile = eq_settings.get("target_profile", "--")
        residual = eq_settings.get("residual_regularization") or {}
        used_fallback = bool(eq_settings.get("used_spectrum_fallback", False))
        headroom = eq_settings.get("headroom_validation") or {}
        headroom_after = headroom.get("after") if isinstance(headroom, dict) else None
        headroom_safe = (
            bool(headroom.get("safe", True)) if isinstance(headroom, dict) else True
        )
        headroom_advisory = (
            bool(headroom.get("advisory", False))
            if isinstance(headroom, dict)
            else False
        )
        headroom_scale = (
            headroom.get("gain_scale") if isinstance(headroom, dict) else None
        )

        self.confidence_label.setText(
            "Confidence: "
            f"overall {_format_percent(confidence)} | "
            f"EQ {_format_percent(eq_confidence)} | "
            f"capture {_format_percent(capture_confidence)}"
        )
        self.confidence_label.setStyleSheet(status_chip_style(state))
        self.error_label.setText(
            f"Target error: {_format_db(before)} -> {_format_db(after)}"
        )
        self.error_label.setStyleSheet(
            status_chip_style("ok" if after is not None else "idle")
        )
        self.gain_scale_label.setText(
            f"Validation: {_format_percent(validation_confidence)} | gain scale {_format_percent(scale)}"
        )
        self.gain_scale_label.setStyleSheet(status_chip_style("info"))
        if isinstance(residual, dict) and "max_regularized_correction_db" in residual:
            requested = _format_db(residual.get("max_requested_correction_db"))
            regularized = _format_db(residual.get("max_regularized_correction_db"))
            narrow = _format_db(residual.get("max_narrow_residual_db"))
            self.gain_scale_label.setText(
                f"{self.gain_scale_label.text()} | correction {requested}->{regularized} | narrow {narrow}"
            )
        if isinstance(headroom_after, dict):
            pre_tp_headroom = _format_db(
                headroom_after.get("pre_limiter_true_peak_headroom_db")
            )
            limiter_gr = _format_db(headroom_after.get("limiter_gain_reduction_db"))
            true_peak_gr = _format_db(
                headroom_after.get("true_peak_limiter_gain_reduction_db")
            )
            headroom_status = (
                "advisory only (Rust simulator unavailable)"
                if headroom_advisory
                else "safe correction"
                if headroom_safe
                else "headroom risk"
            )
            self.gain_scale_label.setText(
                f"{self.gain_scale_label.text()} | {headroom_status}: "
                f"TP headroom {pre_tp_headroom}, LIM GR {limiter_gr}, TP GR {true_peak_gr}"
            )
            self.gain_scale_label.setStyleSheet(
                status_chip_style("info" if headroom_safe else "warn")
            )
            if headroom_scale is not None and float(headroom_scale) < 1.0:
                self.gain_scale_label.setText(
                    f"{self.gain_scale_label.text()} | headroom scale {_format_percent(headroom_scale)}"
                )
        self.target_profile_label.setText(
            f"Target profile: {target_profile}"
            + (" | fallback spectrum" if used_fallback else "")
        )
        self.target_profile_label.setStyleSheet(status_chip_style("info"))
        self.diagnostics_group.setVisible(True)

    def _on_analysis_failed(self, error: str, generation: int | None = None):
        """Handle analysis failure."""
        if generation is not None and generation != self._analysis_generation:
            return
        if self.recording_state != "analyzing":
            return
        if self._close_requested:
            return
        if DEBUG:
            logger.debug("Analysis failed: %s", error)
        self._clear_eq_candidate()
        self.recording_state = "ready"
        self.warning_label.setText(f"❌ {error}")
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        self.diagnostics_group.setVisible(False)
        self.start_button.setText("Record Again")
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)

    def _on_retake_clicked(self):
        """Discard and re-record."""
        reply = QMessageBox.question(
            self,
            "Discard Recording?",
            "Discard current recording and start over?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_recording_ui()

    def _on_cancel_clicked(self):
        """Cancel with confirmation if recording."""
        if self.recording_state == "recording":
            reply = QMessageBox.question(
                self,
                "Cancel Recording?",
                "Discard recording and return to main window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.recording_timer.stop()
                self._cancel_analysis_workers()
                self._cleanup_recording_tap()
                self.reject()
        else:
            self.recording_timer.stop()
            self._cancel_analysis_workers()
            self._cleanup_recording_tap()
            self.reject()

    def _reset_recording_ui(self):
        """Reset UI to initial idle state."""
        self.compare_button.setEnabled(False)
        self._capture_start_timer.stop()
        self.recording_timer.stop()
        self._cancel_analysis_workers()
        self._cleanup_recording_tap()

        # Stop processor if we started it ourselves
        self._stop_owned_processor()

        self.recording_state = "idle"
        self._clear_eq_candidate()
        self.audio_data = None
        self.preview_audio_data = None
        self._capture_context_key = None
        self.progress_bar.setValue(0)
        self.time_label.setText(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.level_meter.set_levels(-120.0, -120.0)
        self.warning_label.setText("Ready to record")
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.diagnostics_group.setVisible(False)
        self.start_button.setText("Start Calibration")
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(False)
        self.curve_combo.setEnabled(True)
        self.target_mode_combo.setEnabled(True)
        self.smoothing_combo.setEnabled(True)

    def _clear_eq_candidate(self) -> None:
        """Discard the only candidate that may be applied."""
        self.eq_settings = None
        self._candidate_target_metadata = None
        self._candidate_metadata = None

    def _stop_owned_processor(self):
        """Stop the processor only if this dialog started it."""
        if not getattr(self, "_started_processor", False):
            return

        parent = _find_processor_owner(self.parent())
        if parent:
            try:
                if DEBUG:
                    logger.debug("Stopping audio processor owned by calibration dialog")
                parent.processor.stop()
            except Exception:
                if DEBUG:
                    logger.debug("Error stopping processor", exc_info=True)
        self._started_processor = False

    def _cleanup_recording_tap(self):
        """Best-effort cleanup for tap/mute state across cancel/close paths."""
        parent = _find_processor_owner(self.parent())

        if not parent:
            return

        try:
            parent.processor.stop_raw_recording()
        except RuntimeError as e:
            if "No recording in progress" not in str(e):
                logger.warning("Failed to stop raw recording during cleanup: %s", e)
        except Exception as e:
            logger.warning("Failed to stop raw recording during cleanup: %s", e)

        try:
            _set_temporary_mute(parent, _TEMPORARY_MUTE_REASON, False)
        except Exception as e:
            logger.warning("Failed to unmute output during cleanup: %s", e)

        try:
            parent.processor.set_recovery_suppressed(False)
        except Exception as e:
            logger.warning("Failed to re-enable recovery after cleanup: %s", e)

    def _cancel_analysis_workers(self) -> None:
        """Cancel work without dropping ownership of a running QThread."""
        self._analysis_generation += 1
        for worker in tuple(self._analysis_workers):
            if worker.isRunning():
                worker.stop()
        self.analysis_worker = None

    def _on_analysis_thread_finished(self, worker: AnalysisWorker) -> None:
        if worker in self._analysis_workers:
            self._analysis_workers.remove(worker)
        if self.analysis_worker is worker:
            self.analysis_worker = None
        worker.deleteLater()
        if self._close_requested:
            self._finish_close()

    def _finish_close(self) -> None:
        if not self._close_requested or any(
            worker.isRunning() for worker in self._analysis_workers
        ):
            return
        for worker in tuple(self._analysis_workers):
            worker.deleteLater()
        self._analysis_workers.clear()
        self._close_requested = False
        self.audio_data = None
        self.preview_audio_data = None
        QDialog.done(self, int(QDialog.DialogCode.Accepted if self._close_result else QDialog.DialogCode.Rejected))

    def _wait_for_analysis_workers(self) -> None:
        """Keep application shutdown from destroying a running QThread."""
        self._request_close(False)
        for worker in tuple(self._analysis_workers):
            if worker.isRunning():
                worker.stop()
                worker.wait()
        self._analysis_workers.clear()
        self.analysis_worker = None

    def _request_close(self, accepted: bool) -> None:
        if self._close_requested:
            return
        self._close_requested = True
        self._close_result = accepted
        self.recording_state = "idle"
        self._clear_eq_candidate()
        self._capture_start_timer.stop()
        self.recording_timer.stop()
        self._cancel_analysis_workers()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        if any(worker.isRunning() for worker in self._analysis_workers):
            self.start_button.setEnabled(False)
            self.retake_btn.setEnabled(False)
            self.warning_label.setText("Canceling analysis...")
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))
            return
        self._finish_close()

    def closeEvent(self, event):
        """Ensure recording state is cleaned up if dialog is closed directly."""
        self._request_close(False)
        event.ignore()

    def accept(self):
        self._request_close(True)

    def reject(self):
        self._request_close(False)

    def get_selected_curve(self):
        """
        Return the selected target curve key.

        Returns:
            str: Target curve key ('broadcast', 'podcast', 'streaming', or 'flat')
        """
        return self.curve_combo.currentData()

    def get_selected_target_mode(self):
        """Return the selected target behavior."""
        return self.target_mode_combo.currentData()

    def get_selected_smoothing_strength(self):
        """Return the selected Auto-EQ smoothing strength."""
        return self.smoothing_combo.currentData()
