"""
Recording dialog for the Auto-EQ feature

DEBUG: Added terminal logging for the recording workflow
"""

import logging
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from typing import Any, NamedTuple

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QMessageBox,
    QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, Signal
import numpy as np

from ..config import (
    EQSettings,
    Preset,
    TARGET_CURVES,
    build_eq_candidate_settings,
)
from ..analysis.noise_reference import capture_is_silent
from ..analysis.voice_setup import raw_capture_is_clipped
from .analysis_worker import AnalysisWorker, CaptureAnalysisWorker
from .accessibility import bind_label, set_accessible_group
from .auto_eq_explanation import (
    TARGET_MODE_LABELS,
    explain_auto_eq,
    headroom_advice,
    target_profile_text,
)
from .components import StepHeader
from .layout_constants import (
    MARGIN_PANEL,
    SUBDUED_TEXT_STYLE,
    configure_resizable_dialog,
    configure_responsive_combo,
    create_scrollable_dialog_body,
    status_chip_style,
)
from .eq_compare import EQComparisonWidget
from .level_check import (
    LEVEL_CHECK_SECONDS,
    LEVEL_HOLD_UPDATES,
    LevelVerdict,
    clipped_message,
    level_choices,
    live_verdict,
    verdict_from_audio,
)
from .level_meter import LevelMeter, MeterWidget
from .setup_messages import (
    numbers_from,
    plain_error,
    plain_stale,
    quality_tooltip,
    quality_word,
    reason_code,
    report_setup_result,
)
from .capture_session import (
    CaptureSession,
    LIVE_OUTPUT_MUTED_NOTICE,
    active_device_identities as _active_device_identities,
    device_label as _device_label,
    device_name as _device_name,
    find_eq_state_owner as _find_eq_state_owner,
    find_processor_owner as _find_processor_owner,
    owner_calibration_context_key as _owner_calibration_context_key,
    processor_sample_rate as _processor_sample_rate,
    restart_processor_for_route as _restart_processor_for_route,
    route_identities_match as _route_identities_match,
    selected_device_identities as _selected_device_identities,
    set_temporary_mute as _set_temporary_mute,
    start_selected_route as _start_selected_route,
    sync_owner_processing_controls as _sync_owner_processing_controls,
)
from .calibration_support import (
    AutoEqCandidate,
    RAINBOW_PASSAGE,
    candidate_metadata as _candidate_metadata,
    chain_settings as _chain_settings,
    filtered_capture_for_analysis as _filtered_capture_for_analysis,
    format_db as _format_db,
    format_percent as _format_percent,
    on_analysis_thread_finished,
    wait_for_analysis_workers,
)
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

RECORDING_DURATION = 10.0  # Seconds

def _eq_chain_settings(owner: Any, incumbent: Preset) -> dict[str, Any]:
    """The chain as the EQ will act in it.

    Raw monitoring and bypass skip the EQ, so the normal path is the one to check.
    """
    chain = _chain_settings(
        owner, full_chain=True, input_pre_filtered=False, preset=incumbent
    )
    chain["processing_mode"] = "normal"
    return chain


# What each capture-quality word means for Auto-EQ.
_QUALITY_MEANING = {
    "Good capture": "clear and steady enough to base a correction on.",
    "Usable capture": "good enough for a gentle correction.",
    "Weak capture": "too unclear to base a correction on.",
}


def _headroom_not_worsened(headroom: Mapping[str, Any]) -> bool:
    """Whether the current EQ is over the headroom rule and the offered EQ leaves no more excess."""
    return headroom.get("status") == "risk_not_worsened"


def _headroom_declined(headroom: Mapping[str, Any]) -> bool:
    """Whether the correction was left out because it would leave less room than the current EQ."""
    return headroom.get("status") == "risk_worse"


class _CaptureAnalysis(NamedTuple):
    """What a finished take needs before the dialog can decide about it."""

    audio: np.ndarray
    silent: bool


def _analyze_capture(raw_audio: np.ndarray, sample_rate: int) -> _CaptureAnalysis:
    """The slow half of a finished take; it runs on a worker thread."""
    audio = _filtered_capture_for_analysis(raw_audio, sample_rate)
    return _CaptureAnalysis(audio, capture_is_silent(audio))


class CalibrationDialog(QDialog):
    """Auto-EQ calibration dialog with target curve selection."""

    # Signal emitted when auto-EQ is applied (emits target curve name)
    auto_eq_applied = Signal(str)

    STEPS = ("Set up", "Record", "Review")
    # Which step each state of the unchanged state machine belongs to.
    STEP_FOR_STATE = {
        "idle": 0,
        "level_check": 1,
        "level_result": 1,
        "recording": 1,
        "capture_analyzing": 1,
        "analyzing": 1,
        "ready": 2,
    }

    @property
    def recording_state(self) -> str:
        return self._recording_state

    @recording_state.setter
    def recording_state(self, state: str) -> None:
        self._recording_state = state
        if "step_header" in self.__dict__:
            self._show_step_for_state()

    def _show_step_for_state(self) -> None:
        step = self.STEP_FOR_STATE.get(self._recording_state, 0)
        self.step_header.set_current(step)
        self.recording_group.setTitle("Review" if step == 2 else "Recording")
        # Compare needs a recording and a proposal, so it belongs to the review alone.
        self.compare_button.setVisible(step == 2)
        self.curve_group.setVisible(step == 0)
        # The passage stays up before recording so it can be read ahead.
        self.instructions_group.setVisible(step < 2)
        # The meter only matters while something is being recorded.
        for widget in (self.progress_bar, self.time_label, self.level_meter_widget):
            widget.setVisible(step == 1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-EQ Voice Tone")
        self.setModal(True)  # Modal dialog - blocks main window

        # Recording state
        self.recording_state = "idle"  # idle, recording, analyzing, ready
        self.audio_data: np.ndarray | None = None
        self.preview_audio_data: np.ndarray | None = None
        self.eq_settings: dict | None = None
        self._candidate_target_metadata: tuple[str, str, str] | None = None
        self._candidate_metadata: dict[str, Any] | None = None
        self._candidate_proposal: AutoEqCandidate | None = None
        self._analysis_incumbent_preset: Preset | None = None
        self._analysis_headroom_chain: dict[str, Any] | None = None
        self._capture_context_key: str | None = None
        self._recent_rms_db: deque[float] = deque(maxlen=LEVEL_HOLD_UPDATES)
        self._recent_peak_db: deque[float] = deque(maxlen=LEVEL_HOLD_UPDATES)
        # The last result: the code that explains it and whether there was one to discard.
        self._outcome_code = "unknown"
        self._had_result = False
        # Whether this result was already reported to the main window.
        self._reported = False
        # Whether this dialog has reported the outcome of an analysis, which a refused retake
        # does not replace.
        self._result_reported = False
        # The level check before the take: its verdict and, if the user went on past it, which.
        self._level_verdict: LevelVerdict | None = None
        self._level_override: str | None = None
        self._recording_duration = RECORDING_DURATION
        self.analysis_worker: AnalysisWorker | CaptureAnalysisWorker | None = None
        self._analysis_workers: list[AnalysisWorker | CaptureAnalysisWorker] = []
        self._analysis_generation = 0
        self._close_requested = False
        self._close_result = False
        self._started_processor = False  # Track if we started processor ourselves
        self._capture_session: CaptureSession | None = None
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
        self.step_header = StepHeader(self.STEPS)
        outer_layout.addWidget(self.step_header)

        self.content_scroll_area, layout = create_scrollable_dialog_body(self)
        self.content_scroll_area.setAccessibleName("Auto-EQ content")
        outer_layout.addWidget(self.content_scroll_area, stretch=1)

        # Tone preset selector group
        self.curve_group = curve_group = QGroupBox("Tone preset")
        curve_layout = QVBoxLayout(curve_group)

        # Curve dropdown
        curve_input_layout = QFormLayout()
        curve_input_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        curve_input_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        curve_label = QLabel("Tone preset:")

        self.curve_combo = QComboBox()
        for key, curve in TARGET_CURVES.items():
            self.curve_combo.addItem(curve.name, key)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        configure_responsive_combo(self.curve_combo)
        curve_input_layout.addRow(curve_label, self.curve_combo)
        bind_label(curve_label, self.curve_combo)

        target_mode_label = QLabel("Matching:")
        self.target_mode_combo = QComboBox()
        for mode, label in TARGET_MODE_LABELS.items():
            self.target_mode_combo.addItem(label, mode)
        self.target_mode_combo.setToolTip(
            "Subtle aims for small tone changes, never more than 2 dB from flat. "
            "Full strength aims for the tone preset's whole curve, up to 5 dB. "
            "Your recording decides whether a correction is applied and how far "
            "headroom lets it go."
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
            "Conservative smoothing ignores narrow dips and peaks in the recording. "
            "Broad is safest but less detailed."
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
        self.instructions_group = instructions_group = QGroupBox(
            "Read this passage aloud"
        )
        instructions_layout = QVBoxLayout(instructions_group)

        # Scrollable text area for Rainbow Passage
        # A label grows with the passage, so no line is cut and nothing scrolls inside the page.
        passage_text = QLabel(RAINBOW_PASSAGE)
        passage_text.setWordWrap(True)
        passage_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        passage_text.setAccessibleName("Auto-EQ passage")
        passage_text.setAccessibleDescription(
            "Read-only passage to speak while Auto-EQ records."
        )
        instructions_layout.addWidget(passage_text)

        layout.addWidget(instructions_group)

        self.mute_notice = QLabel(LIVE_OUTPUT_MUTED_NOTICE)
        self.mute_notice.setWordWrap(True)
        self.mute_notice.setStyleSheet(SUBDUED_TEXT_STYLE)
        layout.addWidget(self.mute_notice)

        # Recording UI group
        self.recording_group = QGroupBox("Recording")
        recording_layout = QVBoxLayout(self.recording_group)
        self.recording_group.setVisible(False)  # Hidden until user clicks Start

        # Progress bar - solid continuous style
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # Hide "0%" text
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setAccessibleName("Auto-EQ recording progress")
        recording_layout.addWidget(self.progress_bar)

        # Time remaining label below progress bar
        self.time_label = QLabel(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.time_label.setAccessibleName("Auto-EQ time remaining")
        recording_layout.addWidget(self.time_label)

        # Level meter (vertical) for real-time validation
        info_layout = QHBoxLayout()
        self.level_meter = LevelMeter(label="Level", show_scale=True)
        self.level_meter_widget = MeterWidget(self.level_meter)
        info_layout.addWidget(self.level_meter_widget)
        recording_layout.addLayout(info_layout)

        # Validation warning label
        self.warning_label = QLabel("Ready to record")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.warning_label.setAccessibleName("Auto-EQ status")
        self.warning_label.setWordWrap(True)
        recording_layout.addWidget(self.warning_label)

        self.diagnostics_group = QGroupBox("Result")
        diagnostics_layout = QVBoxLayout(self.diagnostics_group)
        self.confidence_label = QLabel("Capture quality: --")
        self.error_label = QLabel("Distance from the tone preset: --")
        self.gain_scale_label = QLabel("Headroom check: --")
        self.target_profile_label = QLabel("Tone preset: --")
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
            "The first line rates the recording, not your voice or your microphone. "
            "Hover a line for the numbers behind it."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.eq_compare = EQComparisonWidget()
        diagnostics_layout.addWidget(self.eq_compare)
        diagnostics_layout.addWidget(hint_label)
        self.diagnostics_group.setVisible(False)
        recording_layout.addWidget(self.diagnostics_group)

        # Recording controls
        control_layout = QHBoxLayout()

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        control_layout.addWidget(self.cancel_btn)

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

        layout.addWidget(self.recording_group)
        layout.addStretch(1)

        # Start button (opens recording section)
        self.start_button = QPushButton("Check Level")
        self.start_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.start_button.clicked.connect(self._on_start_clicked)

        # Going on past a level that is not good is a separate, plainly named action.
        self.level_override_btn = QPushButton("Continue Anyway")
        self.level_override_btn.setVisible(False)
        self.level_override_btn.clicked.connect(self._continue_past_level)

        # The actions stay in view while the page above them scrolls.
        control_layout.setContentsMargins(MARGIN_PANEL, 0, MARGIN_PANEL, MARGIN_PANEL)
        control_layout.addStretch(1)
        control_layout.addWidget(self.level_override_btn)
        control_layout.addWidget(self.start_button)
        outer_layout.addLayout(control_layout)
        self._show_step_for_state()

        set_accessible_group(
            (
                (self.retake_btn, "Retake Auto-EQ recording", None),
                (self.cancel_btn, "Cancel Auto-EQ", None),
                (self.level_override_btn, "Continue past the level warning", None),
                (self.level_meter_widget, "Auto-EQ input level", None),
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
        """Handle Start Recording button click based on recording state."""
        if self.recording_state == "idle":
            # Show recording UI and start with the level check
            self.recording_group.setVisible(True)
            self._level_override = None
            self._start_recording("level_check")
        elif self.recording_state == "level_result":
            verdict = self._level_verdict
            self._start_recording(
                "recording" if verdict is not None and verdict.code == "good" else "level_check"
            )
        elif self.recording_state in ("recording", "level_check"):
            # User tried to stop early - not allowed
            QMessageBox.information(
                self,
                "Recording",
                "Please let the recording finish: Auto-EQ needs the full 10 seconds.",
            )
        elif self.recording_state == "analyzing":
            return
        elif self.recording_state == "ready":
            if self.eq_settings is not None:
                self._apply_eq_settings()
            else:
                self._reset_recording_ui()
                self._start_recording("level_check")

    def _continue_past_level(self) -> None:
        """Record the passage although the level check did not say the level was good."""
        verdict = self._level_verdict
        if self.recording_state != "level_result" or verdict is None or not verdict.can_override:
            return
        self._level_override = verdict.code
        self._start_recording("recording")

    def _apply_eq_settings(self):
        """Apply auto-EQ settings to main window and close dialog."""
        if self.recording_state != "ready" or self.eq_settings is None:
            return
        if self._candidate_target_metadata is None:
            return
        eq_settings = deepcopy(self.eq_settings)
        if DEBUG:
            logger.debug("Applying EQ settings")

        # Find the application state owner through the dialog parent chain.
        parent = _find_eq_state_owner(self.parent())

        if not parent:
            QMessageBox.critical(
                self, "Error", plain_error("Could not find audio processor")
            )
            return

        identity_error = self._candidate_identity_error(eq_settings, parent)
        if identity_error is not None:
            QMessageBox.critical(
                self,
                "Recording out of date",
                f"{plain_stale(identity_error)} Nothing was changed.",
            )
            return

        get_typed_eq_settings = getattr(parent.eq_state, "get_eq_settings", None)
        if not callable(get_typed_eq_settings):
            QMessageBox.critical(
                self,
                "Error",
                "AudioForge could not check the new EQ, so nothing was changed. "
                "Close this window and open it again.",
            )
            return
        apply_configuration = getattr(parent, "apply_processing_configuration", None)
        if not callable(apply_configuration):
            QMessageBox.critical(
                self,
                "Error",
                "AudioForge cannot apply the EQ in one step, so nothing was changed. "
                "Close this window and open it again.",
            )
            return
        mode_getter = getattr(parent, "_processing_mode", None)
        if not callable(mode_getter):
            QMessageBox.critical(
                self,
                "Error",
                "AudioForge could not read the current processing mode, so nothing was "
                "changed. Close this window and open it again.",
            )
            return
        try:
            snapshot_preset = parent._get_current_preset()
            snapshot_mode = str(mode_getter())
        except Exception as error:
            logger.warning(
                "Failed to snapshot complete configuration before candidate apply",
                exc_info=True,
            )
            QMessageBox.critical(
                self,
                "Error",
                "AudioForge could not save your current settings first, so nothing was "
                f"changed ({error}).",
            )
            return
        if not isinstance(snapshot_preset, Preset):
            QMessageBox.critical(
                self,
                "Error",
                "AudioForge could not save your current settings first, so nothing was changed.",
            )
            return

        try:
            proposal = self._candidate_proposal
            if proposal is None:
                raise ValueError("validated EQ proposal is unavailable")
            candidate_preset = proposal.proposed_preset
            apply_configuration(candidate_preset, processing_mode=snapshot_mode)
            parent.eq_state.set_auto_eq_diagnostics(eq_settings)
            accepted_eq = get_typed_eq_settings()
            if not isinstance(accepted_eq, EQSettings):
                raise TypeError("EQ state returned invalid typed settings")
            expected_eq = candidate_preset.eq
            if accepted_eq.to_dict() != expected_eq.to_dict():
                raise ValueError("EQ state did not install the validated candidate")
            eq_settings.update(expected_eq.to_dict())
        except Exception as error:
            restore_error = None
            try:
                apply_configuration(snapshot_preset, processing_mode=snapshot_mode)
            except Exception as restore_exc:
                restore_error = restore_exc
                logger.warning(
                    "Failed to restore processing configuration after candidate apply failure",
                    exc_info=True,
                )
            logger.warning("Failed to apply Auto-EQ candidate", exc_info=True)
            message = (
                f"The EQ could not be applied, so your settings were left as they were ({error})."
            )
            if restore_error is not None:
                message = (
                    f"The EQ could not be applied ({error}), and your previous settings "
                    f"could not be fully put back ({restore_error})."
                )
            QMessageBox.critical(self, "Error", message)
            self._report("failed", ["apply_failed"], [message])
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
        self._report("applied", [self._outcome_code], [self.warning_label.text()])

        if DEBUG:
            logger.debug(
                "EQ settings applied, signal emitted for curve=%s", target_curve
            )

        # Close dialog
        self.accept()

    def _compare_recording(self) -> None:
        preview_audio_data = self.preview_audio_data
        if (
            self.recording_state != "ready"
            or self.audio_data is None
            or preview_audio_data is None
            or self.eq_settings is None
        ):
            return
        owner = _find_eq_state_owner(self.parent())
        if owner is None:
            return
        error = self._candidate_identity_error(self.eq_settings, owner)
        if error:
            QMessageBox.warning(self, "Recording out of date", plain_stale(error))
            return
        from .listening_comparison_dialog import ListeningComparisonDialog

        try:
            proposal = self._candidate_proposal
            if proposal is None:
                raise ValueError("validated EQ proposal is unavailable")
            chain = proposal.chain
        except (TypeError, ValueError) as error:
            QMessageBox.warning(
                self,
                "Comparison unavailable",
                f"The comparison could not be prepared ({error}). Record again.",
            )
            return
        dialog = ListeningComparisonDialog(
            audio_data=preview_audio_data,
            sample_rate=_processor_sample_rate(owner),
            current_settings=proposal.incumbent_preset.eq.to_dict(),
            proposed_settings=proposal.proposed_preset.eq.to_dict(),
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
        proposal = self._candidate_proposal
        if proposal is None:
            return "Validated Auto-EQ proposal is unavailable"
        headroom = eq_settings.get("headroom_validation")
        if (
            not isinstance(headroom, Mapping)
            or headroom.get("authoritative") is not True
            or (headroom.get("safe") is not True and not _headroom_not_worsened(headroom))
        ):
            return "Auto-EQ candidate lacks safe native full-chain validation"
        try:
            validated_eq = EQSettings.from_dict(
                dict(eq_settings.get("validated_candidate_eq") or {})
            )
        except (TypeError, ValueError):
            return "Validated Auto-EQ candidate is missing"
        if validated_eq.to_dict() != proposal.proposed_preset.eq.to_dict():
            return "Auto-EQ candidate changed after headroom validation"
        candidate = eq_settings.get("_candidate")
        expected = self._candidate_metadata
        if not isinstance(candidate, Mapping) or not isinstance(expected, Mapping):
            return "Auto-EQ candidate identity is missing"
        if candidate.get("scope") != "eq_only":
            return "Auto-EQ candidate scope is invalid"
        if tuple(candidate.get("allowed_scope") or ()) != ("eq",):
            return "Auto-EQ candidate scope is incomplete"
        if candidate.get("incumbent_signature") != proposal.incumbent_signature:
            return "Auto-EQ incumbent identity is unavailable"
        if candidate.get("configuration_signature") != proposal.proposed_signature:
            return "Auto-EQ proposal identity is unavailable"
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
            return "The audio route or input settings changed after capture"
        try:
            incumbent = parent._get_current_preset()
            if not isinstance(incumbent, Preset):
                return "Current processing configuration is unavailable"
            current_chain = _eq_chain_settings(parent, incumbent)
            if not proposal.matches(incumbent, current_chain):
                return "Processing configuration changed after analysis"
        except Exception:
            return "Current processing configuration cannot be verified"
        generation = capture.get("generation")
        if generation != self._analysis_generation:
            return "Auto-EQ candidate is stale"
        return None

    def _start_recording(self, phase: str = "recording"):
        """Start the non-blocking level check or the passage take."""
        if DEBUG:
            logger.debug("Start recording clicked")

        self._clear_eq_candidate()
        self._cancel_analysis_workers()

        # Get parent's processor (MainWindow has it)
        parent = _find_processor_owner(self.parent())

        if not parent:
            QMessageBox.critical(
                self, "Error", plain_error("Could not find audio processor")
            )
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
                        "Recording canceled. The microphone and speakers that were "
                        "already running were kept."
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
                        "AudioForge could not switch to your selected devices.\n\n"
                        f"{e}\n\nCheck that they are connected and not used by another app.",
                    )
                    return
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
                    "AudioForge could not start the microphone and speakers.\n\n"
                    f"{e}\n\nCheck that they are connected and not used by another app, "
                    "then try again.",
                )
                return

        self._capture_context_key = _owner_calibration_context_key(parent)
        self.recording_state = phase
        self.level_override_btn.setVisible(False)
        self._recording_duration = (
            LEVEL_CHECK_SECONDS if phase == "level_check" else RECORDING_DURATION
        )
        self.time_label.setText(f"Time remaining: {self._recording_duration:.0f}s")
        self.start_button.setText("Checking Level..." if phase == "level_check" else "Recording...")
        self.start_button.setEnabled(False)  # Prevent early stop

        # Lock Auto-EQ target controls during recording
        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)

        # Let DSP loop settle without blocking the UI thread.
        self._capture_start_timer.start(100)

        self.warning_label.setText(
            "Read the first lines of the passage at your normal volume for 3 seconds."
            if phase == "level_check"
            else "Recording... Read the passage aloud at your normal volume."
        )
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))

    def _begin_recording_capture(self):
        """Start Rust-side recording and poll progress from the main Qt thread."""
        self._capture_start_timer.stop()
        if self.recording_state not in ("recording", "level_check"):
            return

        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_recording_failed(plain_error("Could not find audio processor"))
            return

        self._recent_rms_db.clear()
        self._recent_peak_db.clear()
        self._capture_session = CaptureSession(parent, _TEMPORARY_MUTE_REASON)
        try:
            self._capture_session.start(self._recording_duration, before_cleanup=True)
        except Exception as e:
            self._on_recording_failed(plain_error(f"Recording error: {e}"))
            return

        self.recording_timer.start()
        if DEBUG:
            logger.debug("Started main-thread recording capture")

    def _poll_recording_progress(self):
        """Poll recording state from the main Qt thread."""
        if self.recording_state not in ("recording", "level_check"):
            self.recording_timer.stop()
            return

        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_recording_failed(plain_error("Could not find audio processor"))
            return

        try:
            progress_float = float(parent.processor.recording_progress())
            failure = (
                self._capture_session.failure_reason(progress_float)
                if self._capture_session is not None
                else None
            )
            if failure is not None:
                self.recording_timer.stop()
                self._on_recording_failed(plain_error(failure))
                return
            progress_pct = int(progress_float * 100)
            self._on_progress_update(progress_pct)
            self._on_time_remaining(
                max(0.0, self._recording_duration * (1.0 - progress_float))
            )
            self._on_level_update(
                float(parent.processor.recording_level_db()),
                float(parent.processor.get_input_peak_db()),
            )

            if progress_pct >= 100 or parent.processor.is_recording_complete():
                self.recording_timer.stop()
                session, self._capture_session = self._capture_session, None
                audio = (
                    session.stop_recording()
                    if session is not None
                    else parent.processor.stop_raw_recording()
                )
                if audio is None:
                    self._on_recording_failed(plain_error("Recording failed - no audio data"))
                    return
                audio_array = np.asarray(audio, dtype=np.float32)
                self._on_recording_complete(audio_array)
        except Exception as e:
            self.recording_timer.stop()
            self._on_recording_failed(plain_error(f"Recording error: {e}"))

    def _on_progress_update(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def _on_time_remaining(self, seconds: float):
        """Update time remaining label."""
        if seconds > 0:
            self.time_label.setText(f"Time remaining: {seconds:.0f}s")
        else:
            self.time_label.setText("Complete")
            self.time_label.setStyleSheet(message_text_style("ok", strong=True))

    def _on_level_update(self, rms_db: float, peak_db: float):
        """Update level meter with validation warnings."""
        self.level_meter.set_levels(rms_db, peak_db)

        # The latest block dips on every pause between words; judge the loudest of a full
        # window of recent ones, and say nothing about the level until there is one.
        self._recent_rms_db.append(rms_db)
        self._recent_peak_db.append(peak_db)
        verdict = live_verdict(
            self._recent_rms_db,
            self._recent_peak_db,
            again="check again" if self.recording_state == "level_check" else "record again",
        )
        if verdict is not None:
            self.warning_label.setText(verdict.message)
            self.warning_label.setStyleSheet(message_text_style(verdict.state, strong=True))

    def _finish_level_check(self, audio_data: np.ndarray) -> None:
        """Judge the check from its samples and offer the next step the verdict allows."""
        try:
            sample_rate = _processor_sample_rate(_find_processor_owner(self.parent()))
        except Exception:
            sample_rate = 48_000
        verdict = verdict_from_audio(np.asarray(audio_data, dtype=np.float32), sample_rate)
        choices = level_choices(verdict, "Record Passage")
        self._level_verdict = verdict
        self.recording_state = "level_result"
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.time_label.setText("Level checked")
        self.warning_label.setText(f"{verdict.message} {choices.note}".strip())
        self.warning_label.setStyleSheet(message_text_style(verdict.state, strong=True))
        self.start_button.setText(choices.primary)
        self.start_button.setEnabled(True)
        self.level_override_btn.setVisible(choices.override is not None)
        if choices.override is not None:
            self.level_override_btn.setText(choices.override)

    def _on_recording_complete(self, audio_data: np.ndarray):
        """Handle recording completion."""
        if self.recording_state == "level_check":
            self._finish_level_check(audio_data)
            return
        if DEBUG:
            logger.debug("Recording complete: %d samples", len(audio_data))

            rms = np.mean(audio_data**2) ** 0.5
            peak_db = 20 * np.log10(max(np.abs(audio_data).max(), 1e-6))
            rms_db = 20 * np.log10(max(rms, 1e-6))
            logger.debug("Audio stats - Peak: %.1f dB, RMS: %.1f dB", peak_db, rms_db)

        raw_audio = np.ascontiguousarray(
            np.asarray(audio_data, dtype=np.float32).reshape(-1).copy()
        )
        try:
            sample_rate = _processor_sample_rate(_find_processor_owner(self.parent()))
        except Exception:
            sample_rate = 48_000
        # Say so now, while the fix is obvious, as Voice Setup does for the same takes.
        if not np.isfinite(raw_audio).all():
            self._reject_capture(
                "The recording held invalid audio data, which usually means the audio "
                "driver glitched. Restart processing and record again."
            )
            return
        if raw_capture_is_clipped(raw_audio, sample_rate):
            self._reject_capture(clipped_message(raw_audio, sample_rate))
            return
        self._cancel_analysis_workers()
        generation = self._analysis_generation + 1
        self._analysis_generation = generation
        self.recording_state = "capture_analyzing"
        self.progress_bar.setRange(0, 0)
        self.start_button.setEnabled(False)
        self.warning_label.setText("Checking the recording...")
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        worker = CaptureAnalysisWorker(partial(_analyze_capture, raw_audio, sample_rate))
        self.analysis_worker = worker
        self._analysis_workers.append(worker)
        worker.result_ready.connect(
            lambda result, token=generation: self._on_capture_analyzed(
                result, raw_audio, token
            )
        )
        worker.failed.connect(
            lambda error, token=generation: self._on_capture_analysis_failed(error, token)
        )
        worker.finished.connect(
            lambda finished_worker=worker: self._on_analysis_thread_finished(
                finished_worker
            )
        )
        worker.start()

    def _on_capture_analysis_failed(self, error: str, generation: int) -> None:
        if generation != self._analysis_generation or self._close_requested:
            return
        self._on_recording_failed(plain_error(f"Recording error: {error}"))

    def _on_capture_analyzed(
        self, analysis: _CaptureAnalysis, raw_audio: np.ndarray, generation: int
    ) -> None:
        """Decide about a finished take once its post-processing is done."""
        if generation != self._analysis_generation or self._close_requested:
            return
        if analysis.silent:
            self._reject_capture(
                "No sound came from the microphone. Check that it is not muted and "
                "that the right input is selected, then record again."
            )
            return
        self.preview_audio_data = raw_audio
        self.audio_data = analysis.audio
        self._clear_eq_candidate()
        self.recording_state = "analyzing"
        self.progress_bar.setRange(0, 0)  # the analysis reports no steady progress

        # Update UI
        self.start_button.setText("Analyzing...")
        self.start_button.setEnabled(False)
        self.retake_btn.setVisible(True)

        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)

        # Show completion message
        self.warning_label.setText("Recording complete. Checking it against the tone preset.")
        self.warning_label.setStyleSheet(message_text_style("ok", strong=True))

        if DEBUG:
            logger.debug("Audio captured; starting analysis")

        if DEBUG:
            logger.debug("Starting analysis worker")
        self._start_analysis()

    def _on_recording_failed(self, error: str):
        """Handle recording failure."""
        self._reset_recording_ui()
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        self._report("failed", ["recording_failed"], [error])

    def _reject_capture(self, message: str) -> None:
        """Refuse a finished take and leave the dialog ready to record it again."""
        self._reset_recording_ui()
        self.warning_label.setText(message)
        self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        self._report("refused", ["recording_rejected"], [message])

    def _start_analysis(self):
        """Start analysis worker."""
        if self.audio_data is None:
            self._on_analysis_failed("There is no recording to check. Record again.")
            return

        self._cancel_analysis_workers()

        # Get parent's processor for sample rate
        parent = _find_processor_owner(self.parent())

        if not parent:
            self._on_analysis_failed(plain_error("Could not find audio processor"))
            return

        try:
            sample_rate = _processor_sample_rate(parent)
            target_preset = self.get_selected_curve()
            target_mode = self.get_selected_target_mode()
            smoothing_strength = self.get_selected_smoothing_strength()
        except Exception as error:
            self._on_analysis_failed(
                f"Auto-EQ could not start checking the recording ({error}). "
                "Close and reopen this window, then try again."
            )
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
        if self.preview_audio_data is None:
            self._on_analysis_failed(
                "The unprocessed recording is missing, so the safety check could not run. "
                "Record again."
            )
            return
        try:
            incumbent = parent._get_current_preset()
            if not isinstance(incumbent, Preset):
                raise TypeError("current processing configuration is unavailable")
            incumbent = Preset.from_dict(incumbent.to_dict())
            chain_settings = _eq_chain_settings(parent, incumbent)
        except Exception as error:
            self._on_analysis_failed(
                f"AudioForge could not read your current settings for the safety check ({error}). "
                "Close and reopen this window, then try again."
            )
            return
        self._analysis_incumbent_preset = incumbent
        self._analysis_headroom_chain = deepcopy(chain_settings)

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
                headroom_candidate_base_eq_settings=incumbent.eq.to_dict(),
                headroom_audio_data=self.preview_audio_data,
                headroom_chain_settings=chain_settings,
            )
        except Exception as error:
            self._on_analysis_failed(
                f"Auto-EQ could not start checking the recording ({error}). "
                "Close and reopen this window, then try again."
            )
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
            lambda error, token=generation: self._on_analysis_failed(
                plain_error(error), token
            )
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
            self._on_analysis_failed("Auto-EQ lost track of its settings. Record again.")
            return
        if DEBUG:
            logger.debug("Analysis complete")
            logger.debug(
                "Band gains: %s", [round(g, 1) for g in eq_settings["band_gains"]]
            )
            max_gain = max(abs(g) for g in eq_settings["band_gains"])
            logger.debug("Max correction: %.1f dB", round(max_gain, 1))

        eq_settings = deepcopy(eq_settings)
        headroom = eq_settings.get("headroom_validation")
        validated_eq_payload = eq_settings.get("validated_candidate_eq")
        headroom_error = None
        if not isinstance(headroom, Mapping):
            headroom_error = "full-chain headroom validation result is unavailable"
        elif headroom.get("authoritative") is not True:
            detail = headroom.get("reason")
            headroom_error = str(detail or "native full-chain headroom validation is unavailable")
        elif _headroom_declined(headroom):
            pass  # the analysis left the correction out and says why; there is nothing to apply
        elif headroom.get("safe") is not True and not _headroom_not_worsened(headroom):
            headroom_error = "the merged candidate did not pass full-chain headroom validation"
        elif not isinstance(validated_eq_payload, Mapping):
            headroom_error = "validated merged EQ candidate is unavailable"
        elif (
            self._analysis_incumbent_preset is None
            or self._analysis_headroom_chain is None
        ):
            headroom_error = "incumbent processing snapshot is unavailable"

        proposal = None
        if (
            headroom_error is None
            and isinstance(validated_eq_payload, Mapping)
            and self._analysis_incumbent_preset is not None
            and self._analysis_headroom_chain is not None
        ):
            try:
                validated_eq = EQSettings.from_dict(dict(validated_eq_payload))
                expected_eq = build_eq_candidate_settings(
                    self._analysis_incumbent_preset.eq,
                    eq_settings.get("band_freqs", ()),
                    eq_settings.get("band_gains", ()),
                    eq_settings.get("band_qs", ()),
                    layer="correction",
                    enabled=bool(eq_settings.get("enabled", True)),
                )
                if validated_eq.to_dict() != expected_eq.to_dict():
                    raise ValueError(
                        "headroom candidate does not match the final correction"
                    )
                proposal = AutoEqCandidate.create(
                    self._analysis_incumbent_preset,
                    validated_eq,
                    self._analysis_headroom_chain,
                )
            except (TypeError, ValueError) as error:
                headroom_error = str(error)

        candidate = deepcopy(self._candidate_metadata or {})
        if isinstance(candidate, dict):
            candidate["verified_stages"] = []
            if proposal is not None:
                candidate["incumbent_signature"] = proposal.incumbent_signature
                candidate["configuration_signature"] = proposal.proposed_signature
        eq_settings["_candidate"] = candidate
        apply_recommended = bool(eq_settings.get("apply_recommended", False))
        if headroom_error is not None:
            reasons = list(eq_settings.get("abstention_reasons") or [])
            reason = f"Full-chain headroom validation failed: {headroom_error}"
            if reason not in reasons:
                reasons.append(reason)
            eq_settings["abstention_reasons"] = reasons
            eq_settings["recommendation_status"] = "abstain"
            eq_settings["apply_recommended"] = False
            apply_recommended = False
        elif isinstance(headroom, Mapping) and _headroom_declined(headroom):
            # Left out on purpose: the current EQ stays whatever else the result says.
            eq_settings["recommendation_status"] = "abstain"
            eq_settings["apply_recommended"] = False
            apply_recommended = False
        if apply_recommended:
            self._candidate_proposal = proposal
            self.eq_settings = eq_settings
        else:
            self._clear_eq_candidate()
        self.recording_state = "ready"
        self._had_result = True
        self._reported = False
        self.compare_button.setEnabled(apply_recommended)
        # The analysis treats a missing status as "apply"; so does the wording.
        explanation = explain_auto_eq(
            {
                **eq_settings,
                "recommendation_status": eq_settings.get("recommendation_status")
                or ("apply" if apply_recommended else "abstain"),
            }
        )
        text = f"{explanation.summary}. {' '.join(explanation.details)}"
        if apply_recommended:
            largest = max(abs(g) for g in eq_settings["band_gains"])
            text += f" Largest change {largest:.1f} dB."
        advice = headroom_advice(headroom, self._analysis_headroom_chain, apply_recommended)
        if advice:
            text += f" {advice}"
        self.warning_label.setText(text)
        raw_reasons = [
            str(reason)
            for key in ("abstention_reasons", "recommendation_reasons")
            for reason in eq_settings.get(key) or ()
        ]
        # The sentence above is for the person; the analysis's own words stay on hover.
        self.warning_label.setToolTip("; ".join(raw_reasons))
        self.warning_label.setStyleSheet(
            message_text_style("ok" if apply_recommended else "warn", strong=True)
        )
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._show_analysis_diagnostics(eq_settings)
        self.start_button.setText(
            "Apply EQ Settings" if apply_recommended else "Record Again"
        )
        self.start_button.setEnabled(True)
        self._outcome_code = explanation.outcome_code
        if not apply_recommended:
            self._report(
                "refused",
                [explanation.outcome_code, *(reason_code(reason) for reason in raw_reasons)],
                [text],
                eq_settings,
            )
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
        word, state = quality_word(capture_confidence)
        applied = bool(eq_settings.get("apply_recommended", False))
        before = eq_settings.get("validation_before_error_db")
        after = eq_settings.get("validation_after_error_db")
        scale = eq_settings.get("validation_gain_scale")
        target_profile = target_profile_text(eq_settings.get("target_profile"))
        residual = eq_settings.get("residual_regularization") or {}
        used_fallback = bool(eq_settings.get("used_spectrum_fallback", False))
        headroom = eq_settings.get("headroom_validation") or {}
        # A correction left out is shown with the figures that ruled it out.
        headroom_after = (
            headroom.get("before" if _headroom_declined(headroom) else "after")
            if isinstance(headroom, dict)
            else None
        )
        headroom_safe = (
            bool(headroom.get("safe", False)) if isinstance(headroom, dict) else False
        )
        headroom_advisory = (
            bool(headroom.get("advisory", False))
            if isinstance(headroom, dict)
            else False
        )
        headroom_scale = (
            headroom.get("gain_scale") if isinstance(headroom, dict) else None
        )

        self.confidence_label.setText(f"{word}: {_QUALITY_MEANING[word]}")
        self.confidence_label.setToolTip(
            quality_tooltip(
                capture_confidence,
                f"How well the recording supports each frequency band: {_format_percent(eq_confidence)}. "
                f"Safety and benefit of the correction: {_format_percent(validation_confidence)} "
                "(this part starts at 35% and rises as the correction gets closer to the tone preset, "
                f"so it says little about the recording). Overall score: {_format_percent(confidence)}.",
            )
        )
        self.confidence_label.setStyleSheet(status_chip_style(state))
        self.error_label.setText(
            "Distance from the tone preset: "
            + (
                f"{_format_db(before)} now, {_format_db(after)} with this correction."
                if applied
                else f"{_format_db(before)}; no correction was applied."
            )
        )
        error_details = f"The correction is used at {_format_percent(scale)} of its fitted strength."
        if isinstance(residual, dict) and "max_regularized_correction_db" in residual:
            requested = _format_db(residual.get("max_requested_correction_db"))
            regularized = _format_db(residual.get("max_regularized_correction_db"))
            error_details += (
                f" Smoothing took the largest change from {requested} to {regularized}."
            )
        self.error_label.setToolTip(error_details)
        self.error_label.setStyleSheet(
            status_chip_style("ok" if after is not None else "idle")
        )
        headroom_text = "not available"
        headroom_details = ""
        self.gain_scale_label.setStyleSheet(status_chip_style("info"))
        if isinstance(headroom_after, dict):
            pre_tp_headroom = _format_db(
                headroom_after.get("pre_limiter_true_peak_headroom_db")
            )
            limiter_gr = _format_db(headroom_after.get("limiter_gain_reduction_db"))
            true_peak_gr = _format_db(
                headroom_after.get("true_peak_limiter_gain_reduction_db")
            )
            if headroom_advisory:
                headroom_text = "estimate only"
            elif (
                isinstance(headroom, dict)
                and headroom.get("status") == "unavailable"
            ):
                headroom_text = "not available"
            elif headroom_safe:
                headroom_text = "passed"
            elif _headroom_not_worsened(headroom):
                headroom_text = "not met even without the EQ"
            elif _headroom_declined(headroom):
                headroom_text = "the correction would leave less room than your current EQ"
            else:
                headroom_text = "the correction may push the limiter"
            headroom_details = (
                f"Room left before the loudest peaks reach the limit: {pre_tp_headroom}. "
                f"The limiter would turn peaks down by {limiter_gr}, "
                f"and the final peak limiter by {true_peak_gr}."
            )
            self.gain_scale_label.setStyleSheet(
                status_chip_style("info" if headroom_safe else "warn")
            )
            if headroom_scale is not None and float(headroom_scale) < 1.0:
                headroom_details += (
                    f" The correction was reduced to {_format_percent(headroom_scale)} to fit."
                )
        self.gain_scale_label.setText(f"Headroom check: {headroom_text}.")
        self.gain_scale_label.setToolTip(headroom_details)
        self.target_profile_label.setText(
            f"Tone preset: {self.curve_combo.currentText()}"
        )
        self.target_profile_label.setToolTip(
            f"Tone preset and matching: {target_profile}."
            + (
                " The recording had too few clear voiced sounds, so a rougher average was used."
                if used_fallback
                else ""
            )
        )
        self.target_profile_label.setStyleSheet(status_chip_style("info"))
        proposal = self._candidate_proposal
        self.eq_compare.setVisible(proposal is not None and applied)
        if proposal is not None and applied:
            try:
                sample_rate = float(_processor_sample_rate(_find_processor_owner(self.parent())))
            except Exception:
                sample_rate = 48_000.0
            mode = getattr(_find_eq_state_owner(self.parent()), "_processing_mode", lambda: "")()
            self.eq_compare.set_curves(
                proposal.incumbent_preset.eq,
                proposal.proposed_preset.eq,
                sample_rate,
                mode if mode in ("raw", "bypass") else "",
            )
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
        self.progress_bar.setRange(0, 100)
        self.warning_label.setText(error)
        self.warning_label.setToolTip("")
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        self.diagnostics_group.setVisible(False)
        self.start_button.setText("Record Again")
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(False)
        self.target_mode_combo.setEnabled(False)
        self.smoothing_combo.setEnabled(False)
        self._report("failed", ["analysis_failed"], [error])

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
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._recording_duration = RECORDING_DURATION
        self.time_label.setText(f"Time remaining: {RECORDING_DURATION:.0f}s")
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.level_meter.set_unavailable()
        self._had_result = False
        self.warning_label.setToolTip("")
        self.warning_label.setText("Ready to record")
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.diagnostics_group.setVisible(False)
        self.start_button.setText("Check Level")
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(False)
        self._level_verdict = self._level_override = None
        self.level_override_btn.setVisible(False)
        self.curve_combo.setEnabled(True)
        self.target_mode_combo.setEnabled(True)
        self.smoothing_combo.setEnabled(True)

    def _report(
        self,
        outcome: str,
        reason_codes: list[str],
        messages: list[str],
        result: Mapping[str, Any] | None = None,
    ) -> None:
        """Tell the main window how this run ended, for the diagnostics export."""
        refused_take = "recording_rejected" in reason_codes
        if refused_take and self._result_reported:
            return
        if not refused_take and outcome != "failed":
            self._result_reported = True
        measured: dict[str, Any] = numbers_from(
            result or self.eq_settings or {},
            (
                "analysis_confidence",
                "capture_confidence",
                "eq_confidence",
                "validation_confidence",
                "validation_before_error_db",
                "validation_after_error_db",
                "validation_gain_scale",
                "apply_recommended",
            ),
        )
        if self._level_override is not None:
            measured["level_override"] = self._level_override
        self._reported = True
        report_setup_result(
            _find_eq_state_owner(self.parent()),
            "auto_eq",
            outcome,
            reason_codes=reason_codes,
            messages=messages,
            numbers=measured,
        )

    def _clear_eq_candidate(self) -> None:
        """Discard the only candidate that may be applied."""
        self.eq_settings = None
        self._candidate_proposal = None
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
        session, self._capture_session = self._capture_session, None
        if session is not None:
            session.cleanup()

    def _cancel_analysis_workers(self) -> None:
        """Cancel work without dropping ownership of a running QThread."""
        self._analysis_generation += 1
        for worker in tuple(self._analysis_workers):
            if worker.isRunning():
                worker.stop()
        self.analysis_worker = None

    def _on_analysis_thread_finished(self, worker) -> None:
        on_analysis_thread_finished(self, worker)

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

    # A real method, so Qt drops the aboutToQuit connection when the dialog is destroyed.
    def _wait_for_analysis_workers(self) -> None:
        wait_for_analysis_workers(self)

    def _request_close(self, accepted: bool) -> None:
        if self._close_requested:
            return
        if not accepted and self._had_result and not self._reported:
            self._report(
                "cancelled",
                ["closed_before_applying"],
                ["Auto-EQ was closed before the correction was applied."],
            )
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
            str: Selected target key from the built-in curve catalog
        """
        return self.curve_combo.currentData()

    def get_selected_target_mode(self):
        """Return the selected target behavior."""
        return self.target_mode_combo.currentData()

    def get_selected_smoothing_strength(self):
        """Return the selected Auto-EQ smoothing strength."""
        return self.smoothing_combo.currentData()
