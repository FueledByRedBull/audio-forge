"""
Recording dialog for the Auto-EQ feature

DEBUG: Added terminal logging for the recording workflow
"""

import logging
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QVBoxLayout
import numpy as np

from ..config import (
    EQSettings,
    Preset,
    TARGET_CURVES,
    build_eq_candidate_settings,
)
from ..analysis.noise_reference import capture_is_silent
from ..analysis.voice_setup import raw_capture_is_clipped
from .analysis_worker import (
    AnalysisWorker,
    CaptureAnalysis,
    CaptureAnalysisWorker,
    start_capture_analysis,
)
from .auto_eq_explanation import (
    TARGET_MODE_LABELS,
    explain_auto_eq,
    headroom_advice,
    target_profile_text,
)
from .calibration_view import CalibrationView
from .components import session_is_ending
from .layout_constants import configure_resizable_dialog
from .level_check import (
    LEVEL_CHECK_SECONDS,
    LEVEL_HOLD_UPDATES,
    LevelVerdict,
    clipped_message,
    level_choices,
    live_verdict,
    verdict_from_audio,
)
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
    device_label as _device_label,
    device_name as _device_name,
    find_processor_owner as _find_processor_owner,
    owner_calibration_context_key as _owner_calibration_context_key,
    processor_sample_rate as _processor_sample_rate,
)
from .engine_session import (
    AudioLease,
    AudioUnavailable,
    EngineSession,
    RouteMismatch,
    unavailable_text,
)
from .processing_configuration import ProcessingConfiguration
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
from .quick_scene import QML_DIR, load_scene, quick_theme
from .quick_shell import register_qml_types

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


def _analyze_capture(raw_audio: np.ndarray, sample_rate: int) -> CaptureAnalysis:
    """The slow half of a finished take; it runs on a worker thread."""
    audio = _filtered_capture_for_analysis(raw_audio, sample_rate)
    return CaptureAnalysis(audio, capture_is_silent(audio))


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
        if "view_model" in self.__dict__:
            self._show_step_for_state()

    def _show_step_for_state(self) -> None:
        step = self.STEP_FOR_STATE.get(self._recording_state, 0)
        self.view_model.set_page(
            step=step,
            recordingTitle="Review" if step == 2 else "Recording",
            # Compare needs a recording and a proposal, so it belongs to the review alone.
            compareShown=step == 2,
            curveShown=step == 0,
            # The passage stays up before recording so it can be read ahead.
            instructionsShown=step < 2,
            # Only a recording mutes the live output.
            mutedShown=self._recording_state in ("level_check", "recording"),
            # The meter only matters while something is being recorded.
            meterShown=step == 1,
        )

    def __init__(self, parent=None, *, configuration: ProcessingConfiguration | None = None):
        super().__init__(parent)
        self.setWindowTitle("Auto-EQ Voice Tone")
        self.setModal(True)  # Modal dialog - blocks main window
        # Without one, Auto-EQ can measure and recommend but not apply.
        self._configuration = configuration

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
        self._audio_lease: AudioLease | None = None
        self._capture_session: CaptureSession | None = None
        self._capture_start_timer = QTimer(self)
        self._capture_start_timer.setSingleShot(True)
        self._capture_start_timer.timeout.connect(self._begin_recording_capture)
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self._poll_recording_progress)
        self._load_error = self._setup_ui()
        if self._load_error is not None:
            # Nothing is recorded; exec() says why and closes.
            logger.error("Auto-EQ calibration did not open: %s", self._load_error)
            return
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_analysis_workers)
        # The scene's own minimum grows with the text size; the dialog opens at least that wide.
        minimum_width = max(480, int(self.quick_view.rootObject().property("minimumWidth")))
        configure_resizable_dialog(
            self,
            preferred_width=max(720, minimum_width),
            preferred_height=800,
            minimum_width=minimum_width,
            minimum_height=360,
        )

    @property
    def load_error(self) -> str | None:
        """Why the scene did not load, or None; a dialog that did not load records nothing."""
        return self._load_error

    def exec(self) -> int:
        if self._load_error is not None:
            QMessageBox.critical(
                self.parentWidget(),
                self.windowTitle(),
                f"This window could not open.\n\n{self._load_error}",
            )
            return int(QDialog.DialogCode.Rejected)
        return super().exec()

    def _setup_ui(self) -> str | None:
        """Build the view model and load the scene; return why it did not load, or None."""
        self.view_model = CalibrationView(
            steps=self.STEPS,
            curves=tuple((key, curve.name) for key, curve in TARGET_CURVES.items()),
            modes=tuple(TARGET_MODE_LABELS.items()),
            smoothings=(
                ("conservative", "Conservative"),
                ("balanced", "Balanced"),
                ("broad", "Broad"),
            ),
            passage=RAINBOW_PASSAGE,
            curve_changed=self._on_curve_changed,
            start=self._on_start_clicked,
            override=self._continue_past_level,
            cancel=self._on_cancel_clicked,
            retake=self._on_retake_clicked,
            compare=self._compare_recording,
            parent=self,
        )
        self.level_meter = self.view_model.level_meter
        self.eq_compare = self.view_model.eq_compare
        self.view_model.set_page(timeText=f"Time remaining: {RECORDING_DURATION:.0f}s")
        self._show_step_for_state()
        # Initialize with first curve description
        self._on_curve_changed(0)

        register_qml_types()
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view,
            QML_DIR / "AutoEqCalibration.qml",
            quick_theme(),
            {"vm": self.view_model},
        )
        self.quick_view.setAccessibleName("Auto-EQ content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)
        return reason

    def _on_curve_changed(self, index: int):
        """Update description when curve selection changes."""
        if index < 0:
            return
        curve = TARGET_CURVES[self.view_model.selected("curve")]
        self.view_model.set_page(curveDescription=curve.description)

    def _on_start_clicked(self):
        """Handle Start Recording button click based on recording state."""
        if self.recording_state == "idle":
            # Show recording UI and start with the level check
            self.view_model.set_page(recordingShown=True)
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

        configuration = self._configuration
        parent = _find_processor_owner(self.parent())

        if configuration is None or not parent:
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

        try:
            checkpoint = configuration.checkpoint()
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

        eq_state = configuration.states.eq
        try:
            proposal = self._candidate_proposal
            if proposal is None:
                raise ValueError("validated EQ proposal is unavailable")
            candidate_preset = proposal.proposed_preset
            configuration.apply(candidate_preset, processing_mode=checkpoint.processing_mode)
            eq_state.set_auto_eq_diagnostics(eq_settings)
            accepted_eq = eq_state.get_eq_settings()
            if not isinstance(accepted_eq, EQSettings):
                raise TypeError("EQ state returned invalid typed settings")
            expected_eq = candidate_preset.eq
            if accepted_eq.to_dict() != expected_eq.to_dict():
                raise ValueError("EQ state did not install the validated candidate")
            eq_settings.update(expected_eq.to_dict())
        except Exception as error:
            restore_error = None
            try:
                configuration.restore(checkpoint)
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
        self._report("applied", [self._outcome_code], [self.view_model.warning_text])

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
        owner = _find_processor_owner(self.parent())
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
        owner.engine_session.hold_mute("listening_comparison", True)
        try:
            keep = dialog.exec() == int(QDialog.DialogCode.Accepted)
        finally:
            owner.engine_session.hold_mute("listening_comparison", False)
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
        if self._configuration is None:
            return "Current processing configuration is unavailable"
        try:
            incumbent = self._configuration.current
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

        lease = self._borrow_audio(parent.engine_session)
        if lease is None:
            return
        self._audio_lease = lease

        self._capture_context_key = _owner_calibration_context_key(parent)
        self.recording_state = phase
        self._recording_duration = (
            LEVEL_CHECK_SECONDS if phase == "level_check" else RECORDING_DURATION
        )
        self.view_model.set_page(
            overrideShown=False,
            timeText=f"Time remaining: {self._recording_duration:.0f}s",
            startText="Checking Level..." if phase == "level_check" else "Recording...",
            startEnabled=False,  # Prevent early stop
            # Lock Auto-EQ target controls during recording
            choicesEnabled=False,
        )

        # Let DSP loop settle without blocking the UI thread.
        self._capture_start_timer.start(100)

        self.view_model.set_warning(
            "Read the first lines of the passage at your normal volume for 3 seconds."
            if phase == "level_check"
            else "Recording... Read the passage aloud at your normal volume.",
            state="info",
            strong=True,
        )

    def _borrow_audio(self, session: EngineSession) -> AudioLease | None:
        """The selected route's audio for this take, or None after saying why there is none."""
        try:
            return session.borrow(_TEMPORARY_MUTE_REASON)
        except RouteMismatch as mismatch:
            selected_input, selected_output = (_device_name(item) for item in mismatch.selected)
            active_input, active_output = (_device_name(item) for item in mismatch.active)
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
                self.view_model.set_warning(
                    "Recording canceled. The microphone and speakers that were "
                    "already running were kept.",
                    state="warn",
                    strong=False,
                )
                return None
            try:
                return session.borrow(_TEMPORARY_MUTE_REASON, switch_route=True)
            except AudioUnavailable as e:
                QMessageBox.critical(
                    self,
                    "Audio Error",
                    unavailable_text(
                        e,
                        "AudioForge could not switch to your selected devices.",
                        "Check that they are connected and not used by another app.",
                    ),
                )
                return None
        except AudioUnavailable as e:
            QMessageBox.critical(
                self,
                "Audio Error",
                unavailable_text(
                    e,
                    "AudioForge could not start the microphone and speakers.",
                    "Check that they are connected and not used by another app, then try again.",
                ),
            )
            return None

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
        assert (
            self._audio_lease is not None and self._audio_lease.held
        ), "a take starts only after _start_recording borrowed the audio"
        self._capture_session = CaptureSession(parent.processor, self._audio_lease)
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
        self.view_model.set_page(progress=value)

    def _on_time_remaining(self, seconds: float):
        """Update time remaining label."""
        if seconds > 0:
            self.view_model.set_page(timeText=f"Time remaining: {seconds:.0f}s")
        else:
            self.view_model.set_page(timeText="Complete", timeDone=True)

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
            self.view_model.set_warning(verdict.message, state=verdict.state, strong=True)

    def _sample_rate_or_default(self) -> int:
        """The processor's rate; 48 kHz, the rate Voice Setup and Auto-EQ require, when it cannot be read."""
        try:
            return _processor_sample_rate(_find_processor_owner(self.parent()))
        except Exception as error:
            logger.warning("Processor sample rate unreadable (%s); assuming 48000 Hz", error)
            return 48_000

    def _finish_level_check(self, audio_data: np.ndarray) -> None:
        """Judge the check from its samples and offer the next step the verdict allows."""
        sample_rate = self._sample_rate_or_default()
        verdict = verdict_from_audio(np.asarray(audio_data, dtype=np.float32), sample_rate)
        choices = level_choices(verdict, "Record Passage")
        self._level_verdict = verdict
        self.recording_state = "level_result"
        self.view_model.set_warning(
            f"{verdict.message} {choices.note}".strip(), state=verdict.state, strong=True
        )
        self.view_model.set_page(
            timeText="Level checked",
            timeDone=False,
            startText=choices.primary,
            startEnabled=True,
            overrideShown=choices.override is not None,
            **({} if choices.override is None else {"overrideText": choices.override}),
        )

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
        sample_rate = self._sample_rate_or_default()
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
        self.view_model.set_page(progressBusy=True, startEnabled=False)
        self.view_model.set_warning("Checking the recording...", state="info", strong=True)
        start_capture_analysis(
            self,
            partial(_analyze_capture, raw_audio, sample_rate),
            lambda result, token=generation: self._on_capture_analyzed(
                result, raw_audio, token
            ),
            lambda error, token=generation: self._on_capture_analysis_failed(error, token),
        )

    def _on_capture_analysis_failed(self, error: str, generation: int) -> None:
        if generation != self._analysis_generation or self._close_requested:
            return
        self._on_recording_failed(plain_error(f"Recording error: {error}"))

    def _on_capture_analyzed(
        self, analysis: CaptureAnalysis, raw_audio: np.ndarray, generation: int
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

        # Update UI
        self.view_model.set_page(
            progressBusy=True,  # the analysis reports no steady progress
            startText="Analyzing...",
            startEnabled=False,
            retakeShown=True,
            choicesEnabled=False,
        )

        # Show completion message
        self.view_model.set_warning(
            "Recording complete. Checking it against the tone preset.", state="ok", strong=True
        )

        if DEBUG:
            logger.debug("Audio captured; starting analysis")

        if DEBUG:
            logger.debug("Starting analysis worker")
        self._start_analysis()

    def _on_recording_failed(self, error: str):
        """Handle recording failure."""
        self._reset_recording_ui()
        self.view_model.set_warning(error, state="bad", strong=True)
        self._report("failed", ["recording_failed"], [error])

    def _reject_capture(self, message: str) -> None:
        """Refuse a finished take and leave the dialog ready to record it again."""
        self._reset_recording_ui()
        self.view_model.set_warning(message, state="bad", strong=True)
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
        self.view_model.set_page(choicesEnabled=False)
        if self.preview_audio_data is None:
            self._on_analysis_failed(
                "The unprocessed recording is missing, so the safety check could not run. "
                "Record again."
            )
            return
        try:
            if self._configuration is None:
                raise TypeError("current processing configuration is unavailable")
            incumbent = self._configuration.current
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
        self.view_model.set_warning(f"Analyzing: {step_name}")

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
        self.view_model.set_page(compareEnabled=apply_recommended)
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
        raw_reasons = [
            str(reason)
            for key in ("abstention_reasons", "recommendation_reasons")
            for reason in eq_settings.get(key) or ()
        ]
        # The sentence above is for the person; the analysis's own words stay on hover.
        self.view_model.set_warning(
            text,
            state="ok" if apply_recommended else "warn",
            strong=True,
            tip="; ".join(raw_reasons),
        )
        self.view_model.set_page(progressBusy=False, progress=100)
        self._show_analysis_diagnostics(eq_settings)
        self.view_model.set_page(
            startText="Apply EQ Settings" if apply_recommended else "Record Again",
            startEnabled=True,
        )
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

        self.view_model.set_chip(
            "confidence",
            f"{word}: {_QUALITY_MEANING[word]}",
            state=state,
            tip=quality_tooltip(
                capture_confidence,
                f"How well the recording supports each frequency band: {_format_percent(eq_confidence)}. "
                f"Safety and benefit of the correction: {_format_percent(validation_confidence)} "
                "(this part starts at 35% and rises as the correction gets closer to the tone preset, "
                f"so it says little about the recording). Overall score: {_format_percent(confidence)}.",
            ),
        )
        error_text = "Distance from the tone preset: " + (
            f"{_format_db(before)} now, {_format_db(after)} with this correction."
            if applied
            else f"{_format_db(before)}; no correction was applied."
        )
        error_details = f"The correction is used at {_format_percent(scale)} of its fitted strength."
        if isinstance(residual, dict) and "max_regularized_correction_db" in residual:
            requested = _format_db(residual.get("max_requested_correction_db"))
            regularized = _format_db(residual.get("max_regularized_correction_db"))
            error_details += (
                f" Smoothing took the largest change from {requested} to {regularized}."
            )
        self.view_model.set_chip(
            "error", error_text, state="ok" if after is not None else "idle", tip=error_details
        )
        headroom_text = "not available"
        headroom_details = ""
        headroom_state = "info"
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
            headroom_state = "info" if headroom_safe else "warn"
            if headroom_scale is not None and float(headroom_scale) < 1.0:
                headroom_details += (
                    f" The correction was reduced to {_format_percent(headroom_scale)} to fit."
                )
        self.view_model.set_chip(
            "gain", f"Headroom check: {headroom_text}.", state=headroom_state, tip=headroom_details
        )
        self.view_model.set_chip(
            "target",
            f"Tone preset: {self.view_model.curve_name()}",
            state="info",
            tip=f"Tone preset and matching: {target_profile}."
            + (
                " The recording had too few clear voiced sounds, so a rougher average was used."
                if used_fallback
                else ""
            ),
        )
        proposal = self._candidate_proposal
        self.view_model.set_page(eqCompareShown=proposal is not None and applied)
        if proposal is not None and applied:
            sample_rate = float(self._sample_rate_or_default())
            mode = self._configuration.processing_mode if self._configuration is not None else ""
            self.eq_compare.set_curves(
                proposal.incumbent_preset.eq,
                proposal.proposed_preset.eq,
                sample_rate,
                mode if mode in ("raw", "bypass") else "",
            )
        self.view_model.set_page(diagnosticsShown=True)

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
        self.view_model.set_warning(error, state="warn", strong=True, tip="")
        self.view_model.set_page(
            progressBusy=False,
            diagnosticsShown=False,
            startText="Record Again",
            startEnabled=True,
            choicesEnabled=False,
        )
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
        self.view_model.set_page(compareEnabled=False)
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
        self._recording_duration = RECORDING_DURATION
        self.level_meter.set_unavailable()
        self._had_result = False
        self.view_model.set_warning("Ready to record", state="idle", strong=False, tip="")
        self._level_verdict = self._level_override = None
        self.view_model.set_page(
            progressBusy=False,
            progress=0,
            timeText=f"Time remaining: {RECORDING_DURATION:.0f}s",
            timeDone=False,
            diagnosticsShown=False,
            startText="Check Level",
            startEnabled=True,
            retakeShown=False,
            overrideShown=False,
            choicesEnabled=True,
        )

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
            _find_processor_owner(self.parent()),
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
        if self._audio_lease is not None:
            self._audio_lease.release()

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
            self.view_model.set_page(startEnabled=False, retakeEnabled=False)
            self.view_model.set_warning("Canceling analysis...", state="info", strong=True)
            return
        self._finish_close()

    def closeEvent(self, event):
        """Ensure recording state is cleaned up if dialog is closed directly."""
        if session_is_ending():
            # A veto here would stop Windows from signing out: stop the work, join its threads, close.
            self._wait_for_analysis_workers()
            self._finish_close()
            event.accept()
            return
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
        return self.view_model.selected("curve")

    def get_selected_target_mode(self):
        """Return the selected target behavior."""
        return self.view_model.selected("mode")

    def get_selected_smoothing_strength(self):
        """Return the selected Auto-EQ smoothing strength."""
        return self.view_model.selected("smoothing")
