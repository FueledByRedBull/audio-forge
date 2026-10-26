"""Wizard dialog for Auto Voice Setup."""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from collections.abc import Callable, Mapping
from functools import partial
import json
import logging
import time
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..analysis.noise_reference import (
    SILENT_REFERENCE_GUIDANCE,
    SILENT_REFERENCE_REASON,
    CaptureMetadata,
    analyze_noise_reference,
    capture_is_silent,
)
from ..analysis.voice_setup import (
    GATE_MODE_LABELS,
    MAX_VERIFICATION_BAD_CAPTURES,
    _normalise_limiter_settings,
    analyze_voice_setup,
    raw_capture_is_clipped,
    reduce_for_verification,
    reducible_targets,
    reduction_metrics,
    validate_voice_setup_verification,
    verification_stop,
)
from ..config import (
    EQ_FREQUENCIES,
    EQSettings,
    Preset,
    TARGET_CURVES,
    build_eq_candidate_settings,
)
from .accessibility import bind_label, set_accessible_group
from .capture_session import (
    CaptureSession,
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
from .analysis_worker import (
    CancellableWorker,
    CaptureAnalysis,
    CaptureAnalysisWorker,
    start_capture_analysis,
)
from .calibration_support import (
    RAINBOW_PASSAGE,
    candidate_metadata as _candidate_metadata,
    chain_settings as _chain_settings,
    filtered_capture_for_analysis as _filtered_capture_for_analysis,
    format_db as _format_db,
    format_percent as _format_percent,
    on_analysis_thread_finished,
    wait_for_analysis_workers,
)
from .components import StepHeader, live_output_muted_notice, session_is_ending, update_label
from .layout_constants import (
    MARGIN_PANEL,
    SUBDUED_TEXT_STYLE,
    configure_resizable_dialog,
    configure_responsive_combo,
    create_scrollable_dialog_body,
    fit_spinbox_to_contents,
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
    room_noise_message,
    verdict_from_audio,
)
from .level_meter import LevelMeter, MeterWidget
from .setup_messages import (
    capture_issue_codes,
    capture_issue_lines,
    numbers_from,
    other_reasons,
    plain_advice,
    plain_error,
    plain_incomplete,
    plain_reasons,
    plain_stale,
    quality_tooltip,
    quality_word,
    reason_code,
    report_setup_result,
    verification_summary,
)
from .theme import (
    DESCRIPTION_LABEL_STYLE,
    PRIMARY_ACTION_BUTTON_STYLE,
    PROGRESS_BAR_STYLE,
    PROGRESS_LABEL_STYLE,
    message_text_style,
)

logger = logging.getLogger(__name__)

NOISE_RECORDING_DURATION = 2.0
VOICE_RECORDING_DURATION = 10.0
_VOICE_SETUP_MUTE_REASON = "auto_voice_setup"
# "&&" is a literal ampersand; a single "&" makes Qt underline the next character.
_APPLY_VERIFY_AGAIN_LABEL = "Apply && Verify Again"
_UNSTEADY_ROOM_ADVICE = "Setup will play safe; retake it if that was a one-off sound."
# A click does not stop the take, but it makes the whole room-noise reference weak.
_CLICK_REASONS = frozenset(
    {
        "room-noise capture contains isolated clipped samples",
        "room-noise capture contains strong transients",
    }
)
_CLICK_ADVICE = (
    "A click or bump was heard in the room noise. "
    "Press Retake and record the room noise again for a better result."
)
_GATE_WORDS = {
    "Threshold Only": "level only",
    "VAD Assisted": "level plus speech detection",
    "VAD Only": "speech detection only",
}
# Modes in which nothing is processed; setup analyses, applies and verifies as Normal.
_UNPROCESSED_MODES = ("raw", "bypass")
_VERIFIED_STAGES = (
    "input_cleanup",
    "gate",
    "suppression",
    "eq",
    "deesser",
    "compressor",
    "limiter",
)


def _candidate_eq(base_eq: EQSettings, eq_candidate: Mapping[str, Any]) -> EQSettings:
    """The EQ a setup result proposes: its correction over the tone the base keeps."""
    if not eq_candidate.get("apply_recommended", True):
        # Auto-EQ abstained: no tone change, so the user's own EQ stays as it is.
        return base_eq
    if "bands" in eq_candidate:
        eq_payload: dict[str, Any] = {
            key: deepcopy(eq_candidate[key])
            for key in ("schema_version", "enabled", "bands", "layers")
            if key in eq_candidate
        }
        eq_payload.setdefault("schema_version", base_eq.schema_version)
        eq_payload.setdefault("enabled", True)
        return EQSettings.from_dict(eq_payload)
    return build_eq_candidate_settings(
        base_eq,
        eq_candidate.get("band_freqs", ()),
        eq_candidate.get("band_gains", ()),
        eq_candidate.get("band_qs", ()),
        layer="correction",
        enabled=bool(eq_candidate.get("enabled", True)),
    )


def _candidate_preset(
    parent: Any,
    setup_result: Mapping[str, Any],
) -> tuple[Preset, float | None] | None:
    """Build a complete candidate for MainWindow's transactional apply API."""
    getter = getattr(parent, "_get_current_preset", None)
    if not callable(getter):
        return None
    base = getter()
    if not isinstance(base, Preset):
        return None
    payload = deepcopy(base.to_dict())
    # Raw monitoring already reads as Normal here; Bypass is left behind too.
    payload["bypass"] = False

    sections = {
        "gate": setup_result.get("gate_settings"),
        "deesser": setup_result.get("deesser_settings"),
        "compressor": setup_result.get("compressor_settings"),
        "limiter": setup_result.get("limiter_settings"),
        "rnnoise": setup_result.get("suppressor_settings"),
    }
    for name, candidate in sections.items():
        if not isinstance(candidate, Mapping):
            if name == "rnnoise":
                continue
            raise ValueError(f"candidate {name} settings are incomplete")
        base_section = payload.get(name)
        if not isinstance(base_section, Mapping):
            raise ValueError(f"current {name} settings are unavailable")
        payload[name] = {
            key: deepcopy(value)
            for key, value in candidate.items()
            if key in base_section
        }
        for key, value in base_section.items():
            payload[name].setdefault(key, deepcopy(value))

    eq_candidate = setup_result.get("eq_settings")
    if not isinstance(eq_candidate, Mapping):
        raise ValueError("candidate EQ settings are incomplete")
    payload["eq"] = _candidate_eq(base.eq, eq_candidate).to_dict()
    reliability = setup_result.get("compressor_settings", {}).get(
        "noise_reference_reliability"
    )
    if isinstance(reliability, (int, float)) and not isinstance(reliability, bool):
        reliability_value = float(reliability)
        if np.isfinite(reliability_value):
            return Preset.from_dict(payload), reliability_value
    return Preset.from_dict(payload), None


def _suppressor_settings(owner: Any) -> dict[str, Any] | None:
    state = getattr(owner, "noise_suppression_state", None)
    if state is None:
        return None
    return state.get_settings()


def _candidate_eq_settings_error(eq_settings: Any) -> str | None:
    if not isinstance(eq_settings, Mapping):
        return "candidate EQ settings are missing"
    if "bands" in eq_settings:
        try:
            EQSettings.from_dict(
                {
                    key: eq_settings[key]
                    for key in ("schema_version", "enabled", "bands", "layers")
                    if key in eq_settings
                }
            )
        except (TypeError, ValueError):
            return "candidate EQ bands are incomplete"
        return None
    if any(
        key not in eq_settings
        or not isinstance(eq_settings[key], (list, tuple))
        or len(eq_settings[key]) != len(EQ_FREQUENCIES)
        for key in ("band_freqs", "band_gains", "band_qs")
    ):
        return "candidate EQ bands are incomplete"
    return None


def _correction_depth_db(eq_settings: Mapping[str, Any]) -> float:
    """Largest gain among the enabled bands of the correction layer."""
    if "bands" not in eq_settings:
        # The analysis hands over the correction layer as plain arrays.
        return max(abs(gain) for gain in eq_settings["band_gains"])
    correction = (eq_settings.get("layers") or {}).get("correction", ())
    return max(
        (abs(band["gain_db"]) for band in correction if band["enabled"]), default=0.0
    )


def _correction_layer(eq_settings: Mapping[str, Any]) -> dict[str, Any]:
    """The correction layer's frequencies, gains and Qs, as the analysis arrays do."""
    if "bands" not in eq_settings:
        return {
            key: eq_settings[key]
            for key in ("enabled", "band_freqs", "band_gains", "band_qs")
            if key in eq_settings
        }
    layer = (eq_settings.get("layers") or {}).get("correction")
    view: dict[str, Any] = {"enabled": eq_settings.get("enabled", True)}
    if layer:
        view["band_freqs"] = [band["frequency_hz"] for band in layer]
        view["band_gains"] = [band["gain_db"] if band["enabled"] else 0.0 for band in layer]
        view["band_qs"] = [band["q"] for band in layer]
    else:
        view["band_gains"] = [0.0] * len(EQ_FREQUENCIES)
    return view


def _candidate_settings_error(
    setup_result: Mapping[str, Any] | None,
) -> str | None:
    """Return a user-facing reason when a candidate cannot be applied."""
    if not isinstance(setup_result, Mapping):
        return "voice setup result is missing"
    try:
        limiter_settings = _normalise_limiter_settings(
            setup_result.get("limiter_settings"), require_complete=True
        )
    except ValueError:
        return "candidate limiter settings are invalid"
    if limiter_settings is None:
        return "candidate limiter settings are incomplete"
    eq_error = _candidate_eq_settings_error(setup_result.get("eq_settings"))
    if eq_error is not None and setup_result.get("eq_error"):
        return str(setup_result["eq_error"])
    return eq_error


def _voice_setup_context_matches(
    captured: object,
    current: object,
    *,
    allow_unprocessed_to_normal: bool = False,
) -> bool:
    if captured == current:
        return True
    if (
        not allow_unprocessed_to_normal
        or not isinstance(captured, str)
        or not isinstance(current, str)
    ):
        return False
    try:
        captured_parts = json.loads(captured)
        current_parts = json.loads(current)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(captured_parts, list)
        and isinstance(current_parts, list)
        and len(captured_parts) == len(current_parts) == 5
        and captured_parts[:4] == current_parts[:4]
        and captured_parts[4] in _UNPROCESSED_MODES
        and current_parts[4] == "normal"
    )


class VoiceSetupWorker(CancellableWorker):
    """Background worker for multi-stage voice setup analysis."""

    step_progress = Signal(str, int)
    result_ready = Signal(dict)
    _failure_log = "Voice Setup analysis failed"

    def __init__(
        self,
        noise_audio: np.ndarray,
        voice_audio: np.ndarray,
        sample_rate: int,
        target_preset: str,
        *,
        vad_available: bool,
        dynamics_intensity: str,
        custom_target_p95_db: float,
        custom_peak_cap_db: float,
        limiter_settings: Mapping[str, Any] | None,
        noise_metadata: CaptureMetadata | None,
        voice_metadata: CaptureMetadata | None,
        target_lufs: float | None = None,
        noise_model: str = "rnnoise",
        suppressor_strength: float = 1.0,
        incumbent_settings: Mapping[str, Any] | None = None,
        raw_noise_audio: np.ndarray | None = None,
        raw_speech_audio: np.ndarray | None = None,
        input_cleanup_mode: str = "off",
        processing_mode: str = "normal",
    ) -> None:
        super().__init__()
        self.noise_audio = noise_audio
        self.voice_audio = voice_audio
        self.sample_rate = sample_rate
        self.target_preset = target_preset
        self.vad_available = vad_available
        self.dynamics_intensity = dynamics_intensity
        self.custom_target_p95_db = custom_target_p95_db
        self.custom_peak_cap_db = custom_peak_cap_db
        self.limiter_settings = (
            dict(limiter_settings) if limiter_settings is not None else None
        )
        self.noise_metadata = noise_metadata
        self.voice_metadata = voice_metadata
        self.target_lufs = target_lufs
        self.noise_model = noise_model
        self.suppressor_strength = suppressor_strength
        self.incumbent_settings = deepcopy(incumbent_settings)
        self.raw_noise_audio = raw_noise_audio
        self.raw_speech_audio = raw_speech_audio
        self.input_cleanup_mode = input_cleanup_mode
        self.processing_mode = processing_mode

    def _run(self) -> None:
        if self._should_stop():
            return
        result = analyze_voice_setup(
            self.noise_audio,
            self.voice_audio,
            self.sample_rate,
            self.target_preset,
            vad_available=self.vad_available,
            dynamics_intensity=self.dynamics_intensity,
            custom_target_p95_db=self.custom_target_p95_db,
            custom_peak_cap_db=self.custom_peak_cap_db,
            target_lufs=self.target_lufs,
            limiter_settings=self.limiter_settings,
            noise_metadata=self.noise_metadata,
            speech_metadata=self.voice_metadata,
            cancel_check=self._should_stop,
            progress_callback=self.step_progress.emit,
            noise_model=self.noise_model,
            suppressor_strength=self.suppressor_strength,
            incumbent_settings=self.incumbent_settings,
            raw_noise_audio=self.raw_noise_audio,
            raw_speech_audio=self.raw_speech_audio,
            input_cleanup_mode=self.input_cleanup_mode,
            processing_mode=self.processing_mode,
        )
        if self._should_stop():
            return
        self.result_ready.emit(result)


class VoiceSetupVerificationWorker(CancellableWorker):
    """Run second-passage chain verification off the UI thread."""

    step_progress = Signal(str, int)
    result_ready = Signal(dict)
    _failure_log = "Voice Setup verification failed"

    def __init__(
        self,
        noise_audio: np.ndarray,
        original_speech_audio: np.ndarray,
        verification_speech_audio: np.ndarray,
        sample_rate: int,
        setup_result: dict[str, Any],
        target_preset: str,
        *,
        raw_noise_audio: np.ndarray | None = None,
        raw_verification_speech_audio: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.noise_audio = noise_audio
        self.original_speech_audio = original_speech_audio
        self.verification_speech_audio = verification_speech_audio
        self.raw_noise_audio = raw_noise_audio
        self.raw_verification_speech_audio = raw_verification_speech_audio
        self.sample_rate = sample_rate
        self.setup_result = deepcopy(setup_result)
        self.target_preset = target_preset

    def _run(self) -> None:
        if self._should_stop():
            return
        self.step_progress.emit("Validating the combined processing chain...", 55)
        result = validate_voice_setup_verification(
            self.noise_audio,
            self.original_speech_audio,
            self.verification_speech_audio,
            self.sample_rate,
            self.setup_result,
            self.target_preset,
            raw_noise_audio=self.raw_noise_audio,
            raw_verification_speech_audio=self.raw_verification_speech_audio,
            cancel_check=self._should_stop,
        )
        if self._should_stop():
            return
        self.step_progress.emit("Finalizing verification...", 95)
        self.result_ready.emit(result)


def _analyze_capture(
    raw_audio: np.ndarray,
    sample_rate: int,
    phase: str,
    metadata: CaptureMetadata | None,
) -> CaptureAnalysis:
    """The slow half of a finished take; it runs on a worker thread."""
    audio = _filtered_capture_for_analysis(raw_audio, sample_rate)
    if phase == "noise_recording":
        quality = analyze_noise_reference(
            audio,
            None,
            int(metadata.sample_rate or 48_000) if metadata is not None else 48_000,
            noise_metadata=metadata,
        )
        return CaptureAnalysis(audio, False, quality)
    if phase == "voice_recording":
        return CaptureAnalysis(audio, capture_is_silent(audio))
    return CaptureAnalysis(audio, False)


class VoiceSetupDialog(QDialog):
    """Record room tone and speech, then recommend a voice chain."""

    setup_applied = Signal(str)

    STEPS = ("Target", "Level", "Room noise", "Voice", "Review")
    # Which step each state of the state machine belongs to.
    STEP_FOR_STATE = {
        "idle": 0,
        "level_check": 1,
        "level_result": 1,
        "noise_recording": 2,
        "noise_ready": 3,
        "voice_recording": 3,
        "analyzing": 3,
        "verification_recording": 3,
        "verification_analyzing": 3,
        "completed": 4,
        "verification_ready": 4,
        "final_comparison_ready": 4,
    }

    @property
    def setup_state(self) -> str:
        return self._setup_state

    @setup_state.setter
    def setup_state(self, state: str) -> None:
        self._setup_state = state
        if "step_header" in self.__dict__:
            self._show_step_for_state()

    def _show_step_for_state(self) -> None:
        # A take being checked stays on the step it was recorded on.
        state = self._capture_phase if self._setup_state == "capture_analyzing" else self._setup_state
        step = self.STEP_FOR_STATE.get(state, 0)
        if self._setup_state == "idle" and self._failed_step is not None:
            # A take that failed stays on its own step, with its message.
            step = self._failed_step
        self.step_header.set_current(step)
        self.recording_group.setTitle("Review" if step == 4 else "Recording")
        # Compare needs a recording and a proposal, so it belongs to the review alone.
        self.compare_button.setVisible(step == 4)
        self.curve_group.setVisible(step == 0)
        self.dynamics_group.setVisible(step == 0)
        self.noise_group.setVisible(step == 2)
        # The level check is spoken from the passage, so it is shown from the first step on.
        self.voice_group.setVisible(step in (1, 3))
        self.review_page.setVisible(step == 4)
        # Only a recording mutes the live output.
        self.mute_notice.setVisible(
            self._setup_state
            in ("level_check", "noise_recording", "voice_recording", "verification_recording")
        )
        if "start_button" in self.__dict__:
            suggestions = self._advisory_only and self._setup_state == "completed"
            self.start_button.setStyleSheet("" if suggestions else PRIMARY_ACTION_BUTTON_STYLE)
        # The meter only matters while something is being recorded.
        for widget in (self.progress_bar, self.time_label, self.level_meter_widget):
            widget.setVisible(step in (1, 2, 3))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Voice Setup")
        self.setModal(True)

        self.setup_state = "idle"
        self.noise_audio: np.ndarray | None = None
        self.voice_audio: np.ndarray | None = None
        self.preview_noise_audio: np.ndarray | None = None
        self.preview_voice_audio: np.ndarray | None = None
        self.preview_verification_audio: np.ndarray | None = None
        self.noise_metadata: CaptureMetadata | None = None
        self.voice_metadata: CaptureMetadata | None = None
        self._current_capture_metadata: CaptureMetadata | None = None
        self._pre_setup_snapshot: dict[str, Any] | None = None
        self._final_comparison_pending = False
        self.setup_result: dict[str, Any] | None = None
        self._summary_result: dict[str, Any] | None = None
        self._candidate_metadata: dict[str, Any] | None = None
        # What each stage was set to before this setup, for the "Before:" lines.
        self._review_current: dict[str, Mapping[str, Any]] = {}
        self._capture_context_key: str | None = None
        self.analysis_worker: (
            VoiceSetupWorker | VoiceSetupVerificationWorker | CaptureAnalysisWorker | None
        ) = None
        self._analysis_workers: list[
            VoiceSetupWorker | VoiceSetupVerificationWorker | CaptureAnalysisWorker
        ] = []
        self._analysis_generation = 0
        # The recording phase a take in "capture_analyzing" came from.
        self._capture_phase = "voice_recording"
        self._close_requested = False
        self._close_result = False
        self._started_processor = False
        self._recording_duration = NOISE_RECORDING_DURATION
        self._capture_session: CaptureSession | None = None
        self._verification_audio: np.ndarray | None = None
        self._verification_adjustment_count = 0
        self._verification_bad_capture_count = 0
        self._verification_last_reduction_metrics: dict[str, float] | None = None
        self._recent_rms_db: deque[float] = deque(maxlen=LEVEL_HOLD_UPDATES)
        self._recent_peak_db: deque[float] = deque(maxlen=LEVEL_HOLD_UPDATES)
        # A result that did not pass every check: Apply is then the secondary action.
        self._advisory_only = False
        # The level check before the takes: its verdict and, if the user went on past it, which.
        self._level_verdict: LevelVerdict | None = None
        self._level_override: str | None = None
        self._failed_step: int | None = None
        # Whether this run has already told the main window how it ended.
        self._reported = False
        # Whether this dialog has reported the outcome of an analysis, which a refused retake
        # does not replace.
        self._result_reported = False

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
            preferred_width=760,
            preferred_height=820,
            minimum_width=500,
            minimum_height=380,
        )

    def _current_target_lufs(self) -> float:
        state = getattr(self.parent(), "compressor_state", None)
        getter = getattr(state, "get_settings", None)
        if callable(getter):
            try:
                settings = getter()
                value = float(
                    settings.get("target_lufs", -18.0)
                    if isinstance(settings, Mapping)
                    else -18.0
                )
                return min(-12.0, max(-24.0, value))
            except (TypeError, ValueError):
                pass
        return -18.0

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.step_header = StepHeader(self.STEPS)
        outer_layout.addWidget(self.step_header)

        self.content_scroll_area, layout = create_scrollable_dialog_body(self)
        self.content_scroll_area.setAccessibleName("Auto Voice Setup content")
        outer_layout.addWidget(self.content_scroll_area, stretch=1)

        self.curve_group = QGroupBox("Tone preset")
        curve_layout = QVBoxLayout(self.curve_group)

        curve_input = QHBoxLayout()
        curve_label = QLabel("Tone preset:")
        curve_input.addWidget(curve_label)
        self.curve_combo = QComboBox()
        for key, curve in TARGET_CURVES.items():
            self.curve_combo.addItem(curve.name, key)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_changed)
        configure_responsive_combo(self.curve_combo)
        curve_input.addWidget(self.curve_combo)
        bind_label(curve_label, self.curve_combo)
        curve_layout.addLayout(curve_input)

        self.curve_description = QLabel()
        self.curve_description.setWordWrap(True)
        self.curve_description.setStyleSheet(DESCRIPTION_LABEL_STYLE)
        curve_layout.addWidget(self.curve_description)
        layout.addWidget(self.curve_group)

        self.dynamics_group = QGroupBox("Dynamics")
        dynamics_layout = QVBoxLayout(self.dynamics_group)
        dynamics_row = QFormLayout()
        dynamics_row.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        dynamics_row.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        dynamics_label = QLabel("Compression:")
        self.dynamics_combo = QComboBox()
        for label, key in (
            ("Gentle", "gentle"),
            ("Balanced", "balanced"),
            ("Dense", "dense"),
            ("Custom", "custom"),
        ):
            self.dynamics_combo.addItem(label, key)
        config = getattr(self.parent(), "config", None)
        configured = getattr(config, "voice_setup_dynamics_intensity", "balanced")
        configured_index = self.dynamics_combo.findData(configured)
        self.dynamics_combo.setCurrentIndex(max(0, configured_index))
        configure_responsive_combo(self.dynamics_combo)
        dynamics_row.addRow(dynamics_label, self.dynamics_combo)
        bind_label(dynamics_label, self.dynamics_combo)
        self.custom_p95_spin = QDoubleSpinBox()
        self.custom_p95_spin.setRange(1.0, 8.0)
        self.custom_p95_spin.setSuffix(" dB")
        self.custom_p95_spin.setValue(
            float(getattr(config, "voice_setup_custom_p95_db", 3.5))
        )
        fit_spinbox_to_contents(self.custom_p95_spin)
        custom_p95_label = QLabel("Typical evening-out:")
        dynamics_row.addRow(custom_p95_label, self.custom_p95_spin)
        self.custom_peak_spin = QDoubleSpinBox()
        self.custom_peak_spin.setRange(1.5, 12.0)
        self.custom_peak_spin.setSuffix(" dB")
        self.custom_peak_spin.setValue(
            float(getattr(config, "voice_setup_custom_peak_cap_db", 8.0))
        )
        fit_spinbox_to_contents(self.custom_peak_spin)
        custom_peak_label = QLabel("Strongest squeeze allowed:")
        dynamics_row.addRow(custom_peak_label, self.custom_peak_spin)
        bind_label(
            custom_p95_label,
            self.custom_p95_spin,
            name="Typical compressor evening-out in decibels",
        )
        bind_label(
            custom_peak_label,
            self.custom_peak_spin,
            name="Strongest compressor squeeze allowed in decibels",
        )
        dynamics_layout.addLayout(dynamics_row)
        target_lufs_label = QLabel("Target loudness:")
        self.target_lufs_spin = QDoubleSpinBox()
        self.target_lufs_spin.setRange(-24.0, -12.0)
        self.target_lufs_spin.setSingleStep(1.0)
        self.target_lufs_spin.setSuffix(" LUFS")
        self.target_lufs_spin.setToolTip(
            "Loudness target is independent of the selected tone curve."
        )
        self.target_lufs_spin.setValue(self._current_target_lufs())
        fit_spinbox_to_contents(self.target_lufs_spin)
        dynamics_row.addRow(target_lufs_label, self.target_lufs_spin)
        bind_label(target_lufs_label, self.target_lufs_spin, name="Voice setup target loudness")
        dynamics_hint = QLabel(
            "Compression decides how much louder and quieter words are evened out. "
            "Tone and target loudness are set separately."
        )
        dynamics_hint.setWordWrap(True)
        dynamics_hint.setStyleSheet(SUBDUED_TEXT_STYLE)
        dynamics_layout.addWidget(dynamics_hint)
        layout.addWidget(self.dynamics_group)
        self.dynamics_combo.currentIndexChanged.connect(
            self._on_dynamics_intensity_changed
        )
        self.custom_p95_spin.valueChanged.connect(self._on_dynamics_intensity_changed)
        self.custom_peak_spin.valueChanged.connect(self._on_dynamics_intensity_changed)
        self._on_dynamics_intensity_changed(self.dynamics_combo.currentIndex())

        self.noise_group = noise_group = QGroupBox("Capture room noise")
        noise_layout = QVBoxLayout(noise_group)
        noise_hint = QLabel(
            "Stay quiet for 2 seconds so the wizard can measure room noise "
            "and set the gate safely."
        )
        noise_hint.setWordWrap(True)
        noise_layout.addWidget(noise_hint)
        layout.addWidget(noise_group)

        self.voice_group = voice_group = QGroupBox("Read this passage aloud")
        voice_layout = QVBoxLayout(voice_group)
        # A label grows with the passage, so no line is cut and nothing scrolls inside the page.
        passage_text = QLabel(RAINBOW_PASSAGE)
        passage_text.setWordWrap(True)
        passage_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        passage_text.setAccessibleDescription(
            "Read-only passage to speak during Auto Voice Setup."
        )
        voice_layout.addWidget(passage_text)
        layout.addWidget(voice_group)

        self.mute_notice = live_output_muted_notice()
        layout.addWidget(self.mute_notice)

        self.recording_group = QGroupBox("Recording")
        recording_layout = QVBoxLayout(self.recording_group)
        self.recording_group.setVisible(False)

        self.phase_label = QLabel("Ready to start setup")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        recording_layout.addWidget(self.phase_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setAccessibleName("Voice Setup progress")
        recording_layout.addWidget(self.progress_bar)

        self.time_label = QLabel("Time remaining: 2s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        recording_layout.addWidget(self.time_label)

        meter_layout = QHBoxLayout()
        self.level_meter = LevelMeter(label="Level", show_scale=True)
        self.level_meter_widget = MeterWidget(self.level_meter)
        meter_layout.addWidget(self.level_meter_widget)
        recording_layout.addLayout(meter_layout)

        self.warning_label = QLabel("Ready to record")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.warning_label.setWordWrap(True)
        recording_layout.addWidget(self.warning_label)

        self.summary_group = QGroupBox("Recommended Settings")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_group.setVisible(False)
        self.overall_label = QLabel("Overall: --")
        self.eq_label = QLabel("EQ: --")
        self.gate_label = QLabel("Noise gate: --")
        self.deesser_label = QLabel("De-esser: --")
        self.compressor_label = QLabel("Compressor: --")
        for label in (
            self.overall_label,
            self.eq_label,
            self.gate_label,
            self.deesser_label,
            self.compressor_label,
        ):
            label.setStyleSheet(status_chip_style("idle"))
            label.setWordWrap(True)
            summary_layout.addWidget(label)
        hint = QLabel(
            "Voice Setup tunes tone, noise suppression, the noise gate, the "
            "de-esser and the compressor. The limiter is left as it is. Hover a "
            "line for the numbers behind it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.eq_compare = EQComparisonWidget()
        summary_layout.addWidget(self.eq_compare)
        summary_layout.addWidget(hint)
        self.capture_issues_layout = QVBoxLayout()
        summary_layout.insertLayout(1, self.capture_issues_layout)
        # The summary's own visibility still follows the results; the page
        # around it follows the step.
        self.review_page = QWidget()
        review_layout = QVBoxLayout(self.review_page)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.addWidget(self.summary_group)

        controls = QHBoxLayout()
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.setVisible(False)
        self.retake_btn.clicked.connect(self._on_retake_clicked)
        controls.addWidget(self.retake_btn)

        self.compare_button = QPushButton("Compare Recording")
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._compare_recording)
        controls.addWidget(self.compare_button)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self.recording_group)
        layout.addWidget(self.review_page)
        layout.addStretch(1)

        self.start_button = QPushButton("Start Voice Setup")
        self.start_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.start_button.clicked.connect(self._on_start_clicked)

        # Going on past a level that is not good is a separate, plainly named action.
        self.level_override_btn = QPushButton("Continue Anyway")
        self.level_override_btn.setVisible(False)
        self.level_override_btn.clicked.connect(self._continue_past_level)

        # The actions stay in view while the page above them scrolls.
        controls.setContentsMargins(MARGIN_PANEL, 0, MARGIN_PANEL, MARGIN_PANEL)
        controls.insertWidget(0, self.cancel_btn)
        controls.insertStretch(2, 1)
        controls.addWidget(self.level_override_btn)
        controls.addWidget(self.start_button)
        outer_layout.addLayout(controls)
        self._show_step_for_state()

        set_accessible_group(
            (
                (self.retake_btn, "Retake Voice Setup recording", None),
                (self.cancel_btn, "Cancel Auto Voice Setup", None),
                (self.level_meter_widget, "Voice Setup input level", None),
            )
        )
        self.setTabOrder(self.curve_combo, self.dynamics_combo)
        self.setTabOrder(self.dynamics_combo, self.custom_p95_spin)
        self.setTabOrder(self.custom_p95_spin, self.custom_peak_spin)
        self.setTabOrder(self.custom_peak_spin, self.target_lufs_spin)
        # The order of the button row, left to right.
        self.setTabOrder(self.target_lufs_spin, self.cancel_btn)
        self.setTabOrder(self.cancel_btn, self.retake_btn)
        self.setTabOrder(self.retake_btn, self.compare_button)
        self.setTabOrder(self.compare_button, self.level_override_btn)
        self.setTabOrder(self.level_override_btn, self.start_button)

        self._on_curve_changed(0)

    def _on_curve_changed(self, index: int) -> None:
        if index < 0:
            return
        curve_key = self.curve_combo.currentData()
        curve = TARGET_CURVES[curve_key]
        self.curve_description.setText(curve.description)

    def _on_dynamics_intensity_changed(self, _index: int) -> None:
        custom = self.dynamics_combo.currentData() == "custom"
        self.custom_p95_spin.setEnabled(custom)
        self.custom_peak_spin.setEnabled(custom)
        config = getattr(self.parent(), "config", None)
        if config is not None:
            config.voice_setup_dynamics_intensity = str(
                self.dynamics_combo.currentData()
            )
            config.voice_setup_custom_p95_db = self.custom_p95_spin.value()
            config.voice_setup_custom_peak_cap_db = self.custom_peak_spin.value()

    def _on_start_clicked(self) -> None:
        if self.setup_state == "idle":
            self.recording_group.setVisible(True)
            # A take that failed is recorded again; the level choice made before it stays.
            retake_noise = self._failed_step == 2
            self._failed_step = None
            if retake_noise:
                self._start_recording_phase("noise_recording")
            else:
                self._level_override = None
                self._start_recording_phase("level_check")
        elif self.setup_state == "level_result":
            verdict = self._level_verdict
            self._start_recording_phase(
                "noise_recording"
                if verdict is not None and verdict.code == "good"
                else "level_check"
            )
        elif self.setup_state == "noise_ready":
            self._start_recording_phase("voice_recording")
        elif self.setup_state == "final_comparison_ready":
            self._compare_recording()
        elif self.setup_state == "completed" and self.setup_result is not None:
            self._apply_setup()
        elif self.setup_state == "verification_ready":
            self._start_recording_phase("verification_recording")
        elif self.setup_state in {"level_check", "noise_recording", "voice_recording"}:
            QMessageBox.information(
                self, "Recording", "Please let the recording finish."
            )

    def _continue_past_level(self) -> None:
        """Go on to the room noise although the level check did not say it was good."""
        verdict = self._level_verdict
        if self.setup_state != "level_result" or verdict is None or not verdict.can_override:
            return
        self._level_override = verdict.code
        self._start_recording_phase("noise_recording")

    def _start_recording_phase(self, phase: str) -> None:
        self.compare_button.setEnabled(False)
        self._cancel_analysis_workers()
        if not self._ensure_processor_ready():
            return

        self.setup_state = phase
        self.level_override_btn.setVisible(False)
        self.curve_combo.setEnabled(False)
        self.dynamics_group.setEnabled(False)
        self.target_lufs_spin.setEnabled(False)
        self.start_button.setEnabled(False)
        self.retake_btn.setVisible(False)
        if phase != "verification_recording":
            # The proposal stays on the review page while its verification take is recorded.
            self.summary_group.setVisible(False)

        if phase == "level_check":
            self._recording_duration = LEVEL_CHECK_SECONDS
            self.start_button.setText("Checking Level...")
            self.phase_label.setText("Checking your level")
            self.warning_label.setText(
                "Read the first lines of the passage at your normal volume for 3 seconds."
            )
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        elif phase == "noise_recording":
            self.preview_noise_audio = None
            self._recording_duration = NOISE_RECORDING_DURATION
            self.start_button.setText("Recording Noise...")
            self.phase_label.setText("Capturing room noise")
            self.warning_label.setText(
                "Stay quiet and keep the room as it normally is."
            )
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        elif phase == "voice_recording":
            self.preview_voice_audio = None
            self._recording_duration = VOICE_RECORDING_DURATION
            self.start_button.setText("Recording Voice...")
            self.phase_label.setText("Capturing speech")
            self.warning_label.setText("Speak naturally into the microphone.")
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        else:
            self.preview_verification_audio = None
            self._recording_duration = VOICE_RECORDING_DURATION
            self.start_button.setText("Recording Verification...")
            self.phase_label.setText("Capturing a second passage")
            self.warning_label.setText(
                "Read the passage again at the same distance and volume. AudioForge "
                "will run this recording through the new settings and compare it "
                "with the first one."
            )
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))

        self.progress_bar.setValue(0)
        self.time_label.setText(f"Time remaining: {self._recording_duration:.0f}s")
        self._capture_start_timer.start(100)

    def _ensure_processor_ready(self) -> bool:
        parent = _find_processor_owner(self.parent())
        if not parent:
            QMessageBox.critical(self, "Error", plain_error("Could not find audio processor"))
            return False

        processor_was_running = parent.processor.is_running()
        selected_identities = _selected_device_identities(parent)
        selected_input = _device_name(selected_identities[0])
        selected_output = _device_name(selected_identities[1])

        if processor_was_running:
            active_identities = _active_device_identities(parent.processor)
            active_pair = (
                _device_name(active_identities[0]),
                _device_name(active_identities[1]),
            )
            if not _route_identities_match(selected_identities, active_identities):
                reply = QMessageBox.question(
                    self,
                    "Switch Devices for Voice Setup?",
                    "Voice setup should record from your selected devices.\n\n"
                    f"Selected input: {_device_label(selected_input, '(Default Input)')}\n"
                    f"Selected output: {_device_label(selected_output, '(Default Output)')}\n\n"
                    f"Active input: {_device_label(active_pair[0], '(Default Input)')}\n"
                    f"Active output: {_device_label(active_pair[1], '(Default Output)')}\n\n"
                    "Switch now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.warning_label.setText(
                        "Setup canceled. The microphone and speakers that were "
                        "already running were kept."
                    )
                    self.warning_label.setStyleSheet(message_text_style("warn"))
                    # The state, button text and selector stay as the dialog had them.
                    self.start_button.setEnabled(True)
                    return False

                try:
                    _restart_processor_for_route(
                        parent.processor, selected_identities, active_identities
                    )
                    _set_temporary_mute(parent, _VOICE_SETUP_MUTE_REASON, False)
                except Exception as exc:
                    _sync_owner_processing_controls(parent)
                    QMessageBox.critical(
                        self,
                        "Audio Error",
                        "AudioForge could not switch to your selected devices.\n\n"
                        f"{exc}\n\nCheck that they are connected and not used by another app.",
                    )
                    return False
            return True

        try:
            _start_selected_route(parent)
            self._started_processor = True
            return True
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Audio Error",
                "AudioForge could not start the microphone and speakers.\n\n"
                f"{exc}\n\nCheck that they are connected and not used by another app, "
                "then try again.",
            )
            return False

    def _begin_recording_capture(self) -> None:
        self._capture_start_timer.stop()
        if self.setup_state not in {
            "level_check",
            "noise_recording",
            "voice_recording",
            "verification_recording",
        }:
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_recording_failed(plain_error("Could not find audio processor"))
            return

        context_key = _owner_calibration_context_key(parent)
        if self.setup_state in {"level_check", "noise_recording"}:
            self._capture_context_key = context_key
        elif not _voice_setup_context_matches(
            self._capture_context_key,
            context_key,
            allow_unprocessed_to_normal=(
                self.setup_state == "verification_recording"
                and isinstance(self._pre_setup_snapshot, Mapping)
                and self._pre_setup_snapshot.get("processing_mode") in _UNPROCESSED_MODES
            ),
        ):
            self._on_recording_failed(
                "The audio route, input settings or processing mode changed during setup. "
                "Start the setup again without changing them."
            )
            return

        self._recent_rms_db.clear()
        self._recent_peak_db.clear()
        try:
            get_input = getattr(parent.processor, "get_active_input_device", None)
            get_mode = getattr(parent.processor, "get_input_channel_mode", None)
            self._current_capture_metadata = CaptureMetadata(
                captured_at_unix_s=time.time(),
                input_device=str(get_input() or "") if callable(get_input) else "",
                sample_rate=_processor_sample_rate(parent),
                channel_mode=str(get_mode() or "") if callable(get_mode) else "",
                channel_count=1,
            )
            self._capture_session = CaptureSession(parent, _VOICE_SETUP_MUTE_REASON)
            self._capture_session.start(
                self._recording_duration,
                before_cleanup=True,
            )
        except Exception as exc:
            self._on_recording_failed(plain_error(f"Recording error: {exc}"))
            return

        self.recording_timer.start()

    def _poll_recording_progress(self) -> None:
        if self.setup_state not in {
            "level_check",
            "noise_recording",
            "voice_recording",
            "verification_recording",
        }:
            self.recording_timer.stop()
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_recording_failed(plain_error("Could not find audio processor"))
            return

        try:
            progress = float(parent.processor.recording_progress())
            failure = (
                self._capture_session.failure_reason(progress)
                if self._capture_session is not None
                else None
            )
            if failure is not None:
                self.recording_timer.stop()
                self._on_recording_failed(plain_error(failure))
                return
            progress_pct = int(progress * 100)
            self.progress_bar.setValue(progress_pct)
            self._update_time_remaining(
                max(0.0, self._recording_duration * (1.0 - progress))
            )
            self._update_level(
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
                self._on_recording_complete(np.asarray(audio, dtype=np.float32))
        except Exception as exc:
            self.recording_timer.stop()
            self._on_recording_failed(plain_error(f"Recording error: {exc}"))

    def _update_time_remaining(self, seconds: float) -> None:
        if seconds > 0:
            self.time_label.setText(f"Time remaining: {seconds:.0f}s")
        else:
            self.time_label.setText("Complete!")
            self.time_label.setStyleSheet(message_text_style("ok", strong=True))

    def _update_level(self, rms_db: float, peak_db: float) -> None:
        self.level_meter.set_levels(rms_db, peak_db)
        self._recent_rms_db.append(rms_db)
        self._recent_peak_db.append(peak_db)

        if self.setup_state == "noise_recording":
            self._show_live_message(*room_noise_message(self._recent_rms_db))
            return

        # The loudest of a full window of blocks: the pauses between words are not "quiet".
        verdict = live_verdict(
            self._recent_rms_db,
            self._recent_peak_db,
            again="check again" if self.setup_state == "level_check" else "record again",
        )
        if verdict is not None:
            self._show_live_message(verdict.message, verdict.state)

    def _show_live_message(self, message: str, state: str) -> None:
        """Update the live label only when it changed; this runs on every 100 ms poll."""
        update_label(self.warning_label, message, message_text_style(state, strong=True))

    def _finish_level_check(self, audio_data: np.ndarray) -> None:
        """Judge the check from its samples and offer the next step the verdict allows."""
        metadata = self._current_capture_metadata
        sample_rate = int(metadata.sample_rate) if metadata is not None and metadata.sample_rate else 48_000
        verdict = verdict_from_audio(np.asarray(audio_data, dtype=np.float32), sample_rate)
        choices = level_choices(verdict, "Record Room Noise")
        self._level_verdict = verdict
        self.setup_state = "level_result"
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.time_label.setText("Level checked")
        self.phase_label.setText("Level checked")
        self.warning_label.setText(f"{verdict.message} {choices.note}".strip())
        self.warning_label.setStyleSheet(message_text_style(verdict.state, strong=True))
        self.start_button.setText(choices.primary)
        self.start_button.setEnabled(True)
        self.level_override_btn.setVisible(choices.override is not None)
        if choices.override is not None:
            self.level_override_btn.setText(choices.override)

    def _on_recording_complete(self, audio_data: np.ndarray) -> None:
        if self.setup_state == "level_check":
            self._finish_level_check(audio_data)
            return
        self.start_button.setEnabled(True)
        self.retake_btn.setVisible(True)
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)

        raw_audio = np.ascontiguousarray(
            np.asarray(audio_data, dtype=np.float32).reshape(-1).copy()
        )
        sample_rate = (
            int(self._current_capture_metadata.sample_rate)
            if self._current_capture_metadata is not None
            and self._current_capture_metadata.sample_rate
            else 48_000
        )

        # The pre-filter cannot take non-finite samples, in any step.
        if not np.isfinite(raw_audio).all():
            self._reject_capture(
                "The recording held invalid audio data, which usually means the audio "
                "driver glitched. Restart processing and record again."
            )
            return
        # The analysis refuses a clipped capture; say so now, while the fix is obvious,
        # instead of failing after the voice step with the bad capture still in place.
        if self.setup_state != "verification_recording" and raw_capture_is_clipped(
            raw_audio, sample_rate
        ):
            self._reject_capture(
                "A loud sound was picked up in the room-noise recording. "
                "Record the room noise again while the room is quiet."
                if self.setup_state == "noise_recording"
                else clipped_message(raw_audio, sample_rate)
            )
            return

        phase = self.setup_state
        self._cancel_analysis_workers()
        generation = self._analysis_generation + 1
        self._analysis_generation = generation
        self._capture_phase = phase
        self.setup_state = "capture_analyzing"
        self.start_button.setEnabled(False)
        self.warning_label.setText("Checking the recording...")
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        start_capture_analysis(
            self,
            partial(
                _analyze_capture,
                raw_audio,
                sample_rate,
                phase,
                self._current_capture_metadata,
            ),
            lambda result, token=generation: self._on_capture_analyzed(
                result, raw_audio, phase, token
            ),
            lambda error, token=generation: self._on_capture_analysis_failed(
                error, phase, token
            ),
        )

    def _on_capture_analysis_failed(self, error: str, phase: str, generation: int) -> None:
        if generation != self._analysis_generation or self._close_requested:
            return
        self.setup_state = phase
        self._on_recording_failed(plain_error(f"Recording error: {error}"))

    def _on_capture_analyzed(
        self,
        analysis: CaptureAnalysis,
        raw_audio: np.ndarray,
        phase: str,
        generation: int,
    ) -> None:
        """Decide about a finished take once its post-processing is done."""
        if generation != self._analysis_generation or self._close_requested:
            return
        self.setup_state = phase
        self.start_button.setEnabled(True)
        analysis_audio = analysis.audio

        if self.setup_state == "noise_recording":
            self.preview_noise_audio = raw_audio
            self.noise_audio = analysis_audio
            self.noise_metadata = self._current_capture_metadata
            quick_quality = analysis.noise_quality
            assert quick_quality is not None
            if quick_quality.must_recapture:
                self.noise_metadata = None
                guidance = quick_quality.guidance or (
                    "Keep quiet and record steady room tone again.",
                )
                self._reject_capture(" ".join(guidance))
                return
            self.setup_state = "noise_ready"
            self.start_button.setText("Record Voice")
            self.phase_label.setText("Room noise captured")
            if quick_quality.status == "invalid":
                found = plain_reasons(quick_quality.reasons)
                note = (
                    f"{found[0]} " if found else "The room noise was not steady. "
                ) + _UNSTEADY_ROOM_ADVICE
            elif SILENT_REFERENCE_REASON in quick_quality.reasons:
                note = SILENT_REFERENCE_GUIDANCE
            elif _CLICK_REASONS.intersection(quick_quality.reasons):
                note = _CLICK_ADVICE
            else:
                note = ""
            self.warning_label.setText(
                (f"{note} " if note else "")
                + "Now read the passage aloud for the full 10 seconds."
            )
            self.warning_label.setStyleSheet(
                message_text_style("warn" if note else "ok", strong=True)
            )
            self.progress_bar.setValue(0)
            self.time_label.setText(f"Time remaining: {VOICE_RECORDING_DURATION:.0f}s")
            return

        if self.setup_state == "verification_recording":
            self.preview_verification_audio = raw_audio
            self._complete_verification(analysis_audio)
            return

        if analysis.silent:
            # Silence while talking, unlike silent room noise, is a microphone fault.
            self._reject_capture(
                "No sound came from the microphone. Check that it is not muted and "
                "that the right input is selected, then record again."
            )
            return

        self.preview_voice_audio = raw_audio
        self.voice_audio = analysis_audio
        self.voice_metadata = self._current_capture_metadata
        self.setup_state = "analyzing"
        self.start_button.setEnabled(False)
        self.start_button.setText("Analyzing...")
        self.phase_label.setText("Analyzing recordings")
        # The recording's "Complete!" would read as the analysis being done.
        self.time_label.setText("Analysis in progress")
        self.warning_label.setText("Building recommendations for your voice chain.")
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        self._start_analysis()

    def _start_analysis(self) -> None:
        if self.noise_audio is None or self.voice_audio is None:
            self._on_analysis_failed(plain_error("Missing room-noise or speech capture."))
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_analysis_failed(plain_error("Could not find audio processor"))
            return

        vad_available = False
        try:
            vad_available = bool(parent.processor.is_vad_available())
        except Exception:
            vad_available = False

        limiter_settings: Mapping[str, Any] | None = None
        target_lufs = self.target_lufs_spin.value()
        limiter_state = getattr(parent, "limiter_state", None)
        get_limiter_settings = getattr(limiter_state, "get_settings", None)
        if callable(get_limiter_settings):
            candidate_limiter_settings = get_limiter_settings()
            if isinstance(candidate_limiter_settings, Mapping):
                limiter_settings = candidate_limiter_settings

        self._cancel_analysis_workers()
        generation = self._analysis_generation + 1
        self._analysis_generation = generation
        target_preset = self.get_selected_curve()
        dynamics_intensity = str(self.dynamics_combo.currentData())
        custom_target_p95_db = self.custom_p95_spin.value()
        custom_peak_cap_db = self.custom_peak_spin.value()
        suppression = _suppressor_settings(parent)
        incumbent = _chain_settings(parent)
        if hasattr(parent, "eq_state"):
            incumbent["eq"] = deepcopy(parent.eq_state.get_settings())
        if hasattr(parent, "gate_state"):
            incumbent["gate"] = deepcopy(parent.gate_state.get_settings())
        if suppression is not None:
            incumbent["rnnoise"] = suppression
        raw_noise_audio = self.preview_noise_audio
        raw_speech_audio = self.preview_voice_audio
        try:
            current_chain = _chain_settings(
                parent,
                full_chain=True,
                input_pre_filtered=raw_speech_audio is None,
            )
            input_cleanup_mode = str(current_chain["input_cleanup_mode"])
            processing_mode = str(current_chain["processing_mode"])
            # Raw monitoring and Bypass are both left behind when the candidate
            # is applied, so the analysis matches the Normal path it will run.
            if processing_mode in _UNPROCESSED_MODES:
                processing_mode = "normal"
        except Exception as error:
            self._on_analysis_failed(
                f"AudioForge could not read your current audio settings ({error}). "
                "Close and reopen this window, then try again."
            )
            return
        self._candidate_metadata = _candidate_metadata(
            "full_voice_setup",
            target={
                "curve": target_preset,
                "target_lufs": float(target_lufs),
                "dynamics_intensity": dynamics_intensity,
            },
            capture={
                "generation": generation,
                "sample_rate": _processor_sample_rate(parent),
                "noise_samples": int(self.noise_audio.size),
                "voice_samples": int(self.voice_audio.size),
                "context_key": self._capture_context_key,
            },
            options={
                "vad_available": vad_available,
                "dynamics_intensity": dynamics_intensity,
                "custom_target_p95_db": float(custom_target_p95_db),
                "custom_peak_cap_db": float(custom_peak_cap_db),
            },
            allowed_scope=("eq", "gate", "deesser", "compressor", "limiter", "suppression")
            if suppression is not None else ("eq", "gate", "deesser", "compressor", "limiter"),
        )
        self.curve_group.setEnabled(False)
        self.dynamics_group.setEnabled(False)
        self.progress_bar.setValue(0)
        worker = VoiceSetupWorker(
            self.noise_audio,
            self.voice_audio,
            _processor_sample_rate(parent),
            target_preset,
            vad_available=vad_available,
            dynamics_intensity=dynamics_intensity,
            custom_target_p95_db=custom_target_p95_db,
            custom_peak_cap_db=custom_peak_cap_db,
            limiter_settings=limiter_settings,
            noise_metadata=self.noise_metadata,
            voice_metadata=self.voice_metadata,
            target_lufs=target_lufs,
            noise_model=str((suppression or {}).get("model", "rnnoise")),
            suppressor_strength=float((suppression or {}).get("strength", 1.0)),
            incumbent_settings=incumbent,
            raw_noise_audio=raw_noise_audio,
            raw_speech_audio=raw_speech_audio,
            input_cleanup_mode=input_cleanup_mode,
            processing_mode=processing_mode,
        )
        self.analysis_worker = worker
        self._analysis_workers.append(worker)
        worker.step_progress.connect(
            lambda step_name, percentage, token=generation: self._on_analysis_step(
                step_name, percentage, token
            )
        )
        worker.result_ready.connect(
            lambda result, token=generation: self._on_analysis_complete(result, token)
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

    def _on_analysis_step(
        self,
        step_name: str,
        percentage: int,
        generation: int | None = None,
    ) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        self.warning_label.setText(step_name)
        self.progress_bar.setValue(percentage)

    def _on_analysis_complete(
        self, setup_result: dict[str, Any], generation: int | None = None
    ) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        setup_result = deepcopy(setup_result)
        self._reported = False
        candidate = deepcopy(self._candidate_metadata or {})
        if isinstance(candidate, dict):
            candidate["verified_stages"] = []
        setup_result["_candidate"] = candidate
        self._reset_verification_state()
        self.setup_result = setup_result
        self._review_current.clear()
        self._final_comparison_pending = False
        candidate_error = _candidate_settings_error(setup_result)
        candidate_complete = candidate_error is None
        self.compare_button.setEnabled(candidate_complete)
        self.curve_group.setEnabled(not candidate_complete)
        self.dynamics_group.setEnabled(not candidate_complete)
        self.target_lufs_spin.setEnabled(not candidate_complete)
        self.setup_state = (
            "completed"
            if candidate_complete
            else ("noise_ready" if self.noise_audio is not None else "idle")
        )
        self.start_button.setText(
            "Apply Voice Setup"
            if candidate_complete
            else ("Record Voice Again" if self.noise_audio is not None else "Start Voice Setup")
        )
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        if not candidate_complete:
            self.retake_btn.setVisible(True)
        diagnostics = setup_result["diagnostics"]
        advisory = candidate_complete and not diagnostics.get("apply_recommended", False)
        self._advisory_only = advisory
        self.phase_label.setText(
            "Recommendations incomplete"
            if not candidate_complete
            else ("Suggestions only: not every check passed" if advisory else "Recommendations ready")
        )
        if advisory:
            self.start_button.setText("Apply Suggestions")
            self.start_button.setStyleSheet("")
        if not candidate_complete:
            self.warning_label.setText(
                "The recommendations are incomplete. "
                f"{plain_incomplete(candidate_error)} Record the voice passage again to retry."
            )
            state = "warn"
        elif not advisory:
            self.warning_label.setText("All checks passed. Review the settings below, then apply them.")
            state = "ok"
        else:
            # The capture problems are said once, in the lines of the review.
            reasons = plain_reasons(other_reasons(diagnostics)) or (
                []
                if capture_issue_lines(diagnostics, setup_result.get("compressor_settings"))
                else ["The recording was too weak to check the settings fully."]
            )
            self.warning_label.setText(
                " ".join(
                    ["These settings did not pass every check, so treat them as suggestions.", *reasons]
                )
            )
            state = "warn"
        self.warning_label.setStyleSheet(message_text_style(state, strong=True))
        self.progress_bar.setValue(100)
        self._show_summary(setup_result)
        if not candidate_complete:
            self._report(
                "refused",
                ["incomplete_result"],
                [self.warning_label.text()],
            )
        elif advisory:
            self._report(
                "advisory",
                [
                    *(reason_code(str(reason)) for reason in diagnostics.get("uncertainty_reasons") or ()),
                    *capture_issue_codes(diagnostics),
                ]
                or ["weak_capture"],
                [self.warning_label.text()],
            )

    def _review_text(
        self,
        text: str,
        recommended: Mapping[str, Any],
        state_name: str,
        describe: Callable[[Mapping[str, Any]], str],
        comparable: Callable[[Mapping[str, Any]], Mapping[str, Any]] = lambda settings: settings,
    ) -> str:
        """Add what the stage was set to before this setup, when the owner knows."""
        try:
            # Read once: later renders happen with the candidate already applied.
            if state_name not in self._review_current:
                current = getattr(self.parent(), state_name).get_settings()
                if not isinstance(current, Mapping):
                    return text
                self._review_current[state_name] = deepcopy(current)
            current = self._review_current[state_name]
            before = describe(current)
        except Exception:
            return text
        if before != text:
            return f"{text}\nBefore: {before}"
        # The line shows only the headline values of a stage.
        proposed, previous = comparable(recommended), comparable(current)
        for key in proposed.keys() & previous.keys():
            try:
                same = bool(np.allclose(proposed[key], previous[key], rtol=0.0, atol=1e-3))
            except (TypeError, ValueError):
                same = proposed[key] == previous[key]
            if not same:
                return f"{text} (only finer settings change)"
        return f"{text} (unchanged)"

    def _show_capture_issues(self, setup_result: Mapping[str, Any]) -> None:
        """One line under the headline for each reason the capture was weak."""
        while self.capture_issues_layout.count():
            item = self.capture_issues_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for line in capture_issue_lines(
            setup_result["diagnostics"], setup_result.get("compressor_settings")
        ):
            label = QLabel(line)
            label.setWordWrap(True)
            label.setStyleSheet(status_chip_style("warn"))
            self.capture_issues_layout.addWidget(label)

    def _report(
        self,
        outcome: str,
        reason_codes: list[str],
        messages: list[str],
        numbers: Mapping[str, Any] | None = None,
    ) -> None:
        """Tell the main window how this run ended, for the diagnostics export."""
        # A refused verification take puts the previous settings back, which the earlier
        # result no longer describes, so it is always reported.
        refused_take = (
            "recording_rejected" in reason_codes
            and self.setup_state != "verification_recording"
        )
        if refused_take and self._result_reported:
            return
        if not refused_take and outcome != "failed":
            self._result_reported = True
        diagnostics = (self.setup_result or {}).get("diagnostics") or {}
        measured: dict[str, Any] = numbers_from(
            diagnostics,
            (
                "setup_confidence",
                "capture_confidence",
                "noise_referenced_snr_db",
                "speech_rms_db",
                "speech_peak_db",
                "noise_rms_db",
                "short_term_lufs",
                "vad_active_duration_s",
                "weak_capture",
                "apply_recommended",
                "loudness_lowered_db",
            ),
        )
        measured.update(numbers or {})
        if self._level_override is not None:
            measured["level_override"] = self._level_override
        self._reported = True
        report_setup_result(
            _find_eq_state_owner(self.parent()),
            "voice_setup",
            outcome,
            reason_codes=reason_codes,
            messages=messages,
            numbers=measured,
        )

    def _show_eq_comparison(self, eq_settings: Mapping[str, Any] | None) -> None:
        """Draw the proposed EQ over the one from before this setup, or nothing if there is none."""
        self.eq_compare.setVisible(False)
        owner = _find_eq_state_owner(self.parent())
        if eq_settings is None or owner is None:
            return
        try:
            snapshot = self._pre_setup_snapshot
            preset = (
                snapshot["preset"]
                if isinstance(snapshot, Mapping)
                else owner._get_current_preset().to_dict()
            )
            now = EQSettings.from_dict(preset["eq"])
            mode = (
                snapshot.get("processing_mode")
                if isinstance(snapshot, Mapping)
                else owner._processing_mode()
            )
            self.eq_compare.set_curves(
                now,
                _candidate_eq(now, eq_settings),
                float(_processor_sample_rate(owner)),
                mode if mode in _UNPROCESSED_MODES else "",
            )
        except Exception:
            logger.warning("The EQ comparison could not be drawn", exc_info=True)
            return
        self.eq_compare.setVisible(True)

    def _refresh_review(self) -> None:
        """Redraw the review lines after the applied settings or their status changed."""
        # Only a result that was drawn once is complete enough to draw again.
        if self.setup_result is not None and self._summary_result is self.setup_result:
            self._show_summary(self.setup_result)

    def _show_summary(self, setup_result: dict[str, Any]) -> None:
        diagnostics = setup_result["diagnostics"]
        overall_conf = float(diagnostics["setup_confidence"])
        word, state = quality_word(
            overall_conf, weak=bool(diagnostics.get("weak_capture", False))
        )
        if self._final_comparison_pending:
            state = "ok"
            self.overall_label.setText(
                "These settings passed the check on your second recording."
            )
        elif diagnostics.get("apply_recommended", False):
            self.overall_label.setText(f"{word}. All checks passed.")
        else:
            # Settings that did not pass the checks are never "Good", however good the capture.
            state = "bad" if state == "bad" else "warn"
            self.overall_label.setText(
                f"{word}. These settings did not pass every check, so they are suggestions only."
            )
        details = [quality_tooltip(overall_conf)]
        self.overall_label.setToolTip(" ".join(details))
        self.overall_label.setStyleSheet(status_chip_style(state))
        joint = diagnostics.get("joint_tuning")
        if isinstance(joint, Mapping):
            found = plain_reasons(joint.get("reasons", []))
            if found:
                outcome = found[0]
            elif joint.get("withheld_for") == "questionable_noise_reference":
                outcome = (
                    "Noise suppression and the gate were not tuned because the room-noise "
                    "recording was not steady. Your current settings are kept."
                )
            elif joint.get("apply_recommended"):
                outcome = "New noise suppression and gate settings passed the checks."
            else:
                outcome = "Your current noise suppression and gate settings are kept."
            self.overall_label.setText(self.overall_label.text() + f"\n{outcome}")
            details.extend(found[1:])
            self.overall_label.setToolTip(" ".join(details))
        self._show_capture_issues(setup_result)

        eq_settings = setup_result.get("eq_settings")
        eq_error = _candidate_eq_settings_error(eq_settings)
        if eq_error is None:
            assert isinstance(eq_settings, Mapping)

            def describe_eq(settings: Mapping[str, Any]) -> str:
                return f"max correction {_correction_depth_db(settings):.1f} dB"

            tone_text = "Tone: " + (
                self._review_text(
                    describe_eq(eq_settings),
                    eq_settings,
                    "eq_state",
                    describe_eq,
                    _correction_layer,
                )
                if eq_settings.get("apply_recommended", True)
                else "kept as it is"
            )
            before_eq = self._review_current.get("eq_state", {})
            if (
                self._pre_setup_snapshot is None
                and eq_settings.get("enabled", True)
                and eq_settings.get("apply_recommended", True)
                and before_eq.get("enabled", True) is False
            ):
                tone_active = any(
                    band["enabled"]
                    and (
                        band["gain_db"] != 0.0
                        # These shapes change the sound at 0 dB gain.
                        or band["filter_type"] not in ("bell", "low_shelf", "high_shelf")
                    )
                    for band in before_eq.get("bands", ())
                )
                tone_text += (
                    "\nThe EQ is off now; Apply will switch it on"
                    + (
                        ", and your current tone settings will take effect too."
                        if tone_active
                        else "."
                    )
                )
            self.eq_label.setText(tone_text)
            self.eq_label.setToolTip(
                "Quality score of the tone step "
                f"{_format_percent(eq_settings.get('analysis_confidence', 0.0))}: "
                "how consistent your recording was across the voice range."
            )
            self.eq_label.setStyleSheet(
                status_chip_style(
                    "ok"
                    if diagnostics.get("apply_recommended", False) or self._final_comparison_pending
                    else "info"
                )
            )
            self._show_eq_comparison(eq_settings)
        else:
            self._show_eq_comparison(None)
            self.eq_label.setText(
                "Tone left unchanged. "
                f"{plain_incomplete(str(setup_result.get('eq_error') or eq_error))}"
            )
            self.eq_label.setToolTip("")
            self.eq_label.setStyleSheet(status_chip_style("warn"))

        gate = setup_result["gate_settings"]
        gate_mode = str(diagnostics["gate_mode_label"])
        self.gate_label.setText(
            "Noise gate: "
            + self._review_text(
                f"{_GATE_WORDS.get(gate_mode, gate_mode)}, "
                f"opens above {_format_db(gate['threshold_db'])}",
                gate,
                "gate_state",
                lambda settings: (
                    f"{_GATE_WORDS.get(GATE_MODE_LABELS[settings['gate_mode']], '')}, "
                    f"opens above {_format_db(settings['threshold_db'])}"
                ),
            )
        )
        self.gate_label.setToolTip(f"Speech detection sensitivity {gate['vad_threshold']:.2f}")
        self.gate_label.setStyleSheet(status_chip_style("info"))

        deesser = setup_result["deesser_settings"]
        deesser_state = "ok" if deesser["enabled"] else "info"

        def describe_deesser(settings: Mapping[str, Any]) -> str:
            return (
                f"{'on' if settings['enabled'] else 'off'}, "
                f"amount {settings['auto_amount'] * 100.0:.0f}%"
            )

        self.deesser_label.setText(
            "De-esser: "
            + self._review_text(
                describe_deesser(deesser), deesser, "deesser_state", describe_deesser
            )
        )
        self.deesser_label.setToolTip(
            f"Band {deesser['low_cut_hz']:.0f} to {deesser['high_cut_hz']:.0f} Hz"
        )
        self.deesser_label.setStyleSheet(status_chip_style(deesser_state))

        compressor = setup_result["compressor_settings"]
        compressor_diag = diagnostics.get("compressor_calibration") or {}
        intensity = str(
            compressor.get("dynamics_intensity")
            or diagnostics.get("dynamics_intensity")
            or "balanced"
        )
        if compressor.get("dynamics_customized"):
            intensity = "customized"
        measured_gr = compressor_diag.get("measured_gain_reduction_db")
        makeup_text = (
            "auto makeup"
            if compressor["auto_makeup_enabled"]
            else f"{compressor['makeup_gain_db']:.1f} dB makeup"
        )

        def describe_compressor(settings: Mapping[str, Any]) -> str:
            loudness = (
                f"aiming for {settings['target_lufs']:.0f} LUFS"
                if settings["auto_makeup_enabled"]
                else f"{settings['makeup_gain_db']:.1f} dB makeup"
            )
            return (
                f"{'on' if settings.get('enabled', True) else 'off'}, "
                f"{settings['ratio']:.1f}:1 above "
                f"{_format_db(settings['threshold_db'])}, {loudness}"
            )

        lowered = float(diagnostics.get("loudness_lowered_db") or 0.0)
        if lowered <= 0.0:
            loudness_note = ""
        elif compressor["auto_makeup_enabled"]:
            loudness_note = (
                f"\nYou asked for {diagnostics['compressor_target_lufs']:.0f} LUFS; "
                "it was lowered so peaks do not overload the limiter."
            )
        else:
            loudness_note = (
                f"\nMakeup was held {round(lowered, 1):g} dB lower so peaks do not overload the limiter."
            )
        self.compressor_label.setText(
            "Compressor: "
            + self._review_text(
                describe_compressor(compressor),
                compressor,
                "compressor_state",
                describe_compressor,
            )
            + loudness_note
        )
        self.compressor_label.setToolTip(
            f"Compression style: {intensity.capitalize()}; "
            f"{'automatic loudness' if compressor['auto_makeup_enabled'] else makeup_text}; "
            f"on your recording it evens out your voice by {_format_db(measured_gr)}."
        )
        self.compressor_label.setStyleSheet(status_chip_style("info"))
        self.summary_group.setVisible(True)
        self._summary_result = setup_result

    def _candidate_identity_error(self, parent: Any) -> str | None:
        if self.setup_result is None:
            return "voice setup result is missing"
        candidate = self.setup_result.get("_candidate")
        expected = self._candidate_metadata
        if not isinstance(candidate, Mapping) or not isinstance(expected, Mapping):
            return "candidate identity is missing"
        if candidate.get("scope") != "full_voice_setup":
            return "candidate scope is not full voice setup"
        allowed_scope = tuple(candidate.get("allowed_scope") or ())
        if allowed_scope not in (("eq", "gate", "deesser", "compressor", "limiter"),
                                 ("eq", "gate", "deesser", "compressor", "limiter", "suppression")):
            return "candidate scope is incomplete"
        if self.setup_result.get("suppressor_settings") is not None and "suppression" not in allowed_scope:
            return "candidate suppression scope is missing"
        for key in ("scope", "allowed_scope", "target", "options", "capture_identity"):
            if candidate.get(key) != expected.get(key):
                return "candidate options or capture identity changed"

        target = candidate.get("target")
        options = candidate.get("options")
        if not isinstance(target, Mapping) or not isinstance(options, Mapping):
            return "candidate target or options are missing"
        if not isinstance(target.get("target_lufs"), (int, float)):
            return "candidate target loudness is invalid"
        if target.get("curve") is None or target.get("dynamics_intensity") is None:
            return "candidate target is incomplete"
        if any(
            key not in options
            for key in (
                "dynamics_intensity",
                "custom_target_p95_db",
                "custom_peak_cap_db",
            )
        ):
            return "candidate options are incomplete"

        capture = candidate.get("capture_identity")
        if not isinstance(capture, Mapping):
            return "candidate capture identity is missing"
        if self.noise_audio is None or self.voice_audio is None:
            return "voice setup capture is incomplete"
        if capture.get("noise_samples") != int(self.noise_audio.size):
            return "room-noise capture changed after analysis"
        if capture.get("voice_samples") != int(self.voice_audio.size):
            return "voice capture changed after analysis"
        try:
            if capture.get("sample_rate") != _processor_sample_rate(parent):
                return "audio sample rate changed after analysis"
        except RuntimeError:
            return "candidate capture sample rate is unavailable"
        if not _voice_setup_context_matches(
            capture.get("context_key"),
            _owner_calibration_context_key(parent),
            allow_unprocessed_to_normal=(
                isinstance(self._pre_setup_snapshot, Mapping)
                and self._pre_setup_snapshot.get("processing_mode") in _UNPROCESSED_MODES
            ),
        ):
            return (
                "The audio route, input settings or processing mode changed after capture"
            )
        return None

    def _compare_recording(self) -> None:
        if (
            self.setup_state not in {"completed", "final_comparison_ready"}
            or self.voice_audio is None
            or self.setup_result is None
        ):
            return
        owner = _find_eq_state_owner(self.parent())
        if owner is None:
            return
        error = self._candidate_identity_error(owner)
        if error:
            QMessageBox.warning(self, "Recording out of date", plain_stale(error))
            return
        from .listening_comparison_dialog import ListeningComparisonDialog

        try:
            snapshot = self._pre_setup_snapshot
            if self._final_comparison_pending and isinstance(snapshot, Mapping):
                current_chain = deepcopy(snapshot.get("comparison_chain_settings") or {})
                preset_payload = snapshot.get("preset")
                if not isinstance(preset_payload, Mapping):
                    raise ValueError("original processing settings are unavailable")
                current_settings = Preset.from_dict(dict(preset_payload)).eq.to_dict()
                if not current_chain:
                    raise ValueError("original processing settings are unavailable")
            else:
                current_chain = _chain_settings(
                    owner,
                    full_chain=True,
                    input_pre_filtered=self.preview_voice_audio is None,
                )
                current_settings = owner.eq_state.get_settings()
            # The EQ Apply writes: the correction layer over the retained tone layer.
            candidate_info = _candidate_preset(owner, self.setup_result)
            if candidate_info is None:
                raise ValueError("current preset is unavailable")
            proposed_eq = candidate_info[0].eq.to_dict()
        except Exception as error:
            QMessageBox.warning(
                self,
                "Comparison unavailable",
                f"The comparison could not be prepared ({error}). Close and reopen this window.",
            )
            return
        proposed_chain = deepcopy(current_chain)
        proposed_chain.update(
            {
                name: deepcopy(self.setup_result[f"{name}_settings"])
                for name in ("deesser", "compressor", "limiter")
            }
        )
        # Raw monitoring and Bypass are both left behind while applying setup.
        proposed_chain["processing_mode"] = "normal"
        gate_settings = self.setup_result.get("gate_settings")
        if isinstance(gate_settings, Mapping):
            proposed_chain["gate"] = deepcopy(gate_settings)
        suppression = self.setup_result.get("suppressor_settings")
        if isinstance(suppression, Mapping):
            proposed_chain["rnnoise"] = deepcopy(suppression)
        dialog = ListeningComparisonDialog(
            audio_data=self.preview_voice_audio
            if self.preview_voice_audio is not None
            else self.voice_audio,
            sample_rate=_processor_sample_rate(owner),
            current_settings=current_settings,
            proposed_settings=proposed_eq,
            current_chain_settings=current_chain,
            proposed_chain_settings=proposed_chain,
            parent=self,
        )
        _set_temporary_mute(owner, "listening_comparison", True)
        try:
            keep = dialog.exec() == int(QDialog.DialogCode.Accepted)
        finally:
            _set_temporary_mute(owner, "listening_comparison", False)
            dialog.deleteLater()
        if keep:
            if self._final_comparison_pending:
                self._finalize_verified_setup()
            else:
                self._apply_setup()
        elif self._final_comparison_pending:
            restored = self._restore_pre_setup_snapshot()
            self._final_comparison_pending = False
            self._refresh_review()
            self.setup_state = "completed"
            self.compare_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.start_button.setText(_APPLY_VERIFY_AGAIN_LABEL)
            self.phase_label.setText(
                "Final comparison declined; previous settings put back"
                if restored
                else "Final comparison declined; previous settings not fully put back"
            )
            self.warning_label.setText(
                "You kept your old settings. The new ones are still here if you change your mind. "
                + (
                    "Your old settings are active."
                    if restored
                    else "Your old settings could not be fully put back; press Retake to try again."
                )
            )
            self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
            self._report(
                "rolled_back",
                ["final_comparison_declined"],
                [self.warning_label.text()],
            )

    def _apply_setup(self) -> None:
        parent = _find_eq_state_owner(self.parent())
        if not parent or self.setup_result is None:
            QMessageBox.critical(
                self,
                "Error",
                "The settings could not be applied. Close this window and open it again.",
            )
            return

        candidate_error = _candidate_settings_error(self.setup_result)
        if candidate_error is not None:
            QMessageBox.critical(
                self,
                "Incomplete Voice Setup",
                f"{plain_incomplete(candidate_error)} Nothing was changed. "
                "Record the voice passage again.",
            )
            return

        if self._pre_setup_snapshot is None:
            self._reset_verification_state()

        self._final_comparison_pending = False

        identity_error = self._candidate_identity_error(parent)
        if identity_error is not None:
            QMessageBox.critical(
                self,
                "Recording out of date",
                f"{plain_stale(identity_error)} Nothing was changed.",
            )
            return

        diagnostics = self.setup_result.get("diagnostics") or {}
        if not diagnostics.get("apply_recommended", False):
            reasons = plain_reasons(diagnostics.get("uncertainty_reasons") or ()) or [
                "The recording was too weak to check the settings fully."
            ]
            reply = QMessageBox.question(
                self,
                "Apply Suggested Settings?",
                "These settings did not pass every check:\n\n"
                + "\n".join(f"- {reason}" for reason in reasons)
                + "\n\nApply them anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if self._pre_setup_snapshot is None:
            apply_configuration = getattr(parent, "apply_processing_configuration", None)
            getter = getattr(parent, "_get_current_preset", None)
            mode_getter = getattr(parent, "_processing_mode", None)
            if not callable(apply_configuration) or not callable(getter):
                QMessageBox.critical(
                    self,
                    "Error",
                    "AudioForge cannot apply these settings in one step, so nothing was "
                    "changed. Close this window and open it again.",
                )
                return
            try:
                current_preset = getter()
                if not isinstance(current_preset, Preset):
                    raise TypeError("current processing configuration is unavailable")
                compressor_settings = parent.compressor_state.get_settings(
                    include_calibration=True
                )
                comparison_chain = _chain_settings(
                    parent,
                    full_chain=True,
                    input_pre_filtered=self.preview_voice_audio is None,
                    preset=current_preset,
                )
                if not comparison_chain:
                    raise RuntimeError("current processing chain is unavailable")
                if not callable(mode_getter):
                    raise TypeError("processing mode cannot be captured")
                processing_mode = str(mode_getter())
            except Exception as error:
                logger.warning(
                    "Failed to capture the complete voice setup rollback snapshot",
                    exc_info=True,
                )
                QMessageBox.critical(
                    self,
                    "Error",
                    "AudioForge could not save your current settings first, so nothing was "
                    f"changed ({error}).",
                )
                return
            snapshot: dict[str, Any] = {
                "preset": current_preset.to_dict(),
                "comparison_chain_settings": comparison_chain,
                "processing_mode": processing_mode,
                "calibration_context_key": _owner_calibration_context_key(parent),
            }
            snapshot["compressor_metadata"] = {
                key: deepcopy(compressor_settings[key])
                for key in (
                    "dynamics_intensity",
                    "dynamics_profile",
                    "dynamics_customized",
                )
                if key in compressor_settings
            }
            reliability = compressor_settings.get("noise_reference_reliability")
            if isinstance(reliability, (int, float)) and not isinstance(reliability, bool):
                reliability_value = float(reliability)
                if np.isfinite(reliability_value):
                    snapshot["noise_reference_reliability"] = reliability_value
            self._pre_setup_snapshot = deepcopy(snapshot)
        try:
            self._apply_candidate_configuration(parent)
            self._sync_setup_result_with_applied_states(parent)
        except Exception as exc:
            restored = self._restore_pre_setup_snapshot()
            message = (
                f"The new settings could not be applied, so your previous settings were put back ({exc})."
                if restored
                else "The new settings could not be applied, and your previous settings could not "
                f"be fully put back ({exc}). Press Retake to try again."
            )
            QMessageBox.critical(self, "Error", message)
            self._report("failed", ["apply_failed"], [message])
            return
        mode_note = self._mode_switch_note()
        self.setup_state = "verification_ready"
        self.compare_button.setEnabled(False)
        self.start_button.setText("Record Verification Passage")
        self.phase_label.setText("New settings are on for now")
        self.warning_label.setText(
            "The new settings are on for now. Read the passage once more so "
            "AudioForge can check them on a second recording."
            + (f" {mode_note}" if mode_note else "")
        )
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        self._refresh_review()

    def _mode_switch_note(self) -> str:
        """Say when setup took the user out of Bypass or Raw monitor."""
        mode = (self._pre_setup_snapshot or {}).get("processing_mode")
        name = {"bypass": "Bypass", "raw": "Raw monitor"}.get(str(mode))
        return f"{name} was switched to Normal so the settings take effect." if name else ""

    def _apply_candidate_configuration(self, parent: Any) -> None:
        if self.setup_result is None:
            raise ValueError("voice setup result is missing")
        candidate_error = _candidate_settings_error(self.setup_result)
        if candidate_error is not None:
            raise ValueError(candidate_error)
        candidate = self.setup_result.get("_candidate")
        if not isinstance(candidate, Mapping) or candidate.get("scope") != "full_voice_setup":
            raise ValueError("candidate scope is not full voice setup")
        identity_error = self._candidate_identity_error(parent)
        if identity_error is not None:
            raise ValueError(identity_error)
        if not isinstance(self.setup_result.get("limiter_settings"), Mapping) or not isinstance(
            self.setup_result.get("eq_settings"), Mapping
        ):
            raise ValueError("candidate settings are incomplete")
        apply_configuration = getattr(parent, "apply_processing_configuration", None)
        if not callable(apply_configuration):
            raise ValueError("processing owner does not support transactional configuration")
        candidate_info = _candidate_preset(parent, self.setup_result)
        if candidate_info is None:
            raise ValueError("current preset is unavailable")
        candidate_preset, reliability = candidate_info
        compressor_metadata = self.setup_result.get("compressor_settings")
        metadata: dict[str, Any] = {}
        if isinstance(compressor_metadata, Mapping):
            metadata = {
                key: deepcopy(compressor_metadata[key])
                for key in (
                    "dynamics_intensity",
                    "dynamics_profile",
                    "dynamics_customized",
                )
                if key in compressor_metadata
            }
        if metadata and reliability is None:
            try:
                current_compressor = parent.compressor_state.get_settings(
                    include_calibration=True
                )
                current_reliability = current_compressor.get(
                    "noise_reference_reliability"
                )
                if (
                    isinstance(current_reliability, (int, float))
                    and not isinstance(current_reliability, bool)
                    and np.isfinite(float(current_reliability))
                ):
                    reliability = float(current_reliability)
            except Exception:
                logger.warning(
                    "Could not preserve current noise-reference reliability while applying voice setup",
                    exc_info=True,
                )
        apply_configuration(
            candidate_preset,
            noise_reference_reliability=reliability,
            compressor_metadata=metadata or None,
        )
        if hasattr(parent, "status_message"):
            parent.status_message.showMessage(
                "New voice settings are on for now; a second recording will check them", 5000
            )

    def _sync_setup_result_with_applied_states(self, parent: Any) -> None:
        """Use the values the controls accepted for the verification pass."""
        if self.setup_result is None:
            raise ValueError("voice setup result is missing")

        if self.setup_result.get("suppressor_settings") is not None:
            self.setup_result["suppressor_settings"] = _suppressor_settings(parent)
        compressor = dict(self.setup_result.get("compressor_settings") or {})
        compressor.update(
            parent.compressor_state.get_settings(include_calibration=True)
        )
        self.setup_result["compressor_settings"] = compressor

        deesser = dict(self.setup_result.get("deesser_settings") or {})
        deesser.update(parent.deesser_state.get_settings())
        self.setup_result["deesser_settings"] = deesser

        self.setup_result["limiter_settings"] = dict(
            parent.limiter_state.get_settings()
        )

        eq_settings = self.setup_result.get("eq_settings")
        if eq_settings is not None:
            applied_eq = parent.eq_state.get_settings()
            eq_settings = dict(eq_settings)
            for key in ("schema_version", "bands", "layers"):
                if key in applied_eq:
                    eq_settings[key] = deepcopy(applied_eq[key])
            for key in ("enabled", "band_freqs", "band_gains", "band_qs"):
                if key not in applied_eq:
                    raise ValueError(f"EQ state did not report {key}")
                value = applied_eq[key]
                eq_settings[key] = list(value) if key != "enabled" else bool(value)
            self.setup_result["eq_settings"] = eq_settings

        # Keep the exact post-apply chain beside the candidate. Verification
        # uses the raw capture with this full-chain snapshot; the filtered
        # arrays remain the analysis/reference captures above.
        self.setup_result["verification_chain_settings"] = _chain_settings(
            parent,
            full_chain=True,
            input_pre_filtered=False,
        )

    def _reset_verification_state(self) -> None:
        self._verification_audio = None
        self._verification_adjustment_count = 0
        self._verification_bad_capture_count = 0
        self._verification_last_reduction_metrics = None

    def _verification_settings_snapshot(self) -> tuple[Any, ...]:
        if self.setup_result is None:
            return ()
        return tuple(
            (name, deepcopy(self.setup_result.get(name)))
            for name in (
                "eq_settings",
                "deesser_settings",
                "compressor_settings",
                "limiter_settings",
            )
        )

    def _apply_verification_reduction(
        self,
        parent: Any,
        reduction_targets: list[str],
    ) -> tuple[bool, str | None]:
        """Apply only the controls implicated by the latest verification."""
        if self.setup_result is None:
            return False, "verification context is missing"

        before = self._verification_settings_snapshot()
        before_sections = {
            name: deepcopy(self.setup_result.get(name))
            for name in (
                "deesser_settings",
                "compressor_settings",
            )
        }
        previous_adjusted_target = self.setup_result.get(
            "verification_adjusted_target_lufs"
        )
        reduction = reduce_for_verification(
            self.setup_result.get("compressor_settings") or {},
            self.setup_result.get("deesser_settings") or {},
            reduction_targets,
        )
        adjustment_notes = reduction.notes
        adjusted_target_lufs = reduction.adjusted_target_lufs

        self.setup_result["compressor_settings"] = reduction.compressor
        self.setup_result["deesser_settings"] = reduction.deesser
        if self._verification_settings_snapshot() == before:
            for name, value in before_sections.items():
                self.setup_result[name] = value
            return False, "verification adjustment made no settings change"

        try:
            self._apply_candidate_configuration(parent)
            self._sync_setup_result_with_applied_states(parent)
        except Exception:
            for name, value in before_sections.items():
                self.setup_result[name] = value
            if previous_adjusted_target is None:
                self.setup_result.pop("verification_adjusted_target_lufs", None)
            else:
                self.setup_result[
                    "verification_adjusted_target_lufs"
                ] = previous_adjusted_target
            raise
        if self._verification_settings_snapshot() == before:
            for name, value in before_sections.items():
                self.setup_result[name] = value
            if previous_adjusted_target is None:
                self.setup_result.pop("verification_adjusted_target_lufs", None)
            else:
                self.setup_result[
                    "verification_adjusted_target_lufs"
                ] = previous_adjusted_target
            return False, "applied panel settings made no verification progress"
        if adjusted_target_lufs is not None:
            self.setup_result["verification_adjusted_target_lufs"] = (
                adjusted_target_lufs
            )
        return True, "; ".join(adjustment_notes) or None

    def _start_verification_worker(self, audio_data: np.ndarray) -> None:
        if self._close_requested:
            return
        setup_result = self.setup_result
        noise_audio = self.noise_audio
        voice_audio = self.voice_audio
        if (
            setup_result is None
            or noise_audio is None
            or voice_audio is None
        ):
            self._on_verification_failed(
                "Setup lost track of its recordings. Start the setup again."
            )
            return
        processor_owner = _find_processor_owner(self.parent())
        if processor_owner is None:
            self._on_verification_failed(plain_error("Could not find audio processor"))
            return

        self._cancel_analysis_workers()
        generation = self._analysis_generation + 1
        self._analysis_generation = generation
        self.setup_state = "verification_analyzing"
        self.start_button.setEnabled(False)
        self.start_button.setText("Validating...")
        self.curve_combo.setEnabled(False)
        self.phase_label.setText("Checking the new settings")
        self.time_label.setText("Analysis in progress")
        self.warning_label.setText("Comparing your second recording with the first...")
        worker = VoiceSetupVerificationWorker(
            noise_audio,
            voice_audio,
            audio_data,
            _processor_sample_rate(processor_owner),
            setup_result,
            str(
                ((self.setup_result or {}).get("_candidate") or {})
                .get("target", {})
                .get("curve", self.get_selected_curve())
            ),
            raw_noise_audio=self.preview_noise_audio,
            raw_verification_speech_audio=self.preview_verification_audio,
        )
        self.analysis_worker = worker
        self._analysis_workers.append(worker)
        worker.step_progress.connect(
            lambda step_name, percentage, token=generation: self._on_analysis_step(
                step_name, percentage, token
            )
        )
        worker.result_ready.connect(
            lambda result, token=generation: self._on_verification_complete(
                result, token
            )
        )
        worker.failed.connect(
            lambda error, token=generation: self._on_verification_failed(
                plain_error(error), token
            )
        )
        worker.finished.connect(
            lambda finished_worker=worker: self._on_analysis_thread_finished(
                finished_worker
            )
        )
        worker.start()

    def _restore_pre_setup_snapshot(self) -> bool:
        parent = _find_eq_state_owner(self.parent())
        if self._pre_setup_snapshot is None:
            return True
        if parent is None:
            logger.warning("Voice setup rollback owner disappeared")
            return False
        snapshot = self._pre_setup_snapshot
        preset_payload = snapshot.get("preset")
        apply_configuration = getattr(parent, "apply_processing_configuration", None)
        if not isinstance(preset_payload, Mapping) or not callable(apply_configuration):
            logger.warning(
                "Voice setup rollback owner does not provide transactional configuration"
            )
            return False
        try:
            snapshot_context = snapshot.get("calibration_context_key")
            current_context = _owner_calibration_context_key(parent)
            reliability = snapshot.get("noise_reference_reliability")
            if (
                not snapshot_context
                or not current_context
                or not _voice_setup_context_matches(
                    snapshot_context,
                    current_context,
                    allow_unprocessed_to_normal=(
                        snapshot.get("processing_mode") in _UNPROCESSED_MODES
                    ),
                )
            ):
                reliability = 0.0
            compressor_metadata = deepcopy(snapshot.get("compressor_metadata") or {})
            if not isinstance(compressor_metadata, Mapping):
                compressor_metadata = {}
            apply_configuration(
                Preset.from_dict(dict(preset_payload)),
                noise_reference_reliability=(
                    float(reliability)
                    if isinstance(reliability, (int, float))
                    and not isinstance(reliability, bool)
                    and np.isfinite(float(reliability))
                    else None
                ),
                processing_mode=(
                    str(snapshot["processing_mode"])
                    if isinstance(snapshot.get("processing_mode"), str)
                    else None
                ),
                compressor_metadata=dict(compressor_metadata) or None,
            )
        except Exception as exc:
            logger.warning("Voice setup transactional rollback failed: %s", exc)
            return False
        self._pre_setup_snapshot = None
        self._refresh_review()
        return True

    def _complete_verification(self, audio_data: np.ndarray) -> None:
        if (
            self.setup_result is None
            or self.noise_audio is None
            or self.voice_audio is None
        ):
            restored = self._restore_pre_setup_snapshot()
            self._on_analysis_failed(
                "Setup lost track of its recordings. "
                + (
                    "Your previous settings were put back."
                    if restored
                    else "Your previous settings could not be fully put back; press Retake to try again."
                )
            )
            return
        self._verification_audio = np.ascontiguousarray(
            np.asarray(audio_data, dtype=np.float32).reshape(-1).copy()
        )
        self._start_verification_worker(self._verification_audio)

    def _finalize_verified_setup(self) -> None:
        """Persist and close after the verified candidate is accepted."""
        if self.setup_result is None:
            return
        parent = _find_eq_state_owner(self.parent())
        if parent is None:
            QMessageBox.critical(
                self, "Voice Setup", plain_error("Could not find audio processor")
            )
            return
        identity_error = self._candidate_identity_error(parent)
        if identity_error is not None:
            QMessageBox.warning(self, "Recording out of date", plain_stale(identity_error))
            restored = self._restore_pre_setup_snapshot()
            self._final_comparison_pending = False
            self._refresh_review()
            self.setup_state = "completed"
            self.compare_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.start_button.setText(_APPLY_VERIFY_AGAIN_LABEL)
            self.warning_label.setText(
                f"{plain_stale(identity_error)} "
                + (
                    "Your previous settings were put back."
                    if restored
                    else "Your previous settings could not be fully put back; press Retake to try again."
                )
            )
            self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
            self._report("rolled_back", ["stale_result"], [self.warning_label.text()])
            return
        result = self.setup_result.get("verification") or {}
        advice = result.get("suppressor_dulling_advice")
        candidate = self.setup_result.get("_candidate")
        if isinstance(candidate, dict):
            candidate["verified_stages"] = list(_VERIFIED_STAGES)
        if hasattr(parent, "status_message"):
            parent.status_message.showMessage("Verified voice setup applied", 5000)
        metrics_text = verification_summary(result)
        mode_note = self._mode_switch_note()
        QMessageBox.information(
            self,
            "Voice Setup Verified",
            "Your new settings were checked on a second recording and passed, and you "
            "kept them after the final comparison.\n\n"
            + metrics_text
            + (f"\n\n{plain_advice(advice)}" if advice else "")
            + (f"\n\n{mode_note}" if mode_note else ""),
        )
        self._report(
            "verified",
            list(result.get("reason_codes") or ["accepted"]),
            [metrics_text],
            numbers_from(
                result,
                (
                    "spectral_target_error_before_db",
                    "spectral_target_error_after_db",
                    "snr_change_db",
                    "output_true_peak_db",
                    "compressor_gain_reduction_p95_db",
                ),
            ),
        )
        self._pre_setup_snapshot = None
        self._final_comparison_pending = False
        target_curve = (
            candidate.get("target", {}).get("curve")
            if isinstance(candidate, Mapping)
            else None
        ) or self.get_selected_curve()
        self.setup_applied.emit(str(target_curve))
        from .calibration_history import persist_calibration

        persist_calibration(
            self,
            parent,
            "full_voice_setup",
            str(target_curve),
            _VERIFIED_STAGES,
        )
        self.accept()

    def _on_verification_complete(
        self, result: dict[str, Any], generation: int | None = None
    ) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        if self.setup_result is None:
            self._on_verification_failed(
                "Verification context disappeared.", generation
            )
            return
        self.setup_result["verification"] = result
        decision = str(result.get("decision", "retry"))
        reason = " ".join(plain_reasons(result.get("reasons") or ()))
        advice = result.get("suppressor_dulling_advice")
        advice_suffix = f" {plain_advice(advice)}" if advice else ""
        metrics_text = verification_summary(result)
        parent = _find_eq_state_owner(self.parent())
        if parent is None:
            self._on_verification_failed(plain_error("Could not find audio processor"))
            return
        if decision == "accept":
            identity_error = self._candidate_identity_error(parent)
            if identity_error is not None:
                restored = self._restore_pre_setup_snapshot()
                candidate = self.setup_result.get("_candidate")
                if isinstance(candidate, dict):
                    candidate["verified_stages"] = []
                self._final_comparison_pending = False
                self.setup_state = "completed"
                self.compare_button.setEnabled(True)
                self.start_button.setEnabled(True)
                self.start_button.setText(_APPLY_VERIFY_AGAIN_LABEL)
                self.phase_label.setText(
                    "Check passed, but the settings are out of date"
                    if restored
                    else "Check passed; previous settings not fully put back"
                )
                self.warning_label.setText(
                    f"{plain_stale(identity_error)} "
                    + (
                        "Your previous settings were put back."
                        if restored
                        else "Your previous settings could not be fully put back; press Retake to try again."
                    )
                )
                self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
                self._report("rolled_back", ["stale_result"], [self.warning_label.text()])
                return
            self._final_comparison_pending = True
            self._refresh_review()
            self.setup_state = "final_comparison_ready"
            self.compare_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.start_button.setText("Compare Final Candidate")
            self.phase_label.setText("Check passed; final comparison ready")
            adjustment_note = self.setup_result.get("verification_adjustment_note")
            adjusted_target_lufs = self.setup_result.get(
                "verification_adjusted_target_lufs"
            )
            adjustment_suffix = (
                f" Automatic adjustment: {adjustment_note}."
                if adjustment_note
                else ""
            )
            if isinstance(adjusted_target_lufs, (int, float)):
                adjustment_suffix += (
                    f" Effective auto target loudness: {float(adjusted_target_lufs):.0f} LUFS."
                )
            self.warning_label.setText(
                "The new settings passed the check on your second recording. The "
                "comparison is ready: keeping the new settings applies them, while "
                "closing the comparison puts your old settings back."
                + adjustment_suffix
                + advice_suffix
            )
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))
            return
        if decision == "reduce" and parent is not None:
            reduction_targets = reducible_targets(result.get("reduction_targets"))
            current_metrics = reduction_metrics(result)
            stop = verification_stop(
                self._verification_adjustment_count,
                reduction_targets,
                self._verification_last_reduction_metrics,
                current_metrics,
            )
            if stop == "no_target":
                self._on_verification_failed(
                    "Setup could not decide what to change. Record the passage again.",
                    generation,
                )
                return
            if self._verification_audio is None:
                self._on_verification_failed(
                    "The second recording is no longer available. Record it again.",
                    generation,
                )
                return
            if stop == "three_adjustments":
                self._on_verification_failed(
                    f"{reason} Setup could not make the settings safe on your recording "
                    "after three adjustments. Lower the microphone level and record again."
                    + advice_suffix,
                    generation,
                )
                return
            if stop == "no_help":
                self._on_verification_failed(
                    f"{reason} The last adjustment did not help. Lower the microphone "
                    "level and record again."
                    + advice_suffix,
                    generation,
                )
                return
            if current_metrics:
                self._verification_last_reduction_metrics = current_metrics
            try:
                changed, adjustment_note = self._apply_verification_reduction(
                    parent,
                    reduction_targets,
                )
            except Exception as exc:
                self._on_verification_failed(
                    f"The gentler settings could not be applied ({exc}).",
                    generation,
                )
                return
            if not changed:
                self._on_verification_failed(
                    f"{reason} Setup could not adjust the settings any further."
                    " Lower the microphone level and record again."
                    + advice_suffix,
                    generation,
                )
                return
            self._verification_adjustment_count += 1
            self._refresh_review()
            self.setup_state = "verification_ready"
            self._final_comparison_pending = False
            self.start_button.setText("Verify Reduced Processing")
            if adjustment_note:
                previous_note = str(
                    self.setup_result.get("verification_adjustment_note") or ""
                )
                self.setup_result["verification_adjustment_note"] = "; ".join(
                    part for part in (previous_note, adjustment_note) if part
                )
            self._start_verification_worker(self._verification_audio)
            return
        elif decision == "rollback":
            restored = self._restore_pre_setup_snapshot()
            self._final_comparison_pending = False
            self.setup_state = "completed"
            self.compare_button.setEnabled(True)
            self.start_button.setText(_APPLY_VERIFY_AGAIN_LABEL)
            reason = (
                f"{reason} "
                + (
                    "Your previous settings were put back."
                    if restored
                    else "Your previous settings could not be fully put back; "
                    "press Retake to try again."
                )
            ).strip()
        else:
            self._verification_bad_capture_count += 1
            if self._verification_bad_capture_count >= MAX_VERIFICATION_BAD_CAPTURES:
                self._on_verification_failed(
                    f"{reason} Setup stopped after three tries.",
                    generation,
                )
                return
            self.setup_state = "verification_ready"
            self.start_button.setText(
                "Retry Verification "
                f"({self._verification_bad_capture_count}/{MAX_VERIFICATION_BAD_CAPTURES})"
            )
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(decision == "rollback")
        if decision == "rollback":
            self.phase_label.setText(
                "The new settings did not pass the check; previous settings "
                + ("put back" if restored else "not fully put back")
            )
        else:
            self.phase_label.setText("The second recording did not match the first; try again")
        # A retry discards the take, so there is nothing to advise on.
        self.warning_label.setText(
            f"{reason} {metrics_text}" + (advice_suffix if decision == "rollback" else "")
        )
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        if decision == "rollback":
            self._report(
                "rolled_back",
                list(result.get("reason_codes") or ["verification_rollback"]),
                [self.warning_label.text()],
            )

    def _on_verification_failed(
        self, error: str, generation: int | None = None
    ) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        restored = self._restore_pre_setup_snapshot()
        self._final_comparison_pending = False
        self.setup_state = "completed" if self.setup_result is not None else "noise_ready"
        self.compare_button.setEnabled(self.setup_result is not None)
        self.start_button.setEnabled(True)
        self.start_button.setText(
            _APPLY_VERIFY_AGAIN_LABEL
            if self.setup_result is not None
            else "Record Voice"
        )
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        self.phase_label.setText(
            "Check failed; previous settings put back"
            if restored
            else "Check failed; previous settings not fully put back"
        )
        if not restored:
            self.warning_label.setText(
                f"{error} Your previous settings could not be fully put back; "
                "press Retake to try again."
            )
        self.summary_group.setVisible(self.setup_result is not None)
        self._report("failed", ["verification_failed"], [self.warning_label.text()])

    def _on_analysis_failed(self, error: str, generation: int | None = None) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        self.voice_audio = None
        self.preview_voice_audio = None
        self.setup_result = None
        self._reset_verification_state()
        self._candidate_metadata = None
        self._final_comparison_pending = False
        self.curve_group.setEnabled(True)
        self.dynamics_group.setEnabled(True)
        self.target_lufs_spin.setEnabled(True)
        self.setup_state = "noise_ready" if self.noise_audio is not None else "idle"
        self.start_button.setEnabled(True)
        self.start_button.setText(
            "Record Voice" if self.noise_audio is not None else "Start Voice Setup"
        )
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        self.phase_label.setText("Setup could not analyse the recordings")
        self.summary_group.setVisible(False)
        self._report("failed", ["analysis_failed"], [error])

    def _reject_capture(self, message: str) -> None:
        """Refuse a finished take and leave the dialog ready to record it again."""
        noise_step = self.setup_state == "noise_recording"
        # A verification take keeps the status its failure already set.
        verification_step = self.setup_state == "verification_recording"
        self._on_recording_failed(message, refused=True)
        self.progress_bar.setValue(0)
        self.time_label.setText("Ready to record again")
        if noise_step:
            self.phase_label.setText("Room-noise capture rejected")
        elif not verification_step:
            self.phase_label.setText("Voice capture rejected")

    def _on_recording_failed(self, error: str, *, refused: bool = False) -> None:
        verification_failed = self.setup_state == "verification_recording"
        noise_recording_failed = self.setup_state == "noise_recording"
        self.recording_timer.stop()
        # The take is over: the meter and the clock stop showing it.
        self.level_meter.set_unavailable()
        self.level_override_btn.setVisible(False)
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        self._report(
            "refused" if refused else "failed",
            ["recording_rejected" if refused else "recording_failed"],
            [error],
        )
        if verification_failed:
            restored = self._restore_pre_setup_snapshot()
            self._reset_verification_state()
            self._final_comparison_pending = False
            self.start_button.setText(_APPLY_VERIFY_AGAIN_LABEL)
            self.setup_state = "completed"
            self.compare_button.setEnabled(True)
            self.retake_btn.setVisible(True)
            self.phase_label.setText(
                "Second recording failed; previous settings put back"
                if restored
                else "Second recording failed; previous settings not fully put back"
            )
            if not restored:
                self.warning_label.setText(
                    f"{error} Your previous settings could not be fully put back; "
                    "press Retake to try again."
                )
        else:
            if noise_recording_failed:
                self._capture_context_key = None
                self.noise_audio = None
                self.preview_noise_audio = None
            elif self.setup_state == "voice_recording":
                self.voice_audio = None
                self.preview_voice_audio = None
            self.start_button.setText(
                "Retake Room Noise"
                if noise_recording_failed
                else "Start Voice Setup" if self.noise_audio is None else "Record Voice"
            )
            self._failed_step = (1 if self.setup_state == "level_check" else 2) if self.noise_audio is None else None
            self.setup_state = "idle" if self.noise_audio is None else "noise_ready"
            self.dynamics_group.setEnabled(True)
            self.target_lufs_spin.setEnabled(True)
        self._cleanup_recording_tap()

    def _on_retake_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Discard Setup?",
            "Discard the current setup captures and start over?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_setup_ui()

    def _on_cancel_clicked(self) -> None:
        if self.setup_state in {"noise_recording", "voice_recording"}:
            reply = QMessageBox.question(
                self,
                "Cancel Recording?",
                "Discard the current capture and return to the main window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.reject()

    def _reset_setup_ui(self) -> None:
        self.compare_button.setEnabled(False)
        if not self._restore_pre_setup_snapshot():
            self.phase_label.setText("Reset blocked; previous settings not fully put back")
            self.warning_label.setText(
                "Your previous settings could not be fully put back. Press Retake to try again."
            )
            self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
            return
        self._final_comparison_pending = False
        self._capture_start_timer.stop()
        self.recording_timer.stop()
        self._cancel_analysis_workers()
        self._cleanup_recording_tap()
        self._stop_owned_processor()

        self._failed_step = None
        self._reported = False
        self.setup_state = "idle"
        self.noise_audio = None
        self.voice_audio = None
        self.preview_noise_audio = None
        self.preview_voice_audio = None
        self.preview_verification_audio = None
        self.noise_metadata = None
        self.voice_metadata = None
        self._current_capture_metadata = None
        self.setup_result = None
        self._reset_verification_state()
        self._candidate_metadata = None
        self._capture_context_key = None
        self.curve_group.setEnabled(True)
        self.dynamics_group.setEnabled(True)
        self.target_lufs_spin.setEnabled(True)
        self.progress_bar.setValue(0)
        self.level_meter.set_unavailable()
        self._advisory_only = False
        self._level_verdict = self._level_override = None
        self.level_override_btn.setVisible(False)
        self.phase_label.setText("Ready to start setup")
        self.time_label.setText(f"Time remaining: {NOISE_RECORDING_DURATION:.0f}s")
        self.time_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.warning_label.setText("Ready to record")
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.start_button.setText("Start Voice Setup")
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        self.retake_btn.setVisible(False)
        self.summary_group.setVisible(False)

    def _stop_owned_processor(self) -> None:
        if not self._started_processor:
            return

        parent = _find_processor_owner(self.parent())
        if parent:
            try:
                parent.processor.stop()
            except Exception as exc:
                logger.warning("Failed to stop owned processor: %s", exc)
        self._started_processor = False

    def _cleanup_recording_tap(self) -> None:
        session, self._capture_session = self._capture_session, None
        if session is not None:
            session.cleanup()

    def _cancel_analysis_workers(self) -> None:
        """Cancel work without dropping ownership of a running QThread."""
        verification_was_active = self.setup_state == "verification_analyzing"
        self._analysis_generation += 1
        for worker in tuple(self._analysis_workers):
            if worker.isRunning():
                worker.stop()
        self.analysis_worker = None
        if verification_was_active:
            restored = self._restore_pre_setup_snapshot()
            self._final_comparison_pending = False
            self.setup_state = "completed" if self.setup_result is not None else "noise_ready"
            self.compare_button.setEnabled(self.setup_result is not None)
            self.start_button.setEnabled(True)
            self.start_button.setText(
                _APPLY_VERIFY_AGAIN_LABEL
                if self.setup_result is not None
                else "Record Voice"
            )
            self.curve_combo.setEnabled(True)
            self.phase_label.setText(
                "Check canceled; previous settings put back"
                if restored
                else "Check canceled; previous settings not fully put back"
            )

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
        self.noise_audio = None
        self.voice_audio = None
        self.preview_noise_audio = None
        self.preview_voice_audio = None
        self.preview_verification_audio = None
        QDialog.done(self, int(QDialog.DialogCode.Accepted if self._close_result else QDialog.DialogCode.Rejected))

    # A real method, so Qt drops the aboutToQuit connection when the dialog is destroyed.
    def _wait_for_analysis_workers(self) -> None:
        wait_for_analysis_workers(self)

    def _request_close(self, accepted: bool) -> None:
        if self._close_requested:
            return
        self.setup_state = "idle"
        self._capture_start_timer.stop()
        self.recording_timer.stop()
        self._cancel_analysis_workers()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        if not self._restore_pre_setup_snapshot():
            self.phase_label.setText("Close blocked; previous settings not fully put back")
            self.warning_label.setText(
                "Your previous settings could not be fully put back. Press Retake to try again."
            )
            self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
            return
        self._final_comparison_pending = False
        if not accepted and self.setup_result is not None and not self._reported:
            self._report(
                "cancelled",
                ["closed_before_finishing"],
                ["Setup was closed before the new settings were kept."],
            )
        self._close_requested = True
        self._close_result = accepted
        if any(worker.isRunning() for worker in self._analysis_workers):
            self.start_button.setEnabled(False)
            self.retake_btn.setEnabled(False)
            self.warning_label.setText("Canceling analysis...")
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))
            return
        self._finish_close()

    def get_selected_curve(self) -> str:
        return str(self.curve_combo.currentData() or "broadcast")

    def closeEvent(self, event) -> None:
        if session_is_ending():
            # A veto here would stop Windows from signing out: stop the work, join its threads, close.
            self._wait_for_analysis_workers()
            self._finish_close()
            event.accept()
            return
        self._request_close(False)
        event.ignore()

    def accept(self) -> None:
        self._request_close(True)

    def reject(self) -> None:
        """Never leave a temporary candidate active when the dialog closes."""
        self._request_close(False)
