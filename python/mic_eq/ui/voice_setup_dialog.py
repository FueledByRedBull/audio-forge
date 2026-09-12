"""Wizard dialog for Auto Voice Setup."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import logging
import threading
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
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
    QTextEdit,
    QVBoxLayout,
)

from ..analysis.noise_reference import CaptureMetadata, analyze_noise_reference
from ..analysis.cancellation import AnalysisCancelled
from ..analysis.voice_setup import (
    _normalise_limiter_settings,
    analyze_voice_setup,
    validate_voice_setup_verification,
)
from ..config import EQ_FREQUENCIES, TARGET_CURVES
from .accessibility import bind_label, set_accessible_group
from .calibration_dialog import (
    RAINBOW_PASSAGE,
    TOO_LOUD_DB,
    TOO_QUIET_DB,
    _candidate_metadata,
    _chain_settings,
    _filtered_capture_for_analysis,
    _active_device_identities,
    _owner_calibration_context_key,
    _restart_processor_for_route,
    _route_identities_match,
    _selected_device_identities,
    _set_temporary_mute,
    _sync_owner_processing_controls,
    _device_label,
    _device_name,
    _diagnostic_state,
    _find_eq_panel_owner,
    _find_processor_owner,
    _format_db,
    _format_percent,
    _processor_sample_rate,
    _start_selected_route,
)
from .layout_constants import (
    SUBDUED_TEXT_STYLE,
    configure_resizable_dialog,
    configure_responsive_combo,
    create_scrollable_dialog_body,
    fit_spinbox_to_contents,
    status_chip_style,
)
from .level_meter import LevelMeter
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


def _suppressor_settings(owner: Any) -> dict[str, Any] | None:
    if not all(hasattr(owner, name) for name in ("rnnoise_checkbox", "strength_slider", "model_combo")):
        return None
    return {"enabled": owner.rnnoise_checkbox.isChecked(),
            "strength": owner.strength_slider.value() / 100.0,
            "model": owner.model_combo.currentData() or "rnnoise"}


def _apply_suppressor_settings(owner: Any, settings: Mapping[str, Any]) -> None:
    current = _suppressor_settings(owner)
    if current is None:
        raise ValueError("Suppression controls are unavailable")
    strength = settings.get("strength")
    if not isinstance(settings.get("enabled"), bool) or isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
        raise ValueError("Invalid suppression settings")
    if settings.get("model") != current["model"]:
        parent_model = settings.get("model")
        if not isinstance(parent_model, str) or not callable(
            getattr(owner, "_apply_noise_model", None)
        ):
            raise ValueError("Invalid suppression model")
        owner._apply_noise_model(parent_model)
    owner.strength_slider.setValue(round(strength * 100))
    owner.rnnoise_checkbox.setChecked(settings["enabled"])


def _candidate_eq_settings_error(eq_settings: Any) -> str | None:
    if not isinstance(eq_settings, Mapping):
        return "candidate EQ settings are missing"
    if any(
        key not in eq_settings
        or not isinstance(eq_settings[key], (list, tuple))
        or len(eq_settings[key]) != len(EQ_FREQUENCIES)
        for key in ("band_freqs", "band_gains", "band_qs")
    ):
        return "candidate EQ bands are incomplete"
    return None


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
    return _candidate_eq_settings_error(setup_result.get("eq_settings"))


class VoiceSetupWorker(QThread):
    """Background worker for multi-stage voice setup analysis."""

    step_progress = pyqtSignal(str, int)
    result_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

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
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request cooperative cancellation."""
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            if self._should_stop():
                return
            self.step_progress.emit("Analyzing room noise and speech...", 20)
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
                noise_model=self.noise_model,
                suppressor_strength=self.suppressor_strength,
                incumbent_settings=self.incumbent_settings,
            )
            if self._should_stop():
                return
            self.step_progress.emit("Finalizing recommendations...", 95)
            self.result_ready.emit(result)
        except AnalysisCancelled:
            return
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceSetupVerificationWorker(QThread):
    """Run second-passage downstream DSP verification off the UI thread."""

    step_progress = pyqtSignal(str, int)
    result_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        noise_audio: np.ndarray,
        original_speech_audio: np.ndarray,
        verification_speech_audio: np.ndarray,
        sample_rate: int,
        setup_result: dict[str, Any],
        target_preset: str,
    ) -> None:
        super().__init__()
        self.noise_audio = noise_audio
        self.original_speech_audio = original_speech_audio
        self.verification_speech_audio = verification_speech_audio
        self.sample_rate = sample_rate
        self.setup_result = deepcopy(setup_result)
        self.target_preset = target_preset
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request cooperative cancellation."""
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            if self._should_stop():
                return
            self.step_progress.emit("Validating downstream DSP stages...", 55)
            result = validate_voice_setup_verification(
                self.noise_audio,
                self.original_speech_audio,
                self.verification_speech_audio,
                self.sample_rate,
                self.setup_result,
                self.target_preset,
                cancel_check=self._should_stop,
            )
            if self._should_stop():
                return
            self.step_progress.emit("Finalizing verification...", 95)
            self.result_ready.emit(result)
        except AnalysisCancelled:
            return
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceSetupDialog(QDialog):
    """Record room tone and speech, then recommend a voice chain."""

    setup_applied = pyqtSignal(str)

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
        self.setup_result: dict[str, Any] | None = None
        self._candidate_metadata: dict[str, Any] | None = None
        self._capture_context_key: str | None = None
        self.analysis_worker: VoiceSetupWorker | VoiceSetupVerificationWorker | None = None
        self._analysis_workers: list[
            VoiceSetupWorker | VoiceSetupVerificationWorker
        ] = []
        self._analysis_generation = 0
        self._close_requested = False
        self._close_result = False
        self._started_processor = False
        self._recording_duration = NOISE_RECORDING_DURATION

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
        panel = getattr(self.parent(), "compressor_panel", None)
        getter = getattr(panel, "get_compressor_settings", None)
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
        self.content_scroll_area, layout = create_scrollable_dialog_body(self)
        self.content_scroll_area.setAccessibleName("Auto Voice Setup content")
        outer_layout.addWidget(self.content_scroll_area)

        self.curve_group = QGroupBox("Step 1: Select Target Curve")
        curve_layout = QVBoxLayout(self.curve_group)

        curve_input = QHBoxLayout()
        curve_label = QLabel("Target Curve:")
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

        self.dynamics_group = QGroupBox("Step 2: Select Dynamics Intensity")
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
        self.custom_p95_spin.setSuffix(" dB p95")
        self.custom_p95_spin.setValue(
            float(getattr(config, "voice_setup_custom_p95_db", 3.5))
        )
        fit_spinbox_to_contents(self.custom_p95_spin)
        custom_p95_label = QLabel("Target p95 reduction:")
        dynamics_row.addRow(custom_p95_label, self.custom_p95_spin)
        self.custom_peak_spin = QDoubleSpinBox()
        self.custom_peak_spin.setRange(1.5, 12.0)
        self.custom_peak_spin.setSuffix(" dB peak cap")
        self.custom_peak_spin.setValue(
            float(getattr(config, "voice_setup_custom_peak_cap_db", 8.0))
        )
        fit_spinbox_to_contents(self.custom_peak_spin)
        custom_peak_label = QLabel("Peak reduction cap:")
        dynamics_row.addRow(custom_peak_label, self.custom_peak_spin)
        bind_label(
            custom_p95_label,
            self.custom_p95_spin,
            name="Target compressor p95 gain reduction",
        )
        bind_label(
            custom_peak_label,
            self.custom_peak_spin,
            name="Compressor peak gain-reduction cap",
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
            "Compression intensity controls density. Tone and target loudness "
            "are independent."
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

        noise_group = QGroupBox("Step 3: Capture Room Noise")
        noise_layout = QVBoxLayout(noise_group)
        noise_hint = QLabel(
            "Stay quiet for 2 seconds so the wizard can measure room noise "
            "and set the gate safely."
        )
        noise_hint.setWordWrap(True)
        noise_layout.addWidget(noise_hint)
        layout.addWidget(noise_group)

        voice_group = QGroupBox("Step 4: Read Passage Aloud")
        voice_layout = QVBoxLayout(voice_group)
        passage_text = QTextEdit()
        passage_text.setPlainText(RAINBOW_PASSAGE)
        passage_text.setReadOnly(True)
        passage_text.setMaximumHeight(130)
        passage_text.setAccessibleName("Voice Setup passage")
        passage_text.setAccessibleDescription(
            "Read-only passage to speak during Auto Voice Setup."
        )
        voice_layout.addWidget(passage_text)
        layout.addWidget(voice_group)

        self.recording_group = QGroupBox("Step 5: Record, Analyze, And Verify")
        recording_layout = QVBoxLayout(self.recording_group)
        self.recording_group.setVisible(False)

        self.phase_label = QLabel("Ready to start setup")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setStyleSheet(PROGRESS_LABEL_STYLE)
        self.phase_label.setAccessibleName("Voice Setup phase")
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
        self.time_label.setAccessibleName("Voice Setup time remaining")
        recording_layout.addWidget(self.time_label)

        meter_layout = QHBoxLayout()
        self.level_meter = LevelMeter(label="Level", show_scale=True)
        self.level_meter.setMinimumHeight(120)
        meter_layout.addWidget(self.level_meter)
        recording_layout.addLayout(meter_layout)

        self.warning_label = QLabel("Ready to record")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet(message_text_style("idle"))
        self.warning_label.setAccessibleName("Voice Setup status")
        self.warning_label.setWordWrap(True)
        recording_layout.addWidget(self.warning_label)

        self.summary_group = QGroupBox("Recommended Settings")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_group.setVisible(False)
        self.overall_label = QLabel("Overall: --")
        self.eq_label = QLabel("EQ: --")
        self.gate_label = QLabel("Gate/VAD: --")
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
            "This wizard tunes EQ, gate/VAD, de-esser, and compressor. "
            "Limiter settings are left unchanged."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(SUBDUED_TEXT_STYLE)
        summary_layout.addWidget(hint)
        recording_layout.addWidget(self.summary_group)

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
        controls.addWidget(self.cancel_btn)
        recording_layout.addLayout(controls)
        layout.addWidget(self.recording_group)

        self.start_button = QPushButton("Start Voice Setup")
        self.start_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.start_button.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_button)

        set_accessible_group(
            (
                (self.start_button, "Start Auto Voice Setup", None),
                (self.retake_btn, "Retake Voice Setup recording", None),
                (self.cancel_btn, "Cancel Auto Voice Setup", None),
                (self.level_meter, "Voice Setup input level", None),
            )
        )
        self.setTabOrder(self.curve_combo, self.dynamics_combo)
        self.setTabOrder(self.dynamics_combo, self.custom_p95_spin)
        self.setTabOrder(self.custom_p95_spin, self.custom_peak_spin)
        self.setTabOrder(self.custom_peak_spin, self.target_lufs_spin)
        self.setTabOrder(self.target_lufs_spin, self.start_button)
        self.setTabOrder(self.start_button, self.retake_btn)
        self.setTabOrder(self.retake_btn, self.cancel_btn)

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
            self._start_recording_phase("noise_recording")
        elif self.setup_state == "noise_ready":
            self._start_recording_phase("voice_recording")
        elif self.setup_state == "completed" and self.setup_result is not None:
            self._apply_setup()
        elif self.setup_state == "verification_ready":
            self._start_recording_phase("verification_recording")
        elif self.setup_state in {"noise_recording", "voice_recording"}:
            QMessageBox.information(
                self, "Recording", "Please let the recording finish."
            )

    def _start_recording_phase(self, phase: str) -> None:
        self.compare_button.setEnabled(False)
        self._cancel_analysis_workers()
        if not self._ensure_processor_ready():
            return

        self.setup_state = phase
        self.curve_combo.setEnabled(False)
        self.dynamics_group.setEnabled(False)
        self.target_lufs_spin.setEnabled(False)
        self.start_button.setEnabled(False)
        self.retake_btn.setVisible(False)
        self.summary_group.setVisible(False)

        if phase == "noise_recording":
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
                "Read naturally again. The raw passage will be rendered through "
                "the proposed EQ, de-esser, compressor, and limiter stages and "
                "compared with the first capture. Gate, suppression, and input "
                "cleanup, plus live loudness adaptation, are excluded from this "
                "offline check."
            )
            self.warning_label.setStyleSheet(message_text_style("info", strong=True))

        self.progress_bar.setValue(0)
        self.time_label.setText(f"Time remaining: {self._recording_duration:.0f}s")
        self._capture_start_timer.start(100)

    def _ensure_processor_ready(self) -> bool:
        parent = _find_processor_owner(self.parent())
        if not parent:
            QMessageBox.critical(self, "Error", "Could not find audio processor")
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
                        "Setup canceled: using current stream devices."
                    )
                    self.warning_label.setStyleSheet(message_text_style("warn"))
                    self.start_button.setEnabled(True)
                    self.curve_combo.setEnabled(True)
                    self.setup_state = (
                        "idle" if self.noise_audio is None else "noise_ready"
                    )
                    self.start_button.setText(
                        "Start Voice Setup"
                        if self.noise_audio is None
                        else "Record Voice"
                    )
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
                        f"Failed to switch audio devices for setup:\n{exc}",
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
                f"Failed to start audio processing:\n{exc}\n\n"
                "Check that audio devices are connected and not already in use.",
            )
            return False

    def _begin_recording_capture(self) -> None:
        self._capture_start_timer.stop()
        if self.setup_state not in {
            "noise_recording",
            "voice_recording",
            "verification_recording",
        }:
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        context_key = _owner_calibration_context_key(parent)
        if self.setup_state == "noise_recording":
            self._capture_context_key = context_key
        elif context_key != self._capture_context_key:
            self._on_recording_failed(
                "Audio route or input cleanup context changed during setup."
            )
            return

        try:
            _set_temporary_mute(parent, _VOICE_SETUP_MUTE_REASON, True)
            get_input = getattr(parent.processor, "get_active_input_device", None)
            get_mode = getattr(parent.processor, "get_input_channel_mode", None)
            self._current_capture_metadata = CaptureMetadata(
                captured_at_unix_s=time.time(),
                input_device=str(get_input() or "") if callable(get_input) else "",
                sample_rate=_processor_sample_rate(parent),
                channel_mode=str(get_mode() or "") if callable(get_mode) else "",
                channel_count=1,
            )
            parent.processor.set_recovery_suppressed(True)
            parent.processor.start_raw_recording(
                self._recording_duration,
                before_cleanup=True,
            )
        except Exception as exc:
            self._on_recording_failed(f"Recording error: {exc}")
            return

        self.recording_timer.start()

    def _poll_recording_progress(self) -> None:
        if self.setup_state not in {
            "noise_recording",
            "voice_recording",
            "verification_recording",
        }:
            self.recording_timer.stop()
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_recording_failed("Could not find audio processor")
            return

        try:
            progress = float(parent.processor.recording_progress())
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
                audio = parent.processor.stop_raw_recording()
                _set_temporary_mute(parent, _VOICE_SETUP_MUTE_REASON, False)
                if audio is None:
                    self._on_recording_failed("Recording failed - no audio data")
                    return
                self._on_recording_complete(np.asarray(audio, dtype=np.float32))
        except Exception as exc:
            self.recording_timer.stop()
            self._on_recording_failed(f"Recording error: {exc}")

    def _update_time_remaining(self, seconds: float) -> None:
        if seconds > 0:
            self.time_label.setText(f"Time remaining: {seconds:.0f}s")
        else:
            self.time_label.setText("Complete!")
            self.time_label.setStyleSheet(message_text_style("ok", strong=True))

    def _update_level(self, rms_db: float, peak_db: float) -> None:
        self.level_meter.set_levels(rms_db, peak_db)

        if self.setup_state == "noise_recording":
            if rms_db > -35.0:
                self.warning_label.setText(
                    "Room noise is fairly loud. Results may be conservative."
                )
                self.warning_label.setStyleSheet(
                    message_text_style("warn", strong=True)
                )
            else:
                self.warning_label.setText("Noise floor capture looks usable.")
                self.warning_label.setStyleSheet(message_text_style("ok", strong=True))
            return

        if rms_db < TOO_QUIET_DB:
            self.warning_label.setText("Too quiet. Move closer to the mic.")
            self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        elif peak_db > TOO_LOUD_DB:
            self.warning_label.setText("Too loud. Back off slightly to avoid clipping.")
            self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        else:
            self.warning_label.setText("Voice level looks good.")
            self.warning_label.setStyleSheet(message_text_style("ok", strong=True))

    def _on_recording_complete(self, audio_data: np.ndarray) -> None:
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
        analysis_audio = _filtered_capture_for_analysis(raw_audio, sample_rate)

        if self.setup_state == "noise_recording":
            self.preview_noise_audio = raw_audio
            self.noise_audio = analysis_audio
            self.noise_metadata = self._current_capture_metadata
            quick_quality = analyze_noise_reference(
                self.noise_audio,
                None,
                int(self.noise_metadata.sample_rate or 48_000)
                if self.noise_metadata is not None
                else 48_000,
                noise_metadata=self.noise_metadata,
            )
            if quick_quality.status == "invalid":
                self.noise_audio = None
                self.preview_noise_audio = None
                self.noise_metadata = None
                self.setup_state = "idle"
                self.start_button.setText("Retake Room Noise")
                self.phase_label.setText("Room-noise capture rejected")
                guidance = quick_quality.guidance or (
                    "Keep quiet and record steady room tone again.",
                )
                self.warning_label.setText(" ".join(guidance))
                self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
                return
            self.setup_state = "noise_ready"
            self.start_button.setText("Record Voice")
            self.phase_label.setText("Room noise captured")
            self.warning_label.setText(
                "Now read the passage aloud for the full 10 seconds."
            )
            self.warning_label.setStyleSheet(message_text_style("ok", strong=True))
            self.progress_bar.setValue(0)
            self.time_label.setText(f"Time remaining: {VOICE_RECORDING_DURATION:.0f}s")
            return

        if self.setup_state == "verification_recording":
            self.preview_verification_audio = raw_audio
            self._complete_verification(analysis_audio)
            return

        self.preview_voice_audio = raw_audio
        self.voice_audio = analysis_audio
        self.voice_metadata = self._current_capture_metadata
        self.setup_state = "analyzing"
        self.start_button.setEnabled(False)
        self.start_button.setText("Analyzing...")
        self.phase_label.setText("Analyzing recordings")
        self.warning_label.setText("Building recommendations for your voice chain.")
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))
        self._start_analysis()

    def _start_analysis(self) -> None:
        if self.noise_audio is None or self.voice_audio is None:
            self._on_analysis_failed("Missing room-noise or speech capture.")
            return

        parent = _find_processor_owner(self.parent())
        if not parent:
            self._on_analysis_failed("Could not find audio processor")
            return

        vad_available = False
        try:
            vad_available = bool(parent.processor.is_vad_available())
        except Exception:
            vad_available = False

        limiter_settings: Mapping[str, Any] | None = None
        target_lufs = self.target_lufs_spin.value()
        compressor_panel = getattr(parent, "compressor_panel", None)
        get_limiter_settings = getattr(compressor_panel, "get_limiter_settings", None)
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
        if hasattr(parent, "eq_panel"):
            incumbent["eq"] = deepcopy(parent.eq_panel.get_settings())
        if hasattr(parent, "gate_panel"):
            incumbent["gate"] = deepcopy(parent.gate_panel.get_settings())
        if suppression is not None:
            incumbent["rnnoise"] = suppression
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
            lambda error, token=generation: self._on_analysis_failed(error, token)
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
        candidate = deepcopy(self._candidate_metadata or {})
        if isinstance(candidate, dict):
            candidate["verified_stages"] = []
        setup_result["_candidate"] = candidate
        self.setup_result = setup_result
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
        self.phase_label.setText(
            "Recommendations ready" if candidate_complete else "Recommendations incomplete"
        )
        diagnostics = setup_result["diagnostics"]
        if not candidate_complete:
            self.warning_label.setText(
                "Recommendations are incomplete: "
                f"{candidate_error}. Record the voice passage again to retry."
            )
            state = "warn"
        elif diagnostics.get("apply_recommended", False):
            self.warning_label.setText("Review the validated settings and apply them.")
            state = "ok"
        else:
            reasons = diagnostics.get("uncertainty_reasons") or [
                "capture confidence is weak"
            ]
            self.warning_label.setText(
                "Advisory recommendations only: "
                + "; ".join(str(reason) for reason in reasons)
            )
            state = "warn"
        self.warning_label.setStyleSheet(message_text_style(state, strong=True))
        self.progress_bar.setValue(100)
        self._show_summary(setup_result)

    def _show_summary(self, setup_result: dict[str, Any]) -> None:
        diagnostics = setup_result["diagnostics"]
        overall_conf = float(diagnostics["setup_confidence"])
        state = _diagnostic_state(overall_conf)
        self.overall_label.setText(
            "Overall: "
            f"{_format_percent(overall_conf)} | "
            f"capture {_format_percent(diagnostics['capture_confidence'])} | "
            f"uncertainty {_format_percent(diagnostics['recommendation_uncertainty'])}"
        )
        self.overall_label.setStyleSheet(status_chip_style(state))
        joint = diagnostics.get("joint_tuning")
        if isinstance(joint, Mapping):
            outcome = "candidate passed held-out checks" if joint.get("apply_recommended") else "current settings retained"
            self.overall_label.setText(self.overall_label.text() + f"\nSuppression + gate: {outcome}.")
            self.overall_label.setToolTip("; ".join(str(reason) for reason in joint.get("reasons", [])))

        eq_settings = setup_result.get("eq_settings")
        eq_error = _candidate_eq_settings_error(eq_settings)
        if eq_error is None:
            assert isinstance(eq_settings, Mapping)
            self.eq_label.setText(
                "EQ: "
                f"{_format_percent(eq_settings.get('analysis_confidence', 0.0))} | "
                f"max correction {max(abs(g) for g in eq_settings['band_gains']):.1f} dB"
            )
            self.eq_label.setStyleSheet(status_chip_style("ok"))
        else:
            self.eq_label.setText(
                f"EQ: skipped | {setup_result.get('eq_error') or eq_error}"
            )
            self.eq_label.setStyleSheet(status_chip_style("warn"))

        gate = setup_result["gate_settings"]
        self.gate_label.setText(
            "Gate/VAD: "
            f"{diagnostics['gate_mode_label']} | "
            f"threshold {_format_db(gate['threshold_db'])} | "
            f"VAD {gate['vad_threshold']:.2f}"
        )
        self.gate_label.setStyleSheet(status_chip_style("info"))

        deesser = setup_result["deesser_settings"]
        deesser_state = "ok" if deesser["enabled"] else "info"
        deesser_text = "enabled" if deesser["enabled"] else "left off"
        self.deesser_label.setText(
            "De-esser: "
            f"{deesser_text} | auto {deesser['auto_amount'] * 100.0:.0f}% | "
            f"{deesser['low_cut_hz']:.0f}-{deesser['high_cut_hz']:.0f} Hz"
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
        self.compressor_label.setText(
            "Compressor: "
            f"{compressor['ratio']:.1f}:1 @ {_format_db(compressor['threshold_db'])} | "
            f"{makeup_text} | target {compressor['target_lufs']:.0f} LUFS | "
            f"{intensity} | gain reduction {_format_db(measured_gr)}"
        )
        self.compressor_label.setStyleSheet(status_chip_style("info"))
        self.summary_group.setVisible(True)

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
        if capture.get("context_key") != _owner_calibration_context_key(parent):
            return "audio route or input cleanup context changed after capture"
        return None

    def _compare_recording(self) -> None:
        if self.setup_state != "completed" or self.voice_audio is None or self.setup_result is None:
            return
        owner = _find_eq_panel_owner(self.parent())
        if owner is None:
            return
        error = self._candidate_identity_error(owner)
        if error:
            QMessageBox.warning(self, "Stale recording", error)
            return
        from .listening_comparison_dialog import ListeningComparisonDialog

        try:
            current_chain = _chain_settings(
                owner,
                full_chain=True,
                input_pre_filtered=self.preview_voice_audio is None,
            )
        except Exception as error:
            QMessageBox.warning(self, "Comparison unavailable", str(error))
            return
        proposed_chain = deepcopy(current_chain)
        proposed_chain.update(
            {
                name: deepcopy(self.setup_result[f"{name}_settings"])
                for name in ("deesser", "compressor", "limiter")
            }
        )
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
            current_settings=owner.eq_panel.get_settings(),
            proposed_settings=self.setup_result["eq_settings"],
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
            self._apply_setup()

    def _apply_setup(self) -> None:
        parent = _find_eq_panel_owner(self.parent())
        if not parent or self.setup_result is None:
            QMessageBox.critical(self, "Error", "Could not apply voice setup.")
            return

        candidate_error = _candidate_settings_error(self.setup_result)
        if candidate_error is not None:
            QMessageBox.critical(
                self,
                "Incomplete Voice Setup",
                "The voice setup recommendations are incomplete ("
                f"{candidate_error}). Record the voice passage again; no changes were applied.",
            )
            return

        identity_error = self._candidate_identity_error(parent)
        if identity_error is not None:
            QMessageBox.critical(
                self,
                "Stale Voice Setup Candidate",
                f"{identity_error}; no changes were applied.",
            )
            return

        diagnostics = self.setup_result.get("diagnostics") or {}
        if not diagnostics.get("apply_recommended", False):
            reasons = diagnostics.get("uncertainty_reasons") or [
                "capture confidence is weak"
            ]
            reply = QMessageBox.question(
                self,
                "Apply Advisory Settings?",
                "These settings did not reach validated confidence:\n\n"
                + "\n".join(f"- {reason}" for reason in reasons)
                + "\n\nApply them anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if self._pre_setup_snapshot is None:
            limiter_settings = self.setup_result.get("limiter_settings")
            if not isinstance(limiter_settings, Mapping):
                QMessageBox.critical(
                    self,
                    "Error",
                    "Voice setup did not include limiter settings; no changes were applied.",
                )
                return
            self._pre_setup_snapshot = deepcopy(
                {
                    "gate": parent.gate_panel.get_settings(),
                    "deesser": parent.deesser_panel.get_settings(),
                    "compressor": parent.compressor_panel.get_compressor_settings(
                        include_calibration=True
                    ),
                    "limiter": parent.compressor_panel.get_limiter_settings(),
                    "eq": parent.eq_panel.get_settings(),
                    "suppression": _suppressor_settings(parent),
                    "calibration_context_key": _owner_calibration_context_key(parent),
                }
            )
        try:
            self._apply_candidate_panels(parent)
            self._sync_setup_result_with_applied_panels(parent)
        except Exception as exc:
            restored = self._restore_pre_setup_snapshot()
            message = (
                f"Could not apply candidate settings; changes were restored: {exc}"
                if restored
                else f"Could not apply candidate settings; restore is incomplete: {exc}"
            )
            QMessageBox.critical(self, "Error", message)
            return
        self.setup_state = "verification_ready"
        self.compare_button.setEnabled(False)
        self.start_button.setText("Record Verification Passage")
        self.phase_label.setText("Candidate applied temporarily")
        self.warning_label.setText(
            "Read the passage once more to accept, reduce, retry, or roll back "
            "using downstream EQ, de-esser, compressor, and limiter measurements. "
            "Gate, suppression, input cleanup, and live loudness adaptation are "
            "excluded."
        )
        self.warning_label.setStyleSheet(message_text_style("info", strong=True))

    def _apply_candidate_panels(self, parent: Any) -> None:
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
        limiter_settings = self.setup_result.get("limiter_settings")
        eq_settings = self.setup_result.get("eq_settings")
        if not isinstance(limiter_settings, Mapping) or not isinstance(
            eq_settings, Mapping
        ):
            raise ValueError("candidate settings are incomplete")
        bands = list(
            zip(
                eq_settings["band_freqs"],
                eq_settings["band_gains"],
                eq_settings["band_qs"],
            )
        )
        parent.gate_panel.set_settings(self.setup_result["gate_settings"])
        suppression = self.setup_result.get("suppressor_settings")
        if suppression is not None:
            _apply_suppressor_settings(parent, suppression)
        parent.deesser_panel.set_settings(self.setup_result["deesser_settings"])
        parent.compressor_panel.set_compressor_settings(
            self.setup_result["compressor_settings"]
        )
        parent.compressor_panel.set_limiter_settings(dict(limiter_settings))
        if "bands" in eq_settings:
            parent.eq_panel.set_settings(dict(eq_settings))
        else:
            parent.eq_panel.set_settings({"enabled": True})
            parent.eq_panel.apply_auto_eq_results(bands, diagnostics=eq_settings)
        if hasattr(parent, "status_bar"):
            parent.status_bar.showMessage(
                "Voice setup candidate applied; verification required", 5000
            )

    def _sync_setup_result_with_applied_panels(self, parent: Any) -> None:
        """Use the values the controls accepted for the verification pass."""
        if self.setup_result is None:
            raise ValueError("voice setup result is missing")

        if self.setup_result.get("suppressor_settings") is not None:
            self.setup_result["suppressor_settings"] = _suppressor_settings(parent)
        compressor = dict(self.setup_result.get("compressor_settings") or {})
        compressor.update(
            parent.compressor_panel.get_compressor_settings(include_calibration=True)
        )
        self.setup_result["compressor_settings"] = compressor

        deesser = dict(self.setup_result.get("deesser_settings") or {})
        deesser.update(parent.deesser_panel.get_settings())
        self.setup_result["deesser_settings"] = deesser

        self.setup_result["limiter_settings"] = dict(
            parent.compressor_panel.get_limiter_settings()
        )

        eq_settings = self.setup_result.get("eq_settings")
        if eq_settings is not None:
            applied_eq = parent.eq_panel.get_settings()
            eq_settings = dict(eq_settings)
            for key in ("schema_version", "bands", "layers"):
                if key in applied_eq:
                    eq_settings[key] = deepcopy(applied_eq[key])
            for key in ("enabled", "band_freqs", "band_gains", "band_qs"):
                if key not in applied_eq:
                    raise ValueError(f"EQ panel did not report {key}")
                value = applied_eq[key]
                eq_settings[key] = list(value) if key != "enabled" else bool(value)
            self.setup_result["eq_settings"] = eq_settings

    def _restore_pre_setup_snapshot(self) -> bool:
        parent = _find_eq_panel_owner(self.parent())
        if self._pre_setup_snapshot is None:
            return True
        if parent is None:
            logger.warning("Voice setup rollback owner disappeared")
            return False
        snapshot = self._pre_setup_snapshot
        errors: list[str] = []
        compressor_snapshot = deepcopy(snapshot["compressor"])
        snapshot_context = snapshot.get("calibration_context_key")
        current_context = _owner_calibration_context_key(parent)
        if not snapshot_context or not current_context or snapshot_context != current_context:
            compressor_snapshot.pop("noise_reference_reliability", None)
        for label, restore in (
            ("gate", lambda: parent.gate_panel.set_settings(snapshot["gate"])),
            ("de-esser", lambda: parent.deesser_panel.set_settings(snapshot["deesser"])),
            (
                "compressor",
                lambda: parent.compressor_panel.set_compressor_settings(
                    compressor_snapshot
                ),
            ),
            (
                "limiter",
                lambda: parent.compressor_panel.set_limiter_settings(
                    snapshot["limiter"]
                ),
            ),
            ("EQ", lambda: parent.eq_panel.set_settings(snapshot["eq"])),
        ):
            try:
                restore()
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        if snapshot.get("suppression") is not None:
            try:
                _apply_suppressor_settings(parent, snapshot["suppression"])
            except Exception as exc:
                errors.append(f"suppression: {exc}")
        if errors:
            logger.warning("Voice setup rollback was incomplete: %s", "; ".join(errors))
            return False
        self._pre_setup_snapshot = None
        return True

    def _complete_verification(self, audio_data: np.ndarray) -> None:
        if (
            self.setup_result is None
            or self.noise_audio is None
            or self.voice_audio is None
        ):
            restored = self._restore_pre_setup_snapshot()
            self._on_analysis_failed(
                "Verification context is incomplete; "
                + ("settings restored." if restored else "restore is incomplete.")
            )
            return
        self.phase_label.setText("Validating downstream DSP stages")
        self.warning_label.setText("Comparing the second passage...")
        processor_owner = _find_processor_owner(self.parent())
        if processor_owner is None:
            self._on_verification_failed(
                "Audio processor disappeared during verification."
            )
            return

        self._cancel_analysis_workers()
        generation = self._analysis_generation + 1
        self._analysis_generation = generation
        self.setup_state = "verification_analyzing"
        self.start_button.setEnabled(False)
        self.start_button.setText("Validating...")
        self.curve_combo.setEnabled(False)
        worker = VoiceSetupVerificationWorker(
            self.noise_audio,
            self.voice_audio,
            audio_data,
            _processor_sample_rate(processor_owner),
            self.setup_result,
            str(
                ((self.setup_result or {}).get("_candidate") or {})
                .get("target", {})
                .get("curve", self.get_selected_curve())
            ),
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
                error, token
            )
        )
        worker.finished.connect(
            lambda finished_worker=worker: self._on_analysis_thread_finished(
                finished_worker
            )
        )
        worker.start()

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
        reason = "; ".join(result.get("reasons") or ())
        metrics_text = (
            "Target error "
            f"{float(result.get('spectral_target_error_before_db', 0.0)):.1f}"
            "→"
            f"{float(result.get('spectral_target_error_after_db', 0.0)):.1f} dB; "
            "compressor p95 "
            f"{float(result.get('compressor_gain_reduction_p95_db', 0.0)):.1f} dB; "
            "true peak "
            f"{float(result.get('output_true_peak_db', -120.0)):.1f} dBTP; "
            "SNR change "
            f"{float(result.get('snr_change_db', 0.0)):+.1f} dB."
        )
        parent = _find_eq_panel_owner(self.parent())
        if decision == "accept":
            identity_error = self._candidate_identity_error(parent)
            if identity_error is not None:
                restored = self._restore_pre_setup_snapshot()
                candidate = self.setup_result.get("_candidate")
                if isinstance(candidate, dict):
                    candidate["verified_stages"] = []
                self.setup_state = "completed"
                self.start_button.setEnabled(True)
                self.start_button.setText("Apply & Verify Again")
                self.phase_label.setText(
                    "Verification accepted, but candidate is stale"
                    if restored
                    else "Verification accepted; restore is incomplete"
                )
                self.warning_label.setText(
                    f"{identity_error}; "
                    + (
                        "settings were restored."
                        if restored
                        else "previous settings could not be fully restored; retry rollback."
                    )
                )
                self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
                return
            candidate = self.setup_result.get("_candidate")
            if isinstance(candidate, dict):
                candidate["verified_stages"] = [
                    "eq",
                    "deesser",
                    "compressor",
                    "limiter",
                ]
            if parent is not None and hasattr(parent, "status_bar"):
                parent.status_bar.showMessage("Verified voice setup applied", 5000)
            QMessageBox.information(
                self,
                "Voice Setup Verified",
                "Downstream EQ, de-esser, compressor, and limiter verification "
                "accepted the candidate.\n\n"
                + metrics_text
                + "\n\nGate, suppression, and input cleanup were excluded; "
                "live loudness adaptation was also excluded; "
                "this validates engineering constraints, not listening preference.",
            )
            self._pre_setup_snapshot = None
            target_curve = (
                candidate.get("target", {}).get("curve")
                if isinstance(candidate, Mapping)
                else None
            ) or self.get_selected_curve()
            self.setup_applied.emit(str(target_curve))
            from .calibration_history import persist_calibration

            persist_calibration(self, parent, "full_voice_setup", str(target_curve),
                                ("eq", "deesser", "compressor", "limiter"))
            self.accept()
            return
        if decision == "reduce" and parent is not None:
            compressor = self.setup_result["compressor_settings"]
            compressor["threshold_db"] = min(
                -6.0,
                float(compressor["threshold_db"]) + 2.0,
            )
            compressor["ratio"] = max(1.5, float(compressor["ratio"]) * 0.9)
            deesser = self.setup_result["deesser_settings"]
            deesser["auto_amount"] = max(
                0.0,
                float(deesser["auto_amount"]) * 0.85,
            )
            try:
                self._apply_candidate_panels(parent)
                self._sync_setup_result_with_applied_panels(parent)
            except Exception as exc:
                self._on_verification_failed(
                    f"Could not apply reduced candidate settings: {exc}"
                )
                return
            self.setup_state = "verification_ready"
            self.start_button.setText("Verify Reduced Processing")
        elif decision == "rollback":
            restored = self._restore_pre_setup_snapshot()
            self.setup_state = "completed"
            self.start_button.setText("Apply & Verify Again")
            if not restored:
                reason = f"{reason} Candidate rollback is incomplete."
        else:
            self.setup_state = "verification_ready"
            self.start_button.setText("Retry Verification")
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(decision == "rollback")
        self.phase_label.setText(f"Verification decision: {decision.upper()}")
        self.warning_label.setText(f"{reason} {metrics_text}")
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))

    def _on_verification_failed(
        self, error: str, generation: int | None = None
    ) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        restored = self._restore_pre_setup_snapshot()
        self.setup_state = "completed" if self.setup_result is not None else "noise_ready"
        self.start_button.setEnabled(True)
        self.start_button.setText(
            "Apply & Verify Again"
            if self.setup_result is not None
            else "Record Voice"
        )
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet(message_text_style("warn", strong=True))
        self.phase_label.setText(
            "Verification failed; settings restored"
            if restored
            else "Verification failed; restore is incomplete"
        )
        if not restored:
            self.warning_label.setText(
                f"{error} Previous settings could not be fully restored; retry rollback."
            )
        self.summary_group.setVisible(self.setup_result is not None)

    def _on_analysis_failed(self, error: str, generation: int | None = None) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        if self._close_requested:
            return
        self.voice_audio = None
        self.preview_voice_audio = None
        self.setup_result = None
        self._candidate_metadata = None
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
        self.phase_label.setText("Analysis failed")
        self.summary_group.setVisible(False)

    def _on_recording_failed(self, error: str) -> None:
        verification_failed = self.setup_state == "verification_recording"
        noise_recording_failed = self.setup_state == "noise_recording"
        self.recording_timer.stop()
        self.start_button.setEnabled(True)
        self.curve_combo.setEnabled(True)
        self.warning_label.setText(error)
        self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
        if verification_failed:
            restored = self._restore_pre_setup_snapshot()
            self.start_button.setText("Apply & Verify Again")
            self.setup_state = "completed"
            if not restored:
                self.warning_label.setText(
                    f"{error} Previous settings could not be fully restored; retry rollback."
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
                "Start Voice Setup" if self.noise_audio is None else "Record Voice"
            )
            self.setup_state = "idle" if self.noise_audio is None else "noise_ready"
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
            self.phase_label.setText("Reset blocked; settings restore is incomplete")
            self.warning_label.setText(
                "Previous settings could not be fully restored; retry rollback."
            )
            self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
            return
        self._capture_start_timer.stop()
        self.recording_timer.stop()
        self._cancel_analysis_workers()
        self._cleanup_recording_tap()
        self._stop_owned_processor()

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
        self._candidate_metadata = None
        self._capture_context_key = None
        self.curve_group.setEnabled(True)
        self.dynamics_group.setEnabled(True)
        self.target_lufs_spin.setEnabled(True)
        self.progress_bar.setValue(0)
        self.level_meter.set_levels(-120.0, -120.0)
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
        parent = _find_processor_owner(self.parent())
        if not parent:
            return

        try:
            parent.processor.stop_raw_recording()
        except RuntimeError as exc:
            if "No recording in progress" not in str(exc):
                logger.warning("Failed to stop raw recording during cleanup: %s", exc)
        except Exception as exc:
            logger.warning("Failed to stop raw recording during cleanup: %s", exc)

        try:
            _set_temporary_mute(parent, _VOICE_SETUP_MUTE_REASON, False)
        except Exception as exc:
            logger.warning("Failed to unmute output during cleanup: %s", exc)

        try:
            parent.processor.set_recovery_suppressed(False)
        except Exception as exc:
            logger.warning("Failed to re-enable recovery after cleanup: %s", exc)

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
            self.setup_state = "completed" if self.setup_result is not None else "noise_ready"
            self.start_button.setEnabled(True)
            self.start_button.setText(
                "Apply & Verify Again"
                if self.setup_result is not None
                else "Record Voice"
            )
            self.curve_combo.setEnabled(True)
            self.phase_label.setText(
                "Verification canceled; settings restored"
                if restored
                else "Verification canceled; restore is incomplete"
            )

    def _on_analysis_thread_finished(
        self, worker: VoiceSetupWorker | VoiceSetupVerificationWorker
    ) -> None:
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
        self.noise_audio = None
        self.voice_audio = None
        self.preview_noise_audio = None
        self.preview_voice_audio = None
        self.preview_verification_audio = None
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
        self.setup_state = "idle"
        self._capture_start_timer.stop()
        self.recording_timer.stop()
        self._cancel_analysis_workers()
        self._cleanup_recording_tap()
        self._stop_owned_processor()
        if not self._restore_pre_setup_snapshot():
            self.phase_label.setText("Close blocked; settings restore is incomplete")
            self.warning_label.setText(
                "Previous settings could not be fully restored; retry rollback."
            )
            self.warning_label.setStyleSheet(message_text_style("bad", strong=True))
            return
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
        self._request_close(False)
        event.ignore()

    def accept(self) -> None:
        self._request_close(True)

    def reject(self) -> None:
        """Never leave a temporary candidate active when the dialog closes."""
        self._request_close(False)
