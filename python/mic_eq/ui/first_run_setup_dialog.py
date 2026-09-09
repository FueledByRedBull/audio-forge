"""Resumable first-run setup for selecting a route and running checks."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QComboBox,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..config import coerce_device_identity, save_config
from ..config_parts.app_config import FIRST_RUN_SETUP_STEPS
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    PRIMARY_ACTION_BUTTON_STYLE,
    SECONDARY_ACTION_BUTTON_STYLE,
    SPACING_NORMAL,
    SPACING_SECTION,
    configure_resizable_dialog,
)
from .theme import DESCRIPTION_LABEL_STYLE, message_text_style


DEFAULT_SETUP_STEPS = ("devices", "route", "voice")


STEP_CONTENT = {
    "devices": (
        "1. Choose devices",
        "Select the microphone and destination for your calls, games, or recording. "
        "Choose a virtual cable when another app should receive AudioForge output.",
        "Use Selected Route",
    ),
    "route": (
        "2. Check speech and clipping",
        "Start processing, speak at a normal level, and watch the input/output "
        "levels and clipping indicators. Then check your call, game, or recording "
        "app for the microphone signal and confirm it below. AudioForge cannot "
        "confirm destination-app reception automatically.",
        "Check Speech & Clipping",
    ),
    "latency": (
        "Advanced latency calibration",
        "Measure the selected route when you use a loopback cable or speaker-to-microphone path. "
        "This is optional and is available from the main Options menu as well.",
        "Open Latency Calibration",
    ),
    "voice": (
        "3. Calibrate the voice chain",
        "Run voice setup to tune the chain. After it finishes, check the result in your destination app.",
        "Open Voice Setup",
    ),
}


def route_health_reason(processor: object) -> tuple[bool, str]:
    """Return a conservative live-stream health decision for setup."""
    is_running = getattr(processor, "is_running", None)
    if not callable(is_running) or not bool(is_running()):
        return False, "Processing did not start. Check device availability and retry."

    diagnostics_getter = getattr(processor, "get_runtime_diagnostics", None)
    diagnostics = diagnostics_getter() if callable(diagnostics_getter) else {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    error_fields = (
        "input_callback_error_count",
        "output_callback_error_count",
        "input_stream_error_count",
        "output_stream_error_count",
    )
    try:
        callback_error_present = any(
            int(diagnostics.get(field, 0) or 0) > 0 for field in error_fields
        )
    except (TypeError, ValueError, OverflowError):
        return (
            False,
            "Audio stream health data was invalid. Restart AudioForge and retry.",
        )
    if callback_error_present:
        return (
            False,
            "The audio stream reported an error. Check the selected route and retry.",
        )

    for label, getter_name in (
        ("input", "get_input_callback_age_ms"),
        ("output", "get_output_callback_age_ms"),
    ):
        getter = getattr(processor, getter_name, None)
        if not callable(getter):
            return (
                False,
                f"The {label} stream health check is unavailable. Retry the route.",
            )
        try:
            raw_age = getter()
        except (TypeError, ValueError, OverflowError):
            return (
                False,
                f"The {label} stream health check could not be read. Retry the route.",
            )
        if (
            isinstance(raw_age, bool)
            or not isinstance(raw_age, (int, float))
            or not 0.0 <= float(raw_age) < float("inf")
        ):
            return False, f"The {label} stream health value is invalid. Retry the route."
        age_ms = float(raw_age)
        if age_ms > 2_000.0:
            return (
                False,
                f"The {label} audio stream stopped responding ({age_ms:.0f} ms). Retry the route.",
            )
    return (
        True,
        "Audio stream is healthy. Confirm the microphone signal in your destination app.",
    )


class FirstRunSetupDialog(QDialog):
    """Persisted setup flow for selecting a route and running checks."""

    def __init__(self, owner: Any, *, restart_completed: bool = False):
        super().__init__(owner)
        self.owner = owner
        self.config = owner.config
        self._finalized = False
        self.setWindowTitle("AudioForge Setup")
        self.setModal(True)

        if restart_completed and self.config.first_run_setup_state == "completed":
            self.config.first_run_setup_steps = {
                step: "pending" for step in FIRST_RUN_SETUP_STEPS
            }
            self.config.first_run_setup_step = "devices"
        elif self.config.first_run_setup_state == "completed_with_skips":
            self.config.first_run_setup_steps = {
                step: ("pending" if state == "skipped" else state)
                for step, state in self.config.first_run_setup_steps.items()
            }
            self.config.first_run_setup_step = next(
                (
                    step
                    for step in FIRST_RUN_SETUP_STEPS
                    if self.config.first_run_setup_steps.get(step) == "pending"
                ),
                "devices",
            )
        self.config.first_run_setup_state = "in_progress"
        self._advanced_latency_mode = (
            self.config.first_run_setup_step == "latency"
            and self.config.first_run_setup_steps.get("latency") != "completed"
        )
        self._route_check_phase = "speech"
        self._route_check_pending = False
        self._step_index = self._initial_step_index()
        self._save_progress()
        self._route_check_timer = QTimer(self)
        self._route_check_timer.setSingleShot(True)
        self._route_check_timer.timeout.connect(self._finish_route_check)

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACING_SECTION)
        self.progress = QProgressBar()
        self.progress.setRange(0, len(DEFAULT_SETUP_STEPS))
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(DESCRIPTION_LABEL_STYLE)
        layout.addWidget(self.description_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(message_text_style("info"))
        layout.addWidget(self.status_label)

        self.mute_checkbox = QCheckBox("Mute Output")
        self.mute_checkbox.setToolTip(
            "Keep transmission muted until you uncheck this control. "
            "Calibration and stream recovery may use a separate temporary mute."
        )
        self.mute_checkbox.setChecked(bool(getattr(self.owner, "user_muted", False)))
        self.mute_checkbox.toggled.connect(self._on_mute_toggled)
        layout.addWidget(self.mute_checkbox)

        self.device_selection_group = QGroupBox("Audio route")
        device_layout = QGridLayout(self.device_selection_group)
        input_label = QLabel("Microphone:")
        self.input_device_selector = QComboBox()
        bind_label(
            input_label,
            self.input_device_selector,
            name="Setup microphone",
        )
        device_layout.addWidget(input_label, 0, 0)
        device_layout.addWidget(self.input_device_selector, 0, 1)

        output_label = QLabel("Destination:")
        self.output_device_selector = QComboBox()
        bind_label(
            output_label,
            self.output_device_selector,
            name="Setup destination",
        )
        device_layout.addWidget(output_label, 1, 0)
        device_layout.addWidget(self.output_device_selector, 1, 1)
        device_layout.setColumnStretch(1, 1)
        layout.addWidget(self.device_selection_group)
        self.input_device_selector.setModel(self.owner.input_combo.model())
        self.output_device_selector.setModel(self.owner.output_combo.model())
        self.input_device_selector.setCurrentIndex(
            self.owner.input_combo.currentIndex()
        )
        self.output_device_selector.setCurrentIndex(
            self.owner.output_combo.currentIndex()
        )

        button_row = QGridLayout()
        button_row.setSpacing(SPACING_NORMAL)
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.back_button.clicked.connect(self._go_back)
        button_row.addWidget(self.back_button, 0, 0)

        self.skip_button = QPushButton("Skip This Step")
        self.skip_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.skip_button.clicked.connect(self._skip_step)
        button_row.addWidget(self.skip_button, 0, 1)

        self.pause_button = QPushButton("Pause and Close")
        self.pause_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.pause_button.clicked.connect(self.reject)
        button_row.addWidget(self.pause_button, 1, 0)

        self.action_button = QPushButton()
        self.action_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.action_button.clicked.connect(self._run_current_step)
        button_row.addWidget(self.action_button, 1, 1)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)
        layout.addLayout(button_row)

        self.advanced_latency_button = QPushButton("Advanced Latency Calibration...")
        self.advanced_latency_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.advanced_latency_button.clicked.connect(self._run_latency_calibration)
        self.advanced_latency_button.setEnabled(
            callable(getattr(self.owner, "_on_latency_calibration_clicked", None))
        )
        layout.addWidget(self.advanced_latency_button)

        set_accessible_group(
            (
                (self.progress, "Setup progress", None),
                (self.mute_checkbox, "Mute output during setup", None),
                (self.input_device_selector, "Setup microphone", None),
                (self.output_device_selector, "Setup destination", None),
                (self.back_button, "Previous setup step", None),
                (self.skip_button, "Skip current setup step", None),
                (self.pause_button, "Pause setup", None),
                (self.action_button, "Run current setup step", None),
                (
                    self.advanced_latency_button,
                    "Open advanced latency calibration",
                    None,
                ),
            )
        )
        self.setTabOrder(self.mute_checkbox, self.input_device_selector)
        self.setTabOrder(self.input_device_selector, self.output_device_selector)
        self.setTabOrder(self.output_device_selector, self.back_button)
        self.setTabOrder(self.back_button, self.skip_button)
        self.setTabOrder(self.skip_button, self.pause_button)
        self.setTabOrder(self.pause_button, self.action_button)
        self.setTabOrder(self.action_button, self.advanced_latency_button)
        self._render_step()
        configure_resizable_dialog(
            self,
            preferred_width=620,
            preferred_height=340,
            minimum_width=440,
            minimum_height=280,
        )

    def _initial_step_index(self) -> int:
        current = self.config.first_run_setup_step
        if current in DEFAULT_SETUP_STEPS:
            current_index = DEFAULT_SETUP_STEPS.index(current)
            if self.config.first_run_setup_steps.get(current) != "completed":
                return current_index
        for index, step in enumerate(DEFAULT_SETUP_STEPS):
            if self.config.first_run_setup_steps.get(step) != "completed":
                return index
        return len(DEFAULT_SETUP_STEPS) - 1

    @property
    def current_step(self) -> str:
        if self._advanced_latency_mode:
            return "latency"
        return DEFAULT_SETUP_STEPS[self._step_index]

    def _save_progress(self) -> None:
        self.config.first_run_setup_step = self.current_step
        save_config(self.config)

    def _render_step(self) -> None:
        step = self.current_step
        if step != "route":
            self._route_check_timer.stop()
            self._route_check_pending = False
            self._route_check_phase = "speech"
            self.action_button.setEnabled(True)
        title, description, action = STEP_CONTENT[step]
        state = self.config.first_run_setup_steps.get(step, "pending")
        completed_count = sum(
            self.config.first_run_setup_steps.get(default_step) == "completed"
            for default_step in DEFAULT_SETUP_STEPS
        )
        self.progress.setValue(completed_count)
        self.progress.setFormat(
            f"{completed_count}/{len(DEFAULT_SETUP_STEPS)} completed"
        )
        self.title_label.setText(f"<h2>{title}</h2>")
        self.description_label.setText(description)
        self.device_selection_group.setVisible(step == "devices")
        self.advanced_latency_button.setVisible(step != "latency")
        self.mute_checkbox.blockSignals(True)
        self.mute_checkbox.setChecked(bool(getattr(self.owner, "user_muted", False)))
        self.mute_checkbox.blockSignals(False)
        self.status_label.setText(
            "This step was completed. You can run it again or continue."
            if state == "completed"
            else "Ready. Progress is saved if you close this window."
        )
        self.status_label.setStyleSheet(message_text_style("info"))
        if step == "route" and self._route_check_phase == "destination":
            self.action_button.setText("Confirm Destination Signal")
        else:
            self.action_button.setText(action)
        self.back_button.setEnabled(
            self._advanced_latency_mode or self._step_index > 0
        )

    def _set_status(self, message: str, state: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(message_text_style(state))

    def _on_mute_toggled(self, checked: bool) -> None:
        """Route the onboarding control through the main window's mute owner."""
        self.owner.user_mute_checkbox.setChecked(checked)

    def _selected_devices_ready(self) -> bool:
        input_combo = self.input_device_selector
        output_combo = self.output_device_selector
        if self.current_step != "devices":
            input_combo = self.owner.input_combo
            output_combo = self.owner.output_combo
        input_identity = coerce_device_identity(
            input_combo.currentData()
        )
        output_identity = coerce_device_identity(
            output_combo.currentData()
        )
        return input_identity is not None and output_identity is not None

    def _apply_selected_devices(self) -> bool:
        input_index = self.input_device_selector.currentIndex()
        output_index = self.output_device_selector.currentIndex()
        if input_index < 0 or output_index < 0:
            return False

        input_combo = self.owner.input_combo
        output_combo = self.owner.output_combo
        if input_index >= input_combo.count() or output_index >= output_combo.count():
            return False
        route_changed = (
            input_combo.currentIndex() != input_index
            or output_combo.currentIndex() != output_index
        )
        if route_changed and self.owner.processor.is_running():
            self.owner._stop_processing()
            if self.owner.processor.is_running():
                return False
        input_signals_blocked = input_combo.blockSignals(True)
        output_signals_blocked = output_combo.blockSignals(True)
        try:
            input_combo.setCurrentIndex(input_index)
            output_combo.setCurrentIndex(output_index)
        except (RuntimeError, TypeError, ValueError):
            return False
        finally:
            input_combo.blockSignals(input_signals_blocked)
            output_combo.blockSignals(output_signals_blocked)
        if route_changed:
            self._route_check_phase = "speech"
            self.owner._on_device_changed()
        return (
            input_combo.currentIndex() == input_index
            and output_combo.currentIndex() == output_index
            and self._selected_devices_ready()
        )

    def _run_current_step(self) -> None:
        step = self.current_step
        if step == "devices":
            if not self._selected_devices_ready():
                self._set_status(
                    "Both an input and output endpoint must be available and selected.",
                    "error",
                )
                return
            if not self._apply_selected_devices():
                self._set_status(
                    "Those devices are no longer available. Return to the main window, "
                    "refresh the device list, and try again.",
                    "error",
                )
                return
            self._complete_step("Microphone and destination selected.")
            return
        if step == "route":
            if not self._selected_devices_ready():
                self._route_check_phase = "speech"
                self._set_status(
                    "The selected route is unavailable. Return to step 1.", "error"
                )
                return
            if getattr(self.owner, "user_muted", False):
                action = (
                    "confirm the destination signal"
                    if self._route_check_phase == "destination"
                    else "run this check"
                )
                self._set_status(
                    f"Output is muted. Uncheck Mute Output above, then {action}.",
                    "warn",
                )
                return
            if self._route_check_phase == "destination":
                healthy, reason = route_health_reason(self.owner.processor)
                if not healthy:
                    self._route_check_phase = "speech"
                    self._set_status(reason, "error")
                    self.action_button.setText("Check Speech & Clipping")
                    return
                self._complete_step("Destination signal confirmed by user.")
                return
            if not self.owner.processor.is_running():
                self.owner._start_processing()
            self.action_button.setEnabled(False)
            self._route_check_pending = True
            self._set_status("Checking the audio stream...", "info")
            self._route_check_timer.start(750)
            return
        if step == "latency":
            self._run_latency_calibration()
            return
        if step == "voice":
            self.hide()
            try:
                applied = bool(self.owner._on_auto_voice_setup_clicked())
            finally:
                self.show()
            if applied:
                self._complete_step(
                    "Voice setup applied settings after downstream validation. "
                    "Check the result in your destination app."
                )
            else:
                self._set_status(
                    "Voice Setup closed without applying a chain. Retry or skip this step.",
                    "warn",
                )

    def _finish_route_check(self) -> None:
        if not self._route_check_pending or self.current_step != "route":
            return
        self._route_check_timer.stop()
        self._route_check_pending = False
        self.action_button.setEnabled(True)
        healthy, reason = route_health_reason(self.owner.processor)
        if healthy:
            self._route_check_phase = "destination"
            self._set_status(
                f"{reason} Check your destination app, then confirm that it receives "
                "your microphone signal.",
                "success",
            )
            self.action_button.setText("Confirm Destination Signal")
        else:
            self._route_check_phase = "speech"
            self.action_button.setText("Check Speech & Clipping")
            self._set_status(reason, "error")

    def _run_latency_calibration(self) -> None:
        self.hide()
        try:
            saved = bool(self.owner._on_latency_calibration_clicked())
        finally:
            self.show()
        if saved:
            self.config.first_run_setup_steps["latency"] = "completed"
            self._set_status("A measured latency profile was saved for this route.", "success")
            if self._advanced_latency_mode:
                self._advanced_latency_mode = False
            self._save_progress()
            self._render_step()
        else:
            self._set_status(
                "No latency result was saved. Retry when needed, or continue with the "
                "standard setup.",
                "warn",
            )

    def _complete_step(self, message: str) -> None:
        self.config.first_run_setup_steps[self.current_step] = "completed"
        self._set_status(message, "success")
        self._advance_or_finish()

    def _skip_step(self) -> None:
        self.config.first_run_setup_steps[self.current_step] = "skipped"
        self._advance_or_finish()

    def _advance_or_finish(self) -> None:
        if self._advanced_latency_mode:
            self._advanced_latency_mode = False
            self._step_index = self._initial_step_index()
            self._save_progress()
            self._render_step()
            return
        if self._step_index < len(DEFAULT_SETUP_STEPS) - 1:
            self._step_index += 1
            self._save_progress()
            self._render_step()
            return
        self._finish_setup()

    def _go_back(self) -> None:
        if self._advanced_latency_mode:
            self._advanced_latency_mode = False
            self._step_index = max(0, self._step_index - 1)
            self._save_progress()
            self._render_step()
            return
        if self._step_index == 0:
            return
        self._step_index -= 1
        self._route_check_phase = "speech"
        self._save_progress()
        self._render_step()

    def _finish_setup(self) -> None:
        skipped = any(
            self.config.first_run_setup_steps.get(step) == "skipped"
            for step in DEFAULT_SETUP_STEPS
        )
        pending = any(
            self.config.first_run_setup_steps.get(step) == "pending"
            for step in DEFAULT_SETUP_STEPS
        )
        if pending:
            QMessageBox.information(
                self,
                "Setup Paused",
                "Some steps are still pending. Progress was saved and can be resumed later.",
            )
            return
        self.config.first_run_setup_state = (
            "completed_with_skips" if skipped else "completed"
        )
        self._finalized = True
        save_config(self.config)
        self.accept()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._route_check_timer.stop()
        if not self._finalized:
            self.config.first_run_setup_state = "in_progress"
            self._save_progress()
        super().closeEvent(event)

    def reject(self) -> None:
        self._route_check_timer.stop()
        self._route_check_pending = False
        super().reject()


__all__ = ["FirstRunSetupDialog", "route_health_reason"]
