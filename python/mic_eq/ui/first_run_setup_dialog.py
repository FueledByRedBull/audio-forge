"""Thin, resumable shell around AudioForge's existing setup workflows."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..config import coerce_device_identity, save_config
from ..config_parts.app_config import FIRST_RUN_SETUP_STEPS
from .accessibility import set_accessible_group
from .layout_constants import (
    PRIMARY_ACTION_BUTTON_STYLE,
    SECONDARY_ACTION_BUTTON_STYLE,
    SPACING_NORMAL,
    SPACING_SECTION,
)
from .theme import DESCRIPTION_LABEL_STYLE, message_text_style


STEP_CONTENT = {
    "devices": (
        "1. Select the route",
        "Choose the microphone and output or virtual cable in the main window. "
        "AudioForge remembers stable Windows endpoint identities, including duplicate names.",
        "Check Selected Devices",
    ),
    "route": (
        "2. Verify the live route",
        "Start the existing processing path and verify that both Windows audio callbacks remain "
        "healthy. This checks stream operation; it does not claim that a cable is audibly patched.",
        "Run Route Check",
    ),
    "latency": (
        "3. Measure route latency",
        "Open the existing latency calibration workflow. Save a result to complete this step, "
        "or skip it and continue with engine-only latency reporting.",
        "Open Latency Calibration",
    ),
    "voice": (
        "4. Calibrate the voice chain",
        "Open the existing Auto Voice Setup workflow. This shell does not duplicate capture, "
        "analysis, or apply logic.",
        "Open Auto Voice Setup",
    ),
}


def route_health_reason(processor: object) -> tuple[bool, str]:
    """Return a conservative live-stream health decision for the setup shell."""
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
        return False, "Audio callback diagnostics were invalid. Restart AudioForge and retry."
    if callback_error_present:
        return False, "A Windows audio callback reported an error. Retry after checking the route."

    for label, getter_name in (
        ("input", "get_input_callback_age_ms"),
        ("output", "get_output_callback_age_ms"),
    ):
        getter = getattr(processor, getter_name, None)
        if not callable(getter):
            return False, f"The {label} callback heartbeat is unavailable. Retry the route."
        try:
            raw_age = getter()
        except (TypeError, ValueError, OverflowError):
            return False, f"The {label} callback heartbeat could not be read. Retry the route."
        if (
            isinstance(raw_age, bool)
            or not isinstance(raw_age, (int, float))
            or not 0.0 <= float(raw_age) < float("inf")
        ):
            return False, f"The {label} callback heartbeat is invalid. Retry the route."
        age_ms = float(raw_age)
        if age_ms > 2_000.0:
            return False, f"The {label} callback is stale ({age_ms:.0f} ms). Retry the route."
    return True, "Both native audio streams are active without reported callback errors."


class FirstRunSetupDialog(QDialog):
    """Persisted setup navigator that delegates every operation to the main window."""

    def __init__(self, owner: Any, *, restart_completed: bool = False):
        super().__init__(owner)
        self.owner = owner
        self.config = owner.config
        self._finalized = False
        self.setWindowTitle("AudioForge Setup")
        self.setModal(True)
        self.setMinimumWidth(620)

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
        self._step_index = self._initial_step_index()
        self._save_progress()
        self._route_check_timer = QTimer(self)
        self._route_check_timer.setSingleShot(True)
        self._route_check_timer.timeout.connect(self._finish_route_check)

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACING_SECTION)
        self.progress = QProgressBar()
        self.progress.setRange(0, len(FIRST_RUN_SETUP_STEPS))
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

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACING_NORMAL)
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.back_button.clicked.connect(self._go_back)
        button_row.addWidget(self.back_button)

        self.skip_button = QPushButton("Skip This Step")
        self.skip_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.skip_button.clicked.connect(self._skip_step)
        button_row.addWidget(self.skip_button)
        button_row.addStretch()

        self.pause_button = QPushButton("Pause and Close")
        self.pause_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.pause_button.clicked.connect(self.reject)
        button_row.addWidget(self.pause_button)

        self.action_button = QPushButton()
        self.action_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.action_button.clicked.connect(self._run_current_step)
        button_row.addWidget(self.action_button)
        layout.addLayout(button_row)

        set_accessible_group(
            (
                (self.progress, "Setup progress", None),
                (self.back_button, "Previous setup step", None),
                (self.skip_button, "Skip current setup step", None),
                (self.pause_button, "Pause setup", None),
                (self.action_button, "Run current setup step", None),
            )
        )
        self.setTabOrder(self.back_button, self.skip_button)
        self.setTabOrder(self.skip_button, self.pause_button)
        self.setTabOrder(self.pause_button, self.action_button)
        self._render_step()

    def _initial_step_index(self) -> int:
        current = self.config.first_run_setup_step
        if current in FIRST_RUN_SETUP_STEPS:
            current_index = FIRST_RUN_SETUP_STEPS.index(current)
            if self.config.first_run_setup_steps.get(current) != "completed":
                return current_index
        for index, step in enumerate(FIRST_RUN_SETUP_STEPS):
            if self.config.first_run_setup_steps.get(step) != "completed":
                return index
        return len(FIRST_RUN_SETUP_STEPS) - 1

    @property
    def current_step(self) -> str:
        return FIRST_RUN_SETUP_STEPS[self._step_index]

    def _save_progress(self) -> None:
        self.config.first_run_setup_step = self.current_step
        save_config(self.config)

    def _render_step(self) -> None:
        step = self.current_step
        title, description, action = STEP_CONTENT[step]
        state = self.config.first_run_setup_steps.get(step, "pending")
        completed_count = sum(
            value == "completed"
            for value in self.config.first_run_setup_steps.values()
        )
        self.progress.setValue(completed_count)
        self.progress.setFormat(
            f"{completed_count}/{len(FIRST_RUN_SETUP_STEPS)} completed"
        )
        self.title_label.setText(f"<h2>{title}</h2>")
        self.description_label.setText(description)
        self.status_label.setText(
            "This step was completed. You can run it again or continue."
            if state == "completed"
            else "Ready. Progress is saved if you close this window."
        )
        self.status_label.setStyleSheet(message_text_style("info"))
        self.action_button.setText(action)
        self.back_button.setEnabled(self._step_index > 0)

    def _set_status(self, message: str, state: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(message_text_style(state))

    def _selected_devices_ready(self) -> bool:
        input_identity = coerce_device_identity(self.owner.input_combo.currentData())
        output_identity = coerce_device_identity(self.owner.output_combo.currentData())
        return input_identity is not None and output_identity is not None

    def _run_current_step(self) -> None:
        step = self.current_step
        if step == "devices":
            if not self._selected_devices_ready():
                self._set_status(
                    "Both an input and output endpoint must be available and selected.", "error"
                )
                return
            self._complete_step("Selected input and output endpoints are available.")
            return
        if step == "route":
            if not self._selected_devices_ready():
                self._set_status("The selected route is unavailable. Return to step 1.", "error")
                return
            if not self.owner.processor.is_running():
                self.owner._start_processing()
            self.action_button.setEnabled(False)
            self._set_status("Checking native input and output callbacks...", "info")
            self._route_check_timer.start(750)
            return
        if step == "latency":
            self.hide()
            try:
                saved = bool(self.owner._on_latency_calibration_clicked())
            finally:
                self.show()
            if saved:
                self._complete_step("A measured latency profile was saved for this route.")
            else:
                self._set_status(
                    "No latency result was saved. Retry, or skip this optional step honestly.",
                    "warn",
                )
            return
        if step == "voice":
            self.hide()
            try:
                applied = bool(self.owner._on_auto_voice_setup_clicked())
            finally:
                self.show()
            if applied:
                self._complete_step("Auto Voice Setup applied a validated chain.")
            else:
                self._set_status(
                    "Voice Setup closed without applying a chain. Retry or skip this step.",
                    "warn",
                )

    def _finish_route_check(self) -> None:
        self._route_check_timer.stop()
        self.action_button.setEnabled(True)
        healthy, reason = route_health_reason(self.owner.processor)
        if healthy:
            self._complete_step(reason)
        else:
            self._set_status(reason, "error")

    def _complete_step(self, message: str) -> None:
        self.config.first_run_setup_steps[self.current_step] = "completed"
        self._set_status(message, "success")
        self._advance_or_finish()

    def _skip_step(self) -> None:
        self.config.first_run_setup_steps[self.current_step] = "skipped"
        self._advance_or_finish()

    def _advance_or_finish(self) -> None:
        if self._step_index < len(FIRST_RUN_SETUP_STEPS) - 1:
            self._step_index += 1
            self._save_progress()
            self._render_step()
            return
        self._finish_setup()

    def _go_back(self) -> None:
        if self._step_index == 0:
            return
        self._step_index -= 1
        self._save_progress()
        self._render_step()

    def _finish_setup(self) -> None:
        skipped = any(
            state == "skipped" for state in self.config.first_run_setup_steps.values()
        )
        pending = any(
            state == "pending" for state in self.config.first_run_setup_steps.values()
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
        super().reject()


__all__ = ["FirstRunSetupDialog", "route_health_reason"]
