"""Resumable first-run setup for selecting a route and running checks."""

from __future__ import annotations

import logging
import sys
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout

from ..config import coerce_device_identity, save_config
from ..config_parts.app_config import FIRST_RUN_SETUP_STEPS
from .first_run_setup_view import FirstRunSetupView
from .health import RecentStreamHealth
from .layout_constants import configure_resizable_dialog
from .quick_scene import QML_DIR, load_scene, quick_theme
from .quick_shell import register_qml_types


logger = logging.getLogger(__name__)

DEFAULT_SETUP_STEPS = ("devices", "route", "voice")

MUTE_TIP = (
    "Keep transmission muted until you turn this off. "
    "Calibration and stream recovery may use a separate temporary mute."
)


STEP_CONTENT = {
    "devices": (
        "1. Choose devices",
        "Select the microphone and destination for your calls, games, or recording. "
        + (
            "Choose a virtual cable when another app should receive AudioForge output."
            if sys.platform == "win32"
            else "Choose AudioForge Mic Input when another app should receive AudioForge "
            "output; that app then records from AudioForge Mic."
        ),
        "Use this route",
    ),
    "route": (
        "2. Check speech and clipping",
        "Start processing, speak at a normal level, and watch the input and output "
        "levels and clipping indicators. Then check your call, game, or recording "
        "app for the microphone signal and confirm it below. AudioForge cannot "
        "check automatically that the other app receives it.",
        "Check levels",
    ),
    "latency": (
        "Advanced latency calibration",
        "Measure the selected route when you use a loopback cable or a "
        "speaker-to-microphone path. This is optional and is also available on the "
        "Settings page.",
        "Open latency calibration",
    ),
    "voice": (
        "3. Calibrate the voice chain",
        "Run voice setup to tune the chain. After it finishes, check the result in your destination app.",
        "Open Voice Setup",
    ),
}


# The main button says one of these; the scene sizes the dialog for the longest.
ACTION_LABELS = [*(content[2] for content in STEP_CONTENT.values()), "Confirm signal", "Retry save"]


def route_health_reason(
    processor: object, *, health_window: RecentStreamHealth | None = None,
    now: float | None = None,
) -> tuple[bool, str]:
    """Return a conservative live-stream health decision for setup."""
    is_running = getattr(processor, "is_running", None)
    try:
        running = callable(is_running) and bool(is_running())
    except Exception:
        running = False
    if not running:
        return False, "Processing did not start. Check device availability and retry."

    diagnostics_getter = getattr(processor, "get_runtime_diagnostics", None)
    try:
        diagnostics = diagnostics_getter() if callable(diagnostics_getter) else {}
    except Exception:
        return False, "Audio stream health could not be read. Retry the route."
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    window = health_window if health_window is not None else RecentStreamHealth()
    try:
        recent_events = window.observe(diagnostics, now=now)
    except ValueError:
        return (
            False,
            "Audio stream health data was invalid. Restart AudioForge and retry.",
        )
    if recent_events - {"gate_chatter_event_count"}:
        return (
            False,
            "The audio stream reported an error or recovery recently. "
            "Wait for five stable seconds, then retry the route.",
        )
    for name in ("is_recovering", "is_recovery_requested"):
        getter = getattr(processor, name, None)
        if callable(getter):
            try:
                if getter():
                    return False, "Audio stream recovery is pending. Check the selected route."
            except (TypeError, ValueError, OSError, RuntimeError):
                return False, "Audio stream recovery status could not be read. Retry the route."

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
        except (TypeError, ValueError, OverflowError, OSError, RuntimeError):
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


class FirstRunSetupLoadError(RuntimeError):
    """The dialog's Qt Quick scene did not load, so there is nothing to show."""


class FirstRunSetupDialog(QDialog):
    """Persisted setup flow for selecting a route and running checks."""

    def __init__(self, owner: Any, *, restart_completed: bool = False):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("AudioForge Setup")
        self.setModal(True)
        # The scene comes first: if it cannot load, the saved progress is left as it was.
        self._setup_ui()
        self._stream_health = getattr(owner, "_stream_health", None) or RecentStreamHealth()
        self.config = owner.config
        self._finalized = False
        self._progress_unsaved = False
        # A re-run never writes "in_progress", so quitting mid-run cannot re-prompt at launch.
        self._exit_state = (
            self.config.first_run_setup_state
            if self.config.first_run_setup_state in {"completed", "completed_with_skips"}
            else "in_progress"
        )

        # A paused re-run resumes; only a fully finished setup restarts from scratch.
        paused = any(
            self.config.first_run_setup_steps.get(step) == "pending"
            for step in DEFAULT_SETUP_STEPS
        )
        if (
            restart_completed
            and self.config.first_run_setup_state == "completed"
            and not paused
        ):
            self.config.first_run_setup_steps = {
                step: "pending" for step in FIRST_RUN_SETUP_STEPS
            }
            self.config.first_run_setup_step = "devices"
        elif self._exit_state != "in_progress" and (
            paused or self._exit_state == "completed_with_skips"
        ):
            for step in DEFAULT_SETUP_STEPS:
                if self.config.first_run_setup_steps.get(step) == "skipped":
                    self.config.first_run_setup_steps[step] = "pending"
            self.config.first_run_setup_step = next(
                (
                    step
                    for step in DEFAULT_SETUP_STEPS
                    if self.config.first_run_setup_steps.get(step) == "pending"
                ),
                "devices",
            )
        self.config.first_run_setup_state = self._exit_state
        self._advanced_latency_mode = (
            self.config.first_run_setup_step == "latency"
            and self.config.first_run_setup_steps.get("latency") != "completed"
        )
        self._route_check_phase = "speech"
        self._route_check_pending = False
        self._started_processing = False
        self._step_index = self._initial_step_index()
        self._save_progress()
        self._route_check_timer = QTimer(self)
        self._route_check_timer.setSingleShot(True)
        self._route_check_timer.timeout.connect(self._finish_route_check)
        self._route_feedback_timer = QTimer(self)
        self._route_feedback_timer.setInterval(100)
        self._route_feedback_timer.timeout.connect(self._update_route_feedback)

        owner_mute_checkbox = getattr(self.owner, "user_mute_checkbox_action", None)
        if owner_mute_checkbox is not None and hasattr(owner_mute_checkbox, "toggled"):
            owner_mute_checkbox.toggled.connect(self._sync_mute_control)
        self._render_step()
        # The scene's own minimum grows with the text size; the dialog opens at least that wide.
        minimum_width = max(440, int(self.quick_view.rootObject().property("minimumWidth")))
        configure_resizable_dialog(
            self,
            preferred_width=max(620, minimum_width),
            preferred_height=660,
            minimum_width=minimum_width,
            minimum_height=340,
        )

    def _setup_ui(self) -> None:
        input_choice, output_choice = self.owner.input_choice, self.owner.output_choice
        # The dialog holds a snapshot; a device refresh can reorder the owner.
        self._input_identities = [
            input_choice.itemData(index) for index in range(input_choice.count())
        ]
        self._output_identities = [
            output_choice.itemData(index) for index in range(output_choice.count())
        ]
        self.view_model = FirstRunSetupView(
            pause=self.reject,
            back=self._go_back,
            skip=self._skip_step,
            act=self._run_current_step,
            open_latency=self._run_latency_calibration,
            request_mute=self._on_mute_toggled,
            action_labels=ACTION_LABELS,
            inputs=[input_choice.itemText(index) for index in range(input_choice.count())],
            outputs=[output_choice.itemText(index) for index in range(output_choice.count())],
            input_index=input_choice.currentIndex(),
            output_index=output_choice.currentIndex(),
            mute_tip=MUTE_TIP,
            parent=self,
        )
        self.view_model.set_muted(bool(getattr(self.owner, "user_muted", False)))
        self.view_model.set_page(
            latencyEnabled=callable(getattr(self.owner, "_on_latency_calibration_clicked", None))
        )
        register_qml_types()
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view,
            QML_DIR / "FirstRunSetup.qml",
            quick_theme(),
            {"vm": self.view_model},
        )
        if reason is not None:
            # Nothing is connected yet; leave the parent without a half-built child.
            self.setParent(None)
            raise FirstRunSetupLoadError(f"The setup view did not load: {reason}")
        self.quick_view.setAccessibleName("Audio setup content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)

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

    def _set_route_feedback_active(self, active: bool) -> None:
        if active:
            if not self._route_feedback_timer.isActive():
                self._route_feedback_timer.start()
            self._update_route_feedback()
            return

        self._route_feedback_timer.stop()
        for meter in (self.view_model.input_meter, self.view_model.output_meter):
            meter.decay_timer.stop()

    def _update_route_feedback(self) -> None:
        if self.current_step != "route":
            return
        running = self.owner.processor.is_running()
        for direction, meter in (
            ("input", self.view_model.input_meter),
            ("output", self.view_model.output_meter),
        ):
            if running:
                source_meter = getattr(self.owner, f"{direction}_meter")
                source = self.owner.shell_texts[f"{direction}_health_label"]
                if not getattr(source_meter, "measurement_available", True):
                    meter.set_unavailable()
                elif not getattr(source_meter, "peak_available", True):
                    meter.set_rms_level(source_meter.rms_db)
                else:
                    meter.set_levels(source_meter.rms_db, source_meter.peak_db)
                text, state = source.text, source.state
            else:
                meter.set_unavailable()
                text, state = f"{direction.title()}: --", "idle"
            self.view_model.set_health(direction, text, state)

    def _save_progress(self) -> bool:
        self.config.first_run_setup_step = self.current_step
        try:
            saved = save_config(self.config)
        except (OSError, TypeError, ValueError):
            saved = False
        self._progress_unsaved = not bool(saved)
        return bool(saved)

    def _default_setup_finished(self) -> bool:
        return all(
            self.config.first_run_setup_steps.get(step) in {"completed", "skipped"}
            for step in DEFAULT_SETUP_STEPS
        )

    def _render_step(self) -> None:
        step = self.current_step
        if step != "route":
            self._route_check_timer.stop()
            self._route_check_pending = False
            self._started_processing = False
            self._route_check_phase = "speech"
            self._set_route_feedback_active(False)
        else:
            self._set_route_feedback_active(True)
        title, description, action = STEP_CONTENT[step]
        state = self.config.first_run_setup_steps.get(step, "pending")
        completed_count = sum(
            self.config.first_run_setup_steps.get(default_step) == "completed"
            for default_step in DEFAULT_SETUP_STEPS
        )
        if (
            step == "voice"
            and self._progress_unsaved
            and self._default_setup_finished()
        ):
            action = "Retry save"
        elif step == "route" and self._route_check_phase == "destination":
            action = "Confirm signal"
        self.view_model.set_page(
            title=title,
            description=description,
            progressCount=completed_count,
            progressTotal=len(DEFAULT_SETUP_STEPS),
            progressText=f"{completed_count} of {len(DEFAULT_SETUP_STEPS)} steps completed",
            devicesShown=step == "devices",
            routeShown=step == "route",
            latencyShown=step != "latency",
            actionText=action,
            backEnabled=self._advanced_latency_mode or self._step_index > 0,
            **({} if step == "route" else {"actionEnabled": True}),
        )
        self._sync_mute_control()
        if self._progress_unsaved:
            self.view_model.set_status(
                "Progress is unsaved. You can continue, but some steps may need to be "
                "repeated if settings cannot be saved.",
                "warn",
            )
        else:
            self.view_model.set_status(
                "This step was completed. You can run it again or continue."
                if state == "completed"
                else "Ready. Progress is saved if you close this window.",
                "info",
            )

    def _set_status(self, message: str, state: str) -> None:
        if self._progress_unsaved:
            message += " Progress is unsaved."
        self.view_model.set_status(message, state)

    def _on_mute_toggled(self, checked: bool) -> None:
        """Route the onboarding control through the main window's mute owner."""
        self.view_model.set_muted(checked)
        self.owner.user_mute_checkbox_action.setChecked(checked)

    def _sync_mute_control(self, _checked: bool | None = None) -> None:
        """Mirror the live mute owner when a hotkey or tray action changes it."""
        self.view_model.set_muted(bool(getattr(self.owner, "user_muted", False)))

    def _picked_identity(self, identities: list, index: int) -> Any:
        return identities[index] if 0 <= index < len(identities) else None

    def _selected_devices_ready(self) -> bool:
        if self.current_step == "devices":
            input_data = self._picked_identity(
                self._input_identities, self.view_model.input_index
            )
            output_data = self._picked_identity(
                self._output_identities, self.view_model.output_index
            )
        else:
            input_data = self.owner.input_choice.currentData()
            output_data = self.owner.output_choice.currentData()
        return (
            coerce_device_identity(input_data) is not None
            and coerce_device_identity(output_data) is not None
        )

    def _apply_selected_devices(self) -> bool:
        input_combo = self.owner.input_choice
        output_combo = self.owner.output_choice
        # The dialog holds a snapshot; a device refresh can reorder the owner.
        input_identity = self._picked_identity(
            self._input_identities, self.view_model.input_index
        )
        output_identity = self._picked_identity(
            self._output_identities, self.view_model.output_index
        )
        if input_identity is None or output_identity is None:
            return False
        input_index = input_combo.findData(input_identity)
        output_index = output_combo.findData(output_identity)
        if input_index < 0 or output_index < 0:
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
                    "Both a microphone and a destination must be available and selected.",
                    "bad",
                )
                return
            if not self._apply_selected_devices():
                self._set_status(
                    "Those devices are no longer available. Return to the main window, "
                    "refresh the device list, and try again.",
                    "bad",
                )
                return
            self._complete_step("Microphone and destination selected.")
            return
        if step == "route":
            if not self._selected_devices_ready():
                self._route_check_phase = "speech"
                self._set_status(
                    "The selected route is unavailable. Return to step 1.", "bad"
                )
                return
            if getattr(self.owner, "user_muted", False):
                action = (
                    "confirm the destination signal"
                    if self._route_check_phase == "destination"
                    else "run this check"
                )
                self._set_status(
                    f"Output is muted. Turn off Mute output above, then {action}.",
                    "warn",
                )
                return
            if self._route_check_phase == "destination":
                healthy, reason = route_health_reason(
                    self.owner.processor, health_window=self._stream_health
                )
                if not healthy:
                    self._route_check_phase = "speech"
                    self._set_status(reason, "bad")
                    self.view_model.set_page(actionText="Check levels")
                    return
                self._complete_step("You confirmed the destination signal.")
                return
            if not self.owner.processor.is_running():
                self.owner._start_processing()
                self._started_processing = self.owner.processor.is_running()
            self.view_model.set_page(actionEnabled=False)
            self._route_check_pending = True
            self._set_status("Checking the audio stream...", "info")
            self._route_check_timer.start(750)
            return
        if step == "latency":
            self._run_latency_calibration()
            return
        if step == "voice":
            if self._progress_unsaved and self._default_setup_finished():
                self._finish_setup()
                return
            self.hide()
            try:
                applied = bool(self.owner._on_auto_voice_setup_clicked())
            except Exception:
                logger.exception("Voice Setup failed inside the first-run setup")
                applied = False
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
        self.view_model.set_page(actionEnabled=True)
        healthy, reason = route_health_reason(
            self.owner.processor, health_window=self._stream_health
        )
        if healthy:
            self._route_check_phase = "destination"
            self._set_status(
                f"{reason} Check your destination app, then confirm that it receives "
                "your microphone signal.",
                "ok",
            )
            self.view_model.set_page(actionText="Confirm signal")
        else:
            self._route_check_phase = "speech"
            self.view_model.set_page(actionText="Check levels")
            self._set_status(reason, "bad")

    def _run_latency_calibration(self) -> None:
        self._set_route_feedback_active(False)
        self.hide()
        try:
            saved = bool(self.owner._on_latency_calibration_clicked())
        except Exception:
            logger.exception("Latency calibration failed inside the first-run setup")
            saved = False
        finally:
            self.show()
            self._set_route_feedback_active(self.current_step == "route")
        if saved:
            self.config.first_run_setup_steps["latency"] = "completed"
            if self._advanced_latency_mode:
                self._advanced_latency_mode = False
                self._step_index = self._initial_step_index()
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
        self._set_status(message, "ok")
        self._advance_or_finish()

    def _skip_step(self) -> None:
        self.config.first_run_setup_steps[self.current_step] = "skipped"
        self._advance_or_finish()

    def _advance_or_finish(self) -> None:
        if self._advanced_latency_mode:
            self._advanced_latency_mode = False
            self._step_index = self._initial_step_index()
        elif self._step_index < len(DEFAULT_SETUP_STEPS) - 1:
            self._step_index += 1
        else:
            self._finish_setup()
            return
        self._save_progress()
        self._render_step()

    def _go_back(self) -> None:
        if self._advanced_latency_mode:
            self._advanced_latency_mode = False
        elif self._step_index == 0:
            return
        self._step_index = max(0, self._step_index - 1)
        self._route_check_phase = "speech"
        self._save_progress()
        self._render_step()

    def _finish_setup(self) -> bool:
        skipped = any(
            self.config.first_run_setup_steps.get(step) == "skipped"
            for step in DEFAULT_SETUP_STEPS
        )
        pending = any(
            self.config.first_run_setup_steps.get(step) == "pending"
            for step in DEFAULT_SETUP_STEPS
        )
        if pending:
            if not self._save_progress():
                self._render_step()
                return False
            QMessageBox.information(
                self,
                "Setup paused",
                "Some steps are still pending. Progress was saved and can be resumed later.",
            )
            return True
        previous_state = self.config.first_run_setup_state
        self.config.first_run_setup_state = (
            "completed_with_skips" if skipped else "completed"
        )
        if not self._save_progress():
            self.config.first_run_setup_state = previous_state
            self._finalized = False
            self._render_step()
            return False
        self._finalized = True
        self.accept()
        return True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._set_route_feedback_active(self.current_step == "route")

    def hideEvent(self, event) -> None:
        self._route_check_timer.stop()
        self._route_check_pending = False
        self.view_model.set_page(actionEnabled=True)
        self._set_route_feedback_active(False)
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.reject()
        event.accept()

    def reject(self) -> None:
        self._route_check_timer.stop()
        self._route_check_pending = False
        self._set_route_feedback_active(False)
        if self._started_processing and self.owner.processor.is_running():
            self.owner._stop_processing()
        self._started_processing = False
        if not self._finalized:
            self.config.first_run_setup_state = self._exit_state
            if not self._save_progress():
                self.owner.status_message.showMessage(
                    "Setup progress could not be saved; some steps may need to be repeated",
                    6000,
                )
        super().reject()


__all__ = ["FirstRunSetupDialog", "route_health_reason"]
