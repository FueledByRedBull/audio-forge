"""Tests for the resumable first-run setup flow."""

from __future__ import annotations

import logging

import pytest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QComboBox, QWidget

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.config_parts.app_config import FIRST_RUN_SETUP_STEPS
from mic_eq.ui.first_run_setup_dialog import (
    FirstRunSetupDialog,
    route_health_reason,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.shell_state import TextState
from mic_eq.ui.theme import PALETTE


class _Processor:
    def __init__(self, *, running: bool = False, age_ms: float = 4.0):
        self.running = running
        self.age_ms = age_ms
        self.diagnostics: dict[str, object] = {}
        self.input_rms = -24.0
        self.input_peak = -12.0
        self.output_rms = -24.0
        self.output_peak = -12.0

    def is_running(self):
        return self.running

    def get_runtime_diagnostics(self):
        return dict(self.diagnostics)

    def get_input_callback_age_ms(self):
        return self.age_ms

    def get_output_callback_age_ms(self):
        return self.age_ms

    def get_input_rms_db(self):
        return self.input_rms

    def get_input_peak_db(self):
        return self.input_peak

    def get_output_rms_db(self):
        return self.output_rms

    def get_output_peak_db(self):
        return self.output_peak


class _MeterState:
    def __init__(self, rms_db: float, peak_db: float):
        self.rms_db = rms_db
        self.peak_db = peak_db


class _Owner(QWidget):
    def __init__(self, config: AppConfig, processor: _Processor):
        super().__init__()
        self.config = config
        self.processor = processor
        input_identity = DeviceIdentity(
            name="Mic",
            endpoint_id="input-id",
            direction="input",
            name_ordinal=0,
        )
        self.input_choice = QComboBox()
        self.input_choice.addItem(input_identity.name, input_identity)
        output_identity = DeviceIdentity(
            name="Cable",
            endpoint_id="output-id",
            direction="output",
            name_ordinal=0,
        )
        self.output_choice = QComboBox()
        self.output_choice.addItem(output_identity.name, output_identity)
        self.latency_saved = False
        self.voice_applied = False
        self.voice_setup_calls = 0
        self.status_message = Mock()
        self.user_muted = False
        self.input_meter = _MeterState(processor.input_rms, processor.input_peak)
        self.output_meter = _MeterState(processor.output_rms, processor.output_peak)
        self.shell_texts = {
            "input_health_label": TextState(self, text="Input: OK"),
            "output_health_label": TextState(self, text="Output: OK"),
        }
        for state in self.shell_texts.values():
            state.update(status="ok")
        self._temporary_mute_reasons = {"calibration"}
        self.mute_toggles = []
        self.user_mute_checkbox = QCheckBox(self)
        self.user_mute_checkbox_action = self.user_mute_checkbox
        self.user_mute_checkbox.toggled.connect(self._on_user_mute_toggled)
        self.start_calls = 0
        self.stop_calls = 0
        self.device_change_calls = 0
        self.device_change_routes = []

    def _start_processing(self):
        self.start_calls += 1
        self.processor.running = True

    def _stop_processing(self):
        self.stop_calls += 1
        self.processor.running = False

    def _on_device_changed(self):
        self.device_change_calls += 1
        self.device_change_routes.append(
            (self.input_choice.currentData(), self.output_choice.currentData())
        )

    def _on_latency_calibration_clicked(self):
        return self.latency_saved

    def _on_auto_voice_setup_clicked(self):
        self.voice_setup_calls += 1
        return self.voice_applied

    def _on_user_mute_toggled(self, checked):
        self.user_muted = bool(checked)
        self.mute_toggles.append(self.user_muted)


@pytest.fixture
def save_config_ok(monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )


def opening_setup_dialogs(qapp):
    """Build the dialog and show it, so the scene takes real clicks and keys.

    The generator behind the `open_setup` fixtures of the modules that drive the scene.
    """
    made: list[FirstRunSetupDialog] = []

    def make(owner: Any, **kwargs: Any) -> FirstRunSetupDialog:
        dialog = FirstRunSetupDialog(owner, **kwargs)
        made.append(dialog)
        dialog.show()
        qapp.processEvents()
        dialog.activateWindow()
        dialog.quick_view.setFocus()
        qapp.processEvents()
        return dialog

    yield make
    for dialog in made:
        dialog.hide()
        dialog.deleteLater()
    qapp.processEvents()


@pytest.fixture
def open_setup(qapp):
    yield from opening_setup_dialogs(qapp)


def scene_items(root: Any, *, only_visible: bool = False):
    """The scene's items by their visual parents: a Repeater's items are not QObject children."""
    for child in root.childItems():
        if child.isVisible() or not only_visible:
            yield child
            yield from scene_items(child, only_visible=only_visible)


def scene_item(dialog: Any, name: str) -> Any:
    item = next(
        (i for i in scene_items(dialog.quick_view.rootObject()) if i.objectName() == name), None
    )
    assert item is not None, name
    return item


def scene_text(dialog: Any, name: str) -> str:
    return scene_item(dialog, name).property("text")


def at_scene_minimum(dialog: Any, qapp: Any) -> None:
    """Size the dialog to the narrowest its scene allows, at the smallest height.

    The dialog is never wider than its screen, and a screen without fonts (the offscreen one)
    cannot hold the scene's minimum; the scene is judged at its own minimum all the same.
    """
    width = max(dialog.minimumWidth(), int(dialog.quick_view.rootObject().property("minimumWidth")))
    dialog.resize(width, dialog.minimumHeight())
    qapp.processEvents()
    qapp.processEvents()


def cut_short_texts(dialog: Any) -> list[tuple[str, str]]:
    """The visible texts of the scene that show an ellipsis: (item name, text)."""
    return [
        (item.objectName(), item.property("text"))
        for item in scene_items(dialog.quick_view.rootObject(), only_visible=True)
        if item.property("truncated")
    ]


def press(dialog: Any, name: str) -> None:
    """Click the middle of a scene item, as a mouse does."""
    item = scene_item(dialog, name)
    centre = item.mapToScene(item.boundingRect().center()).toPoint()
    QTest.mouseClick(dialog.quick_view, Qt.MouseButton.LeftButton, pos=centre)


def choose(dialog: Any, picker: str, index: int) -> None:
    """Pick the choice at `index` of a drop-down with the arrow keys."""
    item = scene_item(dialog, picker)
    item.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    key = Qt.Key.Key_Down if index > item.property("currentIndex") else Qt.Key.Key_Up
    for _ in range(abs(index - item.property("currentIndex"))):
        QTest.keyClick(dialog.quick_view, key)
    assert item.property("currentIndex") == index


def _route_step_config() -> AppConfig:
    return AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )


def test_route_health_requires_running_recent_error_free_callbacks():
    processor = _Processor(running=False)
    assert route_health_reason(processor)[0] is False

    processor.running = True
    assert route_health_reason(processor)[0] is True

    processor.age_ms = 2_100.0
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "stopped responding" in reason

    processor.age_ms = 4.0
    processor.diagnostics["output_callback_error_count"] = 1
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "reported an error" in reason

    processor.diagnostics["output_callback_error_count"] = "invalid"
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "stream health data was invalid" in reason


def test_route_health_fails_closed_without_valid_callback_heartbeats():
    class MissingHeartbeatProcessor:
        def is_running(self):
            return True

        def get_runtime_diagnostics(self):
            return {}

    healthy, reason = route_health_reason(MissingHeartbeatProcessor())
    assert healthy is False
    assert "stream health check is unavailable" in reason

    processor = _Processor(running=True, age_ms=float("nan"))
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "stream health value is invalid" in reason


def test_route_health_allows_stable_recovery_without_clearing_history():
    from mic_eq.ui.health import RecentStreamHealth

    processor = _Processor(running=True)
    processor.diagnostics["output_callback_error_count"] = 1
    processor.diagnostics["stream_restart_count"] = 2
    window = RecentStreamHealth()
    assert not route_health_reason(processor, health_window=window, now=0.0)[0]
    assert route_health_reason(processor, health_window=window, now=5.0)[0]
    assert processor.diagnostics["output_callback_error_count"] == 1
    assert processor.diagnostics["stream_restart_count"] == 2
    processor.diagnostics["stream_restart_count"] += 1
    assert not route_health_reason(processor, health_window=window, now=6.0)[0]


def test_route_health_ignores_output_drift_retiming():
    from mic_eq.ui.health import RecentStreamHealth

    processor = _Processor(running=True)
    window = RecentStreamHealth()
    assert route_health_reason(processor, health_window=window, now=0.0)[0]
    processor.diagnostics["jitter_dropped_samples"] = 3_845
    assert route_health_reason(processor, health_window=window, now=1.0)[0]


def test_setup_resumes_at_saved_step_and_delegates_route_check(qapp, save_config_ok, open_setup):
    config = _route_step_config()
    owner = _Owner(config, _Processor())
    dialog = open_setup(owner)

    assert dialog.current_step == "route"
    press(dialog, "actionButton")
    assert owner.start_calls == 1
    assert dialog._route_check_timer.isActive()
    assert not scene_item(dialog, "actionButton").isEnabled()
    dialog._finish_route_check()
    assert not dialog._route_check_timer.isActive()
    assert config.first_run_setup_steps["route"] == "pending"
    assert "Confirm signal" in scene_text(dialog, "actionButton")
    press(dialog, "actionButton")
    assert config.first_run_setup_steps["route"] == "completed"
    assert dialog.current_step == "voice"


@pytest.mark.parametrize(
    ("running_before", "stop_calls"),
    [(False, 1), (True, 0)],
    ids=["started", "already_running"],
)
def test_closing_during_level_check_stops_only_processing_it_started(
    qapp, save_config_ok, running_before, stop_calls
):
    config = _route_step_config()
    owner = _Owner(config, _Processor(running=running_before))
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()
    assert dialog._route_check_timer.isActive()
    dialog.reject()

    assert owner.stop_calls == stop_calls
    assert owner.processor.running is running_before


def test_setup_mute_control_tracks_external_owner_changes(qapp, open_setup):
    owner = _Owner(AppConfig(), _Processor())
    dialog = open_setup(owner)
    mute = scene_item(dialog, "muteSwitch")

    assert not mute.property("checked")
    owner.user_mute_checkbox.setChecked(True)
    assert mute.property("checked")
    owner.user_mute_checkbox.setChecked(False)
    assert not mute.property("checked")


def test_setup_records_skips_and_can_resume_them(qapp, save_config_ok):
    config = AppConfig()
    owner = _Owner(config, _Processor(running=True))
    dialog = FirstRunSetupDialog(owner)

    for _step in range(3):
        dialog._skip_step()

    assert config.first_run_setup_state == "completed_with_skips"
    assert all(
        config.first_run_setup_steps[step] == "skipped"
        for step in ("devices", "route", "voice")
    )
    assert config.first_run_setup_steps["latency"] == "pending"

    resumed = FirstRunSetupDialog(owner)
    assert resumed.current_step == "devices"
    assert all(
        config.first_run_setup_steps[step] == "pending"
        for step in ("devices", "route", "voice")
    )
    assert config.first_run_setup_steps["latency"] == "pending"


def test_setup_applies_selected_route_from_wizard(qapp, save_config_ok, open_setup):
    config = AppConfig()
    owner = _Owner(config, _Processor())
    input_desired = DeviceIdentity(
        name="USB Mic",
        endpoint_id="usb-input-id",
        direction="input",
        name_ordinal=0,
    )
    output_desired = DeviceIdentity(
        name="Virtual Cable",
        endpoint_id="virtual-output-id",
        direction="output",
        name_ordinal=0,
    )
    owner.input_choice.addItem(input_desired.name, input_desired)
    owner.output_choice.addItem(output_desired.name, output_desired)

    dialog = open_setup(owner)
    choose(dialog, "inputPicker", 1)
    choose(dialog, "outputPicker", 1)
    press(dialog, "actionButton")

    assert owner.input_choice.currentData() == input_desired
    assert owner.output_choice.currentData() == output_desired
    assert owner.device_change_calls == 1
    assert owner.device_change_routes == [(input_desired, output_desired)]
    assert config.first_run_setup_steps["devices"] == "completed"
    assert dialog.current_step == "route"


def test_setup_applies_snapshot_identities_after_owner_choices_reorder(qapp, open_setup):
    owner = _Owner(AppConfig(), _Processor())
    selected_input = DeviceIdentity(name="Selected Mic", endpoint_id="mic-2", direction="input")
    selected_output = DeviceIdentity(name="Selected Cable", endpoint_id="cable-2", direction="output")
    owner.input_choice.addItem("Selected Mic", selected_input)
    owner.output_choice.addItem("Selected Cable", selected_output)
    dialog = open_setup(owner)
    choose(dialog, "inputPicker", 1)
    choose(dialog, "outputPicker", 1)

    owner.input_choice.clear()
    owner.input_choice.addItem("Replacement", DeviceIdentity(name="Replacement", endpoint_id="mic-1", direction="input"))
    owner.input_choice.addItem("Selected Mic", selected_input)
    owner.output_choice.clear()
    owner.output_choice.addItem("Replacement", DeviceIdentity(name="Replacement", endpoint_id="cable-1", direction="output"))
    owner.output_choice.addItem("Selected Cable", selected_output)

    assert dialog._apply_selected_devices()
    assert owner.input_choice.currentData() == selected_input
    assert owner.output_choice.currentData() == selected_output
    assert owner.device_change_routes == [(selected_input, selected_output)]


def test_setup_rejects_snapshot_identity_removed_from_owner_choices(qapp, open_setup):
    owner = _Owner(AppConfig(), _Processor())
    selected_input = DeviceIdentity(name="Selected Mic", endpoint_id="mic-2", direction="input")
    owner.input_choice.addItem("Selected Mic", selected_input)
    dialog = open_setup(owner)
    choose(dialog, "inputPicker", 1)
    choose(dialog, "outputPicker", 0)

    owner.input_choice.clear()
    replacement = DeviceIdentity(name="Replacement", endpoint_id="mic-1", direction="input")
    owner.input_choice.addItem("Replacement", replacement)

    assert not dialog._apply_selected_devices()
    assert owner.input_choice.currentData() == replacement
    assert owner.device_change_calls == 0


def test_setup_restarts_processing_for_new_route(qapp, save_config_ok, open_setup):
    config = AppConfig()
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    owner.input_choice.addItem(
        "USB Mic",
        DeviceIdentity(
            name="USB Mic",
            endpoint_id="usb-input-id",
            direction="input",
            name_ordinal=0,
        ),
    )
    owner.output_choice.addItem(
        "Virtual Cable",
        DeviceIdentity(
            name="Virtual Cable",
            endpoint_id="virtual-output-id",
            direction="output",
            name_ordinal=0,
        ),
    )

    dialog = open_setup(owner)
    choose(dialog, "inputPicker", 1)
    choose(dialog, "outputPicker", 1)
    press(dialog, "actionButton")

    assert owner.stop_calls == 1
    assert processor.is_running() is False
    press(dialog, "actionButton")
    assert owner.start_calls == 1
    assert dialog._route_check_timer.isActive()


def test_setup_keeps_same_running_route(qapp, save_config_ok):
    config = AppConfig()
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert owner.stop_calls == 0
    assert processor.is_running() is True


def test_setup_destination_confirmation_rechecks_mute_and_stream(qapp, save_config_ok):
    config = _route_step_config()
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    dialog = FirstRunSetupDialog(owner)

    dialog._route_check_phase = "destination"
    owner.user_muted = True
    dialog._run_current_step()
    assert config.first_run_setup_steps["route"] == "pending"
    assert "Output is muted" in scene_text(dialog, "statusText")

    owner.user_muted = False
    processor.running = False
    dialog._run_current_step()
    assert dialog._route_check_phase == "speech"
    assert "did not start" in scene_text(dialog, "statusText")


def test_setup_ignores_late_route_timer_after_navigation(qapp, save_config_ok):
    config = _route_step_config()
    owner = _Owner(config, _Processor())
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()
    assert dialog._route_check_pending is True
    dialog._go_back()
    dialog._finish_route_check()

    assert dialog.current_step == "devices"
    assert config.first_run_setup_steps["route"] == "pending"
    assert "Confirm Destination" not in scene_text(dialog, "actionButton")
    assert scene_item(dialog, "actionButton").isEnabled()


def test_setup_missing_devices_stays_on_route_selection(qapp, save_config_ok):
    owner = _Owner(AppConfig(), _Processor())
    owner.output_choice.clear()
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert dialog.current_step == "devices"
    assert owner.config.first_run_setup_steps["devices"] == "pending"
    assert "must be available" in scene_text(dialog, "statusText")


def test_setup_keeps_latency_optional_and_advanced(qapp, save_config_ok, open_setup):
    config = AppConfig()
    owner = _Owner(config, _Processor())
    owner.latency_saved = True
    dialog = open_setup(owner)

    press(dialog, "skipButton")
    press(dialog, "skipButton")
    assert dialog.current_step == "voice"
    assert config.first_run_setup_steps["latency"] == "pending"
    assert scene_item(dialog, "latencyButton").isEnabled()

    press(dialog, "latencyButton")
    assert config.first_run_setup_steps["latency"] == "completed"
    assert dialog.current_step == "voice"


@pytest.mark.parametrize("step", ["latency", "voice"])
def test_a_failing_setup_step_is_logged_with_its_traceback(
    qapp, save_config_ok, monkeypatch, caplog, step, open_setup
):
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="voice",
        first_run_setup_steps={
            "devices": "completed", "route": "completed", "latency": "completed", "voice": "pending",
        },
    )
    owner = _Owner(config, _Processor())

    def broken():
        raise RuntimeError("dialog could not be built")

    monkeypatch.setattr(owner, "_on_latency_calibration_clicked", broken)
    monkeypatch.setattr(owner, "_on_auto_voice_setup_clicked", broken)
    dialog = open_setup(owner)

    if step == "latency":
        press(dialog, "latencyButton")
    else:
        assert dialog.current_step == "voice"
        press(dialog, "actionButton")

    assert any(
        r.levelno >= logging.ERROR and r.exc_info and "could not be built" in str(r.exc_info[1])
        for r in caplog.records
    )
    assert scene_text(dialog, "statusText")
    assert dialog.isVisible(), "the dialog is back after the failed step"


def test_setup_resumes_legacy_saved_latency_step(qapp, save_config_ok):
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="latency",
        first_run_setup_steps={
            "devices": "completed",
            "route": "completed",
            "latency": "pending",
            "voice": "pending",
        },
    )
    owner = _Owner(config, _Processor())
    owner.latency_saved = True
    dialog = FirstRunSetupDialog(owner)

    assert dialog.current_step == "latency"
    dialog._run_current_step()
    assert config.first_run_setup_steps["latency"] == "completed"
    assert dialog.current_step == "voice"


def test_setup_mute_control_preserves_temporary_mute_owner(qapp, save_config_ok, open_setup):
    owner = _Owner(AppConfig(), _Processor())
    dialog = open_setup(owner)

    press(dialog, "muteSwitch")

    assert owner.user_muted is True
    assert owner.mute_toggles == [True]
    assert owner.user_mute_checkbox.isChecked()
    assert scene_item(dialog, "muteSwitch").property("checked")
    press(dialog, "muteSwitch")
    assert owner.user_muted is False
    assert owner.mute_toggles == [True, False]
    assert not owner.user_mute_checkbox.isChecked()
    assert not scene_item(dialog, "muteSwitch").property("checked")
    assert owner._temporary_mute_reasons == {"calibration"}


def test_setup_save_false_keeps_navigation_usable_and_marks_progress_unsaved(
    qapp, monkeypatch
):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: False
    )
    owner = _Owner(AppConfig(), _Processor())
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert dialog.current_step == "route"
    assert "Check speech" in scene_text(dialog, "titleText")
    assert owner.config.first_run_setup_steps["devices"] == "completed"
    assert dialog._progress_unsaved is True
    assert "Progress is unsaved" in scene_text(dialog, "statusText")


@pytest.mark.parametrize("failure", [OSError("read-only"), TypeError("bad JSON")])
def test_setup_save_exceptions_keep_dialog_usable(qapp, monkeypatch, failure):
    def fail_save(_config):
        raise failure

    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", fail_save
    )
    dialog = FirstRunSetupDialog(_Owner(AppConfig(), _Processor()))

    assert dialog.current_step == "devices"
    assert dialog._progress_unsaved is True
    assert "Progress is unsaved" in scene_text(dialog, "statusText")

    event = QCloseEvent()
    dialog.closeEvent(event)
    assert event.isAccepted() is True
    assert dialog._progress_unsaved is True
    assert "could not be saved" in dialog.owner.status_message.showMessage.call_args.args[0]


def test_setup_failed_finish_offers_retry_without_repeating_voice(qapp, monkeypatch):
    save_results = iter((True, False, True))
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda _config: next(save_results),
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="voice",
        first_run_setup_steps={
            "devices": "completed",
            "route": "completed",
            "latency": "pending",
            "voice": "pending",
        },
    )
    owner = _Owner(config, _Processor())
    owner.voice_applied = True
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert config.first_run_setup_steps["voice"] == "completed"
    assert config.first_run_setup_state == "in_progress"
    assert dialog._finalized is False
    assert scene_text(dialog, "actionButton") == "Retry save"
    assert owner.voice_setup_calls == 1
    assert "Progress is unsaved" in scene_text(dialog, "statusText")

    dialog._run_current_step()

    assert config.first_run_setup_state == "completed"
    assert dialog._finalized is True
    assert owner.voice_setup_calls == 1


def test_route_feedback_mirrors_main_window_and_stops_timers_when_leaving(
    qapp, save_config_ok, open_setup
):
    config = _route_step_config()
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    dialog = open_setup(owner)
    input_meter, output_meter = dialog.view_model.input_meter, dialog.view_model.output_meter

    assert scene_item(dialog, "routePanel").isVisible()
    assert dialog._route_feedback_timer.isActive()
    assert input_meter.rms_db == processor.input_rms
    assert "Input: OK" in scene_text(dialog, "inputHealth")

    processor.input_peak = 0.0
    processor.diagnostics["clip_event_count"] = 1
    owner.input_meter.peak_db = processor.input_peak
    owner.shell_texts["input_health_label"].update(text="Input: CLIPPING", status="bad")
    dialog._update_route_feedback()
    chip = scene_item(dialog, "inputHealth")
    assert "Input: CLIPPING" in chip.property("text")
    assert chip.property("tone") == "bad"
    assert chip.property("color").name() == PALETTE.status_bad_surface

    dialog.resize(dialog.minimumSize())
    qapp.processEvents()
    qapp.processEvents()
    scroll = scene_item(dialog, "contentScroll")
    assert scroll.property("contentHeight") > scroll.height()
    action = scene_item(dialog, "actionButton")
    assert action.height() >= action.property("implicitHeight")
    assert action.mapToScene(action.boundingRect().bottomRight()).y() <= dialog.quick_view.height()
    dialog.hide()
    assert not dialog._route_feedback_timer.isActive()
    assert not input_meter.decay_timer.isActive()
    dialog.show()
    assert dialog._route_feedback_timer.isActive()

    press(dialog, "backButton")

    assert dialog.current_step == "devices"
    assert not scene_item(dialog, "routePanel").isVisible()
    assert not dialog._route_feedback_timer.isActive()
    assert not input_meter.decay_timer.isActive()
    assert not output_meter.decay_timer.isActive()


def test_resume_skipped_voice_does_not_enter_optional_latency(qapp, save_config_ok):
    config = AppConfig(
        first_run_setup_state="completed_with_skips",
        first_run_setup_step="voice",
        first_run_setup_steps={"devices": "completed", "route": "completed",
                               "latency": "pending", "voice": "skipped"},
    )
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()))
    assert dialog.current_step == "voice"
    assert config.first_run_setup_steps["latency"] == "pending"
    dialog.reject()


def test_route_feedback_shows_idle_levels_when_processor_stops(qapp, save_config_ok):
    config = _route_step_config()
    processor = _Processor(running=True)
    dialog = FirstRunSetupDialog(_Owner(config, processor))
    input_meter, output_meter = dialog.view_model.input_meter, dialog.view_model.output_meter

    processor.running = False
    dialog._update_route_feedback()

    assert input_meter.rms_db == -60.0
    assert output_meter.rms_db == -60.0
    assert not input_meter.measurement_available
    assert not output_meter.measurement_available
    assert scene_text(dialog, "inputHealth") == "Input: --"
    assert scene_text(dialog, "outputHealth") == "Output: --"
    assert scene_item(dialog, "inputHealth").property("tone") == "idle"


@pytest.mark.parametrize("entry_state", ["completed", "completed_with_skips"])
@pytest.mark.parametrize("how", ["reject", "pause"])
def test_closing_rerun_setup_restores_completed_state(
    qapp, monkeypatch, entry_state, how
):
    saved_states = []
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda config: saved_states.append(config.first_run_setup_state) or True,
    )
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.QMessageBox.information",
        lambda *_args: None,
    )
    outcome = "skipped" if entry_state == "completed_with_skips" else "completed"
    config = AppConfig(
        first_run_setup_state=entry_state,
        first_run_setup_steps={"devices": outcome, "route": "completed",
                               "latency": "pending", "voice": outcome},
    )
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()), restart_completed=True)
    saves_after_constructor = len(saved_states)
    dialog._skip_step()
    saves_after_step_change = len(saved_states)

    if how == "reject":
        dialog.reject()
    else:
        assert dialog._finish_setup() is True

    assert config.first_run_setup_state == entry_state
    assert 0 < saves_after_constructor < saves_after_step_change < len(saved_states)
    assert set(saved_states) == {entry_state}


def test_finishing_rerun_setup_ends_completed(qapp, monkeypatch):
    saved_states = []
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda config: saved_states.append(config.first_run_setup_state) or True,
    )
    config = AppConfig(first_run_setup_state="completed")
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()), restart_completed=True)
    for step in ("devices", "route", "voice"):
        config.first_run_setup_steps[step] = "completed"

    assert dialog._finish_setup() is True

    assert config.first_run_setup_state == "completed"
    assert dialog._finalized is True
    assert set(saved_states) == {"completed"}


def test_skipping_through_rerun_setup_ends_completed_with_skips(qapp, monkeypatch):
    saved_states = []
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda config: saved_states.append(config.first_run_setup_state) or True,
    )
    config = AppConfig(first_run_setup_state="completed")
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()), restart_completed=True)

    for _step in range(3):
        dialog._skip_step()

    assert config.first_run_setup_state == "completed_with_skips"
    assert dialog._finalized is True
    assert set(saved_states) <= {"completed", "completed_with_skips"}
    assert saved_states[0] == "completed"


@pytest.mark.parametrize("entry_state", ["not_started", "in_progress"])
def test_closing_first_run_setup_stays_in_progress(qapp, save_config_ok, entry_state):
    config = AppConfig(first_run_setup_state=entry_state)
    FirstRunSetupDialog(_Owner(config, _Processor())).reject()

    assert config.first_run_setup_state == "in_progress"


@pytest.mark.parametrize("how", ["reject", "pause"])
def test_rerun_resumes_paused_progress_from_settings(qapp, monkeypatch, how):
    saved_states = []
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda config: saved_states.append(config.first_run_setup_state) or True,
    )
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.QMessageBox.information",
        lambda *_args: None,
    )
    config = AppConfig(
        first_run_setup_state="completed",
        first_run_setup_steps={step: "completed" for step in FIRST_RUN_SETUP_STEPS},
    )
    owner = _Owner(config, _Processor())
    dialog = FirstRunSetupDialog(owner, restart_completed=True)
    config.first_run_setup_steps["devices"] = "completed"
    dialog._advance_or_finish()
    if how == "reject":
        dialog.reject()
    else:
        assert dialog._finish_setup() is True

    resumed = FirstRunSetupDialog(owner, restart_completed=True)

    assert config.first_run_setup_steps["devices"] == "completed"
    assert resumed.current_step == "route"
    assert set(saved_states) == {"completed"}


def test_fully_completed_setup_restarts_from_settings(qapp, save_config_ok):
    config = AppConfig(
        first_run_setup_state="completed",
        first_run_setup_step="voice",
        first_run_setup_steps={step: "completed" for step in FIRST_RUN_SETUP_STEPS},
    )
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()), restart_completed=True)

    assert dialog.current_step == "devices"
    assert set(config.first_run_setup_steps.values()) == {"pending"}


@pytest.mark.parametrize("entry_state", ["completed", "completed_with_skips"])
def test_failed_save_on_close_keeps_entry_state_in_memory(qapp, monkeypatch, entry_state):
    save_results = iter((True, False))
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda _config: next(save_results),
    )
    config = AppConfig(first_run_setup_state=entry_state)
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()), restart_completed=True)

    dialog.reject()

    assert config.first_run_setup_state == entry_state
    assert "could not be saved" in dialog.owner.status_message.showMessage.call_args.args[0]


@pytest.mark.parametrize(
    ("state", "steps", "prompts"),
    [
        ("not_started", {}, True),
        ("in_progress", {}, True),
        ("completed", {}, False),
        ("completed", {"devices": "pending"}, False),
        ("completed_with_skips", {"voice": "skipped"}, False),
        ("completed_with_skips", {"devices": "pending"}, False),
    ],
)
def test_launch_prompts_only_for_unfinished_first_run(
    monkeypatch, state, steps, prompts
):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUDIOFORGE_SMOKE_TEST", raising=False)
    shown = []
    # Unbound call on a minimal stand-in: MainWindow construction is heavy and
    # the method only reads config and calls _show_first_run_setup.
    window = SimpleNamespace(
        config=AppConfig(first_run_setup_state=state, first_run_setup_steps=steps),
        _show_first_run_setup=lambda **kwargs: shown.append(kwargs),
    )

    MainWindow._maybe_show_first_run_setup(window)  # type: ignore[arg-type]

    assert shown == ([{"restart_completed": False}] if prompts else [])


def test_rerun_failed_save_messages_do_not_promise_a_launch_prompt(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: False
    )
    config = AppConfig(first_run_setup_state="completed")
    dialog = FirstRunSetupDialog(_Owner(config, _Processor()), restart_completed=True)

    assert "launch" not in scene_text(dialog, "statusText")
    dialog.reject()

    assert "launch" not in dialog.owner.status_message.showMessage.call_args.args[0]


def test_onboarding_route_check_stops_for_user_mute(qapp, save_config_ok) -> None:
    class Owner(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.config = _route_step_config()
            self.user_muted = True
            self.start_calls = 0
            self.input_choice = QComboBox()
            self.input_choice.addItem(
                "Mic", DeviceIdentity(name="Mic", endpoint_id="in", direction="input")
            )
            self.output_choice = QComboBox()
            self.output_choice.addItem(
                "Cable",
                DeviceIdentity(name="Cable", endpoint_id="out", direction="output"),
            )
            self.processor = SimpleNamespace(is_running=lambda: False)

        def _start_processing(self) -> None:
            self.start_calls += 1

    owner = Owner()
    dialog = FirstRunSetupDialog(owner)
    try:
        dialog._run_current_step()
        assert owner.start_calls == 0
        assert "Output is muted" in scene_text(dialog, "statusText")
    finally:
        dialog.close()
        owner.deleteLater()
        qapp.processEvents()
