"""Tests for the resumable first-run setup flow."""

from __future__ import annotations

import logging

import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QCheckBox, QComboBox, QWidget

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.config_parts.app_config import FIRST_RUN_SETUP_STEPS
from mic_eq.ui.first_run_setup_dialog import (
    FirstRunSetupDialog,
    route_health_reason,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.shell_state import TextState
from mic_eq.ui.theme import status_chip_style


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


def test_setup_resumes_at_saved_step_and_delegates_route_check(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )
    owner = _Owner(config, _Processor())
    dialog = FirstRunSetupDialog(owner)

    assert dialog.current_step == "route"
    dialog._run_current_step()
    assert owner.start_calls == 1
    assert dialog._route_check_timer.isActive()
    dialog._finish_route_check()
    assert not dialog._route_check_timer.isActive()
    assert config.first_run_setup_steps["route"] == "pending"
    assert "Confirm signal" in dialog.action_button.text()
    dialog._run_current_step()
    assert config.first_run_setup_steps["route"] == "completed"
    assert dialog.current_step == "voice"


@pytest.mark.parametrize(
    ("running_before", "stop_calls"),
    [(False, 1), (True, 0)],
    ids=["started", "already_running"],
)
def test_closing_during_level_check_stops_only_processing_it_started(
    qapp, monkeypatch, running_before, stop_calls
):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )
    owner = _Owner(config, _Processor(running=running_before))
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()
    assert dialog._route_check_timer.isActive()
    dialog.reject()

    assert owner.stop_calls == stop_calls
    assert owner.processor.running is running_before


def test_setup_mute_control_tracks_external_owner_changes(qapp):
    owner = _Owner(AppConfig(), _Processor())
    dialog = FirstRunSetupDialog(owner)

    assert not dialog.mute_checkbox.isChecked()
    owner.user_mute_checkbox.setChecked(True)
    assert dialog.mute_checkbox.isChecked()
    owner.user_mute_checkbox.setChecked(False)
    assert not dialog.mute_checkbox.isChecked()


def test_setup_records_skips_and_can_resume_them(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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


def test_setup_applies_selected_route_from_wizard(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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

    dialog = FirstRunSetupDialog(owner)
    dialog.input_device_selector.setCurrentIndex(1)
    dialog.output_device_selector.setCurrentIndex(1)
    dialog._run_current_step()

    assert owner.input_choice.currentData() == input_desired
    assert owner.output_choice.currentData() == output_desired
    assert owner.device_change_calls == 1
    assert owner.device_change_routes == [(input_desired, output_desired)]
    assert config.first_run_setup_steps["devices"] == "completed"
    assert dialog.current_step == "route"


def test_setup_applies_snapshot_identities_after_owner_choices_reorder(qapp):
    owner = _Owner(AppConfig(), _Processor())
    selected_input = DeviceIdentity(name="Selected Mic", endpoint_id="mic-2", direction="input")
    selected_output = DeviceIdentity(name="Selected Cable", endpoint_id="cable-2", direction="output")
    owner.input_choice.addItem("Selected Mic", selected_input)
    owner.output_choice.addItem("Selected Cable", selected_output)
    dialog = FirstRunSetupDialog(owner)
    dialog.input_device_selector.setCurrentIndex(1)
    dialog.output_device_selector.setCurrentIndex(1)

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


def test_setup_rejects_snapshot_identity_removed_from_owner_choices(qapp):
    owner = _Owner(AppConfig(), _Processor())
    selected_input = DeviceIdentity(name="Selected Mic", endpoint_id="mic-2", direction="input")
    owner.input_choice.addItem("Selected Mic", selected_input)
    dialog = FirstRunSetupDialog(owner)
    dialog.input_device_selector.setCurrentIndex(1)
    dialog.output_device_selector.setCurrentIndex(1)

    owner.input_choice.clear()
    replacement = DeviceIdentity(name="Replacement", endpoint_id="mic-1", direction="input")
    owner.input_choice.addItem("Replacement", replacement)

    assert not dialog._apply_selected_devices()
    assert owner.input_choice.currentData() == replacement
    assert owner.device_change_calls == 0


def test_setup_restarts_processing_for_new_route(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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

    dialog = FirstRunSetupDialog(owner)
    dialog.input_device_selector.setCurrentIndex(1)
    dialog.output_device_selector.setCurrentIndex(1)
    dialog._run_current_step()

    assert owner.stop_calls == 1
    assert processor.is_running() is False
    dialog._run_current_step()
    assert owner.start_calls == 1
    assert dialog._route_check_timer.isActive()


def test_setup_keeps_same_running_route(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig()
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert owner.stop_calls == 0
    assert processor.is_running() is True


def test_setup_destination_confirmation_rechecks_mute_and_stream(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    dialog = FirstRunSetupDialog(owner)

    dialog._route_check_phase = "destination"
    owner.user_muted = True
    dialog._run_current_step()
    assert config.first_run_setup_steps["route"] == "pending"
    assert "Output is muted" in dialog.status_label.text()

    owner.user_muted = False
    processor.running = False
    dialog._run_current_step()
    assert dialog._route_check_phase == "speech"
    assert "did not start" in dialog.status_label.text()


def test_setup_ignores_late_route_timer_after_navigation(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )
    owner = _Owner(config, _Processor())
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()
    assert dialog._route_check_pending is True
    dialog._go_back()
    dialog._finish_route_check()

    assert dialog.current_step == "devices"
    assert config.first_run_setup_steps["route"] == "pending"
    assert "Confirm Destination" not in dialog.action_button.text()
    assert dialog.action_button.isEnabled()


def test_setup_missing_devices_stays_on_route_selection(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    owner = _Owner(AppConfig(), _Processor())
    owner.output_choice.clear()
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert dialog.current_step == "devices"
    assert owner.config.first_run_setup_steps["devices"] == "pending"
    assert "must be available" in dialog.status_label.text()


def test_setup_keeps_latency_optional_and_advanced(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig()
    owner = _Owner(config, _Processor())
    owner.latency_saved = True
    dialog = FirstRunSetupDialog(owner)

    dialog._skip_step()
    dialog._skip_step()
    assert dialog.current_step == "voice"
    assert config.first_run_setup_steps["latency"] == "pending"
    assert dialog.advanced_latency_button.isEnabled()

    dialog._run_latency_calibration()
    assert config.first_run_setup_steps["latency"] == "completed"
    assert dialog.current_step == "voice"


@pytest.mark.parametrize("step", ["latency", "voice"])
def test_a_failing_setup_step_is_logged_with_its_traceback(qapp, monkeypatch, caplog, step):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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
    dialog = FirstRunSetupDialog(owner)

    if step == "latency":
        dialog._run_latency_calibration()
    else:
        assert dialog.current_step == "voice"
        dialog._run_current_step()

    assert any(
        r.levelno >= logging.ERROR and r.exc_info and "could not be built" in str(r.exc_info[1])
        for r in caplog.records
    )
    assert dialog.status_label.text()


def test_setup_resumes_legacy_saved_latency_step(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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


def test_setup_mute_control_preserves_temporary_mute_owner(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    owner = _Owner(AppConfig(), _Processor())
    dialog = FirstRunSetupDialog(owner)

    dialog.mute_checkbox.setChecked(True)

    assert owner.user_muted is True
    assert owner.mute_toggles == [True]
    assert owner.user_mute_checkbox.isChecked()
    dialog.mute_checkbox.setChecked(False)
    assert owner.user_muted is False
    assert owner.mute_toggles == [True, False]
    assert not owner.user_mute_checkbox.isChecked()
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
    assert "Check speech" in dialog.title_label.text()
    assert owner.config.first_run_setup_steps["devices"] == "completed"
    assert dialog._progress_unsaved is True
    assert "Progress is unsaved" in dialog.status_label.text()


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
    assert "Progress is unsaved" in dialog.status_label.text()

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
    assert dialog.action_button.text() == "Retry save"
    assert owner.voice_setup_calls == 1
    assert "Progress is unsaved" in dialog.status_label.text()

    dialog._run_current_step()

    assert config.first_run_setup_state == "completed"
    assert dialog._finalized is True
    assert owner.voice_setup_calls == 1


def test_route_feedback_mirrors_main_window_and_stops_timers_when_leaving(
    qapp, monkeypatch
):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    dialog = FirstRunSetupDialog(owner)

    assert not dialog.route_feedback_group.isHidden()
    assert dialog._route_feedback_timer.isActive()
    assert dialog.route_input_meter.rms_db == processor.input_rms
    assert "Input: OK" in dialog.route_input_health_label.text()

    processor.input_peak = 0.0
    processor.diagnostics["clip_event_count"] = 1
    owner.input_meter.peak_db = processor.input_peak
    owner.shell_texts["input_health_label"].update(text="Input: CLIPPING", status="bad")
    dialog._update_route_feedback()
    assert "Input: CLIPPING" in dialog.route_input_health_label.text()
    assert dialog.route_input_health_label.styleSheet() == status_chip_style("bad")

    dialog.show()
    dialog.resize(440, 340)
    qapp.processEvents()
    scrollbar = dialog.content_scroll_area.verticalScrollBar()
    assert scrollbar is not None and scrollbar.maximum() > 0
    assert dialog.action_button.height() >= dialog.action_button.minimumSizeHint().height()
    dialog.hide()
    assert not dialog._route_feedback_timer.isActive()
    assert not dialog.route_input_meter.decay_timer.isActive()
    dialog.show()
    assert dialog._route_feedback_timer.isActive()

    dialog._go_back()

    assert dialog.current_step == "devices"
    assert dialog.route_feedback_group.isHidden()
    assert not dialog._route_feedback_timer.isActive()
    assert not dialog.route_input_meter.decay_timer.isActive()
    assert not dialog.route_output_meter.decay_timer.isActive()


def test_resume_skipped_voice_does_not_enter_optional_latency(qapp, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True)
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


def test_route_feedback_shows_idle_levels_when_processor_stops(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="route",
        first_run_setup_steps={
            "devices": "completed",
            "route": "pending",
            "latency": "pending",
            "voice": "pending",
        },
    )
    processor = _Processor(running=True)
    dialog = FirstRunSetupDialog(_Owner(config, processor))

    processor.running = False
    dialog._update_route_feedback()

    assert dialog.route_input_meter.rms_db == -60.0
    assert dialog.route_output_meter.rms_db == -60.0
    assert not dialog.route_input_meter.measurement_available
    assert not dialog.route_output_meter.measurement_available
    assert dialog.route_input_health_label.text() == "Input: --"
    assert dialog.route_output_health_label.text() == "Output: --"


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
def test_closing_first_run_setup_stays_in_progress(qapp, monkeypatch, entry_state):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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


def test_fully_completed_setup_restarts_from_settings(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )
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

    assert "launch" not in dialog.status_label.text()
    dialog.reject()

    assert "launch" not in dialog.owner.status_message.showMessage.call_args.args[0]


def test_onboarding_route_check_stops_for_user_mute(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True
    )

    class Owner(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.config = AppConfig(
                first_run_setup_state="in_progress",
                first_run_setup_step="route",
                first_run_setup_steps={
                    "devices": "completed",
                    "route": "pending",
                    "latency": "pending",
                    "voice": "pending",
                },
            )
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
        assert "Output is muted" in dialog.status_label.text()
    finally:
        dialog.close()
        owner.deleteLater()
        qapp.processEvents()
