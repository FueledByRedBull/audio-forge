"""Tests for the resumable first-run setup flow."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QWidget

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.ui.first_run_setup_dialog import (
    FirstRunSetupDialog,
    route_health_reason,
)


class _Processor:
    def __init__(self, *, running: bool = False, age_ms: float = 4.0):
        self.running = running
        self.age_ms = age_ms
        self.diagnostics: dict[str, object] = {}

    def is_running(self):
        return self.running

    def get_runtime_diagnostics(self):
        return dict(self.diagnostics)

    def get_input_callback_age_ms(self):
        return self.age_ms

    def get_output_callback_age_ms(self):
        return self.age_ms


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
        self.input_combo = QComboBox()
        self.input_combo.addItem(input_identity.name, input_identity)
        output_identity = DeviceIdentity(
            name="Cable",
            endpoint_id="output-id",
            direction="output",
            name_ordinal=0,
        )
        self.output_combo = QComboBox()
        self.output_combo.addItem(output_identity.name, output_identity)
        self.latency_saved = False
        self.voice_applied = False
        self.user_muted = False
        self._temporary_mute_reasons = {"calibration"}
        self.mute_toggles = []
        self.user_mute_checkbox = QCheckBox(self)
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
            (self.input_combo.currentData(), self.output_combo.currentData())
        )

    def _on_latency_calibration_clicked(self):
        return self.latency_saved

    def _on_auto_voice_setup_clicked(self):
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


def test_setup_resumes_at_saved_step_and_delegates_route_check(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
    assert "Confirm Destination Signal" in dialog.action_button.text()
    dialog._run_current_step()
    assert config.first_run_setup_steps["route"] == "completed"
    assert dialog.current_step == "voice"


def test_setup_records_skips_and_can_resume_them(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
    owner.input_combo.addItem(input_desired.name, input_desired)
    owner.output_combo.addItem(output_desired.name, output_desired)

    dialog = FirstRunSetupDialog(owner)
    dialog.input_device_selector.setCurrentIndex(1)
    dialog.output_device_selector.setCurrentIndex(1)
    dialog._run_current_step()

    assert owner.input_combo.currentData() == input_desired
    assert owner.output_combo.currentData() == output_desired
    assert owner.device_change_calls == 1
    assert owner.device_change_routes == [(input_desired, output_desired)]
    assert config.first_run_setup_steps["devices"] == "completed"
    assert dialog.current_step == "route"


def test_setup_restarts_processing_for_new_route(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
    )
    config = AppConfig()
    processor = _Processor(running=True)
    owner = _Owner(config, processor)
    owner.input_combo.addItem(
        "USB Mic",
        DeviceIdentity(
            name="USB Mic",
            endpoint_id="usb-input-id",
            direction="input",
            name_ordinal=0,
        ),
    )
    owner.output_combo.addItem(
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
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
    )
    owner = _Owner(AppConfig(), _Processor())
    owner.output_combo.clear()
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert dialog.current_step == "devices"
    assert owner.config.first_run_setup_steps["devices"] == "pending"
    assert "must be available" in dialog.status_label.text()


def test_setup_keeps_latency_optional_and_advanced(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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


def test_setup_resumes_legacy_saved_latency_step(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
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
