"""Tests for the resumable first-run orchestration shell."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.ui.first_run_setup_dialog import (
    FirstRunSetupDialog,
    route_health_reason,
)


class _Combo:
    def __init__(self, value: object):
        self.value = value

    def currentData(self):
        return self.value


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
        self.input_combo = _Combo(
            DeviceIdentity(
                name="Mic",
                endpoint_id="input-id",
                direction="input",
                name_ordinal=0,
            )
        )
        self.output_combo = _Combo(
            DeviceIdentity(
                name="Cable",
                endpoint_id="output-id",
                direction="output",
                name_ordinal=0,
            )
        )
        self.latency_saved = False
        self.voice_applied = False
        self.start_calls = 0

    def _start_processing(self):
        self.start_calls += 1
        self.processor.running = True

    def _on_latency_calibration_clicked(self):
        return self.latency_saved

    def _on_auto_voice_setup_clicked(self):
        return self.voice_applied


def test_route_health_requires_running_recent_error_free_callbacks():
    processor = _Processor(running=False)
    assert route_health_reason(processor)[0] is False

    processor.running = True
    assert route_health_reason(processor)[0] is True

    processor.age_ms = 2_100.0
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "stale" in reason

    processor.age_ms = 4.0
    processor.diagnostics["output_callback_error_count"] = 1
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "reported an error" in reason

    processor.diagnostics["output_callback_error_count"] = "invalid"
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "diagnostics were invalid" in reason


def test_route_health_fails_closed_without_valid_callback_heartbeats():
    class MissingHeartbeatProcessor:
        def is_running(self):
            return True

        def get_runtime_diagnostics(self):
            return {}

    healthy, reason = route_health_reason(MissingHeartbeatProcessor())
    assert healthy is False
    assert "heartbeat is unavailable" in reason

    processor = _Processor(running=True, age_ms=float("nan"))
    healthy, reason = route_health_reason(processor)
    assert healthy is False
    assert "heartbeat is invalid" in reason


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
    assert config.first_run_setup_steps["route"] == "completed"
    assert dialog.current_step == "latency"


def test_setup_records_skips_honestly_and_can_resume_them(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
    )
    config = AppConfig()
    owner = _Owner(config, _Processor(running=True))
    dialog = FirstRunSetupDialog(owner)

    for _step in range(4):
        dialog._skip_step()

    assert config.first_run_setup_state == "completed_with_skips"
    assert set(config.first_run_setup_steps.values()) == {"skipped"}

    resumed = FirstRunSetupDialog(owner)
    assert resumed.current_step == "devices"
    assert set(config.first_run_setup_steps.values()) == {"pending"}


def test_setup_missing_devices_stays_on_route_selection(qapp, monkeypatch):
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
    )
    owner = _Owner(AppConfig(), _Processor())
    owner.output_combo.value = None
    dialog = FirstRunSetupDialog(owner)

    dialog._run_current_step()

    assert dialog.current_step == "devices"
    assert owner.config.first_run_setup_steps["devices"] == "pending"
    assert "must be available" in dialog.status_label.text()
