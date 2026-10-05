"""Quiet launch policy with real Qt selections and fake audio/OS boundaries."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QComboBox

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow


@pytest.fixture
def startup_owner(qapp, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(main_window.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(main_window.login_startup, "registration_state", lambda _: "configured")
    input_id = DeviceIdentity(name="Mic", endpoint_id="capture-A", direction="input")
    output_id = DeviceIdentity(name="Cable", endpoint_id="render-A", direction="output")
    config = AppConfig(last_input_device_identity=input_id, last_output_device_identity=output_id)
    owner: Any = SimpleNamespace(
        config=config, _login_startup=True, _login_startup_deadline=None,
        _login_startup_next_retry=0.0, _login_startup_message="",
        _login_startup_route=(input_id, output_id), _startup_restore_ready=True,
        _tray_icon=SimpleNamespace(isVisible=lambda: True), _output_mute_error=None,
        _temporary_mute_reasons=set(), _quitting=False,
        processor=SimpleNamespace(is_running=lambda: False),
        input_combo=QComboBox(), output_combo=QComboBox(), stop_btn=Mock(),
        status_bar=Mock(), _update_session_summary=Mock(), _sync_processing_controls=Mock(),
        _refresh_devices=Mock(), _apply_input_preferences_for_current_route=Mock(),
        _apply_latency_compensation_for_current_devices=Mock(),
        _apply_bound_preset_for_current_route=Mock(return_value=True),
        _current_device_route_key=Mock(return_value="saved-route"),
        _start_processing=Mock(return_value=True), _setup_desktop_integration=Mock(),
    )
    owner.input_combo.addItem("Mic", input_id)
    owner.output_combo.addItem("Cable", output_id)
    owner._combo_identities = MainWindow._combo_identities
    for name in ("begin_login_startup", "_service_login_startup", "_cancel_login_startup"):
        setattr(owner, name, MethodType(getattr(MainWindow, name), owner))
    return owner, clock


def test_login_starts_once_on_exact_saved_route(startup_owner):
    owner, clock = startup_owner
    owner.begin_login_startup()
    owner._service_login_startup()
    clock[0] += 20
    owner._service_login_startup()
    owner._start_processing.assert_called_once_with(interactive=False)
    assert owner._login_startup_deadline is None


def test_missing_endpoint_waits_without_selecting_same_name_replacement(startup_owner):
    owner, clock = startup_owner
    replacement = DeviceIdentity(name="Mic", endpoint_id="capture-B", direction="input")
    owner.input_combo.setItemData(0, replacement)
    owner.begin_login_startup()
    owner._start_processing.assert_not_called()
    assert owner.config.last_input_device_identity.endpoint_id == "capture-A"
    assert owner._login_startup_deadline is not None
    # A reconnect can change the friendly name and ordinal while retaining the endpoint.
    returned = DeviceIdentity(name="Renamed Mic", endpoint_id="capture-A", direction="input")
    owner.input_combo.setItemData(0, returned)
    clock[0] += 2
    owner._service_login_startup()
    owner._start_processing.assert_called_once_with(interactive=False)


@pytest.mark.parametrize("unsupported", ["missing_id", "restore_failed", "config_warning"])
def test_unsupported_restoration_stays_stopped(startup_owner, unsupported):
    owner, _ = startup_owner
    if unsupported == "missing_id":
        owner._login_startup_route[0].endpoint_id = ""
    elif unsupported == "restore_failed":
        owner._startup_restore_ready = False
    else:
        owner.config.load_warning = "Unreadable config"
    owner.begin_login_startup()
    owner._start_processing.assert_not_called()
    assert owner._login_startup_deadline is None
    assert "stopped" in owner._login_startup_message.lower()


def test_wait_is_bounded_and_does_not_retry_after_deadline(startup_owner):
    owner, clock = startup_owner
    owner.output_combo.clear()
    owner.begin_login_startup()
    clock[0] += 61
    owner._service_login_startup()
    count = owner._refresh_devices.call_count
    clock[0] += 61
    owner._service_login_startup()
    assert owner._refresh_devices.call_count == count
    owner._start_processing.assert_not_called()
    assert "stopped" in owner._login_startup_message.lower()


def test_stop_cancels_pending_start_even_when_audio_was_never_running(startup_owner):
    owner, clock = startup_owner
    owner.output_combo.clear()
    owner.begin_login_startup()
    MainWindow._stop_processing(owner)
    owner.output_combo.addItem("Cable", owner._login_startup_route[1])
    clock[0] += 2
    owner._service_login_startup()
    owner._start_processing.assert_not_called()
    assert owner._login_startup_deadline is None


def test_external_opt_out_cancels_pending_start(startup_owner, monkeypatch):
    owner, clock = startup_owner
    owner.output_combo.clear()
    owner.begin_login_startup()
    monkeypatch.setattr(main_window.login_startup, "registration_state", lambda _: "absent")
    owner.output_combo.addItem("Cable", owner._login_startup_route[1])
    clock[0] += 2
    owner._service_login_startup()
    owner._start_processing.assert_not_called()
    assert owner._login_startup_deadline is None


def test_no_tray_exits_without_starting_audio_or_showing_window(startup_owner, monkeypatch):
    owner, clock = startup_owner
    owner._tray_icon = None
    app = SimpleNamespace(exit=Mock())
    monkeypatch.setattr(main_window.QGuiApplication, "instance", lambda: app)
    owner.begin_login_startup()
    assert owner._login_startup_deadline is not None
    clock[0] += 61
    owner._service_login_startup()
    app.exit.assert_called_once_with(1)
    owner._start_processing.assert_not_called()
    assert "tray" in owner._login_startup_message.lower()


def test_route_edit_cancels_retry_before_persisting_the_explicit_new_route(startup_owner):
    owner, clock = startup_owner
    owner.output_combo.clear()
    owner.begin_login_startup()
    owner._combo_device_identity = MainWindow._combo_device_identity
    owner._device_name_from_identity = MainWindow._device_name_from_identity
    owner._save_config_safely = Mock(return_value=True)
    owner._sync_calibration_evidence = Mock()
    replacement = DeviceIdentity(name="Chosen Cable", endpoint_id="render-B", direction="output")
    owner.output_combo.addItem("Chosen Cable", replacement)
    MainWindow._on_device_changed(owner)
    clock[0] += 2
    owner._service_login_startup()
    owner._start_processing.assert_not_called()
    assert owner.config.last_output_device_identity.endpoint_id == "render-B"
    assert owner._login_startup_deadline is None


def test_failed_route_preset_cannot_start_audio(startup_owner):
    from mic_eq.config import DevicePresetBinding

    owner, _ = startup_owner
    owner.config.auto_apply_device_presets = True
    owner.config.device_preset_bindings["saved-route"] = DevicePresetBinding(preset_id="builtin:voice")
    owner._apply_bound_preset_for_current_route.return_value = False
    owner.begin_login_startup()
    owner._start_processing.assert_not_called()
    assert "preset" in owner._login_startup_message.lower()


@pytest.mark.parametrize("mute_failure", [None, "before", "after"])
def test_quiet_start_restores_mute_before_exact_route_and_stops_on_post_start_failure(
    startup_owner, monkeypatch, mute_failure
):
    owner, _ = startup_owner
    events = []
    owner._combo_device_identity = MainWindow._combo_device_identity
    owner._device_selection_to_name = lambda combo: combo.currentData().name
    owner.start_btn = Mock()
    owner.refresh_btn = Mock()
    owner._stream_recovery = Mock()
    owner._sync_meter_timer = Mock()

    def apply_mute():
        events.append("mute")
        if (mute_failure == "before" and len(events) == 1) or (
            mute_failure == "after" and len(events) == 3
        ):
            owner._output_mute_error = "unavailable"

    def start(*_args, **kwargs):
        events.append("start")
        assert kwargs["input_device_endpoint_id"] == "capture-A"
        assert kwargs["output_device_endpoint_id"] == "render-A"

    owner._apply_output_mute = apply_mute
    owner.processor.start = start
    owner.processor.stop = lambda: events.append("stop")
    modal = Mock(side_effect=AssertionError("Quiet start must not show a modal"))
    monkeypatch.setattr(main_window.QMessageBox, "critical", modal)
    assert MainWindow._start_processing(owner, interactive=False) == (mute_failure is None)
    assert events == {
        None: ["mute", "start", "mute"],
        "before": ["mute"],
        "after": ["mute", "start", "mute", "stop"],
    }[mute_failure]
    modal.assert_not_called()


def test_login_does_not_open_first_run_setup(startup_owner, monkeypatch):
    owner, _ = startup_owner
    owner._show_first_run_setup = Mock()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUDIOFORGE_SMOKE_TEST", raising=False)
    MainWindow._maybe_show_first_run_setup(owner)
    owner._show_first_run_setup.assert_not_called()


def test_missing_requested_startup_preset_is_not_authorized_by_last_used_fallback(monkeypatch):
    previous_key = next(iter(main_window.BUILTIN_PRESETS))
    owner: Any = SimpleNamespace(
        config=AppConfig(startup_preset="builtin:missing", last_preset=f"builtin:{previous_key}"),
        input_combo=Mock(), output_combo=Mock(), status_bar=Mock(),
        _apply_preset=Mock(return_value=True), _restore_ui_state=Mock(),
        _apply_latency_compensation_for_current_devices=Mock(),
        _apply_bound_preset_for_current_route=Mock(), _save_config_safely=Mock(return_value=True),
    )
    monkeypatch.setattr(main_window, "list_presets", lambda: [])
    MainWindow._restore_from_config(owner)
    owner._apply_preset.assert_called_once()
    assert not owner._startup_restore_ready


def test_error_after_native_start_stops_audio_on_quiet_path(startup_owner, monkeypatch):
    owner, _ = startup_owner
    running = [False]
    owner._combo_device_identity = MainWindow._combo_device_identity
    owner._device_selection_to_name = lambda combo: combo.currentData().name
    owner._apply_output_mute = Mock()
    owner.processor.is_running = lambda: running[0]
    owner.processor.start = lambda *_args, **_kwargs: running.__setitem__(0, True)
    owner.processor.stop = lambda: running.__setitem__(0, False)
    owner._apply_latency_compensation_for_current_devices.side_effect = RuntimeError("latency failed")
    monkeypatch.setattr(main_window.QMessageBox, "critical", Mock())
    assert not MainWindow._start_processing(owner, interactive=False)
    assert not running[0]


def test_manual_refresh_resumes_route_preset_behavior_after_login_finishes(startup_owner, monkeypatch):
    owner, _ = startup_owner
    owner.begin_login_startup()
    owner._current_device_route_key.side_effect = ["old-route", "new-route"]
    owner._current_capture_format_context = Mock(return_value=None)
    owner._sync_calibration_evidence = Mock()
    owner._save_config_safely = Mock(return_value=True)
    owner.device_warning_banner = Mock()
    for name in ("_combo_device_identity", "_identity_from_device_info",
                 "_find_combo_index_by_identity", "_default_combo_index",
                 "_preferred_output_combo_index"):
        setattr(owner, name, getattr(MainWindow, name))
    owner._select_combo_identity = MethodType(MainWindow._select_combo_identity, owner)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [owner._login_startup_route[0]])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [owner._login_startup_route[1]])
    MainWindow._refresh_devices(owner)
    owner._apply_bound_preset_for_current_route.assert_called_once()


def test_opt_out_does_not_report_running_audio_as_stopped(startup_owner, monkeypatch):
    owner, _ = startup_owner
    owner.processor.is_running = lambda: True
    owner._refresh_login_startup_action = Mock()
    monkeypatch.setattr(main_window.login_startup, "set_login_startup", Mock(return_value=True))
    MainWindow._on_login_startup_toggled(owner, False)
    messages = [str(call.args[0]).lower() for call in owner.status_bar.showMessage.call_args_list]
    assert not any("audio remains stopped" in message for message in messages)
