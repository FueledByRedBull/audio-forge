"""Quiet launch policy on a real window with fake audio, device and clock boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QSystemTrayIcon

from mic_eq.config import AppConfig, DeviceIdentity, DevicePresetBinding, build_device_route_key
from mic_eq.ui import main_window
from mic_eq.ui.main_window import BUILTIN_PRESETS, MainWindow


def _endpoint(name, endpoint_id):
    """One record of the native device enumeration."""
    return SimpleNamespace(
        name=name, is_default=True, endpoint_id=endpoint_id, host_api="WASAPI",
        sample_rate=48_000, channels=2, name_ordinal=0,
    )


def _engine():
    """Audio engine stand-in that remembers whether it is running."""
    running = []
    engine = Mock()
    engine.is_running.side_effect = lambda: bool(running)
    engine.start.side_effect = lambda *_args, **_kwargs: running.append(True) or "started"
    engine.stop.side_effect = running.clear
    return engine


@pytest.fixture
def startup_window(qapp, monkeypatch):
    """A real login-launched window plus the clock and the device list it sees."""
    clock = [10.0]
    devices = {
        "input": [_endpoint("Mic", "capture-A")],
        "output": [_endpoint("Cable", "render-A")],
        "lookups": 0,
    }

    def list_outputs():
        devices["lookups"] += 1
        return devices["output"]

    config = AppConfig(
        last_input_device_identity=DeviceIdentity(
            name="Mic", endpoint_id="capture-A", direction="input"
        ),
        last_output_device_identity=DeviceIdentity(
            name="Cable", endpoint_id="render-A", direction="output"
        ),
    )
    monkeypatch.setattr(main_window, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(main_window.login_startup, "registration_state", lambda _: "configured")
    monkeypatch.setattr(main_window, "load_config", lambda: config)
    monkeypatch.setattr(main_window, "save_config", lambda _config: True)
    monkeypatch.setattr(main_window, "list_presets", lambda: [])
    monkeypatch.setattr(main_window, "list_input_devices", lambda: devices["input"])
    monkeypatch.setattr(main_window, "list_output_devices", list_outputs)
    window = MainWindow(login_startup=True)
    window.processor = _engine()
    window._tray_icon = QSystemTrayIcon(window)
    window._tray_icon.show()
    # Tests drive updates directly instead of through the event loop.
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    yield window, clock, devices
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    if window._tray_icon is not None:
        window._tray_icon.hide()
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_login_starts_once_on_exact_saved_route(startup_window):
    window, clock, _ = startup_window
    window.begin_login_startup()
    window._service_login_startup()
    clock[0] += 20
    window._service_login_startup()
    window.processor.start.assert_called_once()
    started = window.processor.start.call_args.kwargs
    assert started["input_device_endpoint_id"] == "capture-A"
    assert started["output_device_endpoint_id"] == "render-A"
    assert window._login_startup_deadline is None
    assert window.stop_btn_action.isEnabled() and not window.start_btn_action.isEnabled()


def test_missing_endpoint_waits_without_selecting_same_name_replacement(startup_window):
    window, clock, devices = startup_window
    devices["input"] = [_endpoint("Mic", "capture-B")]
    window.begin_login_startup()
    window.processor.start.assert_not_called()
    assert window.config.last_input_device_identity.endpoint_id == "capture-A"
    assert window._login_startup_deadline is not None
    # Waiting leaves Stop available to cancel the pending start.
    assert window.stop_btn_action.isEnabled()
    # A reconnect can change the friendly name and ordinal while retaining the endpoint.
    devices["input"] = [_endpoint("Renamed Mic", "capture-A")]
    clock[0] += 2
    window._service_login_startup()
    window.processor.start.assert_called_once()


@pytest.mark.parametrize("unsupported", ["missing_id", "restore_failed", "config_warning"])
def test_unsupported_restoration_stays_stopped(startup_window, unsupported):
    window, _, _ = startup_window
    if unsupported == "missing_id":
        window._login_startup_route[0].endpoint_id = ""
    elif unsupported == "restore_failed":
        window._startup_restore_ready = False
    else:
        window.config.load_warning = "Unreadable config"
    window.begin_login_startup()
    window.processor.start.assert_not_called()
    assert window._login_startup_deadline is None
    assert "stopped" in window._login_startup_message.lower()


def test_wait_is_bounded_and_does_not_retry_after_deadline(startup_window):
    window, clock, devices = startup_window
    devices["output"] = []
    window.begin_login_startup()
    clock[0] += 61
    window._service_login_startup()
    count = devices["lookups"]
    clock[0] += 61
    window._service_login_startup()
    assert devices["lookups"] == count
    window.processor.start.assert_not_called()
    assert "stopped" in window._login_startup_message.lower()


def test_stop_cancels_pending_start_even_when_audio_was_never_running(startup_window):
    window, clock, devices = startup_window
    devices["output"] = []
    window.begin_login_startup()
    window.stop_btn_action.trigger()
    devices["output"] = [_endpoint("Cable", "render-A")]
    clock[0] += 2
    window._service_login_startup()
    window.processor.start.assert_not_called()
    assert window._login_startup_deadline is None


def test_external_opt_out_cancels_pending_start(startup_window, monkeypatch):
    window, clock, devices = startup_window
    devices["output"] = []
    window.begin_login_startup()
    monkeypatch.setattr(main_window.login_startup, "registration_state", lambda _: "absent")
    devices["output"] = [_endpoint("Cable", "render-A")]
    clock[0] += 2
    window._service_login_startup()
    window.processor.start.assert_not_called()
    assert window._login_startup_deadline is None


def test_no_tray_exits_without_starting_audio_or_showing_window(startup_window, monkeypatch):
    window, clock, _ = startup_window
    window._tray_icon = None
    app_exit = Mock()
    monkeypatch.setattr(main_window.QGuiApplication, "exit", app_exit)
    window.begin_login_startup()
    assert window._login_startup_deadline is not None
    clock[0] += 61
    window._service_login_startup()
    app_exit.assert_called_once_with(1)
    window.processor.start.assert_not_called()
    assert not window.isVisible()
    assert "tray" in window._login_startup_message.lower()


def test_route_edit_cancels_retry_before_persisting_the_explicit_new_route(startup_window):
    window, clock, devices = startup_window
    devices["output"] = []
    window.begin_login_startup()
    devices["output"] = [_endpoint("Chosen Cable", "render-B")]
    window.refresh_btn_action.trigger()
    window.output_choice.setCurrentIndex(0)
    clock[0] += 2
    window._service_login_startup()
    window.processor.start.assert_not_called()
    assert window.config.last_output_device_identity.endpoint_id == "render-B"
    assert window._login_startup_deadline is None


def test_failed_route_preset_cannot_start_audio(startup_window):
    window, _, _ = startup_window
    window.config.auto_apply_device_presets = True
    route_key = window._current_device_route_key()
    assert route_key is not None
    window.config.device_preset_bindings[route_key] = DevicePresetBinding(
        preset_id="builtin:missing"
    )
    window.begin_login_startup()
    window.processor.start.assert_not_called()
    assert "preset" in window._login_startup_message.lower()


@pytest.mark.parametrize("mute_failure", [None, "before", "after"])
def test_quiet_start_restores_mute_before_exact_route_and_stops_on_post_start_failure(
    startup_window, monkeypatch, mute_failure
):
    window, _, _ = startup_window
    events = []
    started = []
    running = []

    def set_output_mute(_muted):
        events.append("mute")
        if (mute_failure == "before" and len(events) == 1) or (
            mute_failure == "after" and len(events) == 3
        ):
            raise OSError("unavailable")

    def start(*_args, **kwargs):
        events.append("start")
        started.append(kwargs)
        running.append(True)

    def stop():
        events.append("stop")
        running.clear()

    processor = Mock()
    processor.is_running.side_effect = lambda: bool(running)
    processor.set_output_mute.side_effect = set_output_mute
    processor.start.side_effect = start
    processor.stop.side_effect = stop
    window.processor = processor
    modal = Mock(side_effect=AssertionError("Quiet start must not show a modal"))
    monkeypatch.setattr(main_window.QMessageBox, "critical", modal)
    assert window._start_processing(interactive=False) == (mute_failure is None)
    assert events == {
        None: ["mute", "start", "mute"],
        "before": ["mute"],
        "after": ["mute", "start", "mute", "stop"],
    }[mute_failure]
    assert all(
        kwargs["input_device_endpoint_id"] == "capture-A"
        and kwargs["output_device_endpoint_id"] == "render-A"
        for kwargs in started
    )
    modal.assert_not_called()
    # Only a start that kept the mute state leaves Stop, not Start, on offer.
    assert window.stop_btn_action.isEnabled() == (mute_failure is None)
    assert window.start_btn_action.isEnabled() == (mute_failure is not None)


def test_login_does_not_open_first_run_setup(startup_window, monkeypatch):
    window, _, _ = startup_window
    window.config.first_run_setup_state = "not_started"
    show_setup = Mock()
    monkeypatch.setattr(window, "_show_first_run_setup", show_setup)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUDIOFORGE_SMOKE_TEST", raising=False)
    window._maybe_show_first_run_setup()
    show_setup.assert_not_called()


def test_missing_requested_startup_preset_is_not_authorized_by_last_used_fallback(startup_window):
    window, _, _ = startup_window
    previous_key = next(iter(BUILTIN_PRESETS))
    window.config.startup_preset = "builtin:missing"
    window.config.last_preset = f"builtin:{previous_key}"
    window._restore_from_config()
    assert window.current_preset_name == BUILTIN_PRESETS[previous_key].name
    assert not window._startup_restore_ready


def test_error_after_native_start_stops_audio_on_quiet_path(startup_window, monkeypatch):
    window, _, _ = startup_window
    monkeypatch.setattr(
        window, "_apply_latency_compensation_for_current_devices",
        Mock(side_effect=RuntimeError("latency failed")),
    )
    monkeypatch.setattr(main_window.QMessageBox, "critical", Mock())
    assert not window._start_processing(interactive=False)
    window.processor.start.assert_called_once()
    assert not window.processor.is_running()
    assert not window.stop_btn_action.isEnabled()


def test_manual_refresh_resumes_route_preset_behavior_after_login_finishes(startup_window):
    window, _, devices = startup_window
    window.begin_login_startup()
    window.stop_btn_action.trigger()
    other = _endpoint("Other Cable", "render-B")
    other_identity = MainWindow._identity_from_device_info(other, "output")
    window.config.last_output_device_identity = other_identity
    window.config.device_preset_bindings[
        build_device_route_key(
            MainWindow._identity_from_device_info(devices["input"][0], "input"), other_identity
        )
    ] = DevicePresetBinding(preset_id="builtin:voice")
    devices["output"] = [other]
    window.refresh_btn_action.trigger()
    assert window.current_preset_name == BUILTIN_PRESETS["voice"].name


def test_opt_out_does_not_report_running_audio_as_stopped(startup_window, monkeypatch):
    window, _, _ = startup_window
    window.processor.is_running.side_effect = lambda: True
    shown = []
    window.status_message.changed.connect(lambda: shown.append(window.status_message.text))
    monkeypatch.setattr(main_window.login_startup, "set_login_startup", Mock(return_value=True))
    window._on_login_startup_toggled(False)
    assert not any("audio remains stopped" in message.lower() for message in shown)


@pytest.fixture
def packaged_registry(startup_window, monkeypatch):
    """The login-shortcut menu of a frozen copy over a fake registration store."""
    window = startup_window[0]
    registry = {"state": "absent", "calls": [], "error": None}

    def set_login_startup(executable, enabled):
        registry["calls"].append((executable, enabled))
        if registry["error"] is not None:
            raise registry["error"]
        registry["state"] = "configured" if enabled else "absent"
        return True

    monkeypatch.setattr(main_window.login_startup, "registration_state", lambda _executable: registry["state"])
    monkeypatch.setattr(main_window.login_startup, "set_login_startup", set_login_startup)
    with monkeypatch.context() as frozen:
        frozen.setattr(sys, "frozen", True, raising=False)
        yield window, registry


@pytest.mark.parametrize(
    ("state", "checked", "enabled", "text"),
    [
        ("absent", False, True, "No login shortcut configured"),
        ("configured", True, True, "Shortcut configured; Windows may disable it"),
        ("other", False, False, "Login shortcut belongs to another copy or is unrecognized"),
        ("unreadable", False, False, "Login shortcut could not be read; left unchanged"),
    ],
)
def test_login_shortcut_menu_follows_the_registration_state(
    packaged_registry, state, checked, enabled, text
):
    window, registry = packaged_registry
    registry["state"] = state
    window._refresh_login_startup_action()
    assert window._login_startup_action.isChecked() is checked
    assert window._login_startup_action.isEnabled() is enabled
    assert window._login_startup_status_action.text() == text


def test_login_shortcut_menu_is_disabled_outside_the_packaged_executable(startup_window):
    window = startup_window[0]
    assert not hasattr(sys, "frozen")
    window._refresh_login_startup_action()
    assert not window._login_startup_action.isEnabled()
    assert window._login_startup_status_action.text() == "Login startup requires the packaged executable"


def test_ticking_the_login_shortcut_registers_this_copy(packaged_registry):
    window, registry = packaged_registry
    window._refresh_login_startup_action()
    window._login_startup_action.trigger()
    assert registry["calls"] == [(Path(sys.executable), True)]
    assert window._login_startup_action.isChecked()
    assert window._login_startup_status_action.text() == "Shortcut configured; Windows may disable it"


def test_failed_login_shortcut_change_is_reported_and_leaves_the_menu_unchanged(packaged_registry):
    window, registry = packaged_registry
    registry["error"] = PermissionError("Startup folder is not writable")
    window._refresh_login_startup_action()
    window._login_startup_action.trigger()
    assert registry["calls"] == [(Path(sys.executable), True)]
    assert window.status_message.text == "Startup folder is not writable"
    assert not window._login_startup_action.isChecked()
    assert window._login_startup_status_action.text() == "No login shortcut configured"

