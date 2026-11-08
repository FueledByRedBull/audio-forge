"""On Linux AudioForge never records its own microphone.

With no other microphone the session manager makes AudioForge Mic the system default, and an
input that follows the default would record AudioForge's own output in a loop. Start refuses, and
a running input stops once; the engine's own restart is never left to reopen it.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtWidgets import QMessageBox

from mic_eq import mic_eq_core
from mic_eq.config import DeviceIdentity
from mic_eq.ui import capture_session, device_selection, main_window
from mic_eq.ui.device_selection import (
    FOLLOW_DEFAULT_INPUT_ENDPOINT,
    OWN_MICROPHONE_MESSAGE,
    OwnMicrophoneError,
    records_own_microphone,
)
from mic_eq.ui.engine_session import AudioUnavailable

FOLLOW_DEFAULT = DeviceIdentity(
    name="Follow system default",
    is_default=True,
    endpoint_id=FOLLOW_DEFAULT_INPUT_ENDPOINT,
    direction="input",
)
USB_MIC = DeviceIdentity(name="USB Microphone", endpoint_id="pipewire:alsa_input.usb", direction="input")
DESTINATION = DeviceIdentity(
    name="AudioForge Mic Input", endpoint_id="pipewire:audioforge_mic_input", direction="output"
)


class _Engine:
    """The audio engine, faked: it opens no device and answers at once.

    Its restart does what the native one does: a pending request reopens the streams.
    """

    def __init__(self, native: Any) -> None:
        self.native = native
        self.running = False
        self.restart_requested = False
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self.native, name)

    def start(self, *_args, **_kwargs) -> str:
        self.calls.append("start")
        self.running = True
        return "started"

    def stop(self) -> None:
        self.calls.append("stop")
        self.running = False
        self.restart_requested = False  # the native stop takes the pending request with it

    def is_running(self) -> bool:
        return self.running

    def is_recovery_requested(self) -> bool:
        return self.restart_requested

    def service_recovery(self) -> bool | None:
        self.calls.append("service_recovery")
        if not self.restart_requested:
            return None
        self.restart_requested = False
        self.running = True
        self.calls.append("reopened")
        return True

    def lifecycle(self) -> list[str]:
        return [call for call in self.calls if call != "service_recovery"]


def _platform(monkeypatch, name: str, default_is_own: bool = False) -> SimpleNamespace:
    """A desktop of the given platform whose default microphone is, or is not, AudioForge Mic."""
    desktop = SimpleNamespace(default_is_own=default_is_own, queries=0, watch_failed=False)

    def query() -> bool:
        desktop.queries += 1
        return desktop.default_is_own

    monkeypatch.setattr(device_selection, "sys", SimpleNamespace(platform=name))
    monkeypatch.setattr(mic_eq_core, "default_source_is_virtual_microphone", query, raising=False)
    monkeypatch.setattr(
        mic_eq_core, "default_source_watch_failed", lambda: desktop.watch_failed, raising=False
    )
    return desktop


@pytest.fixture
def desktop(monkeypatch) -> SimpleNamespace:
    return _platform(monkeypatch, "linux")


@pytest.fixture(autouse=True)
def no_refusals_yet(monkeypatch) -> None:
    """The count is the session's, kept in the module."""
    monkeypatch.setattr(device_selection, "_refusals", 0)


def _refusals() -> int:
    return device_selection.own_microphone_diagnostics()["own_microphone_refusal_count"]  # type: ignore[return-value]


@pytest.fixture
def window(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    engine = _Engine(window.processor)
    window.processor = window.engine_session.processor = engine
    window.input_choice.addItem(FOLLOW_DEFAULT.name, FOLLOW_DEFAULT)
    window.input_choice.addItem(USB_MIC.name, USB_MIC)
    window.output_choice.addItem(DESTINATION.name, DESTINATION)
    monkeypatch.setattr(window, "_apply_input_preferences_for_current_route", lambda: None)
    window.dialogs = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda _parent, title, text: window.dialogs.append((title, text))
    )
    return window, engine


def _banner(window: Any):
    return window.shell_texts["config_warning_banner"]


def _start(window: Any, how: str) -> None:
    if how == "button":
        window._on_start_clicked()
    else:
        window.engine_session.start_now(interactive=how == "setup")


def _tick(window: Any, times: int = 1) -> None:
    for _ in range(times):
        window._update_diagnostics()


# ---- which inputs would record AudioForge Mic ------------------------------------------------


@pytest.mark.parametrize(
    ("input_device", "default_is_own", "expected"),
    [
        (FOLLOW_DEFAULT, True, True),
        (None, True, True),  # no identity at all starts the default input
        (FOLLOW_DEFAULT, False, False),
        (USB_MIC, True, False),  # a saved specific microphone never follows the default
        (USB_MIC.to_dict(), True, False),
    ],
)
def test_only_an_input_that_follows_the_default_can_record_audioforge_mic(
    monkeypatch, input_device, default_is_own, expected
) -> None:
    _platform(monkeypatch, "linux", default_is_own)

    assert records_own_microphone(input_device) is expected


def test_other_platforms_never_ask_the_native_core(monkeypatch) -> None:
    desktop = _platform(monkeypatch, "win32", default_is_own=True)

    assert records_own_microphone(FOLLOW_DEFAULT) is False
    assert records_own_microphone(None) is False
    assert desktop.queries == 0


# ---- Start -----------------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["button", "setup", "login"])
def test_start_refuses_a_follow_default_input_that_would_record_audioforge_mic(
    window, desktop, how
) -> None:
    window, engine = window
    desktop.default_is_own = True

    _start(window, how)

    assert "start" not in engine.calls and not engine.running
    assert window.status_message.text == f"Could not start processing: {OWN_MICROPHONE_MESSAGE}"
    # The usual start-failure path: a dialog for a person who pressed Start, none at login.
    if how == "login":
        assert window.dialogs == []
    else:
        # The message alone: advice to refresh the Output list cannot help here.
        assert window.dialogs == [("Error Starting Processing", OWN_MICROPHONE_MESSAGE)]
    # The notice stays on the banner after the dialog is gone.
    assert _banner(window).shown and OWN_MICROPHONE_MESSAGE in _banner(window).text
    # Stopped and consistent: Start can be pressed again once a microphone is back.
    assert window.start_btn_action.isEnabled() and not window.stop_btn_action.isEnabled()
    assert window.input_choice.enabled and window.output_choice.enabled
    assert window._processing is False
    assert _refusals() == 1


def test_another_failed_start_still_gets_the_generic_advice(window, desktop) -> None:
    window, engine = window
    engine.start = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("audio device is busy")
    )

    window.engine_session.start_now()

    (title, text), = window.dialogs
    assert text.startswith("Failed to start audio processing:\n\naudio device is busy")
    assert "Try these steps" in text
    assert not _banner(window).shown and _refusals() == 0


def test_start_works_again_once_a_real_microphone_returns(window, desktop) -> None:
    window, engine = window
    desktop.default_is_own = True
    window.engine_session.start_now()
    assert not engine.running

    desktop.default_is_own = False
    assert window.engine_session.start_now() is True

    assert engine.running and engine.lifecycle() == ["start"]
    assert window.stop_btn_action.isEnabled()


@pytest.mark.parametrize("how", ["button", "setup", "login"])
def test_a_saved_specific_microphone_still_starts_while_audioforge_mic_is_the_default(
    window, desktop, how
) -> None:
    window, engine = window
    window.input_choice.setCurrentIndex(1)
    desktop.default_is_own = True

    _start(window, how)

    assert engine.running and engine.lifecycle() == ["start"]
    assert window.dialogs == [] and _refusals() == 0


def test_start_is_unaffected_off_linux(window, monkeypatch) -> None:
    window, engine = window
    desktop = _platform(monkeypatch, "win32", default_is_own=True)

    assert window.engine_session.start_now() is True

    assert engine.running and window.dialogs == [] and desktop.queries == 0


def test_voice_setup_and_calibration_starts_refuse_the_same_input(window, desktop) -> None:
    """The dialogs open their streams through the same route start."""
    window, engine = window
    desktop.default_is_own = True

    with pytest.raises(AudioUnavailable) as refused:
        window.engine_session.borrow("auto_eq_calibration")
    assert isinstance(refused.value.__cause__, OwnMicrophoneError)
    assert engine.calls == []
    assert _refusals() == 1  # the dialogs' refusals are counted too

    # A switch to that input fails and puts the running route back.
    with pytest.raises(RuntimeError, match="previous route restored") as raised:
        capture_session.restart_processor_for_route(
            engine, (FOLLOW_DEFAULT, DESTINATION), (USB_MIC, DESTINATION)
        )
    assert OWN_MICROPHONE_MESSAGE in str(raised.value)
    assert engine.lifecycle() == ["stop", "start"] and engine.running
    assert _refusals() == 2


def test_a_start_by_a_dialog_clears_the_notice_of_an_earlier_refusal(window, desktop) -> None:
    """Auto-EQ, Voice Setup and latency calibration start the engine without the window."""
    window, engine = window
    desktop.default_is_own = True
    window.engine_session.start_now()
    assert _banner(window).shown

    desktop.default_is_own = False
    window.engine_session.borrow("auto_voice_setup")

    assert engine.running and not _banner(window).shown


# ---- while running ---------------------------------------------------------------------------


def _running_on_the_default(window: Any, engine: _Engine, desktop: SimpleNamespace) -> None:
    window.engine_session.start_now()
    assert engine.running
    engine.calls.clear()
    assert desktop.default_is_own is False


def test_a_running_input_stops_when_the_default_becomes_audioforge_mic(
    window, desktop, caplog
) -> None:
    window, engine = window
    _running_on_the_default(window, engine, desktop)
    _tick(window)  # a healthy tick leaves it alone
    assert engine.running

    desktop.default_is_own = True  # the last real microphone was unplugged
    with caplog.at_level("WARNING"):
        _tick(window)

    assert not engine.running and engine.lifecycle() == ["stop"]
    assert window.status_message.text == OWN_MICROPHONE_MESSAGE
    assert _banner(window).shown and OWN_MICROPHONE_MESSAGE in _banner(window).text
    assert window.start_btn_action.isEnabled() and not window.stop_btn_action.isEnabled()
    assert window.input_choice.enabled
    assert _refusals() == 1
    assert [r.getMessage() for r in caplog.records if "Stopping audio" in r.getMessage()] == [
        f"Stopping audio: {OWN_MICROPHONE_MESSAGE}"
    ]


def test_the_stop_is_reported_once_and_never_restarts_by_itself(window, desktop, caplog) -> None:
    window, engine = window
    _running_on_the_default(window, engine, desktop)
    desktop.default_is_own = True

    with caplog.at_level("WARNING"):
        _tick(window, 40)  # ten seconds of polling while no microphone exists

    assert engine.lifecycle() == ["stop"] and not engine.running
    assert sum("Stopping audio" in r.getMessage() for r in caplog.records) == 1
    assert _refusals() == 1
    assert window.dialogs == []


def test_a_slow_stop_is_not_repeated_while_the_engine_is_still_stopping(
    window, desktop, qapp
) -> None:
    window, engine = window
    _running_on_the_default(window, engine, desktop)
    window._engine.threaded = True
    gate = threading.Event()

    def slow_stop() -> None:
        engine.calls.append("stop")
        assert gate.wait(10), "the test never released the engine"
        engine.running = False

    engine.stop = slow_stop  # type: ignore[method-assign]
    desktop.default_is_own = True
    try:
        _tick(window, 6)  # the first asks for the stop; the rest find the engine busy
        gate.set()
        deadline = time.monotonic() + 5
        while not window.start_btn_action.isEnabled():
            assert time.monotonic() < deadline, "the stop never finished"
            qapp.processEvents()
            time.sleep(0.005)
    finally:
        gate.set()
        window._engine.wait()

    assert engine.lifecycle() == ["stop"] and _refusals() == 1


@pytest.mark.parametrize("running", [True, False], ids=["engine-running", "failed-restart-left-streams-closed"])
def test_the_engines_own_restart_cannot_reopen_the_loop(window, desktop, running) -> None:
    """A pulled USB microphone makes the engine ask for a restart, which would follow the default."""
    window, engine = window
    _running_on_the_default(window, engine, desktop)
    engine.running = running
    engine.restart_requested = True
    desktop.default_is_own = True

    _tick(window, 8)

    assert "reopened" not in engine.calls and not engine.running
    # The request was cancelled by the stop; it was never serviced into the loop.
    assert engine.lifecycle() == ["stop"] and not engine.restart_requested
    assert window.status_message.text == OWN_MICROPHONE_MESSAGE


def test_without_the_condition_the_restart_service_runs_as_before(window, desktop) -> None:
    window, engine = window
    _running_on_the_default(window, engine, desktop)
    engine.restart_requested = True

    _tick(window)

    assert "reopened" in engine.calls and engine.running
    assert window.status_message.text.startswith("Audio restarted automatically")


def test_a_real_microphone_returning_does_not_restart_the_audio(window, desktop) -> None:
    window, engine = window
    _running_on_the_default(window, engine, desktop)
    desktop.default_is_own = True
    _tick(window)

    desktop.default_is_own = False  # WirePlumber moved the default back
    _tick(window, 8)

    assert not engine.running and engine.lifecycle() == ["stop"]
    assert _banner(window).shown  # still says why, until the person presses Start

    window.engine_session.start_now()
    assert engine.running and not _banner(window).shown


def test_a_saved_specific_microphone_keeps_running_when_audioforge_mic_is_the_default(
    window, desktop
) -> None:
    window, engine = window
    window.input_choice.setCurrentIndex(1)
    _running_on_the_default(window, engine, desktop)
    desktop.default_is_own = True

    _tick(window, 8)

    assert engine.running and engine.lifecycle() == []
    assert _refusals() == 0


def test_a_running_input_is_left_alone_off_linux(window, monkeypatch) -> None:
    window, engine = window
    desktop = _platform(monkeypatch, "win32", default_is_own=True)
    window.engine_session.start_now()
    engine.calls.clear()

    _tick(window, 8)

    assert engine.running and engine.lifecycle() == [] and desktop.queries == 0
    assert not _banner(window).shown


# ---- diagnostics -----------------------------------------------------------------------------


def _exported(window: Any, tmp_path, monkeypatch) -> dict[str, Any]:
    destination = tmp_path / "support.json"
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "JSON Files (*.json)"),
    )
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *_args, **_kwargs: None)
    window._export_diagnostics()
    assert not window.dialogs, window.dialogs
    return json.loads(destination.read_text(encoding="utf-8"))["runtime"]


def test_the_export_counts_every_refusal_and_stop(window, desktop, tmp_path, monkeypatch) -> None:
    window, engine = window
    assert _exported(window, tmp_path, monkeypatch)["own_microphone_refusal_count"] == 0

    desktop.default_is_own = True
    window.engine_session.start_now(interactive=False)  # refused at Start
    with pytest.raises(AudioUnavailable):
        window.engine_session.borrow("latency_calibration")  # refused for a dialog
    desktop.default_is_own = False
    window.engine_session.start_now(interactive=False)
    desktop.default_is_own = True
    _tick(window)  # stopped while running

    assert _exported(window, tmp_path, monkeypatch)["own_microphone_refusal_count"] == 3


@pytest.mark.parametrize("watch_failed", [False, True])
def test_the_export_says_whether_the_default_microphone_can_be_watched(
    window, desktop, tmp_path, monkeypatch, watch_failed
) -> None:
    """When PipeWire refuses the metadata binding the guard is off, and a report must show it."""
    window, _engine = window
    desktop.watch_failed = watch_failed

    runtime = _exported(window, tmp_path, monkeypatch)

    assert runtime["default_source_watch_failed"] is watch_failed


def test_the_export_has_no_such_fields_off_linux(window, tmp_path, monkeypatch) -> None:
    window, _engine = window
    _platform(monkeypatch, "win32")

    runtime = _exported(window, tmp_path, monkeypatch)

    assert "own_microphone_refusal_count" not in runtime
    assert "default_source_watch_failed" not in runtime
