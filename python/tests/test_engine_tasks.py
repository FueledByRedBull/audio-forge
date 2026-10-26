"""Start, Stop, recovery and device listing run on a worker: the window stays alive meanwhile."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from mic_eq.config import DeviceIdentity
from mic_eq.ui import main_window
from mic_eq.ui.engine_tasks import attempt


class _Panic(BaseException):
    """What a native panic looks like to Python: pyo3_runtime.PanicException is no Exception."""


class _SlowEngine:
    """The audio engine, faked: its calls wait for the test, and open no device."""

    def __init__(self, native: Any) -> None:
        self.native = native
        self.running = False
        self.calls: list[str] = []
        self.status_reads = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()  # a test that wants a call to hang clears this
        self.start_error: BaseException | None = None
        self.recovery_result: bool | None = None
        self.recovery_error: BaseException | None = None

    def __getattr__(self, name: str):
        return getattr(self.native, name)

    def lifecycle(self) -> list[str]:
        return [call for call in self.calls if call != "mute"]

    def _hang(self) -> None:
        self.entered.set()
        assert self.release.wait(10), "the test never released the engine"

    def start(self, *_args, **_kwargs) -> str:
        self.calls.append("start")
        self._hang()
        if self.running:
            raise RuntimeError("Already running")
        if self.start_error is not None:
            raise self.start_error
        self.running = True
        return "started"

    def stop(self) -> None:
        self.calls.append("stop")
        self._hang()
        self.running = False

    def service_recovery(self):
        self.calls.append("service_recovery")
        self._hang()
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovery_result

    def is_running(self) -> bool:
        self.status_reads += 1
        return self.running

    def set_output_mute(self, muted: bool) -> None:
        self.calls.append("mute")
        self.native.set_output_mute(muted)


def wait_until(qapp, condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Keep the event loop running until ``condition`` holds."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the engine worker"
        qapp.processEvents()
        time.sleep(0.005)


def run_loop_for(qapp, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)


@pytest.fixture
def engine_window(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    engine = _SlowEngine(window.processor)
    window.processor = engine
    window.input_choice.addItem("Microphone", DeviceIdentity(name="Microphone", direction="input"))
    window.output_choice.addItem("Destination", DeviceIdentity(name="Destination", direction="output"))
    monkeypatch.setattr(window, "_apply_input_preferences_for_current_route", lambda: None)
    yield window, engine
    engine.release.set()
    window._engine.wait()


class _Ticks:
    """Counts a 10 ms timer: it only advances while the event loop runs."""

    def __init__(self) -> None:
        self.count = 0
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self) -> None:
        self.count += 1


def test_start_returns_at_once_and_the_event_loop_runs_while_the_engine_starts(
    engine_window, qapp
):
    window, engine = engine_window
    engine.release.clear()
    ticks = _Ticks()
    # A start that blocked the GUI thread would only come back through this.
    rescue = threading.Timer(3.0, engine.release.set)
    rescue.start()
    try:
        began = time.monotonic()
        window.start_btn_action.trigger()
        assert time.monotonic() - began < 1.0, "Start held the GUI thread"
        assert engine.entered.wait(2.0)

        before = ticks.count
        run_loop_for(qapp, 0.25)
        assert ticks.count - before >= 5
        assert not window.start_btn_action.isEnabled() and not window.stop_btn_action.isEnabled()
        assert not window.input_choice.enabled and not window.refresh_btn_action.isEnabled()

        engine.release.set()
        wait_until(qapp, lambda: window.stop_btn_action.isEnabled())
        assert engine.running and not window.start_btn_action.isEnabled()
        assert window.status_message.text == "Processing: started"
    finally:
        rescue.cancel()
        ticks.timer.stop()


def test_a_second_click_while_the_engine_starts_does_not_start_it_twice(engine_window, qapp):
    window, engine = engine_window
    engine.release.clear()

    window.start_btn_action.trigger()
    assert engine.entered.wait(2.0)
    window.start_btn_action.trigger()
    window._on_start_clicked()  # the menu and its shortcut take the same way in
    run_loop_for(qapp, 0.1)
    engine.release.set()
    wait_until(qapp, lambda: window.stop_btn_action.isEnabled())

    assert engine.lifecycle() == ["start"]


def test_stop_runs_on_the_worker_and_locks_the_controls_until_it_is_done(engine_window, qapp):
    window, engine = engine_window
    window._start_processing()
    assert engine.running and window.stop_btn_action.isEnabled()
    engine.calls.clear()
    engine.release.clear()
    engine.entered.clear()
    ticks = _Ticks()

    window.stop_btn_action.trigger()
    assert engine.entered.wait(2.0)
    before = ticks.count
    run_loop_for(qapp, 0.2)
    assert ticks.count - before >= 5
    window.stop_btn_action.trigger()
    assert not window.stop_btn_action.isEnabled() and not window.start_btn_action.isEnabled()

    engine.release.set()
    wait_until(qapp, lambda: window.start_btn_action.isEnabled())
    ticks.timer.stop()
    assert engine.lifecycle() == ["stop"] and not engine.running
    assert window.status_message.text == "Processing stopped"
    assert window.input_choice.enabled and window.refresh_btn_action.isEnabled()
    assert not window.stop_btn_action.isEnabled()


def test_a_start_that_fails_is_reported_as_before(engine_window, qapp, monkeypatch):
    window, engine = engine_window
    engine.start_error = RuntimeError("audio device is busy")
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda _parent, title, text: shown.append(f"{title}: {text}")
    )

    window.start_btn_action.trigger()
    wait_until(qapp, lambda: bool(shown))

    assert shown[0].startswith("Error Starting Processing: Failed to start audio processing")
    assert "audio device is busy" in shown[0] and "Press the refresh button" in shown[0]
    assert window.status_message.text == "Could not start processing: audio device is busy"
    assert window.start_btn_action.isEnabled() and not window.stop_btn_action.isEnabled()
    assert window.input_choice.enabled and window.refresh_btn_action.isEnabled()
    window.start_btn_action.trigger()  # and it can be tried again
    wait_until(qapp, lambda: len(shown) == 2)


def test_refresh_lists_the_devices_on_the_worker(engine_window, qapp, monkeypatch):
    window, _engine = engine_window
    entered, release = threading.Event(), threading.Event()
    listed_on: list[threading.Thread] = []

    def list_inputs():
        listed_on.append(threading.current_thread())
        entered.set()
        assert release.wait(10)
        return [
            SimpleNamespace(
                name="Mic", is_default=True, endpoint_id="mic", host_api="WASAPI",
                sample_rate=48_000, channels=1, name_ordinal=0,
            )
        ]

    monkeypatch.setattr(main_window, "list_input_devices", list_inputs)
    ticks = _Ticks()
    try:
        window.refresh_btn_action.trigger()
        assert entered.wait(2.0)
        before = ticks.count
        run_loop_for(qapp, 0.2)
        assert ticks.count - before >= 5
        assert not window.refresh_btn_action.isEnabled() and not window.start_btn_action.isEnabled()
        window.refresh_btn_action.trigger()  # one listing at a time

        release.set()
        wait_until(qapp, lambda: window.refresh_btn_action.isEnabled())
    finally:
        release.set()
        ticks.timer.stop()
    assert len(listed_on) == 1 and listed_on[0] is not threading.main_thread()
    assert [window.input_choice.itemText(i) for i in range(window.input_choice.count())] == [
        "(Default) Mic"
    ]


def test_a_listing_that_fails_is_shown_on_the_device_banner(engine_window, qapp, monkeypatch):
    window, _engine = engine_window

    def fail():
        raise OSError("no audio service")

    monkeypatch.setattr(main_window, "list_input_devices", fail)

    window.refresh_btn_action.trigger()
    assert not window.refresh_btn_action.isEnabled()
    wait_until(qapp, lambda: window.refresh_btn_action.isEnabled())

    banner = window.shell_texts["device_warning_banner"]
    assert banner.shown and "Could not list input devices (no audio service)" in banner.text


def test_the_engine_recovery_runs_on_the_worker_one_call_at_a_time(engine_window, qapp):
    window, engine = engine_window
    window._start_processing()
    engine.calls.clear()
    engine.recovery_result = True
    engine.release.clear()
    engine.entered.clear()

    tick = window._service_stream_recovery

    began = time.monotonic()
    tick()
    tick()  # the first call is still out
    assert time.monotonic() - began < 1.0
    assert engine.entered.wait(2.0)
    run_loop_for(qapp, 0.1)
    assert window.stop_btn_action.isEnabled()  # a poll never locks the controls
    engine.release.set()
    wait_until(qapp, lambda: "restarted automatically" in window.status_message.text)

    assert engine.calls == ["service_recovery"]
    tick()
    wait_until(qapp, lambda: engine.calls == ["service_recovery"] * 2)


def test_a_stall_is_left_to_the_engines_own_recovery(engine_window, monkeypatch):
    """The supervisor in the engine is the only stall detector: a tick polls and restarts nothing."""
    window, engine = engine_window
    window._engine.threaded = False  # the fake engine answers at once
    clock = [1000.0]
    monkeypatch.setattr("mic_eq.ui.main_window.time.monotonic", lambda: clock[0])
    window._start_processing()
    # The stalls the window used to judge for itself, long past any warm-up or grace time:
    # output callbacks gone while the input lives, and a loud input with a silent output.
    engine.get_input_callback_age_ms = lambda: 10
    engine.get_output_callback_age_ms = lambda: 9_000
    engine.get_input_rms_db = lambda: -20.0
    engine.get_output_rms_db = lambda: -120.0
    engine.calls.clear()

    for _ in range(60):
        clock[0] += 1.0
        window._update_diagnostics()

    assert engine.calls == ["service_recovery"] * 60
    assert engine.running


def test_the_poll_shows_the_engines_back_off_and_adds_none_of_its_own(engine_window):
    window, engine = engine_window
    window._engine.threaded = False
    window._start_processing()
    engine.calls.clear()
    stall = "input_cb_age_ms=3100, output_cb_age_ms=12, output_write_age_ms=3200"
    engine.get_last_restart_reason = lambda: stall
    engine.get_last_stream_error = lambda: "Restart failed for selected devices: gone"
    retrying = (
        "The selected devices are not available, so AudioForge keeps retrying: "
        "Restart failed for selected devices: gone"
    )

    # What the engine answers after a stall: waiting out its back-off, a restart whose devices are
    # gone, the back-off again, then a restart that works. Every tick asks; none restarts itself.
    for answer, shown in (
        (None, "Processing: started"),
        (False, retrying),
        (None, retrying),
        (True, "Audio restarted automatically after: audio stalled; you may have heard a short gap"),
    ):
        engine.recovery_result = answer
        window._update_diagnostics()
        assert window.status_message.text == shown

    assert engine.calls == ["service_recovery"] * 4


def test_a_restart_names_its_cause_unless_it_was_a_stall(engine_window):
    window, engine = engine_window
    window._engine.threaded = False
    engine.recovery_result = True
    engine.get_last_restart_reason = lambda: "DSP thread panicked: boom"

    window._service_stream_recovery()

    assert window.status_message.text == (
        "Audio restarted automatically after: DSP thread panicked: boom; you may have heard a short gap"
    )


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize(
    "open_dialog",
    ["_on_latency_calibration_clicked", "_on_auto_eq_clicked", "_on_auto_voice_setup_clicked"],
)
def test_a_calibration_dialog_pauses_the_engines_recovery_while_it_is_open(
    engine_window, monkeypatch, open_dialog, before
):
    window, engine = engine_window
    monkeypatch.setattr(window, "_latency_profile_key", lambda: "route")
    monkeypatch.setattr(window, "_current_latency_profile", lambda: None)
    for name in ("LatencyCalibrationDialog", "CalibrationDialog", "VoiceSetupDialog"):
        monkeypatch.setattr(main_window, name, lambda *_args, **_kwargs: Mock())
    during: list[bool] = []

    def exec_dialog(_dialog) -> int:
        during.append(engine.is_recovery_suppressed())
        return 0

    monkeypatch.setattr(main_window, "_exec_and_release", exec_dialog)
    engine.set_recovery_suppressed(before)

    getattr(window, open_dialog)()

    assert during == [True]
    assert engine.is_recovery_suppressed() is before


def test_a_stop_during_a_recovery_poll_stays_stopped(engine_window, qapp):
    window, engine = engine_window
    window._start_processing()
    engine.calls.clear()
    engine.release.clear()
    engine.entered.clear()
    engine.recovery_result = True
    window._service_stream_recovery()
    assert engine.entered.wait(2.0)

    window.stop_btn_action.trigger()  # queued behind the poll on the one worker
    engine.release.set()
    wait_until(qapp, lambda: window.start_btn_action.isEnabled())
    engine.recovery_result = None  # the stop took the pending request with it
    window._update_diagnostics()
    window._engine.wait()
    run_loop_for(qapp, 0.1)

    assert engine.calls == ["service_recovery", "stop", "service_recovery"]
    assert not engine.running


def test_the_timers_leave_the_engine_alone_while_a_call_is_out(engine_window, qapp):
    window, engine = engine_window
    engine.release.clear()

    window.start_btn_action.trigger()
    assert engine.entered.wait(2.0)
    engine.status_reads = 0
    window._update_meters()
    window._update_diagnostics()
    window._sync_meter_timer()
    window._sync_processing_controls()
    reads = engine.status_reads
    engine.release.set()
    wait_until(qapp, lambda: window.stop_btn_action.isEnabled())

    assert reads == 0


def test_closing_during_a_start_waits_for_it_and_then_stops_the_audio(engine_window, qapp):
    window, engine = engine_window
    engine.release.clear()
    window.start_btn_action.trigger()
    assert engine.entered.wait(2.0)
    threading.Timer(0.3, engine.release.set).start()

    window.close()

    assert engine.lifecycle() == ["start", "stop"] and not engine.running


def test_a_click_that_runs_at_once_does_not_flicker_the_controls(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window
    window.input_choice.addItem("Microphone", DeviceIdentity(name="Microphone", direction="input"))
    window.output_choice.addItem("Destination", DeviceIdentity(name="Destination", direction="output"))
    monkeypatch.setattr(window, "_apply_input_preferences_for_current_route", lambda: None)
    window._engine.threaded = False  # the synchronous mode the controls must not flicker in
    changes: list[str] = []
    window.refresh_btn_action.changed.connect(lambda: changes.append("refresh"))
    window.start_btn_action.changed.connect(lambda: changes.append("start"))

    window.refresh_btn_action.trigger()

    # A button that is disabled and enabled again has lost the keyboard focus on the way.
    assert changes == []


def test_a_native_panic_is_an_error_but_an_interrupt_still_interrupts():
    def panic():
        raise _Panic("PyAudioProcessor is unsendable, but sent to another thread")

    failure = attempt(panic)
    assert isinstance(failure, RuntimeError) and "unsendable" in str(failure)
    assert isinstance(failure.__cause__, _Panic)

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        attempt(interrupt)


def test_a_panic_in_a_start_unlocks_the_controls_and_is_reported(engine_window, qapp, monkeypatch):
    window, engine = engine_window
    engine.start_error = _Panic("PyAudioProcessor is unsendable, but sent to another thread")
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, title, text: shown.append(text))

    window.start_btn_action.trigger()
    wait_until(qapp, lambda: bool(shown))

    assert "unsendable" in shown[0]
    assert window.start_btn_action.isEnabled() and window._engine_busy is None


def test_a_panic_in_the_recovery_poll_lets_the_next_tick_ask_again(engine_window, qapp):
    window, engine = engine_window
    window._start_processing()
    engine.recovery_error = _Panic("unsendable")

    tick = window._service_stream_recovery

    tick()
    wait_until(qapp, lambda: not window._recovery_polling)
    tick()
    wait_until(qapp, lambda: engine.calls.count("service_recovery") == 2)


def test_shutdown_does_not_stop_the_engine_beside_a_call_that_is_still_out(
    engine_window, qapp, monkeypatch
):
    window, engine = engine_window
    monkeypatch.setattr(main_window, "ENGINE_CALL_WAIT_MS", 100)
    window._start_processing()
    engine.calls.clear()
    engine.release.clear()
    engine.entered.clear()
    window.stop_btn_action.trigger()
    assert engine.entered.wait(2.0)

    began = time.monotonic()
    window._shutdown()
    assert time.monotonic() - began < 2.0

    assert engine.lifecycle() == ["stop"]  # not a second stop on this thread beside the first
    engine.release.set()
    window._engine.wait()
    run_loop_for(qapp, 0.1)
    assert window.status_message.text != "Processing stopped"  # the late answer is dropped


def test_the_late_answer_of_a_start_after_shutdown_is_dropped(engine_window, qapp, monkeypatch):
    window, engine = engine_window
    monkeypatch.setattr(main_window, "ENGINE_CALL_WAIT_MS", 100)
    engine.release.clear()
    window.start_btn_action.trigger()
    assert engine.entered.wait(2.0)

    window._shutdown()
    engine.release.set()
    window._engine.wait()
    run_loop_for(qapp, 0.1)

    assert window.status_message.text != "Processing: started"
    assert engine.lifecycle() == ["start"]


def test_a_recovery_poll_holds_the_getters_off_the_engine_while_it_is_out(engine_window, qapp):
    window, engine = engine_window
    window._start_processing()
    engine.release.clear()
    engine.entered.clear()
    window._service_stream_recovery()
    assert engine.entered.wait(2.0)

    engine.status_reads = 0
    window._update_meters()
    window._update_diagnostics()
    window._sync_meter_timer()
    reads = engine.status_reads
    engine.release.set()
    wait_until(qapp, lambda: not window._recovery_polling)

    assert reads == 0


@pytest.mark.parametrize("reopened", [True, False])
def test_a_start_click_during_a_recovery_poll_waits_for_its_answer(
    engine_window, qapp, monkeypatch, reopened
):
    window, engine = engine_window
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, _title, text: shown.append(text))
    engine.release.clear()
    window._service_stream_recovery()
    assert engine.entered.wait(2.0)

    window.start_btn_action.trigger()
    engine.running = reopened  # what the retry that is out did to the streams
    engine.recovery_result = reopened
    engine.release.set()
    wait_until(qapp, lambda: not window._recovery_polling and window._engine_busy is None)
    window._engine.wait()
    run_loop_for(qapp, 0.1)

    expected = ["service_recovery"] if reopened else ["service_recovery", "start"]
    assert engine.lifecycle() == expected
    assert engine.running and not shown

