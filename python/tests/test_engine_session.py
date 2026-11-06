"""The engine session at its interface: Start/Stop races, recovery, mute owners, leases, quit."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import logging

import pytest

from mic_eq.config import DeviceIdentity
from mic_eq.ui.engine_session import (
    START_AFTER_STOP_LOCKOUT_S,
    AudioBorrowed,
    AudioReleased,
    AudioUnavailable,
    BackgroundStopFailed,
    ControlsChanged,
    EngineSession,
    MuteApplied,
    RestartFailed,
    Restarted,
    RouteMismatch,
    SessionView,
    Started,
    StartFailed,
    StopFailed,
    Stopped,
)
from engine_fakes import DESTINATION, MIC, FakeEngine, FakePreparation, ManualClock, ManualTasks


class _Panic(BaseException):
    """What a native panic looks like to Python: pyo3_runtime.PanicException is no Exception."""


class Rig:
    def __init__(self) -> None:
        self.engine = FakeEngine()
        self.tasks = ManualTasks()
        self.clock = ManualClock()
        self.prepare = FakePreparation()
        self.prepare.engine = self.engine
        self.session = EngineSession(self.engine, self.tasks, self.prepare, clock=self.clock)
        self.views: list[SessionView] = []
        self.session.changed.connect(self.views.append)

    def events(self, kind: type | None = None) -> list:
        return [view.event for view in self.views if kind is None or isinstance(view.event, kind)]

    def running(self) -> None:
        """The audio runs, started from the window; the record starts after it."""
        assert self.session.start_now()
        self.engine.calls.clear()
        self.views.clear()


@pytest.fixture
def rig(qapp) -> Rig:
    return Rig()


# --- Start and Stop --------------------------------------------------------------------------


def test_a_start_runs_on_the_worker_with_the_controls_locked_and_the_mute_on_both_sides(rig):
    rig.session.set_user_muted(True)  # the persisted mute
    rig.engine.calls.clear()
    rig.session.request_start("button")

    assert rig.session.view.busy == "start" and rig.engine.calls == ["mute"]
    rig.tasks.run()

    # The mute goes on before the native start and again after the window's compensation.
    assert rig.engine.calls == ["mute", "start", "compensation", "mute"] and rig.engine.muted is True
    started = rig.events(Started)
    assert started == [Started("started", True)]
    assert rig.session.view.busy is None and isinstance(rig.views[-1].event, ControlsChanged)


def test_a_start_or_stop_while_a_call_is_out_is_ignored(rig):
    rig.session.request_start("button")
    rig.session.request_start("button")
    rig.session.request_start("menu")
    rig.session.request_stop("button")
    rig.session.request_refresh(lambda: None, lambda _listing: None)
    rig.tasks.run()

    assert rig.engine.lifecycle() == ["start", "compensation"]
    rig.session.request_stop("button")
    rig.session.request_stop("menu")
    rig.session.request_start("menu")
    rig.tasks.run()
    assert rig.engine.lifecycle() == ["start", "compensation", "stop"]
    assert rig.events(Stopped) == [Stopped("button")]


def test_a_window_start_that_has_nothing_to_start_answers_without_the_engine(rig):
    rig.prepare.answer = False  # the window refused (no destination, or a failed recovery)
    assert rig.session.start_now() is False
    rig.prepare.answer = True  # already running
    assert rig.session.start_now() is True
    rig.session.request_start("menu")

    assert rig.engine.calls == [] and rig.tasks.pending == []


def test_a_button_start_right_after_a_button_stop_is_the_second_click_of_a_double_click(rig):
    rig.running()
    rig.session.request_stop("button")
    rig.tasks.run()
    rig.clock.now += START_AFTER_STOP_LOCKOUT_S / 2

    rig.session.request_start("button")
    assert rig.engine.lifecycle() == ["stop"] and rig.tasks.pending == []

    rig.clock.now += START_AFTER_STOP_LOCKOUT_S
    rig.session.request_start("button")
    rig.tasks.run()
    assert rig.engine.lifecycle() == ["stop", "start", "compensation"]


@pytest.mark.parametrize("stop_source", ["menu", "button"])
def test_only_the_shared_button_is_locked_out_after_a_stop(rig, stop_source):
    rig.running()
    rig.session.request_stop(stop_source)
    rig.tasks.run()

    rig.session.request_start("menu")  # the menu, its shortcut: no double-click on one button
    rig.tasks.run()
    assert rig.engine.lifecycle() == ["stop", "start", "compensation"]

    rig.clock.now += START_AFTER_STOP_LOCKOUT_S
    rig.session.request_stop("menu")
    rig.tasks.run()
    rig.session.request_start("button")  # a menu Stop never locks the button either
    rig.tasks.run()
    assert rig.engine.lifecycle() == ["stop", "start", "compensation", "stop", "start", "compensation"]


def test_a_start_that_fails_or_panics_unlocks_and_is_reported(rig):
    rig.engine.start_error = RuntimeError("audio device is busy")
    rig.session.request_start("button")
    rig.tasks.run()

    [failed] = rig.events(StartFailed)
    assert str(failed.error) == "audio device is busy" and failed.interactive
    assert rig.session.view.busy is None

    rig.engine.start_error = None
    rig.session.request_start("button")

    def panic(*_args, **_kwargs):
        raise _Panic("unsendable")

    rig.engine.start = panic
    rig.tasks.run()
    panic = rig.events(StartFailed)[-1]
    assert isinstance(panic.error, RuntimeError) and "unsendable" in str(panic.error)
    assert rig.session.view.busy is None


def test_a_login_start_whose_mute_cannot_be_set_stops_again(rig):
    rig.engine.mute_error = RuntimeError("mute refused")
    assert rig.session.start_now(interactive=False) is False  # refused before the start
    assert rig.engine.lifecycle() == []

    calls = 0

    def mute_after_start(muted: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("mute refused")

    rig.engine.set_output_mute = mute_after_start
    assert rig.session.start_now(interactive=False) is False
    assert rig.engine.lifecycle() == ["start", "compensation", "stop"] and not rig.engine.running
    assert "could not be restored" in str(rig.events(StartFailed)[-1].error)


def test_a_failed_login_start_stops_audio_it_left_running(rig):
    def start_and_fail(*_args, **_kwargs):
        rig.engine.running = True
        raise RuntimeError("half started")

    rig.engine.start = start_and_fail
    assert rig.session.start_now(interactive=False) is False
    assert rig.engine.lifecycle() == ["stop"] and not rig.engine.running


@pytest.mark.parametrize("synchronous", [False, True])
def test_stop_cancels_a_restart_that_is_waiting_to_retry(rig, synchronous):
    assert rig.session.stop_now() is False  # nothing to stop: the controls are redrawn
    assert isinstance(rig.views[-1].event, ControlsChanged)
    rig.engine.restart_requested = True  # a failed restart: the streams are closed, the retry waits

    if synchronous:
        assert rig.session.stop_now() is True
    else:
        rig.session.request_stop("button")
        rig.tasks.run()

    assert rig.engine.lifecycle() == ["stop"] and not rig.engine.restart_requested
    assert rig.events(Stopped)[-1] == Stopped("window" if synchronous else "button")


def test_a_stop_that_fails_is_reported_and_unlocks(rig):
    rig.running()
    rig.engine.stop_error = OSError("device lost")
    rig.session.request_stop("own_microphone")
    rig.tasks.run()

    [failed] = rig.events(StopFailed)
    assert str(failed.error) == "device lost" and failed.source == "own_microphone"
    assert rig.session.view.busy is None and rig.engine.running


def test_a_background_stop_queues_behind_the_call_that_holds_the_engine(rig):
    rig.running()
    rig.session.tick()
    rig.session.stop_in_background()
    assert [call.__name__ for call, _done in rig.tasks.pending[1:]] == ["stop"]

    rig.engine.stop_error = RuntimeError("stop unavailable")
    rig.tasks.run()
    assert rig.engine.lifecycle() == ["service_recovery", "stop"]
    assert [str(event.error) for event in rig.events(BackgroundStopFailed)] == ["stop unavailable"]


# --- recovery ------------------------------------------------------------------------------


def test_the_recovery_poll_runs_one_call_at_a_time_and_never_locks_the_controls(rig):
    rig.running()
    rig.session.tick()
    rig.session.tick()  # the first poll is still out

    assert len(rig.tasks.pending) == 1
    assert rig.session.view.recovering and rig.session.view.busy is None
    assert rig.session.view.in_use
    rig.tasks.run()
    rig.session.tick()
    rig.tasks.run()
    assert rig.engine.calls == ["service_recovery"] * 2


def test_the_poll_reports_what_the_engines_recovery_did_and_restarts_nothing_itself(rig):
    rig.running()
    stall = "input_cb_age_ms=3100, output_cb_age_ms=12, output_write_age_ms=3200"
    for answer, reason in ((None, ""), (False, ""), (True, stall), (True, "DSP thread panicked: boom")):
        rig.engine.recovery_result = answer
        rig.engine.restart_reason = reason
        rig.engine.stream_error = "Restart failed for selected devices: gone"
        rig.session.tick()
        rig.tasks.run()

    assert rig.events(RestartFailed) == [RestartFailed("Restart failed for selected devices: gone")]
    assert rig.events(Restarted) == [Restarted("audio stalled"), Restarted("DSP thread panicked: boom")]
    # A restart that worked gets the window's latency compensation, before its reason is read.
    assert rig.engine.calls == ["service_recovery"] * 3 + ["compensation", "service_recovery", "compensation"]


def test_a_failing_recovery_poll_is_logged_once_per_streak_and_the_next_tick_asks_again(rig, caplog):
    def poll(error: BaseException | None, ticks: int = 1) -> None:
        rig.engine.recovery_error = error
        for _ in range(ticks):
            rig.session.tick()
            rig.tasks.run()

    with caplog.at_level(logging.WARNING, logger="mic_eq.ui.engine_session"):
        poll(_Panic("unsendable"), 3)
        poll(None)  # works again: the streak ends
        poll(RuntimeError("service_recovery failed"))

    assert rig.engine.calls == ["service_recovery"] * 5
    assert [record.message for record in caplog.records] == ["Rust recovery service failed"] * 2


@pytest.mark.parametrize("reopened", [True, False])
def test_a_start_during_a_recovery_poll_waits_for_its_answer(rig, reopened):
    rig.session.tick()
    rig.session.request_start("button")
    assert len(rig.tasks.pending) == 1  # not queued beside the poll

    rig.engine.running = reopened  # what the retry that is out did to the streams
    rig.engine.recovery_result = reopened
    rig.tasks.run()
    rig.tasks.run()

    expected = (
        ["service_recovery", "compensation"] if reopened
        else ["service_recovery", "start", "compensation"]
    )
    assert rig.engine.lifecycle() == expected
    assert rig.engine.running and not rig.events(StartFailed)


def test_a_start_waiting_on_recovery_is_not_locked_out_by_a_recent_button_stop(rig):
    rig.running()
    rig.session.request_stop("button")
    rig.tasks.run()
    rig.session.tick()
    rig.session.request_start("menu")  # within the lockout, so only a non-button Start gets here
    rig.tasks.run()
    rig.tasks.run()

    assert rig.engine.lifecycle() == ["stop", "service_recovery", "start", "compensation"]


def test_a_stop_during_a_recovery_poll_stays_stopped(rig):
    rig.running()
    rig.engine.recovery_result = True
    rig.session.tick()
    rig.session.request_stop("button")  # queued behind the poll on the one worker
    rig.tasks.run()
    rig.engine.recovery_result = None  # the stop took the pending request with it
    rig.session.tick()
    rig.tasks.run()

    assert rig.engine.lifecycle() == ["service_recovery", "compensation", "stop", "service_recovery"]
    assert not rig.engine.running


# --- output mute -----------------------------------------------------------------------------


def test_temporary_mutes_never_clear_the_users_mute_or_each_other(rig):
    rig.session.set_user_muted(True)
    rig.session.hold_mute("calibration", True)
    rig.session.hold_mute("configuration", True)
    rig.session.hold_mute("calibration", False)
    assert rig.engine.muted is True and rig.session.view.mute_reasons == {"configuration"}

    rig.session.hold_mute("configuration", False)
    assert rig.engine.muted is True and rig.session.view.user_muted
    rig.session.set_user_muted(False)
    assert rig.engine.muted is False and not rig.session.view.mute_requested
    rig.session.hold_mute("", True)
    assert rig.session.view.mute_reasons == {"temporary"}
    assert all(isinstance(event, MuteApplied) for event in rig.events())


def test_a_mute_the_engine_refused_is_pending_not_silencing(rig):
    rig.engine.mute_error = RuntimeError("mute unavailable")
    rig.session.set_user_muted(True)

    view = rig.session.view
    assert view.mute_error and view.mute_requested and not view.output_muted
    rig.engine.mute_error = None
    rig.session.set_user_muted(True)
    assert rig.session.view.output_muted and not rig.session.view.mute_error


# --- leases ----------------------------------------------------------------------------------


@pytest.mark.parametrize("call", ["start", "recovery"])
def test_a_borrow_while_a_call_is_out_is_refused_until_it_answers(rig, call):
    if call == "start":
        rig.session.request_start("button")
    else:
        rig.session.tick()

    with pytest.raises(AudioUnavailable, match="try again"):
        rig.session.borrow("test_my_sound")
    assert rig.engine.lifecycle() == [] and rig.events(AudioBorrowed) == []

    rig.tasks.run()
    lease = rig.session.borrow("test_my_sound")
    assert lease.started == (call == "recovery")


def test_a_borrow_that_cannot_read_the_route_says_so_and_owns_nothing(rig, caplog):
    rig.running()

    def unreadable() -> str:
        raise RuntimeError("route unreadable")

    rig.engine.get_active_input_device = unreadable
    with caplog.at_level(logging.ERROR, logger="mic_eq.ui.engine_session"):
        with pytest.raises(AudioUnavailable, match="route unreadable"):
            rig.session.borrow("test_my_sound")
    assert "route unreadable" in caplog.text and rig.events(AudioBorrowed) == []


def test_a_lease_stops_only_audio_it_started_and_only_once(rig):
    lease = rig.session.borrow("auto_eq_calibration")
    assert lease.started and rig.engine.lifecycle() == ["start"]
    assert rig.events(AudioBorrowed) == [AudioBorrowed("auto_eq_calibration", started=True)]
    assert rig.session.borrow("auto_eq_calibration") is lease  # the same take again: still its audio

    assert lease.held
    assert lease.release().stopped
    assert not lease.held and not lease.release().stopped
    assert rig.engine.lifecycle() == ["start", "stop"]

    rig.running()
    other = rig.session.borrow("auto_eq_calibration")
    assert not other.started and not other.release().stopped and rig.engine.running


def test_a_lease_leaves_alone_audio_the_window_restarted(rig):
    lease = rig.session.borrow("latency_calibration")
    rig.session.request_stop("button")
    rig.tasks.run()
    rig.clock.now += 1.0
    rig.session.request_start("button")
    rig.tasks.run()

    assert not lease.release().stopped and rig.engine.running


def test_a_lease_stops_a_restart_its_audio_left_waiting(rig):
    lease = rig.session.borrow("auto_voice_setup")
    rig.engine.running = False
    rig.engine.restart_requested = True
    assert lease.release().stopped and not rig.engine.restart_requested

    lease = rig.session.borrow("auto_voice_setup")
    rig.engine.running = False  # the engine lost it, and does not retry
    assert not lease.release().stopped and rig.engine.lifecycle() == ["start", "stop", "start"]


def test_a_failed_lease_stop_is_logged_reported_and_keeps_ownership(rig, caplog):
    lease = rig.session.borrow("latency_calibration")
    rig.engine.stop_error = RuntimeError("stop refused")
    with caplog.at_level(logging.WARNING, logger="mic_eq.ui.engine_session"):
        report = lease.release()

    assert not report.stopped and str(report.error) == "stop refused"
    assert "stop refused" in caplog.text and "latency_calibration" in caplog.text
    assert rig.events(AudioReleased)[-1].report is report
    rig.engine.stop_error = None
    assert lease.release().stopped and not rig.engine.running


@pytest.mark.parametrize("probe", ["is_running", "is_recovery_requested"])
def test_a_lease_release_whose_engine_probe_fails_is_reported_and_keeps_ownership(rig, caplog, probe):
    lease = rig.session.borrow("latency_calibration")
    rig.engine.running = False  # is_recovery_requested is asked only of audio that has stopped
    original = getattr(rig.engine, probe)

    def fail() -> bool:
        raise RuntimeError("probe failed")

    setattr(rig.engine, probe, fail)
    with caplog.at_level(logging.WARNING, logger="mic_eq.ui.engine_session"):
        report = lease.release()

    assert not report.stopped and str(report.error) == "probe failed"
    assert "probe failed" in caplog.text and rig.events(AudioReleased)[-1].report is report
    setattr(rig.engine, probe, original)
    rig.engine.running = True
    assert lease.release().stopped and not rig.engine.running


def test_a_lease_mute_the_engine_refused_is_let_go_again_by_the_next_release(rig):
    lease = rig.session.borrow("test_my_sound")
    lease.mute(True)
    rig.engine.mute_error = RuntimeError("mute unavailable")
    lease.mute(False)

    assert lease.muted() and rig.session.view.mute_error
    rig.engine.mute_error = None
    lease.release()
    assert not lease.muted() and rig.engine.muted is False and not rig.session.view.mute_error


def test_a_lease_mute_is_its_own_and_goes_with_it(rig):
    lease = rig.session.borrow("test_my_sound")
    lease.mute(True)
    assert lease.muted() and rig.session.view.mute_reasons == {"test_my_sound"}
    rig.session.set_user_muted(True)

    lease.release()
    assert not lease.muted() and rig.session.view.mute_reasons == frozenset()
    assert rig.engine.muted is True  # the user's mute stays


def test_borrowing_audio_on_another_route_asks_and_moves_it_with_the_old_route_as_fallback(rig):
    other = (DeviceIdentity(name="Headset", direction="input"), DESTINATION)
    rig.prepare.route = other
    rig.running()
    rig.prepare.route = (MIC, DESTINATION)

    with pytest.raises(RouteMismatch) as mismatch:
        rig.session.borrow("auto_eq_calibration")
    assert mismatch.value.selected == (MIC, DESTINATION)
    assert mismatch.value.active[0] is not None and mismatch.value.active[0].name == "Headset"
    assert rig.engine.lifecycle() == []

    lease = rig.session.borrow("auto_eq_calibration", switch_route=True)
    assert not lease.started and rig.engine.lifecycle() == ["stop", "start"]
    assert rig.engine.calls[-1] == "mute"  # the mute owners hold on the new route too
    assert rig.engine.route[0] is not None and rig.engine.route[0].name == "Microphone"

    rig.prepare.route = other
    rig.engine.start_error = RuntimeError("busy")
    with pytest.raises(AudioUnavailable, match="busy; failed to restore previous route"):
        rig.session.borrow("auto_eq_calibration", switch_route=True)
    assert isinstance(rig.views[-1].event, ControlsChanged)


def test_a_lease_keeps_its_audio_when_a_retake_moves_it_to_the_selected_route(rig):
    lease = rig.session.borrow("auto_eq_calibration")
    rig.prepare.route = (DeviceIdentity(name="Headset", direction="input"), DESTINATION)

    assert rig.session.borrow("auto_eq_calibration", switch_route=True) is lease
    assert lease.release().stopped and not rig.engine.running


def test_a_borrow_that_cannot_start_says_why_and_owns_nothing(rig):
    rig.engine.start_error = RuntimeError("no microphone")
    with pytest.raises(AudioUnavailable, match="no microphone"):
        rig.session.borrow("test_my_sound")

    rig.engine.start_error = None
    rig.prepare.route = (MIC, None)
    with pytest.raises(AudioUnavailable, match="Select a destination"):
        rig.session.borrow("test_my_sound")
    assert rig.events(AudioBorrowed) == []


# --- quit and shutdown ---------------------------------------------------------------------


def test_quit_asks_only_while_live_audio_would_stop(rig):
    asked: list[bool] = []

    def ask():
        asked.append(True)
        return "cancel"

    assert rig.session.quit_choice(ask, confirm=True) == "quit"
    rig.running()
    assert rig.session.quit_choice(ask, confirm=False) == "quit"
    assert rig.session.quit_choice(ask, confirm=True) == "cancel"
    assert asked == [True]


def test_shutdown_drops_late_answers_and_stops_only_beside_no_call(rig):
    rig.running()
    rig.session.request_stop("button")
    saved: list[bool] = []

    assert rig.session.shutdown(100, save=lambda: saved.append(True) or True) is True
    assert saved == [True] and rig.engine.lifecycle() == []  # no second stop beside the one out

    rig.tasks.run()
    assert rig.engine.lifecycle() == ["stop"] and rig.events(Stopped) == []
    rig.session.tick()
    rig.session.request_start("menu")
    assert rig.tasks.pending == []

    rig.session.reopen()
    rig.clock.now += 1.0
    rig.session.request_start("button")
    rig.tasks.run()
    rig.session.shutdown(100, save=lambda: False)
    assert rig.engine.lifecycle() == ["stop", "start", "compensation", "stop"]


def test_shutdown_still_stops_the_audio_when_saving_raises(rig):
    rig.running()

    def save() -> bool:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        rig.session.shutdown(100, save=save)
    assert rig.engine.lifecycle() == ["stop"] and not rig.engine.running


def test_the_late_answer_of_a_start_after_shutdown_is_dropped(rig):
    rig.session.request_start("button")
    rig.session.shutdown(100, save=lambda: True)
    rig.tasks.run()

    assert rig.events(Started) == [] and rig.engine.lifecycle() == ["start"]
