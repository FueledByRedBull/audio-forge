"""One owner for the audio session: Start, Stop, the engine's own recovery, and output mute.

The window and the dialogs ask this module for a transition; it runs the blocking engine calls
on ``tasks`` (``engine_tasks.EngineTasks`` in the app), keeps the busy, recovering and muted
state, and emits ``changed`` with a ``SessionView`` after every transition. The window redraws
from that view and shows the prompts its ``event`` asks for; this module shows nothing itself.
Everything here runs on the GUI thread, and so does every ``changed``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import logging
import time
from typing import Any, Literal, Protocol

from PySide6.QtCore import QObject, Signal

from ..config import DeviceIdentity
from .capture_session import (
    active_device_identities,
    restart_processor_for_route,
    route_identities_match,
)
from .device_selection import start_processor_for_route
from .engine_tasks import attempt

logger = logging.getLogger(__name__)

Route = tuple[DeviceIdentity | None, DeviceIdentity | None]
StartSource = Literal["button", "menu"]
StopSource = Literal["button", "menu", "window", "own_microphone"]
QuitChoice = Literal["quit", "tray", "cancel"]

# Start and Stop share one button that turns back into Start as soon as a stop ends (about
# 120 ms), so a button Start this soon after a button Stop is the second click of a double-click.
START_AFTER_STOP_LOCKOUT_S = 0.5
# How the engine's supervisor words a stall restart's reason (callback ages, in ms).
STALL_REASON_PREFIX = "input_cb_age_ms="


class Preparation(Protocol):
    """What the window does around the engine calls; it owns the route widgets and the prompts."""

    def start_route(self, interactive: bool) -> Route | bool:
        """Before a window Start: the route to start, or the final answer when there is nothing to start."""
        ...

    def selected_route(self) -> Route:
        """The route the window has selected now; a dialog's borrow starts or checks it."""
        ...

    def engine_started(self) -> None:
        """After a window Start or an automatic restart worked: the streams are new, so redo what depends on them."""
        ...


class Tasks(Protocol):
    threaded: bool

    def submit(self, call: Callable[[], Any], done: Callable[[Any], object]) -> None: ...

    def wait(self, timeout_ms: int = -1) -> bool: ...


@dataclass(frozen=True)
class ControlsChanged:
    """An engine call went out or came back, or a Stop found nothing to stop."""


@dataclass(frozen=True)
class MuteApplied:
    """The mute owners changed and the engine was told (``SessionView.mute_error`` if it refused)."""


@dataclass(frozen=True)
class Started:
    result: object
    interactive: bool


@dataclass(frozen=True)
class StartFailed:
    error: Exception
    interactive: bool


@dataclass(frozen=True)
class Stopped:
    source: StopSource


@dataclass(frozen=True)
class StopFailed:
    error: Exception
    source: StopSource


@dataclass(frozen=True)
class Restarted:
    """The engine's own recovery reopened the streams; ``reason`` is empty when it gave none."""

    reason: str


@dataclass(frozen=True)
class RestartFailed:
    """The engine's restart found the devices gone; it keeps retrying."""

    error: str


@dataclass(frozen=True)
class BackgroundStopFailed:
    error: Exception


@dataclass(frozen=True)
class AudioBorrowed:
    purpose: str
    started: bool


@dataclass(frozen=True)
class AudioReleased:
    purpose: str
    report: StopReport


SessionEvent = (
    ControlsChanged | MuteApplied | Started | StartFailed | Stopped | StopFailed | Restarted
    | RestartFailed | BackgroundStopFailed | AudioBorrowed | AudioReleased
)


@dataclass(frozen=True)
class SessionView:
    """What the session holds; whether the engine runs is the engine's, and the window asks it."""

    busy: str | None = None  # the call Start, Stop and Refresh wait on: "start", "stop" or "refresh"
    recovering: bool = False  # a recovery poll is out on the worker
    user_muted: bool = False
    mute_reasons: frozenset[str] = field(default_factory=frozenset)
    mute_error: bool = False  # the engine refused the last mute change
    event: SessionEvent | None = None

    @property
    def in_use(self) -> bool:
        """A call is out on the worker; nothing may ask the engine until it comes back."""
        return self.busy is not None or self.recovering

    @property
    def mute_requested(self) -> bool:
        return self.user_muted or bool(self.mute_reasons)

    @property
    def output_muted(self) -> bool:
        """A mute the engine accepted is silencing the output (not one still pending)."""
        return not self.mute_error and self.mute_requested


@dataclass(frozen=True)
class StopReport:
    stopped: bool  # this release stopped the audio
    error: Exception | None = None  # the stop failed; the lease still owns the audio


class RouteMismatch(Exception):
    """The audio runs on another route than the selected one; borrow with ``switch_route`` to move it."""

    def __init__(self, selected: Route, active: Route) -> None:
        super().__init__("The selected route does not match the active stream")
        self.selected = selected
        self.active = active


class AudioUnavailable(Exception):
    """The audio could not be started or moved; the message is the engine's."""


class AudioLease:
    """Audio a dialog borrowed: it stops only what it started, and only while that still runs."""

    def __init__(self, session: EngineSession, purpose: str, started: bool) -> None:
        self._session = session
        self.purpose = purpose
        self.started = started
        self._released = False
        self._muted = False

    def mute(self, on: bool = True) -> None:
        """Hold or let go of the output mute under this lease's purpose."""
        self._session.hold_mute(self.purpose, bool(on))
        # A release the engine refused stays pending: the next release or cleanup tries again.
        self._muted = bool(on) or self._session.view.mute_error

    def muted(self) -> bool:
        return self._muted

    @property
    def held(self) -> bool:
        """The lease has not been released yet, or its stop failed and it still owns the audio."""
        return not self._released

    def release(self) -> StopReport:
        """Let the mute go and stop the audio this lease started; a failed stop keeps it owned."""
        if self._muted:
            self.mute(False)
        if self._released:
            return StopReport(stopped=False)
        report = self._session._release(self)
        self._released = report.error is None
        return report


class EngineSession(QObject):
    changed = Signal(object)  # SessionView

    def __init__(
        self,
        processor: Any,
        tasks: Tasks,
        prepare: Preparation,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.processor = processor
        self._tasks = tasks
        self._prepare = prepare
        self._clock = clock
        self._busy: str | None = None
        self._recovering = False
        self._start_after_recovery = False
        self._recovery_failing = False
        self._button_stop_at: float | None = None
        self._user_muted = False
        self._mute_reasons: set[str] = set()
        self._mute_error = False
        self._closed = False
        self._owner: AudioLease | None = None  # the lease whose start the running audio is

    # --- state -------------------------------------------------------------------------------

    @property
    def view(self) -> SessionView:
        return self._view(None)

    @property
    def closed(self) -> bool:
        return self._closed

    def _view(self, event: SessionEvent | None) -> SessionView:
        return SessionView(
            busy=self._busy,
            recovering=self._recovering,
            user_muted=self._user_muted,
            mute_reasons=frozenset(self._mute_reasons),
            mute_error=self._mute_error,
            event=event,
        )

    def _emit(self, event: SessionEvent) -> None:
        self.changed.emit(self._view(event))

    def _restart_pending(self) -> bool:
        """The engine, streams stopped, waits to retry a failed restart; a Stop cancels it."""
        getter = getattr(self.processor, "is_recovery_requested", None)
        return callable(getter) and bool(getter())

    def _run(self, name: str, call: Callable[[], Any], done: Callable[[Any], object]) -> None:
        """Run a blocking engine call on the worker, with Start, Stop and Refresh locked until ``done`` ran."""
        if self._closed:
            return
        if self._tasks.threaded:  # a call that returns at once would only cost the focus of a button
            self._busy = name
            self._emit(ControlsChanged())

        def finish(outcome: Any) -> None:
            self._busy = None
            if self._closed:
                return
            try:
                done(outcome)
            finally:
                self._emit(ControlsChanged())

        self._tasks.submit(call, finish)

    # --- Start -------------------------------------------------------------------------------

    def request_start(self, source: StartSource) -> None:
        """Start on the worker. Ignored while a call is out; a button Start right after a button Stop is dropped."""
        if self._busy:
            return
        if (
            source == "button"
            and self._button_stop_at is not None
            and self._clock() - self._button_stop_at < START_AFTER_STOP_LOCKOUT_S
        ):
            return
        if self._recovering:
            # Queued behind that call, the start would meet the streams a retry reopened.
            self._start_after_recovery = True
            return
        self._start_on_worker()

    def _start_on_worker(self) -> None:
        if self._busy:
            return
        route = self._route_to_start(True)
        if not isinstance(route, bool):
            self._run(
                "start",
                partial(start_processor_for_route, self.processor, *route),
                partial(self._finish_start, interactive=True),
            )

    def start_now(self, *, interactive: bool = True) -> bool:
        """Start and wait for the engine on this thread; login startup and Guided Setup need the answer."""
        route = self._route_to_start(interactive)
        if isinstance(route, bool):
            return route
        return self._finish_start(
            attempt(partial(start_processor_for_route, self.processor, *route)), interactive
        )

    def _route_to_start(self, interactive: bool) -> Route | bool:
        route = self._prepare.start_route(interactive)
        if isinstance(route, bool):
            return route
        # The persisted mute goes on before the native start, so the first buffer is safe.
        self._apply_mute()
        if not interactive and self._mute_error:
            return False
        return route

    def _finish_start(self, outcome: Any, interactive: bool) -> bool:
        try:
            if isinstance(outcome, Exception):
                raise outcome
            self._prepare.engine_started()
            self._apply_mute()
            if not interactive and self._mute_error:
                self.processor.stop()
                raise RuntimeError("Output mute state could not be restored")
        except Exception as error:
            logger.exception("Start processing failed")
            if not interactive and self.processor.is_running():
                try:
                    self.processor.stop()
                except Exception:
                    logger.exception("Could not stop audio after failed login startup")
            self._emit(StartFailed(error, interactive))
            return False
        self._owner = None
        self._emit(Started(outcome, interactive))
        return True

    # --- Stop --------------------------------------------------------------------------------

    def request_stop(self, source: StopSource = "button") -> None:
        """Stop on the worker; it also cancels a restart that waits to retry."""
        if self._busy or not self._stop_needed():
            return
        if source == "button":
            self._button_stop_at = self._clock()
        self._run("stop", self.processor.stop, partial(self._finish_stop, source=source))

    def stop_now(self) -> bool:
        """Stop and wait for the engine on this thread."""
        if not self._stop_needed():
            return False
        return self._finish_stop(attempt(self.processor.stop), "window")

    def _stop_needed(self) -> bool:
        if self.processor.is_running() or self._restart_pending():
            return True
        self._emit(ControlsChanged())
        return False

    def _finish_stop(self, outcome: Any, source: StopSource) -> bool:
        if isinstance(outcome, Exception):
            logger.error("Stop processing failed", exc_info=outcome)
            self._emit(StopFailed(outcome, source))
            return False
        self._owner = None
        self._emit(Stopped(source))
        return True

    def stop_in_background(self) -> None:
        """Stop after a failed configuration write: queued behind a start or recovery that holds the engine."""
        self._tasks.submit(self.processor.stop, self._background_stop_done)

    def _background_stop_done(self, outcome: Any) -> None:
        if self._closed:
            return
        if isinstance(outcome, Exception):
            logger.error("Stopping audio after a failed configuration failed", exc_info=outcome)
            self._emit(BackgroundStopFailed(outcome))
        else:
            self._owner = None

    def request_refresh(self, listing: Callable[[], Any], done: Callable[[Any], object]) -> None:
        """List the devices on the worker; Start and Stop wait for it like for each other."""
        if not self._busy:
            self._run("refresh", listing, done)

    # --- recovery ----------------------------------------------------------------------------

    def tick(self) -> None:
        """Poll the engine's own stream recovery; it detects stalls, backs off and restarts."""
        if self._busy or self._recovering or self._closed:
            return
        # Not locked like a click: it asks the engine every tick and mostly has nothing to do.
        self._recovering = True
        self._tasks.submit(lambda: self.processor.service_recovery(), self._recovery_serviced)

    def _recovery_serviced(self, outcome: Any) -> None:
        self._recovering = False
        start, self._start_after_recovery = self._start_after_recovery, False
        if self._closed:
            return
        try:
            if isinstance(outcome, Exception):
                raise outcome
            if outcome is not None:
                if outcome:
                    self._prepare.engine_started()
                    try:
                        reason = self.processor.get_last_restart_reason() or ""
                    except Exception:
                        reason = ""
                    if reason.startswith(STALL_REASON_PREFIX):
                        reason = "audio stalled"
                    self._emit(Restarted(reason))
                else:
                    try:
                        error = self.processor.get_last_stream_error() or ""
                    except Exception:
                        error = ""
                    self._emit(RestartFailed(error))
            self._recovery_failing = False
        except Exception:
            # Runs every tick: log the first failure of a streak, not each repeat.
            if not self._recovery_failing:
                logger.warning("Rust recovery service failed", exc_info=True)
            self._recovery_failing = True
        if start:
            self._start_on_worker()

    # --- output mute -------------------------------------------------------------------------

    def set_user_muted(self, on: bool) -> None:
        self._user_muted = bool(on)
        self._apply_mute()

    def hold_mute(self, reason: str, on: bool) -> None:
        """Hold or let go of one temporary mute; letting it go never changes the user's mute."""
        reason = str(reason or "temporary")
        if on:
            self._mute_reasons.add(reason)
        else:
            self._mute_reasons.discard(reason)
        self._apply_mute()

    def _apply_mute(self) -> None:
        try:
            self.processor.set_output_mute(self._user_muted or bool(self._mute_reasons))
            self._mute_error = False
        except Exception:
            self._mute_error = True
            logger.warning("Failed to apply output mute state", exc_info=True)
        self._emit(MuteApplied())

    # --- dialogs -----------------------------------------------------------------------------

    def borrow(self, purpose: str, *, switch_route: bool = False) -> AudioLease:
        """Audio on the selected route for a dialog: started if stopped, moved there with ``switch_route``.

        Raises ``RouteMismatch`` when the audio runs elsewhere and ``switch_route`` is false, and
        ``AudioUnavailable`` when a call is out, the route cannot be read, or the start or the move
        failed (a failed move puts the old route back).
        """
        if self.view.in_use:
            raise AudioUnavailable("Audio is starting, stopping or recovering. Wait a moment and try again.")
        try:
            selected = self._prepare.selected_route()
            active = active_device_identities(self.processor) if self.processor.is_running() else None
        except Exception as error:
            logger.exception("Could not read the audio route")
            raise AudioUnavailable(str(error)) from error
        if active is not None:
            owner = self._owner
            if not route_identities_match(selected, active):
                if not switch_route:
                    raise RouteMismatch(selected, active)
                try:
                    restart_processor_for_route(self.processor, selected, active)
                except Exception as error:
                    self._emit(ControlsChanged())
                    raise AudioUnavailable(str(error)) from error
                self._apply_mute()
                self._emit(AudioBorrowed(purpose, started=False))
            if owner is not None and owner.purpose == purpose:
                return owner  # still the audio that lease started, on this route now
            return AudioLease(self, purpose, started=False)
        try:
            start_processor_for_route(self.processor, *selected)
        except Exception as error:
            raise AudioUnavailable(str(error)) from error
        lease = AudioLease(self, purpose, started=True)
        self._owner = lease
        self._emit(AudioBorrowed(purpose, started=True))
        return lease

    def _release(self, lease: AudioLease) -> StopReport:
        if not lease.started or self._owner is not lease:
            return StopReport(stopped=False)
        outcome = attempt(lambda: bool(self.processor.is_running() or self._restart_pending()))
        if outcome is False:
            self._owner = None
            return StopReport(stopped=False)
        if outcome is True:
            outcome = attempt(self.processor.stop)
        if isinstance(outcome, Exception):
            logger.warning("Stopping the audio started for %s failed: %s", lease.purpose, outcome)
            report = StopReport(stopped=False, error=outcome)
        else:
            self._owner = None
            report = StopReport(stopped=True)
        self._emit(AudioReleased(lease.purpose, report))
        return report

    # --- quit --------------------------------------------------------------------------------

    def quit_choice(self, ask: Callable[[], QuitChoice], *, confirm: bool) -> QuitChoice:
        """Ask before a quit stops live audio; without ``confirm`` or live audio the answer is "quit"."""
        if not confirm or not self.processor.is_running():
            return "quit"
        return ask()

    def shutdown(self, wait_ms: int, *, save: Callable[[], bool]) -> bool:
        """Drop the answers of calls still out, wait for them, ``save``, then stop the audio.

        The stop is skipped beside a call that outlasted the wait: it would wait behind it on
        this thread, and the exit ends the audio anyway. Returns what ``save`` returned.
        """
        self._closed = True
        idle = self._tasks.wait(wait_ms)
        if not idle:
            logger.warning("An engine call is still out at shutdown; the exit ends the audio")
        try:
            return save()
        finally:
            if idle and self.processor.is_running():
                self.processor.stop()

    def reopen(self) -> None:
        """A window shown again after a refused quit takes answers again, and must shut down again."""
        self._closed = False
