"""Test adapters for the engine session's seams: the engine, the worker and the clock."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

from mic_eq.config import DeviceIdentity
from mic_eq.ui.capture_session import selected_device_identities
from mic_eq.ui.engine_session import AudioLease, EngineSession, Route
from mic_eq.ui.engine_tasks import attempt

MIC = DeviceIdentity(name="Microphone", direction="input")
DESTINATION = DeviceIdentity(name="Destination", direction="output")


class FakeEngine:
    """The native AudioProcessor's lifecycle calls, answered at once and recorded in order."""

    def __init__(self) -> None:
        self.running = False
        self.restart_requested = False
        self.calls: list[str] = []
        self.muted: bool | None = None
        self.route: Route = (None, None)
        self.start_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.mute_error: Exception | None = None
        self.recovery_result: bool | None = None
        self.recovery_error: BaseException | None = None
        self.restart_reason = ""
        self.stream_error = ""

    def lifecycle(self) -> list[str]:
        return [call for call in self.calls if call != "mute"]

    def start(self, input_name, output_name, *_ordinals, **_endpoints) -> str:
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error
        if self.running:
            raise RuntimeError("Already running")
        self.running = True
        self.route = (
            None if input_name is None else DeviceIdentity(name=input_name, direction="input"),
            DeviceIdentity(name=output_name, direction="output"),
        )
        return "started"

    def stop(self) -> None:
        self.calls.append("stop")
        if self.stop_error is not None:
            raise self.stop_error
        self.running = False
        self.restart_requested = False  # the native stop takes a pending restart with it

    def is_running(self) -> bool:
        return self.running

    def is_recovery_requested(self) -> bool:
        return self.restart_requested

    def service_recovery(self) -> bool | None:
        self.calls.append("service_recovery")
        if self.recovery_error is not None:
            raise self.recovery_error
        return self.recovery_result

    def get_last_restart_reason(self) -> str:
        return self.restart_reason

    def get_last_stream_error(self) -> str:
        return self.stream_error

    def set_output_mute(self, muted: bool) -> None:
        self.calls.append("mute")
        if self.mute_error is not None:
            raise self.mute_error
        self.muted = muted

    def get_active_input_device(self) -> str | None:
        return self.route[0].name if self.route[0] is not None else None

    def get_active_output_device(self) -> str | None:
        return self.route[1].name if self.route[1] is not None else None


class ManualTasks:
    """The engine worker, run by the test: each call and its answer wait for ``run``."""

    threaded = True

    def __init__(self) -> None:
        self.pending: list[tuple[Callable[[], Any], Callable[[Any], object]]] = []

    def submit(self, call: Callable[[], Any], done: Callable[[Any], object]) -> None:
        self.pending.append((call, done))

    def run(self, count: int | None = None) -> None:
        """Run the next ``count`` calls (all of them by default), in order, each with its answer."""
        for _ in range(len(self.pending) if count is None else count):
            call, done = self.pending.pop(0)
            done(attempt(call))

    def wait(self, timeout_ms: int = -1) -> bool:
        del timeout_ms
        return not self.pending


class ManualClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakePreparation:
    """The window's side of a start: the route, and what it does once the engine started."""

    def __init__(self, route: Route = (MIC, DESTINATION)) -> None:
        self.route = route
        self.answer: Route | bool | None = None  # what start_route says instead of the route
        self.prepared: list[bool] = []
        self.engine: FakeEngine | None = None

    def start_route(self, interactive: bool) -> Route | bool:
        self.prepared.append(interactive)
        if self.answer is not None:
            return self.answer
        if self.engine is not None and self.engine.running:
            return True
        return self.route

    def selected_route(self) -> Route:
        return self.route

    def engine_started(self) -> None:
        if self.engine is not None:
            self.engine.calls.append("compensation")


class OwnerPreparation(FakePreparation):
    """The route a fake window's own pickers hold, read when asked, as the main window does."""

    def __init__(self, owner: Any) -> None:
        super().__init__()
        self.owner = owner

    def selected_route(self) -> Route:
        return selected_device_identities(self.owner)

    def start_route(self, interactive: bool) -> Route | bool:
        self.route = self.selected_route()
        return super().start_route(interactive)


def fake_session(
    engine: Any | None = None,
    *,
    route: Route = (MIC, DESTINATION),
    owner: Any | None = None,
    tasks: Any | None = None,
    clock: ManualClock | None = None,
) -> EngineSession:
    """A session over the fakes; the dialogs' fake owners carry one as ``engine_session``.

    With ``owner`` the route is what that fake window's pickers hold, else ``route``.
    """
    prepare = OwnerPreparation(owner) if owner is not None else FakePreparation(route)
    if isinstance(engine, FakeEngine):
        prepare.engine = engine
    return EngineSession(
        engine if engine is not None else FakeEngine(),
        tasks if tasks is not None else ManualTasks(),
        prepare,
        clock=clock if clock is not None else ManualClock(),
    )


def lend_audio(dialog: Any) -> bool:
    """Stands in for a setup dialog's own borrow: the audio is ready under a lease that does nothing."""
    dialog._audio_lease = Mock(spec=AudioLease)
    return True
