"""Blocking audio-engine calls (start, stop, recovery, device listing) off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

# Off until the native AudioProcessor can be called from a thread other than the one that made it:
# it is `unsendable` in rust-core (python_api.rs), so a worker's call would panic. Even then the
# window only stays alive if start, stop, service_recovery and the device listings release the GIL.
OFFLOAD_ENGINE_CALLS = False


def attempt(call: Callable[[], Any]) -> Any:
    """The call's result, or the exception it raised."""
    try:
        return call()
    except Exception as error:
        return error


class EngineTasks(QObject):
    """One worker thread that runs engine calls one at a time, in the order they were submitted.

    The outcome of each call (its result, or the exception it raised) goes to its callback on the
    GUI thread, which may therefore touch widgets. While ``threaded`` is false the call and its
    callback run at once, on the calling thread.
    """

    _finished = Signal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.threaded = OFFLOAD_ENGINE_CALLS
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._finished.connect(self._deliver)

    def submit(self, call: Callable[[], Any], done: Callable[[Any], object]) -> None:
        if self.threaded:
            self._pool.start(lambda: self._finished.emit(done, attempt(call)))
        else:
            done(attempt(call))

    def wait(self, timeout_ms: int = -1) -> bool:
        """Wait for the calls that were submitted; false if they outlast the timeout."""
        return self._pool.waitForDone(timeout_ms)

    @Slot(object, object)
    def _deliver(self, done: Callable[[Any], object], outcome: Any) -> None:
        done(outcome)
