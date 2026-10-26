"""Blocking audio-engine calls (start, stop, recovery, device listing) off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThreadPool, Signal, Slot


def attempt(call: Callable[[], Any]) -> Any:
    """The call's result, or the error it raised.

    A native panic (pyo3_runtime.PanicException) is no Exception; it comes back as a RuntimeError,
    or the worker would die with the answer unsent and the window locked. Interrupts pass through.
    """
    try:
        return call()
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as error:
        return error
    except BaseException as panic:
        failure = RuntimeError(f"{type(panic).__name__}: {panic}")
        failure.__cause__ = panic
        return failure


class EngineTasks(QObject):
    """One worker thread that runs engine calls one at a time, in the order they were submitted.

    The outcome of each call (its result, or the exception it raised) goes to its callback on the
    GUI thread, which may therefore touch widgets. While ``threaded`` is false the call and its
    callback run at once, on the calling thread.

    The native AudioProcessor can be called from any thread (a frozen pyclass in rust-core,
    python_api.rs), and start, stop, service_recovery and the device listings release the GIL, so the
    window stays alive while they run. Getters and setters never wait for a start in progress.
    """

    _finished = Signal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.threaded = True
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
