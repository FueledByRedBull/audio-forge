"""The setup dialogs' analysis workers: tokens, cancelling, and keeping a running thread owned."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QEvent

from mic_eq.ui.analysis_worker import CaptureAnalysisWorker, WorkerSet


class _Recorder:
    def __init__(self) -> None:
        self.finished = 0
        self.results: list[object] = []
        self.errors: list[str] = []

    def on_finished(self) -> None:
        self.finished += 1


def _blocked(release: threading.Event, entered: threading.Event):
    def analyze() -> str:
        entered.set()
        release.wait(5.0)
        return "result"

    return analyze


def test_a_new_run_or_a_cancel_makes_every_older_token_stale() -> None:
    workers = WorkerSet(lambda: None)
    first = workers.begin()
    assert workers.is_current(first)

    second = workers.begin()
    assert not workers.is_current(first) and workers.is_current(second)

    workers.cancel()
    assert not workers.is_current(second)
    assert not workers.is_current(None)


def test_a_finished_capture_delivers_its_result_and_drops_the_worker(qapp) -> None:
    recorder = _Recorder()
    workers = WorkerSet(recorder.on_finished)

    workers.start_capture(lambda: "result", recorder.results.append, recorder.errors.append)
    worker = workers.current
    assert isinstance(worker, CaptureAnalysisWorker) and workers.workers == [worker]
    assert worker.wait(5000)
    qapp.processEvents()

    assert recorder.results == ["result"] and recorder.errors == []
    assert workers.workers == [] and workers.current is None
    assert recorder.finished == 1


def test_a_cancelled_worker_stays_owned_until_its_thread_ends(qapp) -> None:
    recorder = _Recorder()
    workers = WorkerSet(recorder.on_finished)
    release, entered = threading.Event(), threading.Event()
    workers.start_capture(_blocked(release, entered), recorder.results.append, recorder.errors.append)
    worker = workers.current
    assert worker is not None and entered.wait(5.0)

    workers.cancel()

    assert workers.current is None and workers.workers == [worker]
    assert workers.running() and worker._should_stop()
    release.set()
    assert worker.wait(5000)
    qapp.processEvents()
    assert not workers.running() and workers.workers == []
    assert recorder.results == [] and recorder.finished == 1


def test_join_stops_and_waits_for_every_running_worker(qapp) -> None:
    recorder = _Recorder()
    workers = WorkerSet(recorder.on_finished)
    entered = threading.Event()

    def analyze() -> None:
        entered.set()
        while not workers.workers[0]._should_stop():
            time.sleep(0.005)

    workers.start_capture(analyze, recorder.results.append, recorder.errors.append)
    worker = workers.current
    assert worker is not None and entered.wait(5.0)

    workers.join()

    assert worker.isFinished()
    assert workers.workers == [] and workers.current is None and not workers.running()
    qapp.processEvents()


def test_discard_hands_finished_workers_to_qt(qapp) -> None:
    workers = WorkerSet(lambda: None)
    worker = CaptureAnalysisWorker(lambda: None)
    workers.track(worker)
    deleted: list[bool] = []
    worker.destroyed.connect(lambda *_args: deleted.append(True))

    workers.discard()

    assert workers.workers == []
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert deleted == [True]
