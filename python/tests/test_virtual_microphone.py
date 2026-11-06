"""AudioForge's PipeWire microphone is built again after PipeWire restarts."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from mic_eq import mic_eq_core
from mic_eq.ui.virtual_microphone import CHECK_SECONDS, RETRY_SECONDS, VirtualMicrophoneWatch


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def pipewire(monkeypatch):
    """The two native calls: whether the microphone exists, and building it, which can fail."""
    state: Any = SimpleNamespace(
        running=True, looks=0, builds=0, error=None, built=threading.Semaphore(0),
        hold=None, build_thread=None,
    )

    def running() -> bool:
        state.looks += 1
        return state.running

    def create() -> None:
        state.builds += 1
        state.build_thread = threading.current_thread()
        try:
            if state.hold is not None:
                state.hold.wait(timeout=5)
            if state.error is not None:
                raise state.error
            state.running = True
        finally:
            state.built.release()

    monkeypatch.setattr(mic_eq_core, "virtual_microphone_running", running, raising=False)
    monkeypatch.setattr(mic_eq_core, "create_virtual_microphone", create, raising=False)
    return state


def _settled(watch: VirtualMicrophoneWatch) -> None:
    deadline = time.monotonic() + 5
    while watch._rebuilding and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not watch._rebuilding


def test_a_microphone_that_exists_is_left_alone(pipewire) -> None:
    VirtualMicrophoneWatch(_Clock()).poll()

    assert pipewire.looks == 1 and pipewire.builds == 0


def test_a_dropped_microphone_is_built_again_off_the_calling_thread(pipewire) -> None:
    pipewire.running = False
    pipewire.hold = threading.Event()  # building waits on PipeWire
    watch = VirtualMicrophoneWatch(_Clock())

    watch.poll()  # returns while the build is still waiting

    assert pipewire.built.acquire(timeout=0.2) is False
    pipewire.hold.set()
    assert pipewire.built.acquire(timeout=5)
    assert pipewire.build_thread is not threading.current_thread()
    _settled(watch)
    assert pipewire.running


def test_it_looks_only_every_few_seconds(pipewire) -> None:
    clock = _Clock()
    watch = VirtualMicrophoneWatch(clock)

    watch.poll()
    clock.now += CHECK_SECONDS - 0.1
    watch.poll()
    assert pipewire.looks == 1

    clock.now += 0.2
    watch.poll()
    assert pipewire.looks == 2


def test_a_rebuild_in_progress_is_not_started_twice(pipewire) -> None:
    pipewire.running = False
    pipewire.hold = threading.Event()
    clock = _Clock()
    watch = VirtualMicrophoneWatch(clock)

    watch.poll()
    clock.now += CHECK_SECONDS
    watch.poll()
    pipewire.hold.set()
    assert pipewire.built.acquire(timeout=5)
    _settled(watch)

    assert pipewire.builds == 1


def test_a_rebuild_that_fails_is_tried_again_only_after_a_pause(pipewire) -> None:
    pipewire.running = False
    pipewire.error = RuntimeError("PipeWire is not running")
    clock = _Clock()
    watch = VirtualMicrophoneWatch(clock)

    watch.poll()
    assert pipewire.built.acquire(timeout=5)
    _settled(watch)
    assert pipewire.builds == 1

    clock.now += CHECK_SECONDS
    watch.poll()
    assert pipewire.builds == 1  # too soon

    pipewire.error = None
    clock.now += RETRY_SECONDS
    watch.poll()
    assert pipewire.built.acquire(timeout=5)
    _settled(watch)
    assert pipewire.builds == 2 and pipewire.running


def test_a_stopped_watch_looks_and_rebuilds_no_more(pipewire) -> None:
    pipewire.running = False
    clock = _Clock()
    watch = VirtualMicrophoneWatch(clock)
    watch.stop()

    clock.now += RETRY_SECONDS
    watch.poll()

    assert pipewire.looks == 0 and pipewire.builds == 0


def test_quitting_stops_the_watch(isolated_main_window) -> None:
    window: Any = isolated_main_window
    watch = Mock()
    window.virtual_microphone_watch = watch

    window._shutdown()

    watch.stop.assert_called_once_with()


def test_a_window_shown_again_after_quitting_watches_again(isolated_main_window) -> None:
    window: Any = isolated_main_window
    watch = Mock()
    window.virtual_microphone_watch = watch

    window._shutdown()
    watch.resume.assert_not_called()
    window.show()

    watch.resume.assert_called_once_with()


def test_the_diagnostics_timer_polls_the_watch(isolated_main_window) -> None:
    window: Any = isolated_main_window
    watch = Mock()
    window.virtual_microphone_watch = watch

    window._update_diagnostics()

    watch.poll.assert_called_once_with()


def test_an_outage_is_reported_once_however_many_retries_it_takes(pipewire, caplog):
    pipewire.running = False
    pipewire.error = RuntimeError("no PipeWire")
    clock = _Clock()
    watch = VirtualMicrophoneWatch(clock=clock)
    with caplog.at_level("WARNING"):
        for attempt in range(3):
            watch.poll()
            assert pipewire.built.acquire(timeout=5)
            pipewire.build_thread.join(timeout=5)
            clock.now += RETRY_SECONDS + CHECK_SECONDS + 1
    assert pipewire.builds == 3
    assert sum("is gone" in record.message for record in caplog.records) == 1
