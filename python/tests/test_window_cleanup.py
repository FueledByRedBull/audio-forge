"""A test's main window never leaves its native audio processor to be dropped on another thread.

The processor is unsendable: dropping it anywhere but the thread that made it is reported as an unraisable
exception. A window is Python garbage in a reference cycle once its test ends, so the next collection to
run frees it, and one that runs in an analysis worker thread is the wrong thread. ``conftest.py`` collects on
the main thread before the next test starts; the second test needs the first to run first.
"""

import gc
import sys
import threading

import pytest

window_left_behind = False


def test_a_test_leaves_a_main_window_behind(isolated_main_window):
    global window_left_behind
    assert isolated_main_window.processor is not None
    window_left_behind = True


def test_the_next_test_finds_no_processor_left_for_a_worker_thread(qapp):
    if not window_left_behind:
        pytest.skip("needs the window the test before it leaves; run the whole file")
    dropped: list[str] = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda unraisable: dropped.append(str(unraisable.exc_value))
    try:
        worker = threading.Thread(target=gc.collect)
        worker.start()
        worker.join()
    finally:
        sys.unraisablehook = previous

    assert not dropped
