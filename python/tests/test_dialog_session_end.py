"""A setup dialog that is open must not veto Windows signing out or shutting down.

The main window sets ``audioforge_session_ending`` on the application while the session ends. Each
dialog then stops what it runs, waits for its threads and accepts the close. Any other close is as
it always was: the dialog asks its work to stop and closes when that has stopped.
"""
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false
# (the sibling test modules are found by pytest at run time; the dialogs get stand-in workers and sinks)

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QDialog

from mic_eq.ui.analysis_worker import start_capture_analysis
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog
from mic_eq.ui.listening_comparison_dialog import ListeningComparisonDialog
from test_listening_comparison import _settings
from test_setup_level_check_flow import voice  # noqa: F401  (a fixture)
from test_voice_setup_integration import real_main_window  # noqa: F401  (a fixture)

REJECTED = int(QDialog.DialogCode.Rejected)


@pytest.fixture
def session_ending(qapp):
    qapp.setProperty("audioforge_session_ending", True)
    yield
    qapp.setProperty("audioforge_session_ending", False)


class _StoppableThread(QThread):
    """A worker that runs until it is asked to stop, as the analysis threads do."""

    def __init__(self) -> None:
        super().__init__()
        self._stopped = threading.Event()

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        self._stopped.wait(10)


class _Session:
    """A capture session that only records being cleaned up."""

    def __init__(self) -> None:
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class _Playback:
    """The pieces of audio playback a dialog stops: a sink and its buffer."""

    def __init__(self) -> None:
        self.stopped = self.deleted = False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.stopped = True

    def deleteLater(self) -> None:
        self.deleted = True


def _blocked_analysis(dialog) -> QThread:
    """A take's post-processing that runs until the dialog asks it to stop."""
    running = threading.Event()

    def analyze():
        running.set()
        while not dialog._analysis_workers[0]._should_stop():
            time.sleep(0.005)

    start_capture_analysis(dialog, analyze, lambda _result: None, lambda _error: None)
    worker = dialog._analysis_workers[0]
    assert running.wait(5) and worker.isRunning()
    return worker


def _settle(qapp) -> None:
    qapp.processEvents()
    qapp.processEvents()


@pytest.mark.parametrize("ending", [True, False], ids=["session-ending", "normal"])
def test_voice_setup_stops_its_recording_and_analysis_when_closed(
    voice, qapp, ending  # noqa: F811
) -> None:
    qapp.setProperty("audioforge_session_ending", ending)
    try:
        voice.show()
        session = _Session()
        voice.setup_state = "voice_recording"
        voice._capture_session = session
        voice.recording_timer.start()
        worker = _blocked_analysis(voice)

        closed = voice.close()
        alive = worker.isRunning()
        if not ending:
            worker.wait()
        _settle(qapp)

        assert not voice.recording_timer.isActive() and session.cleaned
        if ending:
            # Accepted at once, with the thread joined: nothing is left to wait for.
            assert closed and not voice.isVisible() and not alive
            assert voice.result() == REJECTED and voice._analysis_workers == []
        else:
            # As before: the close is vetoed while the thread winds down, then the dialog closes itself.
            assert not closed
            assert not voice.isVisible() and voice.result() == REJECTED
    finally:
        qapp.setProperty("audioforge_session_ending", False)


def test_voice_setup_closes_for_a_session_end_even_when_its_old_settings_cannot_be_put_back(
    voice, session_ending, qapp, monkeypatch  # noqa: F811
) -> None:
    voice.show()
    monkeypatch.setattr(voice, "_restore_pre_setup_snapshot", lambda: False)
    assert voice.close() and not voice.isVisible()


def test_voice_setup_still_refuses_a_normal_close_when_its_old_settings_cannot_be_put_back(
    voice, qapp, monkeypatch  # noqa: F811
) -> None:
    voice.show()
    monkeypatch.setattr(voice, "_restore_pre_setup_snapshot", lambda: False)
    assert not voice.close() and voice.isVisible()
    assert "Close blocked" in voice.phase_label.text()


@pytest.mark.parametrize("ending", [True, False], ids=["session-ending", "normal"])
def test_auto_eq_stops_its_recording_and_analysis_when_closed(
    isolated_main_window, qapp, ending
) -> None:
    dialog = CalibrationDialog(isolated_main_window)
    qapp.setProperty("audioforge_session_ending", ending)
    try:
        dialog.show()
        session = _Session()
        dialog.recording_state = "recording"
        dialog._capture_session = session
        dialog.recording_timer.start()
        worker = _blocked_analysis(dialog)

        closed = dialog.close()
        alive = worker.isRunning()
        if not ending:
            worker.wait()
        _settle(qapp)

        assert not dialog.recording_timer.isActive() and session.cleaned
        if ending:
            assert closed and not dialog.isVisible() and not alive
            assert dialog.result() == REJECTED and dialog._analysis_workers == []
        else:
            assert not closed
            assert not dialog.isVisible() and dialog.result() == REJECTED
    finally:
        qapp.setProperty("audioforge_session_ending", False)
        dialog.deleteLater()


@pytest.mark.parametrize("ending", [True, False], ids=["session-ending", "normal"])
def test_latency_calibration_stops_its_measurement_when_closed(
    isolated_main_window, qapp, ending
) -> None:
    dialog = LatencyCalibrationDialog(isolated_main_window)
    qapp.setProperty("audioforge_session_ending", ending)
    try:
        dialog.show()
        session = _Session()
        worker = dialog.worker = _StoppableThread()
        worker.finished.connect(lambda: dialog._on_thread_finished(worker))
        worker.start()
        dialog._capture_session = session
        assert worker.isRunning()

        closed = dialog.close()
        alive = worker.isRunning()
        if not ending:
            worker.wait()
        _settle(qapp)

        assert session.cleaned
        if ending:
            assert closed and not dialog.isVisible() and not alive
            assert dialog.worker is None and dialog.result() == REJECTED
        else:
            assert not closed
            assert not dialog.isVisible() and dialog.result() == REJECTED
    finally:
        qapp.setProperty("audioforge_session_ending", False)
        dialog.deleteLater()


@pytest.mark.parametrize("ending", [True, False], ids=["session-ending", "normal"])
def test_the_listening_comparison_stops_playback_and_rendering_when_closed(
    qapp, monkeypatch, ending
) -> None:
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    dialog = ListeningComparisonDialog(
        audio_data=np.zeros(480, dtype=np.float32), sample_rate=48_000,
        current_settings=_settings(), proposed_settings=_settings(),
    )
    qapp.setProperty("audioforge_session_ending", ending)
    try:
        dialog.show()
        sink, buffer = _Playback(), _Playback()
        dialog._audio_sink, dialog._audio_buffer = sink, buffer
        worker = dialog._worker = _StoppableThread()
        worker.finished.connect(lambda: setattr(dialog, "_worker", None) or dialog._finish_close())
        worker.start()

        closed = dialog.close()
        alive = worker.isRunning()
        if not ending:
            worker.wait()
        _settle(qapp)

        assert sink.stopped and buffer.stopped
        if ending:
            assert closed and not dialog.isVisible() and not alive
            assert dialog.decision is False and dialog.result() == REJECTED
        else:
            assert not closed
            assert not dialog.isVisible() and dialog.decision is False
    finally:
        qapp.setProperty("audioforge_session_ending", False)
        dialog.deleteLater()
