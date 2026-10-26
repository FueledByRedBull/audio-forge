"""A finished take is post-processed off the GUI thread, with the same outcomes as before."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import pytest

from mic_eq.analysis.noise_reference import SILENT_REFERENCE_GUIDANCE
from mic_eq.ui import calibration_dialog, voice_setup_dialog
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.calibration_support import filtered_capture_for_analysis
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from capture_helpers import finish_take, settle
from test_auto_eq_candidate_lifecycle import _Owner
from test_voice_setup import _NO_SOUND, _TOO_SHORT, _make_noise, _make_voice
from test_voice_setup_integration import real_main_window  # noqa: F401  (a fixture)

RATE = 48_000


class _Spy:
    """Wraps an analysis function: records the thread it ran on and can hold it open."""

    def __init__(self, monkeypatch, module, name: str, *, gated: bool = False) -> None:
        self.threads: list[threading.Thread] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        if not gated:
            self.release.set()
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            self.threads.append(threading.current_thread())
            self.entered.set()
            self.release.wait(10.0)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    @property
    def off_the_gui_thread(self) -> bool:
        return bool(self.threads) and all(
            thread is not threading.main_thread() for thread in self.threads
        )


def _speech() -> np.ndarray:
    return _make_voice(RATE, seconds=1.0)


# --- Voice Setup -------------------------------------------------------------------------


@pytest.fixture
def voice(real_main_window, monkeypatch):  # noqa: F811
    dialog = VoiceSetupDialog(parent=real_main_window)
    analyses: list[int] = []
    setattr(dialog, "analyses", analyses)  # noqa: B010 (read back by the tests)
    monkeypatch.setattr(dialog, "_start_analysis", lambda: analyses.append(1))
    yield dialog
    dialog.reject()
    settle(dialog)
    dialog.deleteLater()


def _take(dialog: VoiceSetupDialog, state: str, audio: np.ndarray, *, finish=finish_take) -> None:
    dialog.setup_state = state
    dialog.phase_label.setText("Capturing")
    dialog._current_capture_metadata = None
    finish(dialog, audio)


def _gated_voice_take(dialog: VoiceSetupDialog, monkeypatch, state: str) -> _Spy:
    """Finish a take of the phase ``state`` and hold its post-processing open; returns the spy that holds it."""
    prefilter = _Spy(monkeypatch, voice_setup_dialog, "_filtered_capture_for_analysis", gated=True)
    if state == "voice_recording":
        dialog.noise_audio = _make_noise(RATE)
    dialog.setup_state = state
    dialog._current_capture_metadata = None
    dialog._on_recording_complete(_speech() if state == "voice_recording" else _make_noise(RATE))
    assert prefilter.entered.wait(5.0)
    return prefilter


def test_a_good_room_noise_take_is_kept_filtered_and_voice_comes_next(voice) -> None:
    raw = _make_noise(RATE)
    _take(voice, "noise_recording", raw)

    assert voice.setup_state == "noise_ready"
    assert voice.phase_label.text() == "Room noise captured"
    assert voice.start_button.text() == "Record Voice"
    assert voice.start_button.isEnabled()
    assert voice.warning_label.text() == "Now read the passage aloud for the full 10 seconds."
    assert np.array_equal(voice.preview_noise_audio, raw)
    assert np.array_equal(voice.noise_audio, filtered_capture_for_analysis(raw, RATE))


def test_a_silent_room_noise_take_moves_on_with_the_silent_reference_advice(voice) -> None:
    _take(voice, "noise_recording", np.zeros(2 * RATE, dtype=np.float32))

    assert voice.setup_state == "noise_ready"
    assert voice.noise_audio is not None
    assert voice.warning_label.text().startswith(SILENT_REFERENCE_GUIDANCE)
    assert voice.warning_label.text().endswith("Now read the passage aloud for the full 10 seconds.")


def test_a_room_noise_take_the_analysis_cannot_use_is_rejected(voice) -> None:
    _take(voice, "noise_recording", _make_noise(RATE, seconds=0.5))

    assert voice.setup_state == "idle"
    assert voice.noise_audio is None and voice.preview_noise_audio is None
    assert voice.phase_label.text() == "Room-noise capture rejected"
    assert voice.start_button.text() == "Retake Room Noise"
    assert all(token in voice.warning_label.text() for token in _TOO_SHORT)


def test_a_silent_voice_take_is_a_microphone_fault(voice) -> None:
    voice.noise_audio = _make_noise(RATE)
    _take(voice, "voice_recording", np.zeros(RATE, dtype=np.float32))

    assert voice.setup_state == "noise_ready"
    assert voice.voice_audio is None and voice.preview_voice_audio is None
    assert voice.phase_label.text() == "Voice capture rejected"
    assert voice.start_button.text() == "Record Voice"
    assert all(token in voice.warning_label.text() for token in _NO_SOUND)
    assert voice.analyses == []


def test_a_good_voice_take_starts_the_setup_analysis(voice) -> None:
    voice.noise_audio = _make_noise(RATE)
    raw = _speech()
    _take(voice, "voice_recording", raw)

    assert voice.setup_state == "analyzing"
    assert voice.analyses == [1]
    assert voice.start_button.text() == "Analyzing..."
    assert not voice.start_button.isEnabled()
    assert voice.phase_label.text() == "Analyzing recordings"
    assert voice.warning_label.text() == "Building recommendations for your voice chain."
    assert np.array_equal(voice.preview_voice_audio, raw)
    assert np.array_equal(voice.voice_audio, filtered_capture_for_analysis(raw, RATE))


def test_a_verification_take_goes_to_the_verification_with_the_filtered_audio(
    voice, monkeypatch
) -> None:
    received: list[np.ndarray] = []
    monkeypatch.setattr(voice, "_complete_verification", received.append)
    raw = _speech()
    _take(voice, "verification_recording", raw)

    assert voice.setup_state == "verification_recording"
    assert np.array_equal(voice.preview_verification_audio, raw)
    assert len(received) == 1
    assert np.array_equal(received[0], filtered_capture_for_analysis(raw, RATE))


def test_voice_setup_post_processing_runs_off_the_gui_thread(voice, monkeypatch) -> None:
    prefilter = _Spy(monkeypatch, voice_setup_dialog, "_filtered_capture_for_analysis")
    reference = _Spy(monkeypatch, voice_setup_dialog, "analyze_noise_reference")

    _take(voice, "noise_recording", _make_noise(RATE))

    assert prefilter.off_the_gui_thread
    assert reference.off_the_gui_thread
    assert voice.setup_state == "noise_ready"


def test_voice_setup_waits_in_its_own_state_and_then_continues(voice, monkeypatch) -> None:
    prefilter = _gated_voice_take(voice, monkeypatch, "voice_recording")

    assert voice.setup_state == "capture_analyzing"
    assert not voice.start_button.isEnabled()
    voice._on_start_clicked()  # nothing to do while the take is checked
    assert voice.setup_state == "capture_analyzing"
    prefilter.release.set()
    settle(voice)

    assert voice.setup_state == "analyzing" and voice.analyses == [1]


def test_closing_voice_setup_while_a_take_is_checked_drops_the_result(voice, monkeypatch) -> None:
    prefilter = _gated_voice_take(voice, monkeypatch, "voice_recording")

    voice.reject()
    assert voice._close_requested
    assert voice.warning_label.text() == "Canceling analysis..."
    prefilter.release.set()
    settle(voice)

    assert voice.analyses == []
    assert voice.setup_state == "idle"
    assert voice.voice_audio is None and voice.preview_voice_audio is None
    assert not voice._close_requested  # the close went through once the worker was done
    assert voice.result() == voice.DialogCode.Rejected
    assert not voice._analysis_workers


def test_starting_over_while_a_take_is_checked_drops_the_result(voice, monkeypatch) -> None:
    prefilter = _gated_voice_take(voice, monkeypatch, "noise_recording")

    voice._reset_setup_ui()
    prefilter.release.set()
    settle(voice)

    assert voice.setup_state == "idle"
    assert voice.noise_audio is None and voice.preview_noise_audio is None
    assert voice.phase_label.text() == "Ready to start setup"


def test_application_quit_joins_a_take_still_being_checked(voice, monkeypatch) -> None:
    prefilter = _gated_voice_take(voice, monkeypatch, "noise_recording")
    worker = voice._analysis_workers[0]
    threading.Timer(0.1, prefilter.release.set).start()

    voice._wait_for_analysis_workers()

    assert not worker.isRunning()
    assert not voice._analysis_workers
    settle(voice)
    assert voice.noise_audio is None


def test_a_take_that_cannot_be_post_processed_fails_like_a_recording_error(
    voice, monkeypatch
) -> None:
    def broken(_audio, _rate):
        raise RuntimeError("Native input conditioning is unavailable")

    monkeypatch.setattr(voice_setup_dialog, "_filtered_capture_for_analysis", broken)
    _take(voice, "noise_recording", _make_noise(RATE))

    assert voice.setup_state == "idle"
    assert voice.noise_audio is None
    assert voice.start_button.isEnabled()
    assert "Native input conditioning is unavailable" in voice.warning_label.text()


# --- Auto-EQ -----------------------------------------------------------------------------


class _Session:
    def __init__(self, *_args) -> None:
        pass

    def cleanup(self) -> None:
        pass


@pytest.fixture
def auto_eq(qapp, monkeypatch):
    monkeypatch.setattr(calibration_dialog, "CaptureSession", _Session)
    monkeypatch.setattr(calibration_dialog, "_start_selected_route", lambda _owner: None)
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    analyses: list[int] = []
    setattr(dialog, "analyses", analyses)  # noqa: B010 (read back by the tests)
    monkeypatch.setattr(dialog, "_start_analysis", lambda: analyses.append(1))
    yield dialog
    dialog.reject()
    settle(dialog)
    owner.close()


def _auto_take(dialog: CalibrationDialog, audio: np.ndarray, finish: Callable = finish_take) -> None:
    dialog.recording_state = "recording"
    finish(dialog, audio)


def _gated_auto_eq_take(dialog: CalibrationDialog, monkeypatch) -> _Spy:
    """Finish a take and hold its post-processing open; returns the spy that holds it."""
    prefilter = _Spy(monkeypatch, calibration_dialog, "_filtered_capture_for_analysis", gated=True)
    dialog.recording_state = "recording"
    dialog._on_recording_complete(_speech())
    assert prefilter.entered.wait(5.0)
    return prefilter


def test_auto_eq_keeps_a_good_take_filtered_and_starts_the_analysis(auto_eq) -> None:
    raw = _speech()
    _auto_take(auto_eq, raw)

    assert auto_eq.recording_state == "analyzing"
    assert auto_eq.analyses == [1]
    assert auto_eq.start_button.text() == "Analyzing..."
    assert not auto_eq.start_button.isEnabled()
    assert auto_eq.retake_btn.isHidden() is False
    assert auto_eq.warning_label.text() == "Recording complete. Checking it against the tone preset."
    assert (auto_eq.progress_bar.minimum(), auto_eq.progress_bar.maximum()) == (0, 0)
    assert np.array_equal(auto_eq.preview_audio_data, raw)
    assert np.array_equal(auto_eq.audio_data, filtered_capture_for_analysis(raw, RATE))


def test_auto_eq_refuses_a_silent_take_as_a_microphone_fault(auto_eq) -> None:
    _auto_take(auto_eq, np.zeros(RATE, dtype=np.float32))

    assert auto_eq.recording_state == "idle"
    assert auto_eq.audio_data is None and auto_eq.preview_audio_data is None
    assert auto_eq.analyses == []
    assert all(token in auto_eq.warning_label.text() for token in _NO_SOUND)
    assert auto_eq.start_button.text() == "Check Level"


def test_auto_eq_waits_in_its_own_state_and_then_continues(auto_eq, monkeypatch) -> None:
    prefilter = _gated_auto_eq_take(auto_eq, monkeypatch)

    assert auto_eq.recording_state == "capture_analyzing"
    assert not auto_eq.start_button.isEnabled()
    auto_eq._on_start_clicked()
    assert auto_eq.recording_state == "capture_analyzing"
    prefilter.release.set()
    settle(auto_eq)

    assert auto_eq.recording_state == "analyzing" and auto_eq.analyses == [1]


def test_closing_auto_eq_while_a_take_is_checked_drops_the_result(auto_eq, monkeypatch) -> None:
    prefilter = _gated_auto_eq_take(auto_eq, monkeypatch)

    auto_eq.reject()
    assert auto_eq._close_requested
    assert auto_eq.warning_label.text() == "Canceling analysis..."
    prefilter.release.set()
    settle(auto_eq)

    assert auto_eq.analyses == []
    assert auto_eq.recording_state == "idle"
    assert auto_eq.audio_data is None and auto_eq.preview_audio_data is None
    assert not auto_eq._close_requested
    assert auto_eq.result() == auto_eq.DialogCode.Rejected
    assert not auto_eq._analysis_workers


def test_auto_eq_start_over_while_a_take_is_checked_drops_the_result(auto_eq, monkeypatch) -> None:
    prefilter = _gated_auto_eq_take(auto_eq, monkeypatch)

    auto_eq._reset_recording_ui()
    prefilter.release.set()
    settle(auto_eq)

    assert auto_eq.recording_state == "idle"
    assert auto_eq.analyses == []
    assert auto_eq.audio_data is None


def test_auto_eq_application_quit_joins_a_take_still_being_checked(auto_eq, monkeypatch) -> None:
    prefilter = _gated_auto_eq_take(auto_eq, monkeypatch)
    worker = auto_eq._analysis_workers[0]
    threading.Timer(0.1, prefilter.release.set).start()

    auto_eq._wait_for_analysis_workers()

    assert not worker.isRunning()
    assert not auto_eq._analysis_workers
    settle(auto_eq)
    assert auto_eq.audio_data is None


def test_auto_eq_take_that_cannot_be_post_processed_fails_like_a_recording_error(
    auto_eq, monkeypatch
) -> None:
    def broken(_audio, _rate):
        raise RuntimeError("Native input conditioning is unavailable")

    monkeypatch.setattr(calibration_dialog, "_filtered_capture_for_analysis", broken)
    _auto_take(auto_eq, _speech())

    assert auto_eq.recording_state == "idle"
    assert auto_eq.audio_data is None
    assert "Native input conditioning is unavailable" in auto_eq.warning_label.text()
