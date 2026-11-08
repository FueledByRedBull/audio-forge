"""The 3-second level check that opens Voice Setup and Auto-EQ."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import numpy as np
import pytest

from mic_eq.ui import calibration_dialog
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.platform_text import SOUND_SETTINGS
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from capture_helpers import RATE, _voice, finish_take
from engine_fakes import lend_audio
from test_auto_eq_candidate_lifecycle import _Owner
from test_voice_setup_integration import real_main_window  # noqa: F401  (a fixture)

GOOD = _voice(0.3)
HOT = _voice(0.95)
QUIET = _voice(0.003)
SILENT = np.zeros(3 * RATE, dtype=np.float32)
UNREADABLE = np.full(1000, np.nan, dtype=np.float32)

HOT_MESSAGE = (
    "Your voice peaks at -0.4 dB, close to the limit: lower the microphone level "
    f"by about 6 dB (in {SOUND_SETTINGS} or on the microphone), then check again."
)


class _Session:
    """A capture session that records only what the dialog asks of it."""

    instances: list["_Session"] = []

    def __init__(self, *_args) -> None:
        self.started: list[float] = []
        self.cleaned = False
        _Session.instances.append(self)

    def start(self, duration, **_kwargs) -> None:
        self.started.append(duration)

    def cleanup(self) -> None:
        self.cleaned = True


# --- Voice Setup -------------------------------------------------------------------------


@pytest.fixture
def voice(real_main_window, monkeypatch):  # noqa: F811
    _Session.instances = []
    monkeypatch.setattr("mic_eq.ui.voice_setup_dialog.CaptureSession", _Session)
    dialog = VoiceSetupDialog(parent=real_main_window)
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: lend_audio(dialog))
    yield dialog
    dialog.reject()
    dialog.deleteLater()


def _check(dialog: VoiceSetupDialog, take: np.ndarray) -> None:
    """Start the level check and finish it with ``take``."""
    dialog._on_start_clicked()
    dialog._capture_start_timer.stop()
    dialog._on_recording_complete(take)


def test_voice_setup_opens_with_a_three_second_level_check(voice) -> None:
    voice._on_start_clicked()

    assert voice.setup_state == "level_check"
    assert voice._recording_duration == 3.0
    assert voice.view_model.shown["phaseText"] == "Checking your level"
    assert "3 seconds" in voice.view_model.warning_text
    assert voice.view_model.shown["step"] == 1 and voice.STEPS[1] == "Level"
    assert voice.view_model.shown["voiceShown"]  # the passage to speak from
    assert not voice.view_model.shown["noiseShown"]

    voice._capture_start_timer.stop()
    voice._begin_recording_capture()
    voice.recording_timer.stop()
    assert _Session.instances[-1].started == [3.0]  # the check is captured like a take


def test_a_good_level_goes_on_to_the_room_noise_with_one_click(voice) -> None:
    _check(voice, GOOD)

    assert voice.setup_state == "level_result"
    assert voice._level_verdict.code == "good"
    assert "good" in voice.view_model.warning_text
    assert f"{voice._level_verdict.peak_db:.1f} dB" in voice.view_model.warning_text
    assert voice.view_model.shown["startText"] == "Record Room Noise"
    assert not voice.view_model.shown["overrideShown"]

    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice._recording_duration == 2.0


def test_a_hot_level_stops_before_any_take_and_can_be_overridden_by_name(voice) -> None:
    _check(voice, HOT)

    assert voice.setup_state == "level_result"
    assert voice.view_model.warning_text.startswith(HOT_MESSAGE)
    assert "may clip" in voice.view_model.warning_text
    assert voice.view_model.shown["startText"] == "Check Level Again"
    assert voice.view_model.shown["overrideShown"]
    assert voice.view_model.shown["overrideText"] == "Continue Despite Clipping Risk"

    voice._on_start_clicked()  # the primary action checks again; it never records
    voice._capture_start_timer.stop()
    assert voice.setup_state == "level_check"

    voice._on_recording_complete(HOT)
    voice._continue_past_level()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice._level_override == "too_hot"


def test_a_quiet_level_may_continue_with_a_warning(voice) -> None:
    _check(voice, QUIET)

    assert voice.setup_state == "level_result"
    text = voice.view_model.warning_text
    assert text.startswith("Your voice is quiet:") and "-40 dB or louder" in text
    assert "less reliable" in text
    assert voice.view_model.shown["overrideText"] == "Continue Anyway"

    voice._continue_past_level()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice._level_override == "too_quiet"


@pytest.mark.parametrize(
    ("take", "start"),
    [(SILENT, "No sound came from the microphone."), (UNREADABLE, "The recording could not be read.")],
    ids=["silent", "unreadable"],
)
def test_nothing_heard_cannot_be_continued_past(voice, take, start) -> None:
    _check(voice, take)

    assert voice.setup_state == "level_result"
    assert voice.view_model.warning_text.startswith(start)
    assert not voice.view_model.shown["overrideShown"]
    assert voice.view_model.shown["startText"] == "Check Level Again"


def test_a_failed_check_returns_to_the_start(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._on_recording_failed("The microphone stopped sending sound.")

    assert voice.setup_state == "idle"
    assert voice.view_model.shown["startText"] == "Start Voice Setup"
    assert voice.view_model.shown["startEnabled"]
    assert not voice.view_model.shown["overrideShown"]


def test_closing_during_the_check_releases_the_capture(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()
    voice._begin_recording_capture()
    session = _Session.instances[-1]

    voice.reject()

    assert session.cleaned and voice.setup_state == "idle"
    assert voice.recording_timer.isActive() is False


def test_retaking_after_a_level_result_starts_over(voice, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes)
    _check(voice, HOT)

    voice._on_retake_clicked()

    assert voice.setup_state == "idle"
    assert not voice.view_model.shown["overrideShown"]
    assert voice._level_override is None


def test_the_check_meter_speaks_the_same_words_as_the_verdict(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._update_level(-20.0, -0.4)

    assert voice.view_model.warning_text == HOT_MESSAGE


# --- Auto-EQ -----------------------------------------------------------------------------


@pytest.fixture
def auto_eq(qapp, monkeypatch):
    _Session.instances = []
    monkeypatch.setattr(calibration_dialog, "CaptureSession", _Session)
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    yield dialog
    dialog.reject()
    owner.close()


def _check_auto(dialog: CalibrationDialog, take: np.ndarray) -> None:
    dialog._on_start_clicked()
    dialog._capture_start_timer.stop()
    dialog._on_recording_complete(take)


def test_auto_eq_opens_with_a_three_second_level_check(auto_eq) -> None:
    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    assert auto_eq.recording_state == "level_check"
    assert auto_eq.view_model.shown["startText"] == "Checking Level..."
    assert auto_eq.view_model.shown["step"] == 1 and auto_eq.STEPS[1] == "Record"
    assert "3 seconds" in auto_eq.view_model.warning_text

    auto_eq._begin_recording_capture()
    auto_eq.recording_timer.stop()
    assert _Session.instances[-1].started == [3.0]


def test_auto_eq_good_level_starts_the_ten_second_take_with_one_click(auto_eq) -> None:
    _check_auto(auto_eq, GOOD)

    assert auto_eq.recording_state == "level_result"
    assert auto_eq.view_model.shown["startText"] == "Record Passage"
    assert not auto_eq.view_model.shown["overrideShown"]

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    assert auto_eq.recording_state == "recording"
    auto_eq._begin_recording_capture()
    auto_eq.recording_timer.stop()
    assert _Session.instances[-1].started == [10.0]


def test_auto_eq_hot_level_blocks_the_take_until_named_override(auto_eq) -> None:
    _check_auto(auto_eq, HOT)

    assert auto_eq.view_model.warning_text.startswith(HOT_MESSAGE)
    assert auto_eq.view_model.shown["startText"] == "Check Level Again"
    assert auto_eq.view_model.shown["overrideText"] == "Continue Despite Clipping Risk"

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()
    assert auto_eq.recording_state == "level_check"

    auto_eq._on_recording_complete(HOT)
    auto_eq._continue_past_level()
    auto_eq._capture_start_timer.stop()

    assert auto_eq.recording_state == "recording"
    assert auto_eq._level_override == "too_hot"


def test_auto_eq_quiet_level_continues_with_a_warning(auto_eq) -> None:
    _check_auto(auto_eq, QUIET)

    verdict = auto_eq._level_verdict
    text = auto_eq.view_model.warning_text
    assert verdict.code == "too_quiet" and verdict.change_db > 0
    assert "quiet" in text and "-40 dB or louder" in text
    assert f"Raise the microphone level by about {verdict.change_db} dB" in text
    assert "less reliable" in text
    assert auto_eq.view_model.shown["overrideText"] == "Continue Anyway"


def test_auto_eq_nothing_heard_cannot_be_continued_past(auto_eq) -> None:
    _check_auto(auto_eq, SILENT)

    text = auto_eq.view_model.warning_text
    assert auto_eq._level_verdict.code == "silent"
    assert "No sound" in text and "not muted" in text and "right input is selected" in text
    assert not auto_eq.view_model.shown["overrideShown"]


def test_auto_eq_keeps_ownership_of_a_processor_it_started_for_the_check(auto_eq) -> None:
    owner = auto_eq.parent()
    starts: list[bool] = []
    owner.processor.is_running = lambda: bool(starts)  # running once started
    owner.processor.start = lambda *_route, **_endpoints: starts.append(True)
    owner.processor.stop = starts.clear
    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()
    lease = auto_eq._audio_lease
    assert lease is not None and lease.started
    auto_eq._on_recording_complete(GOOD)

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    assert auto_eq._audio_lease is lease and starts == [True]


@pytest.mark.parametrize(
    ("fixture", "update_method"),
    [("voice", "_update_level"), ("auto_eq", "_on_level_update")],
    ids=["voice_setup", "auto_eq"],
)
def test_the_meter_names_the_window_peak_in_a_pause(request, fixture, update_method) -> None:
    dialog = request.getfixturevalue(fixture)
    update = getattr(dialog, update_method)
    dialog._on_start_clicked()
    dialog._capture_start_timer.stop()

    update(-25.0, -7.0)  # a loud word
    for _ in range(14):
        update(-70.0, -34.8)  # the pause after it

    shown = dialog.view_model.warning_text if fixture == "auto_eq" else dialog.view_model.warning_text
    assert shown == "The level is good: your voice peaks at -7.0 dB."


def test_auto_eq_closing_during_the_check_releases_the_capture(auto_eq) -> None:
    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()
    auto_eq._begin_recording_capture()
    session = _Session.instances[-1]

    auto_eq.reject()

    assert session.cleaned and auto_eq.recording_state == "idle"


# --- The room-noise take ------------------------------------------------------------------


def _room_noise(click: bool = False) -> np.ndarray:
    take = (0.003 * np.random.default_rng(3).standard_normal(2 * RATE)).astype(np.float32)
    if click:
        take[40_000:40_005] = 1.0  # shorter than a flat top, so the take is not refused
    return take


def _loud_room_noise() -> np.ndarray:
    take = _room_noise()
    take[1000:1100] = 1.0
    return take


def _record_room_noise(voice: VoiceSetupDialog, take: np.ndarray) -> None:
    _check(voice, QUIET)
    voice._continue_past_level()
    voice._capture_start_timer.stop()
    assert voice.setup_state == "noise_recording"
    finish_take(voice, take)


def test_retake_room_noise_starts_the_room_noise_take_and_keeps_the_level_choice(voice) -> None:
    _record_room_noise(voice, _loud_room_noise())

    assert voice.setup_state == "idle"
    assert voice.view_model.shown["startText"] == "Retake Room Noise"
    text = voice.view_model.warning_text
    assert "A loud sound was picked up" in text
    assert "Record the room noise again" in text and "room is quiet" in text
    assert "play safe" not in text
    assert voice.parent().last_setup_result["outcome"] == "refused"

    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice.view_model.shown["phaseText"] == "Capturing room noise"
    assert voice._level_override == "too_quiet"
    assert voice._recording_duration == 2.0


def test_retaking_after_a_failed_room_noise_take_records_the_room_noise_too(voice) -> None:
    _check(voice, GOOD)
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._on_recording_failed("The microphone stopped sending sound.")

    assert voice.view_model.shown["startText"] == "Retake Room Noise"
    voice._on_start_clicked()
    voice._capture_start_timer.stop()
    assert voice.setup_state == "noise_recording"


def test_a_click_in_the_room_noise_is_pointed_out_and_setup_can_go_on(voice) -> None:
    _record_room_noise(voice, _room_noise(click=True))

    assert voice.setup_state == "noise_ready"
    assert voice.view_model.shown["startEnabled"] and voice.view_model.shown["startText"] == "Record Voice"
    text = voice.view_model.warning_text
    assert "A click or bump was heard" in text and "Press Retake and record the room noise again" in text
    assert "read the passage aloud" in text and "10 seconds" in text
    assert (voice.view_model.warning_state, voice.view_model.warning_strong) == ("warn", True)

    voice._on_start_clicked()
    voice._capture_start_timer.stop()
    assert voice.setup_state == "voice_recording"


def test_a_clean_room_noise_take_says_nothing_about_clicks(voice) -> None:
    _record_room_noise(voice, _room_noise())

    assert voice.setup_state == "noise_ready"
    text = voice.view_model.warning_text
    assert "read the passage aloud" in text and "10 seconds" in text
    assert "click" not in text
    assert (voice.view_model.warning_state, voice.view_model.warning_strong) == ("ok", True)


def test_the_live_message_is_not_restyled_while_it_stays_the_same(voice) -> None:
    styles: list[str] = []
    voice.view_model.statusChanged.connect(lambda: styles.append(voice.view_model.warning_state))

    voice._show_live_message("Room is quiet", "ok")
    voice._show_live_message("Room is quiet", "ok")
    voice._show_live_message("Room is quiet", "ok")
    assert len(styles) == 1

    voice._show_live_message("Room is noisy", "warn")
    assert len(styles) == 2 and voice.view_model.warning_text == "Room is noisy"


def test_auto_eq_logs_the_assumed_rate_when_the_processor_rate_cannot_be_read(
    auto_eq, monkeypatch, caplog
) -> None:
    def unreadable(_owner):
        raise RuntimeError("Processor sample rate is unavailable")

    monkeypatch.setattr(calibration_dialog, "_processor_sample_rate", unreadable)
    with caplog.at_level("WARNING", logger=calibration_dialog.logger.name):
        _check_auto(auto_eq, GOOD)

    assert auto_eq.recording_state == "level_result"
    assert "assuming 48000 Hz" in caplog.text and "unavailable" in caplog.text
