"""The 3-second level check that opens Voice Setup and Auto-EQ."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import numpy as np
import pytest

from mic_eq.ui import calibration_dialog
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.theme import message_text_style
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from test_auto_eq_candidate_lifecycle import _Owner
from test_voice_setup_integration import real_main_window  # noqa: F401  (a fixture)

RATE = 48_000


def _voice(level: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (level * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


GOOD = _voice(0.3)
HOT = _voice(0.95)
QUIET = _voice(0.003)
SILENT = np.zeros(3 * RATE, dtype=np.float32)
UNREADABLE = np.full(1000, np.nan, dtype=np.float32)

HOT_MESSAGE = (
    "Your voice peaks at -0.4 dB, close to the limit: lower the microphone level "
    "by about 6 dB (in Windows or on the microphone), then check again."
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
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: True)
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
    assert voice.phase_label.text() == "Checking your level"
    assert "3 seconds" in voice.warning_label.text()
    assert voice.step_header._labels[1].text() == "Level"
    assert voice.voice_group.isHidden() is False  # the passage to speak from
    assert voice.noise_group.isHidden()


def test_the_check_is_captured_like_a_take(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()
    voice._begin_recording_capture()
    voice.recording_timer.stop()

    assert _Session.instances[-1].started == [3.0]


def test_a_good_level_goes_on_to_the_room_noise_with_one_click(voice) -> None:
    _check(voice, GOOD)

    assert voice.setup_state == "level_result"
    assert voice._level_verdict.code == "good"
    assert "good" in voice.warning_label.text()
    assert f"{voice._level_verdict.peak_db:.1f} dB" in voice.warning_label.text()
    assert voice.start_button.text() == "Record Room Noise"
    assert voice.level_override_btn.isHidden()

    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice._recording_duration == 2.0


def test_a_hot_level_stops_before_any_take_and_can_be_overridden_by_name(voice) -> None:
    _check(voice, HOT)

    assert voice.setup_state == "level_result"
    assert voice.warning_label.text().startswith(HOT_MESSAGE)
    assert "may clip" in voice.warning_label.text()
    assert voice.start_button.text() == "Check Level Again"
    assert voice.level_override_btn.isHidden() is False
    assert voice.level_override_btn.text() == "Continue Despite Clipping Risk"

    voice._on_start_clicked()  # the primary action checks again; it never records
    voice._capture_start_timer.stop()
    assert voice.setup_state == "level_check"

    voice._on_recording_complete(HOT)
    voice.level_override_btn.click()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice._level_override == "too_hot"


def test_a_quiet_level_may_continue_with_a_warning(voice) -> None:
    _check(voice, QUIET)

    assert voice.setup_state == "level_result"
    text = voice.warning_label.text()
    assert text.startswith("Your voice is quiet:") and "-40 dB or louder" in text
    assert "less reliable" in text
    assert voice.level_override_btn.text() == "Continue Anyway"

    voice.level_override_btn.click()
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
    assert voice.warning_label.text().startswith(start)
    assert voice.level_override_btn.isHidden()
    assert voice.start_button.text() == "Check Level Again"


def test_a_failed_check_returns_to_the_start(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._on_recording_failed("The microphone stopped sending sound.")

    assert voice.setup_state == "idle"
    assert voice.start_button.text() == "Start Voice Setup"
    assert voice.start_button.isEnabled()
    assert voice.level_override_btn.isHidden()


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
    assert voice.level_override_btn.isHidden()
    assert voice._level_override is None


def test_the_check_meter_speaks_the_same_words_as_the_verdict(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._update_level(-20.0, -0.4)

    assert voice.warning_label.text() == HOT_MESSAGE


def test_the_voice_setup_meter_names_the_window_peak_in_a_pause(voice) -> None:
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._update_level(-25.0, -7.0)  # a loud word
    for _ in range(14):
        voice._update_level(-70.0, -34.8)  # the pause after it

    assert voice.warning_label.text() == "The level is good: your voice peaks at -7.0 dB."


# --- Auto-EQ -----------------------------------------------------------------------------


@pytest.fixture
def auto_eq(qapp, monkeypatch):
    _Session.instances = []
    monkeypatch.setattr(calibration_dialog, "CaptureSession", _Session)
    monkeypatch.setattr(calibration_dialog, "_start_selected_route", lambda _owner: None)
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
    assert auto_eq.start_button.text() == "Checking Level..."
    assert auto_eq.step_header._labels[1].text() == "Record"
    assert "3 seconds" in auto_eq.warning_label.text()

    auto_eq._begin_recording_capture()
    auto_eq.recording_timer.stop()
    assert _Session.instances[-1].started == [3.0]


def test_auto_eq_good_level_starts_the_ten_second_take_with_one_click(auto_eq) -> None:
    _check_auto(auto_eq, GOOD)

    assert auto_eq.recording_state == "level_result"
    assert auto_eq.start_button.text() == "Record Passage"
    assert auto_eq.level_override_btn.isHidden()

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    assert auto_eq.recording_state == "recording"
    auto_eq._begin_recording_capture()
    auto_eq.recording_timer.stop()
    assert _Session.instances[-1].started == [10.0]


def test_auto_eq_hot_level_blocks_the_take_until_named_override(auto_eq) -> None:
    _check_auto(auto_eq, HOT)

    assert auto_eq.warning_label.text().startswith(HOT_MESSAGE)
    assert auto_eq.start_button.text() == "Check Level Again"
    assert auto_eq.level_override_btn.text() == "Continue Despite Clipping Risk"

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()
    assert auto_eq.recording_state == "level_check"

    auto_eq._on_recording_complete(HOT)
    auto_eq.level_override_btn.click()
    auto_eq._capture_start_timer.stop()

    assert auto_eq.recording_state == "recording"
    assert auto_eq._level_override == "too_hot"


def test_auto_eq_quiet_level_continues_with_a_warning(auto_eq) -> None:
    _check_auto(auto_eq, QUIET)

    verdict = auto_eq._level_verdict
    text = auto_eq.warning_label.text()
    assert verdict.code == "too_quiet" and verdict.change_db > 0
    assert "quiet" in text and "-40 dB or louder" in text
    assert f"Raise the microphone level by about {verdict.change_db} dB" in text
    assert "less reliable" in text
    assert auto_eq.level_override_btn.text() == "Continue Anyway"


def test_auto_eq_nothing_heard_cannot_be_continued_past(auto_eq) -> None:
    _check_auto(auto_eq, SILENT)

    text = auto_eq.warning_label.text()
    assert auto_eq._level_verdict.code == "silent"
    assert "No sound" in text and "not muted" in text and "right input is selected" in text
    assert auto_eq.level_override_btn.isHidden()


def test_auto_eq_keeps_ownership_of_a_processor_it_started_for_the_check(auto_eq) -> None:
    owner = auto_eq.parent()
    owner.processor.is_running = lambda: auto_eq._started_processor  # running once started
    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()
    assert auto_eq._started_processor is True
    auto_eq._on_recording_complete(GOOD)

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    assert auto_eq._started_processor is True


def test_auto_eq_meter_names_the_window_peak_in_a_pause(auto_eq) -> None:
    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    auto_eq._on_level_update(-25.0, -7.0)  # a loud word
    for _ in range(14):
        auto_eq._on_level_update(-70.0, -34.8)  # the pause after it

    assert auto_eq.warning_label.text() == "The level is good: your voice peaks at -7.0 dB."


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
    voice.level_override_btn.click()
    voice._capture_start_timer.stop()
    assert voice.setup_state == "noise_recording"
    voice._on_recording_complete(take)


def test_retake_room_noise_starts_the_room_noise_take_and_keeps_the_level_choice(voice) -> None:
    _record_room_noise(voice, _loud_room_noise())

    assert voice.setup_state == "idle"
    assert voice.start_button.text() == "Retake Room Noise"
    text = voice.warning_label.text()
    assert "A loud sound was picked up" in text
    assert "Record the room noise again" in text and "room is quiet" in text
    assert "play safe" not in text
    assert voice.parent().last_setup_result["outcome"] == "refused"

    voice.start_button.click()
    voice._capture_start_timer.stop()

    assert voice.setup_state == "noise_recording"
    assert voice.phase_label.text() == "Capturing room noise"
    assert voice._level_override == "too_quiet"
    assert voice._recording_duration == 2.0


def test_retaking_after_a_failed_room_noise_take_records_the_room_noise_too(voice) -> None:
    _check(voice, GOOD)
    voice._on_start_clicked()
    voice._capture_start_timer.stop()

    voice._on_recording_failed("The microphone stopped sending sound.")

    assert voice.start_button.text() == "Retake Room Noise"
    voice.start_button.click()
    voice._capture_start_timer.stop()
    assert voice.setup_state == "noise_recording"


def test_a_click_in_the_room_noise_is_pointed_out_and_setup_can_go_on(voice) -> None:
    _record_room_noise(voice, _room_noise(click=True))

    assert voice.setup_state == "noise_ready"
    assert voice.start_button.isEnabled() and voice.start_button.text() == "Record Voice"
    text = voice.warning_label.text()
    assert "A click or bump was heard" in text and "Press Retake and record the room noise again" in text
    assert "read the passage aloud" in text and "10 seconds" in text
    assert voice.warning_label.styleSheet() == message_text_style("warn", strong=True)

    voice.start_button.click()
    voice._capture_start_timer.stop()
    assert voice.setup_state == "voice_recording"


def test_a_clean_room_noise_take_says_nothing_about_clicks(voice) -> None:
    _record_room_noise(voice, _room_noise())

    assert voice.setup_state == "noise_ready"
    text = voice.warning_label.text()
    assert "read the passage aloud" in text and "10 seconds" in text
    assert "click" not in text
    assert voice.warning_label.styleSheet() == message_text_style("ok", strong=True)


def test_the_start_button_is_named_by_what_it_says(voice, auto_eq) -> None:
    from PySide6.QtGui import QAccessible

    def accessible_name(button) -> str:
        interface = QAccessible.queryAccessibleInterface(button)
        assert interface is not None
        return interface.text(QAccessible.Text.Name)

    for dialog, take in ((voice, GOOD), (auto_eq, HOT)):
        assert accessible_name(dialog.start_button) == dialog.start_button.text()
        dialog._on_start_clicked()
        dialog._capture_start_timer.stop()
        dialog._on_recording_complete(take)
        assert accessible_name(dialog.start_button) == dialog.start_button.text()


def _says_live_output_is_muted(label) -> bool:
    text = label.text().lower()
    # A screen reader reads the sentence itself, not a fixed name.
    read_aloud = label.accessibleName() in ("", label.text())
    return label.isVisibleTo(label.window()) and read_aloud and "muted" in text and "call" in text


def test_voice_setup_says_the_live_output_is_muted_while_it_records(voice) -> None:
    assert _says_live_output_is_muted(voice.mute_notice)

    voice._on_start_clicked()

    assert _says_live_output_is_muted(voice.mute_notice)


def test_auto_eq_says_the_live_output_is_muted_while_it_records(auto_eq) -> None:
    assert _says_live_output_is_muted(auto_eq.mute_notice)

    auto_eq._on_start_clicked()
    auto_eq._capture_start_timer.stop()

    assert _says_live_output_is_muted(auto_eq.mute_notice)
