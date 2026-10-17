"""The shared microphone level verdicts."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mic_eq.analysis.voice_setup import raw_capture_is_clipped
from mic_eq.ui import level_check
from mic_eq.ui.level_check import (
    LEVEL_HOLD_UPDATES,
    clipped_message,
    live_verdict,
    room_noise_message,
    verdict_from_audio,
    verdict_from_levels,
)

RATE = 48_000


def _voice(level: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (level * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def test_a_comfortable_voice_is_good():
    verdict = verdict_from_audio(_voice(0.3), RATE)

    assert verdict.code == "good" and not verdict.can_override
    assert "-10.5 dB" in verdict.message


def test_a_hot_voice_gets_the_lowering_advice_from_its_peak():
    verdict = verdict_from_audio(_voice(0.95), RATE)  # -0.4 dB, not clipped

    assert verdict.code == "too_hot"
    assert "-0.4 dB" in verdict.message and "by about 6 dB" in verdict.message
    assert "clipping" not in verdict.message
    assert verdict.change_db == -6
    assert verdict.can_override


def test_a_clipped_voice_is_hot_and_says_so():
    take = _voice(0.5)
    take[1000:1010] = 1.0  # a flat top

    verdict = verdict_from_audio(take, RATE)

    assert verdict.code == "too_hot"
    assert "clipping" in verdict.message


def test_one_stray_sample_does_not_make_a_quiet_voice_hot():
    take = _voice(0.2)
    take[500] = 1.0

    assert verdict_from_audio(take, RATE).code == "good"


def test_a_quiet_voice_may_continue_with_a_warning_and_says_how_much_to_raise():
    verdict = verdict_from_audio(_voice(0.003), RATE)  # about -50 dB

    assert verdict.code == "too_quiet"
    assert verdict.can_override
    assert "-40 dB or louder" in verdict.message
    assert f"by about {verdict.change_db} dB" in verdict.message
    # What the advice asks for lands the peak at the target, not over the limit.
    assert verdict.peak_db is not None and verdict.peak_db + verdict.change_db <= -6.0 + 1.0


def test_pauses_between_words_do_not_make_a_voice_quiet():
    take = _voice(0.3)
    take[RATE // 2 : RATE] = 0.0
    take[2 * RATE :] = 0.0

    assert verdict_from_audio(take, RATE).code == "good"


@pytest.mark.parametrize(
    ("take", "code"),
    [
        (np.zeros(3 * RATE, dtype=np.float32), "silent"),
        (np.full(100, np.nan, dtype=np.float32), "unreadable"),
        (np.zeros(0, dtype=np.float32), "unreadable"),
    ],
)
def test_silence_and_unreadable_audio_cannot_continue(take, code):
    verdict = verdict_from_audio(take, RATE)

    assert verdict.code == code
    assert not verdict.can_override
    if code == "silent":
        assert "muted" in verdict.message


def test_the_levels_form_gives_the_same_verdict_as_the_audio_form():
    assert verdict_from_levels(-0.4, -10.0).code == "too_hot"
    assert verdict_from_levels(-12.0, -45.0).code == "too_quiet"
    assert verdict_from_levels(-12.0, -20.0).code == "good"


def test_the_live_verdict_waits_out_the_pauses_and_says_a_peak_at_once():
    assert live_verdict([-70.0] * (LEVEL_HOLD_UPDATES - 1), [-60.0] * (LEVEL_HOLD_UPDATES - 1)) is None

    quiet = live_verdict([-70.0] * LEVEL_HOLD_UPDATES, [-60.0] * LEVEL_HOLD_UPDATES)
    assert quiet is not None and quiet.code == "too_quiet"

    good = live_verdict([-25.0] * 3 + [-70.0] * 12, [-12.0] * 15)
    assert good is not None and good.code == "good"

    hot = live_verdict([-20.0], [-1.0])
    assert hot is not None and hot.code == "too_hot" and "-1.0 dB" in hot.message


def test_the_live_verdict_names_the_loudest_peak_of_the_window_not_the_latest_block():
    # A loud word, then a pause: the latest block's peak is -34.8 dB, the window's is -7.0 dB.
    rms = [-25.0] * 3 + [-70.0] * 12
    peaks = [-7.0] + [-12.0] * 2 + [-34.8] * 12

    verdict = live_verdict(rms, peaks)

    assert verdict is not None and verdict.code == "good"
    assert "-7.0 dB" in verdict.message and "-34.8" not in verdict.message
    assert verdict.peak_db == -7.0


def test_a_level_just_below_zero_prints_as_zero():
    hot = verdict_from_levels(-0.04, -10.0)
    clipped = verdict_from_levels(-0.04, -10.0, clipped=True)

    assert "peaks at 0.0 dB, close to the limit" in hot.message
    assert "peaks reach 0.0 dB" in clipped.message
    assert "-0.0" not in hot.message + clipped.message
    assert room_noise_message([-0.3])[0].startswith("The room is loud (0 dB)")


def test_the_room_noise_message_names_the_level_when_the_room_is_loud():
    message, state = room_noise_message([-50.0, -30.0])

    assert state == "warn" and "-30 dB" in message
    quiet, quiet_state = room_noise_message([-50.0])
    assert quiet_state == "ok" and quiet != message and "loud" not in quiet


def _clicked(width: int, level: float = 0.05) -> np.ndarray:
    take = _voice(level)
    take[5000 : 5000 + width] = 1.0
    return take


@pytest.mark.parametrize(("width", "code"), [(1, "good"), (3, "good"), (6, "too_hot")])
def test_a_click_is_judged_by_the_rule_that_refuses_a_take(width, code):
    take = _clicked(width)

    verdict = verdict_from_audio(take, RATE)

    assert verdict.code == code
    assert raw_capture_is_clipped(take, RATE) == (code == "too_hot")
    if code == "too_hot":
        assert "clipping" in verdict.message
        assert verdict.peak_db is not None and verdict.peak_db > -0.1


def test_the_advice_says_where_to_change_the_level():
    hot = verdict_from_levels(-0.4, -10.0)
    quiet = verdict_from_levels(-12.0, -45.0)

    assert "(in Windows or on the microphone)" in hot.message
    assert "(in Windows or on the microphone)" in quiet.message


def test_a_microphone_far_below_the_target_is_not_asked_for_an_impossible_gain():
    verdict = verdict_from_levels(-60.0, -90.0)

    assert verdict.code == "too_quiet" and 0 < verdict.change_db <= 30
    assert "as far as it goes" in verdict.message and "63" not in verdict.message
    assert "by about" in verdict_from_levels(-12.0, -45.0).message


def test_a_voice_that_starts_in_the_last_third_of_a_second_is_not_called_quiet():
    take = _voice(0.0005)
    take[int(2.7 * RATE) :] = _voice(0.3, 0.3)

    assert verdict_from_audio(take, RATE).code == "good"


@pytest.mark.parametrize("level", [0.003, 0.3, 0.0])
def test_the_clipped_take_message_never_asks_for_a_zero_or_negative_amount(level):
    take = _voice(level) if level else _voice(0.3)
    if not level:
        take[1000:1010] = 1.0

    message = clipped_message(take, RATE)

    found = re.search(r"by about (-?\d+) dB", message)
    assert found is not None and int(found.group(1)) > 0
    assert "(in Windows or on the microphone)" in message


def test_a_voice_just_under_the_threshold_is_printed_below_it():
    message = verdict_from_levels(-10.0, -40.4).message

    assert "-41 dB when you speak" in message
    assert "-40 dB when you speak" not in message
    assert "needs -40 dB or louder" in message


def test_a_take_with_almost_no_speech_asks_for_speech_instead_of_a_raise():
    take = np.random.default_rng(1).standard_normal(3 * RATE).astype(np.float32) * 1e-4
    take[-RATE // 5 :] += _voice(0.3, 0.2)  # 0.2 s of speech

    verdict = verdict_from_audio(take, RATE)

    assert verdict.code == "too_quiet" and verdict.can_override
    assert "speech" in verdict.message and "Speak" in verdict.message
    assert "Raise" not in verdict.message and verdict.change_db == 0
    assert "-80 dB" in verdict_from_levels(-10.0, -80.0).message


@pytest.mark.parametrize(
    "verdict",
    [
        verdict_from_levels(-3.5, -41.0),
        live_verdict([-42.0] * LEVEL_HOLD_UPDATES, [0.0] * LEVEL_HOLD_UPDATES),
    ],
)
def test_a_quiet_voice_with_a_peak_near_the_limit_is_not_told_to_raise_the_level(verdict):
    assert verdict.code == "too_quiet" and verdict.can_override
    assert verdict.change_db == 0
    assert "Raise" not in verdict.message and "by about" not in verdict.message
    assert "Move closer" in verdict.message and "speak more evenly" in verdict.message


@pytest.mark.parametrize(("peak", "speech"), [(-10.0, -41.5), (-5.0, -42.0), (-12.0, -45.0), (-8.0, -40.4)])
def test_the_advised_raise_crosses_the_threshold_without_making_the_peak_too_hot(peak, speech):
    verdict = verdict_from_levels(peak, speech)

    assert verdict.code == "too_quiet" and verdict.change_db > 0
    assert speech + verdict.change_db >= -40.0
    assert peak + verdict.change_db <= -3.0


@pytest.fixture
def lower_target(monkeypatch):
    monkeypatch.setattr(level_check, "LOWER_LEVEL_TARGET", True)


def test_the_lower_level_target_is_off_by_default():
    assert level_check.LOWER_LEVEL_TARGET is False
    assert (level_check.too_loud_db(), level_check.target_peak_db()) == (-3.0, -6.0)


@pytest.mark.parametrize(
    ("peak", "code", "change"),
    [(-3.1, "good", 0), (-3.0, "good", 0), (-2.9, "too_hot", -4), (-5.9, "good", 0)],
)
def test_with_the_switch_off_the_old_boundaries_hold(peak, code, change):
    verdict = verdict_from_levels(peak, -20.0)

    assert (verdict.code, verdict.change_db) == (code, change)
    assert level_check.target_peak_db() == -6.0


def test_with_the_switch_on_the_limit_is_minus_eight(lower_target):
    assert verdict_from_levels(-8.1, -20.0).code == "good"
    assert verdict_from_levels(-8.0, -20.0).code == "good"
    assert verdict_from_levels(-7.9, -20.0).code == "too_hot"
    assert level_check.too_loud_db() == -8.0 and level_check.target_peak_db() == -11.0


@pytest.mark.parametrize(("peak", "lower"), [(-7.9, 4), (-5.0, 6), (-0.4, 11)])
def test_with_the_switch_on_the_lowering_advice_lands_at_minus_eleven(lower_target, peak, lower):
    verdict = verdict_from_levels(peak, -20.0)

    assert verdict.code == "too_hot" and verdict.change_db == -lower
    assert f"by about {lower} dB" in verdict.message
    assert -11.0 - 1.0 < peak - lower <= -11.0 + 0.05


def test_with_the_switch_on_a_hot_take_from_audio_and_a_clipped_one_aim_at_minus_eleven(lower_target):
    hot = verdict_from_audio(_voice(0.5), RATE)  # -6.0 dB
    assert (hot.code, hot.change_db) == ("too_hot", -5)
    assert verdict_from_audio(_voice(0.35), RATE).code == "good"  # -9.1 dB
    assert "by about 11 dB" in clipped_message(_voice(0.95), RATE)


def test_with_the_switch_on_a_quiet_voices_raise_stops_at_the_new_target(lower_target):
    verdict = verdict_from_levels(-20.0, -45.0)  # the speech target asks for +20, the peak target for +9

    assert verdict.code == "too_quiet" and verdict.change_db == 9
    assert verdict.peak_db is not None and verdict.peak_db + verdict.change_db == -11.0


def test_with_the_switch_on_a_quiet_peak_near_the_new_limit_is_not_told_to_raise(lower_target):
    verdict = verdict_from_levels(-8.5, -41.0)  # +1 dB reaches -40 but the peak would pass -8

    assert verdict.code == "too_quiet" and verdict.change_db == 0
    assert "Move closer" in verdict.message


def test_with_the_switch_on_the_live_verdict_uses_the_new_limit(lower_target):
    hot = live_verdict([-20.0], [-7.0])
    assert hot is not None and hot.code == "too_hot"
    assert live_verdict([-20.0], [-9.0]) is None
    quiet = live_verdict([-42.0] * LEVEL_HOLD_UPDATES, [-7.0] * LEVEL_HOLD_UPDATES)
    assert quiet is not None and quiet.code == "too_quiet"


def test_with_the_switch_off_the_live_verdict_and_raise_are_unchanged():
    assert live_verdict([-20.0], [-7.0]) is None
    assert verdict_from_levels(-20.0, -45.0).change_db == 14
