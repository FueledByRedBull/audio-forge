"""The shared microphone level verdicts."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import re

import numpy as np
import pytest

from collections import deque

from mic_eq.analysis.voice_setup import SPEECH_MIN_DURATION_S, raw_capture_is_clipped
from mic_eq.ui import level_check
from mic_eq.ui.calibration_support import TOO_QUIET_DB
from mic_eq.ui.level_check import (
    LEVEL_HOLD_UPDATES,
    clipped_message,
    level_text,
    live_verdict,
    room_noise_message,
    take_rules,
    verdict_from_audio,
    verdict_from_levels,
)
from mic_eq.ui.platform_text import SOUND_SETTINGS
from capture_helpers import RATE, _voice


def test_a_comfortable_voice_is_good():
    verdict = verdict_from_audio(_voice(0.3), RATE)

    assert verdict.code == "good" and not verdict.can_override
    assert "-10.5 dB" in verdict.message


def test_a_hot_voice_gets_the_lowering_advice_from_its_peak():
    verdict = verdict_from_audio(_voice(0.95), RATE)  # -0.4 dB, not clipped

    assert verdict.code == "too_hot"
    assert "-0.4 dB" in verdict.message and "clipping" not in verdict.message
    assert "lower the microphone level by about 6 dB" in verdict.message
    assert "then check again" in verdict.message
    assert verdict.change_db == -6
    assert verdict.can_override


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
        assert verdict.message.startswith("No sound came from the microphone.")
        assert "not muted" in verdict.message


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
    assert room_noise_message([-50.0]) == ("The room is quiet enough.", "ok")


def _clicked(width: int, level: float, position: int) -> np.ndarray:
    take = _voice(level)
    take[position : position + width] = 1.0
    return take


@pytest.mark.parametrize(
    ("width", "level", "position", "code"),
    [
        (1, 0.05, 5000, "good"),
        (3, 0.05, 5000, "good"),
        (6, 0.05, 5000, "too_hot"),
        (10, 0.5, 1000, "too_hot"),
        (1, 0.2, 500, "good"),
    ],
    ids=["one_sample", "three_samples", "six_samples", "flat_top_in_a_loud_voice", "stray_sample_in_a_quiet_voice"],
)
def test_a_click_is_judged_by_the_rule_that_refuses_a_take(width, level, position, code):
    take = _clicked(width, level, position)

    verdict = verdict_from_audio(take, RATE)

    assert verdict.code == code
    assert raw_capture_is_clipped(take, RATE) == (code == "too_hot")
    if code == "too_hot":
        assert verdict.message.startswith("Your voice is clipping")
        assert verdict.peak_db is not None and verdict.peak_db > -0.1


def test_the_advice_says_where_to_change_the_level():
    hot = verdict_from_levels(-0.4, -10.0)
    quiet = verdict_from_levels(-12.0, -45.0)

    assert f"(in {SOUND_SETTINGS} or on the microphone)" in hot.message
    assert f"(in {SOUND_SETTINGS} or on the microphone)" in quiet.message


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
    assert f"(in {SOUND_SETTINGS} or on the microphone)" in message


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
    assert verdict.message.startswith("Little or no speech was heard")
    assert "Speak for the whole check, then check again." in verdict.message
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


@pytest.mark.parametrize(
    ("peak", "code", "change"),
    [(-3.0, "good", 0), (-2.9, "too_hot", -4), (-5.9, "good", 0)],
)
def test_the_hot_boundary_and_the_lowering_advice(peak, code, change):
    verdict = verdict_from_levels(peak, -20.0)

    assert (verdict.code, verdict.change_db) == (code, change)
    assert level_check.TARGET_PEAK_DB == -6.0


def test_the_live_verdict_and_the_raise_follow_the_limit():
    assert live_verdict([-20.0], [-7.0]) is None
    assert verdict_from_levels(-20.0, -45.0).change_db == 14


def _rule(rules, name):
    return next(rule for rule in rules if rule.name == name)


@pytest.mark.parametrize(
    ("loudest", "speech_s", "clipped", "states"),
    [
        (-40.1, 3.1, False, {"Too quiet": "watch", "Clipped": "fine", "Too short": "fine"}),
        (-39.9, 3.1, False, {"Too quiet": "fine", "Clipped": "fine", "Too short": "fine"}),
        (-39.9, 2.9, False, {"Too quiet": "fine", "Clipped": "fine", "Too short": "watch"}),
        (-20.0, 5.0, True, {"Too quiet": "fine", "Clipped": "refused", "Too short": "fine"}),
    ],
)
def test_take_rules_warn_on_quiet_and_refuse_clipped_and_short_takes(loudest, speech_s, clipped, states):
    rms = [-70.0] * (LEVEL_HOLD_UPDATES - 1) + [loudest]
    rules = take_rules(rms, [loudest + 10.0] * LEVEL_HOLD_UPDATES, speech_s, clipped)

    assert [rule.name for rule in rules] == ["Too quiet", "Clipped", "Too short"]
    assert {rule.name: rule.state for rule in rules} == states
    words = {"fine": "Fine so far", "watch": "Keep reading"}
    for rule in rules:
        if rule.state in words and rule.name != "Too quiet":
            assert rule.status_word == words[rule.state], rule
    quiet = _rule(rules, "Too quiet")
    assert quiet.detail == f"Your words under {TOO_QUIET_DB:.0f} dB"
    # A quiet take is never refused, so its row warns rather than crossing it out.
    assert quiet.status_word == ("Too quiet now" if states["Too quiet"] == "watch" else "Fine so far")
    if clipped:
        assert _rule(rules, "Clipped").status_word == "Clipped"


def test_too_short_rule_names_the_real_minimum(monkeypatch):
    rule = _rule(take_rules([], [], 0.0, False), "Too short")
    assert rule.detail == f"Under {SPEECH_MIN_DURATION_S:g} seconds recorded"
    assert "3 seconds" in rule.detail

    monkeypatch.setattr(level_check, "SPEECH_MIN_DURATION_S", 5.0)
    assert _rule(take_rules([], [], 4.9, False), "Too short").state == "watch"
    assert _rule(take_rules([], [], 5.0, False), "Too short").state == "fine"
    assert "Under 5 seconds" in _rule(take_rules([], [], 0.0, False), "Too short").detail


def test_a_quiet_row_waits_out_the_pauses_like_the_live_verdict():
    rms: deque[float] = deque(maxlen=LEVEL_HOLD_UPDATES)
    peaks: deque[float] = deque(maxlen=LEVEL_HOLD_UPDATES)
    states = []
    for level in [-70.0] * LEVEL_HOLD_UPDATES + [-25.0] + [-70.0] * LEVEL_HOLD_UPDATES:
        rms.append(level)
        peaks.append(level + 10.0)
        states.append(_rule(take_rules(rms, peaks, 5.0, False), "Too quiet").status_word)

    assert states[: LEVEL_HOLD_UPDATES - 1] == ["Keep reading"] * (LEVEL_HOLD_UPDATES - 1)
    assert states[LEVEL_HOLD_UPDATES - 1] == "Too quiet now"
    # One loud block holds the row at fine for as long as it stays in the window.
    assert states[LEVEL_HOLD_UPDATES : 2 * LEVEL_HOLD_UPDATES] == ["Fine so far"] * LEVEL_HOLD_UPDATES
    assert states[2 * LEVEL_HOLD_UPDATES] == "Too quiet now"


@pytest.mark.parametrize(
    ("rms", "peaks"),
    [
        ([], []),
        ([-70.0] * (LEVEL_HOLD_UPDATES - 1), [-60.0] * (LEVEL_HOLD_UPDATES - 1)),
        ([-70.0] * LEVEL_HOLD_UPDATES, [-60.0] * LEVEL_HOLD_UPDATES),
        ([-40.0] * LEVEL_HOLD_UPDATES, [-30.0] * LEVEL_HOLD_UPDATES),
        ([-45.0] * LEVEL_HOLD_UPDATES, [-1.0] * LEVEL_HOLD_UPDATES),
        ([-20.0], [-1.0]),
        ([-25.0] * 3 + [-70.0] * 12, [-12.0] * 15),
        ([-50.0] * LEVEL_HOLD_UPDATES, [-0.5] * LEVEL_HOLD_UPDATES),
    ],
)
def test_the_quiet_row_warns_exactly_when_the_live_verdict_says_too_quiet(rms, peaks):
    verdict = live_verdict(rms, peaks)
    rule = _rule(take_rules(rms, peaks, 5.0, False), "Too quiet")

    assert (rule.status_word == "Too quiet now") == (verdict is not None and verdict.code == "too_quiet")
    assert rule.state != "refused"


@pytest.mark.parametrize(
    "take",
    [
        _voice(0.3),
        np.clip(_voice(2.0), -1.0, 1.0),
        np.concatenate([_voice(0.3), np.ones(int(RATE * 0.001), dtype=np.float32)]),
        _voice(0.3, seconds=2.0),
    ],
)
def test_the_clip_and_length_rows_follow_the_checks_that_refuse_a_take(take):
    clipped = raw_capture_is_clipped(take, RATE)
    rules = take_rules([], [], take.size / RATE, clipped)

    assert (_rule(rules, "Clipped").state == "refused") == clipped
    # The analysis refuses a voice capture shorter than this many samples.
    too_short = take.size < int(RATE * SPEECH_MIN_DURATION_S)
    assert (_rule(rules, "Too short").state != "fine") == too_short


def test_the_level_reading_names_the_level_and_the_peak():
    assert level_text(-22.4, -8.43) == "Input level -22 dB \u00b7 peak -8.4 dB"
    assert level_text(-0.2, -0.04) == "Input level 0 dB \u00b7 peak 0.0 dB"
