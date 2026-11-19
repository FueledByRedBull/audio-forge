"""The Overview's status sentence and the chain rows' lines, without Qt."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from mic_eq.ui.chain_presentation import ChainInputs, describe_chain
from mic_eq.ui.overview_presentation import (
    MUTE_PENDING_ISSUE,
    RECOMMEND_SETUP_HINT,
    UNMUTE_PENDING_ISSUE,
    OverviewInputs,
    describe_overview,
)

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
LIVE = OverviewInputs(
    now=NOW, running=True, input_name="Microphone", output_name="CABLE Input",
    backend="RNNoise", output_peak_db=-3.4,
)


@pytest.mark.parametrize(
    ("changes", "state"),
    [
        ({}, "live"),
        ({"issues": ("Lower the input gain.",)}, "live_issue"),
        ({"issues": ("a",), "muted": True}, "muted"),
        # Callers hear silence while muted, so the mute outranks a clipping microphone.
        ({"issues": ("a",), "muted": True, "input_bad": True}, "muted"),
        ({"input_bad": True}, "clipping"),
        ({"issues": ("a",), "muted": True, "input_bad": True, "device_bad": True}, "device_lost"),
        # A refused mute is no mute: what callers get is still the microphone.
        ({"mute_error": True}, "live_issue"),
        ({"mute_error": True, "input_bad": True}, "clipping"),
        # No microphone is a stopped state; running audio whose device went away has lost it.
        ({"muted": True, "input_bad": True, "device_bad": True, "no_mic": True}, "device_lost"),
        ({"no_mic": True}, "live"),
        ({"running": False, "muted": True, "input_bad": True, "device_bad": True, "no_mic": True}, "no_mic"),
        ({"running": False}, "ready"),
        ({"running": False, "restart_pending": True}, "device_lost"),
        ({"running": False, "no_mic": True, "restart_pending": True}, "no_mic"),
        # A device judged bad while stopped is stale; stopped audio is just ready.
        ({"running": False, "device_bad": True, "input_bad": True}, "ready"),
    ],
)
def test_overview_priority_order(changes, state) -> None:
    assert describe_overview(replace(LIVE, **changes)).state == state


def test_restart_is_offered_only_for_running_audio_whose_device_failed() -> None:
    assert describe_overview(replace(LIVE, device_bad=True)).offer_restart
    # Restarting would cancel the engine's own retry, and nothing would reconnect gone devices.
    assert not describe_overview(replace(LIVE, running=False, restart_pending=True)).offer_restart
    for changes in ({}, {"running": False}, {"input_bad": True}, {"muted": True}, {"no_mic": True, "running": False}):
        assert not describe_overview(replace(LIVE, **changes)).offer_restart, changes


def test_each_state_says_what_is_happening_in_words() -> None:
    def view(**changes):
        return describe_overview(replace(LIVE, **changes))

    live = view()
    assert (live.headline, live.detail, live.tone) == (
        "Live · your voice is clean", "RNNoise · peaks under -3 dB", "ok",
    )
    assert live.nav_line == "Live · all stages"
    assert view(backend="").detail == "Noise suppression off · peaks under -3 dB"
    assert view(mode="bypass").headline == "Live · Bypass, voice chain off"
    assert view(mode="raw").headline == "Live · Raw Monitor, no processing"
    assert view(mode="raw").detail == "Peaks under -3 dB"

    ready = view(running=False)
    assert ready.headline == "Ready when you are" and ready.tone == "idle"
    assert ready.detail == "Microphone > CABLE Input · Default preset"
    assert ready.nav_line == "Ready · not running"
    assert view(running=False, output_name="").detail == "Choose an output, then press Start"

    issues = view(issues=("Lower the input gain.", "Check the output."))
    assert issues.headline == issues.nav_line == "Live · 2 issues"
    assert issues.detail == "Lower the input gain."
    assert view(issues=("x",)).headline == "Live · 1 issue"

    muted = view(muted=True)
    assert muted.headline == "Muted · others hear silence" and muted.tone == "mute"
    assert "Ctrl+M" in muted.detail

    refused = view(mute_error=True, mute_requested=True)
    assert (refused.headline, refused.detail, refused.tone) == ("Live · 1 issue", MUTE_PENDING_ISSUE, "warn")
    assert "callers still hear" in refused.detail and "silence" not in refused.headline
    # The refused mute comes first among the issues and is counted with them.
    both = view(mute_error=True, mute_requested=True, issues=("Lower the input gain.",))
    assert (both.headline, both.detail) == ("Live · 2 issues", MUTE_PENDING_ISSUE)
    # A refused unmute may leave callers hearing silence, so it never claims they hear the mic.
    unmute = view(mute_error=True)
    assert (unmute.headline, unmute.detail) == ("Live · 1 issue", UNMUTE_PENDING_ISSUE)
    assert "still hear your microphone" not in unmute.detail

    clipping = view(input_bad=True, input_advice="Turn the microphone down.")
    assert clipping.headline == "Live · your mic is clipping" and clipping.tone == "bad"
    assert clipping.detail.startswith("Turn the microphone down. ")
    assert "before AudioForge" in clipping.detail

    lost = view(device_bad=True, device_advice="Check that the device is connected.")
    assert lost.headline == "Your callers can't hear you"
    assert lost.detail == "Check that the device is connected."
    assert lost.callers_affected and lost.tone == "callers"
    assert not view().callers_affected

    no_mic = view(no_mic=True, running=False)
    assert no_mic.headline == "No microphone found" and no_mic.tone == "warn"
    assert "Refresh audio devices" in no_mic.detail
    assert no_mic.nav_line == "Stopped · no microphone"


def test_the_ribbon_waits_while_stopped() -> None:
    assert describe_overview(LIVE).ribbon_caption == "Level · last 10 seconds"
    assert describe_overview(replace(LIVE, running=False)).ribbon_caption == "Level · waiting for audio"


@pytest.mark.parametrize(
    ("ago", "line"),
    [
        (timedelta(hours=1), "Last Auto Voice Setup: today"),
        (timedelta(days=1), "Last Auto Voice Setup: yesterday"),
        (timedelta(days=3), "Last Auto Voice Setup: 3 days ago"),
    ],
)
def test_setup_is_recommended_until_one_was_saved(ago, line) -> None:
    first = describe_overview(LIVE)
    assert first.recommend_setup and first.setup_line == RECOMMEND_SETUP_HINT
    done = describe_overview(replace(LIVE, last_setup=NOW - ago))
    assert not done.recommend_setup
    assert done.setup_line == line


def test_stopped_chain_rows_read_idle_with_empty_bars() -> None:
    lines = describe_chain(ChainInputs(compressor_reduction_db=4.0, deesser_reduction_db=2.0,
                                       limiter_reduction_db=1.0))
    for key in ("deesser", "compressor", "limiter"):
        assert lines[key].line == "Idle" and lines[key].activity == 0
    assert all(line.activity == 0 for line in lines.values())


def test_chain_rows_name_each_stage_setting() -> None:
    lines = describe_chain(ChainInputs(
        running=True, strength=0.72, compressor_reduction_db=2.14, deesser_reduction_db=24.0,
    ))
    assert {key: line.line for key, line in lines.items()} == {
        "cleanup": "HP 80 Hz",
        "gate": "-40 dB",
        "suppression": "RNNoise · 72%",
        "deesser": "Reduction 24.0 dB",
        "eq": "10 bands",
        "compressor": "Reduction 2.1 dB",
        "limiter": "Idle",
    }
    assert lines["deesser"].activity == 1.0
    assert lines["compressor"].activity == pytest.approx(2.14 / 12)
    off = describe_chain(ChainInputs(
        running=True, cleanup_mode="Gentle", gate_enabled=False, suppression_enabled=False,
        deesser_enabled=False, eq_enabled=False, compressor_enabled=False, limiter_enabled=False,
    ))
    assert off["cleanup"].line == "HP 80 Hz · Gentle"
    assert {key: line.line for key, line in off.items() if key != "cleanup"} == dict.fromkeys(
        ("gate", "suppression", "deesser", "eq", "compressor", "limiter"), "Off"
    )
    assert describe_chain(ChainInputs(eq_bands=1, gate_threshold_db=-42.5))["eq"].line == "1 band"
    assert describe_chain(ChainInputs(gate_threshold_db=-42.5))["gate"].line == "-42.5 dB"


def test_ready_lists_the_route_and_no_mic_lists_what_was_checked() -> None:
    ready = describe_overview(replace(LIVE, running=False, output_virtual=True))
    assert ready.checks_title == "Route check · All set"
    assert ready.checks == (
        ("Microphone found · Microphone", "ok"),
        ("Output set to CABLE Input", "ok"),
        ("In Discord, OBS, Zoom or Teams, choose the other end of CABLE Input as your microphone", "note"),
    )
    unrouted = describe_overview(replace(LIVE, running=False, output_name=""))
    assert unrouted.checks_title == "Route check"
    assert unrouted.checks[-1] == ("No output chosen yet", "warn")

    no_mic = describe_overview(replace(LIVE, running=False, no_mic=True, input_name=""))
    assert no_mic.checks_title == "What AudioForge checked"
    assert [kind for _text, kind in no_mic.checks] == ["warn", "ok", "note"]
    assert no_mic.checks[0][0] == "No input devices detected · the Input list is empty"
    assert no_mic.checks[1][0] == "Output found · CABLE Input"
    nothing = describe_overview(replace(LIVE, running=False, no_mic=True, output_name="", output_found=False))
    assert nothing.checks[1] == ("No output devices detected", "warn")
    assert describe_overview(LIVE).checks == ()
