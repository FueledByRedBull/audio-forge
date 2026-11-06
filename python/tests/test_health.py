"""Health decision tests for compact runtime diagnostics."""

from dataclasses import replace
from typing import Any

import pytest

from mic_eq.runtime_snapshot import RuntimeSnapshot
from mic_eq.ui.health import (
    DEVICE_STARTUP_GRACE_SECONDS,
    NO_CALLBACK_YET,
    NOISE_SELECTION_DROPPED_WARNING,
    Chip,
    HealthMemory,
    HealthReadings,
    HealthReport,
    advice_for,
    health_summary,
    input_health_state,
    output_health_state,
    runtime_health,
)
from mic_eq.ui.shell_specs import LATENCY_LOWER_BOUND_NOTE_NO_BUFFER, TEXT_SPECS


def _observe(window, diagnostics: dict[str, Any], *, now: float) -> frozenset[str]:
    return window.observe(RuntimeSnapshot.parse(diagnostics), now=now)


def test_nonfinite_limiter_history_cannot_look_like_healthy_output():
    for invalid in (float("nan"), float("inf")):
        assert output_health_state(rms_db=-20, limiter_history_db=invalid)[1] == "idle"
        assert output_health_state(rms_db=-20, true_peak_limiter_history_db=invalid)[1] == "idle"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_health_measurements_are_unavailable(value):
    assert input_health_state(rms_db=value) == ("Input: --", "idle")
    assert output_health_state(rms_db=value) == ("Output: --", "idle")
    text, state = output_health_state(rms_db=-18.0, true_peak_db=value, short_term_lufs=value)
    assert state == "ok" and "nan" not in text and "inf" not in text


def test_recent_health_recovers_without_erasing_cumulative_counts():
    from mic_eq.ui.health import RecentStreamHealth

    health = RecentStreamHealth()
    diagnostics = {"output_callback_error_count": 1, "stream_restart_count": 2}
    assert _observe(health, diagnostics, now=0.0) == frozenset(diagnostics)
    assert _observe(health, diagnostics, now=4.9) == frozenset(diagnostics)
    assert not _observe(health, diagnostics, now=5.0)
    assert diagnostics == {"output_callback_error_count": 1, "stream_restart_count": 2}
    diagnostics["output_callback_error_count"] = 2
    assert _observe(health, diagnostics, now=6.0) == {"output_callback_error_count"}
    diagnostics["output_callback_error_count"] = 0  # Counter reset on a new stream.
    assert not _observe(health, diagnostics, now=11.0)


def test_a_counter_reset_ends_its_recent_event():
    # A restart within the five-second window used to leave the old event
    # standing while the counter read 0: "Gate: CHATTERING (0 times)".
    from mic_eq.ui.health import RecentStreamHealth

    health = RecentStreamHealth()
    assert _observe(health, {"gate_chatter_event_count": 12}, now=0.0) == {"gate_chatter_event_count"}
    assert not _observe(health, {"gate_chatter_event_count": 0}, now=1.0)
    assert _observe(health, {"gate_chatter_event_count": 1}, now=2.0) == {"gate_chatter_event_count"}

    # Events the new stream counted before the next tick are real ones.
    health = RecentStreamHealth()
    assert _observe(health, {"gate_chatter_event_count": 12}, now=0.0) == {"gate_chatter_event_count"}
    assert _observe(health, {"gate_chatter_event_count": 3}, now=1.0) == {"gate_chatter_event_count"}
    assert not _observe(health, {"gate_chatter_event_count": 3}, now=7.0)


def test_output_drift_retiming_is_not_a_recent_stream_event():
    from mic_eq.ui.health import RecentStreamHealth

    health = RecentStreamHealth()
    assert not _observe(health, {"jitter_dropped_samples": 0}, now=0.0)
    assert not _observe(health, {"jitter_dropped_samples": 3845}, now=1.0)
    assert _observe(health, {"jitter_dropped_samples": 3845, "input_dropped_samples": 1}, now=2.0) == {
        "input_dropped_samples"
    }


def test_recent_health_rejects_invalid_counter_evidence():
    from mic_eq.ui.health import RecentStreamHealth

    for value in (float("nan"), -1, "invalid", True, 1.5):
        with pytest.raises(ValueError):
            _observe(RecentStreamHealth(), {"input_dropped_samples": value}, now=0.0)


def test_input_health_flags_dense_signal_from_low_crest_factor():
    text, state = input_health_state(rms_db=-24.0, crest_factor_db=2.2)

    assert state == "warn"
    assert text == "Input: ALREADY COMPRESSED (2.2 dB)"


def test_input_health_preserves_ok_state_for_normal_speech_levels():
    text, state = input_health_state(rms_db=-24.0, crest_factor_db=12.0)

    assert (text, state) == ("Input: OK (-24 dB average)", "ok")
    assert input_health_state(rms_db=-24.0, peak_db=-8.04)[0] == "Input: OK (-24 dB average, peaks -8.0 dB)"


def test_loud_speech_rms_is_not_a_warning_but_peaks_near_full_scale_are():
    # A voice almost never reaches -3 dB RMS; the old rule watched the wrong number.
    assert input_health_state(rms_db=-2.0)[1] == "ok"
    assert input_health_state(rms_db=-20.0, peak_db=-3.0)[1] == "ok"
    text, state = input_health_state(rms_db=-20.0, peak_db=-0.3)
    assert (text, state) == ("Input: PEAKS TOO HIGH (-0.3 dB)", "warn")


def test_the_warning_boundary_is_judged_on_the_number_shown():
    # -2.96 prints as -3.0 dB, which is fine; the text and the state cannot disagree.
    assert input_health_state(rms_db=-20.0, peak_db=-2.96) == ("Input: OK (-20 dB average, peaks -3.0 dB)", "ok")
    assert input_health_state(rms_db=-20.0, peak_db=-2.94) == ("Input: PEAKS TOO HIGH (-2.9 dB)", "warn")
    assert input_health_state(rms_db=-20.0, crest_factor_db=2.96)[1] == "ok"
    assert input_health_state(rms_db=-20.0, crest_factor_db=2.94) == ("Input: ALREADY COMPRESSED (2.9 dB)", "warn")
    assert input_health_state(rms_db=-65.4) == ("Input: OK (-65 dB average)", "ok")
    assert input_health_state(rms_db=-65.6) == ("Input: LOW (-66 dB)", "warn")
    assert output_health_state(rms_db=-1.2)[1] == "ok"
    assert output_health_state(rms_db=-0.6) == ("Output: OK (-1 dB average)", "ok")


def test_a_value_that_rounds_to_zero_never_prints_a_minus_sign():
    assert input_health_state(rms_db=-0.3)[0] == "Input: OK (0 dB average)"
    assert output_health_state(rms_db=-18.0, true_peak_db=-0.04)[0] == "Output: OK (-18 dB average, peaks 0.0 dB)"
    assert output_health_state(rms_db=-0.4) == ("Output: VERY LOUD (0 dB)", "warn")
    assert output_health_state(rms_db=-18.0, true_peak_headroom_db=-0.04)[0] == "Output: LITTLE HEADROOM (0.0 dB)"
    assert output_health_state(rms_db=-18.0, true_peak_db=-0.04, true_peak_delta=True)[0] == (
        "Output: PEAK ABOVE CEILING (peak 0.0 dB)"
    )


def test_clipping_is_a_bad_state_that_names_the_peak_and_wins_over_everything():
    text, state = input_health_state(
        rms_db=-80.0, peak_db=1.24, full_scale_samples=7, phase_rescue_active=True,
    )
    assert (text, state) == ("Input: CLIPPED (peak 1.2 dB, full scale 7 times)", "bad")
    assert input_health_state(rms_db=-20.0, full_scale_samples=6)[0] == (
        "Input: CLIPPED (peak 0.0 dB, full scale 6 times)"
    )


@pytest.mark.parametrize(
    ("peak", "lower_by"),
    [(-0.3, 6), (-2.9, 4), (-2.0, 4), (-1.0, 5), (0.0, 6), (0.1, 7), (1.6, 8)],
)
def test_input_peak_advice_names_the_peak_and_a_whole_decibel_drop_towards_minus_six(peak, lower_by):
    advice = advice_for(input_health_state(rms_db=-20.0, peak_db=peak)[0])
    assert f"Input peaks at {peak:.1f} dB" in advice
    assert f"Lower the microphone level by about {lower_by} dB" in advice
    # Says why it matters, not only what to do.
    assert "cut off" in advice


def test_clipped_advice_is_stronger_and_still_has_the_number():
    advice = advice_for(input_health_state(rms_db=-20.0, peak_db=0.0, full_scale_samples=9)[0])
    # Stronger than the same peak without clipping: its own advice, carrying the count.
    assert advice != advice_for(input_health_state(rms_db=-20.0, peak_db=0.0)[0])
    assert "clipped" in advice
    assert "9 times" in advice and "0.0 dB" in advice
    assert "Lower the microphone level by about 6 dB" in advice
    assert "held" not in advice and "flat" not in advice


def test_full_scale_with_the_limiter_off_is_bad_and_with_it_on_is_normal():
    kwargs: dict[str, Any] = dict(rms_db=-12.0, at_full_scale=True, peak_db=2.3)
    text, state = output_health_state(limiter_enabled=False, **kwargs)
    assert (text, state) == ("Output: AT FULL SCALE (peak 2.3 dB)", "bad")
    advice = advice_for(text)
    assert "limiter is off" in advice and "Turn the limiter on or lower the makeup gain" in advice
    assert "about 4 dB" in advice
    # An output sitting at its ceiling is what a limiter is for.
    assert output_health_state(limiter_enabled=True, **kwargs)[1] == "ok"
    assert output_health_state(limiter_enabled=False, rms_db=-12.0)[1] == "ok"


def test_output_health_warns_on_low_true_peak_headroom():
    text, state = output_health_state(
        rms_db=-18.0,
        true_peak_db=-1.0,
        true_peak_headroom_db=0.4,
    )

    assert (text, state) == ("Output: LITTLE HEADROOM (0.4 dB)", "warn")
    assert "Lower the makeup gain by about 1 dB" in advice_for(text)


def test_headroom_advice_names_the_latest_peak_and_which_side_of_the_ceiling_it_is_on():
    below = advice_for("Output: LITTLE HEADROOM (0.4 dB)")
    assert "latest output peak is only 0.4 dB below the ceiling" in below
    assert "loudest" not in below
    above = advice_for(output_health_state(rms_db=-18.0, true_peak_headroom_db=-0.3)[0])
    assert "latest output peak is 0.3 dB above the ceiling" in above
    assert "-0.3" not in above and "below the ceiling" not in above
    assert "Lower the makeup gain by about 2 dB" in above


def test_output_health_warns_on_limiter_history():
    text, state = output_health_state(
        rms_db=-18.0,
        limiter_history_db=6.5,
        true_peak_limiter_history_db=0.2,
    )

    assert (text, state) == ("Output: LIMITER WORKING HARD (6.5 dB)", "warn")
    assert "turning the output down by up to 6.5 dB" in advice_for(text)
    assert "by about 7 dB" in advice_for(text)


def test_output_health_ok_includes_true_peak_and_loudness_detail():
    text, state = output_health_state(
        rms_db=-18.0,
        true_peak_db=-3.2,
        true_peak_headroom_db=1.7,
        short_term_lufs=-20.4,
    )

    assert (text, state) == ("Output: OK (-18 dB average, peaks -3.2 dB, -20 LUFS)", "ok")


def test_a_muted_output_reports_its_levels_as_before_the_mute_and_is_not_judged() -> None:
    # The levels are measured before the mute, so they must not read as heard or as a fault.
    assert output_health_state(muted=True, rms_db=-36.0, true_peak_db=-30.4) == (
        "Output: Muted (before mute: -36 dB average, peaks -30.4 dB)", "info",
    )
    assert output_health_state(
        muted=True, rms_db=-0.2, at_full_scale=True, limiter_enabled=False, limiter_history_db=9.0,
    )[1] == "info"
    assert output_health_state(muted=True, rms_db=None) == ("Output: Muted", "info")


def test_every_warning_chip_has_plain_language_advice() -> None:
    low_text, _ = input_health_state(rms_db=-72.0)
    assert "very quiet (about -72 dB" in advice_for(low_text)
    # The longer output token must not be mistaken for anything shorter.
    assert "0.2 dB below the ceiling" in advice_for("Output: LITTLE HEADROOM (0.2 dB)")
    # A drops chip with nothing dropped has nothing to advise.
    assert advice_for("Drops: 0 samples dropped, 0 dropouts, 0 restarts") == ""
    assert advice_for("Drops: nothing dropped, but a restart is pending") == ""
    assert advice_for("Input: OK (-20 dB average)") == ""
    assert advice_for("Input: STRANGE") == ""


@pytest.mark.parametrize(
    ("chip", "happened", "to_do"),
    [
        ("Input: PHASE", "partly cancel each other", "pick Left, Right or Phase-safe mono"),
        ("Input: LOW RUMBLE", "Low rumble is reaching", "keep input cleanup on"),
        ("Input: HUM DETECTED", "steady hum", "Move the microphone cable away from power cables"),
        ("Input: ALREADY COMPRESSED (2.8 dB)", "differ by only 2.8 dB", "Turn off processing in the microphone driver"),
        ("Output: PEAK ABOVE CEILING (peak 0.4 dB)", "went above the limiter ceiling", "lower the makeup gain"),
        ("Output: VERY LOUD (0 dB)", "is 0 dB", "Lower the makeup gain"),
        ("Gate: CHATTERING (3 times)", "rapidly 3 times", "Raise the hold time or lower the threshold"),
        ("Gate: EASED (3 times)", "3 times so far", "lowering the threshold yourself"),
        ("Backend: rnnoise failed", "passing through unprocessed", "Check the Backend setting on the Noise Suppression card"),
        ("Device activity: input 12 ms ago, output 1400 ms ago", "for 1400 ms", "Check that the device is connected"),
        ("Dropouts: 3 (2 in a row)", "dropped out 3 times", "Close heavy applications"),
        ("Buffer: FILLING UP (1200 samples)", "1200 samples waiting", "Close heavy applications"),
        ("Drops: 12 samples dropped, 3 dropouts, 2 restarts", "12 samples so far", "restart processing"),
        ("Recovery: 2 restarts, 4 output recoveries", "restarted (2 times so far)", "check the device connection"),
        ("Input: CLIPPED (peak 0.0 dB, full scale 12 times)", "full scale 12 times in the last few seconds", "about 6 dB"),
    ],
)
def test_each_warning_says_what_happened_and_what_to_do_with_its_number(chip, happened, to_do):
    advice = advice_for(chip)
    assert happened in advice
    # The action comes after the description, as its own sentence.
    assert to_do in advice.split(happened, 1)[1]
    assert advice.endswith(".")


@pytest.mark.parametrize(
    ("chip", "cause", "count"),
    [
        ("Recovery: 0 restarts, 3 output recoveries, restarted just now", "output", "3 times"),
        ("Recovery: 1 restart, 3 output recoveries", "stream restarted", "1 time "),
        ("Recovery: 0 restarts, 0 output recoveries, restart pending", "recovering", ""),
    ],
)
def test_recovery_advice_counts_only_what_happened(chip, cause, count):
    advice = advice_for(chip)
    assert cause in advice and count in advice
    assert "0 times" not in advice and "check the device connection" in advice
    if not count:
        assert not any(character.isdigit() for character in advice)


_READINGS = HealthReadings(
    latency_ms=24.0, dsp_time_ms=0.5, input_buf=128, rnnoise_buf=0,
    input_rms_db=-18.0, output_rms_db=-18.0, input_callback_age_ms=10, output_callback_age_ms=12,
)


def _health(
    diagnostics: dict[str, Any], readings: HealthReadings = _READINGS, *,
    recent: frozenset[str] = frozenset(), memory: HealthMemory = HealthMemory(), now: float = 0.0,
) -> HealthReport:
    return runtime_health(
        RuntimeSnapshot.parse(diagnostics), readings, recent_events=recent, memory=memory, now=now
    )


def test_an_engine_dict_with_nothing_to_report_gives_healthy_chips():
    report = _health({})
    chips = report.chips

    assert chips["latency_label"] == Chip(
        "Latency: at least 24 ms (processing 0.5 ms)", "info",
        TEXT_SPECS["latency_label"].tip + LATENCY_LOWER_BOUND_NOTE_NO_BUFFER,
    )
    assert chips["buffer_label"] == Chip("Buffer: OK (128 samples)", "ok")
    assert chips["input_health_label"] == Chip("Input: OK (-18 dB average)", "ok")
    assert chips["output_health_label"] == Chip("Output: OK (-18 dB average)", "ok")
    assert chips["gate_health_label"] == Chip("Gate: OK", "ok")
    assert chips["callback_health_label"] == Chip("Device activity: input 10 ms ago, output 12 ms ago", "ok")
    assert chips["underrun_health_label"] == Chip("Dropouts: 0", "ok")
    assert (chips["dropped_label"].text, chips["dropped_label"].state) == (
        "Drops: 0 samples dropped, 0 dropouts, 0 restarts", "ok"
    )
    assert chips["dropped_label"].tip is not None and chips["dropped_label"].tip.endswith(
        "Drops: 0 | U:0 | L:0 | NF:0 | RS:0"
    )
    assert chips["backend_diag_label"] == Chip("Backend: rnnoise, working", "ok")
    assert chips["recovery_diag_label"] == Chip(
        "Recovery: 0 restarts, 0 output recoveries", "ok", "No recovery recorded."
    )
    assert report.messages == ()
    assert report.memory == HealthMemory()
    assert health_summary(list(chips.values())) == Chip("Health: OK", "ok", "Everything looks fine.")


def test_counts_that_rose_since_the_last_tick_warn_once_and_the_memory_carries_them():
    diagnostics = {
        "output_underrun_total": 2, "output_true_peak_event_count": 1, "output_true_peak_db": 0.4,
        "input_phase_warning_count": 1,
    }
    first = _health(diagnostics)
    assert first.chips["underrun_health_label"] == Chip("Dropouts: 2", "warn")
    assert first.chips["output_health_label"] == Chip("Output: PEAK ABOVE CEILING (peak 0.4 dB)", "warn")
    assert first.chips["dropped_label"].text == (
        "Drops: 0 samples dropped, 2 dropouts, 0 restarts (needs attention: the output ran out of audio)"
    )
    assert first.memory == HealthMemory(
        output_underrun_total=2, input_phase_warning_count=1, output_true_peak_event_count=1
    )

    second = _health(diagnostics, memory=first.memory)
    assert second.chips["underrun_health_label"] == Chip("Dropouts: 2", "ok")
    assert second.chips["output_health_label"] == Chip("Output: OK (-18 dB average, peaks 0.4 dB)", "ok")
    assert second.chips["dropped_label"].state == "ok"


def test_a_device_that_never_starts_warns_once_the_grace_from_its_first_sighting_is_over():
    waiting = replace(
        _READINGS, input_callback_age_ms=NO_CALLBACK_YET, output_callback_age_ms=NO_CALLBACK_YET, processing=True
    )
    first = _health({}, waiting, now=100.0)
    assert first.chips["callback_health_label"] == Chip(
        "Device activity: input has not started sending sound yet, output has not started playing sound yet",
        "idle",
    )
    assert first.memory.device_wait_since == 100.0

    later = _health({}, waiting, memory=first.memory, now=100.0 + DEVICE_STARTUP_GRACE_SECONDS)
    assert later.chips["callback_health_label"].state == "warn"
    assert later.memory.device_wait_since == 100.0

    # Not processing: nothing is waiting for a start.
    stopped = _health({}, replace(waiting, processing=False), memory=later.memory, now=200.0)
    assert stopped.chips["callback_health_label"].state == "idle"
    assert stopped.memory.device_wait_since is None


def test_a_dropped_selection_and_a_failed_backend_are_announced_once():
    diagnostics = {
        "noise_selection_dropped_count": 1, "noise_model": "deepfilter", "noise_backend_available": False,
        "noise_backend_failed": True, "noise_backend_error": "model missing",
    }
    first = _health(diagnostics)
    assert first.messages == (
        (NOISE_SELECTION_DROPPED_WARNING, 8000),
        (
            "Noise suppression stopped working, so your voice is passing through unprocessed. "
            "Choose another model on the Noise Suppression card. Details: model missing",
            6000,
        ),
    )
    assert first.chips["backend_diag_label"] == Chip("Backend: deepfilter, failed, reported an error", "warn")

    assert _health(diagnostics, memory=first.memory).messages == ()
    working = {**diagnostics, "noise_backend_available": True, "noise_backend_failed": False}
    assert _health(working, memory=first.memory).memory.backend_warning is None


def test_recent_engine_events_and_a_pending_restart_need_attention_but_gate_chatter_is_the_gate():
    report = _health(
        {"stream_restart_count": 1, "last_restart_reason": "device changed"},
        replace(_READINGS, recovery_pending=True),
        recent=frozenset({"stream_restart_count"}),
    )
    assert report.chips["recovery_diag_label"] == Chip(
        "Recovery: 1 restart, 0 output recoveries, restart pending", "warn", "Last recovery: device changed"
    )
    assert report.chips["dropped_label"].text == (
        "Drops: 0 samples dropped, 0 dropouts, 1 restart (needs attention: the audio engine reported a fault)"
    )

    gate = _health({"gate_chatter_event_count": 2}, recent=frozenset({"gate_chatter_event_count"}))
    assert gate.chips["gate_health_label"] == Chip("Gate: CHATTERING (2 times)", "warn")
    assert gate.chips["dropped_label"].text == "Drops: nothing dropped, but the gate is chattering"


def test_the_summary_names_the_worst_row_and_gives_its_advice():
    full_scale = "Output: AT FULL SCALE (peak 1.5 dB)"
    rows = [
        Chip("Input: PEAKS TOO HIGH (-2.0 dB)", "warn"),
        Chip(full_scale, "bad"),
        Chip("Latency: about 10 ms (processing 0.2 ms)", "info"),
    ]
    assert health_summary(rows) == Chip(full_scale, "bad", advice_for(full_scale))
    assert health_summary(rows[2:]) == Chip("Health: IDLE", "idle", TEXT_SPECS["health_advice_label"].text)
    # A mute the engine has not confirmed is worth a look even when every row is fine.
    assert health_summary([Chip("Gate: OK", "ok")], mute_error=True) == Chip(
        "Health: WARN", "warn", "Check the items marked below."
    )
