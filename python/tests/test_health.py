"""Health decision tests for compact runtime diagnostics."""

import pytest

from mic_eq.ui.health import input_health_state, output_health_state


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
    assert health.observe(diagnostics, now=0.0) == frozenset(diagnostics)
    assert health.observe(diagnostics, now=4.9) == frozenset(diagnostics)
    assert not health.observe(diagnostics, now=5.0)
    assert diagnostics == {"output_callback_error_count": 1, "stream_restart_count": 2}
    diagnostics["output_callback_error_count"] = 2
    assert health.observe(diagnostics, now=6.0) == {"output_callback_error_count"}
    diagnostics["output_callback_error_count"] = 0  # Counter reset on a new stream.
    assert not health.observe(diagnostics, now=11.0)


def test_recent_health_rejects_invalid_counter_evidence():
    from mic_eq.ui.health import RecentStreamHealth

    for value in (float("nan"), -1, "invalid", True, 1.5):
        with pytest.raises(ValueError):
            RecentStreamHealth().observe({"input_dropped_samples": value}, now=0.0)


def test_input_health_flags_dense_signal_from_low_crest_factor():
    text, state = input_health_state(rms_db=-24.0, crest_factor_db=2.2)

    assert state == "warn"
    assert "DENSE" in text


def test_input_health_preserves_ok_state_for_normal_speech_levels():
    text, state = input_health_state(rms_db=-24.0, crest_factor_db=12.0)

    assert state == "ok"
    assert "CF:12" in text


def test_output_health_warns_on_low_true_peak_headroom():
    text, state = output_health_state(
        rms_db=-18.0,
        true_peak_db=-1.0,
        true_peak_headroom_db=0.4,
    )

    assert state == "warn"
    assert "LOW TP HEADROOM" in text


def test_output_health_warns_on_limiter_history():
    text, state = output_health_state(
        rms_db=-18.0,
        limiter_history_db=6.5,
        true_peak_limiter_history_db=0.2,
    )

    assert state == "warn"
    assert "LIMITING HARD" in text


def test_output_health_ok_includes_true_peak_and_loudness_detail():
    text, state = output_health_state(
        rms_db=-18.0,
        true_peak_db=-3.2,
        true_peak_headroom_db=1.7,
        short_term_lufs=-20.4,
    )

    assert state == "ok"
    assert "TP:-3.2" in text
    assert "LU:-20" in text
