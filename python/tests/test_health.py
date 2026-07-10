"""Health decision tests for compact runtime diagnostics."""

from mic_eq.ui.health import input_health_state, output_health_state


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
