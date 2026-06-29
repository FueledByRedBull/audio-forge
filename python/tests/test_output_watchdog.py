"""Tests for output callback watchdog logic."""

from mic_eq.ui.stream_recovery import StreamRecoveryManager, update_callback_stall_state


def test_watchdog_ignores_during_calibration():
    now = 100.0
    state, recover = update_callback_stall_state(
        stall_started_at=50.0,
        now=now,
        input_cb_age_ms=100,
        output_cb_age_ms=3000,
        processing_started_at=0.0,
        last_recovery_at=0.0,
        calibration_dialog_open=True,
    )
    assert state is None
    assert recover is False


def test_watchdog_warmup_blocks_recovery():
    now = 10.0
    state, recover = update_callback_stall_state(
        stall_started_at=None,
        now=now,
        input_cb_age_ms=100,
        output_cb_age_ms=3000,
        processing_started_at=6.0,
        last_recovery_at=0.0,
        calibration_dialog_open=False,
    )
    assert state is None
    assert recover is False


def test_watchdog_cooldown_blocks_recovery():
    now = 100.0
    state, recover = update_callback_stall_state(
        stall_started_at=98.0,
        now=now,
        input_cb_age_ms=100,
        output_cb_age_ms=3000,
        processing_started_at=0.0,
        last_recovery_at=90.0,
        calibration_dialog_open=False,
    )
    assert recover is False


def test_watchdog_sets_stall_timer_then_recovers():
    now = 100.0
    state, recover = update_callback_stall_state(
        stall_started_at=None,
        now=now,
        input_cb_age_ms=100,
        output_cb_age_ms=3000,
        processing_started_at=0.0,
        last_recovery_at=0.0,
        calibration_dialog_open=False,
    )
    assert state == now
    assert recover is False

    later = now + 1.6
    state, recover = update_callback_stall_state(
        stall_started_at=state,
        now=later,
        input_cb_age_ms=100,
        output_cb_age_ms=3000,
        processing_started_at=0.0,
        last_recovery_at=0.0,
        calibration_dialog_open=False,
    )
    assert state is None
    assert recover is True


def test_watchdog_resets_when_not_suspicious():
    now = 100.0
    state, recover = update_callback_stall_state(
        stall_started_at=90.0,
        now=now,
        input_cb_age_ms=3000,
        output_cb_age_ms=100,
        processing_started_at=0.0,
        last_recovery_at=0.0,
        calibration_dialog_open=False,
    )
    assert state is None
    assert recover is False


def test_output_stall_sets_timer_then_recovers():
    manager = StreamRecoveryManager(processing_started_at=0.0)

    recover = manager.maybe_recover_output_stall(
        input_rms=-20.0,
        output_rms=-90.0,
        output_buf=30000,
        calibration_dialog_open=False,
        now=100.0,
    )
    assert recover is False
    assert manager.output_stall_started_at == 100.0

    recover = manager.maybe_recover_output_stall(
        input_rms=-20.0,
        output_rms=-90.0,
        output_buf=30000,
        calibration_dialog_open=False,
        now=101.6,
    )
    assert recover is True
    assert manager.output_stall_started_at is None
    assert manager.last_output_recovery_at == 101.6


def test_output_stall_ignores_calibration_and_cooldown():
    manager = StreamRecoveryManager(
        output_stall_started_at=90.0,
        last_output_recovery_at=95.0,
        processing_started_at=0.0,
    )

    recover = manager.maybe_recover_output_stall(
        input_rms=-20.0,
        output_rms=-90.0,
        output_buf=30000,
        calibration_dialog_open=True,
        now=100.0,
    )
    assert recover is False
    assert manager.output_stall_started_at is None

    manager.output_stall_started_at = 99.0
    recover = manager.maybe_recover_output_stall(
        input_rms=-20.0,
        output_rms=-90.0,
        output_buf=30000,
        calibration_dialog_open=False,
        now=100.0,
    )
    assert recover is False
    assert manager.output_stall_started_at == 99.0
