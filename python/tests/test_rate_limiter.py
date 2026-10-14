"""Newer control values must supersede queued writes even after a UI stall."""

from mic_eq.ui import rate_limiter
from mic_eq.ui.compressor_state import CompressorState
from mic_eq.ui.processing_controls import SettingControl
from mic_eq.mic_eq_core import AudioProcessor


def test_overdue_update_cannot_overwrite_new_native_setting(qapp, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now[0])
    processor = AudioProcessor()
    state = CompressorState(processor)
    release = SettingControl(state, "compressor", "release_ms", None)
    # The applied write the window makes at start-up opens the throttle interval.
    state.set_settings({})
    state._rate_limiter.flush()
    try:
        release.setValue(400.0)
        assert state._rate_limiter._pending_fn is not None
        # The UI thread was busy beyond the throttle interval; the old timer
        # has not run when a newer value arrives.
        now[0] += 0.060
        release.setValue(100.0)
        state._rate_limiter._execute_pending()
        assert release.value == 100
        assert processor.get_compressor_release() == 100
        assert not state._rate_limiter._timer.isActive()
    finally:
        processor.stop()


def test_burst_keeps_latest_value_and_flushes_once(qapp, monkeypatch):
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 100.0)
    limiter = rate_limiter.RateLimiter()
    values = []
    for value in (1, 2, 3):
        limiter.call(lambda value=value: values.append(value))
    assert values == [1]
    limiter.flush()
    limiter._execute_pending()
    assert values == [1, 3]


def test_bulk_write_cancels_old_pending_edit_and_runs_synchronously(qapp, monkeypatch):
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: 100.0)
    limiter = rate_limiter.RateLimiter(interval_ms=500)
    values = []
    limiter.call(lambda: values.append("old-immediate"))
    limiter.call(lambda: values.append("stale"))

    limiter.call_now(lambda: values.append("bulk"))
    limiter._execute_pending()

    assert values == ["old-immediate", "bulk"]
    assert limiter._pending_fn is None
    assert not limiter._timer.isActive()
