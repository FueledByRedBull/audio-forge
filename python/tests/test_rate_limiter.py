"""Newer control values must supersede queued writes even after a UI stall."""

from mic_eq.ui import rate_limiter
from mic_eq.ui.compressor_panel import CompressorPanel
from mic_eq.mic_eq_core import AudioProcessor


def test_overdue_update_cannot_overwrite_new_native_setting(qapp, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now[0])
    processor = AudioProcessor()
    panel = CompressorPanel(processor)
    panel._comp_rate_limiter.flush()
    try:
        panel.release_spinbox.setValue(400)
        assert panel._comp_rate_limiter._pending_fn is not None
        # The UI thread was busy beyond the throttle interval; the old timer
        # has not run when a newer value arrives.
        now[0] += 0.060
        panel.release_spinbox.setValue(100)
        panel._comp_rate_limiter._execute_pending()
        assert panel.release_spinbox.value() == 100
        assert processor.get_compressor_release() == 100
        assert not panel._comp_rate_limiter._timer.isActive()
    finally:
        panel.deleteLater()


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
