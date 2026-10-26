"""Unavailable meter state and bounded painting work."""

from PySide6.QtCore import QEventLoop, QTimer

from mic_eq.ui.level_meter import ConfidenceMeter, GainReductionMeter, LevelMeter


def _wait(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def test_level_meter_decays_without_any_widget_and_stops_by_itself(qapp):
    now = [100.0]  # the test's clock, in seconds
    meter = LevelMeter(clock=lambda: now[0])
    meter.PEAK_DECAY_RATE = 200.0  # Fall 10 dB per 50 ms tick.
    ticks = []
    meter.decay_timer.timeout.connect(lambda: ticks.append(1))
    assert not meter.decay_timer.isActive()
    meter.set_levels(-20.0, 0.0)
    assert meter.decay_timer.isActive()
    assert meter.is_clipping
    # One real timer firing shows the timeout drives the decay; the clock moves only when told to.
    now[0] += 0.05
    _wait(125)
    assert len(ticks) >= 1
    assert meter.peak_hold_db < meter.DB_MAX

    # The clip flash clears and the hold reaches the floor without any view.
    for _ in range(100):
        if not meter.decay_timer.isActive():
            break
        now[0] += 0.05
        meter._decay_peak_hold()
    assert not meter.decay_timer.isActive()
    assert meter.peak_hold_db == meter.DB_MIN
    assert not meter.is_clipping
    assert meter.measurement_available


def test_level_meter_timer_stops_when_the_measurement_goes_away(qapp):
    meter = LevelMeter()
    meter.set_levels(-20.0, -10.0)
    assert meter.decay_timer.isActive()
    meter.set_rms_level(-20.0)
    assert not meter.decay_timer.isActive()
    meter.set_levels(-20.0, -10.0)
    assert meter.decay_timer.isActive()
    meter.set_unavailable()
    assert not meter.measurement_available
    assert not meter.decay_timer.isActive()


def test_unknown_peak_keeps_only_measured_rms_and_nonfinite_is_unavailable(qapp):
    meter = LevelMeter()
    meter.set_levels(-18.0, float("nan"))
    assert meter.measurement_available and not meter.peak_available
    assert meter.rms_db == -18.0 and not meter.is_clipping
    meter.set_levels(float("nan"), 0.0)
    assert not meter.measurement_available
    assert not meter.is_clipping


def test_stopped_gain_reduction_and_confidence_do_not_look_like_zero(qapp):
    reduction = GainReductionMeter()
    confidence = ConfidenceMeter()
    reduction.set_gain_reduction(3.0)
    confidence.set_confidence(0.8)
    reduction.set_gain_reduction(None)
    confidence.set_confidence(None)
    assert not reduction.measurement_available
    assert not confidence.measurement_available
    reduction.set_gain_reduction(float("inf"))
    confidence.set_confidence(float("nan"))
    assert not reduction.measurement_available
    assert not confidence.measurement_available
