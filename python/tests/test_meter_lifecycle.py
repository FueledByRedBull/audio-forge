"""Unavailable meter state and bounded painting work."""

from PySide6.QtCore import QEventLoop, QTimer

from mic_eq.ui.level_meter import ConfidenceMeter, GainReductionMeter, LevelMeter


def test_level_meter_does_not_tick_hidden_or_unavailable(qapp):
    meter = LevelMeter()
    ticks = []
    meter.decay_timer.timeout.connect(lambda: ticks.append(1))
    assert not meter.decay_timer.isActive()
    meter.set_levels(-20.0, -10.0)
    assert not meter.decay_timer.isActive()
    meter.show()
    qapp.processEvents()
    loop = QEventLoop()
    QTimer.singleShot(125, loop.quit)
    loop.exec()
    assert len(ticks) >= 1
    assert meter.measurement_available
    meter.hide()
    ticks.clear()
    QTimer.singleShot(125, loop.quit)
    loop.exec()
    assert not ticks and not meter.decay_timer.isActive()
    meter.set_unavailable()
    meter.show()
    qapp.processEvents()
    assert not meter.measurement_available
    assert not meter.decay_timer.isActive()
    meter.close()


def test_unknown_peak_keeps_only_measured_rms_and_nonfinite_is_unavailable(qapp):
    meter = LevelMeter()
    meter.set_levels(-18.0, float("nan"))
    assert meter.measurement_available and not meter.peak_available
    assert meter.rms_db == -18.0 and not meter.is_clipping
    meter.set_levels(float("nan"), 0.0)
    assert not meter.measurement_available
    assert not meter.is_clipping
    meter.close()


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
    reduction.close()
    confidence.close()
