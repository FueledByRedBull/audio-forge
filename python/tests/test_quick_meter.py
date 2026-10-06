"""The Quick meter adapter uses the existing meter painter and notifications."""

from PySide6.QtCore import QObject, QSizeF, Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QSignalSpy

from mic_eq.ui.level_meter import ConfidenceMeter, GainReductionMeter, LevelMeter
from mic_eq.ui.quick_meter import QuickMeterItem


class _MeterSource(QObject):
    repaintRequested = Signal()

    def __init__(self, target):
        super().__init__()
        self.target = target
        target.changed.connect(self.repaintRequested)


class _ObservedMeterItem(QuickMeterItem):
    def __init__(self):
        super().__init__()
        self.repaint_requests = 0

    def _source_changed(self) -> None:
        self.repaint_requests += 1
        super()._source_changed()


def _image(size: tuple[int, int]) -> QImage:
    image = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    return image


def test_quick_meter_paints_the_same_snapshot_as_the_classic_adapter(qapp):
    level = LevelMeter("IN", show_scale=True)
    level.set_levels(-18.0, -4.0)
    reduction = GainReductionMeter()
    reduction.set_gain_reduction(7.3)
    confidence = ConfidenceMeter()
    confidence.set_confidence(0.63)
    confidence.set_threshold(0.71)

    for meter, size in (
        (level, (72, 140)),
        (reduction, (190, 18)),
        (confidence, (190, 20)),
    ):
        item = QuickMeterItem()
        item.setProperty("source", _MeterSource(meter))
        item.setSize(QSizeF(*size))

        expected = _image(size)
        painter = QPainter(expected)
        meter.paint_meter(painter, *size)
        painter.end()

        actual = _image(size)
        painter = QPainter(actual)
        item.paint(painter)
        painter.end()

        assert actual == expected
        item.deleteLater()
        meter.close()

    qapp.processEvents()


def test_meter_changes_and_enabled_state_reach_quick_repaint_source(qapp):
    meter = LevelMeter("OUT", show_scale=False)
    source = _MeterSource(meter)
    item = _ObservedMeterItem()
    repaint_notifications = QSignalSpy(source.repaintRequested)
    item.setProperty("source", source)

    meter.set_levels(-24.0, -12.0)
    assert repaint_notifications.count() == 1
    assert item.repaint_requests == 1

    # The same 50 ms decay path used by the visible classic meter notifies Quick.
    meter._decay_peak_hold()
    assert repaint_notifications.count() == 2
    assert item.repaint_requests == 2

    meter.setEnabled(False)
    assert repaint_notifications.count() == 3
    assert item.repaint_requests == 3

    item.deleteLater()
    meter.close()
    qapp.processEvents()
