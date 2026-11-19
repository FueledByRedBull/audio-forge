"""The Quick meter adapter paints the meter model."""

from PySide6.QtCore import QSizeF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QSignalSpy

from mic_eq.ui.level_meter import ConfidenceMeter, GainReductionMeter, LevelMeter
from mic_eq.ui.quick_meter import QuickMeterItem


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


def test_quick_meter_paints_the_same_snapshot_as_the_model(qapp):
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
        item.setProperty("source", meter)
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

    qapp.processEvents()


def test_meter_changes_and_enabled_state_reach_quick_repaint_source(qapp):
    now = [0.0]
    meter = LevelMeter("OUT", show_scale=False, clock=lambda: now[0])
    item = _ObservedMeterItem()
    repaint_notifications = QSignalSpy(meter.changed)
    item.setProperty("source", meter)

    meter.set_levels(-24.0, -12.0)
    assert repaint_notifications.count() == 1
    assert item.repaint_requests == 1

    # The same 50 ms decay path notifies Quick, whether or not any widget exists.
    now[0] += 0.05
    meter._decay_peak_hold()
    assert repaint_notifications.count() == 2
    assert item.repaint_requests == 2

    meter.set_enabled(False)
    assert repaint_notifications.count() == 3
    assert item.repaint_requests == 3
    meter.set_enabled(False)
    assert repaint_notifications.count() == 3

    item.deleteLater()
    qapp.processEvents()


def test_meter_accessible_text_changes_only_when_its_words_do(qapp):
    meter = LevelMeter("IN")
    meter.set_levels(-20.0, -10.0)
    heard = []
    meter.accessibleTextChanged.connect(lambda: heard.append(meter.property("accessibleText")))
    meter.set_levels(-20.4, -10.0)  # same whole dB, same words
    assert heard == []
    meter.set_muted(True)
    assert heard == ["-20 dB. Muted: the level shown is the level before the mute."]
    assert ConfidenceMeter("VAD").property("accessibleText") == "no reading"


def test_a_horizontal_meter_paints_the_bar_instead_of_the_upright_meter(qapp):
    meter = LevelMeter("IN", show_scale=False)
    meter.set_levels(-12.0, -6.0)
    size = (72, 6)
    item = QuickMeterItem()
    item.setProperty("source", meter)
    item.setSize(QSizeF(*size))
    changes = QSignalSpy(item.horizontalChanged)
    item.setProperty("horizontal", True)
    item.setProperty("horizontal", True)
    assert changes.count() == 1

    expected = _image(size)
    painter = QPainter(expected)
    meter.paint_bar(painter, *size)
    painter.end()
    upright = _image(size)
    painter = QPainter(upright)
    meter.paint_meter(painter, *size)
    painter.end()
    actual = _image(size)
    painter = QPainter(actual)
    item.paint(painter)
    painter.end()

    assert actual == expected
    assert actual != upright
    # The bar fills from the left: the quiet end is lit, the loud end is not.
    assert actual.pixelColor(2, 3) != actual.pixelColor(size[0] - 2, 3)
    item.deleteLater()
    qapp.processEvents()
