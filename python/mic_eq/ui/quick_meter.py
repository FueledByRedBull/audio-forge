"""Qt Quick painter adapter for the shared meter models."""

from __future__ import annotations

from PySide6.QtCore import QObject, Property, Signal, Slot
from PySide6.QtGui import QPainter
from PySide6.QtQuick import QQuickPaintedItem

from .level_meter import MeterModel


class QuickMeterItem(QQuickPaintedItem):
    """Paint meter telemetry directly, without rendering or resizing a widget."""

    sourceChanged = Signal()
    horizontalChanged = Signal()
    segmentedChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: MeterModel | None = None
        self._horizontal = False
        self._segmented = False

    def _get_source(self) -> MeterModel | None:
        return self._source

    def _set_source(self, source: MeterModel | None) -> None:
        if source is self._source:
            return
        if source is not None and not isinstance(source, MeterModel):
            raise TypeError("QuickMeterItem.source must be a meter model")
        if self._source is not None:
            try:
                self._source.changed.disconnect(self._source_changed)
            except (RuntimeError, TypeError):
                pass
        self._source = source
        if source is not None:
            source.changed.connect(self._source_changed)
        self.sourceChanged.emit()
        self.update()

    source = Property(QObject, _get_source, _set_source, notify=sourceChanged)

    def _set_horizontal(self, horizontal: bool) -> None:
        if horizontal != self._horizontal:
            self._horizontal = horizontal
            self.horizontalChanged.emit()
            self.update()

    # A thin bar, quiet at the left, instead of the default upright meter.
    horizontal = Property(bool, lambda self: self._horizontal, _set_horizontal, notify=horizontalChanged)

    def _set_segmented(self, segmented: bool) -> None:
        if segmented != self._segmented:
            self._segmented = segmented
            self.segmentedChanged.emit()
            self.update()

    # The horizontal bar drawn as a row of short steps; it has no effect on the upright meter.
    segmented = Property(bool, lambda self: self._segmented, _set_segmented, notify=segmentedChanged)

    @Slot()  # Register before QML adds delegate properties to the metaobject.
    def _source_changed(self) -> None:
        if self.isVisible():
            self.update()

    def paint(self, painter: QPainter) -> None:
        if self._source is None:
            return
        if not self._horizontal:
            paint = self._source.paint_meter
        elif self._segmented:
            paint = self._source.paint_segmented_bar
        else:
            paint = self._source.paint_bar
        paint(painter, self.width(), self.height())
