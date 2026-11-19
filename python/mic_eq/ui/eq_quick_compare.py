"""Qt Quick painter adapter for the read-only EQ comparison."""

from __future__ import annotations

from PySide6.QtCore import QObject, Property, Signal, Slot
from PySide6.QtGui import QPainter
from PySide6.QtQuick import QQuickPaintedItem

from .eq_compare import EQComparison


class EQComparisonItem(QQuickPaintedItem):
    """Paint the EQ now over the proposed EQ. It takes no input."""

    sourceChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: EQComparison | None = None
        self.setAntialiasing(True)

    def _get_source(self) -> EQComparison | None:
        return self._source

    def _set_source(self, source: EQComparison | None) -> None:
        if source is self._source:
            return
        if source is not None and not isinstance(source, EQComparison):
            raise TypeError("EQComparisonItem.source must be an EQComparison")
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

    @Slot()  # Register before QML adds properties to the metaobject.
    def _source_changed(self) -> None:
        self.update()

    def paint(self, painter: QPainter) -> None:
        if self._source is not None:
            self._source.paint(painter, self.width(), self.height())
