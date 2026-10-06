"""Qt Quick painter adapter for the shared QWidget meter renderers."""

from __future__ import annotations

from PySide6.QtCore import QObject, Property, Signal
from PySide6.QtGui import QPainter
from PySide6.QtQuick import QQuickPaintedItem


class QuickMeterItem(QQuickPaintedItem):
    """Paint meter telemetry directly, without rendering or resizing a widget."""

    sourceChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: QObject | None = None

    def _get_source(self) -> QObject | None:
        return self._source

    def _set_source(self, source: QObject | None) -> None:
        if source is self._source:
            return
        if source is not None:
            target = getattr(source, "target", None)
            if not callable(getattr(target, "paint_meter", None)):
                raise TypeError("QuickMeterItem.source must expose a meter target")
        if self._source is not None:
            repaint_requested = getattr(self._source, "repaintRequested", None)
            if repaint_requested is not None:
                try:
                    repaint_requested.disconnect(self._source_changed)
                except (RuntimeError, TypeError):
                    pass
        self._source = source
        if source is not None:
            repaint_requested = getattr(source, "repaintRequested", None)
            if repaint_requested is not None:
                repaint_requested.connect(self._source_changed)
        self.sourceChanged.emit()
        self.update()

    source = Property(QObject, _get_source, _set_source, notify=sourceChanged)

    def _source_changed(self) -> None:
        if self.isVisible():
            self.update()

    def paint(self, painter: QPainter) -> None:
        source = self._source
        target = getattr(source, "target", None)
        if target is not None:
            target.paint_meter(painter, self.width(), self.height())
