"""Native Qt Quick input and paint adapter for :class:`EQGraphModel`."""

from __future__ import annotations

from PySide6.QtCore import QObject, Property, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter
from PySide6.QtQuick import QQuickPaintedItem

from .eq_graph import EQGraphGeometry, EQGraphModel, EQGraphRenderer


class EQGraphItem(QQuickPaintedItem):
    """A QML graph item that paints and edits the shared model directly."""

    graphChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: EQGraphModel | None = None
        self.setAntialiasing(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def _get_graph(self) -> QObject | None:
        return self._graph

    def _set_graph(self, value: QObject | None) -> None:
        if value is self._graph:
            return
        if value is not None and not isinstance(value, EQGraphModel):
            raise TypeError("EQGraphItem.graph must be an EQGraphModel")
        if self._graph is not None:
            try:
                self._graph.changed.disconnect(self._update_graph)
            except (RuntimeError, TypeError):
                pass
        self._graph = value
        if self._graph is not None:
            self._graph.changed.connect(self._update_graph)
        self.graphChanged.emit()
        self.update()

    graph = Property(QObject, _get_graph, _set_graph, notify=graphChanged)

    def _update_graph(self) -> None:
        self.update()

    def paint(self, painter: QPainter) -> None:
        if self._graph is not None:
            EQGraphRenderer.paint(
                painter,
                self._graph,
                self.width(),
                self.height(),
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        graph = self._graph
        if (
            graph is None
            or event.button() != Qt.MouseButton.LeftButton
            or EQGraphGeometry.nearest_band_handle(
                graph.bands,
                event.position().x(),
                event.position().y(),
                self.width(),
                self.height(),
            )
            is None
        ):
            event.ignore()
            return
        self.forceActiveFocus()
        graph.mouse_press(
            event.position().x(),
            event.position().y(),
            self.width(),
            self.height(),
        )
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        graph = self._graph
        if graph is None or not graph.mouse_move(
            event.position().x(),
            event.position().y(),
            self.width(),
            self.height(),
        ):
            event.ignore()
            return
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        graph = self._graph
        if (
            graph is None
            or event.button() != Qt.MouseButton.LeftButton
            or not graph.mouse_release(
                event.position().x(),
                event.position().y(),
                self.width(),
                self.height(),
            )
        ):
            event.ignore()
            return
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._graph is None or not self._graph.key_press(
            event.key(), event.modifiers()
        ):
            event.ignore()
            return
        event.accept()
