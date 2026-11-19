"""Native Qt Quick input and paint adapter for :class:`EQGraphModel`."""

from __future__ import annotations

from PySide6.QtCore import QObject, Property, Qt, Signal, Slot
from PySide6.QtGui import QHoverEvent, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtQuick import QQuickPaintedItem

from .eq_graph import EQGraphModel, EQGraphRenderer


class EQGraphPreview(QQuickPaintedItem):
    """The shared model's curves, read-only: no handles, input or focus."""

    graphChanged = Signal()
    # The graph item paints the band handles a user drags.
    HANDLES = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: EQGraphModel | None = None
        self.setAntialiasing(True)

    def _get_graph(self) -> QObject | None:
        return self._graph

    def _set_graph(self, value: QObject | None) -> None:
        if value is self._graph:
            return
        if value is not None and not isinstance(value, EQGraphModel):
            raise TypeError("EQGraphItem.graph must be an EQGraphModel")
        if self._graph is not None:
            for signal in (self._graph.changed, self._graph.hoverChanged):
                try:
                    signal.disconnect(self._update_graph)
                except (RuntimeError, TypeError):
                    pass
        self._graph = value
        if self._graph is not None:
            self._graph.changed.connect(self._update_graph)
            self._graph.hoverChanged.connect(self._update_graph)
        self.graphChanged.emit()
        self.update()

    graph = Property(QObject, _get_graph, _set_graph, notify=graphChanged)

    @Slot()  # Register before QML adds properties to the metaobject.
    def _update_graph(self) -> None:
        self.update()

    def paint(self, painter: QPainter) -> None:
        if self._graph is not None:
            EQGraphRenderer.paint(
                painter,
                self._graph,
                self.width(),
                self.height(),
                handles=self.HANDLES,
            )


class EQGraphItem(EQGraphPreview):
    """A QML graph item that paints and edits the shared model directly."""

    HANDLES = True
    # A handle may have moved: the bands, the selection or the item's size changed.
    handlesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        for signal in (self.graphChanged, self.widthChanged, self.heightChanged):
            signal.connect(self.handlesChanged)

    @Slot()
    def _update_graph(self) -> None:
        self.update()
        self.handlesChanged.emit()

    @Slot(int, result=float)
    def handleX(self, band: int) -> float:
        """Where band ``band``'s handle is drawn across the item, for what points at it."""
        if self._graph is None or not 0 <= band < len(self._graph.bands):
            return 0.0
        return self._graph.band_handle_position(band, self.width(), self.height())[0]

    @Slot(int, result=float)
    def handleBottom(self, band: int) -> float:
        """How far down the item band ``band``'s handle ends, its ring included."""
        if self._graph is None or not 0 <= band < len(self._graph.bands):
            return 0.0
        geometry = self._graph.GEOMETRY
        radius = (
            geometry.SELECTED_HANDLE_RADIUS
            if band == self._graph.selected_band_index
            else geometry.HANDLE_RADIUS
        )
        _x, y = self._graph.band_handle_position(band, self.width(), self.height())
        return y + radius + EQGraphRenderer.HANDLE_RING_WIDTH

    def mousePressEvent(self, event: QMouseEvent) -> None:
        graph = self._graph
        if (
            graph is None
            or event.button() != Qt.MouseButton.LeftButton
            or graph.nearest_band_handle(
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
        # Keeps the page's flick from taking the drag over, touch included.
        self.setKeepMouseGrab(True)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        graph = self._graph
        if (
            graph is None
            or event.button() != Qt.MouseButton.LeftButton
            or not graph.request_band_reset(
                event.position().x(),
                event.position().y(),
                self.width(),
                self.height(),
            )
        ):
            event.ignore()
            return
        event.accept()

    def hoverMoveEvent(self, event: QHoverEvent) -> None:
        if self._graph is not None:
            self._graph.set_hover(
                event.position().x(), event.position().y(), self.width(), self.height()
            )

    def hoverLeaveEvent(self, event: QHoverEvent) -> None:
        if self._graph is not None:
            self._graph.clear_hover()

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

    def mouseUngrabEvent(self) -> None:
        # Also runs after an ordinary release, when no drag is left to finish.
        self.setKeepMouseGrab(False)
        if self._graph is not None:
            self._graph.finish_drag()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._graph is None or not self._graph.key_press(
            event.key(), event.modifiers()
        ):
            event.ignore()
            return
        event.accept()
