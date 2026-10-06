"""Classic QWidget adapter for the shared EQ response graph."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import QWidget

from .eq_graph import EQGraphGeometry, EQGraphModel, EQGraphRenderer


class EQCurveWidget(QWidget):
    """Paint and route input for an :class:`EQGraphModel`."""

    bandDragStarted = Signal(int)
    bandDragged = Signal(int, float, float)
    bandDragFinished = Signal(int, float, float)
    bandDragCancelled = Signal(int, float, float)
    bandSelected = Signal(int)

    REGION_STRIP_HEIGHT = EQGraphGeometry.REGION_STRIP_HEIGHT
    REGIONS = EQGraphGeometry.REGIONS
    MARGIN_LEFT = EQGraphGeometry.MARGIN_LEFT
    MARGIN_RIGHT = EQGraphGeometry.MARGIN_RIGHT
    MARGIN_TOP = EQGraphGeometry.MARGIN_TOP
    MARGIN_BOTTOM = EQGraphGeometry.MARGIN_BOTTOM
    FREQUENCY_MIN_HZ = EQGraphGeometry.FREQUENCY_MIN_HZ
    FREQUENCY_MAX_HZ = EQGraphGeometry.FREQUENCY_MAX_HZ
    GAIN_MIN_DB = EQGraphGeometry.GAIN_MIN_DB
    GAIN_MAX_DB = EQGraphGeometry.GAIN_MAX_DB
    DISPLAY_DB_MIN = EQGraphGeometry.DISPLAY_DB_MIN
    DISPLAY_DB_MAX = EQGraphGeometry.DISPLAY_DB_MAX
    HANDLE_RADIUS = EQGraphGeometry.HANDLE_RADIUS
    HIT_RADIUS = EQGraphGeometry.HIT_RADIUS
    GAIN_FILTER_TYPES = EQGraphGeometry.GAIN_FILTER_TYPES

    def __init__(
        self,
        parent=None,
        *,
        graph_model: EQGraphModel | None = None,
    ):
        super().__init__(parent)
        self.graph_model = graph_model or EQGraphModel()
        self.setMinimumHeight(100)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Editable EQ response graph")
        self.setAccessibleDescription(
            "Click and drag a band handle. Left and right change frequency; "
            "up and down change gain for bell and shelf filters. Use left "
            "bracket and right bracket to select a band from the keyboard."
        )
        self.graph_model.changed.connect(self._model_changed)
        self.graph_model.bandDragStarted.connect(self.bandDragStarted.emit)
        self.graph_model.bandDragged.connect(self.bandDragged.emit)
        self.graph_model.bandDragFinished.connect(self.bandDragFinished.emit)
        self.graph_model.bandDragCancelled.connect(self.bandDragCancelled.emit)
        self.graph_model.bandSelected.connect(self.bandSelected.emit)

    def _model_changed(self) -> None:
        self.update()

    @property
    def bands(self) -> list[tuple]:
        return self.graph_model.bands

    @bands.setter
    def bands(self, value: list[tuple]) -> None:
        self.graph_model.bands = value

    @property
    def band_markers(self) -> list[float]:
        return self.graph_model.band_markers

    @band_markers.setter
    def band_markers(self, value: list[float]) -> None:
        self.graph_model.band_markers = value

    @property
    def correction_bands(self) -> list[tuple]:
        return self.graph_model.correction_bands

    @correction_bands.setter
    def correction_bands(self, value: list[tuple]) -> None:
        self.graph_model.correction_bands = value

    @property
    def interaction_warnings(self) -> list:
        return self.graph_model.interaction_warnings

    @interaction_warnings.setter
    def interaction_warnings(self, value: list) -> None:
        self.graph_model.interaction_warnings = value

    @property
    def freq_points(self) -> list[float]:
        return self.graph_model.freq_points

    @property
    def response_db(self) -> list[float]:
        return self.graph_model.response_db

    @response_db.setter
    def response_db(self, value: list[float]) -> None:
        self.graph_model.response_db = value

    @property
    def sample_rate(self) -> float:
        return self.graph_model.sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        self.graph_model.sample_rate = float(value)
        self.graph_model._update_response()
        self.graph_model.changed.emit()

    def select_band(self, band_index: int) -> None:
        self.graph_model.select_band(band_index)

    def frequency_to_x(self, frequency_hz: float) -> float:
        return EQGraphGeometry.frequency_to_x(
            frequency_hz, self.width(), self.height()
        )

    def x_to_frequency(self, x: float) -> float:
        return EQGraphGeometry.x_to_frequency(x, self.width(), self.height())

    def gain_to_y(self, gain_db: float) -> float:
        return EQGraphGeometry.gain_to_y(gain_db, self.width(), self.height())

    def y_to_gain(self, y: float) -> float:
        return EQGraphGeometry.y_to_gain(y, self.width(), self.height())

    def band_handle_position(self, band_index: int) -> tuple[float, float]:
        return EQGraphGeometry.band_handle_position(
            self.bands, band_index, self.width(), self.height()
        )

    def _nearest_band_handle(self, x: float, y: float) -> int | None:
        return EQGraphGeometry.nearest_band_handle(
            self.bands, x, y, self.width(), self.height()
        )

    def set_all_params(self, bands, *, correction=()) -> None:
        self.graph_model.set_all_params(bands, correction=correction)

    def set_band_markers(self, frequencies_hz) -> None:
        self.graph_model.set_band_markers(frequencies_hz)

    def clear_band_markers(self) -> None:
        self.graph_model.clear_band_markers()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        position = event.position()
        if self._nearest_band_handle(position.x(), position.y()) is None:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.graph_model.mouse_press(
            position.x(), position.y(), self.width(), self.height()
        )
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if not self.graph_model.mouse_move(
            position.x(), position.y(), self.width(), self.height()
        ):
            super().mouseMoveEvent(event)
            return
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.graph_model.mouse_release(
            event.position().x(),
            event.position().y(),
            self.width(),
            self.height(),
        ):
            super().mouseReleaseEvent(event)
            return
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self.graph_model.key_press(event.key(), event.modifiers()):
            super().keyPressEvent(event)
            return
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        EQGraphRenderer.paint(
            painter, self.graph_model, self.width(), self.height()
        )
