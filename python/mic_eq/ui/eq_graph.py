"""Shared data, geometry, interaction, and painting for the EQ response graph."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QPainter, QPen

from mic_eq import eq_magnitude_response_v2
from mic_eq.analysis.eq_quality import EqInteractionWarning, evaluate_eq_quality
from mic_eq.config import EQSettings
from .theme import PALETTE, RADIUS_CONTROL, qcolor


class EQGraphGeometry:
    """Size-dependent coordinate mapping and handle hit testing."""

    REGION_STRIP_HEIGHT = 22
    REGIONS = (
        ("SUB BASS", 20.0, 60.0),
        ("BASS", 60.0, 250.0),
        ("LOW MIDS", 250.0, 500.0),
        ("MID RANGE", 500.0, 2000.0),
        ("UPPER MIDS", 2000.0, 6000.0),
        ("HIGHS", 6000.0, 20000.0),
    )

    MARGIN_LEFT = 40
    MARGIN_RIGHT = 10
    MARGIN_TOP = 34
    MARGIN_BOTTOM = 20
    FREQUENCY_MIN_HZ = 20.0
    FREQUENCY_MAX_HZ = 20_000.0
    GAIN_MIN_DB = -12.0
    GAIN_MAX_DB = 12.0
    DISPLAY_DB_MIN = -15.0
    DISPLAY_DB_MAX = 15.0
    HANDLE_RADIUS = 6.0
    HIT_RADIUS = 11.0
    GAIN_FILTER_TYPES = frozenset({"bell", "low_shelf", "high_shelf"})

    @classmethod
    def plot_size(cls, width: float, height: float) -> tuple[float, float]:
        return (
            max(1.0, float(width) - cls.MARGIN_LEFT - cls.MARGIN_RIGHT),
            max(1.0, float(height) - cls.MARGIN_TOP - cls.MARGIN_BOTTOM),
        )

    @classmethod
    def frequency_to_x(cls, frequency_hz: float, width: float, height: float) -> float:
        plot_width, _plot_height = cls.plot_size(width, height)
        frequency = min(
            cls.FREQUENCY_MAX_HZ,
            max(cls.FREQUENCY_MIN_HZ, float(frequency_hz)),
        )
        normalized = (
            math.log10(frequency) - math.log10(cls.FREQUENCY_MIN_HZ)
        ) / (
            math.log10(cls.FREQUENCY_MAX_HZ)
            - math.log10(cls.FREQUENCY_MIN_HZ)
        )
        return cls.MARGIN_LEFT + normalized * plot_width

    @classmethod
    def x_to_frequency(cls, x: float, width: float, height: float) -> float:
        plot_width, _plot_height = cls.plot_size(width, height)
        normalized = min(1.0, max(0.0, (float(x) - cls.MARGIN_LEFT) / plot_width))
        log_frequency = math.log10(cls.FREQUENCY_MIN_HZ) + normalized * (
            math.log10(cls.FREQUENCY_MAX_HZ) - math.log10(cls.FREQUENCY_MIN_HZ)
        )
        return float(round(10.0**log_frequency))

    @classmethod
    def gain_to_y(cls, gain_db: float, width: float, height: float) -> float:
        _plot_width, plot_height = cls.plot_size(width, height)
        gain = min(cls.GAIN_MAX_DB, max(cls.GAIN_MIN_DB, float(gain_db)))
        normalized = (cls.DISPLAY_DB_MAX - gain) / (
            cls.DISPLAY_DB_MAX - cls.DISPLAY_DB_MIN
        )
        return cls.MARGIN_TOP + normalized * plot_height

    @classmethod
    def y_to_gain(cls, y: float, width: float, height: float) -> float:
        _plot_width, plot_height = cls.plot_size(width, height)
        normalized = min(
            1.0,
            max(0.0, (float(y) - cls.MARGIN_TOP) / plot_height),
        )
        display_gain = cls.DISPLAY_DB_MAX - normalized * (
            cls.DISPLAY_DB_MAX - cls.DISPLAY_DB_MIN
        )
        clamped = min(cls.GAIN_MAX_DB, max(cls.GAIN_MIN_DB, display_gain))
        return round(clamped * 10.0) / 10.0

    @classmethod
    def band_handle_position(
        cls,
        bands: list[tuple],
        band_index: int,
        width: float,
        height: float,
    ) -> tuple[float, float]:
        filter_type, frequency, gain, _q, _slope, _enabled = bands[band_index]
        handle_gain = gain if filter_type in cls.GAIN_FILTER_TYPES else 0.0
        return (
            cls.frequency_to_x(frequency, width, height),
            cls.gain_to_y(handle_gain, width, height),
        )

    @classmethod
    def nearest_band_handle(
        cls,
        bands: list[tuple],
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> int | None:
        nearest: tuple[float, int] | None = None
        for index in range(len(bands)):
            handle_x, handle_y = cls.band_handle_position(
                bands, index, width, height
            )
            distance = math.hypot(float(x) - handle_x, float(y) - handle_y)
            if distance > cls.HIT_RADIUS:
                continue
            candidate = (distance, index)
            if nearest is None or candidate < nearest:
                nearest = candidate
        return nearest[1] if nearest else None


class EQGraphModel(QObject):
    """One EQ graph snapshot and interaction state shared by both views."""

    changed = Signal()
    bandDragStarted = Signal(int)
    bandDragged = Signal(int, float, float)
    bandDragFinished = Signal(int, float, float)
    bandDragCancelled = Signal(int, float, float)
    bandSelected = Signal(int)

    GEOMETRY = EQGraphGeometry
    GAIN_FILTER_TYPES = EQGraphGeometry.GAIN_FILTER_TYPES

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.sample_rate = 48_000.0
        self.bands = [
            (
                band.filter_type,
                band.frequency_hz,
                band.gain_db,
                band.q,
                band.slope_db_per_octave,
                band.enabled,
            )
            for band in EQSettings().bands
        ]
        self.band_markers: list[float] = []
        self.correction_bands: list[tuple] = []
        self.interaction_warnings: list[EqInteractionWarning] = []
        self._selected_band_index: int | None = None
        self._drag_band_index: int | None = None
        self._drag_origin: tuple[float, float] | None = None
        self.freq_points = self.generate_log_frequencies(20, 20_000, 100)
        self.response_db = [0.0] * len(self.freq_points)
        self._update_response()

    @staticmethod
    def generate_log_frequencies(
        f_min: float, f_max: float, num_points: int
    ) -> list[float]:
        log_min = math.log10(f_min)
        log_max = math.log10(f_max)
        step = (log_max - log_min) / (num_points - 1)
        return [10 ** (log_min + i * step) for i in range(num_points)]

    @property
    def selected_band_index(self) -> int | None:
        return self._selected_band_index

    @property
    def drag_origin(self) -> tuple[float, float] | None:
        return self._drag_origin

    @property
    def dragging_band_index(self) -> int | None:
        return self._drag_band_index

    def select_band(self, band_index: int) -> None:
        if band_index == self._selected_band_index:
            return
        self._selected_band_index = band_index
        self.changed.emit()
        self.bandSelected.emit(band_index)

    def set_all_params(self, bands, *, correction=()) -> None:
        """Accept native typed tuples or legacy (frequency, gain, Q) tuples."""
        self.correction_bands = list(correction)
        for index, band in enumerate(bands):
            if index >= len(self.bands):
                break
            if len(band) == 3:
                frequency, gain_db, q = band
                filter_type = (
                    "low_shelf"
                    if index == 0
                    else "high_shelf"
                    if index == 9
                    else "bell"
                )
                self.bands[index] = (
                    filter_type,
                    float(frequency),
                    float(gain_db),
                    float(q),
                    12,
                    True,
                )
            elif len(band) == 6:
                filter_type, frequency, gain_db, q, slope, enabled = band
                self.bands[index] = (
                    str(filter_type),
                    float(frequency),
                    float(gain_db),
                    float(q),
                    int(slope),
                    bool(enabled),
                )
            else:
                raise ValueError("EQ bands must contain either 3 or 6 typed fields")
        self._update_response()
        self.changed.emit()

    def set_band_markers(self, frequencies_hz) -> None:
        self.band_markers = [float(freq) for freq in frequencies_hz]
        self.changed.emit()

    def clear_band_markers(self) -> None:
        self.band_markers = []
        self.changed.emit()

    def mouse_press(self, x: float, y: float, width: float, height: float) -> bool:
        band_index = self.GEOMETRY.nearest_band_handle(
            self.bands, x, y, width, height
        )
        if band_index is None:
            return False
        self.select_band(band_index)
        self._drag_band_index = band_index
        band = self.bands[band_index]
        self._drag_origin = (float(band[1]), float(band[2]))
        self.bandDragStarted.emit(band_index)
        return True

    def mouse_move(self, x: float, y: float, width: float, height: float) -> bool:
        if self._drag_band_index is None:
            return False
        frequency, gain = self._update_dragged_band(x, y, width, height)
        self.bandDragged.emit(self._drag_band_index, frequency, gain)
        return True

    def mouse_release(
        self, x: float, y: float, width: float, height: float
    ) -> bool:
        if self._drag_band_index is None:
            return False
        band_index = self._drag_band_index
        frequency, gain = self._update_dragged_band(x, y, width, height)
        self._drag_band_index = None
        self._drag_origin = None
        self.bandDragFinished.emit(band_index, frequency, gain)
        return True

    def key_press(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            direction = -1 if key == Qt.Key.Key_BracketLeft else 1
            current = self._selected_band_index
            self.select_band(
                0 if current is None else (current + direction) % len(self.bands)
            )
            return True
        if self._selected_band_index is None:
            return False
        if key == Qt.Key.Key_Escape and self._drag_origin is not None:
            band_index = self._selected_band_index
            frequency, gain = self._drag_origin
            filter_type, _frequency, _gain, q, slope, enabled = self.bands[band_index]
            self.bands[band_index] = (
                filter_type,
                frequency,
                gain,
                q,
                slope,
                enabled,
            )
            self._drag_band_index = None
            self._drag_origin = None
            self._update_response()
            self.changed.emit()
            self.bandDragCancelled.emit(band_index, frequency, gain)
            return True
        if key not in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            return False
        band_index = self._selected_band_index
        filter_type, frequency, gain, q, slope, enabled = self.bands[band_index]
        coarse = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            direction = -1.0 if key == Qt.Key.Key_Left else 1.0
            octave_step = (1.0 / 12.0) if coarse else (1.0 / 48.0)
            frequency = min(
                self.GEOMETRY.FREQUENCY_MAX_HZ,
                max(
                    self.GEOMETRY.FREQUENCY_MIN_HZ,
                    round(frequency * 2.0 ** (direction * octave_step)),
                ),
            )
        elif filter_type in self.GAIN_FILTER_TYPES:
            direction = 1.0 if key == Qt.Key.Key_Up else -1.0
            gain_step = 1.0 if coarse else 0.1
            gain = min(
                self.GEOMETRY.GAIN_MAX_DB,
                max(
                    self.GEOMETRY.GAIN_MIN_DB,
                    round((gain + direction * gain_step) * 10.0) / 10.0,
                ),
            )
        self.bands[band_index] = (
            filter_type,
            float(frequency),
            float(gain),
            q,
            slope,
            enabled,
        )
        self._update_response()
        self.changed.emit()
        self.bandDragStarted.emit(band_index)
        self.bandDragged.emit(band_index, float(frequency), float(gain))
        self.bandDragFinished.emit(band_index, float(frequency), float(gain))
        return True

    def _drag_parameters(
        self, x: float, y: float, width: float, height: float
    ) -> tuple[float, float]:
        if self._drag_band_index is None:
            raise RuntimeError("no EQ band drag is active")
        filter_type, _frequency, gain, _q, _slope, _enabled = self.bands[
            self._drag_band_index
        ]
        frequency = self.GEOMETRY.x_to_frequency(x, width, height)
        if filter_type in self.GAIN_FILTER_TYPES:
            gain = self.GEOMETRY.y_to_gain(y, width, height)
        return frequency, float(gain)

    def _update_dragged_band(
        self, x: float, y: float, width: float, height: float
    ) -> tuple[float, float]:
        if self._drag_band_index is None:
            raise RuntimeError("no EQ band drag is active")
        frequency, gain = self._drag_parameters(x, y, width, height)
        filter_type, _old_frequency, _old_gain, q, slope, enabled = self.bands[
            self._drag_band_index
        ]
        self.bands[self._drag_band_index] = (
            filter_type,
            frequency,
            gain,
            q,
            slope,
            enabled,
        )
        self._update_response()
        self.changed.emit()
        return frequency, gain

    def _update_response(self) -> None:
        self.response_db = list(
            eq_magnitude_response_v2(
                self.freq_points,
                self.bands,
                self.sample_rate,
            )
        )
        if self.correction_bands:
            correction = self._response(self.correction_bands)
            self.response_db = [
                tone + measured
                for tone, measured in zip(
                    self.response_db, correction, strict=True
                )
            ]
        warnings = []
        for stage in (self.bands, self.correction_bands):
            if not stage:
                continue
            warnings.extend(
                evaluate_eq_quality(
                    [band[1] for band in stage],
                    [band[2] for band in stage],
                    [band[3] for band in stage],
                    self.sample_rate,
                    filter_types=[band[0] for band in stage],
                    enabled=[band[5] for band in stage],
                    slopes_db_per_octave=[band[4] for band in stage],
                ).warnings
            )
        max_index = max(range(len(self.response_db)), key=self.response_db.__getitem__)
        max_boost_db = self.response_db[max_index]
        if max_boost_db > 10.5 and not any(
            warning.kind == "max_boost" for warning in warnings
        ):
            warnings.append(
                EqInteractionWarning(
                    "max_boost",
                    float(self.freq_points[max_index]),
                    min(1.0, (max_boost_db - 10.5) / 6.0),
                    "Combined boost is high",
                )
            )
        warnings.sort(key=lambda warning: warning.severity, reverse=True)
        self.interaction_warnings = warnings

    def _response(self, bands: list[tuple]) -> list[float]:
        return list(
            eq_magnitude_response_v2(self.freq_points, bands, self.sample_rate)
        )


class EQGraphRenderer:
    """Paint one shared graph model into either Qt view's QPainter."""

    GEOMETRY = EQGraphGeometry

    @classmethod
    def paint(
        cls,
        painter: QPainter,
        model: EQGraphModel,
        width: float,
        height: float,
    ) -> None:
        geometry = cls.GEOMETRY
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(PALETTE.data_surface))
        painter.drawRoundedRect(QRect(0, 0, width, height), RADIUS_CONTROL, RADIUS_CONTROL)

        margin_left = geometry.MARGIN_LEFT
        margin_right = geometry.MARGIN_RIGHT
        margin_top = geometry.MARGIN_TOP
        margin_bottom = geometry.MARGIN_BOTTOM
        plot_height = height - margin_top - margin_bottom
        db_min = geometry.DISPLAY_DB_MIN
        db_max = geometry.DISPLAY_DB_MAX
        db_range = db_max - db_min

        def db_to_y(db: float) -> float:
            normalized = (db_max - db) / db_range
            return margin_top + normalized * plot_height

        def freq_to_x(freq: float) -> float:
            return geometry.frequency_to_x(freq, width, height)

        region_font = painter.font()
        region_font.setPointSize(7)
        painter.save()
        painter.setFont(region_font)
        for name, low_hz, high_hz in geometry.REGIONS:
            left = int(freq_to_x(low_hz)) + 1
            right = int(freq_to_x(high_hz)) - 1
            painter.fillRect(
                left,
                0,
                right - left,
                geometry.REGION_STRIP_HEIGHT,
                qcolor(PALETTE.data_surface_raised),
            )
            painter.setPen(qcolor(PALETTE.data_text_muted))
            if painter.fontMetrics().horizontalAdvance(name) < right - left - 4:
                painter.drawText(
                    left,
                    0,
                    right - left,
                    geometry.REGION_STRIP_HEIGHT,
                    Qt.AlignmentFlag.AlignCenter,
                    name,
                )
        painter.restore()

        grid_pen = QPen(qcolor(PALETTE.data_grid), 1)
        painter.setPen(grid_pen)
        for db in (-12, -6, 0, 6, 12):
            y = db_to_y(db)
            painter.drawLine(margin_left, int(y), width - margin_right, int(y))
            if db == 0:
                painter.setPen(qcolor(PALETTE.data_text))
                painter.drawText(5, int(y) + 4, f"{db} dB")
            else:
                painter.setPen(qcolor(PALETTE.data_text_muted))
                painter.drawText(5, int(y) + 4, f"{db:+d}")
            painter.setPen(grid_pen)

        for freq in (100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            x = freq_to_x(freq)
            painter.drawLine(int(x), margin_top, int(x), height - margin_bottom)
            painter.setPen(qcolor(PALETTE.data_text_muted))
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            painter.drawText(int(x) - 10, height - 5, label)
            painter.setPen(grid_pen)

        painter.setPen(QPen(qcolor(PALETTE.data_curve), 2))
        points = [
            (
                int(freq_to_x(freq)),
                int(db_to_y(model.response_db[index])),
            )
            for index, freq in enumerate(model.freq_points)
        ]
        for first, second in zip(points, points[1:]):
            painter.drawLine(first[0], first[1], second[0], second[1])

        if model.band_markers:
            marker_pen = QPen(
                qcolor(PALETTE.data_marker, alpha=150),
                1,
                Qt.PenStyle.DashLine,
            )
            marker_fill = qcolor(PALETTE.data_marker)
            for frequency in model.band_markers:
                if not geometry.FREQUENCY_MIN_HZ <= frequency <= geometry.FREQUENCY_MAX_HZ:
                    continue
                x = int(freq_to_x(frequency))
                nearest_index = min(
                    range(len(model.freq_points)),
                    key=lambda index: abs(model.freq_points[index] - frequency),
                )
                y = int(db_to_y(model.response_db[nearest_index]))
                painter.setPen(marker_pen)
                painter.drawLine(x, margin_top, x, height - margin_bottom)
                painter.setBrush(marker_fill)
                painter.setPen(QPen(marker_fill, 1))
                painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if model.interaction_warnings:
            painter.setPen(QPen(qcolor(PALETTE.data_warning, alpha=180), 2))
            painter.setBrush(qcolor(PALETTE.data_warning, alpha=80))
            for warning in model.interaction_warnings[:6]:
                frequency = warning.frequency_hz
                if not geometry.FREQUENCY_MIN_HZ <= frequency <= geometry.FREQUENCY_MAX_HZ:
                    continue
                x = int(freq_to_x(frequency))
                marker_height = max(8, int(10 + warning.severity * 12))
                painter.drawRect(x - 2, margin_top, 4, marker_height)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        for index, band in enumerate(model.bands):
            x, y = geometry.band_handle_position(model.bands, index, width, height)
            selected = index == model.selected_band_index
            enabled = bool(band[5])
            fill = qcolor(PALETTE.eq_band_colors[index % len(PALETTE.eq_band_colors)])
            radius = geometry.HANDLE_RADIUS + (2.0 if selected else 0.0)
            painter.setPen(
                QPen(qcolor(PALETTE.data_handle_selected), 2.0)
                if selected
                else QPen(fill, 1.5)
            )
            painter.setBrush(fill if enabled else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(round(x - radius)),
                int(round(y - radius)),
                int(round(radius * 2.0)),
                int(round(radius * 2.0)),
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.restore()
