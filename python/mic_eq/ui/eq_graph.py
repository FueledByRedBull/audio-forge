"""Shared data, geometry, interaction, and painting for the EQ response graph."""

from __future__ import annotations

import math
from bisect import bisect_left

from PySide6.QtCore import QLineF, QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPen, QPolygonF

from mic_eq import eq_magnitude_response_v2, input_prefilter_response_db
from mic_eq.analysis.eq_quality import EqInteractionWarning, evaluate_eq_quality
from mic_eq.config import EQSettings
from .theme import PALETTE, RADIUS_CONTROL, qcolor


# A band that is off: the native response gives it exactly 0 dB.
OFF_BAND: tuple[str, float, float, float, int, bool] = ("bell", 1000.0, 0.0, 1.0, 12, False)


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
        # response_db is what the EQ stage is set to do: the tone bands that
        # are on plus the Auto-EQ layer, whether or not the EQ switch is on.
        # The curves below are its parts; a band that is off has none, so it
        # cannot reach the total.
        self.band_curves_db: list[list[float]] = [[] for _ in self.bands]
        self.correction_response_db: list[float] = []
        # total_response_db is what the signal really gets: the fixed input
        # conditioning that comes before the EQ, plus the EQ unless its switch
        # is off. Only the part of the conditioning that does not follow the
        # signal can be drawn; see input_prefilter_response_db.
        self.eq_enabled = True
        self.cleanup_mode = "off"
        self.input_filter_db: list[float] = []
        self.total_response_db: list[float] = []
        self.interaction_warnings: list[EqInteractionWarning] = []
        self._selected_band_index: int | None = None
        self._drag_band_index: int | None = None
        self._drag_origin: tuple[float, float] | None = None
        self._drag_press = (0.0, 0.0)
        self._drag_handle = (0.0, 0.0)
        self._base_freq_points = self.generate_log_frequencies(20, 20_000, 100)
        self.freq_points = list(self._base_freq_points)
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

    def set_all_params(self, bands, *, correction=(), enabled=True) -> None:
        """Accept native typed tuples or legacy (frequency, gain, Q) tuples."""
        self.correction_bands = list(correction)
        self.eq_enabled = bool(enabled)
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

    def set_input_conditioning(self, cleanup_mode: str, sample_rate: float) -> None:
        """Follow what the processor runs: its cleanup mode and sample rate."""
        if (cleanup_mode, float(sample_rate)) == (self.cleanup_mode, self.sample_rate):
            return
        self.cleanup_mode = cleanup_mode
        self.sample_rate = float(sample_rate)
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
        self._drag_press = (x, y)
        self._drag_handle = self.GEOMETRY.band_handle_position(
            self.bands, band_index, width, height
        )
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
        self._update_dragged_band(x, y, width, height)
        return self.finish_drag()

    def finish_drag(self) -> bool:
        """End the active drag where the band stands."""
        band_index = self._drag_band_index
        if band_index is None:
            return False
        _filter_type, frequency, gain, *_ = self.bands[band_index]
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
        # The handle moves by the pointer's travel, so a press that was not
        # dead centre does not jump, and a click leaves the band as it was.
        origin = self._drag_origin
        if origin is not None and (x, y) == self._drag_press:
            return origin
        filter_type, _frequency, gain, _q, _slope, _enabled = self.bands[
            self._drag_band_index
        ]
        press_x, press_y = self._drag_press
        handle_x, handle_y = self._drag_handle
        frequency = self.GEOMETRY.x_to_frequency(
            handle_x + (x - press_x), width, height
        )
        if filter_type in self.GAIN_FILTER_TYPES:
            gain = self.GEOMETRY.y_to_gain(handle_y + (y - press_y), width, height)
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
        # Each band's own frequency is sampled, so a narrow peak or notch is
        # drawn at its true height and not at the nearest of 100 points.
        self.freq_points = sorted(
            {
                *self._base_freq_points,
                *(
                    band[1]
                    for stage in (self.bands, self.correction_bands)
                    for band in stage
                    if band[5]
                ),
            }
        )
        self.response_db = self._response(self.bands)
        self.correction_response_db = (
            self._response(self.correction_bands) if self.correction_bands else []
        )
        if self.correction_response_db:
            self.response_db = [
                tone + measured
                for tone, measured in zip(
                    self.response_db, self.correction_response_db, strict=True
                )
            ]
        self.band_curves_db = [
            self._response(self._band_alone(index)) if band[5] else []
            for index, band in enumerate(self.bands)
        ]
        self.input_filter_db = list(
            input_prefilter_response_db(
                self.freq_points, self.cleanup_mode, self.sample_rate
            )
        )
        self.total_response_db = [
            conditioning + (tone if self.eq_enabled else 0.0)
            for conditioning, tone in zip(self.input_filter_db, self.response_db)
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

    def _band_alone(self, index: int) -> list[tuple]:
        """The tone stage with every other band off: one band's own response."""
        bands = [OFF_BAND] * len(self.bands)
        bands[index] = self.bands[index]
        return bands


class EQGraphRenderer:
    """Paint one shared graph model into either Qt view's QPainter."""

    GEOMETRY = EQGraphGeometry
    TOTAL_WIDTH = 2.0
    SELECTED_BAND_WIDTH = 1.5
    BAND_WIDTH = 1.0
    FILL_ALPHA = 44
    AXIS_TOLERANCE_DB = 0.3
    CASING_WIDTH = 3.0
    MUTED_OPACITY = 0.4

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
        zero_pen = QPen(qcolor(PALETTE.border_strong), 1)
        painter.setPen(grid_pen)
        for db in (-12, -6, 0, 6, 12):
            y = db_to_y(db)
            painter.setPen(zero_pen if db == 0 else grid_pen)
            painter.drawLine(margin_left, int(y), width - margin_right, int(y))
            if db == 0:
                painter.setPen(qcolor(PALETTE.data_text))
                painter.drawText(5, int(y) + 4, f"{db} dB")
            else:
                painter.setPen(qcolor(PALETTE.data_text_muted))
                painter.drawText(5, int(y) + 4, f"{db:+d}")
            painter.setPen(grid_pen)

        for freq in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            x = freq_to_x(freq)
            painter.drawLine(int(x), margin_top, int(x), height - margin_bottom)
            painter.setPen(qcolor(PALETTE.data_text_muted))
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            painter.drawText(int(x) - 10, height - 5, label)
            painter.setPen(grid_pen)

        cls._paint_curves(painter, model, width, height)

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
                above = bisect_left(model.freq_points, frequency)
                nearest_index = min(
                    range(max(above - 1, 0), min(above + 1, len(model.freq_points))),
                    key=lambda index: abs(model.freq_points[index] - frequency),
                )
                y = int(db_to_y(model.total_response_db[nearest_index]))
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

        number_font = painter.font()
        number_font.setPointSize(7)
        number_font.setBold(True)
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
            # An off band is a hollow ring, but a selected one carries a number
            # that must not land on the curve behind it.
            painter.setBrush(
                fill
                if enabled
                else qcolor(PALETTE.data_surface)
                if selected
                else Qt.BrushStyle.NoBrush
            )
            box = QRect(
                int(round(x - radius)),
                int(round(y - radius)),
                int(round(radius * 2.0)),
                int(round(radius * 2.0)),
            )
            painter.drawEllipse(box)
            if selected:
                # The number the band editor shows beside the same color.
                painter.setFont(number_font)
                painter.setPen(
                    qcolor(PALETTE.text_on_accent if enabled else PALETTE.data_text)
                )
                painter.drawText(box, Qt.AlignmentFlag.AlignCenter, str(index + 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.restore()

    @staticmethod
    def _active_runs(values: list[float]) -> list[range]:
        """Segments of one band's curve where it leaves the 0 dB axis.

        A bell is flat far from its centre; drawing that part would only lay a
        colored stripe over the axis. Each run keeps one point of the axis on
        either side so it still meets it. Segment ``i`` joins points ``i`` and
        ``i + 1``.
        """
        runs: list[range] = []
        first: int | None = None
        for index, db in enumerate(values):
            if abs(db) >= EQGraphRenderer.AXIS_TOLERANCE_DB:
                if first is None:
                    first = max(index - 1, 0)
            elif first is not None:
                runs.append(range(first, index))
                first = None
        if first is not None:
            runs.append(range(first, len(values) - 1))
        return runs

    @classmethod
    def _paint_curves(
        cls,
        painter: QPainter,
        model: EQGraphModel,
        width: int,
        height: int,
    ) -> None:
        """Draw, back to front: the fill, each band, the input filter, the total, the Auto-EQ layer.

        With the EQ switch off the bands and the Auto-EQ layer are muted: they
        are what the EQ would do, and the total no longer includes them.
        """
        geometry = cls.GEOMETRY
        plot_width, plot_height = geometry.plot_size(width, height)
        top = geometry.MARGIN_TOP
        pixels_per_db = plot_height / (geometry.DISPLAY_DB_MAX - geometry.DISPLAY_DB_MIN)
        xs = [geometry.frequency_to_x(f, width, height) for f in model.freq_points]

        def ys(values: list[float]) -> list[float]:
            return [top + (geometry.DISPLAY_DB_MAX - db) * pixels_per_db for db in values]

        def segments(y: list[float], which=None) -> list[QLineF]:
            # Qt strokes a polyline with a slow general path, but draws plain
            # square-capped lines fast; the caps overlap, so the joins show no seam.
            return [
                QLineF(xs[i], y[i], xs[i + 1], y[i + 1])
                for i in (range(len(y) - 1) if which is None else which)
            ]

        def pen(color: str, line_width: float, cap=Qt.PenCapStyle.SquareCap) -> QPen:
            return QPen(
                qcolor(color),
                line_width,
                Qt.PenStyle.SolidLine,
                cap,
                Qt.PenJoinStyle.RoundJoin,
            )

        painter.save()
        painter.setClipRect(QRectF(geometry.MARGIN_LEFT, top, plot_width, plot_height))
        zero_y = top + geometry.DISPLAY_DB_MAX * pixels_per_db
        total = [QPointF(x, y) for x, y in zip(xs, ys(model.total_response_db))]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(PALETTE.data_curve, alpha=cls.FILL_ALPHA))
        painter.drawPolygon(
            QPolygonF([QPointF(xs[0], zero_y), *total, QPointF(xs[-1], zero_y)])
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if not model.eq_enabled:
            painter.setOpacity(cls.MUTED_OPACITY)
        selected = model.selected_band_index
        # The selected band last, so it is on top of its neighbours.
        for index in sorted(range(len(model.band_curves_db)), key=lambda i: i == selected):
            curve = model.band_curves_db[index]
            painter.setPen(
                pen(
                    PALETTE.eq_band_colors[index % len(PALETTE.eq_band_colors)],
                    cls.SELECTED_BAND_WIDTH if index == selected else cls.BAND_WIDTH,
                )
            )
            for run in cls._active_runs(curve):
                painter.drawLines(segments(ys(curve), run))
        painter.setOpacity(1.0)

        # Dotted: the input conditioning is not a band and not the Auto-EQ layer.
        conditioning = ys(model.input_filter_db)
        painter.setPen(pen(PALETTE.data_text_muted, cls.BAND_WIDTH))
        for run in cls._active_runs(model.input_filter_db):
            painter.drawLines(segments(conditioning, run[::2]))

        painter.setPen(pen(PALETTE.data_curve, cls.TOTAL_WIDTH, Qt.PenCapStyle.RoundCap))
        painter.drawPolyline(QPolygonF(total))

        if model.correction_response_db:
            # Dashed by leaving out every third segment, which costs a fraction
            # of stroking a dashed polyline. The dark casing keeps the dashes
            # visible on top of the total when the tone layer is flat and the
            # two coincide.
            if not model.eq_enabled:
                painter.setOpacity(cls.MUTED_OPACITY)
            correction = ys(model.correction_response_db)
            dashes = segments(correction, [i for i in range(len(xs) - 1) if i % 3 != 2])
            painter.setPen(pen(PALETTE.data_surface, cls.CASING_WIDTH))
            painter.drawLines(dashes)
            painter.setPen(pen(PALETTE.data_text, cls.BAND_WIDTH))
            painter.drawLines(dashes)
        painter.restore()
