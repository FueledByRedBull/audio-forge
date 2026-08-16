"""
Frequency response curve visualization for parametric EQ
"""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QPainter, QPen
from PyQt6.QtCore import Qt, pyqtSignal

from mic_eq.analysis.eq_quality import (
    EqInteractionWarning,
    evaluate_eq_quality,
)
from mic_eq import eq_magnitude_response_v2
from mic_eq.config import EQSettings
from .theme import PALETTE, qcolor


class EQCurveWidget(QWidget):
    """Widget that displays frequency response curve for 10-band EQ."""

    bandDragStarted = pyqtSignal(int)
    bandDragged = pyqtSignal(int, float, float)
    bandDragFinished = pyqtSignal(int, float, float)
    bandDragCancelled = pyqtSignal(int, float, float)

    MARGIN_LEFT = 40
    MARGIN_RIGHT = 10
    MARGIN_TOP = 10
    MARGIN_BOTTOM = 20
    FREQUENCY_MIN_HZ = 20.0
    FREQUENCY_MAX_HZ = 20_000.0
    GAIN_MIN_DB = -12.0
    GAIN_MAX_DB = 12.0
    DISPLAY_DB_MIN = -15.0
    DISPLAY_DB_MAX = 15.0
    HANDLE_RADIUS = 5.0
    HIT_RADIUS = 11.0
    GAIN_FILTER_TYPES = frozenset({"bell", "low_shelf", "high_shelf"})

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.sample_rate = 48000.0

        # Native v2 tuples: (type, frequency, gain, Q, slope, enabled).
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
        self.band_markers = []
        self.interaction_warnings = []
        self._selected_band_index: int | None = None
        self._drag_band_index: int | None = None
        self._drag_origin: tuple[float, float] | None = None
        # Pre-calculate frequency points for curve (log-spaced)
        self.freq_points = self._generate_log_frequencies(20, 20000, 100)
        self.response_db = [0.0] * len(self.freq_points)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Editable EQ response graph")
        self.setAccessibleDescription(
            "Click and drag a band handle. Left and right change frequency; "
            "up and down change gain for bell and shelf filters. Use left "
            "bracket and right bracket to select a band from the keyboard."
        )

        self._update_response()

    def _generate_log_frequencies(self, f_min, f_max, num_points):
        """Generate logarithmically-spaced frequency points."""
        log_min = math.log10(f_min)
        log_max = math.log10(f_max)
        step = (log_max - log_min) / (num_points - 1)
        return [10 ** (log_min + i * step) for i in range(num_points)]

    def _native_response(self, bands):
        return list(
            eq_magnitude_response_v2(
                self.freq_points,
                bands,
                self.sample_rate,
            )
        )

    def _plot_size(self) -> tuple[float, float]:
        return (
            max(1.0, float(self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT)),
            max(1.0, float(self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)),
        )

    def frequency_to_x(self, frequency_hz: float) -> float:
        """Map one validated frequency to a logical-pixel x coordinate."""
        plot_width, _plot_height = self._plot_size()
        frequency = min(
            self.FREQUENCY_MAX_HZ,
            max(self.FREQUENCY_MIN_HZ, float(frequency_hz)),
        )
        normalized = (
            math.log10(frequency) - math.log10(self.FREQUENCY_MIN_HZ)
        ) / (
            math.log10(self.FREQUENCY_MAX_HZ)
            - math.log10(self.FREQUENCY_MIN_HZ)
        )
        return self.MARGIN_LEFT + normalized * plot_width

    def x_to_frequency(self, x: float) -> float:
        """Map a logical-pixel x coordinate to a clamped 1 Hz value."""
        plot_width, _plot_height = self._plot_size()
        normalized = min(
            1.0,
            max(0.0, (float(x) - self.MARGIN_LEFT) / plot_width),
        )
        log_frequency = math.log10(self.FREQUENCY_MIN_HZ) + normalized * (
            math.log10(self.FREQUENCY_MAX_HZ)
            - math.log10(self.FREQUENCY_MIN_HZ)
        )
        return float(round(10.0**log_frequency))

    def gain_to_y(self, gain_db: float) -> float:
        """Map gain to a logical-pixel y coordinate."""
        _plot_width, plot_height = self._plot_size()
        gain = min(self.GAIN_MAX_DB, max(self.GAIN_MIN_DB, float(gain_db)))
        normalized = (self.DISPLAY_DB_MAX - gain) / (
            self.DISPLAY_DB_MAX - self.DISPLAY_DB_MIN
        )
        return self.MARGIN_TOP + normalized * plot_height

    def y_to_gain(self, y: float) -> float:
        """Map a logical-pixel y coordinate to clamped 0.1 dB precision."""
        _plot_width, plot_height = self._plot_size()
        normalized = min(
            1.0,
            max(0.0, (float(y) - self.MARGIN_TOP) / plot_height),
        )
        display_gain = self.DISPLAY_DB_MAX - normalized * (
            self.DISPLAY_DB_MAX - self.DISPLAY_DB_MIN
        )
        clamped = min(self.GAIN_MAX_DB, max(self.GAIN_MIN_DB, display_gain))
        return round(clamped * 10.0) / 10.0

    def band_handle_position(self, band_index: int) -> tuple[float, float]:
        """Return the visible handle position for tests and hit detection."""
        filter_type, frequency, gain, _q, _slope, _enabled = self.bands[
            band_index
        ]
        handle_gain = gain if filter_type in self.GAIN_FILTER_TYPES else 0.0
        return self.frequency_to_x(frequency), self.gain_to_y(handle_gain)

    def _nearest_band_handle(self, x: float, y: float) -> int | None:
        nearest: tuple[float, int] | None = None
        for index in range(len(self.bands)):
            handle_x, handle_y = self.band_handle_position(index)
            distance = math.hypot(float(x) - handle_x, float(y) - handle_y)
            if distance > self.HIT_RADIUS:
                continue
            candidate = (distance, index)
            if nearest is None or candidate < nearest:
                nearest = candidate
        return nearest[1] if nearest else None

    def _drag_parameters(self, x: float, y: float) -> tuple[float, float]:
        if self._drag_band_index is None:
            raise RuntimeError("no EQ band drag is active")
        filter_type, _frequency, gain, _q, _slope, _enabled = self.bands[
            self._drag_band_index
        ]
        frequency = self.x_to_frequency(x)
        if filter_type in self.GAIN_FILTER_TYPES:
            gain = self.y_to_gain(y)
        return frequency, float(gain)

    def _update_dragged_band(self, x: float, y: float) -> tuple[float, float]:
        if self._drag_band_index is None:
            raise RuntimeError("no EQ band drag is active")
        frequency, gain = self._drag_parameters(x, y)
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
        self.update()
        return frequency, gain

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        position = event.position()
        band_index = self._nearest_band_handle(position.x(), position.y())
        if band_index is None:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._selected_band_index = band_index
        self._drag_band_index = band_index
        band = self.bands[band_index]
        self._drag_origin = (float(band[1]), float(band[2]))
        self.bandDragStarted.emit(band_index)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_band_index is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        frequency, gain = self._update_dragged_band(
            position.x(),
            position.y(),
        )
        self.bandDragged.emit(self._drag_band_index, frequency, gain)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._drag_band_index is None
        ):
            super().mouseReleaseEvent(event)
            return
        position = event.position()
        band_index = self._drag_band_index
        frequency, gain = self._update_dragged_band(
            position.x(),
            position.y(),
        )
        self._drag_band_index = None
        self._drag_origin = None
        self.bandDragFinished.emit(band_index, frequency, gain)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            direction = -1 if key == Qt.Key.Key_BracketLeft else 1
            current = self._selected_band_index
            self._selected_band_index = (
                0 if current is None else (current + direction) % len(self.bands)
            )
            self.update()
            event.accept()
            return
        if self._selected_band_index is None:
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Escape and self._drag_origin is not None:
            band_index = self._selected_band_index
            frequency, gain = self._drag_origin
            filter_type, _frequency, _gain, q, slope, enabled = self.bands[
                band_index
            ]
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
            self.update()
            self.bandDragCancelled.emit(band_index, frequency, gain)
            event.accept()
            return
        if key not in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            super().keyPressEvent(event)
            return
        band_index = self._selected_band_index
        filter_type, frequency, gain, q, slope, enabled = self.bands[band_index]
        coarse = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            direction = -1.0 if key == Qt.Key.Key_Left else 1.0
            octave_step = (1.0 / 12.0) if coarse else (1.0 / 48.0)
            frequency = min(
                self.FREQUENCY_MAX_HZ,
                max(
                    self.FREQUENCY_MIN_HZ,
                    round(frequency * 2.0 ** (direction * octave_step)),
                ),
            )
        elif filter_type in self.GAIN_FILTER_TYPES:
            direction = 1.0 if key == Qt.Key.Key_Up else -1.0
            gain_step = 1.0 if coarse else 0.1
            gain = min(
                self.GAIN_MAX_DB,
                max(self.GAIN_MIN_DB, round((gain + direction * gain_step) * 10.0) / 10.0),
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
        self.update()
        self.bandDragStarted.emit(band_index)
        self.bandDragged.emit(band_index, float(frequency), float(gain))
        self.bandDragFinished.emit(band_index, float(frequency), float(gain))
        event.accept()

    def _update_response(self):
        """Calculate combined frequency response for all bands."""
        self.response_db = self._native_response(self.bands)
        freqs = [band[1] for band in self.bands]
        gains = [
            band[2]
            if band[0] in {"bell", "low_shelf", "high_shelf"} and band[5]
            else 0.0
            for band in self.bands
        ]
        qs = [band[3] for band in self.bands]
        warnings = list(
            evaluate_eq_quality(freqs, gains, qs, self.sample_rate).warnings
        )
        max_index = max(
            range(len(self.response_db)),
            key=self.response_db.__getitem__,
        )
        max_boost_db = self.response_db[max_index]
        if (
            max_boost_db > 10.5
            and not any(warning.kind == "max_boost" for warning in warnings)
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

    def set_band_params(self, band_index, freq, gain_db, q):
        """Update parameters for a single band and redraw."""
        if 0 <= band_index < len(self.bands):
            filter_type, _, _, _, slope, enabled = self.bands[band_index]
            self.bands[band_index] = (
                filter_type,
                float(freq),
                float(gain_db),
                float(q),
                slope,
                enabled,
            )
            self._update_response()
            self.update()  # Trigger repaint

    def set_band_config(
        self,
        band_index,
        filter_type,
        freq,
        gain_db,
        q,
        slope,
        enabled,
    ):
        """Update one complete typed band and redraw."""
        if 0 <= band_index < len(self.bands):
            self.bands[band_index] = (
                str(filter_type),
                float(freq),
                float(gain_db),
                float(q),
                int(slope),
                bool(enabled),
            )
            self._update_response()
            self.update()

    def set_all_params(self, bands):
        """
        Update all bands at once.
        Accept native v2 tuples or legacy (frequency, gain, Q) tuples.
        """
        for i, band in enumerate(bands):
            if i < len(self.bands):
                if len(band) == 3:
                    freq, gain_db, q = band
                    filter_type = (
                        "low_shelf"
                        if i == 0
                        else "high_shelf"
                        if i == 9
                        else "bell"
                    )
                    self.bands[i] = (
                        filter_type,
                        float(freq),
                        float(gain_db),
                        float(q),
                        12,
                        True,
                    )
                elif len(band) == 6:
                    filter_type, freq, gain_db, q, slope, enabled = band
                    self.bands[i] = (
                        str(filter_type),
                        float(freq),
                        float(gain_db),
                        float(q),
                        int(slope),
                        bool(enabled),
                    )
                else:
                    raise ValueError(
                        "EQ bands must contain either 3 legacy or 6 typed fields"
                    )

        self._update_response()
        self.update()  # Trigger repaint

    def set_band_markers(self, frequencies_hz):
        """Show markers for dynamically placed EQ bands."""
        self.band_markers = [float(freq) for freq in frequencies_hz]
        self.update()

    def clear_band_markers(self):
        """Hide dynamic band markers."""
        self.band_markers = []
        self.update()

    def paintEvent(self, event):
        """Draw the frequency response curve."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get widget dimensions
        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, qcolor(PALETTE.data_surface))

        # Define plot area (margins for labels)
        margin_left = self.MARGIN_LEFT
        margin_right = self.MARGIN_RIGHT
        margin_top = self.MARGIN_TOP
        margin_bottom = self.MARGIN_BOTTOM
        plot_height = height - margin_top - margin_bottom

        # Y-axis: -15dB to +15dB
        db_min = self.DISPLAY_DB_MIN
        db_max = self.DISPLAY_DB_MAX
        db_range = db_max - db_min

        def db_to_y(db):
            """Convert dB to y pixel coordinate."""
            normalized = (db_max - db) / db_range  # Invert: higher dB = lower y
            return margin_top + normalized * plot_height

        def freq_to_x(freq):
            """Convert frequency (Hz) to x pixel coordinate (log scale)."""
            return self.frequency_to_x(freq)

        # Draw horizontal grid lines
        grid_pen = QPen(qcolor(PALETTE.data_grid), 1)
        painter.setPen(grid_pen)

        for db in [-12, -6, 0, 6, 12]:
            y = db_to_y(db)
            painter.drawLine(margin_left, int(y), width - margin_right, int(y))

            # Label
            if db == 0:
                painter.setPen(qcolor(PALETTE.data_text))
                painter.drawText(5, int(y) + 4, f"{db} dB")
                painter.setPen(grid_pen)
            else:
                painter.setPen(qcolor(PALETTE.data_text_muted))
                painter.drawText(5, int(y) + 4, f"{db:+d}")
                painter.setPen(grid_pen)

        # Draw vertical grid lines at octave intervals
        for freq in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            if 20 <= freq <= 20000:
                x = freq_to_x(freq)
                painter.drawLine(int(x), margin_top, int(x), height - margin_bottom)

                # Label
                painter.setPen(qcolor(PALETTE.data_text_muted))
                if freq >= 1000:
                    label = f"{freq // 1000}k"
                else:
                    label = str(freq)
                painter.drawText(int(x) - 10, height - 5, label)
                painter.setPen(grid_pen)

        # Draw frequency response curve
        curve_pen = QPen(qcolor(PALETTE.data_curve), 2)
        painter.setPen(curve_pen)

        points = []
        for i, freq in enumerate(self.freq_points):
            x = freq_to_x(freq)
            y = db_to_y(self.response_db[i])
            points.append((int(x), int(y)))

        # Draw line segments
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            painter.drawLine(x1, y1, x2, y2)

        if self.band_markers:
            marker_pen = QPen(
                qcolor(PALETTE.data_marker, alpha=150),
                1,
                Qt.PenStyle.DashLine,
            )
            marker_fill = qcolor(PALETTE.data_marker)
            for freq in self.band_markers:
                if freq < 20.0 or freq > 20_000.0:
                    continue
                x = int(freq_to_x(freq))
                nearest_idx = min(
                    range(len(self.freq_points)),
                    key=lambda idx: abs(self.freq_points[idx] - freq),
                )
                y = int(db_to_y(self.response_db[nearest_idx]))
                painter.setPen(marker_pen)
                painter.drawLine(x, margin_top, x, height - margin_bottom)
                painter.setBrush(marker_fill)
                painter.setPen(QPen(marker_fill, 1))
                painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if self.interaction_warnings:
            warning_pen = QPen(qcolor(PALETTE.data_warning, alpha=180), 2)
            warning_fill = qcolor(PALETTE.data_warning, alpha=80)
            painter.setPen(warning_pen)
            painter.setBrush(warning_fill)
            for warning in self.interaction_warnings[:6]:
                freq = warning.frequency_hz
                if freq < 20.0 or freq > 20_000.0:
                    continue
                x = int(freq_to_x(freq))
                marker_height = max(8, int(10 + warning.severity * 12))
                painter.drawRect(x - 2, margin_top, 4, marker_height)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        # Draw keyboard/mouse-editable handles last so they remain discoverable.
        for index, band in enumerate(self.bands):
            x, y = self.band_handle_position(index)
            selected = index == self._selected_band_index
            enabled = bool(band[5])
            fill = (
                qcolor(PALETTE.data_handle_selected)
                if selected
                else qcolor(PALETTE.data_curve)
                if enabled
                else qcolor(PALETTE.data_handle_disabled)
            )
            outline = (
                qcolor(PALETTE.data_handle_selected_outline)
                if selected
                else qcolor(PALETTE.data_handle_outline)
            )
            radius = self.HANDLE_RADIUS + (1.0 if selected else 0.0)
            painter.setPen(QPen(outline, 1.5))
            painter.setBrush(fill if enabled or selected else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(round(x - radius)),
                int(round(y - radius)),
                int(round(radius * 2.0)),
                int(round(radius * 2.0)),
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)
