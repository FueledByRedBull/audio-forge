"""
Frequency response curve visualization for parametric EQ
"""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt

from mic_eq.analysis.eq_quality import evaluate_eq_quality
from mic_eq import eq_magnitude_response


class EQCurveWidget(QWidget):
    """Widget that displays frequency response curve for 10-band EQ."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.sample_rate = 48000.0

        # Band parameters: (freq, gain_db, q, filter_type)
        # filter_type: 0=lowshelf, 1=peaking, 2=highshelf
        self.bands = []
        self.overlay_bands = []  # Optional second curve for comparison
        self.band_markers = []
        self.interaction_warnings = []
        self.show_overlay = False
        for i in range(10):
            if i == 0:
                filter_type = 0  # Low shelf
            elif i == 9:
                filter_type = 2  # High shelf
            else:
                filter_type = 1  # Peaking

            # Default frequencies
            freqs = [80, 160, 320, 640, 1280, 2500, 5000, 8000, 12000, 16000]
            self.bands.append((freqs[i], 0.0, 1.41, filter_type))

        # Pre-calculate frequency points for curve (log-spaced)
        self.freq_points = self._generate_log_frequencies(20, 20000, 100)
        self.response_db = [0.0] * len(self.freq_points)

        self._update_response()

    def _generate_log_frequencies(self, f_min, f_max, num_points):
        """Generate logarithmically-spaced frequency points."""
        log_min = math.log10(f_min)
        log_max = math.log10(f_max)
        step = (log_max - log_min) / (num_points - 1)
        return [10 ** (log_min + i * step) for i in range(num_points)]

    def _native_response(self, bands):
        parameters = [
            (float(freq), float(gain), float(q)) for freq, gain, q, _ in bands
        ]
        return list(
            eq_magnitude_response(
                self.freq_points,
                parameters,
                self.sample_rate,
            )
        )

    def _update_response(self):
        """Calculate combined frequency response for all bands."""
        self.response_db = self._native_response(self.bands)
        freqs = [band[0] for band in self.bands]
        gains = [band[1] for band in self.bands]
        qs = [band[2] for band in self.bands]
        self.interaction_warnings = list(
            evaluate_eq_quality(freqs, gains, qs, self.sample_rate).warnings
        )

    def set_band_params(self, band_index, freq, gain_db, q):
        """Update parameters for a single band and redraw."""
        if 0 <= band_index < len(self.bands):
            _, _, _, filter_type = self.bands[band_index]
            self.bands[band_index] = (freq, gain_db, q, filter_type)
            self._update_response()
            self.update()  # Trigger repaint

    def set_all_params(self, bands):
        """
        Update all bands at once.
        bands = [(freq, gain, q), ...] for all 10 bands.
        """
        for i, (freq, gain_db, q) in enumerate(bands):
            if i < len(self.bands):
                _, _, _, filter_type = self.bands[i]
                self.bands[i] = (freq, gain_db, q, filter_type)

        self._update_response()
        self.update()  # Trigger repaint

    def set_overlay_params(self, bands):
        """
        Set overlay curve parameters for before/after comparison.

        Args:
            bands: List of (frequency_hz, gain_db, q) tuples for overlay curve
        """
        self.overlay_bands = []
        for i, (freq, gain_db, q) in enumerate(bands):
            if i == 0:
                filter_type = 0  # Low shelf
            elif i == 9:
                filter_type = 2  # High shelf
            else:
                filter_type = 1  # Peaking
            self.overlay_bands.append((freq, gain_db, q, filter_type))
        self.show_overlay = True
        self._update_overlay_response()
        self.update()

    def clear_overlay(self):
        """Remove overlay curve and return to single curve mode."""
        self.overlay_bands = []
        self.show_overlay = False
        self.update()

    def set_band_markers(self, frequencies_hz):
        """Show markers for dynamically placed EQ bands."""
        self.band_markers = [float(freq) for freq in frequencies_hz]
        self.update()

    def clear_band_markers(self):
        """Hide dynamic band markers."""
        self.band_markers = []
        self.update()

    def _update_overlay_response(self):
        """Calculate frequency response for overlay curve."""
        self.overlay_response_db = self._native_response(self.overlay_bands)

    def paintEvent(self, event):
        """Draw the frequency response curve."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get widget dimensions
        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor("#2a2a2a"))

        # Define plot area (margins for labels)
        margin_left = 40
        margin_right = 10
        margin_top = 10
        margin_bottom = 20
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        # Y-axis: -15dB to +15dB
        db_min = -15.0
        db_max = 15.0
        db_range = db_max - db_min

        def db_to_y(db):
            """Convert dB to y pixel coordinate."""
            normalized = (db_max - db) / db_range  # Invert: higher dB = lower y
            return margin_top + normalized * plot_height

        def freq_to_x(freq):
            """Convert frequency (Hz) to x pixel coordinate (log scale)."""
            log_freq = math.log10(freq)
            log_min = math.log10(20)
            log_max = math.log10(20000)
            normalized = (log_freq - log_min) / (log_max - log_min)
            return margin_left + normalized * plot_width

        # Draw horizontal grid lines
        grid_pen = QPen(QColor("#3a3a3a"), 1)
        painter.setPen(grid_pen)

        for db in [-12, -6, 0, 6, 12]:
            y = db_to_y(db)
            painter.drawLine(margin_left, int(y), width - margin_right, int(y))

            # Label
            if db == 0:
                painter.setPen(QColor("#888888"))
                painter.drawText(5, int(y) + 4, f"{db} dB")
                painter.setPen(grid_pen)
            else:
                painter.setPen(QColor("#555555"))
                painter.drawText(5, int(y) + 4, f"{db:+d}")
                painter.setPen(grid_pen)

        # Draw vertical grid lines at octave intervals
        for freq in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            if 20 <= freq <= 20000:
                x = freq_to_x(freq)
                painter.drawLine(int(x), margin_top, int(x), height - margin_bottom)

                # Label
                painter.setPen(QColor("#555555"))
                if freq >= 1000:
                    label = f"{freq // 1000}k"
                else:
                    label = str(freq)
                painter.drawText(int(x) - 10, height - 5, label)
                painter.setPen(grid_pen)

        # Draw frequency response curve
        curve_pen = QPen(QColor("#00d4ff"), 2)  # Cyan
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
            marker_pen = QPen(QColor(255, 212, 64, 150), 1, Qt.PenStyle.DashLine)
            marker_fill = QColor(255, 212, 64)
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
            warning_pen = QPen(QColor(255, 183, 77, 180), 2)
            warning_fill = QColor(255, 183, 77, 80)
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

        # Draw overlay curve if enabled
        if self.show_overlay and self.overlay_bands:
            overlay_pen = QPen(QColor(255, 140, 0), 2)  # Orange for overlay
            painter.setPen(overlay_pen)

            overlay_points = []
            for i, freq in enumerate(self.freq_points):
                x = freq_to_x(freq)
                y = db_to_y(self.overlay_response_db[i])
                overlay_points.append((int(x), int(y)))

            for i in range(len(overlay_points) - 1):
                x1, y1 = overlay_points[i]
                x2, y2 = overlay_points[i + 1]
                painter.drawLine(x1, y1, x2, y2)

            # Add legend when overlay is shown
            legend_x = width - 120
            legend_y = 20

            # Main curve label (Current)
            painter.setPen(QPen(QColor(100, 200, 100), 2))
            painter.drawText(legend_x, legend_y, "Current")

            # Overlay curve label (New)
            painter.setPen(QPen(QColor(255, 140, 0), 2))
            painter.drawText(legend_x, legend_y + 18, "New")
