"""
Frequency response curve visualization for parametric EQ
"""

import math
import cmath
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt


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

    def _calc_biquad_coefficients(self, freq, gain_db, q, filter_type):
        """
        Calculate biquad filter coefficients.
        Returns (b0, b1, b2, a1, a2) normalized so a0 = 1.
        Uses Robert Bristow-Johnson formulas.
        """
        omega = 2 * math.pi * freq / self.sample_rate
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        alpha = sin_omega / (2.0 * q)
        A = 10 ** (gain_db / 40.0)  # sqrt(10^(dB/20))

        if filter_type == 1:  # Peaking
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_omega
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha / A

        elif filter_type == 0:  # Low shelf
            two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_omega + two_sqrt_a_alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_omega)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_omega - two_sqrt_a_alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_omega + two_sqrt_a_alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_omega)
            a2 = (A + 1.0) + (A - 1.0) * cos_omega - two_sqrt_a_alpha

        else:  # High shelf (filter_type == 2)
            two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_omega + two_sqrt_a_alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_omega)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_omega - two_sqrt_a_alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_omega + two_sqrt_a_alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_omega)
            a2 = (A + 1.0) - (A - 1.0) * cos_omega - two_sqrt_a_alpha

        # Normalize so a0 = 1
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    def _biquad_response(self, freq_hz, b0, b1, b2, a1, a2):
        """
        Calculate magnitude response in dB at given frequency.
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        """
        omega = 2 * math.pi * freq_hz / self.sample_rate
        z = cmath.exp(1j * omega)  # z = e^(j*omega)

        # Evaluate transfer function
        numerator = b0 + b1 * (z ** -1) + b2 * (z ** -2)
        denominator = 1 + a1 * (z ** -1) + a2 * (z ** -2)

        h = numerator / denominator
        magnitude_db = 20 * math.log10(abs(h) + 1e-10)  # Add epsilon to avoid log(0)
        return magnitude_db

    def _update_response(self):
        """Calculate combined frequency response for all bands."""
        # Initialize response to 0 dB
        self.response_db = [0.0] * len(self.freq_points)

        # Sum contribution from each band
        for freq, gain_db, q, filter_type in self.bands:
            # Skip bands with 0 gain (optimization)
            if abs(gain_db) < 0.01:
                continue

            # Calculate coefficients for this band
            b0, b1, b2, a1, a2 = self._calc_biquad_coefficients(freq, gain_db, q, filter_type)

            # Add this band's response at each frequency point
            for i, f in enumerate(self.freq_points):
                db = self._biquad_response(f, b0, b1, b2, a1, a2)
                self.response_db[i] += db

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

    def _update_overlay_response(self):
        """Calculate frequency response for overlay curve."""
        self.overlay_response_db = [0.0] * len(self.freq_points)

        for freq, gain_db, q, filter_type in self.overlay_bands:
            if abs(gain_db) < 0.01:
                continue

            b0, b1, b2, a1, a2 = self._calc_biquad_coefficients(freq, gain_db, q, filter_type)

            for i, f in enumerate(self.freq_points):
                db = self._biquad_response(f, b0, b1, b2, a1, a2)
                self.overlay_response_db[i] += db

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
