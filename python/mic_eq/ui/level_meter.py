"""
OBS-style visual level meter widget

Shows RMS level as a filled bar with peak hold indicator.
Color gradient: green → yellow → red
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QLinearGradient, QPen, QFont

from .theme import PALETTE, qcolor


SCALE_TICK_LENGTH = 3
SCALE_TEXT_GAP = 7


def _scale_mark_geometry(
    meter_width: int,
    y: float,
    scale_width: int,
) -> tuple[int, QRectF]:
    """Return separated tick and label geometry for one meter scale mark."""

    tick_end = meter_width + SCALE_TICK_LENGTH
    text_left = tick_end + SCALE_TEXT_GAP
    text_width = max(0, meter_width + scale_width - text_left)
    return tick_end, QRectF(text_left, y - 6, text_width, 12)


class LevelMeter(QWidget):
    """OBS-style vertical level meter with peak hold and dB scale."""

    # Color constants
    COLOR_GREEN = qcolor(PALETTE.meter_safe)
    COLOR_YELLOW = qcolor(PALETTE.meter_caution)
    COLOR_RED = qcolor(PALETTE.meter_danger)
    COLOR_BACKGROUND = qcolor(PALETTE.data_surface)
    COLOR_PEAK_HOLD = qcolor(PALETTE.meter_peak)
    COLOR_CLIP = qcolor(PALETTE.meter_clip)
    COLOR_SCALE = qcolor(PALETTE.meter_scale)

    # dB scale
    DB_MIN = -60.0
    DB_MAX = 0.0

    # Peak hold decay (dB per second)
    PEAK_DECAY_RATE = 20.0

    # Scale marks to display
    SCALE_MARKS = [0, -6, -12, -18, -24, -30, -40, -50, -60]

    def __init__(self, label: str = "", show_scale: bool = True, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.show_scale = show_scale
        self.rms_db = -120.0
        self.peak_db = -120.0
        self.peak_hold_db = -120.0
        self.is_clipping = False
        self.clip_flash_counter = 0

        # Minimum size - wider to accommodate scale
        self.setMinimumWidth(50 if show_scale else 25)
        self.setMinimumHeight(120)

        # Peak hold decay timer
        self.decay_timer = QTimer(self)
        self.decay_timer.timeout.connect(self._decay_peak_hold)
        self.decay_timer.start(50)  # 20 Hz update

    def set_levels(self, rms_db: float, peak_db: float):
        """Update the meter levels."""
        self.rms_db = max(self.DB_MIN, min(self.DB_MAX, rms_db))
        self.peak_db = max(self.DB_MIN, min(self.DB_MAX, peak_db))

        # Update peak hold (only increases)
        if self.peak_db > self.peak_hold_db:
            self.peak_hold_db = self.peak_db

        # Check for clipping
        if peak_db >= -0.5:
            self.is_clipping = True
            self.clip_flash_counter = 10  # Flash for ~0.5 seconds

        self.update()

    def _decay_peak_hold(self):
        """Decay the peak hold indicator over time."""
        decay_amount = self.PEAK_DECAY_RATE * 0.05  # 50ms intervals
        self.peak_hold_db -= decay_amount
        if self.peak_hold_db < self.DB_MIN:
            self.peak_hold_db = self.DB_MIN

        # Decay clip flash
        if self.clip_flash_counter > 0:
            self.clip_flash_counter -= 1
            if self.clip_flash_counter == 0:
                self.is_clipping = False

        self.update()

    def _db_to_y(self, db: float, height: float) -> float:
        """Convert dB value to Y coordinate (0 = top, height = bottom)."""
        # Normalize to 0-1 range
        normalized = (db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        normalized = max(0.0, min(1.0, normalized))
        # Invert because Y=0 is top
        return height * (1.0 - normalized)

    def paintEvent(self, event):
        """Paint the level meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        width = self.width()
        height = self.height()

        # Reserve space for label at bottom
        label_height = 18 if self.label_text else 0

        # Reserve space for scale on the right
        scale_width = 28 if self.show_scale else 0
        meter_width = width - scale_width - 2
        meter_height = height - label_height - 8

        # Draw background for meter
        painter.fillRect(2, 4, meter_width - 2, meter_height, self.COLOR_BACKGROUND)

        # Calculate bar positions
        rms_y = self._db_to_y(self.rms_db, meter_height) + 4
        peak_hold_y = self._db_to_y(self.peak_hold_db, meter_height) + 4

        # Draw RMS bar with gradient
        if self.rms_db > self.DB_MIN:
            bar_rect = QRectF(4, rms_y, meter_width - 6, meter_height + 4 - rms_y)

            # Create gradient
            gradient = QLinearGradient(0, meter_height + 4, 0, 4)
            gradient.setColorAt(0.0, self.COLOR_GREEN)  # Bottom (quiet)
            gradient.setColorAt(0.67, self.COLOR_YELLOW)  # -20 dB
            gradient.setColorAt(0.9, self.COLOR_RED)  # -6 dB
            gradient.setColorAt(1.0, self.COLOR_RED)  # 0 dB

            painter.fillRect(bar_rect, gradient)

        # Draw peak hold line
        if self.peak_hold_db > self.DB_MIN:
            peak_color = (
                self.COLOR_CLIP if self.peak_hold_db >= -0.5 else self.COLOR_PEAK_HOLD
            )
            pen = QPen(peak_color, 2)
            painter.setPen(pen)
            painter.drawLine(4, int(peak_hold_y), meter_width - 2, int(peak_hold_y))

        # Draw scale with numbers
        if self.show_scale:
            painter.setPen(QPen(self.COLOR_SCALE, 1))
            font = QFont()
            font.setPointSize(7)
            font.setBold(False)
            painter.setFont(font)

            for db in self.SCALE_MARKS:
                y = self._db_to_y(db, meter_height) + 4
                tick_end, text_rect = _scale_mark_geometry(
                    meter_width,
                    y,
                    scale_width,
                )

                # Draw tick mark
                painter.drawLine(meter_width, int(y), tick_end, int(y))

                # Draw dB value
                db_text = f"{db:d}" if db != 0 else "0"
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    db_text,
                )

        # Draw clipping indicator at top
        if self.is_clipping and self.clip_flash_counter % 2 == 0:
            painter.fillRect(2, 0, meter_width - 2, 4, self.COLOR_CLIP)

        # Draw label at bottom
        if self.label_text:
            painter.setPen(QPen(qcolor(PALETTE.meter_scale)))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                0,
                height - label_height,
                width,
                label_height,
                Qt.AlignmentFlag.AlignCenter,
                self.label_text,
            )

        painter.end()


class GainReductionMeter(QWidget):
    """Horizontal gain reduction meter (shows compression amount)."""

    COLOR_BACKGROUND = qcolor(PALETTE.data_surface)
    COLOR_REDUCTION = qcolor(PALETTE.meter_reduction)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gain_reduction_db = 0.0
        self.setMinimumHeight(16)
        self.setMaximumHeight(20)

    def set_gain_reduction(self, db: float):
        """Update the gain reduction display (positive dB = reduction)."""
        self.gain_reduction_db = max(0.0, min(24.0, db))
        self.update()

    def paintEvent(self, event):
        """Paint the gain reduction meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Draw background
        painter.fillRect(0, 0, width, height, self.COLOR_BACKGROUND)

        # Draw reduction bar (from right to left)
        if self.gain_reduction_db > 0:
            # Normalize: 0 dB = no bar, 24 dB = full bar
            bar_width = (self.gain_reduction_db / 24.0) * (width - 4)
            bar_rect = QRectF(width - 2 - bar_width, 2, bar_width, height - 4)
            painter.fillRect(bar_rect, self.COLOR_REDUCTION)

        # Draw label
        painter.setPen(QPen(qcolor(PALETTE.meter_scale)))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        text = f"GR: {self.gain_reduction_db:.1f} dB"
        painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter, text)

        painter.end()


class ConfidenceMeter(QWidget):
    """VAD confidence meter showing speech probability (0.0 to 1.0)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(20)
        self.confidence = 0.0
        self.setAutoFillBackground(False)

    def set_confidence(self, value: float):
        """Update confidence value (0.0 to 1.0)."""
        self.confidence = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Background (dark gray)
        painter.fillRect(rect, qcolor(PALETTE.data_surface_raised))

        # Fill based on confidence with color gradient
        fill_width = int(width * self.confidence)

        if fill_width > 0:
            # Gradient from red (low) -> yellow (medium) -> green (high)
            gradient = QLinearGradient(0, 0, width, 0)
            gradient.setColorAt(0.0, qcolor(PALETTE.meter_danger))
            gradient.setColorAt(0.5, qcolor(PALETTE.meter_caution))
            gradient.setColorAt(1.0, qcolor(PALETTE.meter_safe))

            painter.fillRect(0, 0, fill_width, height, gradient)

        # Threshold marker (default 0.5)
        threshold_x = int(width * 0.5)
        painter.setPen(QPen(qcolor(PALETTE.meter_peak), 2))
        painter.drawLine(threshold_x, 0, threshold_x, height)

        painter.end()
