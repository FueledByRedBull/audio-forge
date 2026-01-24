"""
OBS-style visual level meter widget

Shows RMS level as a filled bar with peak hold indicator.
Color gradient: green → yellow → red
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QPen, QFont


class LevelMeter(QWidget):
    """OBS-style vertical level meter with peak hold and dB scale."""

    # Color constants
    COLOR_GREEN = QColor(76, 175, 80)      # -inf to -20 dB
    COLOR_YELLOW = QColor(255, 235, 59)    # -20 to -6 dB
    COLOR_RED = QColor(244, 67, 54)        # -6 to 0 dB
    COLOR_BACKGROUND = QColor(30, 30, 30)
    COLOR_PEAK_HOLD = QColor(255, 255, 255)
    COLOR_CLIP = QColor(255, 0, 0)
    COLOR_SCALE = QColor(150, 150, 150)

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
        self.setMinimumHeight(200)

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
            gradient.setColorAt(0.0, self.COLOR_GREEN)      # Bottom (quiet)
            gradient.setColorAt(0.67, self.COLOR_YELLOW)    # -20 dB
            gradient.setColorAt(0.9, self.COLOR_RED)        # -6 dB
            gradient.setColorAt(1.0, self.COLOR_RED)        # 0 dB

            painter.fillRect(bar_rect, gradient)

        # Draw peak hold line
        if self.peak_hold_db > self.DB_MIN:
            peak_color = self.COLOR_CLIP if self.peak_hold_db >= -0.5 else self.COLOR_PEAK_HOLD
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

                # Draw tick mark
                painter.drawLine(meter_width, int(y), meter_width + 3, int(y))

                # Draw dB value
                db_text = f"{db:d}" if db != 0 else "0"
                text_rect = QRectF(meter_width + 4, y - 6, scale_width - 6, 12)
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, db_text)

        # Draw clipping indicator at top
        if self.is_clipping and self.clip_flash_counter % 2 == 0:
            painter.fillRect(2, 0, meter_width - 2, 4, self.COLOR_CLIP)

        # Draw label at bottom
        if self.label_text:
            painter.setPen(QPen(QColor(200, 200, 200)))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                0, height - label_height, width, label_height,
                Qt.AlignmentFlag.AlignCenter,
                self.label_text
            )

        painter.end()


class StereoLevelMeter(QWidget):
    """Dual level meter for input/output display."""

    def __init__(self, left_label: str = "IN", right_label: str = "OUT", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Horizontal layout for the two meters
        meters_layout = QHBoxLayout()
        meters_layout.setSpacing(4)

        self.left_meter = LevelMeter(left_label, show_scale=True)
        self.right_meter = LevelMeter(right_label, show_scale=True)

        meters_layout.addWidget(self.left_meter)
        meters_layout.addWidget(self.right_meter)

        layout.addLayout(meters_layout)

    def set_input_levels(self, rms_db: float, peak_db: float):
        """Update input meter."""
        self.left_meter.set_levels(rms_db, peak_db)

    def set_output_levels(self, rms_db: float, peak_db: float):
        """Update output meter."""
        self.right_meter.set_levels(rms_db, peak_db)


class GainReductionMeter(QWidget):
    """Horizontal gain reduction meter (shows compression amount)."""

    COLOR_BACKGROUND = QColor(30, 30, 30)
    COLOR_REDUCTION = QColor(255, 152, 0)  # Orange

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
        painter.setPen(QPen(QColor(200, 200, 200)))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        text = f"GR: {self.gain_reduction_db:.1f} dB"
        painter.drawText(
            0, 0, width, height,
            Qt.AlignmentFlag.AlignCenter,
            text
        )

        painter.end()
