"""
OBS-style level meters

Meters are QObject models that hold the readings and know how to paint them;
MeterWidget and the Qt Quick item are thin views over a model.
LevelMeter shows RMS level as a filled bar with peak hold indicator.
Color gradient: green → yellow → red
"""

import math
import time
from collections import deque
from collections.abc import Callable

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Property, QObject, Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QPainter, QLinearGradient, QPen, QFont

from ..analysis.voice_setup import CLIP_FLAT_TOP_S
from .theme import PALETTE, qcolor


SCALE_TICK_LENGTH = 3
SCALE_TEXT_GAP = 7
# A clip lights the marker for this long; a later clip restarts the time.
CLIP_HOLD_SECONDS = 3.0
# The engine counts every sample at or above the level raw_capture_is_clipped
# uses, at its fixed 48 kHz processing rate (devices are resampled to it first).
# The CLIP tag lights when the count rises by the samples a flat top lasts there
# (6). The count has no runs in it, so six separate touches also light the tag:
# it says the signal reached full scale that many times, not that it was held.
CLIP_FLAT_TOP_SAMPLES = round(48_000 * CLIP_FLAT_TOP_S)
# The red strip at the top of the bar stays lit this long after a peak at full scale.
CLIP_STRIP_SECONDS = 0.5
# How far back peaks are remembered for the health advice.
RECENT_PEAK_SECONDS = 10.0


def _linear_to_db(value: float | None) -> float | None:
    """Convert a linear peak to dB; nothing usable gives ``None``."""
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return 20.0 * math.log10(value)


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


class MeterModel(QObject):
    """What every meter view reads: presentation properties and a change signal."""

    changed = Signal()
    # Only when the spoken value differs, so assistive technology is not told about every repaint.
    valueTextChanged = Signal()

    def __init__(self, name: str = "", tip: str = "", parent: QObject | None = None):
        super().__init__(parent)
        self._name, self._tip, self._enabled = name, tip, True
        self._announced_value: str | None = None
        self.changed.connect(self._sync_value_text)

    def _sync_value_text(self) -> None:
        value = self._read_value()
        if value != self._announced_value:
            self._announced_value = value
            self.valueTextChanged.emit()

    def _read_value(self) -> str:
        """The reading as words, in whole units so that it changes only when it matters."""
        return ""

    def set_enabled(self, enabled: bool) -> None:
        if enabled != self._enabled:
            self._enabled = enabled
            self.changed.emit()

    name = Property(str, lambda self: self._name, constant=True)
    toolTip = Property(str, lambda self: self._tip, constant=True)
    enabled = Property(bool, lambda self: self._enabled, set_enabled, notify=changed)
    shown = Property(bool, lambda self: True, constant=True)
    description = Property(str, lambda self: self._describe(), notify=changed)
    valueText = Property(str, lambda self: self._read_value(), notify=valueTextChanged)

    def _describe(self) -> str:
        """What assistive technology adds to the name right now."""
        return ""

    def paint_meter(self, painter: QPainter, width: float, height: float) -> None:
        """Paint this meter into any Qt painter without changing its size."""
        raise NotImplementedError


class LevelMeter(MeterModel):
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

    def __init__(
        self, label: str = "", show_scale: bool = True, name: str = "",
        parent: QObject | None = None, clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(name, parent=parent)
        self.clock = clock
        self.label_text = label
        self.show_scale = show_scale
        self.measurement_available = False
        self.peak_available = False
        self.rms_db = -120.0
        self.peak_db = -120.0
        self.peak_hold_db = -120.0
        self.is_clipping = False
        self._clip_strip_until = 0.0
        self._decayed_at = 0.0
        self.clip_marker = False
        self.muted = False
        self.clip_peak_db = self.DB_MIN
        # How many times the signal reached full scale in the window that lit the marker.
        self.clip_samples = 0
        self._clip_until = 0.0
        self._last_clip_at: float | None = None
        self._last_full_scale_count: int | None = None
        self._full_scale_rises: deque[tuple[float, int]] = deque()
        self._recent_peaks: deque[tuple[float, float]] = deque()

        # Peak hold decay timer
        self.decay_timer = QTimer(self)
        self.decay_timer.timeout.connect(self._decay_peak_hold)
        self.decay_timer.setInterval(50)  # Runs only while a measured peak has something to decay.

    def _sync_decay_timer(self) -> None:
        active = self.measurement_available and (
            self.clip_marker or (self.peak_available and (
                self.peak_hold_db > self.DB_MIN or self.is_clipping))
        )
        if active and not self.decay_timer.isActive():
            self._decayed_at = self.clock()
            self.decay_timer.start()
        elif not active:
            self.decay_timer.stop()

    def set_levels(self, rms_db: float | None, peak_db: float | None):
        """Update measured levels; unavailable peaks are never estimated."""
        if rms_db is None or not math.isfinite(rms_db):
            self.set_unavailable()
            return
        if peak_db is None or not math.isfinite(peak_db):
            self.set_rms_level(rms_db)
            return
        previous = (self.rms_db, self.peak_db, self.measurement_available, self.peak_available)
        self.measurement_available = self.peak_available = True
        self.rms_db = max(self.DB_MIN, min(self.DB_MAX, rms_db))
        self.peak_db = max(self.DB_MIN, min(self.DB_MAX, peak_db))
        if self.peak_db > self.peak_hold_db:
            self.peak_hold_db = self.peak_db
        if peak_db >= -0.5:
            self.is_clipping = True
            self._clip_strip_until = self.clock() + CLIP_STRIP_SECONDS
        self._sync_decay_timer()
        if previous != (self.rms_db, self.peak_db, True, True):
            self.changed.emit()

    def note_peak(
        self,
        peak: float | None,
        clip_peak: float | None,
        full_scale_count: int | None = None,
    ) -> None:
        """Fold the true peaks since the last read (linear) into the hold line and clip marker.

        The hold line follows ``peak``. The marker needs the cumulative
        full-scale sample count to rise by ``CLIP_FLAT_TOP_SAMPLES`` within
        ``CLIP_HOLD_SECONDS``; the first count only sets the baseline. A peak
        says nothing about how often full scale was reached, so it never lights
        the marker. ``clip_peak``, the level at the point being judged (the same
        value as ``peak`` for the input, what actually leaves the app for the
        output), only labels the marker. It clears itself ``CLIP_HOLD_SECONDS``
        after the last clip.
        """
        if not self.measurement_available:
            return
        now = self.clock()
        previous = (self.peak_hold_db, self.clip_marker, self.clip_samples)
        hold_db = _linear_to_db(peak)
        final_db = _linear_to_db(clip_peak)
        if hold_db is not None:
            self.peak_hold_db = max(self.peak_hold_db, min(self.DB_MAX, hold_db))
            self._recent_peaks.append((now, hold_db))
        while self._recent_peaks and now - self._recent_peaks[0][0] >= RECENT_PEAK_SECONDS:
            self._recent_peaks.popleft()
        rise = 0
        if full_scale_count is not None:
            baseline = self._last_full_scale_count
            if baseline is not None:
                # A count below the baseline means a new stream started from zero.
                rise = full_scale_count - baseline if full_scale_count >= baseline else full_scale_count
            self._last_full_scale_count = full_scale_count
        if rise:
            self._full_scale_rises.append((now, rise))
        while self._full_scale_rises and now - self._full_scale_rises[0][0] >= CLIP_HOLD_SECONDS:
            self._full_scale_rises.popleft()
        recent_samples = sum(count for _, count in self._full_scale_rises)
        if rise and recent_samples >= CLIP_FLAT_TOP_SAMPLES:
            if not self.clip_marker:
                self.clip_peak_db = self.DB_MIN
                self.clip_samples = 0
            self.clip_samples = max(self.clip_samples, recent_samples)
            self.clip_marker = True
            self.clip_peak_db = max(self.clip_peak_db, final_db if final_db is not None else 0.0)
            self._last_clip_at = now
            self._clip_until = now + CLIP_HOLD_SECONDS
        self._sync_decay_timer()
        if previous != (self.peak_hold_db, self.clip_marker, self.clip_samples):
            self.changed.emit()

    def recent_peak_db(self, seconds: float) -> float | None:
        """Loudest bar peak (dB, not clamped to the scale) of the last ``seconds``, or ``None``."""
        now = self.clock()
        return max((db for at, db in self._recent_peaks if now - at < seconds), default=None)

    def clipped_within(self, seconds: float) -> bool:
        """Whether the marker last lit less than ``seconds`` ago."""
        return self._last_clip_at is not None and self.clock() - self._last_clip_at < seconds

    def full_scale_samples_within(self, seconds: float) -> int:
        """Times the signal reached full scale in the clip that lit the marker, if that was within ``seconds``."""
        return self.clip_samples if self.clipped_within(seconds) else 0

    def forget_full_scale_count(self) -> None:
        """Make the next count a new baseline, so samples counted meanwhile are not new clips."""
        self._last_full_scale_count = None
        self._full_scale_rises.clear()

    def clear_clip_marker(self) -> None:
        """Drop the marker and the clips that lit it; counting carries on from the last count."""
        lit = self.clip_marker
        self.clip_marker = False
        self.clip_peak_db = self.DB_MIN
        self.clip_samples = 0
        self._last_clip_at = None
        self._full_scale_rises.clear()
        self._sync_decay_timer()
        if lit:
            self.changed.emit()

    def _clear_clip(self) -> None:
        self.clip_marker = False
        self.clip_peak_db = self.DB_MIN
        self.clip_samples = 0
        self._last_clip_at = None
        self.forget_full_scale_count()
        self._recent_peaks.clear()

    def set_rms_level(self, rms_db: float) -> None:
        """Show RMS when no peak measurement is available."""
        if not math.isfinite(rms_db):
            self.set_unavailable()
            return
        previous = (self.rms_db, self.measurement_available, self.peak_available)
        self.measurement_available = True
        self.peak_available = False
        self.rms_db = max(self.DB_MIN, min(self.DB_MAX, rms_db))
        self.peak_db = self.peak_hold_db = self.DB_MIN
        self.is_clipping = False
        self._sync_decay_timer()
        if previous != (self.rms_db, True, False):
            self.changed.emit()

    def set_unavailable(self) -> None:
        changed = self.measurement_available
        self.measurement_available = self.peak_available = False
        self.rms_db = self.peak_db = self.peak_hold_db = self.DB_MIN
        self.is_clipping = False
        self._clear_clip()
        self.decay_timer.stop()
        if changed:
            self.changed.emit()

    def _decay_peak_hold(self):
        """Decay measured peak hold; stop repainting once it reaches the floor."""
        previous = (self.peak_hold_db, self.is_clipping, self.clip_marker)
        now = self.clock()
        # Fall by the time that passed, so a late timer tick does not slow the hold line.
        self.peak_hold_db = max(self.DB_MIN, self.peak_hold_db - self.PEAK_DECAY_RATE * (now - self._decayed_at))
        self._decayed_at = now
        if self.is_clipping and now >= self._clip_strip_until:
            self.is_clipping = False
        if self.clip_marker and now >= self._clip_until:
            self.clip_marker = False
            self.clip_peak_db = self.DB_MIN
        self._sync_decay_timer()
        if previous != (self.peak_hold_db, self.is_clipping, self.clip_marker):
            self.changed.emit()

    def _read_value(self) -> str:
        return f"{self.rms_db:.0f} dB" if self.measurement_available else "no signal"

    def set_muted(self, muted: bool) -> None:
        """Say that the bars show the level before a mute, which nobody hears."""
        if muted != self.muted:
            self.muted = muted
            self.changed.emit()

    def _describe(self) -> str:
        notes = ["Muted: the level shown is the level before the mute."] if self.muted else []
        if self.clip_marker:
            notes.append(
                f"Clipping: the signal reached full scale {self.clip_samples} times "
                f"(peak {self.clip_peak_db:.1f} dB) within the last {CLIP_HOLD_SECONDS:g} seconds."
            )
        return " ".join(notes)

    def _db_to_y(self, db: float, height: float) -> float:
        """Convert dB value to Y coordinate (0 = top, height = bottom)."""
        # Normalize to 0-1 range
        normalized = (db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        normalized = max(0.0, min(1.0, normalized))
        # Invert because Y=0 is top
        return height * (1.0 - normalized)

    def paint_meter(self, painter: QPainter, width: float, height: float) -> None:
        """Paint this meter into any Qt painter without changing its size."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = int(round(width))
        height = int(round(height))

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

        if not self.measurement_available:
            painter.setPen(QPen(self.COLOR_SCALE))
            painter.drawText(QRectF(2, 4, meter_width - 2, meter_height), Qt.AlignmentFlag.AlignCenter, "--")

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
        if self.is_clipping:
            painter.fillRect(2, 0, meter_width - 2, 4, self.COLOR_CLIP)

        # Draw label at bottom; while the clip marker is lit a "CLIP" tag replaces it,
        # so the warning is a word on a shape and does not depend on colour. A mute
        # comes first: the output is silent, so a clip is not heard.
        if self.label_text:
            text = "MUTE" if self.muted else self.label_text
            if self.clip_marker and not self.muted:
                text = "CLIP"
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self.COLOR_CLIP)
                painter.drawRoundedRect(
                    QRectF(2, height - label_height + 1, width - 4, label_height - 2), 3, 3
                )
                painter.setPen(QPen(qcolor(PALETTE.text_on_data_color)))
            else:
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
                text,
            )


class GainReductionMeter(MeterModel):
    """Horizontal gain reduction meter (shows compression amount)."""

    COLOR_BACKGROUND = qcolor(PALETTE.data_surface)
    COLOR_REDUCTION = qcolor(PALETTE.meter_reduction)

    def __init__(self, name: str = "", parent: QObject | None = None):
        super().__init__(name, parent=parent)
        self.measurement_available = False
        self.gain_reduction_db = 0.0

    def _read_value(self) -> str:
        return f"{self.gain_reduction_db:.0f} dB" if self.measurement_available else "no reading"

    def set_gain_reduction(self, db: float | None):
        """Update reduction, keeping unavailable distinct from measured zero."""
        available = db is not None and math.isfinite(db)
        value = max(0.0, min(24.0, db)) if available and db is not None else 0.0
        if (value, available) != (self.gain_reduction_db, self.measurement_available):
            self.gain_reduction_db = value
            self.measurement_available = available
            self.changed.emit()

    def paint_meter(self, painter: QPainter, width: float, height: float) -> None:
        """Paint this meter into any Qt painter without changing its size."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = int(round(width))
        height = int(round(height))

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
        text = f"GR: {self.gain_reduction_db:.1f} dB" if self.measurement_available else "GR: --"
        painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter, text)


class ConfidenceMeter(MeterModel):
    """VAD confidence meter showing speech probability (0.0 to 1.0)."""

    def __init__(self, name: str = "", tip: str = "", parent: QObject | None = None):
        super().__init__(name, tip, parent)
        self.measurement_available = False
        self.confidence = 0.0
        self.threshold = 0.5

    def _read_value(self) -> str:
        return f"{self.confidence * 100:.0f} percent" if self.measurement_available else "no reading"

    def set_confidence(self, value: float | None):
        """Update confidence, or explicitly clear an unavailable measurement."""
        available = value is not None and math.isfinite(value)
        confidence = max(0.0, min(1.0, value)) if available and value is not None else 0.0
        if (confidence, available) != (self.confidence, self.measurement_available):
            self.confidence = confidence
            self.measurement_available = available
            self.changed.emit()

    def set_threshold(self, value: float):
        self.threshold = max(0.0, min(1.0, value))
        self.changed.emit()

    def paint_meter(self, painter: QPainter, width: float, height: float) -> None:
        """Paint this meter into any Qt painter without changing its size."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = int(round(width))
        height = int(round(height))
        rect = QRectF(0, 0, width, height)

        # Background (dark gray)
        painter.fillRect(rect, qcolor(PALETTE.data_surface_raised))

        if not self.measurement_available:
            painter.setPen(QPen(qcolor(PALETTE.meter_scale)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "--")

        # Fill based on confidence with color gradient
        fill_width = int(width * self.confidence)

        if fill_width > 0:
            # Gradient from red (low) -> yellow (medium) -> green (high)
            gradient = QLinearGradient(0, 0, width, 0)
            gradient.setColorAt(0.0, qcolor(PALETTE.meter_danger))
            gradient.setColorAt(0.5, qcolor(PALETTE.meter_caution))
            gradient.setColorAt(1.0, qcolor(PALETTE.meter_safe))

            painter.fillRect(0, 0, fill_width, height, gradient)

        threshold_x = int(width * self.threshold)
        painter.setPen(QPen(qcolor(PALETTE.meter_peak), 2))
        painter.drawLine(threshold_x, 0, threshold_x, height)


class MeterWidget(QWidget):
    """Paint a meter model and mirror its accessible name, tooltip and enabled state."""

    def __init__(self, model: MeterModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.setAccessibleName(model.property("name"))
        self.setToolTip(model.property("toolTip"))
        self.setAccessibleDescription(model.property("description"))
        if isinstance(model, LevelMeter):
            # Wider minimum to accommodate the scale.
            self.setMinimumSize(50 if model.show_scale else 25, 120)
        elif isinstance(model, GainReductionMeter):
            self.setMinimumHeight(16)
            self.setMaximumHeight(20)
        else:
            self.setMinimumHeight(20)
        model.changed.connect(self._model_changed)
        self.setEnabled(model.property("enabled"))

    def _model_changed(self) -> None:
        self.setEnabled(self.model.property("enabled"))
        description = self.model.property("description")
        if description != self.accessibleDescription():
            self.setAccessibleDescription(description)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.model.paint_meter(painter, self.width(), self.height())
