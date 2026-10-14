"""A read-only picture of the EQ now and the EQ a setup proposes."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from mic_eq import eq_magnitude_response_v2
from mic_eq.config import EQSettings

from .eq_graph import format_frequency_hz
from .theme import PALETTE, qcolor

_F_MIN, _F_MAX = 20.0, 20_000.0
_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 40.0, 12.0, 8.0, 22.0
_GRID_FREQUENCIES = (50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0)
_LABELLED_FREQUENCIES = (100.0, 1_000.0, 10_000.0)
# Below this the two curves are the same picture.
_SAME_DB = 0.1
# The modes that skip the EQ stage, as the person knows them.
_UNUSED_MODE_NAMES = {"bypass": "Bypass", "raw": "Raw monitor"}


def eq_response_db(eq: EQSettings, frequencies: np.ndarray, sample_rate: float) -> np.ndarray:
    """What the EQ stage does at each frequency: flat when its switch is off."""
    if not eq.enabled:
        return np.zeros(frequencies.size)
    total = np.zeros(frequencies.size)
    # The native response takes a whole stage of ten bands; an absent layer adds nothing.
    for bands in filter(None, (eq.bands, eq.correction_bands)):
        total += np.asarray(
            eq_magnitude_response_v2(
                frequencies.tolist(), [band.to_native() for band in bands], sample_rate
            ),
            dtype=float,
        )
    return total


class EQComparisonWidget(QWidget):
    """The EQ now (dashed) over the proposed EQ (solid). It takes no input."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAccessibleName("EQ now compared with the proposed EQ")
        self._frequencies = np.geomspace(_F_MIN, _F_MAX, 120)
        self._now = np.zeros(self._frequencies.size)
        self._proposed = np.zeros(self._frequencies.size)
        self._unused_mode_name = ""
        self.setVisible(False)

    def set_curves(
        self, now: EQSettings, proposed: EQSettings, sample_rate: float, unused_mode: str = ""
    ) -> None:
        """Draw both; ``unused_mode`` names a processing mode that skips the EQ now."""
        self._unused_mode_name = _UNUSED_MODE_NAMES.get(unused_mode, "")
        # Each band's own frequency is on the grid, so a narrow peak is drawn at its height.
        marks = [
            band.frequency_hz
            for eq in (now, proposed)
            for bands in (eq.bands, eq.correction_bands or ())
            for band in bands
            if band.enabled and _F_MIN <= band.frequency_hz <= _F_MAX
        ]
        self._frequencies = np.unique(np.concatenate([np.geomspace(_F_MIN, _F_MAX, 120), marks]))
        self._now = eq_response_db(now, self._frequencies, sample_rate)
        self._proposed = eq_response_db(proposed, self._frequencies, sample_rate)
        self.setAccessibleDescription(self.description())
        self.update()

    def largest_change(self) -> tuple[float, float]:
        """The biggest difference the proposal makes, in dB (signed), and the Hz it is at."""
        difference = self._proposed - self._now
        index = int(np.argmax(np.abs(difference)))
        return float(difference[index]), float(self._frequencies[index])

    def description(self) -> str:
        change, frequency = self.largest_change()
        if abs(change) < _SAME_DB:
            return "The proposed EQ is the same as the EQ now."
        now = (
            f"the EQ now, which {self._unused_mode_name} does not use"
            if self._unused_mode_name
            else "the EQ now"
        )
        return (
            f"Largest change: {change:+.1f} dB at {format_frequency_hz(frequency)}. "
            f"Solid line: the proposed EQ. Dashed line: {now}."
        )

    def lines(self) -> list[tuple[str, np.ndarray, bool]]:
        """(label, response in dB, dashed) for each curve; the shape tells them apart, not the colour."""
        now = f"Now (not used in {self._unused_mode_name})" if self._unused_mode_name else "Now"
        return [(now, self._now, True), ("Proposed", self._proposed, False)]

    def _range_db(self) -> float:
        peak = max(float(np.max(np.abs(self._now))), float(np.max(np.abs(self._proposed))), 1.0)
        return max(6.0, 3.0 * math.ceil((peak + 1.0) / 3.0))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = float(self.width()), float(self.height())
        plot = QRectF(
            _MARGIN_LEFT, _MARGIN_TOP,
            width - _MARGIN_LEFT - _MARGIN_RIGHT, height - _MARGIN_TOP - _MARGIN_BOTTOM,
        )
        painter.fillRect(self.rect(), qcolor(PALETTE.data_surface))
        span = self._range_db()
        log_min, log_max = math.log10(_F_MIN), math.log10(_F_MAX)

        def x_of(frequency: float) -> float:
            return plot.left() + plot.width() * (math.log10(frequency) - log_min) / (log_max - log_min)

        def y_of(db: float) -> float:
            return plot.center().y() - plot.height() / 2.0 * max(-1.0, min(1.0, db / span))

        grid = qcolor(PALETTE.data_grid)
        label_font = QFont(self.font())
        label_font.setPointSizeF(max(7.0, label_font.pointSizeF() - 1.0))
        painter.setFont(label_font)
        muted = qcolor(PALETTE.data_text_muted)
        for frequency in _GRID_FREQUENCIES:
            painter.setPen(QPen(grid, 1.0))
            painter.drawLine(QPointF(x_of(frequency), plot.top()), QPointF(x_of(frequency), plot.bottom()))
            if frequency in _LABELLED_FREQUENCIES:
                painter.setPen(muted)
                painter.drawText(
                    QRectF(x_of(frequency) - 24, plot.bottom() + 3, 48, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    format_frequency_hz(frequency),
                )
        for db in (-span, 0.0, span):
            painter.setPen(QPen(grid, 1.5 if db == 0.0 else 1.0))
            painter.drawLine(QPointF(plot.left(), y_of(db)), QPointF(plot.right(), y_of(db)))
            painter.setPen(muted)
            painter.drawText(
                QRectF(0, y_of(db) - 8, _MARGIN_LEFT - 4, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{db:+.0f} dB" if db else "0 dB",
            )
        painter.setClipRect(plot)
        now_color, proposed_color = qcolor(PALETTE.text_muted), qcolor(PALETTE.accent)
        lines = self.lines()
        for (_label, response, dashed), color, thickness in zip(
            lines, (now_color, proposed_color), (1.6, 2.6)
        ):
            path = QPainterPath()
            for index, (frequency, db) in enumerate(zip(self._frequencies, response)):
                point = QPointF(x_of(float(frequency)), y_of(float(db)))
                path.moveTo(point) if index == 0 else path.lineTo(point)
            pen = QPen(color, thickness)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.setClipping(False)
        self._paint_legend(painter, plot, lines, now_color, proposed_color)
        painter.end()

    def _paint_legend(
        self,
        painter: QPainter,
        plot: QRectF,
        lines: list[tuple[str, np.ndarray, bool]],
        now: QColor,
        proposed: QColor,
    ) -> None:
        entries = (
            (lines[0][0], now, True, 1.6),
            (lines[1][0], proposed, False, 2.6),
        )
        widths = [28.0 + painter.fontMetrics().horizontalAdvance(label) + 12.0 for label, *_ in entries]
        x = plot.right() - sum(widths)
        for (label, color, dashed, thickness), width in zip(entries, widths):
            pen = QPen(color, thickness)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, plot.top() + 10), QPointF(x + 22, plot.top() + 10))
            painter.setPen(qcolor(PALETTE.text_primary))
            painter.drawText(QPointF(x + 28, plot.top() + 14), label)
            x += width
