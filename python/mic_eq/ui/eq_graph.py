"""Shared data, geometry, interaction, and painting for the EQ response graph."""

from __future__ import annotations

import math
from bisect import bisect_left

from PySide6.QtCore import Property, QLineF, QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QGuiApplication, QPainter, QPen, QPolygonF

from mic_eq import eq_magnitude_response_v2, input_prefilter_response_db
from mic_eq.analysis.eq_quality import EqInteractionWarning, evaluate_eq_quality
from mic_eq.config import EQSettings
from .eq_state import EQ_FILTER_OPTIONS
from .theme import PALETTE, RADIUS_CONTROL, qcolor


# A band that is off: the native response gives it exactly 0 dB.
OFF_BAND: tuple[str, float, float, float, int, bool] = ("bell", 1000.0, 0.0, 1.0, 12, False)

FILTER_NAMES = {filter_type: name for name, filter_type in EQ_FILTER_OPTIONS}
# The processing modes in which the EQ does not run, as the graph names them.
NOT_RUNNING_MODES = {"bypass": "Bypass", "raw": "Raw monitor"}
GRAPH_HELP = (
    "Click and drag a band handle. Left and right change frequency; "
    "up and down change gain for bell and shelf filters. Use left bracket "
    "and right bracket to select a band from the keyboard. Double-click a "
    "handle, or press Delete or Backspace on the selected band, to reset it. "
    "Moving the pointer over the graph reads out "
    "the frequency and level of the thick curve. "
    "The thick curve is what the signal really gets: the input filter "
    "plus every EQ filter that is on, or the input filter alone when "
    "the EQ switch is off. Thin colored curves are the bands, the "
    "dashed curve is the Auto-EQ layer and the dotted curve is the "
    "input filter: the DC blocker and an 80 Hz high-pass. Adaptive input "
    "cleanup keeps that high-pass, but follows the signal beyond it: it "
    "raises the high-pass to 100 or 120 Hz when it detects rumble and notches "
    "hum, and neither is drawn."
)


def format_frequency_hz(frequency_hz: float, *, exact: bool = False) -> str:
    """Hz below a kilohertz, else kHz with three digits.

    ``exact`` keeps whole hertz, as a band's frequency box shows them: kHz is
    used only when it loses none.
    """
    hertz = round(frequency_hz)
    khz = f"{frequency_hz / 1000.0:.3g}"
    if frequency_hz >= 1000.0 and not (exact and round(float(khz) * 1000.0) != hertz):
        return f"{khz} kHz"
    return f"{hertz} Hz"


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
    # A shelf's own curve is at half its gain at the corner, where its handle
    # sits: the handle follows the pointer and the gain moves twice as far.
    SHELF_HANDLE_RATIO = 0.5

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
    def y_to_db(cls, y: float, width: float, height: float) -> float:
        _plot_width, plot_height = cls.plot_size(width, height)
        normalized = min(
            1.0,
            max(0.0, (float(y) - cls.MARGIN_TOP) / plot_height),
        )
        return cls.DISPLAY_DB_MAX - normalized * (
            cls.DISPLAY_DB_MAX - cls.DISPLAY_DB_MIN
        )

    @classmethod
    def y_to_gain(cls, y: float, width: float, height: float) -> float:
        clamped = min(cls.GAIN_MAX_DB, max(cls.GAIN_MIN_DB, cls.y_to_db(y, width, height)))
        return round(clamped * 10.0) / 10.0

    @classmethod
    def band_handle_position(
        cls,
        bands: list[tuple],
        band_index: int,
        width: float,
        height: float,
        handle_db: list[float] | None = None,
    ) -> tuple[float, float]:
        """Where a handle is drawn: on its band's own curve at its frequency.

        ``handle_db`` is that curve's height per band, as ``EQGraphModel``
        keeps it. Without it the handle sits at the band's gain, which is
        where a bell's curve is.
        """
        filter_type, frequency, gain, _q, _slope, _enabled = bands[band_index]
        if handle_db is not None:
            handle_gain = handle_db[band_index]
        else:
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
        handle_db: list[float] | None = None,
    ) -> int | None:
        nearest: tuple[float, int] | None = None
        for index in range(len(bands)):
            handle_x, handle_y = cls.band_handle_position(
                bands, index, width, height, handle_db
            )
            distance = math.hypot(float(x) - handle_x, float(y) - handle_y)
            if distance > cls.HIT_RADIUS:
                continue
            candidate = (distance, index)
            if nearest is None or candidate < nearest:
                nearest = candidate
        return nearest[1] if nearest else None


class EQGraphModel(QObject):
    """One EQ graph snapshot and interaction state."""

    changed = Signal()
    # The pointer moved over the graph; nothing else did, so no repaint of a
    # recomputed response rides on it.
    hoverChanged = Signal()
    bandResetRequested = Signal(int)
    bandDragStarted = Signal(int)
    bandDragged = Signal(int, float, float)
    bandDragFinished = Signal(int, float, float)
    bandDragCancelled = Signal(int, float, float)
    bandSelected = Signal(int)
    # The accessible description changed at rest: not on every step of a drag,
    # which a screen reader would announce one by one.
    accessibleDescriptionChanged = Signal()

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
        # Where each band's handle rides: its own curve at its own frequency,
        # whether or not the band is on, within the gain range.
        self.handle_db: list[float] = [0.0] * len(self.bands)
        self.correction_response_db: list[float] = []
        # total_response_db is what the signal really gets: the fixed input
        # conditioning that comes before the EQ, plus the EQ unless its switch
        # is off. Only the part of the conditioning that does not follow the
        # signal can be drawn (the DC blocker and the 80 Hz high-pass that
        # cleanup starts from); see input_prefilter_response_db.
        self.eq_enabled = True
        # What the processor runs: "bypass" skips the EQ, "raw" skips the input
        # filter as well. The bands stay editable and drawn either way.
        self.engine_mode = "normal"
        self.cleanup_mode = "off"
        self.input_filter_db: list[float] = []
        self.total_response_db: list[float] = []
        self.interaction_warnings: list[EqInteractionWarning] = []
        self._selected_band_index: int | None = None
        self._drag_band_index: int | None = None
        self._drag_origin: tuple[float, float] | None = None
        self._drag_press = (0.0, 0.0)
        self._drag_handle = (0.0, 0.0)
        self.hover: tuple[int, int] | None = None
        self._base_freq_points = self.generate_log_frequencies(20, 20_000, 100)
        self._drag_started = False
        self.freq_points = list(self._base_freq_points)
        self.response_db = [0.0] * len(self.freq_points)
        self._update_response()
        self._description = self.accessible_description()
        self.changed.connect(self._refresh_description)
        self.bandDragFinished.connect(self._refresh_description)

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
    def eq_running(self) -> bool:
        return self.eq_enabled and self.engine_mode == "normal"

    @property
    def not_running_label(self) -> str:
        mode = NOT_RUNNING_MODES.get(self.engine_mode)
        return f"EQ not running: {mode}" if mode else ""

    @property
    def dragging_band_index(self) -> int | None:
        """The band a drag is moving; a press that has not travelled yet is no drag."""
        return self._drag_band_index if self._drag_started else None

    def select_band(self, band_index: int) -> None:
        if band_index == self._selected_band_index:
            return
        self._selected_band_index = band_index
        self.changed.emit()
        self.bandSelected.emit(band_index)

    def _band_numbers(self, index: int) -> list[str]:
        """Frequency, then gain and Q for the filters that have them."""
        filter_type, frequency, gain, q, _slope, _enabled = self.bands[index]
        numbers = [format_frequency_hz(frequency, exact=True)]
        if filter_type in self.GAIN_FILTER_TYPES:
            numbers.append(f"{gain:+.1f} dB")
        if not filter_type.endswith("_pass"):
            numbers.append(f"Q {q:.2g}")
        return numbers

    def band_readout(self, index: int) -> str:
        return "  ".join([f"Band {index + 1}", *self._band_numbers(index)])

    def _selection_summary(self) -> str:
        index = self._selected_band_index
        if index is None:
            return ""
        band = self.bands[index]
        parts = [FILTER_NAMES[band[0]], *self._band_numbers(index), "on" if band[5] else "off"]
        return f"Band {index + 1} selected: {', '.join(parts)}."

    def accessible_description(self) -> str:
        not_running = self.not_running_label
        if not_running:
            skipped = "The EQ and the input filter are" if self.engine_mode == "raw" else "The EQ is"
            thick = "flat" if self.engine_mode == "raw" else "the input filter alone"
            not_running += (
                f". {skipped} skipped in this mode; the thin band curves show what "
                f"the EQ would do, and the thick curve is {thick}."
            )
        return " ".join(
            part for part in (GRAPH_HELP, not_running, self._selection_summary()) if part
        )

    def _refresh_description(self, *_drag_end) -> None:
        description = self.accessible_description()
        if self.dragging_band_index is None and description != self._description:
            self._description = description
            self.accessibleDescriptionChanged.emit()

    def _announced_description(self) -> str:
        return self._description

    accessibleDescription = Property(
        str, _announced_description, notify=accessibleDescriptionChanged
    )

    def set_engine_mode(self, mode: str) -> None:
        """Follow what the processor runs: normal, bypass or raw monitor."""
        if mode not in ("normal", *NOT_RUNNING_MODES):
            raise ValueError(f"unknown processing mode: {mode!r}")
        if mode == self.engine_mode:
            return
        self.engine_mode = mode
        self._update_response()
        self.changed.emit()

    def band_handle_position(
        self, band_index: int, width: float, height: float
    ) -> tuple[float, float]:
        return self.GEOMETRY.band_handle_position(
            self.bands, band_index, width, height, self.handle_db
        )

    def nearest_band_handle(
        self, x: float, y: float, width: float, height: float
    ) -> int | None:
        return self.GEOMETRY.nearest_band_handle(
            self.bands, x, y, width, height, self.handle_db
        )

    def response_at(self, frequency_hz: float) -> float:
        """The total curve at one frequency, evaluated there and not read off the drawn points.

        A narrow peak is off by up to a dB between two drawn points. This costs
        about 0.01 ms, and it leaves the stored response alone.
        """
        frequencies = [frequency_hz]
        eq = sum(
            eq_magnitude_response_v2(frequencies, stage, self.sample_rate)[0]
            for stage in (self.bands, self.correction_bands)
            if stage
        )
        return self._input_filter(frequencies)[0] + (eq if self.eq_running else 0.0)

    def set_hover(self, x: float, y: float, width: float, height: float) -> None:
        geometry = self.GEOMETRY
        plot_width, plot_height = geometry.plot_size(width, height)
        inside = (
            geometry.MARGIN_LEFT <= x <= geometry.MARGIN_LEFT + plot_width
            and geometry.MARGIN_TOP <= y <= geometry.MARGIN_TOP + plot_height
        )
        self._set_hover((round(x), round(y)) if inside else None)

    def clear_hover(self) -> None:
        self._set_hover(None)

    def _set_hover(self, hover: tuple[int, int] | None) -> None:
        if hover != self.hover:
            self.hover = hover
            self.hoverChanged.emit()

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
        """Take a handle; it drags once the pointer has travelled the platform's drag distance."""
        band_index = self.nearest_band_handle(x, y, width, height)
        if band_index is None:
            return False
        self.select_band(band_index)
        self._drag_band_index = band_index
        self._drag_started = False
        band = self.bands[band_index]
        self._drag_origin = (float(band[1]), float(band[2]))
        self._drag_press = (x, y)
        self._drag_handle = self.band_handle_position(band_index, width, height)
        return True

    def _request_reset(self, band_index: int) -> None:
        self.cancel_drag()
        self.select_band(band_index)
        self.bandResetRequested.emit(band_index)

    def request_band_reset(
        self, x: float, y: float, width: float, height: float
    ) -> bool:
        """A double-click: ask for the band whose handle is under the pointer.

        Qt Quick delivers a double-click's second press as a press first, so a
        drag may be open; it ends unchanged, and its release then writes nothing.
        """
        band_index = self.nearest_band_handle(x, y, width, height)
        if band_index is None:
            return False
        self._request_reset(band_index)
        return True

    def mouse_move(self, x: float, y: float, width: float, height: float) -> bool:
        """Track the pointer for the readout; True while it drags a band."""
        self.set_hover(x, y, width, height)
        if self._drag_band_index is None:
            return False
        if not self._drag_started:
            press_x, press_y = self._drag_press
            distance = QGuiApplication.styleHints().startDragDistance()
            if math.hypot(x - press_x, y - press_y) < distance:
                return True
            self._drag_started = True
            self.bandDragStarted.emit(self._drag_band_index)
        frequency, gain = self._update_dragged_band(x, y, width, height)
        self.bandDragged.emit(self._drag_band_index, frequency, gain)
        return True

    def mouse_release(
        self, x: float, y: float, width: float, height: float
    ) -> bool:
        if self._drag_band_index is None:
            return False
        if self._drag_started:
            self._update_dragged_band(x, y, width, height)
        return self.finish_drag()

    def finish_drag(self) -> bool:
        """End the active drag where the band stands."""
        band_index = self._drag_band_index
        if band_index is None:
            return False
        _filter_type, frequency, gain, *_ = self.bands[band_index]
        started, self._drag_started = self._drag_started, False
        self._drag_band_index = None
        self._drag_origin = None
        if started:
            self.bandDragFinished.emit(band_index, frequency, gain)
        return True

    def cancel_drag(self) -> bool:
        """End the active drag by putting the band back where it was pressed."""
        band_index, origin = self._drag_band_index, self._drag_origin
        if band_index is None or origin is None:
            return False
        if not self._drag_started:
            self._drag_band_index = None
            self._drag_origin = None
            return True
        frequency, gain = origin
        filter_type, _frequency, _gain, q, slope, enabled = self.bands[band_index]
        self.bands[band_index] = (filter_type, frequency, gain, q, slope, enabled)
        self._drag_started = False
        self._drag_band_index = None
        self._drag_origin = None
        self._update_response()
        self.changed.emit()
        self.bandDragCancelled.emit(band_index, frequency, gain)
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
        if key == Qt.Key.Key_Escape and self.cancel_drag():
            return True
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._request_reset(self._selected_band_index)
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
        origin = self._drag_origin
        if self._drag_band_index is None or origin is None:
            raise RuntimeError("no EQ band drag is active")
        # The handle moves by the pointer's travel, so a press that was not
        # dead centre does not jump, and a click leaves the band as it was.
        if (x, y) == self._drag_press:
            return origin
        geometry = self.GEOMETRY
        filter_type = self.bands[self._drag_band_index][0]
        press_x, press_y = self._drag_press
        handle_x, handle_y = self._drag_handle
        frequency = geometry.x_to_frequency(handle_x + (x - press_x), width, height)
        gain = origin[1]
        if filter_type in self.GAIN_FILTER_TYPES:
            travel_db = geometry.y_to_db(
                handle_y + (y - press_y), width, height
            ) - geometry.y_to_db(handle_y, width, height)
            if filter_type.endswith("_shelf"):
                travel_db /= geometry.SHELF_HANDLE_RATIO
            gain = min(
                geometry.GAIN_MAX_DB,
                max(geometry.GAIN_MIN_DB, round((gain + travel_db) * 10.0) / 10.0),
            )
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
        self.handle_db = [
            min(
                self.GEOMETRY.GAIN_MAX_DB,
                max(
                    self.GEOMETRY.GAIN_MIN_DB,
                    eq_magnitude_response_v2(
                        [band[1]], self._band_alone(index, enabled=True), self.sample_rate
                    )[0],
                ),
            )
            for index, band in enumerate(self.bands)
        ]
        self.input_filter_db = self._input_filter(self.freq_points)
        self.total_response_db = [
            conditioning + (tone if self.eq_running else 0.0)
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

    def _input_filter(self, frequencies: list[float]) -> list[float]:
        """The input conditioning as the engine applies it in the current mode."""
        if self.engine_mode == "raw":
            return [0.0] * len(frequencies)
        # Adaptive cleanup runs on the full path only; bypass keeps the fixed
        # high-pass, which is where cleanup starts too.
        cleanup = "off" if self.engine_mode == "bypass" else self.cleanup_mode
        return list(input_prefilter_response_db(frequencies, cleanup, self.sample_rate))

    def _response(self, bands: list[tuple]) -> list[float]:
        return list(
            eq_magnitude_response_v2(self.freq_points, bands, self.sample_rate)
        )

    def _band_alone(self, index: int, *, enabled: bool | None = None) -> list[tuple]:
        """The tone stage with every other band off: one band's own response."""
        bands = [OFF_BAND] * len(self.bands)
        band = self.bands[index]
        bands[index] = band if enabled is None else (*band[:5], enabled)
        return bands


class EQGraphRenderer:
    """Paint one graph model into a QPainter."""

    GEOMETRY = EQGraphGeometry
    TOTAL_WIDTH = 2.0
    SELECTED_BAND_WIDTH = 1.5
    BAND_WIDTH = 1.0
    FILL_ALPHA = 44
    AXIS_TOLERANCE_DB = 0.3
    CASING_WIDTH = 3.0
    MUTED_OPACITY = 0.4
    LABEL_POINT_SIZE = 8
    LABEL_PADDING = (6.0, 3.0)
    READOUT_GAP = 14.0
    CAPTION_INSET = 6.0

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

        caption = cls.caption_layout(painter.font(), model, width, height)
        if caption is not None:
            cls._paint_label(painter, *caption, PALETTE.data_text)

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
            x, y = model.band_handle_position(index, width, height)
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
        cls._paint_readout(painter, model, width, height)
        painter.restore()

    @classmethod
    def _plot_rect(cls, width: float, height: float) -> QRectF:
        geometry = cls.GEOMETRY
        return QRectF(
            geometry.MARGIN_LEFT,
            geometry.MARGIN_TOP,
            *geometry.plot_size(width, height),
        )

    @classmethod
    def _label_font(cls, font: QFont) -> QFont:
        label = QFont(font)
        label.setPointSize(cls.LABEL_POINT_SIZE)
        label.setBold(True)
        return label

    @classmethod
    def _label_size(cls, font: QFont, text: str) -> tuple[float, float]:
        metrics = QFontMetricsF(cls._label_font(font))
        pad_x, pad_y = cls.LABEL_PADDING
        return metrics.horizontalAdvance(text) + 2 * pad_x, metrics.height() + 2 * pad_y

    @classmethod
    def readout_layout(
        cls, font: QFont, model: EQGraphModel, width: float, height: float
    ) -> tuple[str, QRectF] | None:
        """The readout's text and box: a dragged band's numbers, else the total under the pointer.

        The box sits a fixed gap from its anchor, above and to the right, on
        the other side where that would leave the plot. Hover only reads the
        values already computed.
        """
        geometry = cls.GEOMETRY
        dragged = model.dragging_band_index
        if dragged is not None:
            text = model.band_readout(dragged)
            anchor = QPointF(*model.band_handle_position(dragged, width, height))
        elif model.hover is not None:
            hover_x, hover_y = model.hover
            frequency = geometry.x_to_frequency(hover_x, width, height)
            level = round(model.response_at(frequency), 1)
            shown = f"{level:+.1f}" if level else "0.0"
            text = f"{format_frequency_hz(frequency, exact=True)}  {shown} dB"
            anchor = QPointF(hover_x, hover_y)
        else:
            return None
        box_width, box_height = cls._label_size(font, text)
        plot = cls._plot_rect(width, height)
        x = anchor.x() + cls.READOUT_GAP
        if x + box_width > plot.right():
            x = anchor.x() - cls.READOUT_GAP - box_width
        y = anchor.y() - cls.READOUT_GAP - box_height
        if y < plot.top():
            y = anchor.y() + cls.READOUT_GAP
        x = min(max(x, plot.left()), plot.right() - box_width)
        y = min(max(y, plot.top()), plot.bottom() - box_height)
        return text, QRectF(x, y, box_width, box_height)

    @classmethod
    def caption_layout(
        cls, font: QFont, model: EQGraphModel, width: float, height: float
    ) -> tuple[str, QRectF] | None:
        """The "EQ not running" caption in the first plot corner no handle is near."""
        text = model.not_running_label
        if not text:
            return None
        box_width, box_height = cls._label_size(font, text)
        plot = cls._plot_rect(width, height)
        inset = cls.CAPTION_INSET
        left, top = plot.left() + inset, plot.top() + inset
        right, bottom = plot.right() - box_width - inset, plot.bottom() - box_height - inset
        boxes = [
            QRectF(x, y, box_width, box_height)
            for x, y in ((right, top), (left, top), (right, bottom), (left, bottom))
        ]
        reach = cls.GEOMETRY.HANDLE_RADIUS + 2.0
        handles = [
            QPointF(*model.band_handle_position(index, width, height))
            for index in range(len(model.bands))
        ]
        free = next(
            (
                box
                for box in boxes
                if not any(
                    box.adjusted(-reach, -reach, reach, reach).contains(handle)
                    for handle in handles
                )
            ),
            boxes[0],
        )
        return text, free

    @classmethod
    def _paint_label(
        cls, painter: QPainter, text: str, box: QRectF, color: str
    ) -> None:
        painter.save()
        painter.setFont(cls._label_font(painter.font()))
        painter.setPen(QPen(qcolor(PALETTE.border_strong), 1))
        painter.setBrush(qcolor(PALETTE.data_surface_raised))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(qcolor(color))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    @classmethod
    def _paint_readout(
        cls, painter: QPainter, model: EQGraphModel, width: int, height: int
    ) -> None:
        layout = cls.readout_layout(painter.font(), model, width, height)
        if layout is None:
            return
        if model.dragging_band_index is None and model.hover is not None:
            # A guide and a dot on the total, so the number has a place on the curve.
            geometry = cls.GEOMETRY
            x, _y = model.hover
            plot = cls._plot_rect(width, height)
            db = model.response_at(geometry.x_to_frequency(x, width, height))
            curve_y = plot.top() + (geometry.DISPLAY_DB_MAX - db) * plot.height() / (
                geometry.DISPLAY_DB_MAX - geometry.DISPLAY_DB_MIN
            )
            painter.setPen(QPen(qcolor(PALETTE.data_text_muted, alpha=120), 1))
            painter.drawLine(QLineF(x, plot.top(), x, plot.bottom()))
            painter.setPen(QPen(qcolor(PALETTE.data_curve), 2))
            painter.setBrush(qcolor(PALETTE.data_surface))
            painter.drawEllipse(
                QPointF(x, min(max(curve_y, plot.top()), plot.bottom())), 3.5, 3.5
            )
        cls._paint_label(painter, *layout, PALETTE.data_text)

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

        if not model.eq_running:
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
            if not model.eq_running:
                painter.setOpacity(cls.MUTED_OPACITY)
            correction = ys(model.correction_response_db)
            dashes = segments(correction, [i for i in range(len(xs) - 1) if i % 3 != 2])
            painter.setPen(pen(PALETTE.data_surface, cls.CASING_WIDTH))
            painter.drawLines(dashes)
            painter.setPen(pen(PALETTE.data_text, cls.BAND_WIDTH))
            painter.drawLines(dashes)
        painter.restore()
