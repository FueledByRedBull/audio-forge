"""What the EQ graph draws: each band, the Auto-EQ layer, the input filter and their total."""

from __future__ import annotations

import re

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QGuiApplication, QImage, QPainter
from PySide6.QtTest import QSignalSpy

from mic_eq import AudioProcessor, eq_magnitude_response_v2, input_prefilter_response_db
from mic_eq.ui.eq_graph import (
    OFF_BAND,
    EQGraphGeometry,
    EQGraphModel,
    EQGraphRenderer,
    format_frequency_hz,
)
from mic_eq.ui.theme import PALETTE, qcolor

WIDTH, HEIGHT = 900, 300
FLAT = [("bell", 1000.0, 0.0, 1.0, 12, True)] * 10
# type, Hz, dB, Q, slope, on
TONE = [
    ("high_pass", 75.0, 0.0, 0.707, 24, True),
    ("low_shelf", 160.0, 3.5, 0.7, 12, True),
    ("bell", 300.0, -3.0, 1.4, 12, True),
    ("bell", 600.0, 4.0, 1.0, 12, False),
    ("bell", 1400.0, 4.0, 1.2, 12, True),
    ("bell", 2800.0, 5.5, 2.0, 12, True),
    ("bell", 4200.0, -5.0, 3.0, 12, True),
    ("notch", 7000.0, 0.0, 6.0, 12, True),
    ("bell", 9500.0, 2.0, 1.0, 12, True),
    ("high_shelf", 12000.0, -4.0, 0.7, 12, True),
]
AUTO_EQ = [
    ("low_shelf", 72.0, -2.0, 0.75, 12, True),
    ("bell", 145.0, -1.0, 1.0, 12, True),
    ("bell", 290.0, 0.4, 1.1, 12, True),
    ("bell", 580.0, 1.1, 1.2, 12, True),
    ("bell", 1160.0, 1.8, 1.4, 12, True),
    ("bell", 2320.0, 2.2, 1.6, 12, True),
    ("bell", 4640.0, 1.2, 1.5, 12, True),
    ("bell", 7600.0, -0.8, 1.4, 12, True),
    ("bell", 11100.0, -1.1, 1.1, 12, True),
    ("high_shelf", 15100.0, -0.6, 0.8, 12, True),
]


def model_with(tone, correction=(), *, enabled=True, cleanup="off") -> EQGraphModel:
    model = EQGraphModel()
    model.set_all_params(tone, correction=correction, enabled=enabled)
    model.set_input_conditioning(cleanup, model.sample_rate)
    return model


def with_band(bands, index, **fields):
    names = ("filter_type", "frequency_hz", "gain_db", "q", "slope", "enabled")
    band = dict(zip(names, bands[index])) | fields
    return [*bands[:index], tuple(band[name] for name in names), *bands[index + 1 :]]


def paint(model: EQGraphModel) -> QImage:
    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    EQGraphRenderer.paint(painter, model, WIDTH, HEIGHT)
    painter.end()
    return image


def x_at(hz: float) -> int:
    return round(EQGraphGeometry.frequency_to_x(hz, WIDTH, HEIGHT))


def y_at(db: float) -> int:
    return round(EQGraphGeometry.gain_to_y(db, WIDTH, HEIGHT))


def line_x(hz: float) -> int:
    return int(EQGraphGeometry.frequency_to_x(hz, WIDTH, HEIGHT))


def line_y(db: float) -> int:
    return int(EQGraphGeometry.gain_to_y(db, WIDTH, HEIGHT))


def lightness(image: QImage, x: int, y: int) -> int:
    return image.pixelColor(x, y).lightness()


def warmth(image: QImage, x: int, ys: range) -> int:
    """How orange the warmest pixel of a few columns is: red above blue."""
    return max(
        image.pixelColor(column, y).red() - image.pixelColor(column, y).blue()
        for column in range(x - 3, x + 4)
        for y in ys
    )


def apart(pixel, color: str) -> int:
    target = qcolor(color)
    return (
        abs(pixel.red() - target.red())
        + abs(pixel.green() - target.green())
        + abs(pixel.blue() - target.blue())
    )


def distance_to(image: QImage, x: int, ys: range, color: str) -> int:
    """The closest any pixel of a few columns around x comes to a color."""
    return min(
        apart(image.pixelColor(column, y), color)
        for column in range(x - 3, x + 4)
        for y in ys
    )


def segment_middle(model: EQGraphModel, index: int) -> int:
    xs = [EQGraphGeometry.frequency_to_x(f, WIDTH, HEIGHT) for f in model.freq_points]
    return round((xs[index] + xs[index + 1]) / 2)


def segment_near(model: EQGraphModel, hz: float, remainder: int) -> int:
    """A segment index near a frequency whose position in the dash pattern is chosen."""
    start = next(i for i, f in enumerate(model.freq_points) if f >= hz)
    return next(i for i in range(start, start + 3) if i % 3 == remainder)


def test_total_is_the_input_filter_plus_the_bands_that_are_on_plus_the_auto_eq_layer(qapp):
    model = model_with(TONE, AUTO_EQ)

    assert model.band_curves_db[3] == []
    assert all(curve for index, curve in enumerate(model.band_curves_db) if index != 3)
    assert model.correction_response_db
    assert model.input_filter_db
    parts = [curve for curve in model.band_curves_db if curve] + [
        model.correction_response_db,
        model.input_filter_db,
    ]
    assert model.total_response_db == pytest.approx(
        [sum(values) for values in zip(*parts)], abs=1e-9
    )
    # response_db keeps its meaning: what the EQ stages are set to do.
    tone = eq_magnitude_response_v2(model.freq_points, TONE, model.sample_rate)
    auto = eq_magnitude_response_v2(model.freq_points, AUTO_EQ, model.sample_rate)
    assert model.response_db == pytest.approx([a + b for a, b in zip(tone, auto)], abs=1e-9)
    assert model.total_response_db != pytest.approx(model.response_db, abs=0.5)


def test_a_band_that_is_off_has_no_curve_and_no_part_in_the_total(qapp):
    off = model_with(TONE, AUTO_EQ)
    # The same band on, but with nothing to do: what an off band must amount to.
    gone = model_with(with_band(TONE, 3, gain_db=0.0, enabled=True), AUTO_EQ)

    assert off.bands[3][2] == 4.0
    assert off.band_curves_db[3] == []
    shared = set(off.freq_points) & set(gone.freq_points)
    assert len(shared) >= 100
    total = dict(zip(off.freq_points, off.total_response_db))
    other = dict(zip(gone.freq_points, gone.total_response_db))
    assert [total[f] for f in sorted(shared)] == pytest.approx(
        [other[f] for f in sorted(shared)], abs=1e-9
    )
    on = model_with(with_band(TONE, 3, enabled=True), AUTO_EQ)
    assert on.band_curves_db[3]
    assert max(abs(a - b) for a, b in zip(on.total_response_db, off.total_response_db)) > 1.0


def test_band_curves_follow_drag_and_keyboard_edits(qapp):
    model = model_with(TONE, AUTO_EQ)
    band = 4
    before = [list(curve) for curve in model.band_curves_db]
    x, y = EQGraphGeometry.band_handle_position(model.bands, band, WIDTH, HEIGHT)

    assert model.mouse_press(x, y, WIDTH, HEIGHT)
    assert model.mouse_move(x + 60, y - 40, WIDTH, HEIGHT)

    others = [i for i in range(10) if i != band]
    assert len(others) == 9
    assert model.band_curves_db[band] != before[band]
    alone = [OFF_BAND] * 10
    alone[band] = model.bands[band]
    assert model.band_curves_db[band] == pytest.approx(
        eq_magnitude_response_v2(model.freq_points, alone, model.sample_rate),
        abs=1e-12,
    )
    for other in (i for i in others if model.bands[i][5]):
        alone = [OFF_BAND] * 10
        alone[other] = model.bands[other]
        assert model.band_curves_db[other] == pytest.approx(
            eq_magnitude_response_v2(model.freq_points, alone, model.sample_rate),
            abs=1e-12,
        )

    model.mouse_release(x + 60, y - 40, WIDTH, HEIGHT)
    dragged = list(model.band_curves_db[band])
    assert model.key_press(Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
    assert model.band_curves_db[band] != dragged


def test_a_narrow_peak_and_notch_are_sampled_at_their_own_frequency(qapp):
    peak = with_band(FLAT, 4, frequency_hz=1234.0, gain_db=12.0, q=10.0)
    model = model_with(with_band(peak, 5, filter_type="notch", frequency_hz=777.0, q=10.0))

    assert max(model.band_curves_db[4]) == pytest.approx(12.0, abs=1e-6)
    assert min(model.band_curves_db[5]) < -100.0
    assert model.freq_points == sorted(model.freq_points)


def test_the_input_filter_follows_the_cleanup_mode_and_the_sample_rate(qapp):
    off = model_with(TONE, AUTO_EQ, cleanup="off")
    gentle = model_with(TONE, AUTO_EQ, cleanup="gentle")

    for model in (off, gentle):
        assert model.input_filter_db == pytest.approx(
            input_prefilter_response_db(model.freq_points, model.cleanup_mode, 48_000.0)
        )
    low = off.freq_points.index(min(off.freq_points, key=lambda f: abs(f - 40.0)))
    # Cleanup keeps the 80 Hz high-pass, so the static filter cuts the bass in every mode.
    assert off.input_filter_db == pytest.approx(gentle.input_filter_db)
    assert gentle.input_filter_db[low] < -10.0
    # Neither the EQ-only response nor the bands change with the mode.
    assert off.response_db == pytest.approx(gentle.response_db, abs=1e-12)
    assert off.band_curves_db == gentle.band_curves_db
    assert off.total_response_db == pytest.approx(gentle.total_response_db, abs=1e-12)

    changed = []
    off.changed.connect(lambda: changed.append(True))
    off.set_input_conditioning("gentle", 48_000.0)
    assert changed and off.input_filter_db == pytest.approx(gentle.input_filter_db)
    off.set_input_conditioning("gentle", 44_100.0)
    assert off.input_filter_db != pytest.approx(gentle.input_filter_db, abs=1e-3)


def test_the_model_uses_the_rate_the_processor_runs(qapp):
    processor = AudioProcessor()
    try:
        assert EQGraphModel().sample_rate == float(processor.sample_rate())
    finally:
        processor.stop()


def test_with_the_eq_switch_off_the_total_is_the_input_filter_alone(qapp):
    on = model_with(TONE, AUTO_EQ)
    off = model_with(TONE, AUTO_EQ, enabled=False)

    assert off.total_response_db == pytest.approx(off.input_filter_db, abs=1e-12)
    assert off.response_db == pytest.approx(on.response_db, abs=1e-12)
    assert off.band_curves_db == on.band_curves_db


def test_the_eq_switch_takes_the_eq_out_of_the_drawn_total(qapp):
    boost = with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0)
    on, off = model_with(boost), model_with(boost, enabled=False)
    column = x_at(900.0)
    # The bell is about +3 dB here: the total passes through it, or does not.
    nearest = min(range(len(on.freq_points)), key=lambda i: abs(on.freq_points[i] - 900.0))
    row = y_at(on.total_response_db[nearest])

    on_image, off_image = paint(on), paint(off)

    assert max(lightness(on_image, column, row + dy) for dy in (-1, 0, 1)) > 200
    assert max(lightness(off_image, column, row + dy) for dy in (-1, 0, 1)) < 120
    assert lightness(off_image, column, y_at(0.0)) > 200


def test_the_eq_switch_mutes_the_bands_instead_of_hiding_them(qapp):
    # Two bands that nearly cancel, so the total does not hide either curve.
    tone = with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0)
    tone = with_band(tone, 1, frequency_hz=1300.0, gain_db=-6.0)
    column = x_at(850.0)
    plot = range(EQGraphGeometry.MARGIN_TOP + 4, HEIGHT - EQGraphGeometry.MARGIN_BOTTOM)

    on = paint(model_with(tone))
    muted = paint(model_with(tone, enabled=False))
    gone = paint(model_with(with_band(tone, 4, enabled=False), enabled=False))

    assert warmth(on, column, plot) > 150
    assert 30 < warmth(muted, column, plot) < 150
    assert warmth(gone, column, plot) < 15


def test_the_total_is_drawn_where_the_model_says(qapp):
    model = model_with(with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0))
    image = paint(model)

    for hz in (300.0, 700.0, 1700.0, 4000.0):
        index = model.freq_points.index(min(model.freq_points, key=lambda f: abs(f - hz)))
        x = round(EQGraphGeometry.frequency_to_x(model.freq_points[index], WIDTH, HEIGHT))
        y = y_at(model.total_response_db[index])
        assert max(lightness(image, x, y + dy) for dy in (-1, 0, 1)) > 200, hz


def test_the_total_is_filled_back_to_zero_for_boosts_and_cuts(qapp):
    boost = model_with(with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0))
    cut = model_with(with_band(FLAT, 4, frequency_hz=1200.0, gain_db=-6.0))
    x = x_at(1200.0)

    def red(model, db):
        return paint(model).pixelColor(x, y_at(db)).red()

    # Between the curve and the 0 dB line is tinted; beyond the curve is not.
    assert red(boost, 3.0) > red(boost, 10.0) + 20
    assert red(cut, -3.0) > red(cut, -10.0) + 20


def test_an_enabled_band_draws_its_own_color_and_an_off_band_does_not(qapp):
    # Two bands that nearly cancel, so the total does not hide either curve.
    tone = with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0)
    tone = with_band(tone, 1, frequency_hz=1300.0, gain_db=-6.0)
    color = PALETTE.eq_band_colors[4]
    x = x_at(850.0)
    plot = range(EQGraphGeometry.MARGIN_TOP + 4, HEIGHT - EQGraphGeometry.MARGIN_BOTTOM)

    on = paint(model_with(tone))
    off = paint(model_with(with_band(tone, 4, enabled=False)))

    assert distance_to(on, x, plot, color) < 36
    assert distance_to(off, x, plot, color) > 150


def test_the_selected_band_is_drawn_on_top_of_its_neighbours(qapp):
    # Bands 3 and 4 are the same curve, and band 0 cancels both from the total.
    tone = with_band(FLAT, 3, frequency_hz=1200.0, gain_db=6.0)
    tone = with_band(tone, 4, frequency_hz=1200.0, gain_db=6.0)
    tone = with_band(tone, 0, frequency_hz=1200.0, gain_db=-12.0)
    x = x_at(850.0)
    plot = range(EQGraphGeometry.MARGIN_TOP + 4, HEIGHT - EQGraphGeometry.MARGIN_BOTTOM)

    for selected in (3, 4):
        model = model_with(tone)
        model.select_band(selected)
        image = paint(model)
        other = 7 - selected
        winner = distance_to(image, x, plot, PALETTE.eq_band_colors[selected])
        loser = distance_to(image, x, plot, PALETTE.eq_band_colors[other])
        assert winner < 36 and winner + 40 < loser, selected


def test_the_auto_eq_layer_is_dashed_apart_from_the_total(qapp):
    # A tone shelf lifts the total 6 dB above the Auto-EQ plateau at +4 dB.
    tone = with_band(FLAT, 9, filter_type="high_shelf", frequency_hz=20.0, gain_db=6.0)
    auto = with_band(FLAT, 9, filter_type="high_shelf", frequency_hz=20.0, gain_db=4.0)
    model = model_with(tone, auto)
    image = paint(model)

    def brightest(index):
        x, y = segment_middle(model, index), y_at(model.correction_response_db[index])
        return max(lightness(image, x, y + dy) for dy in (-1, 0, 1))

    # Two segments in three are drawn, and the third is a gap in the dashes.
    assert brightest(segment_near(model, 1500.0, 0)) > 120
    assert brightest(segment_near(model, 1500.0, 1)) > 120
    assert brightest(segment_near(model, 1500.0, 2)) < 80


def test_the_auto_eq_dashes_stay_visible_on_the_total_when_the_tone_is_flat(qapp):
    auto = with_band(FLAT, 9, filter_type="high_shelf", frequency_hz=20.0, gain_db=4.0)
    model = model_with(FLAT, auto)
    image = paint(model)

    def column(index):
        x, row = segment_middle(model, index), y_at(model.correction_response_db[index])
        return [lightness(image, x, row + dy) for dy in range(-4, 5)]

    drawn = column(segment_near(model, 1500.0, 0))
    skipped = column(segment_near(model, 1500.0, 2))
    assert sum(abs(a - b) for a, b in zip(drawn, skipped)) > 150


def test_the_input_filter_is_dotted_and_is_not_a_band(qapp):
    # A tone shelf keeps the total well above the input filter's roll-off.
    tone = with_band(FLAT, 9, filter_type="high_shelf", frequency_hz=20.0, gain_db=9.0)
    model = model_with(tone, cleanup="off")
    image = paint(model)
    run = next(
        run
        for run in EQGraphRenderer._active_runs(model.input_filter_db)
        if model.freq_points[run.start] < 80.0 < model.freq_points[run.stop]
    )

    def brightest(index):
        x = segment_middle(model, index)
        y = round(
            (y_at(model.input_filter_db[index]) + y_at(model.input_filter_db[index + 1])) / 2
        )
        return max(lightness(image, x, y + dy) for dy in (-2, -1, 0, 1, 2))

    drawn = [i for i in run if (i - run.start) % 2 == 0 and 60.0 < model.freq_points[i] < 150.0]
    skipped = [i for i in run if (i - run.start) % 2 == 1 and 60.0 < model.freq_points[i] < 150.0]
    assert len(drawn) >= 4 and len(skipped) >= 4
    assert all(brightest(i) > 80 for i in drawn)
    assert all(brightest(i) < 40 for i in skipped)


def test_the_selected_handle_prints_its_number(qapp):
    tone = with_band(FLAT, 4, frequency_hz=2000.0, gain_db=3.0)
    model = model_with(tone)
    x, y = (round(v) for v in EQGraphGeometry.band_handle_position(model.bands, 4, WIDTH, HEIGHT))

    def dark(image):
        return sum(
            image.pixelColor(x + dx, y + dy).lightness() < 90
            for dx in range(-4, 5)
            for dy in range(-4, 5)
        )

    plain = dark(paint(model))
    model.select_band(4)
    assert plain == 0
    assert dark(paint(model)) >= 8


def test_a_selected_band_that_is_off_keeps_its_number_off_the_total(qapp):
    # An off band at 0 dB sits right on a flat total.
    tone = with_band(FLAT, 4, frequency_hz=2000.0, enabled=False)
    model = model_with(tone)
    model.select_band(4)
    x, y = (round(v) for v in EQGraphGeometry.band_handle_position(model.bands, 4, WIDTH, HEIGHT))

    assert apart(paint(model).pixelColor(x + 5, y), PALETTE.data_surface) < 30


def test_markers_sit_on_the_total_at_the_nearest_sample(qapp):
    peak = with_band(FLAT, 4, frequency_hz=1000.0, gain_db=12.0, q=10.0)
    model = model_with(peak)
    marker_hz = 1040.0
    model.set_band_markers([marker_hz])
    nearest = min(range(len(model.freq_points)), key=lambda i: abs(model.freq_points[i] - marker_hz))
    x = line_x(marker_hz)
    y = line_y(model.total_response_db[nearest])

    assert apart(paint(model).pixelColor(x, y), PALETTE.data_marker) < 40


def test_the_axis_has_a_zero_line_and_every_decade_down_to_20_hz(qapp):
    # A +9 dB total keeps the curve clear of the lines being read.
    model = model_with(
        with_band(FLAT, 9, filter_type="high_shelf", frequency_hz=20.0, gain_db=9.0)
    )
    image = paint(model)

    assert lightness(image, 600, line_y(0.0)) > lightness(image, 600, line_y(-6.0)) + 3
    level = line_y(10.5)
    for hz in (20.0, 50.0, 100.0):
        x = line_x(hz)
        assert max(lightness(image, x - 1, level), lightness(image, x, level)) > lightness(image, x + 7, level) + 10, hz


def test_curves_stay_inside_the_plot(qapp):
    # A notch is minus infinity at its centre; it must not reach the axis labels.
    notch = with_band(FLAT, 2, filter_type="notch", frequency_hz=130.0, q=4.0)
    image = paint(model_with(notch))
    background = qcolor(PALETTE.data_surface).rgb()
    bottom = HEIGHT - EQGraphGeometry.MARGIN_BOTTOM
    x = x_at(130.0)

    assert [image.pixel(x + dx, bottom + dy) for dx in (-3, 0, 3) for dy in (2, 3)] == [
        background
    ] * 6


HANDLE_CASES = [
    ("bell", 2000.0, 6.0),
    ("low_shelf", 160.0, 8.0),
    ("high_shelf", 6000.0, -8.0),
    ("high_pass", 120.0, 0.0),
    ("low_pass", 9000.0, 0.0),
    ("notch", 1000.0, 0.0),
]


@pytest.mark.parametrize(("filter_type", "hz", "gain"), HANDLE_CASES)
def test_a_handle_rides_its_own_curve_at_the_band_frequency(qapp, filter_type, hz, gain):
    model = model_with(
        with_band(FLAT, 4, filter_type=filter_type, frequency_hz=hz, gain_db=gain, q=1.0)
    )
    own = model.band_curves_db[4][model.freq_points.index(hz)]

    x, y = model.band_handle_position(4, WIDTH, HEIGHT)

    assert x == pytest.approx(EQGraphGeometry.frequency_to_x(hz, WIDTH, HEIGHT))
    # A notch is minus infinity at its centre: its handle stops at the lowest gain.
    assert y == pytest.approx(
        EQGraphGeometry.gain_to_y(max(-12.0, own), WIDTH, HEIGHT), abs=1e-6
    )
    if filter_type.endswith("_shelf"):
        assert abs(y - EQGraphGeometry.gain_to_y(gain, WIDTH, HEIGHT)) > 10


def test_the_handle_of_a_band_that_is_off_waits_where_its_curve_would_be(qapp):
    shelf = dict(filter_type="high_shelf", frequency_hz=6000.0, gain_db=8.0, q=0.7)
    on = model_with(with_band(FLAT, 4, **shelf))
    off = model_with(with_band(FLAT, 4, enabled=False, **shelf))

    assert off.band_curves_db[4] == []
    assert off.band_handle_position(4, WIDTH, HEIGHT) == pytest.approx(
        on.band_handle_position(4, WIDTH, HEIGHT)
    )


def test_hit_testing_follows_the_drawn_handle(qapp):
    model = model_with(
        with_band(FLAT, 9, filter_type="high_shelf", frequency_hz=6000.0, gain_db=8.0)
    )
    x, y = model.band_handle_position(9, WIDTH, HEIGHT)
    nominal = EQGraphGeometry.gain_to_y(8.0, WIDTH, HEIGHT)

    assert nominal < y - 20
    assert model.nearest_band_handle(x, y + 6, WIDTH, HEIGHT) == 9
    assert model.nearest_band_handle(x, nominal, WIDTH, HEIGHT) is None


@pytest.mark.parametrize(("filter_type", "ratio"), [("bell", 1.0)])
def test_a_gain_handle_follows_the_pointer_and_the_gain_moves_by_the_ratio(
    qapp, filter_type, ratio
):
    model = model_with(
        with_band(FLAT, 4, filter_type=filter_type, frequency_hz=3000.0, gain_db=3.0, q=0.7)
    )
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)
    pixels_per_db = (HEIGHT - 54) / 30.0

    assert model.mouse_press(x, y, WIDTH, HEIGHT)
    assert model.mouse_move(x, y - 20, WIDTH, HEIGHT)

    assert model.bands[4][2] == pytest.approx(3.0 + ratio * 20 / pixels_per_db, abs=0.1)
    assert model.band_handle_position(4, WIDTH, HEIGHT)[1] == pytest.approx(y - 20, abs=1.0)


@pytest.mark.parametrize("q", [0.3, 0.707, 10.0])
@pytest.mark.parametrize("filter_type", ["low_shelf", "high_shelf"])
@pytest.mark.parametrize("travel", [-30, 25])
def test_a_shelf_handle_follows_the_pointer_at_any_q(qapp, filter_type, q, travel):
    model = model_with(
        with_band(FLAT, 4, filter_type=filter_type, frequency_hz=3000.0, gain_db=3.0, q=q)
    )
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)
    own = model.band_curves_db[4][model.freq_points.index(3000.0)]
    pixels_per_db = (HEIGHT - 54) / 30.0

    assert own == pytest.approx(1.5, abs=1e-6)
    assert model.mouse_press(x, y, WIDTH, HEIGHT)
    assert model.mouse_move(x, y + travel, WIDTH, HEIGHT)

    assert model.bands[4][2] == pytest.approx(3.0 - 2.0 * travel / pixels_per_db, abs=0.1)
    assert model.band_handle_position(4, WIDTH, HEIGHT)[1] == pytest.approx(y + travel, abs=1.0)
    assert model.handle_db[4] == pytest.approx(model.bands[4][2] / 2.0, abs=1e-6)


def drag_distance() -> int:
    return QGuiApplication.styleHints().startDragDistance()


def test_a_press_that_stays_under_the_drag_distance_changes_nothing(qapp):
    model = model_with(TONE)
    before = list(model.bands)
    started, finished = QSignalSpy(model.bandDragStarted), QSignalSpy(model.bandDragFinished)
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)
    jitter = drag_distance() - 1

    assert jitter >= 1
    assert model.mouse_press(x, y, WIDTH, HEIGHT)
    assert model.mouse_move(x + jitter, y, WIDTH, HEIGHT)
    assert model.dragging_band_index is None
    assert model.mouse_release(x + jitter, y, WIDTH, HEIGHT)

    assert model.bands == before
    assert started.count() == 0 and finished.count() == 0
    assert model.dragging_band_index is None
    assert not model.mouse_release(x, y, WIDTH, HEIGHT)


def test_a_drag_past_the_drag_distance_follows_the_pointer_from_its_first_step(qapp):
    model = model_with(
        with_band(FLAT, 4, frequency_hz=3000.0, gain_db=3.0, q=1.0)
    )
    started = QSignalSpy(model.bandDragStarted)
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)
    first = drag_distance() + 2

    assert model.mouse_press(x, y, WIDTH, HEIGHT)
    assert started.count() == 0
    assert model.mouse_move(x, y - first, WIDTH, HEIGHT)

    assert started.count() == 1 and model.dragging_band_index == 4
    assert model.band_handle_position(4, WIDTH, HEIGHT)[1] == pytest.approx(y - first, abs=1.0)
    # Back under the distance, the drag stays a drag.
    assert model.mouse_move(x, y - 1, WIDTH, HEIGHT)
    assert model.band_handle_position(4, WIDTH, HEIGHT)[1] == pytest.approx(y - 1, abs=1.0)
    model.mouse_release(x, y - 1, WIDTH, HEIGHT)
    assert started.count() == 1


def test_a_click_on_a_shelf_handle_leaves_it_alone(qapp):
    model = model_with(
        with_band(FLAT, 4, filter_type="low_shelf", frequency_hz=437.5, gain_db=2.35)
    )
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)

    model.mouse_press(x + 4, y + 4, WIDTH, HEIGHT)
    model.mouse_release(x + 4, y + 4, WIDTH, HEIGHT)

    assert model.bands[4][1:3] == (437.5, 2.35)


@pytest.mark.parametrize("filter_type", ["high_pass", "low_pass", "notch"])
def test_pass_and_notch_handles_move_horizontally_and_stay_on_their_curve(
    qapp, filter_type
):
    model = model_with(
        with_band(FLAT, 4, filter_type=filter_type, frequency_hz=1000.0, gain_db=5.0, q=2.0)
    )
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)

    model.mouse_press(x, y, WIDTH, HEIGHT)
    model.mouse_move(x + 80, y - 60, WIDTH, HEIGHT)

    assert model.bands[4][1] > 1500.0
    assert model.bands[4][2] == 5.0
    assert model.band_handle_position(4, WIDTH, HEIGHT)[1] == pytest.approx(y, abs=0.5)


def test_frequencies_read_in_hz_below_a_kilohertz_and_in_khz_above():
    assert [
        format_frequency_hz(hz) for hz in (85.0, 999.0, 1000.0, 1250.0, 12_500.0, 20_000.0)
    ] == ["85 Hz", "999 Hz", "1 kHz", "1.25 kHz", "12.5 kHz", "20 kHz"]


def test_exact_frequencies_never_hide_a_hertz_in_the_kilohertz_rounding():
    # The band's frequency box counts whole hertz; the readouts must agree with it.
    assert [
        format_frequency_hz(hz, exact=True)
        for hz in (85.0, 1000.0, 1001.0, 1234.0, 1250.0, 12_500.0, 20_000.0, 437.5)
    ] == ["85 Hz", "1 kHz", "1001 Hz", "1234 Hz", "1.25 kHz", "12.5 kHz", "20 kHz", "438 Hz"]
    assert format_frequency_hz(1234.0) == "1.23 kHz"
    assert [
        format_frequency_hz(hz, exact=True) for hz in (10_040.0, 10_050.0, 12_340.0, 19_990.0)
    ] == ["10040 Hz", "10050 Hz", "12340 Hz", "19990 Hz"]


def readout_of(model):
    layout = EQGraphRenderer.readout_layout(QFont(), model, WIDTH, HEIGHT)
    assert layout is not None
    return layout


def caption_of(model):
    layout = EQGraphRenderer.caption_layout(QFont(), model, WIDTH, HEIGHT)
    assert layout is not None
    return layout


def read_hover(model, x, y):
    model.set_hover(x, y, WIDTH, HEIGHT)
    return readout_of(model)


def test_hovering_reads_the_total_curve_without_recomputing_the_response(qapp):
    model = model_with(with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0))
    changed, hovered = QSignalSpy(model.changed), QSignalSpy(model.hoverChanged)
    points, total = model.freq_points, model.total_response_db

    text, _rect = read_hover(model, x_at(1250.0), 150)

    match = re.fullmatch(r"([\d.]+) (Hz|kHz)  ([+-]\d+\.\d) dB", text)
    assert match, text
    hz = float(match[1]) * (1000.0 if match[2] == "kHz" else 1.0)
    assert hz == pytest.approx(1250.0, rel=0.02)
    assert float(match[3]) == pytest.approx(model.response_at(hz), abs=0.06)
    assert model.response_at(hz) > 4.0
    assert hovered.count() == 1 and changed.count() == 0
    read_hover(model, x_at(1250.0), 150)
    assert hovered.count() == 1
    # A recomputed response would have replaced both lists.
    assert model.freq_points is points and model.total_response_db is total


def exact_total(model, hz):
    stage = eq_magnitude_response_v2([hz], model.bands, model.sample_rate)[0]
    return stage + input_prefilter_response_db([hz], "off", model.sample_rate)[0]


def test_the_readout_is_exact_at_a_narrow_peak(qapp):
    model = model_with(with_band(FLAT, 4, frequency_hz=1234.0, gain_db=12.0, q=10.0))

    for column in range(x_at(1100.0), x_at(1400.0)):
        text, _rect = read_hover(model, column, 150)
        hz = EQGraphGeometry.x_to_frequency(column, WIDTH, HEIGHT)
        shown = float(text.rsplit("  ", 1)[1].removesuffix(" dB"))
        assert text.startswith(format_frequency_hz(hz, exact=True) + "  ")
        assert shown == pytest.approx(exact_total(model, hz), abs=0.051), text


def test_the_readout_follows_the_correction_layer_and_the_mode(qapp):
    model = model_with(TONE, AUTO_EQ)
    column = x_at(2320.0)
    hz = EQGraphGeometry.x_to_frequency(column, WIDTH, HEIGHT)
    tone = eq_magnitude_response_v2([hz], TONE, model.sample_rate)[0]
    auto = eq_magnitude_response_v2([hz], AUTO_EQ, model.sample_rate)[0]

    assert model.response_at(hz) == pytest.approx(
        tone + auto + input_prefilter_response_db([hz], "off", model.sample_rate)[0], abs=1e-9
    )
    model.set_engine_mode("bypass")
    assert model.response_at(hz) == pytest.approx(
        input_prefilter_response_db([hz], "off", model.sample_rate)[0], abs=1e-9
    )


def test_a_flat_total_reads_zero_without_a_sign(qapp):
    model = model_with(FLAT)

    text, _rect = read_hover(model, x_at(3000.0), 150)

    assert text.endswith("  0.0 dB"), text
    assert "-0.0" not in text and "+0.0" not in text


def test_the_readout_reads_the_total_of_the_mode(qapp):
    model = model_with(FLAT, cleanup="off")

    model.set_engine_mode("raw")
    assert model.response_at(30.0) == 0.0
    model.set_engine_mode("bypass")
    assert model.response_at(30.0) < -5.0


def test_leaving_the_graph_or_the_plot_removes_the_readout(qapp):
    model = model_with(FLAT)
    hovered = QSignalSpy(model.hoverChanged)

    read_hover(model, 400, 150)
    for x, y in ((400, 5), (5, 150), (WIDTH - 2, 150), (400, HEIGHT - 3)):
        model.set_hover(400, 150, WIDTH, HEIGHT)
        model.set_hover(x, y, WIDTH, HEIGHT)
        assert model.hover is None
        assert EQGraphRenderer.readout_layout(QFont(), model, WIDTH, HEIGHT) is None
    model.set_hover(400, 150, WIDTH, HEIGHT)
    model.clear_hover()
    assert model.hover is None and hovered.count() > 2


def test_dragging_a_band_reads_that_bands_numbers_instead(qapp):
    tone = with_band(FLAT, 4, frequency_hz=1250.0, gain_db=2.3, q=1.4)
    tone = with_band(tone, 6, filter_type="high_pass", frequency_hz=75.0)
    model = model_with(tone)
    model.set_hover(400, 150, WIDTH, HEIGHT)

    far = drag_distance() + 5
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)
    model.mouse_press(x, y, WIDTH, HEIGHT)
    model.mouse_move(x, y - far, WIDTH, HEIGHT)
    model.mouse_move(x, y, WIDTH, HEIGHT)
    assert readout_of(model)[0] == "Band 5  1.25 kHz  +2.3 dB  Q 1.4"
    model.finish_drag()

    x, y = model.band_handle_position(6, WIDTH, HEIGHT)
    model.mouse_press(x, y, WIDTH, HEIGHT)
    model.mouse_move(x, y - far, WIDTH, HEIGHT)
    model.mouse_move(x, y, WIDTH, HEIGHT)
    assert readout_of(model)[0] == "Band 7  75 Hz"


@pytest.mark.parametrize(
    ("x", "y"),
    [(42, 36), (WIDTH - 12, 36), (42, HEIGHT - 22), (WIDTH - 12, HEIGHT - 22), (450, 150)],
)
def test_the_readout_stays_in_the_plot_and_off_the_pointer(qapp, x, y):
    model = model_with(FLAT)
    _text, rect = read_hover(model, x, y)
    geometry = EQGraphGeometry
    plot = QRectF(
        geometry.MARGIN_LEFT,
        geometry.MARGIN_TOP,
        *geometry.plot_size(WIDTH, HEIGHT),
    )

    assert plot.contains(rect)
    assert not rect.adjusted(-8, -8, 8, 8).contains(QPointF(x, y))


def test_painting_the_readout_adds_a_guide_and_a_label_and_nothing_else(qapp):
    model = model_with(with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0))
    plain = paint(model)
    x, y = x_at(300.0), 220
    _text, rect = read_hover(model, x, y)
    hovered = paint(model)
    box = rect.toAlignedRect().adjusted(-2, -2, 2, 2)

    centre = rect.center().toPoint()
    assert hovered.pixelColor(centre) != plain.pixelColor(centre)
    assert [
        (column, row)
        for column in range(0, WIDTH, 3)
        for row in range(0, HEIGHT, 3)
        if hovered.pixel(column, row) != plain.pixel(column, row)
        and not box.contains(column, row)
        and abs(column - x) > 4
    ] == []


def test_bypass_runs_the_input_filter_only_and_raw_monitor_runs_nothing(qapp):
    normal = model_with(TONE, AUTO_EQ, cleanup="off")
    bypass = model_with(TONE, AUTO_EQ, cleanup="off")
    raw = model_with(TONE, AUTO_EQ, cleanup="off")

    bypass.set_engine_mode("bypass")
    raw.set_engine_mode("raw")

    assert normal.engine_mode == "normal" and normal.eq_running
    assert not bypass.eq_running and not raw.eq_running
    assert bypass.total_response_db == pytest.approx(normal.input_filter_db, abs=1e-12)
    assert bypass.input_filter_db == pytest.approx(normal.input_filter_db, abs=1e-12)
    assert raw.total_response_db == [0.0] * len(raw.freq_points)
    assert raw.input_filter_db == [0.0] * len(raw.freq_points)
    # What the EQ is set to do is still shown and editable.
    assert raw.band_curves_db == normal.band_curves_db
    assert raw.response_db == pytest.approx(normal.response_db, abs=1e-12)

    raw.set_engine_mode("normal")
    assert raw.total_response_db == pytest.approx(normal.total_response_db, abs=1e-12)
    with pytest.raises(ValueError):
        raw.set_engine_mode("muted")


@pytest.mark.parametrize("cleanup", ["gentle", "strong"])
def test_bypass_draws_the_fixed_high_pass_whatever_the_cleanup_mode(qapp, cleanup):
    # Adaptive cleanup runs on the full path only; bypass applies the fixed 80 Hz high-pass.
    fixed = input_prefilter_response_db(
        model_with(TONE, AUTO_EQ).freq_points, "off", 48_000.0
    )
    cleaned = model_with(TONE, AUTO_EQ, cleanup=cleanup)
    late = model_with(TONE, AUTO_EQ)
    low = cleaned.freq_points.index(min(cleaned.freq_points, key=lambda f: abs(f - 40.0)))

    cleaned.set_engine_mode("bypass")
    late.set_engine_mode("bypass")
    late.set_input_conditioning(cleanup, late.sample_rate)

    for model in (cleaned, late):
        assert model.cleanup_mode == cleanup
        assert model.input_filter_db == pytest.approx(fixed, abs=1e-12)
        assert model.total_response_db == pytest.approx(fixed, abs=1e-12)
        assert model.total_response_db[low] < -10.0
    # Normal mode draws the cleanup's own high-pass, which starts at the same 80 Hz.
    cleaned.set_engine_mode("normal")
    assert cleaned.input_filter_db == pytest.approx(
        input_prefilter_response_db(cleaned.freq_points, cleanup, 48_000.0), abs=1e-12
    )
    assert cleaned.input_filter_db[low] < -10.0
    # Raw monitor skips the conditioning.
    cleaned.set_engine_mode("raw")
    assert cleaned.input_filter_db == [0.0] * len(cleaned.freq_points)
    assert cleaned.total_response_db == [0.0] * len(cleaned.freq_points)


def test_the_graph_says_when_the_eq_is_not_running(qapp):
    model = model_with(FLAT)
    labels, descriptions = [], []
    for mode in ("normal", "bypass", "raw", "normal"):
        model.set_engine_mode(mode)
        labels.append(model.not_running_label)
        descriptions.append(model.accessible_description())

    assert labels == ["", "EQ not running: Bypass", "EQ not running: Raw monitor", ""]
    assert "not running" not in descriptions[0] and "not running" not in descriptions[3]
    assert labels[1] in descriptions[1] and labels[2] in descriptions[2]
    # The general help also mentions the input filter, so judge only the not-running part.
    bypass = descriptions[1][descriptions[1].index(labels[1]):]
    raw = descriptions[2][descriptions[2].index(labels[2]):]
    for text in (bypass, raw):
        assert "the thin band curves show what the EQ would do" in text
    # Bypass skips only the EQ, so the input filter still shapes the thick curve.
    assert "The EQ is skipped in this mode" in bypass and "input filter are skipped" not in bypass
    assert "the thick curve is the input filter alone" in bypass and "flat" not in bypass
    # Raw monitor skips the input filter too, so its thick curve is flat.
    assert "The EQ and the input filter are skipped in this mode" in raw
    assert "the thick curve is flat" in raw and "input filter alone" not in raw


def test_a_graph_that_is_not_running_is_muted_and_captioned(qapp):
    # Two bands that nearly cancel, so the total does not hide either curve.
    tone = with_band(FLAT, 4, frequency_hz=1200.0, gain_db=6.0)
    tone = with_band(tone, 1, frequency_hz=1300.0, gain_db=-6.0)
    column = x_at(850.0)
    plot = range(EQGraphGeometry.MARGIN_TOP + 4, HEIGHT - EQGraphGeometry.MARGIN_BOTTOM)
    model = model_with(tone)
    running = paint(model)
    model.set_engine_mode("bypass")
    stopped = paint(model)
    text, rect = caption_of(model)
    box = rect.toAlignedRect()
    backdrop = stopped.pixelColor(box.left() + 2, box.top() + 2).lightness()
    inked = sum(
        abs(stopped.pixelColor(x, y).lightness() - backdrop) > 60
        for x in range(box.left() + 3, box.right() - 3)
        for y in range(box.top() + 3, box.bottom() - 3)
    )

    assert text == "EQ not running: Bypass"
    assert warmth(running, column, plot) > 150
    assert 30 < warmth(stopped, column, plot) < 150
    assert inked > 40


def test_the_caption_has_no_place_in_a_running_graph(qapp):
    assert EQGraphRenderer.caption_layout(QFont(), model_with(FLAT), WIDTH, HEIGHT) is None


def test_the_caption_keeps_off_the_handles(qapp):
    # Handles under both upper corners, where the caption would go first.
    tone = with_band(FLAT, 3, frequency_hz=15000.0, gain_db=12.0)
    tone = with_band(tone, 4, frequency_hz=40.0, gain_db=12.0)
    model = model_with(tone)
    model.set_engine_mode("raw")

    _text, rect = caption_of(model)
    radius = EQGraphGeometry.HANDLE_RADIUS
    clear = rect.adjusted(-radius, -radius, radius, radius)

    for index in range(10):
        assert not clear.contains(QPointF(*model.band_handle_position(index, WIDTH, HEIGHT)))


def test_the_description_follows_the_selected_band(qapp):
    model = model_with(with_band(FLAT, 4, frequency_hz=1250.0, gain_db=2.3, q=1.4))
    assert "Band 5 selected" not in model.accessible_description()
    model.select_band(4)

    assert "Band 5 selected: Bell, 1.25 kHz, +2.3 dB, Q 1.4, on." in model.accessible_description()
    off = model_with(with_band(FLAT, 2, filter_type="high_pass", frequency_hz=75.0, enabled=False))
    off.select_band(2)
    assert "Band 3 selected: High-pass, 75 Hz, off." in off.accessible_description()


def test_the_description_is_announced_when_a_drag_ends_and_not_during_it(qapp):
    model = model_with(TONE)
    model.select_band(4)
    before = model.property("accessibleDescription")
    announced = QSignalSpy(model.accessibleDescriptionChanged)
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)

    assert model.mouse_press(x, y, WIDTH, HEIGHT)
    for step in range(1, 12):
        assert model.mouse_move(x, y - 6 * step, WIDTH, HEIGHT)
    assert announced.count() == 0
    assert model.property("accessibleDescription") == before
    model.mouse_release(x, y - 66, WIDTH, HEIGHT)

    assert announced.count() == 1
    assert model.bands[4][2] > 4.0
    assert model.property("accessibleDescription") == model.accessible_description()
    assert f"Band 5 selected: Bell, 1.4 kHz, {model.bands[4][2]:+.1f} dB" in model.property(
        "accessibleDescription"
    )
    model.select_band(5)
    assert announced.count() == 2
    model.set_engine_mode("bypass")
    assert announced.count() == 3 and "Bypass" in model.property("accessibleDescription")


def test_the_description_names_the_keys_that_reset_a_band(qapp):
    description = model_with(FLAT).accessible_description()

    assert "Double-click a handle, or press Delete or Backspace on the selected band, to reset it." in description


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_resets_the_selected_band_through_the_same_request(qapp, key):
    model = model_with(TONE)
    requests = QSignalSpy(model.bandResetRequested)

    assert not model.key_press(key, Qt.KeyboardModifier.NoModifier)
    assert requests.count() == 0
    model.select_band(4)
    assert model.key_press(key, Qt.KeyboardModifier.NoModifier)
    assert requests.count() == 1 and requests.at(0)[0] == 4


def test_a_double_click_asks_for_the_band_under_the_pointer_only(qapp):
    model = model_with(with_band(FLAT, 4, frequency_hz=3000.0, gain_db=4.0))
    requests = QSignalSpy(model.bandResetRequested)
    x, y = model.band_handle_position(4, WIDTH, HEIGHT)

    assert not model.request_band_reset(x + 40, y, WIDTH, HEIGHT)
    assert requests.count() == 0
    assert model.request_band_reset(x + 3, y + 3, WIDTH, HEIGHT)
    assert requests.at(0)[0] == 4 and model.selected_band_index == 4
