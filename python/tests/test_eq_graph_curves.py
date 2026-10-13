"""What the EQ graph draws: each band, the Auto-EQ layer, the input filter and their total."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter

from mic_eq import AudioProcessor, eq_magnitude_response_v2, input_prefilter_response_db
from mic_eq.ui.eq_graph import OFF_BAND, EQGraphGeometry, EQGraphModel, EQGraphRenderer
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
    low = off.freq_points.index(min(off.freq_points, key=lambda f: abs(f - 20.0)))
    # The 80 Hz high-pass is only in the static filter when cleanup is off.
    assert off.input_filter_db[low] < gentle.input_filter_db[low] - 15.0
    # Neither the EQ-only response nor the bands change with the mode.
    assert off.response_db == pytest.approx(gentle.response_db, abs=1e-12)
    assert off.band_curves_db == gentle.band_curves_db
    assert off.total_response_db[low] < gentle.total_response_db[low] - 15.0

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
