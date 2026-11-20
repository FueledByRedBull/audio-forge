"""The input analyser on the Equalizer page: the model, the painter, the poll and the palette."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QSignalSpy

from mic_eq.ui.eq_graph import ANALYSER_DESCRIPTION, EQGraphGeometry, EQGraphModel, EQGraphRenderer
from mic_eq.ui.input_analyser import InputAnalyser
from mic_eq.ui.theme import PALETTE, SemanticPalette, build_palette, contrast_ratio, flatten
from test_quick_shell import quick_window  # noqa: F401  (a fixture)

WIDTH, HEIGHT = 900, 300
PLOT_BOTTOM = HEIGHT - EQGraphGeometry.MARGIN_BOTTOM
PLOT_HEIGHT = HEIGHT - EQGraphGeometry.MARGIN_TOP - EQGraphGeometry.MARGIN_BOTTOM
FREQUENCIES = np.array([100.0, 1000.0, 4000.0, 12000.0], dtype=np.float32)
SYSTEM_COLORS = {
    "window": "#102030", "window_text": "#f0e0d0", "gray_text": "#80ff80",
    "highlight": "#ffee00", "highlight_text": "#000011",
}


def levels(value: float) -> np.ndarray:
    return np.full(len(FREQUENCIES), value, dtype=np.float32)


def paint(model: EQGraphModel) -> QImage:
    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    EQGraphRenderer.paint(painter, model, WIDTH, HEIGHT)
    painter.end()
    return image


def pixel(image: QImage, x: int, y: int) -> tuple[int, int, int]:
    color = image.pixelColor(x, y)
    return color.red(), color.green(), color.blue()


# The model


def test_the_model_shows_a_spectrum_until_it_is_cleared(qapp) -> None:
    model = EQGraphModel()
    spy = QSignalSpy(model.spectrumChanged)
    plain_description = model.accessible_description()
    assert model.input_spectrum_db is None and ANALYSER_DESCRIPTION not in plain_description

    model.set_input_spectrum(levels(-60.0), FREQUENCIES)
    assert spy.count() == 1
    assert model.input_spectrum_db is not None and len(model.input_spectrum_db) == 4
    assert list(model.input_spectrum_hz) == list(FREQUENCIES)
    # The phrase is said once, when the analyser appears, and not again for each reading.
    assert ANALYSER_DESCRIPTION in model.accessible_description()
    announced = QSignalSpy(model.accessibleDescriptionChanged)
    model.set_input_spectrum(levels(-50.0), FREQUENCIES)
    assert spy.count() == 2 and announced.count() == 0

    model.clear_input_spectrum()
    assert spy.count() == 3
    assert model.input_spectrum_db is None and len(model.input_spectrum_hz) == 0
    assert model.accessible_description() == plain_description
    model.clear_input_spectrum()
    assert spy.count() == 3, "clearing nothing is no change"


def test_a_spectrum_needs_one_level_for_each_frequency() -> None:
    with pytest.raises(ValueError, match="one level for each band"):
        EQGraphModel().set_input_spectrum(levels(-60.0)[:3], FREQUENCIES)


def test_the_analyser_scale_puts_minus_90_at_the_floor_and_minus_18_at_the_zero_line() -> None:
    geometry = EQGraphGeometry
    floor = geometry.MARGIN_TOP + geometry.plot_size(WIDTH, HEIGHT)[1]
    assert geometry.analyser_to_y(-90.0, WIDTH, HEIGHT) == pytest.approx(floor)
    assert geometry.analyser_to_y(-120.0, WIDTH, HEIGHT) == pytest.approx(floor)
    assert geometry.analyser_to_y(-18.0, WIDTH, HEIGHT) == pytest.approx(geometry.gain_to_y(0.0, WIDTH, HEIGHT))
    # Linear in dBFS: full scale is a quarter of the way past the zero line, and nothing leaves the plot.
    middle = geometry.analyser_to_y(-54.0, WIDTH, HEIGHT)
    assert middle == pytest.approx((floor + geometry.gain_to_y(0.0, WIDTH, HEIGHT)) / 2)
    assert geometry.analyser_to_y(0.0, WIDTH, HEIGHT) == pytest.approx(floor - 0.625 * (floor - geometry.MARGIN_TOP))
    assert geometry.analyser_to_y(60.0, WIDTH, HEIGHT) == pytest.approx(geometry.MARGIN_TOP)


# The painter


def test_the_painter_draws_the_area_only_with_a_spectrum(qapp) -> None:
    model = EQGraphModel()
    without = paint(model)
    # -54 dBFS is a quarter of the plot's height from its floor.
    top = PLOT_BOTTOM - 0.25 * PLOT_HEIGHT
    inside, above, edge_x = round(PLOT_BOTTOM - 0.1 * PLOT_HEIGHT), round(top - 0.2 * PLOT_HEIGHT), 700

    model.set_input_spectrum(levels(-54.0), FREQUENCIES)
    drawn = paint(model)
    red, _green, blue = pixel(drawn, edge_x, inside)
    assert blue - red > 8 and pixel(without, edge_x, inside) != pixel(drawn, edge_x, inside)
    assert pixel(drawn, edge_x, above) == pixel(without, edge_x, above), "nothing above the spectrum"
    # The soft line on the top edge is bluer than the area under it.
    edge = max(pixel(drawn, edge_x, round(top) + step)[2] for step in (-1, 0, 1))
    assert edge > pixel(drawn, edge_x, inside)[2]

    model.clear_input_spectrum()
    assert paint(model) == without


def test_the_area_fades_toward_the_floor(qapp) -> None:
    model = EQGraphModel()
    plain = paint(model)
    model.set_input_spectrum(levels(-30.0), FREQUENCIES)
    drawn = paint(model)

    def tint(y: int) -> int:
        return pixel(drawn, 700, y)[2] - pixel(plain, 700, y)[2]

    assert tint(round(PLOT_BOTTOM - 0.35 * PLOT_HEIGHT)) > tint(round(PLOT_BOTTOM - 0.05 * PLOT_HEIGHT)) > 0


def test_a_spectrum_at_the_floor_draws_nothing(qapp) -> None:
    model = EQGraphModel()
    plain = paint(model)
    model.set_input_spectrum(levels(-120.0), FREQUENCIES)
    assert paint(model) == plain


def test_the_curves_lie_over_the_analyser(qapp) -> None:
    model = EQGraphModel()
    plain = paint(model)
    model.set_input_spectrum(levels(-10.0), FREQUENCIES)
    drawn = paint(model)
    # The total curve at 0 dB runs inside the area; it keeps its own colour, up to the glow's blend.
    y = round(EQGraphGeometry.gain_to_y(0.0, WIDTH, HEIGHT))
    assert y > PLOT_BOTTOM - 0.625 * PLOT_HEIGHT  # under the spectrum's crest
    bright = max(range(y - 1, y + 2), key=lambda row: sum(pixel(plain, 700, row)))
    assert all(abs(a - b) <= 15 for a, b in zip(pixel(drawn, 700, bright), pixel(plain, 700, bright)))


# The poll


class FakeEngine:
    def __init__(self) -> None:
        self.sequence = 1
        self.buffers: list[np.ndarray] = []
        self.counted = 0
        self.failure: Exception | None = None

    def input_spectrum_bin_count(self) -> int:
        self.counted += 1
        return len(FREQUENCIES)

    def input_spectrum_bin_frequencies(self, out) -> None:
        out[:] = FREQUENCIES

    def read_input_spectrum(self, out) -> int:
        if self.failure is not None:
            raise self.failure
        self.buffers.append(out)
        out[:] = -45.0
        return self.sequence


def test_the_poll_reuses_one_buffer_and_repaints_only_for_a_new_frame(qapp) -> None:
    model = EQGraphModel()
    analyser, engine = InputAnalyser(model), FakeEngine()
    spy = QSignalSpy(model.spectrumChanged)

    analyser.poll(engine)
    analyser.poll(engine)
    assert spy.count() == 1, "the same frame is not shown twice"
    engine.sequence = 2
    analyser.poll(engine)
    engine.sequence = 3
    analyser.poll(engine)
    assert spy.count() == 3
    assert len(engine.buffers) == 4 and all(buffer is engine.buffers[0] for buffer in engine.buffers)
    assert model.input_spectrum_db is engine.buffers[0]
    assert engine.counted == 1, "the band count is read once"
    assert list(model.input_spectrum_hz) == list(FREQUENCIES)


def test_no_frame_or_an_unreadable_engine_clears_the_spectrum(qapp) -> None:
    model = EQGraphModel()
    analyser, engine = InputAnalyser(model), FakeEngine()
    analyser.poll(engine)
    assert model.input_spectrum_db is not None

    engine.sequence = 0
    analyser.poll(engine)
    assert model.input_spectrum_db is None

    engine.sequence = 1
    analyser.poll(engine)
    assert model.input_spectrum_db is not None, "the same frame number shows again after a clear"
    engine.failure = RuntimeError("engine gone")
    analyser.poll(engine)
    assert model.input_spectrum_db is None
    analyser.poll(object())  # an engine built without the analyser
    assert model.input_spectrum_db is None


class RunningProcessor:
    """The window's real engine, told to be running and to send a fixed spectrum."""

    def __init__(self, real) -> None:
        self._real = real
        self.running = True
        self.reads = 0
        self.buffers: list[np.ndarray] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def is_running(self) -> bool:
        return self.running

    def input_spectrum_bin_count(self) -> int:
        return len(FREQUENCIES)

    def input_spectrum_bin_frequencies(self, out) -> None:
        out[:] = FREQUENCIES

    def read_input_spectrum(self, out) -> int:
        self.reads += 1
        self.buffers.append(out)
        out[:] = -45.0
        return self.reads


@pytest.fixture
def window_with_engine(quick_window):  # noqa: F811
    from PySide6.QtCore import QTimer

    for timer in quick_window.findChildren(QTimer):
        timer.stop()
    engine = RunningProcessor(quick_window.processor)
    quick_window.processor = engine
    quick_window.shell_page.setIndex(quick_window.EQUALIZER_PAGE_INDEX)
    return quick_window, engine


def test_the_window_reads_the_spectrum_every_other_meter_tick_on_the_equalizer_page(window_with_engine) -> None:
    window, engine = window_with_engine
    graph = window.eq_presentation.graph_model
    for _ in range(6):
        window._update_meters()
    assert engine.reads == 3
    assert graph.input_spectrum_db is not None
    assert all(buffer is engine.buffers[0] for buffer in engine.buffers)


def test_the_window_does_not_read_on_another_page_and_clears_what_was_shown(window_with_engine) -> None:
    window, engine = window_with_engine
    graph = window.eq_presentation.graph_model
    for _ in range(2):
        window._update_meters()
    assert graph.input_spectrum_db is not None
    reads = engine.reads

    window.shell_page.setIndex(0)
    assert graph.input_spectrum_db is None
    for _ in range(6):
        window._update_meters()
    assert engine.reads == reads and graph.input_spectrum_db is None

    window.shell_page.setIndex(window.EQUALIZER_PAGE_INDEX)
    for _ in range(2):
        window._update_meters()
    assert engine.reads > reads and graph.input_spectrum_db is not None


def test_the_window_does_not_read_when_the_stream_is_stopped_and_clears_the_spectrum_on_stop(window_with_engine) -> None:
    window, engine = window_with_engine
    graph = window.eq_presentation.graph_model
    for _ in range(2):
        window._update_meters()
    assert graph.input_spectrum_db is not None
    reads = engine.reads

    engine.running = False
    window._sync_meter_timer()
    assert graph.input_spectrum_db is None
    for _ in range(6):
        window._update_meters()
    assert engine.reads == reads and graph.input_spectrum_db is None


def test_the_window_does_not_read_while_minimized(window_with_engine) -> None:
    window, engine = window_with_engine
    window.showMinimized()
    window._update_input_analyser()
    assert engine.reads == 0
    assert window.eq_presentation.graph_model.input_spectrum_db is None


# The palette


def test_the_analyser_colours_are_blue_and_follow_a_contrast_theme() -> None:
    from PySide6.QtGui import QColor

    for token in (PALETTE.analyser_fill, PALETTE.analyser_line):
        color = QColor(token)
        assert 200 <= color.hslHue() <= 230 and color.blue() > color.red()
    # Its edge stands out from the graph's ground by the 3:1 a graphic needs.
    assert contrast_ratio(flatten(PALETTE.analyser_line, PALETTE.data_surface), PALETTE.data_surface) >= 3

    contrast = build_palette(SYSTEM_COLORS)
    assert contrast.analyser_fill == contrast.analyser_line == SYSTEM_COLORS["gray_text"]
    assert contrast.analyser_fill != SemanticPalette().analyser_fill
    assert contrast_ratio(contrast.analyser_line, contrast.data_surface) >= 3
