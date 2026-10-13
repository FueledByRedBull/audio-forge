"""Shared EQ graph model behavior across QWidget and Qt Quick adapters."""

from dataclasses import replace

import pytest
from PySide6.QtCore import QPoint, QSizeF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QSignalSpy, QTest
from mic_eq import AudioProcessor, input_prefilter_response_db
from mic_eq.ui.eq_curve import EQCurveWidget
from mic_eq.ui.eq_panel import EQPresentation
from mic_eq.ui.eq_quick_graph import EQGraphItem
from mic_eq.ui.eq_state import EQState


def test_eq_presentation_does_not_create_the_classic_graph_until_requested(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)

    assert presentation._curve_widget is None
    assert not presentation.findChildren(EQCurveWidget)

    graph_model = presentation.graph_model
    curve_widget = presentation.curve_widget
    assert curve_widget.graph_model is graph_model
    assert presentation._curve_widget is curve_widget

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_classic_and_quick_graph_adapters_paint_the_same_model(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    model = presentation.graph_model
    curve_widget = presentation.curve_widget
    quick_item = EQGraphItem()
    quick_item.setProperty("graph", model)

    size = (640, 260)
    curve_widget.resize(*size)
    quick_item.setSize(QSizeF(*size))
    model.set_band_markers((80.0, 1000.0, 8000.0))

    classic_image = QImage(
        *size, QImage.Format.Format_ARGB32_Premultiplied
    )
    classic_image.fill(Qt.GlobalColor.transparent)
    curve_widget.render(classic_image)

    quick_image = QImage(
        *size, QImage.Format.Format_ARGB32_Premultiplied
    )
    quick_image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(quick_image)
    quick_item.paint(painter)
    painter.end()

    assert quick_item.graph is model
    # QWidget fills the four unpainted rounded corners with its parent
    # palette; Qt Quick composites those corners over the QML card instead.
    for y in range(6, size[1] - 6):
        for x in range(6, size[0] - 6):
            assert quick_image.pixel(x, y) == classic_image.pixel(x, y)

    quick_item.deleteLater()
    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_classic_graph_escape_cancels_the_shared_model_edit(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    curve_widget = presentation.curve_widget
    curve_widget.resize(640, 260)
    curve_widget.show()
    qapp.processEvents()

    model = presentation.graph_model
    band_index = 4
    original = model.bands[band_index]
    cancelled = QSignalSpy(model.bandDragCancelled)
    finished = QSignalSpy(state.configurationEditFinished)
    start = curve_widget.band_handle_position(band_index)
    target = QPoint(
        round(curve_widget.frequency_to_x(2200.0)),
        round(curve_widget.gain_to_y(4.0)),
    )

    QTest.mousePress(
        curve_widget,
        Qt.MouseButton.LeftButton,
        pos=QPoint(round(start[0]), round(start[1])),
    )
    QTest.mouseMove(curve_widget, target, delay=5)
    QTest.keyClick(curve_widget, Qt.Key.Key_Escape)
    qapp.processEvents()

    assert model.bands[band_index] == original
    assert cancelled.count() == 1
    assert finished.count() == 1
    assert finished.at(0)[0] == "Cancelled EQ graph edit"

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_classic_graph_click_off_centre_leaves_the_band_alone(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    curve_widget = presentation.curve_widget
    curve_widget.resize(640, 260)
    curve_widget.show()
    qapp.processEvents()

    band_index = 4
    # Off the graph's whole-Hz, 0.1 dB grid, so snapping to it would show.
    state.set_band(
        band_index,
        replace(state.get_band(band_index), frequency_hz=437.5, gain_db=2.35),
        edited=False,
    )
    state.flush()
    model = presentation.graph_model
    original = model.bands[band_index]
    assert original[1:3] == (437.5, 2.35)
    edited = QSignalSpy(state.configurationEdited)
    x, y = curve_widget.band_handle_position(band_index)

    QTest.mouseClick(
        curve_widget,
        Qt.MouseButton.LeftButton,
        pos=QPoint(round(x) + 5, round(y) + 5),
    )
    qapp.processEvents()

    assert model.bands[band_index] == original
    assert state.get_band(band_index).frequency_hz == 437.5
    assert state.get_band(band_index).gain_db == 2.35
    assert edited.count() == 0

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_graph_total_follows_the_state_layers_and_skips_bands_that_are_off(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    model = presentation.graph_model

    state.apply_auto_eq_results(
        [
            (72.0, -2.0, 0.75),
            (145.0, -1.0, 1.0),
            (290.0, 0.4, 1.1),
            (580.0, 1.1, 1.2),
            (1160.0, 1.8, 1.4),
            (2320.0, 2.2, 1.6),
            (4640.0, 1.2, 1.5),
            (7600.0, -0.8, 1.4),
            (11100.0, -1.1, 1.1),
            (15100.0, -0.6, 0.8),
        ]
    )
    state.set_band_value(4, "gain_db", 5.0, edited=False)
    state.set_band_value(4, "enabled", False, edited=False)

    # The tone layer is flat but for one band that is off, so only the
    # Auto-EQ layer may reach the total.
    assert model.correction_response_db
    assert model.band_curves_db[4] == []
    assert model.response_db == pytest.approx(model.correction_response_db, abs=1e-9)

    state.set_band_value(4, "enabled", True, edited=False)
    assert model.band_curves_db[4]
    assert model.response_db != pytest.approx(model.correction_response_db, abs=0.5)

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_graph_follows_the_processors_cleanup_mode_and_the_eq_switch(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    model = presentation.graph_model
    state.set_band_value(4, "gain_db", 5.0, edited=False)

    assert model.cleanup_mode == "off"
    assert model.sample_rate == float(processor.sample_rate())
    assert model.total_response_db != pytest.approx(model.input_filter_db, abs=0.5)

    # Changed behind the graph's back, as the window does it: found on the next tick.
    processor.set_input_cleanup_mode("strong")
    QTest.qWait(450)
    assert model.cleanup_mode == "strong"
    assert model.input_filter_db == pytest.approx(
        input_prefilter_response_db(model.freq_points, "strong", model.sample_rate)
    )

    # With the EQ switch off the signal gets the input filter and nothing else.
    state.set_enabled(False)
    assert not model.eq_enabled
    assert model.total_response_db == pytest.approx(model.input_filter_db, abs=1e-12)
    state.set_enabled(True)
    assert model.eq_enabled
    assert model.total_response_db != pytest.approx(model.input_filter_db, abs=0.5)

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()
