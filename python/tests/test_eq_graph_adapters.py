"""Shared EQ graph model behavior across QWidget and Qt Quick adapters."""

from PySide6.QtCore import QPoint, QSizeF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QSignalSpy, QTest
from mic_eq import AudioProcessor
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
