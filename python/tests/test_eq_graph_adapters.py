"""Shared EQ graph model behavior across QWidget and Qt Quick adapters."""

from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QSizeF, Qt
from PySide6.QtGui import (
    QAccessible,
    QGuiApplication,
    QHoverEvent,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
)
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget
from mic_eq import AudioProcessor, input_prefilter_response_db
from mic_eq.config import EQSettings
from mic_eq.ui.eq_curve import EQCurveWidget
from mic_eq.ui.eq_graph import EQGraphModel
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


def test_graph_draws_the_fixed_high_pass_in_bypass_whatever_the_cleanup_mode(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    model = presentation.graph_model
    processor.set_input_cleanup_mode("gentle")
    processor.set_bypass(True)

    presentation.sync_input_conditioning()

    assert (model.engine_mode, model.cleanup_mode) == ("bypass", "gentle")
    assert model.input_filter_db == pytest.approx(
        input_prefilter_response_db(model.freq_points, "off", model.sample_rate)
    )
    processor.set_bypass(False)
    presentation.sync_input_conditioning()
    assert model.input_filter_db == pytest.approx(
        input_prefilter_response_db(model.freq_points, "gentle", model.sample_rate)
    )

    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def _classic_graph(qapp):
    processor = AudioProcessor()
    state = EQState(processor)
    presentation = EQPresentation(state)
    curve_widget = presentation.curve_widget
    curve_widget.resize(640, 260)
    curve_widget.show()
    qapp.processEvents()
    return processor, state, presentation, curve_widget


def _close_graph(qapp, processor, presentation):
    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


def test_classic_graph_hover_reads_out_and_leaving_removes_it(qapp):
    processor, _state, presentation, curve_widget = _classic_graph(qapp)
    model = presentation.graph_model
    changed = QSignalSpy(model.changed)
    points, total = model.freq_points, model.total_response_db
    other = QLineEdit()
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.addWidget(other)
    layout.addWidget(curve_widget)
    window.resize(700, 450)
    window.show()
    QTest.qWaitForWindowActive(window)
    other.setFocus()

    QTest.mouseMove(curve_widget, QPoint(300, 120))
    assert model.hover == (300, 120)
    QTest.mouseMove(curve_widget, QPoint(310, 125))
    assert model.hover == (310, 125)
    QApplication.sendEvent(curve_widget, QEvent(QEvent.Type.Leave))

    assert model.hover is None
    assert changed.count() == 0
    assert model.freq_points is points and model.total_response_db is total
    assert other.hasFocus() and not curve_widget.hasFocus()
    window.deleteLater()
    _close_graph(qapp, processor, presentation)


def test_both_views_describe_the_graph_at_the_end_of_a_drag_and_not_during_it(qapp):
    # A bare model: nothing but the graph itself refreshes the description.
    model = EQGraphModel()
    model.set_all_params(
        [("bell", 1000.0 * (i + 1), 0.0, 1.0, 12, True) for i in range(10)]
    )
    curve_widget = EQCurveWidget(graph_model=model)
    curve_widget.resize(640, 260)
    announced = QSignalSpy(model.accessibleDescriptionChanged)

    assert "selected:" not in curve_widget.accessibleDescription()
    model.select_band(4)
    assert announced.count() == 1
    assert "Band 5 selected: Bell, 5 kHz, +0.0 dB" in curve_widget.accessibleDescription()
    model.select_band(6)
    announced_before = announced.count()
    during = curve_widget.accessibleDescription()
    assert model.property("accessibleDescription") == during

    x, y = curve_widget.band_handle_position(6)
    QTest.mousePress(curve_widget, Qt.MouseButton.LeftButton, pos=QPoint(round(x), round(y)))
    for step in range(1, 12):
        QTest.mouseMove(curve_widget, QPoint(round(x), round(y) - step * 6), delay=5)
        assert curve_widget.accessibleDescription() == during
        assert model.property("accessibleDescription") == during
    assert announced.count() == announced_before
    QTest.mouseRelease(curve_widget, Qt.MouseButton.LeftButton, pos=QPoint(round(x), round(y) - 66))

    gain = model.bands[6][2]
    assert gain > 1.0
    assert announced.count() == announced_before + 1
    assert f"Band 7 selected: Bell, 7 kHz, {gain:+.1f} dB" in curve_widget.accessibleDescription()
    assert model.property("accessibleDescription") == curve_widget.accessibleDescription()


def test_classic_graph_double_click_resets_only_a_band_under_the_pointer(qapp):
    processor, _state, presentation, curve_widget = _classic_graph(qapp)
    requests = QSignalSpy(presentation.graph_model.bandResetRequested)
    x, y = curve_widget.band_handle_position(4)
    handle = QPoint(round(x), round(y))

    QTest.mouseClick(curve_widget, Qt.MouseButton.LeftButton, pos=handle)
    assert requests.count() == 0
    QTest.mouseDClick(curve_widget, Qt.MouseButton.LeftButton, pos=QPoint(round(x), 60))
    assert requests.count() == 0
    QTest.mouseDClick(curve_widget, Qt.MouseButton.LeftButton, pos=handle)

    assert requests.count() == 1 and requests.at(0)[0] == 4
    _close_graph(qapp, processor, presentation)


def test_double_clicking_a_handle_resets_that_band_as_one_graph_edit(qapp):
    processor, state, presentation, curve_widget = _classic_graph(qapp)
    state.set_band_value(3, "gain_db", 2.0, edited=False)
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.set_band_value(4, "q", 3.2, edited=False)
    state.set_band_value(4, "frequency_hz", 1500.0, edited=False)
    state.flush()
    x, y = curve_widget.band_handle_position(4)
    started = QSignalSpy(state.configurationEditStarted)
    finished = QSignalSpy(state.configurationEditFinished)

    QTest.mouseDClick(
        curve_widget, Qt.MouseButton.LeftButton, pos=QPoint(round(x), round(y))
    )
    qapp.processEvents()

    assert state.get_band(4) == EQSettings().bands[4]
    assert state.get_band(3).gain_db == 2.0
    assert processor.get_eq_band_config(4) == ("bell", 1280.0, 0.0, 1.41, 12, True)
    labels = [finished.at(i)[0] for i in range(finished.count())]
    # The press and release before the double-click moved nothing, so they record nothing.
    assert labels[-1] == "Reset EQ band" and labels.count("Reset EQ band") == 1
    assert labels == ["Reset EQ band"]
    assert started.count() == finished.count()
    _close_graph(qapp, processor, presentation)


def _hover_event(x: float, y: float) -> QHoverEvent:
    point = QPointF(x, y)
    return QHoverEvent(QEvent.Type.HoverMove, point, point, point)


def test_quick_graph_hover_reads_out_and_leaving_removes_it(qapp):
    model = EQGraphModel()
    item = EQGraphItem()
    item.setSize(QSizeF(640, 260))
    item.setProperty("graph", model)

    assert item.acceptHoverEvents()
    item.hoverMoveEvent(_hover_event(300, 120))
    assert model.hover == (300, 120)
    item.hoverLeaveEvent(_hover_event(-1, -1))

    assert model.hover is None


def test_quick_graph_double_click_asks_for_a_reset_of_the_handle_under_it(qapp):
    model = EQGraphModel()
    item = EQGraphItem()
    item.setSize(QSizeF(640, 260))
    item.setProperty("graph", model)
    requests = QSignalSpy(model.bandResetRequested)
    x, y = model.band_handle_position(2, 640, 260)

    def double_click(px, py):
        point = QPointF(px, py)
        event = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            point,
            point,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        item.mouseDoubleClickEvent(event)
        return event.isAccepted()

    assert not double_click(x, 60)
    assert requests.count() == 0
    assert double_click(x, y)
    assert requests.count() == 1 and requests.at(0)[0] == 2


def _find_graph_item(item):
    for child in item.childItems():
        if isinstance(child, EQGraphItem):
            return child
        found = _find_graph_item(child)
        if found is not None:
            return found
    return None


def test_the_real_scene_delivers_hover_and_double_click_to_the_graph(
    isolated_main_window, qapp
):
    window = isolated_main_window
    window.resize(1400, 1100)
    window.show()
    qapp.processEvents()
    view = window.centralWidget()
    item = _find_graph_item(view.rootObject())
    assert item is not None
    model = window.eq_presentation.graph_model
    window.eq_state.set_band_value(4, "gain_db", 3.0, edited=False)
    window.eq_state.flush()
    x, y = model.band_handle_position(4, item.width(), item.height())
    handle = item.mapToScene(QPointF(x, y)).toPoint()
    assert view.rect().contains(handle)

    QTest.mouseMove(view, handle + QPoint(0, 40))
    qapp.processEvents()
    assert model.hover is not None
    assert not item.hasActiveFocus()
    QTest.mouseMove(view, QPoint(2, 2))
    qapp.processEvents()
    assert model.hover is None

    QTest.mouseDClick(view, Qt.MouseButton.LeftButton, pos=handle)
    qapp.processEvents()
    assert window.eq_state.get_band(4) == EQSettings().bands[4]
    # Qt Quick sends the second press of a double-click as a press first.
    assert model.dragging_band_index is None
    QTest.mouseMove(view, QPoint(10, 10))
    qapp.processEvents()
    assert window.eq_state.get_band(4) == EQSettings().bands[4]


def test_a_reset_after_the_press_of_a_double_click_survives_the_release(qapp):
    processor, state, presentation, _widget = _classic_graph(qapp)
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.flush()
    model = presentation.graph_model
    x, y = model.band_handle_position(4, 640, 260)
    finished = QSignalSpy(state.configurationEditFinished)

    # Qt Quick: the press part of the double-click, then the double-click, then the release.
    assert model.mouse_press(x, y, 640, 260)
    assert model.request_band_reset(x, y, 640, 260)
    model.mouse_release(x, y, 640, 260)

    assert model.dragging_band_index is None
    assert state.get_band(4) == EQSettings().bands[4]
    assert [finished.at(i)[0] for i in range(finished.count())] == ["Reset EQ band"]

    # A drag that did start is cancelled by the double-click that follows.
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.flush()
    far = QGuiApplication.styleHints().startDragDistance() + 5
    assert model.mouse_press(x, y, 640, 260)
    assert model.mouse_move(x, y - far, 640, 260)
    assert model.request_band_reset(x, y - far, 640, 260)
    model.mouse_release(x, y - far, 640, 260)
    assert [finished.at(i)[0] for i in range(finished.count())][1:] == [
        "Cancelled EQ graph edit",
        "Reset EQ band",
    ]
    assert state.get_band(4) == EQSettings().bands[4]
    _close_graph(qapp, processor, presentation)


def test_a_jittered_click_before_a_double_click_records_only_the_reset(qapp):
    processor, state, presentation, curve_widget = _classic_graph(qapp)
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.flush()
    x, y = curve_widget.band_handle_position(4)
    handle = QPoint(round(x), round(y))
    finished = QSignalSpy(state.configurationEditFinished)

    QTest.mousePress(curve_widget, Qt.MouseButton.LeftButton, pos=handle)
    QTest.mouseMove(curve_widget, handle + QPoint(1, 0))
    QTest.mouseRelease(curve_widget, Qt.MouseButton.LeftButton, pos=handle + QPoint(1, 0))
    assert state.get_band(4).gain_db == 3.0 and state.get_band(4).frequency_hz == 1280.0
    QTest.mouseDClick(curve_widget, Qt.MouseButton.LeftButton, pos=handle + QPoint(1, 0))
    qapp.processEvents()

    assert [finished.at(i)[0] for i in range(finished.count())] == ["Reset EQ band"]
    assert state.get_band(4) == EQSettings().bands[4]
    _close_graph(qapp, processor, presentation)


def test_a_double_click_in_the_real_scene_is_one_undo_step_after_a_jittered_press(
    isolated_main_window, qapp
):
    window = isolated_main_window
    window.resize(1400, 1100)
    window.show()
    qapp.processEvents()
    view = window.centralWidget()
    item = _find_graph_item(view.rootObject())
    assert item is not None
    state = window.eq_state
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.flush()
    window._commit_pending_configuration_snapshot()
    # The committed change moves the page layout; point at the handle where it settles.
    QTest.qWait(100)
    history = window._configuration_history
    before, original = history.size, state.get_band(4)
    model = window.eq_presentation.graph_model
    x, y = model.band_handle_position(4, item.width(), item.height())
    handle = item.mapToScene(QPointF(x, y)).toPoint()
    jittered = handle + QPoint(1, 0)

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=handle)
    QTest.mouseMove(view, jittered)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=jittered)
    qapp.processEvents()
    assert state.get_band(4) == original
    QTest.mouseDClick(view, Qt.MouseButton.LeftButton, pos=jittered)
    qapp.processEvents()

    assert state.get_band(4) == EQSettings().bands[4]
    assert history.size == before + 1
    window.undo_configuration()
    assert state.get_band(4) == original


def test_delete_resets_the_selected_band_in_both_views(qapp):
    processor, state, presentation, curve_widget = _classic_graph(qapp)
    model = presentation.graph_model
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.flush()
    finished = QSignalSpy(state.configurationEditFinished)
    labels = lambda: [finished.at(i)[0] for i in range(finished.count())]  # noqa: E731

    QTest.keyClick(curve_widget, Qt.Key.Key_Delete)
    assert labels() == [] and state.get_band(4).gain_db == 3.0
    model.select_band(4)
    QTest.keyClick(curve_widget, Qt.Key.Key_Delete)
    assert state.get_band(4) == EQSettings().bands[4]
    assert labels() == ["Reset EQ band"]

    item = EQGraphItem()
    item.setProperty("graph", model)
    state.set_band_value(4, "gain_db", 2.0, edited=False)
    state.flush()
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier
    )
    item.keyPressEvent(event)
    assert event.isAccepted()
    assert state.get_band(4) == EQSettings().bands[4]
    assert labels() == ["Reset EQ band", "Reset EQ band"]
    _close_graph(qapp, processor, presentation)


def test_the_real_scene_describes_the_graph_after_a_drag_and_not_during_it(
    isolated_main_window, qapp
):
    window = isolated_main_window
    window.resize(1400, 1100)
    window.show()
    qapp.processEvents()
    view = window.centralWidget()
    item = _find_graph_item(view.rootObject())
    assert item is not None
    model = window.eq_presentation.graph_model
    announced = QSignalSpy(model.accessibleDescriptionChanged)
    QAccessible.setActive(True)
    try:
        _drag_and_describe(view, item, model, announced, qapp)
    finally:
        QAccessible.setActive(False)


def _drag_and_describe(view, item, model, announced, qapp):
    def described() -> str:
        return QAccessible.queryAccessibleInterface(item).text(QAccessible.Text.Description)

    x, y = model.band_handle_position(4, item.width(), item.height())
    handle = item.mapToScene(QPointF(x, y)).toPoint()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=handle)
    qapp.processEvents()
    before, count = described(), announced.count()
    assert "Band 5 selected" in before
    for step in range(1, 12):
        QTest.mouseMove(view, handle + QPoint(0, -step * 4), delay=5)
        assert described() == before
    assert model.dragging_band_index == 4 and announced.count() == count
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=handle + QPoint(0, -44))
    qapp.processEvents()

    assert announced.count() == count + 1
    assert f"{model.bands[4][2]:+.1f} dB" in described()
    assert described() != before
