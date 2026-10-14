"""Shared EQ graph model behavior through the Qt Quick adapter."""

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QSizeF, Qt
from PySide6.QtGui import (
    QAccessible,
    QGuiApplication,
    QHoverEvent,
    QKeyEvent,
    QMouseEvent,
)
from PySide6.QtTest import QSignalSpy, QTest
from mic_eq import AudioProcessor, input_prefilter_response_db
from mic_eq.config import EQSettings
from mic_eq.ui.eq_graph import EQGraphModel
from mic_eq.ui.eq_presentation import EQPresentation
from mic_eq.ui.eq_quick_graph import EQGraphItem
from mic_eq.ui.eq_state import EQState


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


def _graph():
    processor = AudioProcessor()
    state = EQState(processor)
    return processor, state, EQPresentation(state)


def _close_graph(qapp, processor, presentation):
    presentation.deleteLater()
    qapp.processEvents()
    processor.stop()


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
    processor, state, presentation = _graph()
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


def test_backspace_resets_the_selected_band_in_the_quick_graph(qapp):
    processor, state, presentation = _graph()
    model = presentation.graph_model
    state.set_band_value(4, "gain_db", 2.0, edited=False)
    state.flush()
    model.select_band(4)
    finished = QSignalSpy(state.configurationEditFinished)

    item = EQGraphItem()
    item.setProperty("graph", model)
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier
    )
    item.keyPressEvent(event)
    assert event.isAccepted()
    assert state.get_band(4) == EQSettings().bands[4]
    assert [finished.at(i)[0] for i in range(finished.count())] == ["Reset EQ band"]
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
