"""Undo and redo as a user drives them in the Qt Quick scene: real pointer, keys and QML items."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from mic_eq.config import DeviceIdentity
from mic_eq.config_parts.settings import q_from_bandwidth_octaves  # noqa: F401  (kept for parity with sibling tests)
from test_quick_shell import (  # noqa: F401  (quick_window is a fixture)
    _band_box,
    _drag_distance,
    _drag_up,
    _find,
    _graph_item,
    _handle_point,
    _reveal,
    _rebaseline,
    mic_page,
    quick_window,  # noqa: F401
    wide_page,  # noqa: F401
)

CTRL = Qt.KeyboardModifier.ControlModifier
CTRL_SHIFT = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
BAND = 4  # Pressing its handle also selects it, so the editor row shows this band.


class Scene:
    """The mic page of a window the keyboard shortcuts reach."""

    def __init__(self, window, page, graph, qapp):
        self.window, self.page, self.graph, self.qapp = window, page, graph, qapp
        self.view = window.centralWidget()
        self.history = window._configuration_history
        self.graph_model = window.eq_presentation.graph_model

    def undo(self) -> None:
        self.commit()  # the user's pause: a disabled Undo action ignores its shortcut
        QTest.keyClick(self.view, Qt.Key.Key_Z, CTRL)
        self.qapp.processEvents()

    def redo(self) -> None:
        self.commit()
        QTest.keyClick(self.view, Qt.Key.Key_Z, CTRL_SHIFT)
        self.qapp.processEvents()

    def commit(self) -> None:
        self.window._commit_pending_configuration_snapshot()

    def band(self, index: int = BAND):
        return self.window.eq_state.get_band(index)

    def drag_handle(self, index: int = BAND, steps: int = 8) -> None:
        for _ in _drag_up(
            self.view, _handle_point(self.window, self.graph, index), steps, _drag_distance() + 2
        ):
            pass
        self.qapp.processEvents()

    def box(self, name: str):
        """The selected band's value box named ``name``: its proxy and the text it shows."""
        proxy, item = _band_box(self.window, name)
        assert item is not None
        field = next(c for c in item.childItems() if c.metaObject().className().startswith("QQuickTextInput"))
        return proxy, item, field

    def shown(self, name: str) -> str:
        return self.box(name)[2].property("text")

    def fader(self, index: int = BAND):
        eq = self.window.quick_bridge.model["eq"]
        return _find(self.view.rootObject(), "Fader", eq["bands"][index]["gain"])

    def picker(self, key: str, index: int = BAND):
        eq = self.window.quick_bridge.model["eq"]
        return _find(self.view.rootObject(), "Picker", eq["bands"][index][key])


@pytest.fixture
def scene(quick_window, wide_page, qapp):  # noqa: F811
    window = quick_window
    graph = _graph_item(window.centralWidget().rootObject())
    # The debounce that merges a gesture's edits is the clock stand-in: only
    # an explicit commit, undo or redo closes a numeric edit.
    window._history_timer.setInterval(600_000)
    window.activateWindow()
    window.centralWidget().setFocus()
    qapp.processEvents()
    return Scene(window, wide_page, graph, qapp)


def test_a_graph_drag_is_one_undo_step_and_the_whole_scene_follows(scene) -> None:
    start = scene.band()
    assert scene.history.size == 1 and not scene.window._undo_action.isEnabled()

    scene.drag_handle()

    dragged = scene.band()
    assert dragged.gain_db > start.gain_db
    assert scene.history.size == 2
    assert scene.graph_model.bands[BAND][2] == pytest.approx(dragged.gain_db)
    assert scene.shown("gain") == f"{dragged.gain_db:+.1f}"

    scene.undo()
    assert scene.band() == start
    assert scene.history.size == 2 and not scene.history.can_undo
    assert scene.graph_model.bands[BAND][2] == pytest.approx(start.gain_db)
    assert scene.shown("gain") == "0"
    assert scene.shown("frequency") == f"{start.frequency_hz:.0f} Hz"
    assert scene.fader().property("value") == pytest.approx(0)
    assert "Undid" in scene.window.status_message.text

    scene.redo()
    assert scene.band() == dragged
    assert scene.graph_model.bands[BAND][2] == pytest.approx(dragged.gain_db)
    assert scene.shown("gain") == f"{dragged.gain_db:+.1f}"
    assert scene.fader().property("value") == pytest.approx(dragged.gain_db * 10)
    assert not scene.history.can_redo


def test_a_dragged_frequency_comes_back_in_the_frequency_box(scene) -> None:
    start = scene.band()
    point = _handle_point(scene.window, scene.graph, BAND)
    QTest.mousePress(scene.view, Qt.MouseButton.LeftButton, pos=point)
    for dx in range(_drag_distance() + 2, 60, 12):
        QTest.mouseMove(scene.view, point + QPoint(dx, 0), delay=5)
    QTest.mouseRelease(scene.view, Qt.MouseButton.LeftButton, pos=point + QPoint(dx, 0))
    scene.qapp.processEvents()
    moved = scene.band()
    assert moved.frequency_hz > start.frequency_hz
    assert scene.shown("frequency") == f"{moved.frequency_hz:.0f} Hz"

    scene.undo()

    assert scene.band().frequency_hz == start.frequency_hz
    assert scene.shown("frequency") == f"{start.frequency_hz:.0f} Hz"
    assert scene.graph_model.bands[BAND][1] == start.frequency_hz


def test_a_double_click_reset_can_be_undone_and_redone(scene) -> None:
    scene.drag_handle()
    edited = scene.band()
    assert edited.gain_db != 0
    size = scene.history.size

    QTest.mouseDClick(
        scene.view, Qt.MouseButton.LeftButton, pos=_handle_point(scene.window, scene.graph, BAND)
    )
    scene.qapp.processEvents()
    assert scene.band().gain_db == 0
    assert scene.history.size == size + 1
    assert scene.shown("gain") == "0"

    scene.undo()
    assert scene.band() == edited
    assert scene.graph_model.bands[BAND][2] == pytest.approx(edited.gain_db)
    assert scene.shown("gain") == f"{edited.gain_db:+.1f}"

    scene.redo()
    assert scene.band().gain_db == 0
    assert scene.graph_model.bands[BAND][2] == 0
    assert scene.shown("gain") == "0"


def test_a_fader_drag_is_one_undo_step(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    scene.qapp.processEvents()
    fader = scene.fader()
    start = scene.band()
    first = _reveal(scene.page, fader)
    press = QPoint(first.x() - 20, first.y())
    QTest.mousePress(scene.view, Qt.MouseButton.LeftButton, pos=press)
    for step in range(1, 9):
        QTest.mouseMove(scene.view, press + QPoint(step * 6, 0), delay=5)
    QTest.mouseRelease(scene.view, Qt.MouseButton.LeftButton, pos=press + QPoint(48, 0))
    scene.qapp.processEvents()
    moved = scene.band()
    assert moved.gain_db > start.gain_db
    scene.commit()
    assert scene.history.size == 2

    scene.undo()
    assert scene.band() == start
    assert fader.property("value") == pytest.approx(0)
    assert scene.shown("gain") == "0"

    scene.redo()
    assert scene.band() == moved
    assert fader.property("value") == pytest.approx(moved.gain_db * 10)


def test_each_keyboard_step_on_a_fader_undoes_by_that_step(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    fader = scene.fader()
    _reveal(scene.page, fader)
    fader.forceActiveFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(scene.view, Qt.Key.Key_Right)
    assert scene.band().gain_db == pytest.approx(0.1)
    scene.commit()
    assert scene.history.size == 2
    QTest.keyClick(scene.view, Qt.Key.Key_Right)
    assert scene.band().gain_db == pytest.approx(0.2)
    scene.commit()

    scene.undo()
    assert scene.band().gain_db == pytest.approx(0.1)
    assert fader.property("value") == pytest.approx(1)
    scene.undo()
    assert scene.band().gain_db == 0
    assert scene.shown("gain") == "0"


def test_a_typed_value_undoes_to_the_value_it_replaced(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    start = scene.band()
    _proxy, box, field = scene.box("frequency")
    QTest.mouseClick(scene.view, Qt.MouseButton.LeftButton, pos=_reveal(scene.page, box))
    QTest.keyClick(scene.view, Qt.Key.Key_A, CTRL)
    QTest.keyClicks(scene.view, "2500")
    QTest.keyClick(scene.view, Qt.Key.Key_Return)
    scene.qapp.processEvents()
    assert scene.band().frequency_hz == 2500
    scene.commit()
    assert scene.history.size == 2

    # Leave the field, as the user does before reaching for the shortcut.
    scene.graph.forceActiveFocus()
    scene.undo()

    assert scene.band().frequency_hz == start.frequency_hz
    assert field.property("text") == f"{start.frequency_hz:.0f} Hz"
    assert scene.graph_model.bands[BAND][1] == start.frequency_hz
    scene.redo()
    assert scene.band().frequency_hz == 2500
    assert field.property("text") == "2500 Hz"


def test_keyboard_and_wheel_steps_in_a_value_box_undo_one_at_a_time(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    start = scene.band()
    _proxy, box, field = scene.box("frequency")
    centre = _reveal(scene.page, box)
    QTest.mouseClick(scene.view, Qt.MouseButton.LeftButton, pos=centre)
    QTest.keyClick(scene.view, Qt.Key.Key_Up)
    scene.commit()
    QTest.wheelEvent(scene.view.quickWindow(), QPointF(centre), QPoint(0, 120))
    scene.qapp.processEvents()
    assert scene.band().frequency_hz == start.frequency_hz + 20

    scene.graph.forceActiveFocus()
    scene.undo()
    assert scene.band().frequency_hz == start.frequency_hz + 10
    assert field.property("text") == f"{start.frequency_hz + 10:.0f} Hz"
    scene.undo()
    assert scene.band().frequency_hz == start.frequency_hz


def test_a_picker_change_undoes_and_redoes(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    start = scene.band()
    proxy = scene.window.quick_bridge.model["eq"]["bands"][BAND]["type"]
    picker = scene.picker("type")
    _reveal(scene.page, picker)
    picker.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(scene.view, Qt.Key.Key_Down)
    scene.qapp.processEvents()
    changed = scene.band()
    assert changed.filter_type != start.filter_type
    assert picker.property("displayText") == proxy.items[proxy.index] != "Bell"

    scene.undo()
    assert scene.band() == start
    assert picker.property("displayText") == "Bell"
    assert scene.graph_model.bands[BAND][0] == start.filter_type

    scene.redo()
    assert scene.band() == changed
    assert picker.property("displayText") == proxy.items[proxy.index]
    assert scene.graph_model.bands[BAND][0] == changed.filter_type


def test_a_processing_mode_change_undoes_and_redoes(scene) -> None:
    window = scene.window
    mode = window.quick_bridge.model["top"]["mode"]
    picker = _find(scene.view.rootObject(), "Picker", mode)
    assert picker is not None
    assert picker.property("displayText") == "Normal" and not window.processor.is_bypass()
    picker.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(scene.view, Qt.Key.Key_Down)
    scene.qapp.processEvents()
    assert window.processor.is_bypass() and picker.property("displayText") == "Bypass"

    scene.undo()
    assert not window.processor.is_bypass()
    assert window.processing_mode_choice.currentData() == "normal"
    assert picker.property("displayText") == "Normal"

    scene.redo()
    assert window.processor.is_bypass() and picker.property("displayText") == "Bypass"


def test_a_new_edit_after_undo_drops_the_redo_branch(scene) -> None:
    scene.drag_handle(BAND)
    first = scene.band()
    scene.undo()
    assert scene.history.can_redo and scene.window.redo_button_action.isEnabled()

    scene.drag_handle(BAND, steps=3)
    second = scene.band()
    assert second != first and second.gain_db > 0

    assert not scene.history.can_redo
    assert not scene.window.redo_button_action.isEnabled()
    scene.redo()  # nothing to redo: the shortcut leaves the new edit alone
    assert scene.band() == second


def test_the_undo_and_redo_buttons_in_the_scene_follow_the_history(scene) -> None:
    presets = scene.window.quick_bridge.model["presets"]
    assert not presets["redo"].enabled and not presets["undo"].enabled
    scene.drag_handle()
    assert presets["undo"].enabled and not presets["redo"].enabled
    scene.undo()
    assert not presets["undo"].enabled and presets["redo"].enabled
    scene.redo()
    assert presets["undo"].enabled and not presets["redo"].enabled


@pytest.fixture
def routed(scene, monkeypatch):
    """The scene with two inputs and two outputs, the first of each chosen and in the history."""
    window = scene.window
    monkeypatch.setattr(window, "_apply_input_preferences_for_current_route", lambda: None)
    for name in ("Microphone", "Headset"):
        window.input_choice.addItem(name, DeviceIdentity(name=name, direction="input"))
    for name in ("Speakers", "Cable"):
        window.output_choice.addItem(name, DeviceIdentity(name=name, direction="output"))
    _rebaseline(window)
    return scene


def _change_the_route(window) -> tuple:
    top = window.quick_bridge.model["top"]
    top["input"].setIndex(1)
    top["output"].setIndex(1)
    for row in window.quick_bridge.model["settings"]["input"]:
        row["proxy"].setIndex(1)
    top["mute"].click()
    return (
        window.input_choice.currentIndex(),
        window.output_choice.currentIndex(),
        window.input_channel_mode_choice.currentIndex(),
        window.input_cleanup_mode_choice.currentIndex(),
        window.user_muted,
    )


def test_undo_and_redo_leave_the_chosen_route_alone(routed) -> None:
    scene = routed
    route = _change_the_route(scene.window)
    assert route == (1, 1, 1, 1, True)
    _rebaseline(scene.window)  # the route is part of the history's starting point
    scene.drag_handle()

    scene.undo()
    assert scene.band().gain_db == 0
    scene.redo()
    assert scene.band().gain_db > 0

    assert (
        scene.window.input_choice.currentIndex(),
        scene.window.output_choice.currentIndex(),
        scene.window.input_channel_mode_choice.currentIndex(),
        scene.window.input_cleanup_mode_choice.currentIndex(),
        scene.window.user_muted,
    ) == route


def test_muting_is_not_an_undo_step(routed) -> None:
    scene = routed
    size = scene.history.size
    scene.window.quick_bridge.model["top"]["mute"].click()
    scene.commit()
    assert scene.window.user_muted
    assert scene.history.size == size and not scene.window._undo_action.isEnabled()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A device or input-mode change alters the calibration context in the history key, so "
        "every Ctrl+Z first records a phantom 'Manual processing edit' and 'undoes' that; the "
        "real edit is never reached however often it is pressed. The commit lives in "
        "MainWindow._commit_pending_configuration_snapshot, shared with the classic view."
    ),
)
def test_a_route_change_does_not_cost_the_next_undo_press(routed) -> None:
    scene = routed
    scene.drag_handle()
    start_size = scene.history.size
    _change_the_route(scene.window)
    assert scene.history.size == start_size

    scene.undo()

    assert scene.band().gain_db == 0
    assert scene.history.size == start_size


def test_ctrl_z_in_a_value_box_undoes_the_typing_not_the_configuration(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    scene.drag_handle()  # a configuration edit that an app-level undo would take back
    edited = scene.band()
    shown = scene.shown("frequency")
    _proxy, box, field = scene.box("frequency")
    QTest.mouseClick(scene.view, Qt.MouseButton.LeftButton, pos=_reveal(scene.page, box))
    QTest.keyClick(scene.view, Qt.Key.Key_A, CTRL)
    QTest.keyClicks(scene.view, "777")
    assert field.property("text") == "777"

    scene.undo()

    assert field.property("text") == shown
    assert scene.band() == edited
    assert scene.history.can_undo


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A focused value box takes Ctrl+Z for its own text history even when it has nothing to "
        "undo, so the reflex 'type a value, Enter, Ctrl+Z' does nothing. Letting the shortcut "
        "through when the box has no typing to undo is a product decision, and the classic "
        "spin boxes behave the same way."
    ),
)
def test_ctrl_z_right_after_entering_a_value_takes_it_back(scene) -> None:
    scene.window.eq_state.select_band(BAND)
    start = scene.band()
    _proxy, box, _field = scene.box("frequency")
    QTest.mouseClick(scene.view, Qt.MouseButton.LeftButton, pos=_reveal(scene.page, box))
    QTest.keyClick(scene.view, Qt.Key.Key_A, CTRL)
    QTest.keyClicks(scene.view, "2500")
    QTest.keyClick(scene.view, Qt.Key.Key_Return)
    scene.commit()
    assert scene.band().frequency_hz == 2500

    QTest.keyClick(scene.view, Qt.Key.Key_Z, CTRL)  # focus is still in the box
    scene.qapp.processEvents()

    assert scene.band().frequency_hz == start.frequency_hz
