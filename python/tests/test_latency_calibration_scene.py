"""The latency calibration dialog is a Qt Quick scene: it loads, names its controls and takes the keyboard."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import QPointF, Qt, qInstallMessageHandler
from PySide6.QtGui import QAccessible, QFont
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QWidget

from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog, LatencyCalibrationLoadError
from mic_eq.ui.latency_calibration_view import NO_READINGS, LatencyCalibrationView
from test_quick_shell import _AccessibleScene

PROFILE = {
    "capture_format_context": [48_000, 1, 48_000, 2],
    "measured_round_trip_ms": 42.0,
    "total_latency_ms": 53.5,
}
BUTTONS = ["closeButton", "resetButton", "runButton", "acceptButton"]


@pytest.fixture(autouse=True)
def no_qml_messages():
    """Fail a test that makes the scene print a warning or error."""
    messages: list[str] = []
    previous = qInstallMessageHandler(
        lambda _mode, _context, message: messages.append(message) if ".qml" in message else None
    )
    yield messages
    qInstallMessageHandler(previous)
    assert messages == [], "the scene printed: " + "; ".join(messages[:3])


@pytest.fixture
def dialog(qapp):
    made: list[LatencyCalibrationDialog] = []

    def make(profile: dict | None = None) -> LatencyCalibrationDialog:
        dialog = LatencyCalibrationDialog(None, existing_profile=profile)
        made.append(dialog)
        dialog.show()
        qapp.processEvents()
        dialog.activateWindow()
        dialog.quick_view.setFocus()
        qapp.processEvents()
        return dialog

    yield make
    for dialog in made:
        dialog.close()
        dialog.deleteLater()
    qapp.processEvents()


def _all_items(root: Any, *, only_visible: bool = False):
    """The scene's items by their visual parents: a Repeater's items are not QObject children."""
    for child in root.childItems():
        if child.isVisible() or not only_visible:
            yield child
            yield from _all_items(child, only_visible=only_visible)


def _item(dialog: Any, name: str) -> Any:
    item = next((i for i in _all_items(dialog.quick_view.rootObject()) if i.objectName() == name), None)
    assert item is not None, name
    return item


def _items_named(dialog: Any, name: str) -> list:
    return [i for i in _all_items(dialog.quick_view.rootObject()) if i.objectName() == name]


def _focus_item_name(dialog: Any) -> str:
    return dialog.quick_view.quickWindow().activeFocusItem().objectName()


def _within_page(dialog: Any, item: Any) -> bool:
    scroll = _item(dialog, "contentScroll")
    top = item.mapToItem(scroll, QPointF(0, 0)).y()
    return top >= 0 and top + item.height() <= scroll.height()


def test_the_scene_loads_and_every_control_has_its_accessible_name(dialog) -> None:
    made = dialog(PROFILE)
    assert made.quick_view.status() == QQuickWidget.Status.Ready and made.quick_view.errors() == []
    with _AccessibleScene() as scene:
        names = {
            "progressBar": "Latency calibration progress",
            "levelMeter": "Latency calibration capture level",
            "acceptButton": "Accept latency calibration",
            "resetButton": "Reset latency calibration",
            "closeButton": "Close latency calibration",
            "runButton": "Run calibration",
            "detailsButton": "Show details",
        }
        assert {key: scene.name(_item(made, key)) for key in names} == names
        interface = QAccessible.queryAccessibleInterface(made.quick_view)
        assert interface is not None and interface.text(QAccessible.Text.Name) == "Latency calibration content"

        made.view_model.set_status("Moving closer helps")
        assert scene.name(_item(made, "statusText")) == "Moving closer helps"
        total = _item(made, "totalReading")
        made.view_model.set_readings({**NO_READINGS, "total": "12.00 ms"})
        interfaces = [QAccessible.queryAccessibleInterface(child) for child in total.childItems()]
        assert "Total latency: 12.00 ms" in [
            i.text(QAccessible.Text.Name) for i in interfaces if i is not None
        ]
        siblings = QAccessible.queryAccessibleInterface(total.childItems()[1]).parent()
        heard = [siblings.child(i).text(QAccessible.Text.Name) for i in range(siblings.childCount())]
        assert "Total latency: 12.00 ms" in heard and "Total latency" not in heard, "a name is read once"


def test_tab_visits_close_reset_run_then_accept_left_to_right(dialog, qapp) -> None:
    made = dialog(PROFILE)
    view = made.quick_view
    visited: list[str] = []
    for _ in range(8):
        QTest.keyClick(view, Qt.Key.Key_Tab)
        visited.append(view.quickWindow().activeFocusItem().objectName())
    assert [name for name in visited if name in BUTTONS][:4] == BUTTONS
    xs = [_item(made, name).mapToScene(QPointF(0, 0)).x() for name in BUTTONS]
    assert xs == sorted(xs)


def test_enter_presses_the_focused_button_and_escape_closes_the_dialog(dialog, qapp) -> None:
    made = dialog()
    view = made.quick_view
    _item(made, "closeButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert not made.isVisible() and made.result() == made.DialogCode.Rejected

    again = dialog()
    QTest.keyClick(again.quick_view, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not again.isVisible() and again.result() == again.DialogCode.Rejected


@pytest.mark.parametrize(("profile", "command"), [(None, "_on_run_clicked"), (PROFILE, "_on_accept_clicked")])
def test_enter_outside_a_button_does_what_the_main_button_does(dialog, monkeypatch, qapp, profile, command) -> None:
    calls: list[str] = []
    monkeypatch.setattr(LatencyCalibrationDialog, command, lambda self: calls.append(command))
    made = dialog(profile)
    QTest.keyClick(made.quick_view, Qt.Key.Key_Return)
    assert calls == [command]


def test_the_actions_fit_and_the_page_scrolls_only_vertically_in_a_small_dialog(dialog, qapp) -> None:
    made = dialog()
    made.resize(460, 360)
    qapp.processEvents()
    qapp.processEvents()
    made.view_model.toggleDetails()
    QTest.qWait(30)
    scroll, column = _item(made, "contentScroll"), _item(made, "contentColumn")
    right = scroll.mapToScene(QPointF(scroll.width(), 0)).x()
    wide = [
        (item.objectName(), item.mapToScene(QPointF(item.width(), 0)).x())
        for item in _all_items(column, only_visible=True)
        if item.mapToScene(QPointF(item.width(), 0)).x() > right + 0.5
    ]
    assert wide == [], "items run past the right edge of the page"
    assert scroll.property("contentHeight") > scroll.height()
    accept = _item(made, "acceptButton")
    assert accept.mapToScene(QPointF(accept.width(), 0)).x() <= made.quick_view.width()


def test_the_view_model_reports_changes_once_and_forwards_its_commands(qapp) -> None:
    calls: list[str] = []
    model = LatencyCalibrationView(
        run=lambda: calls.append("run"),
        accept=lambda: calls.append("accept"),
        reset=lambda: calls.append("reset"),
        close=lambda: calls.append("close"),
    )
    changes: list[str] = []
    model.statusChanged.connect(lambda: changes.append("status"))
    model.detailsShownChanged.connect(lambda: changes.append("details"))
    model.set_status("Ready")
    model.set_status("Preparing probe...")
    model.toggleDetails()
    assert changes == ["status", "details"] and model.status_text == "Preparing probe..." and model.detailsShown
    model.run()
    model.accept()
    model.reset()
    model.close()
    assert calls == ["run", "accept", "reset", "close"]


def test_the_keyboard_alone_scrolls_a_small_dialog_down_to_its_status_line(dialog, qapp) -> None:
    made = dialog()
    made.resize(460, 360)
    qapp.processEvents()
    qapp.processEvents()
    view = made.quick_view
    scroll, status = _item(made, "contentScroll"), _item(made, "statusText")
    assert scroll.property("contentHeight") > scroll.height() and not _within_page(made, status)

    for _ in range(3):
        if _focus_item_name(made) == "contentScroll":
            break
        QTest.keyClick(view, Qt.Key.Key_Tab)
    assert _focus_item_name(made) == "contentScroll", "the page is a tab stop"

    def press(key: Qt.Key) -> float:
        QTest.keyClick(view, key)
        qapp.processEvents()
        return scroll.property("contentY")

    line = press(Qt.Key.Key_Down)
    assert 0 < line < scroll.height()
    page = press(Qt.Key.Key_PageDown)
    assert page > line
    assert press(Qt.Key.Key_PageUp) < page
    assert press(Qt.Key.Key_Home) == 0
    assert press(Qt.Key.Key_Up) == 0
    last = press(Qt.Key.Key_End)
    assert last == scroll.property("contentHeight") - scroll.height()
    assert press(Qt.Key.Key_Down) == last
    assert _within_page(made, status) and _within_page(made, _item(made, "levelMeter"))


def test_the_page_shows_a_focus_ring_and_a_name_when_it_has_the_keyboard(dialog, qapp) -> None:
    made = dialog()
    ring = _item(made, "scrollFocusRing")
    assert not ring.isVisible()
    _item(made, "contentScroll").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()
    assert ring.isVisible()
    with _AccessibleScene() as scene:
        assert scene.name(_item(made, "contentScroll")) == "Latency calibration page"


def test_a_details_button_that_takes_the_keyboard_is_scrolled_into_view(dialog, qapp) -> None:
    made = dialog()
    made.resize(460, 360)
    qapp.processEvents()
    qapp.processEvents()
    made.view_model.toggleDetails()
    QTest.qWait(30)
    scroll, button = _item(made, "contentScroll"), _item(made, "detailsButton")
    scroll.setProperty("contentY", scroll.property("contentHeight") - scroll.height())
    qapp.processEvents()
    assert not _within_page(made, button)
    button.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()
    assert _within_page(made, button)


def test_no_reading_name_is_cut_short_at_a_large_text_size(dialog, qapp) -> None:
    previous = qapp.font()
    large = QFont(previous)
    large.setPointSize(14)
    qapp.setFont(large)
    try:
        made = dialog()
        made.view_model.toggleDetails()
        QTest.qWait(30)
        labels = _items_named(made, "readingLabel")
        assert len(labels) == 8
        widest = max(label.property("implicitWidth") for label in labels)
        assert made.minimumWidth() >= widest, "the dialog is never narrower than its names"
        made.resize(made.minimumWidth(), 620)
        qapp.processEvents()
        qapp.processEvents()
        assert [
            (label.property("text"), label.width(), label.property("implicitWidth"))
            for label in labels
            if label.property("truncated") or label.width() < label.property("implicitWidth")
        ] == []
    finally:
        qapp.setFont(previous)


@pytest.mark.parametrize(
    ("profile", "button", "command"),
    [
        (None, "runButton", "_on_run_clicked"),
        (PROFILE, "acceptButton", "_on_accept_clicked"),
        (PROFILE, "resetButton", "_on_reset_clicked"),
    ],
)
def test_pressing_a_scene_button_runs_the_dialogs_command(dialog, monkeypatch, qapp, profile, button, command) -> None:
    calls: list[str] = []
    monkeypatch.setattr(LatencyCalibrationDialog, command, lambda self: calls.append(command))
    made = dialog(profile)
    item = _item(made, button)
    item.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(made.quick_view, Qt.Key.Key_Space)
    assert calls == [command]

    centre = item.mapToScene(item.boundingRect().center()).toPoint()
    QTest.mouseClick(made.quick_view, Qt.MouseButton.LeftButton, pos=centre)
    qapp.processEvents()
    assert calls == [command, command]


def test_a_scene_that_does_not_load_raises_before_anything_is_connected(qapp, monkeypatch) -> None:
    connected: list[object] = []
    quitting = SimpleNamespace(aboutToQuit=SimpleNamespace(connect=connected.append))
    monkeypatch.setattr(
        "mic_eq.ui.latency_calibration_dialog.QApplication", SimpleNamespace(instance=lambda: quitting)
    )
    monkeypatch.setattr("mic_eq.ui.latency_calibration_dialog.load_scene", lambda *_args: "no such file")
    parent = QWidget()
    with pytest.raises(LatencyCalibrationLoadError, match="no such file"):
        LatencyCalibrationDialog(parent)
    assert connected == []
    assert parent.findChildren(QDialog) == []
    parent.deleteLater()
    qapp.processEvents()
