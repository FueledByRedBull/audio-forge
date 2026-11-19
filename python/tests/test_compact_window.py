"""The compact call window shows the main window's own state and stands in for it beside a call."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt, qInstallMessageHandler
from PySide6.QtGui import QAccessible, QColor, QGuiApplication, QIcon, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSystemTrayIcon

from mic_eq.ui.compact_window import MINIMUM_HEIGHT, MINIMUM_WIDTH, CompactWindow
from mic_eq.ui.theme import PALETTE

CONTROLS = ["compactMute", "compactPresets", "compactExpand", "compactKeepOnTop"]
READINGS = ["compactToneDot", "compactHeadline", "compactHealth", "compactInputBar", "compactOutputBar", "compactLatency"]


class _Running:
    """The window's processor, reporting that audio runs."""

    def __init__(self, native: Any) -> None:
        self.native = native

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def is_running(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def no_qml_warnings():
    messages: list[str] = []
    previous = qInstallMessageHandler(
        lambda _mode, _context, message: messages.append(message) if ".qml" in message else None
    )
    yield
    qInstallMessageHandler(previous)
    assert messages == [], "the scene printed warnings: " + "; ".join(messages[:3])


@pytest.fixture
def window(isolated_main_window, qapp):
    shown: Any = isolated_main_window
    shown.show()
    qapp.processEvents()
    yield shown
    if shown._tray_icon is not None:
        shown._tray_icon.hide()


def _tray(window: Any, monkeypatch) -> Any:
    base = QPixmap(32, 32)
    base.fill(QColor(PALETTE.card_surface))
    window.setWindowIcon(QIcon(base))
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    window._setup_desktop_integration(tray_only=True)
    return window._tray_menu


def _tray_item(window: Any, monkeypatch, text: str):
    return next(action for action in _tray(window, monkeypatch).actions() if action.text() == text)


def _open(window: Any, qapp) -> CompactWindow:
    window.compact_window_action.trigger()
    qapp.processEvents()
    assert window.compact_window is not None
    return window.compact_window


def _items(root):
    for child in root.childItems():
        yield child
        yield from _items(child)


def _item(compact: CompactWindow, name: str):
    return next(item for item in _items(compact.quick_view.rootObject()) if item.objectName() == name)


def _click(compact: CompactWindow, name: str, qapp) -> None:
    item = _item(compact, name)
    centre = item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()
    QTest.mouseClick(compact.quick_view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
    qapp.processEvents()


def _name(item) -> str:
    return QAccessible.queryAccessibleInterface(item).text(QAccessible.Text.Name)


def test_ctrl_shift_a_opens_it_from_the_main_window_and_closing_it_brings_the_main_window_back(
    window, qapp
) -> None:
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QTest.keyClick(
        window.centralWidget(), Qt.Key.Key_A,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    qapp.processEvents()
    compact = window.compact_window
    assert compact is not None and compact.isVisible()
    assert not window.isVisible()
    assert compact.windowTitle() == "AudioForge"

    compact.close()
    qapp.processEvents()
    assert window.isVisible() and not compact.isVisible()


def test_the_tray_opens_it_and_closing_it_leaves_a_hidden_main_window_in_the_tray(
    window, qapp, monkeypatch
) -> None:
    item = _tray_item(window, monkeypatch, "Compact window")
    window.hide()
    item.trigger()
    qapp.processEvents()
    compact = window.compact_window
    assert compact is not None and compact.isVisible() and not window.isVisible()

    # Opening it again while it shows keeps where it came from.
    item.trigger()
    compact.close()
    qapp.processEvents()
    assert not window.isVisible() and not compact.isVisible()


def test_from_a_shown_main_window_the_tray_hides_it_as_close_to_tray_does(window, qapp, monkeypatch) -> None:
    item = _tray_item(window, monkeypatch, "Compact window")
    notes: list[str] = []
    monkeypatch.setattr(window._tray_icon, "showMessage", lambda title, *_args: notes.append(title))
    item.trigger()
    qapp.processEvents()
    assert not window.isVisible() and window._hidden_to_tray
    assert notes == ["AudioForge is running in the background"]
    window.compact_window.close()
    qapp.processEvents()
    assert window.isVisible()


def test_it_holds_the_status_health_meters_mute_preset_expand_latency_and_no_stop(window, qapp) -> None:
    compact = _open(window, qapp)
    for name in CONTROLS + READINGS:
        assert _item(compact, name).isVisible(), name
    model = window.quick_bridge.model
    assert _item(compact, "compactHeadline").property("text") == model["overview"].headline
    assert _item(compact, "compactInputBar").property("source") is model["meters"][0]
    assert _item(compact, "compactOutputBar").property("source") is model["meters"][1]
    every = list(_items(compact.quick_view.rootObject()))
    assert not [item for item in every if item.property("proxy") in (model["top"]["stop"], model["top"]["start"])]
    assert not [item for item in every if item.property("text") in ("Stop", "Start")]
    # One state: the compact window adds no scene model of its own.
    assert window.compact_window_action.parent() is window
    assert compact.quick_view.rootObject().property("ui") is window.quick_bridge


def test_mute_presets_and_expand_act_through_the_shared_commands(window, qapp, monkeypatch) -> None:
    compact = _open(window, qapp)
    _click(compact, "compactMute", qapp)
    assert window.user_mute_checkbox_action.isChecked()
    assert _item(compact, "compactMute").property("checked")
    window.user_mute_checkbox_action.setChecked(False)
    qapp.processEvents()
    assert not _item(compact, "compactMute").property("checked")

    popped: list = []
    monkeypatch.setattr(window.presets_menu, "popup", lambda position, *_args: popped.append(position))
    presets = _item(compact, "compactPresets")
    assert presets.property("text") == window.shell_texts["preset_status_label"].data["name"]
    _click(compact, "compactPresets", qapp)
    assert len(popped) == 1
    below = presets.mapToGlobal(QPointF(0, presets.height() + 4))
    assert (popped[0].x(), popped[0].y()) == (round(below.x()), round(below.y()))

    _click(compact, "compactExpand", qapp)
    assert window.isVisible() and not compact.isVisible()


def test_keep_on_top_switches_the_window_flag_and_lasts_the_session(window, qapp) -> None:
    compact = _open(window, qapp)
    flag = Qt.WindowType.WindowStaysOnTopHint
    assert compact.windowFlags() & flag and _item(compact, "compactKeepOnTop").property("checked")
    _click(compact, "compactKeepOnTop", qapp)
    assert not compact.windowFlags() & flag and not compact.keepOnTop
    assert compact.isVisible() and not _item(compact, "compactKeepOnTop").property("checked")
    assert not compact.windowFlags() & Qt.WindowType.WindowTransparentForInput

    compact.close()
    _open(window, qapp)
    assert window.compact_window is compact and not compact.windowFlags() & flag
    _click(compact, "compactKeepOnTop", qapp)
    assert compact.windowFlags() & flag and compact.isVisible()


def test_the_meters_run_only_while_it_shows(window, qapp, monkeypatch) -> None:
    monkeypatch.setattr(window, "processor", _Running(window.processor))
    compact = _open(window, qapp)
    assert not window.isVisible()
    assert window.meter_timer.isActive()
    compact.setWindowState(Qt.WindowState.WindowMinimized)
    qapp.processEvents()
    assert not window.meter_timer.isActive()
    compact.setWindowState(Qt.WindowState.WindowNoState)
    qapp.processEvents()
    assert window.meter_timer.isActive()
    compact.hide()
    assert not window.meter_timer.isActive()
    window.meter_timer.stop()


def test_quitting_closes_it(window, qapp, monkeypatch) -> None:
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: None))
    item = _tray_item(window, monkeypatch, "Compact window")
    window.hide()
    item.trigger()
    qapp.processEvents()
    compact = window.compact_window
    assert compact is not None and compact.isVisible()

    window._quit_from_tray()
    qapp.processEvents()
    assert not compact.isVisible() and not window.isVisible()


def test_the_app_quitting_without_a_close_closes_it(window, qapp) -> None:
    compact = _open(window, qapp)
    window._on_about_to_quit()
    qapp.processEvents()
    assert not compact.isVisible() and not window.isVisible()


def test_show_from_the_tray_expands_it(window, qapp) -> None:
    compact = _open(window, qapp)
    window._show_from_tray()
    qapp.processEvents()
    assert window.isVisible() and not compact.isVisible()


def test_at_its_minimum_size_nothing_overlaps_or_leaves_the_window(window, qapp) -> None:
    compact = _open(window, qapp)
    # The scene asks for more only when the text is larger than usual.
    assert MINIMUM_WIDTH <= compact.minimumWidth() <= MINIMUM_WIDTH + 40
    assert MINIMUM_HEIGHT <= compact.minimumHeight() <= MINIMUM_HEIGHT + 40
    compact.resize(compact.minimumSize())
    qapp.processEvents()
    root = compact.quick_view.rootObject()
    assert (root.width(), root.height()) == (compact.minimumWidth(), compact.minimumHeight())
    bounds = QRectF(0, 0, root.width(), root.height())
    rects = {}
    for name in CONTROLS + READINGS:
        item = _item(compact, name)
        rect = item.mapRectToScene(QRectF(0, 0, item.width(), item.height()))
        assert item.width() > 0 and item.height() > 0, name
        assert bounds.contains(rect), name
        rects[name] = rect
    overlaps = [(a, b) for a, b in combinations(rects, 2) if rects[a].intersects(rects[b])]
    assert overlaps == []


def test_every_control_is_a_named_tab_stop_in_reading_order(window, qapp) -> None:
    compact = _open(window, qapp)
    view = compact.quick_view
    stops = []
    for _ in range(len(CONTROLS) + 1):
        QTest.keyClick(view, Qt.Key.Key_Tab)
        stops.append(view.quickWindow().activeFocusItem().objectName())
    assert stops == [*CONTROLS, CONTROLS[0]]
    assert [_name(_item(compact, name)) for name in CONTROLS] == ["Mute output", "Presets", "Expand", "Keep on top"]
    for name in ("compactHeadline", "compactInputBar", "compactOutputBar"):
        assert _name(_item(compact, name)) != "", name
    latency = _item(compact, "compactLatency").childItems()[0]
    assert _name(latency).startswith("Latency")
    health = _item(compact, "compactHealth").childItems()[-1]
    assert _name(health).endswith(window.shell_texts["health_summary_label"].text)


def test_escape_does_nothing(window, qapp) -> None:
    compact = _open(window, qapp)
    _item(compact, "compactMute").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(compact.quick_view, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert compact.isVisible() and not window.isVisible()
    assert not window.user_mute_checkbox_action.isChecked()
