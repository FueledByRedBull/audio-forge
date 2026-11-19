"""The compact call window shows the main window's own state and stands in for it beside a call."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pytest
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, qInstallMessageHandler
from PySide6.QtGui import QAccessible, QColor, QGuiApplication, QIcon, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSystemTrayIcon
from shiboken6 import Shiboken

from mic_eq.ui import quick_shell
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


class _Hides(QObject):
    """Counts the hide events of the widget it watches."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Hide:
            self.count += 1
        return False


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


def _tab_to(compact: CompactWindow, name: str) -> None:
    view = compact.quick_view
    for _ in CONTROLS:
        QTest.keyClick(view, Qt.Key.Key_Tab)
        if view.quickWindow().activeFocusItem().objectName() == name:
            return
    raise AssertionError(f"{name} is not a tab stop")


def _wait_until(condition, what: str = "the condition") -> None:
    for _ in range(80):
        if condition():
            return
        QTest.qWait(25)
    raise AssertionError(f"{what} did not come true")


def _shown_tip(compact: CompactWindow):
    tip = compact.quick_view.rootObject().findChild(QObject, "compactTooltip")
    assert tip is not None
    _wait_until(lambda: tip.property("opened"), "the tooltip opening")
    return tip


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
    # Its own top-level window, so the taskbar and Alt+Tab list it.
    assert compact.parent() is None and compact.isWindow()
    assert not compact.testAttribute(Qt.WidgetAttribute.WA_QuitOnClose)

    # The same keys from the compact window close it again.
    compact.activateWindow()
    QTest.qWaitForWindowActive(compact)
    QTest.keyClick(
        compact.quick_view, Qt.Key.Key_A,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
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


def test_from_a_shown_main_window_it_hides_the_main_window_without_a_tray_notice(
    window, qapp, monkeypatch
) -> None:
    item = _tray_item(window, monkeypatch, "Compact window")
    notes: list[str] = []
    monkeypatch.setattr(window._tray_icon, "showMessage", lambda title, *_args: notes.append(title))
    item.trigger()
    qapp.processEvents()
    assert not window.isVisible()
    assert notes == [] and not window.__dict__.get("_hidden_to_tray")
    assert "tray" not in window.status_message.text
    window.compact_window.close()
    qapp.processEvents()
    assert window.isVisible() and not window.isMinimized()


def test_a_minimized_main_window_goes_back_to_minimized(window, qapp) -> None:
    window.showMinimized()
    qapp.processEvents()
    compact = _open(window, qapp)
    assert compact.isVisible() and not window.isVisible()
    compact.close()
    qapp.processEvents()
    assert window.isVisible() and window.isMinimized()


def test_without_a_tray_closing_it_brings_the_main_window_back(window, qapp, monkeypatch) -> None:
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: False)
    assert window._tray_icon is None
    compact = _open(window, qapp)
    compact.close()
    qapp.processEvents()
    assert window.isVisible() and not compact.isVisible()

    # Even from a hidden main window, which only a tray could bring back.
    window.hide()
    _open(window, qapp)
    compact.close()
    qapp.processEvents()
    assert window.isVisible() and not compact.isVisible()


def test_stopped_and_lost_device_show_the_overview_words(window, qapp, monkeypatch) -> None:
    compact = _open(window, qapp)
    overview = window.quick_bridge.model["overview"]
    headline = _item(compact, "compactHeadline")
    assert overview.state in ("ready", "no_mic")
    assert headline.property("text") == overview.headline != ""

    monkeypatch.setattr(window, "no_microphone", lambda: False)
    monkeypatch.setattr(window, "_restart_pending", lambda: True)
    overview.refresh()
    qapp.processEvents()
    assert overview.state == "device_lost"
    assert headline.property("text") == "Your callers can't hear you"
    dot = _item(compact, "compactToneDot")
    _wait_until(lambda: dot.property("color") == QColor(PALETTE.callers_affected))


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


def test_keep_on_top_switches_the_window_flag_and_lasts_the_session(window, qapp, monkeypatch) -> None:
    monkeypatch.setattr(window, "processor", _Running(window.processor))
    compact = _open(window, qapp)
    flag = Qt.WindowType.WindowStaysOnTopHint
    handle = compact.windowHandle()
    assert compact.windowFlags() & flag and _item(compact, "compactKeepOnTop").property("checked")
    assert handle.flags() & flag
    hides = _Hides()
    compact.installEventFilter(hides)
    _click(compact, "compactKeepOnTop", qapp)
    assert not compact.windowFlags() & flag and not compact.keepOnTop
    assert not handle.flags() & flag and compact.windowHandle() is handle
    assert compact.isVisible() and not _item(compact, "compactKeepOnTop").property("checked")
    assert not compact.windowFlags() & Qt.WindowType.WindowTransparentForInput
    # The window is not recreated, so it never hides and the meters never pause.
    assert hides.count == 0 and window.meter_timer.isActive()
    compact.removeEventFilter(hides)

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
    # Closed for good: it is deleted with the scene it shows.
    assert window.compact_window is None and not window.isVisible()
    assert not Shiboken.isValid(compact) or not compact.isVisible()


def test_the_app_quitting_without_a_close_closes_it(window, qapp) -> None:
    compact = _open(window, qapp)
    window._on_about_to_quit()
    qapp.processEvents()
    # Closed for good: it is deleted with the scene it shows.
    assert window.compact_window is None and not window.isVisible()
    assert not Shiboken.isValid(compact) or not compact.isVisible()


def test_dropping_the_main_scene_closes_it_and_shows_the_main_window(window, qapp) -> None:
    compact = _open(window, qapp)
    quick_shell._scene_stopped(window, "The scene stopped")
    qapp.processEvents()
    assert not compact.isVisible() and window.compact_window is None
    assert window.isVisible() and "quick_bridge" not in window.__dict__


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


def test_its_minimum_follows_the_scene_when_the_content_grows(window, qapp) -> None:
    compact = _open(window, qapp)
    root = compact.quick_view.rootObject()
    before = compact.minimumWidth()
    assert before == max(MINIMUM_WIDTH, root.property("minimumWidth"))
    window.shell_texts["latency_readout_label"].update(text="Measuring the latency of this route, please wait")
    # The scene lays out again on its next frame.
    _wait_until(lambda: root.property("minimumWidth") > before)
    grown = root.property("minimumWidth")
    assert grown > before
    assert compact.minimumWidth() == grown
    assert compact.width() >= grown


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


def test_the_mute_and_presets_buttons_explain_themselves_inside_the_window(window, qapp, monkeypatch) -> None:
    long_name = "A preset with a name far too long for the compact window button"
    compact = _open(window, qapp)
    # The window rewrites the summary from the configuration; hold it so the long name stays.
    monkeypatch.setattr(window, "_update_session_summary", lambda: None)
    window.shell_texts["preset_status_label"].update(data={"name": long_name, "modified": False})
    compact.resize(compact.minimumSize())
    qapp.processEvents()
    model = window.quick_bridge.model
    _tab_to(compact, "compactMute")
    tip = _shown_tip(compact)
    assert tip.property("text") == model["top"]["mute"].toolTip != ""

    _tab_to(compact, "compactPresets")
    tip = _shown_tip(compact)
    assert tip.property("text").startswith(long_name + "\n")
    root = compact.quick_view.rootObject()
    box = tip.property("contentItem").parentItem()
    rect = box.mapRectToScene(QRectF(0, 0, box.width(), box.height()))
    assert QRectF(0, 0, root.width(), root.height()).contains(rect)


def test_escape_after_tabbing_changes_nothing(window, qapp) -> None:
    compact = _open(window, qapp)
    _tab_to(compact, "compactKeepOnTop")
    view = compact.quick_view
    QTest.keyClick(view, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert compact.isVisible() and not window.isVisible()
    assert view.quickWindow().activeFocusItem().objectName() == "compactKeepOnTop"
    assert compact.keepOnTop and not window.user_mute_checkbox_action.isChecked()
