"""The Qt Quick scene keeps long text inside its column at the minimum window size."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from PySide6.QtCore import QObject, QRectF, qInstallMessageHandler
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtQuick import QQuickItem
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtTest import QTest

from mic_eq.config import AppConfig
from mic_eq.ui.main_window import MINIMUM_WINDOW_HEIGHT, MINIMUM_WINDOW_WIDTH, MainWindow
from mic_eq.ui.processing_configuration import PresetOrigin
from mic_eq.ui.quick_shell import InspectorState
from mic_eq.ui.theme import RADIUS_PANEL

INPUT = SimpleNamespace(name="CABLE Output (VB-Audio Virtual Cable)", is_default=True)
OUTPUT = SimpleNamespace(name="Realtek Digital Output (Realtek USB Audio)", is_default=True)
STALE_CALIBRATION = "Saved calibration is stale: devices or settings differ; recalibrate"
LONG_PRESET = "Hands-on test: <weird>/name that keeps going well past any sensible width"
SEGOE_UI = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"


def build_minimum_window(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    status: str,
    *,
    modified: bool = False,
    width: int = MINIMUM_WINDOW_WIDTH,
) -> MainWindow:
    """The real window at its minimum size; only the device lists are faked."""
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [INPUT])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [OUTPUT])
    window = MainWindow()
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    named_preset = window.configuration.current
    named_preset.name = preset
    window.configuration.load(named_preset, PresetOrigin(), remember=False)
    if modified:
        window.gate_state.set_settings({"threshold_db": window.gate_state.get_settings()["threshold_db"] + 1})
        window.configuration.commit_edit()
    # The platform minimum is clamped to the offscreen screen; ask for the real one.
    window.setMinimumSize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)
    window.resize(width, MINIMUM_WINDOW_HEIGHT)
    window.show()
    window.status_message.showMessage(status)
    window.shell_texts["calibration_status_label"].update(text=STALE_CALIBRATION)
    for _ in range(3):
        qapp.processEvents()
        QTest.qWait(30)
    return window


@pytest.fixture
def segoe_font(qapp):
    """Text widths depend on the font, and Qt's offscreen platform has none.

    The icon font decides whether undo and redo are icons or wider text buttons.
    """
    if not SEGOE_UI.is_file():
        pytest.skip("The Windows UI font is not installed")
    families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(SEGOE_UI)))
    for icons in ("SegoeIcons.ttf", "segmdl2.ttf"):
        QFontDatabase.addApplicationFont(str(SEGOE_UI.with_name(icons)))
    previous = qapp.font()
    qapp.setFont(QFont(families[0], 9))
    yield
    qapp.setFont(previous)


@pytest.fixture(params=["short", "short-unsaved", "long"])
def minimum_window(request, qapp, monkeypatch, segoe_font):
    preset = LONG_PRESET if request.param == "long" else "DaBest"
    window = build_minimum_window(
        qapp, monkeypatch, preset, f"Loaded preset: {preset}", modified=request.param == "short-unsaved"
    )
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def scene_rect(item: QQuickItem) -> QRectF:
    return item.mapRectToScene(QRectF(0, 0, item.width(), item.height()))


def named(window: MainWindow, name: str) -> QQuickItem:
    item = cast(QQuickWidget, window.centralWidget()).rootObject().findChild(QObject, name)
    assert isinstance(item, QQuickItem), name
    return item


def shown_items(item: QQuickItem):
    """Every visible item below ``item`` that takes up room."""
    for child in item.childItems():
        if child.isVisible() and child.width() > 0 and child.height() > 0:
            yield child
            yield from shown_items(child)


@pytest.mark.parametrize("minimum_window", ["short", "short-unsaved"], indirect=True)
def test_the_scene_is_built_at_its_minimum_width(minimum_window) -> None:
    assert named(minimum_window, "sidebar").width() > 0
    assert minimum_window.centralWidget().width() == MINIMUM_WINDOW_WIDTH


def test_nothing_in_the_main_pane_runs_into_the_sidebar_or_off_the_window(minimum_window) -> None:
    sidebar_right = scene_rect(named(minimum_window, "sidebar")).right()
    window_right = float(minimum_window.centralWidget().width())
    pane = named(minimum_window, "mainPane")
    overflowing = [
        f"{item.metaObject().className()} {item.objectName()!r} "
        f"left={scene_rect(item).left():.1f} right={scene_rect(item).right():.1f}"
        for item in shown_items(pane)
        if scene_rect(item).left() < sidebar_right - 0.5 or scene_rect(item).right() > window_right + 0.5
    ]
    assert not overflowing, f"the sidebar ends at {sidebar_right:.1f}, the window at {window_right:.1f}: {overflowing}"


def test_a_long_preset_name_is_elided_and_whole_in_the_tooltip(minimum_window) -> None:
    name = named(minimum_window, "presetName")
    long_name = minimum_window.configuration.identity.name == LONG_PRESET
    assert name.property("truncated") is long_name
    hint = next(child for child in name.children() if "Hint" in child.metaObject().className())
    assert hint.property("text") == (LONG_PRESET if long_name else "")


def caption_tip(item: QQuickItem) -> str:
    hint = next(child for child in item.children() if "Hint" in child.metaObject().className())
    return hint.property("text")


def status_caption(window: MainWindow, text: str) -> QQuickItem:
    footer = named(window, "sidebarFooter")
    return next(item for item in shown_items(footer) if item.property("text") == text)


# The unsaved flag only reaches the preset block; the footer reads the calibration and health,
# so one saved and one long-named window cover it.
@pytest.mark.parametrize("minimum_window", ["short", "long"], indirect=True)
def test_an_elided_status_caption_has_its_full_text_as_its_tip(minimum_window) -> None:
    message = named(minimum_window, "statusMessage")
    calibration = status_caption(minimum_window, STALE_CALIBRATION)
    for caption in (message, calibration):
        assert caption_tip(caption) == (caption.property("text") if caption.property("truncated") else "")


@pytest.mark.parametrize("width", [MINIMUM_WINDOW_WIDTH, 1600])
def test_the_calibration_text_wraps_whole_under_the_health_chip(qapp, monkeypatch, segoe_font, width) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Loaded preset: DaBest", width=width)
    try:
        assert window.centralWidget().width() == width
        calibration = status_caption(window, STALE_CALIBRATION)
        assert calibration.property("truncated") is False
        assert calibration.property("lineCount") > 1
        # The note sits 14 px under the Health chip and keeps inside the sidebar.
        gap = scene_rect(calibration).top() - scene_rect(health_chip(window)).bottom()
        assert gap == pytest.approx(14, abs=1)
        assert scene_rect(calibration).right() <= scene_rect(named(window, "sidebar")).right()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("minimum_window", ["short", "long"], indirect=True)
def test_the_sidebar_footer_parts_do_not_overlap(minimum_window) -> None:
    footer = named(minimum_window, "sidebarFooter")
    boxes = sorted(
        (scene_rect(child) for child in footer.childItems() if child.isVisible()),
        key=QRectF.top,
    )
    assert len(boxes) == 4
    assert min(box.width() for box in boxes) > 0, "a part was squeezed out"
    for upper, lower in zip(boxes, boxes[1:]):
        assert upper.bottom() <= lower.top(), f"{upper} runs into {lower}"
    sidebar = scene_rect(named(minimum_window, "sidebar"))
    assert all(sidebar.left() <= box.left() and box.right() <= sidebar.right() for box in boxes)
    assert boxes[-1].bottom() <= sidebar.bottom()
    # The line under the pages holds only the status message.
    message = named(minimum_window, "statusMessage")
    assert message.parentItem() is named(minimum_window, "mainPane")


LONG_HEALTH_TEXTS = [
    "Drops: 256 samples dropped, 0 dropouts, 1 restart (needs attention: the audio engine reported a fault)",
    "Device activity: input has not started sending sound yet, output has not started playing sound yet",
    "Output: LIMITER WORKING HARD (7.5 dB)",
]


def settle(qapp) -> None:
    for _ in range(3):
        qapp.processEvents()
        QTest.qWait(30)


def health_chip(window: MainWindow) -> QQuickItem:
    footer = named(window, "sidebarFooter")
    return next(child for child in footer.childItems() if "Chip" in child.metaObject().className())


@pytest.mark.parametrize("minimum_window", ["short", "long"], indirect=True)
@pytest.mark.parametrize("text", LONG_HEALTH_TEXTS)
def test_a_long_health_text_is_cut_short_and_whole_in_the_tooltip_and_the_name(
    minimum_window, qapp, text
) -> None:
    from PySide6.QtGui import QAccessible

    window = minimum_window
    window._set_health_chip("health_summary_label", text, "warn")
    window.shell_texts["health_summary_label"].update(tip="What to do about it.")
    settle(qapp)

    # The other parts keep their room and nothing runs past the sidebar.
    test_the_sidebar_footer_parts_do_not_overlap(window)
    chip = health_chip(window)
    shown = next(child for child in chip.childItems() if child.metaObject().className() == "QQuickText")
    assert shown.property("truncated") is True
    assert caption_tip(chip) == f"{text}\n\nWhat to do about it."
    was_active = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        toggle = named(window, "healthToggle")
        assert QAccessible.queryAccessibleInterface(toggle).text(QAccessible.Text.Name) == text
    finally:
        QAccessible.setActive(was_active)


@pytest.mark.parametrize("minimum_window", ["short", "long"], indirect=True)
def test_a_short_health_text_keeps_its_natural_width_and_its_own_tooltip(minimum_window, qapp) -> None:
    window = minimum_window
    window._set_health_chip("health_summary_label", "Health: OK", "ok")
    window.shell_texts["health_summary_label"].update(tip="Everything looks fine.")
    settle(qapp)

    chip = health_chip(window)
    shown = next(child for child in chip.childItems() if child.metaObject().className() == "QQuickText")
    assert shown.property("truncated") is False
    assert chip.width() == pytest.approx(chip.implicitWidth())
    assert caption_tip(chip) == "Everything looks fine."


@pytest.mark.parametrize("minimum_window", ["short", "long"], indirect=True)
def test_the_status_message_keeps_its_room_at_the_worst_combination(minimum_window, qapp) -> None:
    window = minimum_window
    transmission = "Transmission: Running / Raw monitor / Mute pending"
    window.shell_texts["transmission_status_label"].update(text=transmission, status="warn")
    window._set_health_chip("health_summary_label", LONG_HEALTH_TEXTS[0], "warn")
    window.status_message.showMessage("Loaded preset: DaBest")
    settle(qapp)

    # The sidebar texts give way in the sidebar; the message keeps the main pane's width.
    test_the_sidebar_footer_parts_do_not_overlap(window)
    message = named(window, "statusMessage")
    assert message.width() >= named(window, "mainPane").width() * 3 / 4
    # The Overview says the transmission now; the sidebar no longer shows it.
    assert all(item.property("text") != transmission for item in shown_items(named(window, "sidebarFooter")))
    calibration = status_caption(window, STALE_CALIBRATION)
    assert calibration.property("truncated") is False
    assert caption_tip(calibration) == ""


def assert_toolbar_fits(window: MainWindow, width: int) -> None:
    """Nothing in the toolbar leaves it, and each part of its row keeps its own room."""
    bar = named(window, "toolbar")
    edge = scene_rect(bar)
    assert edge.right() == pytest.approx(width)
    beyond = [
        f"{item.metaObject().className()} {item.objectName()!r} {scene_rect(item)}"
        for item in shown_items(bar)
        if scene_rect(item).right() > edge.right() + 0.5
        or scene_rect(item).top() < edge.top() - 0.5
        or scene_rect(item).bottom() > edge.bottom() + 0.5
    ]
    assert not beyond, f"the toolbar is {edge}: {beyond}"
    row = named(window, "toolRow")
    parts = sorted(
        (scene_rect(child) for child in row.childItems() if child.isVisible() and child.width() > 0),
        key=QRectF.left,
    )
    for left, right in zip(parts, parts[1:]):
        assert left.right() <= right.left() + 0.5, f"{left} runs into {right}"


# The tightest cells: the minimum width, the `wide` threshold and the widest window kept, at the
# smallest and largest font, with Mute pressed and the mode changed. Other cells leave more room.
@pytest.mark.parametrize("point_size", [9, 14])
@pytest.mark.parametrize("width", [MINIMUM_WINDOW_WIDTH, 1200, 1300])
@pytest.mark.parametrize("state", ["unmuted", "muted-bypass"])
def test_the_toolbar_fits_one_row_at_every_font_size_and_width(
    qapp, monkeypatch, segoe_font, point_size, width, state
) -> None:
    qapp.setFont(QFont(qapp.font().family(), point_size))
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=width)
    try:
        if state == "muted-bypass":
            window.user_mute_checkbox_action.setChecked(True)
            window.processing_mode_choice.setIndex(1)
        settle(qapp)
        assert_toolbar_fits(window, width)
        bar = named(window, "toolbar")
        mute = next(item for item in shown_items(bar) if item.metaObject().className().startswith("MuteBtn"))
        assert mute.property("on") is (state == "muted-bypass")
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("point_size", [9, 14])
def test_the_toolbar_fits_the_no_microphone_reason_at_the_minimum_width(
    qapp, monkeypatch, segoe_font, point_size
) -> None:
    messages: list[str] = []
    previous = qInstallMessageHandler(lambda _mode, _context, message: messages.append(message))
    qapp.setFont(QFont(qapp.font().family(), point_size))
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready")
    try:
        monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
        window._refresh_devices()
        settle(qapp)
        reason = named(window, "startBlockedReason")
        assert reason.isVisible() and reason.property("text") == "No microphone to start from"
        assert_toolbar_fits(window, MINIMUM_WINDOW_WIDTH)
        # Cut short, the reason is whole in its tooltip.
        assert reason.property("tip") == (reason.property("text") if reason.property("truncated") else "")
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
        qInstallMessageHandler(previous)
    assert [message for message in messages if "inding loop" in message or ".qml" in message] == []


def test_a_wide_toolbar_keeps_the_no_microphone_reason_on_one_line(qapp, monkeypatch, segoe_font) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=1440)
    try:
        monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
        window._refresh_devices()
        settle(qapp)
        reason = named(window, "startBlockedReason")
        assert reason.isVisible() and reason.property("lineCount") == 1
        assert not reason.property("truncated")
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("width", "sidebar_width"), [(MINIMUM_WINDOW_WIDTH, 220), (1199, 220), (1200, 256), (1600, 256)]
)
def test_the_sidebar_width_follows_the_wide_breakpoint(qapp, monkeypatch, segoe_font, width, sidebar_width) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=width)
    try:
        assert named(window, "sidebar").width() == sidebar_width
        assert scene_rect(named(window, "mainPane")).left() > sidebar_width
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("width", [MINIMUM_WINDOW_WIDTH, 1199, 1200, 1600])
def test_the_inspector_docks_from_the_wide_breakpoint_and_lies_over_the_pane_below_it(
    qapp, monkeypatch, segoe_font, width
) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=width)
    try:
        before = scene_rect(named(window, "mainPane"))
        inspector = window.findChild(InspectorState)
        assert inspector is not None
        inspector.show("compressor")
        QTest.qWait(400)
        panel = named(window, "inspector")
        glass = scene_rect(panel)
        assert glass.width() == 400
        assert glass.top() > scene_rect(named(window, "toolbar")).bottom()
        assert glass.bottom() < MINIMUM_WINDOW_HEIGHT and glass.right() < width
        pane = scene_rect(named(window, "mainPane"))
        rail = named(window, "inspectorRail")
        if width >= 1200:
            # Docked: the pane makes room, and the glass has nothing behind it to blur.
            # The main panel ends one ground gap before the rail, and the pane keeps its inset.
            main_panel = scene_rect(named(window, "mainPanel"))
            assert main_panel.right() == width - 424 - 10 < glass.left()
            assert pane.right() == main_panel.right() - 16
            assert rail.isVisible() and scene_rect(rail).left() == main_panel.right() + 10
            assert panel.property("backdrop") is None
        else:
            # Over the pane, which keeps its width and shows through the glass.
            assert pane == before and glass.left() < pane.right()
            assert not rail.isVisible()
            assert panel.property("backdrop").objectName() == "mainPane"
        overflowing = [
            f"{item.metaObject().className()} left={scene_rect(item).left():.1f} right={scene_rect(item).right():.1f}"
            for item in shown_items(panel)
            if scene_rect(item).left() < glass.left() - 0.5 or scene_rect(item).right() > glass.right() + 0.5
        ]
        assert not overflowing, f"the inspector spans {glass.left():.1f} to {glass.right():.1f}: {overflowing}"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def navigation_rows(window: MainWindow) -> list[QQuickItem]:
    return [
        item for item in shown_items(named(window, "sidebar")) if item.metaObject().className().startswith("NavRow")
    ]


@pytest.mark.parametrize("width", [MINIMUM_WINDOW_WIDTH, 1600])
def test_the_sidebar_and_the_main_panel_stand_apart_on_the_ground(qapp, monkeypatch, segoe_font, width) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=width)
    try:
        sidebar = named(window, "sidebar")
        panel = named(window, "mainPanel")
        gap = 10
        assert scene_rect(sidebar).left() == gap
        assert scene_rect(panel).left() - scene_rect(sidebar).right() == gap
        assert scene_rect(panel).right() == width - gap
        assert sidebar.property("radius") == panel.property("radius") == RADIUS_PANEL
        assert scene_rect(panel).contains(scene_rect(named(window, "mainPane")))
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(("width", "tiled"), [(MINIMUM_WINDOW_WIDTH, False), (1199, False), (1200, True)])
def test_the_row_icons_show_only_where_the_names_keep_their_room(
    qapp, monkeypatch, segoe_font, width, tiled
) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=width)
    try:
        rows = navigation_rows(window)
        assert len(rows) == 8
        assert [row.property("tiled") for row in rows] == [tiled] * 8
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_the_stopped_health_line_says_it_is_not_running(qapp, monkeypatch, segoe_font) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready")
    try:
        chip = health_chip(window)
        assert chip.property("level") == "idle"
        assert chip.property("shown") == "Health: not running"
        window._set_health_chip("health_summary_label", "Drops: 0", "ok")
        settle(qapp)
        assert chip.property("shown") == "Drops: 0"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_the_stopped_level_card_waits_for_start(qapp, monkeypatch, segoe_font) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready")
    try:
        waiting = named(window, "ribbonWaiting")
        assert waiting.isVisible()
        assert not named(window, "levelRibbon").isVisible()
        assert scene_rect(named(window, "levelRibbonCard")).contains(scene_rect(waiting))
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


# The largest font at the minimum width is where the panels, the rows and the footer are tightest.
def test_the_panels_keep_clear_at_the_minimum_width_and_a_large_font(qapp, monkeypatch, segoe_font) -> None:
    qapp.setFont(QFont(qapp.font().family(), 14))
    window = build_minimum_window(qapp, monkeypatch, LONG_PRESET, "Ready")
    try:
        test_nothing_in_the_main_pane_runs_into_the_sidebar_or_off_the_window(window)
        test_the_sidebar_footer_parts_do_not_overlap(window)
        sidebar = scene_rect(named(window, "sidebar"))
        for row in navigation_rows(window):
            box = scene_rect(row)
            assert sidebar.left() <= box.left() and box.right() <= sidebar.right(), row.property("name")
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_each_route_check_has_its_round_mark_in_the_card(qapp, monkeypatch, segoe_font) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Ready", width=1440)
    try:
        card = named(window, "overviewChecks")
        repeater = next(item for item in card.findChildren(QObject) if item.metaObject().className() == "QQuickRepeater")
        marks = [item for item in shown_items(card) if item.metaObject().className().startswith("CheckMark")]
        assert len(marks) == repeater.property("count") > 0
        assert {mark.property("kind") for mark in marks} <= {"ok", "warn", "note"}
        assert all(mark.width() == mark.height() == 2 * mark.property("radius") for mark in marks)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
