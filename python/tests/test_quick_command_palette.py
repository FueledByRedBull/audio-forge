"""The command palette: Ctrl+K over the main scene, its rows and their effects."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QAccessible, QColor, QKeyEvent
from PySide6.QtQml import QQmlEngine, QQmlExpression
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QDialog

from mic_eq.config import BUILTIN_PRESETS
from mic_eq.ui.command_palette import CALLERS_LINE, CALLERS_MUTED_LINE, NO_MATCH_TEXT, stage_setting
from test_quick_accessibility import no_qml_warnings  # noqa: F401  (an autouse fixture)
from test_quick_shell import _AccessibleScene
from test_quick_shell import quick_window as shell_window  # noqa: F401  (a fixture)


@pytest.fixture
def quick_window(shell_window):  # noqa: F811
    """The real main window with its Qt Quick scene, as test_quick_shell builds it."""
    return shell_window


def _palette(window):
    return window.quick_bridge.model["palette"]


def _scene(window, name: str):
    return window.centralWidget().rootObject().findChild(QObject, name)


def _show(window, qapp) -> None:
    window.resize(1400, 900)
    window.activateWindow()
    window.centralWidget().setFocus()
    qapp.processEvents()


def _press(window, key, qapp, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    QTest.keyClick(window.centralWidget(), key, modifiers)
    qapp.processEvents()


def _type(window, text: str, qapp) -> None:
    QTest.keyClicks(window.centralWidget(), text)
    qapp.processEvents()


def _select(palette, title: str) -> dict:
    palette.setQuery(title)
    index = next(i for i, row in enumerate(palette.rows) if row["title"] == title)
    palette.select(index)
    return palette.rows[index]


def _delegates(window) -> list:
    content = _scene(window, "paletteList").property("contentItem")
    return sorted(
        (item for item in content.childItems() if item.property("modelData") is not None),
        key=lambda item: item.property("index"),
    )


def _setting(window, section: str, key: str):
    stages = {stage["key"]: stage for stage in window.quick_bridge.model["stages"]}
    return stage_setting(stages[section], section, key)


def _items(item):
    yield item
    for child in item.childItems():
        yield from _items(child)


def _scene_rect(item) -> QRectF:
    return item.mapRectToScene(QRectF(0, 0, item.width(), item.height()))


def test_typing_filters_the_rows_and_an_empty_result_says_so(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    palette.show_palette()
    everything = [row["title"] for row in palette.rows]
    assert {"Start processing", "Stop processing", "Mute", "Run Auto Voice Setup", "Test my sound"} <= set(everything)
    assert {"Noise gate", "Noise suppression", "De-esser", "Equalizer", "Compressor", "Limiter"} <= set(everything)
    assert {preset.name for preset in BUILTIN_PRESETS.values()} <= set(everything)
    assert {row["group"] for row in palette.rows} == {"Action", "Stage", "Preset", "Setting"}
    assert palette.current == 0

    # Every word must match, in any case, in the title or the group.
    palette.setQuery("COMP thresh")
    assert [row["title"] for row in palette.rows] == ["Compressor threshold"]
    palette.setQuery("preset")
    assert {row["group"] for row in palette.rows} == {"Preset"}
    assert len(palette.rows) == len(BUILTIN_PRESETS)

    palette.setQuery("no such command")
    qapp.processEvents()
    assert palette.rows == [] and palette.current == -1
    empty = _scene(window, "paletteEmpty")
    assert empty.isVisible() and empty.property("text") == NO_MATCH_TEXT == "No matching commands"
    assert not _scene(window, "paletteList").isVisible()
    palette.setQuery("")
    qapp.processEvents()
    assert [row["title"] for row in palette.rows] == everything
    assert not empty.isVisible()


def test_an_action_row_triggers_the_toolbar_action(quick_window, qapp) -> None:
    window = quick_window
    palette = _palette(window)
    palette.show_palette()
    spy = QSignalSpy(window.user_mute_checkbox_action.triggered)
    _select(palette, "Mute")
    palette.run()
    assert spy.count() == 1
    assert window.engine_session.view.user_muted
    assert not palette.shown
    palette.show_palette()
    # The same row now unmutes, as the toolbar button does.
    _select(palette, "Unmute")
    palette.run()
    assert not window.engine_session.view.user_muted


def test_a_stage_row_toggles_through_the_stage_switch(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    compressor = next(row["toggle"] for row in window.quick_bridge.model["sidebar"]["chain"] if row["name"] == "Compressor")
    before = window.compressor_state.get_settings()["enabled"]
    spy = QSignalSpy(compressor.changed)
    palette.show_palette()
    row = _select(palette, "Compressor")
    assert row["value"] == ("On" if before else "Off")
    palette.run()
    assert window.compressor_state.get_settings()["enabled"] is (not before)
    assert spy.count() >= 1


def test_a_preset_row_loads_through_the_presets_menu(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    preset = list(BUILTIN_PRESETS.values())[-1]
    action = next(item for item in window.builtin_presets_menu.actions() if item.text() == preset.name)
    spy = QSignalSpy(action.triggered)
    palette.show_palette()
    row = _select(palette, preset.name)
    assert row["detail"] == preset.description
    palette.run()
    assert spy.count() == 1
    assert window.configuration.identity.name == preset.name


def test_the_mute_row_is_named_for_what_it_does_now(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    palette.show_palette()
    assert "Mute" in [row["title"] for row in palette.rows]
    # Muted from the toolbar while the palette is open: the row follows.
    window.user_mute_checkbox_action.trigger()
    titles = [row["title"] for row in palette.rows]
    assert "Unmute" in titles and "Mute" not in titles
    window.user_mute_checkbox_action.trigger()
    assert "Mute" in [row["title"] for row in palette.rows]


def test_a_disabled_action_shows_disabled_and_does_nothing(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    assert not window.stop_btn_action.isEnabled()
    spy = QSignalSpy(window.stop_btn_action.triggered)
    palette.show_palette()
    row = _select(palette, "Stop processing")
    assert row["enabled"] is False
    qapp.processEvents()
    assert _delegates(window)[palette.current].property("enabled") is False
    palette.run()
    assert spy.count() == 0
    # It stays open, so another row can be picked.
    assert palette.shown


def test_a_numeric_draft_changes_with_the_arrows_and_esc_discards_it(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    before = window.limiter_state.get_settings()["ceiling_db"]
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    _type(window, "limiter ceiling", qapp)
    assert [row["title"] for row in palette.rows] == ["Limiter ceiling"]
    _press(window, Qt.Key.Key_Left, qapp)
    _press(window, Qt.Key.Key_Left, qapp)
    _press(window, Qt.Key.Key_Right, qapp)
    row = palette.rows[0]
    assert row["draft"] is True
    proxy = _setting(window, "limiter", "ceiling_db")
    assert row["value"] == proxy.format_value(before - float(proxy.property("step")))
    assert row["detail"].startswith("Was ")
    # A draft is not an edit: the setting has not moved.
    assert window.limiter_state.get_settings()["ceiling_db"] == before
    _press(window, Qt.Key.Key_Escape, qapp)
    assert not palette.shown
    assert window.limiter_state.get_settings()["ceiling_db"] == before
    palette.show_palette()
    _select(palette, "Limiter ceiling")
    assert palette.rows[palette.current]["draft"] is False


def test_enter_applies_a_numeric_draft_as_an_undoable_edit(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    before = window.configuration.current.compressor.threshold_db
    after = before + 2 * float(_setting(window, "compressor", "threshold_db").property("step"))
    palette.show_palette()
    _select(palette, "Compressor threshold")
    assert palette.adjust(1) and palette.adjust(1)
    assert window.configuration.current.compressor.threshold_db == before
    palette.run()
    assert not palette.shown
    assert window.configuration.current.compressor.threshold_db == pytest.approx(after)
    window.undo_configuration()
    assert window.configuration.current.compressor.threshold_db == before
    window.redo_configuration()
    assert window.configuration.current.compressor.threshold_db == pytest.approx(after)


def test_left_and_right_on_a_row_that_is_not_numeric_leave_it_alone(quick_window) -> None:
    palette = _palette(quick_window)
    palette.show_palette()
    _select(palette, "Compressor")
    # Not handled, so the search field moves its cursor instead.
    assert palette.adjust(1) is False


@pytest.mark.parametrize("modifier", ["shift", "ctrl"])
def test_shift_or_ctrl_with_an_arrow_moves_the_cursor_not_the_draft(quick_window, qapp, modifier) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    held = Qt.KeyboardModifier.ShiftModifier if modifier == "shift" else Qt.KeyboardModifier.ControlModifier
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    _type(window, "limiter ceiling", qapp)
    field = _scene(window, "paletteSearch")
    assert field.property("cursorPosition") == len("limiter ceiling")
    _press(window, Qt.Key.Key_Left, qapp, held)
    assert palette.rows[0]["draft"] is False
    assert field.property("cursorPosition") < len("limiter ceiling")
    _press(window, Qt.Key.Key_Right, qapp, held)
    assert palette.rows[0]["draft"] is False
    # A plain arrow still steps it.
    _press(window, Qt.Key.Key_Left, qapp)
    assert palette.rows[0]["draft"] is True


def test_a_draft_goes_when_its_setting_is_disabled(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    window.compressor_state.set_value("auto_makeup_enabled", False)
    palette.show_palette()
    _select(palette, "Compressor makeup gain")
    assert palette.adjust(1)
    assert palette.rows[palette.current]["draft"] is True
    # Auto makeup takes the gain over, so the draft could not be applied.
    window.compressor_state.set_value("auto_makeup_enabled", True)
    row = next(row for row in palette.rows if row["title"] == "Compressor makeup gain")
    assert row["enabled"] is False and row["draft"] is False
    window.compressor_state.set_value("auto_makeup_enabled", False)
    row = next(row for row in palette.rows if row["title"] == "Compressor makeup gain")
    assert row["draft"] is False


def test_changing_the_filter_drops_every_draft(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    before = window.limiter_state.get_settings()["ceiling_db"]
    palette.show_palette()
    _select(palette, "Limiter ceiling")
    assert palette.adjust(-1)
    palette.setQuery("limiter ceil")
    assert palette.rows[0]["title"] == "Limiter ceiling" and palette.rows[0]["draft"] is False
    # Nothing to apply: Enter leaves the setting where it was.
    palette.run()
    assert window.limiter_state.get_settings()["ceiling_db"] == before


def test_the_callers_line_is_on_exactly_the_rows_that_change_what_callers_hear(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    palette.show_palette()
    qapp.processEvents()
    # Test my sound and Auto Voice Setup hold the live output muted while they run.
    muting = {"Run Auto Voice Setup", "Test my sound"}
    for row in palette.rows:
        assert row["callers"] == (CALLERS_MUTED_LINE if row["title"] in muting else CALLERS_LINE), row["title"]
        assert row["description"].endswith(row["callers"])
    assert muting <= {row["title"] for row in palette.rows}
    shown = {
        item.property("modelData")["title"]
        for item in _delegates(window)
        if item.findChild(QObject, "paletteCallers").property("visible")
    }
    assert shown and shown == {
        item.property("modelData")["title"] for item in _delegates(window) if item.property("modelData")["callers"]
    }


def test_ctrl_k_opens_the_palette_esc_closes_it_and_focus_returns(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    quick = window.centralWidget().quickWindow()
    _press(window, Qt.Key.Key_Tab, qapp)
    origin = quick.activeFocusItem()
    assert origin is not None and origin.property("visualFocus") is True
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    assert _palette(window).shown
    assert quick.activeFocusItem().objectName() == "paletteSearch"
    assert _scene(window, "commandPalette").isVisible()
    # Tab keeps focus in the palette.
    _press(window, Qt.Key.Key_Tab, qapp)
    assert quick.activeFocusItem().objectName() == "paletteSearch"
    _press(window, Qt.Key.Key_Escape, qapp)
    assert not _palette(window).shown
    assert quick.activeFocusItem() is origin
    # Back as a keyboard move, so its focus ring shows.
    assert origin.property("visualFocus") is True
    # A window shortcut, not a global hotkey, that holding the keys does not repeat.
    action = _palette(window).action
    assert action.shortcut().toString() == "Ctrl+K"
    assert action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
    assert action.autoRepeat() is False and action in window.actions()


def test_focus_goes_to_the_page_when_its_origin_is_hidden(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    quick = window.centralWidget().quickWindow()
    _press(window, Qt.Key.Key_Tab, qapp)
    _press(window, Qt.Key.Key_Tab, qapp)
    origin = quick.activeFocusItem()
    fallback = _scene(window, "commandPalette").property("fallbackFocus")
    assert origin is not None and fallback is not None and origin is not fallback
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    origin.setVisible(False)
    try:
        _press(window, Qt.Key.Key_Escape, qapp)
        assert not _palette(window).shown
        assert quick.activeFocusItem() is fallback
        assert fallback.property("visualFocus") is True
    finally:
        origin.setVisible(True)


def test_enter_runs_the_selected_row_from_the_keyboard(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    before = window.limiter_state.get_settings()["enabled"]
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    _type(window, "limiter", qapp)
    assert palette.rows[0]["title"] == "Limiter" and palette.current == 0
    _press(window, Qt.Key.Key_Down, qapp)
    assert palette.current == 1
    _press(window, Qt.Key.Key_Up, qapp)
    assert palette.current == 0
    _press(window, Qt.Key.Key_Return, qapp)
    assert not palette.shown
    assert window.limiter_state.get_settings()["enabled"] is (not before)


def test_a_click_selects_a_row_and_a_click_on_the_selected_row_runs_it(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    before = window.limiter_state.get_settings()["enabled"]
    palette.show_palette()
    palette.setQuery("limiter")
    # Clicked once the panel has settled.
    QTest.qWait(window.quick_bridge.model["theme"]["motionPanelOpen"] + 100)
    assert palette.current == 0 and palette.rows[0]["title"] == "Limiter"
    palette.clickAt(1)
    assert palette.shown and palette.current == 1
    # A real click on the first row selects it without running it.
    view = window.centralWidget()
    first = _delegates(window)[0]
    centre = first.mapToScene(first.boundingRect().center()).toPoint()
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
    qapp.processEvents()
    assert palette.shown and palette.current == 0
    assert window.limiter_state.get_settings()["enabled"] is before
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
    qapp.processEvents()
    assert not palette.shown
    assert window.limiter_state.get_settings()["enabled"] is (not before)


def test_a_second_ctrl_k_while_open_keeps_the_palette_as_it_is(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    quick = window.centralWidget().quickWindow()
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    _type(window, "limiter", qapp)
    _press(window, Qt.Key.Key_Down, qapp)
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    assert palette.shown and palette.query == "limiter" and palette.current == 1
    assert _scene(window, "paletteSearch").property("text") == "limiter"
    assert quick.activeFocusItem().objectName() == "paletteSearch"


def test_ctrl_k_from_a_text_field_opens_the_palette(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    window.shell_page.setIndex(window.EQUALIZER_PAGE_INDEX)
    qapp.processEvents()
    root = window.centralWidget().rootObject()
    field = next(
        item for item in _items(root)
        if item.metaObject().className().startswith("QQuickTextInput")
        and item.isVisible() and not item.property("readOnly")
    )
    field.forceActiveFocus()
    qapp.processEvents()
    quick = window.centralWidget().quickWindow()
    assert quick.activeFocusItem() is field
    text = field.property("text")
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    assert _palette(window).shown
    assert quick.activeFocusItem().objectName() == "paletteSearch"
    _press(window, Qt.Key.Key_Escape, qapp)
    # The field kept its text: Ctrl+K was never an edit there.
    assert quick.activeFocusItem() is field and field.property("text") == text


def test_ctrl_k_is_kept_from_a_field_that_would_take_it_as_an_edit_key(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    override = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    assert palette.eventFilter(window.centralWidget(), override) is True
    assert not override.isAccepted()
    other = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier)
    assert palette.eventFilter(window.centralWidget(), other) is False


def test_ctrl_k_under_a_modal_dialog_does_not_open_the_palette(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    dialog = QDialog(window)
    dialog.setModal(True)
    dialog.show()
    dialog.activateWindow()
    qapp.processEvents()
    try:
        QTest.keyClick(dialog, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
        qapp.processEvents()
        assert not _palette(window).shown
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_the_panel_stays_inside_the_pane_when_the_window_resizes(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _palette(window).show_palette()
    qapp.processEvents()
    pane = _scene(window, "commandPalette").property("backdrop")
    glass = _scene(window, "paletteGlass")
    for width, height in ((window.minimumWidth(), window.minimumHeight()), (1100, 760), (1400, 900)):
        window.resize(width, height)
        qapp.processEvents()
        assert _palette(window).shown
        assert _scene_rect(pane).contains(_scene_rect(glass)), (width, height)


def test_the_search_field_and_rows_are_named_for_assistive_technology(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    palette.show_palette()
    palette.setQuery("compressor")
    palette.select(next(i for i, row in enumerate(palette.rows) if row["title"] == "Compressor ratio"))
    assert palette.adjust(1)
    qapp.processEvents()
    delegates = _delegates(window)
    assert len(delegates) == len(palette.rows) > 1
    assert any(row["draft"] for row in palette.rows)
    with _AccessibleScene() as scene:
        assert scene.name(_scene(window, "paletteSearch")) == "Search commands"
        assert scene.name(_scene(window, "paletteList")) == "Commands"
        assert scene.name(_scene(window, "paletteGlass")) == "Command palette"
        for item, row in zip(delegates, palette.rows, strict=True):
            expected = f"{row['title']}: {row['value']}" if row["value"] else row["title"]
            assert scene.name(item) == expected
            assert scene.description(item) == row["description"]
            assert row["detail"] in row["description"]


def test_an_empty_result_is_announced(quick_window, monkeypatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr(QAccessible, "isActive", staticmethod(lambda: True))
    monkeypatch.setattr(QAccessible, "updateAccessibility", staticmethod(lambda event: sent.append(event.message())))
    palette = _palette(quick_window)
    palette.show_palette()
    palette.setQuery("no such command")
    assert sent == [NO_MATCH_TEXT]
    palette.setQuery("no such command at all")
    assert sent == [NO_MATCH_TEXT] * 2


def _duration(window, animation: str) -> int:
    """The current length of one of the palette's animations, read by its id in CommandPalette.qml."""
    inside = _scene(window, "paletteGlass")
    value, undefined = QQmlExpression(QQmlEngine.contextForObject(inside), inside, animation + ".duration").evaluate()
    assert not undefined, animation
    return int(value)


@pytest.mark.parametrize("motion", ["reduced", "full"])
def test_under_reduced_motion_the_palette_only_fades(request, monkeypatch, qapp, motion) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1" if motion == "reduced" else "0")
    window = request.getfixturevalue("quick_window")
    _show(window, qapp)
    theme = window.quick_bridge.model["theme"]
    overlay = _scene(window, "commandPalette")
    palette = _palette(window)
    palette.show_palette()
    assert _duration(window, "move") == (0 if motion == "reduced" else theme["motionPanelOpen"])
    assert _duration(window, "fade") == theme["motionCrossfade"] > 0
    QTest.qWait(max(theme["motionPanelOpen"], theme["motionCrossfade"]) + 100)
    assert overlay.property("progress") == 1 and overlay.property("opacity") == 1
    palette.close()
    assert _duration(window, "move") == (0 if motion == "reduced" else theme["motionPanelClose"])
    assert _duration(window, "fade") == theme["motionCrossfade"]
    QTest.qWait(max(theme["motionPanelClose"], theme["motionCrossfade"]) + 100)
    assert overlay.property("progress") == 0 and not overlay.isVisible()


@pytest.fixture(params=["dark", "HC Black", "HC White"])
def themed_window(request, monkeypatch):
    """The window in the dark look, then in the Windows black and white contrast themes."""
    from test_quick_accessibility import _system_themes, use_system_theme

    system = _system_themes()[request.param]
    palette = use_system_theme(monkeypatch, system)
    window = request.getfixturevalue("quick_window")
    vars(window)["system_palette"], vars(window)["system_colors"] = palette, system
    return window


def test_every_text_on_the_selected_row_reads_on_its_fill_in_each_theme(themed_window, qapp) -> None:
    from mic_eq.ui.theme import contrast_ratio, flatten

    window, palette, system = themed_window, themed_window.system_palette, themed_window.system_colors
    _show(window, qapp)
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    _type(window, "limiter ceiling", qapp)
    _press(window, Qt.Key.Key_Left, qapp)
    row = _palette(window).rows[0]
    # A draft shows its value in the accent text; every row says what callers hear.
    assert row["draft"] and row["callers"] and row["detail"] and row["value"] and row["group"]
    (delegate,) = _delegates(window)
    colours = {
        item.property("text"): QColor(item.property("color")).name()
        for item in _items(delegate)
        if item.metaObject().className().startswith("QQuickText") and item.property("text")
    }
    assert set(colours) == {row["title"], row["detail"], row["callers"], row["value"], row["group"]}
    fill = flatten(palette.accent_fill, flatten(palette.glass_modal, palette.app_surface))
    if system:
        # The fill is the highlight: all five are the highlighted text.
        assert set(colours.values()) == {palette.hover_text.lower()}
    else:
        assert colours == {
            row["title"]: palette.hover_text, row["detail"]: palette.text_muted,
            row["callers"]: palette.callers_affected_text, row["value"]: palette.accent_text,
            row["group"]: palette.text_muted,
        }
    assert all(contrast_ratio(colour, fill) >= 4.5 for colour in colours.values()), colours
