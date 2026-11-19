"""The command palette: Ctrl+K over the main scene, its rows and their effects."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import pytest
from PySide6.QtCore import QElapsedTimer, QObject, Qt
from PySide6.QtTest import QSignalSpy, QTest

from mic_eq.config import BUILTIN_PRESETS
from mic_eq.ui.command_palette import CALLERS_LINE, NO_MATCH_TEXT
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


def test_typing_filters_the_rows_and_an_empty_result_says_so(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    palette.open()
    everything = [row["title"] for row in palette.rows]
    assert {"Start processing", "Stop processing", "Mute", "Run Auto Voice Setup", "Test my sound"} <= set(everything)
    assert {"Noise gate", "Noise suppression", "De-esser", "Equalizer", "Compressor", "Limiter"} <= set(everything)
    assert {preset.name for preset in BUILTIN_PRESETS.values()} <= set(everything)
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
    palette.open()
    spy = QSignalSpy(window.user_mute_checkbox_action.triggered)
    _select(palette, "Mute")
    palette.run()
    assert spy.count() == 1
    assert window.engine_session.view.user_muted
    assert not palette.shown
    palette.open()
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
    palette.open()
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
    palette.open()
    row = _select(palette, preset.name)
    assert row["detail"] == preset.description
    palette.run()
    assert spy.count() == 1
    assert window.configuration.identity.name == preset.name


def test_a_page_row_opens_the_page(quick_window) -> None:
    window = quick_window
    palette = _palette(window)
    palette.open()
    _select(palette, "Open Health")
    palette.run()
    assert window.shell_page.index == window.HEALTH_PAGE_INDEX


def test_a_disabled_action_shows_disabled_and_does_nothing(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    assert not window.stop_btn_action.isEnabled()
    spy = QSignalSpy(window.stop_btn_action.triggered)
    palette.open()
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
    assert row["value"] == window.quick_bridge.model["stages"][-1]["rows"][0]["proxy"].format_value(before - 0.1)
    assert row["detail"].startswith("Was ")
    # A draft is not an edit: the setting has not moved.
    assert window.limiter_state.get_settings()["ceiling_db"] == before
    _press(window, Qt.Key.Key_Escape, qapp)
    assert not palette.shown
    assert window.limiter_state.get_settings()["ceiling_db"] == before
    palette.open()
    _select(palette, "Limiter ceiling")
    assert palette.rows[palette.current]["draft"] is False


def test_enter_applies_a_numeric_draft_as_an_undoable_edit(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    before = window.configuration.current.compressor.threshold_db
    palette.open()
    _select(palette, "Compressor threshold")
    assert palette.adjust(1) and palette.adjust(1)
    assert window.configuration.current.compressor.threshold_db == before
    palette.run()
    assert not palette.shown
    assert window.configuration.current.compressor.threshold_db == before + 2
    window.undo_configuration()
    assert window.configuration.current.compressor.threshold_db == before
    window.redo_configuration()
    assert window.configuration.current.compressor.threshold_db == before + 2


def test_left_and_right_on_a_row_that_is_not_numeric_leave_it_alone(quick_window) -> None:
    palette = _palette(quick_window)
    palette.open()
    _select(palette, "Compressor")
    # Not handled, so the search field moves its cursor instead.
    assert palette.adjust(1) is False


def test_the_callers_line_is_on_exactly_the_rows_that_change_what_callers_hear(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    palette.open()
    qapp.processEvents()
    affecting = {"Start processing", "Stop processing", "Mute"}
    for row in palette.rows:
        heard = row["title"] in affecting or row["group"] in {"Stage", "Preset", "Setting"}
        assert row["callers"] == (CALLERS_LINE if heard else ""), row["title"]
        assert (CALLERS_LINE in row["description"]) is heard
    quiet = {row["title"] for row in palette.rows if not row["callers"]}
    assert {"Run Auto Voice Setup", "Test my sound", "Open Health"} <= quiet
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
    assert origin is not None
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
    # A window shortcut, not a global hotkey, that holding the keys does not repeat.
    action = _palette(window).action
    assert action.shortcut().toString() == "Ctrl+K"
    assert action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
    assert action.autoRepeat() is False and action in window.actions()


def test_enter_runs_the_selected_row_from_the_keyboard(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _press(window, Qt.Key.Key_K, qapp, Qt.KeyboardModifier.ControlModifier)
    _type(window, "open", qapp)
    assert _palette(window).rows[0]["title"] == "Open Overview"
    _press(window, Qt.Key.Key_Down, qapp)
    _press(window, Qt.Key.Key_Return, qapp)
    assert not _palette(window).shown
    assert window.shell_page.index == window.HEALTH_PAGE_INDEX


def test_the_search_field_and_rows_are_named_for_assistive_technology(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    palette = _palette(window)
    palette.open()
    _select(palette, "Limiter ceiling")
    palette.setQuery("")
    qapp.processEvents()
    with _AccessibleScene() as scene:
        assert scene.name(_scene(window, "paletteSearch")) == "Search commands"
        assert scene.name(_scene(window, "paletteList")) == "Commands"
        assert scene.name(_scene(window, "paletteGlass")) == "Command palette"
        for item, row in zip(_delegates(window), palette.rows, strict=False):
            expected = f"{row['title']}: {row['value']}" if row["value"] else row["title"]
            assert scene.name(item) == expected
            assert scene.description(item) == row["description"]
            assert row["detail"] in row["description"]


@pytest.mark.parametrize("motion", ["reduced", "full"])
def test_under_reduced_motion_the_palette_only_fades(request, monkeypatch, qapp, motion) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1" if motion == "reduced" else "0")
    window = request.getfixturevalue("quick_window")
    _show(window, qapp)
    theme = window.quick_bridge.model["theme"]
    overlay = _scene(window, "commandPalette")
    _palette(window).open()
    # Sampled while it fades in: half faded, the panel has finished moving only under reduced motion.
    samples = []
    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < theme["motionPanelOpen"] + 300:
        samples.append((overlay.property("opacity"), overlay.property("progress")))
        QTest.qWait(5)
    fading = [progress for opacity, progress in samples if 0.5 <= opacity < 1]
    assert fading
    if motion == "reduced":
        assert all(progress == 1 for progress in fading)
    else:
        assert any(progress < 1 for progress in fading)
    assert overlay.property("progress") == 1 and overlay.property("opacity") == 1
