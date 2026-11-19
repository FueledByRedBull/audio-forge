"""Hold to hear original on the Equalizer page, and the sentence under the selected band."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QAccessible, QKeyEvent
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

from mic_eq.config import EQBandSettings
from mic_eq.ui.engine_session import AudioBorrowed, RestartFailed
from mic_eq.ui.eq_band_hint import band_hint
from mic_eq.ui.eq_graph import EQGraphGeometry
from mic_eq.ui.eq_hold_compare import EQ_OFF_REASON, HOLD_TIP, PILL_TEXT, STOPPED_REASON
from mic_eq.ui.eq_state import EQState
from mic_eq.ui.main_window import MINIMUM_WINDOW_HEIGHT, MINIMUM_WINDOW_WIDTH
from mic_eq.ui.quick_shell import _scene_stopped
from test_quick_accessibility import (  # noqa: F401  (fixtures)
    EQUALIZER,
    _focus,
    _items,
    _with_proxy,
    _kind,
    _show,
    no_qml_warnings,
    quick_window,
    segoe_font,
    shell_window,
)
from test_quick_shell import _AccessibleScene, _graph_item, engine_window  # noqa: F401  (a fixture)

PILL = "Hearing your original voice · callers hear it too · release to return"


# The engine switch, without a scene


class Processor:
    def __init__(self) -> None:
        self.enabled: list[bool] = []
        self.refuse = False

    def set_eq_enabled(self, enabled: bool) -> None:
        if self.refuse:
            raise RuntimeError("engine refused")
        self.enabled.append(enabled)

    def apply_eq_layers(self, _correction, _tone) -> None:
        pass


@pytest.fixture
def state(qapp):
    return EQState(Processor())


def test_holding_turns_the_stage_off_in_the_engine_alone(state) -> None:
    edits: list[str] = []
    state.configurationEdited.connect(edits.append)
    before = state.get_eq_settings()
    assert state.hold_original()
    assert state.processor.enabled[-1] is False and state.original_held
    assert state.get_eq_settings() == before and state.get_eq_settings().enabled
    state.release_original()
    assert state.processor.enabled[-1] is True and not state.original_held
    assert edits == []


def test_an_equalizer_that_is_off_has_no_original_to_hold(state) -> None:
    state.set_enabled(False)
    writes = len(state.processor.enabled)
    assert not state.hold_original()
    assert len(state.processor.enabled) == writes


def test_any_write_ends_the_hold_and_the_engine_runs_what_was_written(state) -> None:
    state.hold_original()
    state.set_enabled(False)
    assert not state.original_held
    assert state.processor.enabled[-1] is False and not state.get_eq_settings().enabled
    # Releasing afterwards does not put the old switch back.
    state.release_original()
    assert state.processor.enabled[-1] is False


def test_a_refused_hold_is_reported_and_nothing_is_held(state) -> None:
    failures: list[str] = []
    state.writeFailed.connect(failures.append)
    state.processor.refuse = True
    assert not state.hold_original()
    assert not state.original_held and failures and "engine refused" in failures[0]


def test_a_refused_release_keeps_the_hold_showing_and_is_reported(state) -> None:
    failures: list[str] = []
    held: list[bool] = []
    state.writeFailed.connect(failures.append)
    state.hold_original()
    state.originalHeldChanged.connect(lambda: held.append(state.original_held))
    state.processor.refuse = True
    state.release_original()
    assert state.original_held and held == [] and failures and "engine refused" in failures[-1]
    state.processor.refuse = False
    state.release_original()
    assert not state.original_held and state.processor.enabled[-1] is True and held == [False]


def test_a_refused_write_during_the_hold_keeps_it_until_the_engine_takes_one(state) -> None:
    failures: list[str] = []
    state.writeFailed.connect(failures.append)
    state.hold_original()
    state.processor.refuse = True
    with pytest.raises(RuntimeError):
        state.set_enabled(False)
    assert state.original_held and failures
    assert state.get_eq_settings().enabled
    state.processor.refuse = False
    state.set_band_value(0, "gain_db", 2.0)
    state.flush()
    assert not state.original_held and state.processor.enabled[-1] is True


# The sentence under the band


@pytest.mark.parametrize(
    ("filter_type", "frequency_hz", "gain_db", "sentence"),
    [
        # Each region, at and just under its boundaries
        ("bell", 40.0, 3.0, "Bell at 40 Hz · boosted, so it adds rumble around here."),
        ("bell", 59.0, 3.0, "Bell at 59 Hz · boosted, so it adds rumble around here."),
        ("bell", 60.0, -3.0, "Bell at 60 Hz · cut, so it reduces warmth around here."),
        ("bell", 249.0, 2.0, "Bell at 249 Hz · boosted, so it adds warmth around here."),
        ("bell", 250.0, 2.0, "Bell at 250 Hz · boosted, so it adds boxiness around here."),
        ("bell", 1280.0, -1.5, "Bell at 1.28 kHz · cut, so it reduces body around here."),
        ("bell", 2000.0, 1.0, "Bell at 2 kHz · boosted, so it adds clarity around here."),
        ("high_shelf", 6000.0, 4.0, "High shelf at 6 kHz · boosted, so it adds air / hiss from here up."),
        ("high_shelf", 20000.0, 4.0, "High shelf at 20 kHz · boosted, so it adds air / hiss from here up."),
        ("low_shelf", 80.0, -3.0, "Low shelf at 80 Hz · cut, so it reduces warmth from here down."),
        # Flat is what the gain box shows as 0: under a twentieth of a dB either way
        ("high_shelf", 3000.0, 0.04,
         "High shelf at 3 kHz · flat, so it changes nothing yet; raise it for more clarity, lower it for less."),
        ("high_shelf", 3000.0, -0.04,
         "High shelf at 3 kHz · flat, so it changes nothing yet; raise it for more clarity, lower it for less."),
        ("high_shelf", 3000.0, -0.06, "High shelf at 3 kHz · cut, so it reduces clarity from here up."),
        # Pass filters speak of the region an octave past the corner; gain plays no part
        ("high_pass", 80.0, 6.0,
         "High-pass at 80 Hz · it removes everything below this frequency, starting with the rumble."),
        ("high_pass", 119.0, 0.0,
         "High-pass at 119 Hz · it removes everything below this frequency, starting with the rumble."),
        ("high_pass", 120.0, 0.0,
         "High-pass at 120 Hz · it removes everything below this frequency, starting with the warmth."),
        ("low_pass", 1000.0, 0.0,
         "Low-pass at 1 kHz · it removes everything above this frequency, starting with the clarity."),
        ("low_pass", 12000.0, -6.0,
         "Low-pass at 12 kHz · it removes everything above this frequency, starting with the air / hiss."),
        ("notch", 50.0, 0.0,
         "Notch at 50 Hz · it takes out a narrow slice of the rumble, for a hum or a ringing tone."),
        ("notch", 1000.0, 9.0,
         "Notch at 1 kHz · it takes out a narrow slice of the body, for a hum or a ringing tone."),
    ],
)
def test_the_sentence_names_the_band_and_what_it_does_there(filter_type, frequency_hz, gain_db, sentence) -> None:
    assert band_hint(filter_type, frequency_hz, gain_db) == sentence


def test_a_band_that_is_off_or_an_equalizer_that_is_off_says_so() -> None:
    assert band_hint("bell", 1000.0, 3.0, band_on=False) == "Band off · switch it on to hear it."
    assert band_hint("bell", 1000.0, 3.0, stage_on=False) == "The equalizer is off, so this band changes nothing."
    assert (
        band_hint("high_pass", 80.0, 0.0, band_on=False, stage_on=False)
        == "The equalizer is off, so this band changes nothing."
    )


# The scene


def _find(window, name: str) -> QQuickItem:
    item = window.centralWidget().rootObject().findChild(QObject, name)
    assert isinstance(item, QQuickItem), name
    return item


def _engine_on(window) -> bool:
    return window.eq_state.processor.is_eq_enabled()


def _compare(window):
    return window.quick_bridge.model["eq"]["compare"]


def _start(window, qapp) -> None:
    window.start_btn_action.trigger()
    for _ in range(200):
        if window._processing:
            break
        QTest.qWait(10)
    qapp.processEvents()
    assert window._processing and _compare(window).enabled


@pytest.fixture
def live(engine_window, qapp):  # noqa: F811
    window, _engine = engine_window
    _show(window, qapp, EQUALIZER)
    _start(window, qapp)
    assert _engine_on(window)
    return window


def _centre(item):
    return item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()


def _key(window, kind: QEvent.Type, key: Qt.Key, text: str, *, repeat: bool = False) -> None:
    QCoreApplication.sendEvent(
        window.centralWidget(), QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier, text, repeat, 1)
    )
    QCoreApplication.processEvents()


def _held(window) -> bool:
    held = _compare(window).held
    assert held == (not _engine_on(window)) == _find(window, "originalPill").isVisible()
    return held


def test_the_button_holds_while_the_mouse_is_down(live, qapp) -> None:
    window = live
    button = _find(window, "holdOriginal")
    view = window.centralWidget()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=_centre(button))
    qapp.processEvents()
    assert _held(window)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=_centre(button))
    qapp.processEvents()
    assert not _held(window)


def test_a_release_outside_the_button_still_returns(live, qapp) -> None:
    window = live
    button = _find(window, "holdOriginal")
    view = window.centralWidget()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=_centre(button))
    qapp.processEvents()
    assert _held(window)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=_centre(_find(window, "bandPanel")))
    qapp.processEvents()
    assert not _held(window)


def test_a_lost_mouse_grab_returns(live, qapp) -> None:
    window = live
    button = _find(window, "holdOriginal")
    QTest.mousePress(window.centralWidget(), Qt.MouseButton.LeftButton, pos=_centre(button))
    qapp.processEvents()
    assert _held(window)
    button.ungrabMouse()
    qapp.processEvents()
    assert not _held(window)


def test_space_on_the_focused_button_holds_and_ignores_auto_repeat(live, qapp) -> None:
    window = live
    _focus(window, _find(window, "holdOriginal"), qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_Space, " ")
    assert _held(window)
    for _ in range(3):
        _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_Space, " ", repeat=True)
        _key(window, QEvent.Type.KeyPress, Qt.Key.Key_Space, " ", repeat=True)
        assert _held(window)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_Space, " ")
    assert not _held(window)


def test_b_on_the_focused_graph_holds_and_ignores_auto_repeat(live, qapp) -> None:
    window = live
    graph = _graph_item(window.centralWidget().rootObject())
    _focus(window, graph, qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b")
    assert _held(window)
    for _ in range(3):
        _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b", repeat=True)
        _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b", repeat=True)
        assert _held(window)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert not _held(window)


def test_b_on_the_focused_button_holds_as_its_keycap_says(live, qapp) -> None:
    window = live
    _focus(window, _find(window, "holdOriginal"), qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b")
    assert _held(window)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b", repeat=True)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b", repeat=True)
    assert _held(window)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert not _held(window)


def test_b_is_no_window_shortcut(live, qapp) -> None:
    window = live
    view = window.centralWidget()
    tone = _with_proxy(window, "Btn", window.quick_bridge.model["eq"]["tone"])
    _focus(window, tone, qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b")
    assert not _held(window)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    for item in (_find(window, "holdOriginal"), _graph_item(view.rootObject())):
        _focus(window, item, qapp)
        QTest.keyPress(view, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
        qapp.processEvents()
        assert not _held(window)
        QTest.keyRelease(view, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)


def test_moving_the_focus_returns(live, qapp) -> None:
    window = live
    button = _find(window, "holdOriginal")
    graph = _graph_item(window.centralWidget().rootObject())
    _focus(window, button, qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_Space, " ")
    assert _held(window)
    _focus(window, graph, qapp)
    assert not _held(window)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b")
    assert _held(window)
    _focus(window, button, qapp)
    assert not _held(window)
    # The keys let go later find nothing to end, and the next hold works.
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_Space, " ")
    assert _held(window)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_Space, " ")
    assert not _held(window)


def _hold(window, qapp) -> None:
    _focus(window, _graph_item(window.centralWidget().rootObject()), qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b")
    assert _held(window)


def _after_hold_ends_elsewhere_the_next_hold_works(window, qapp) -> None:
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    _hold(window, qapp)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert not _held(window)


def test_the_window_losing_focus_returns(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    QCoreApplication.sendEvent(window, QEvent(QEvent.Type.WindowDeactivate))
    qapp.processEvents()
    assert not _held(window)
    _after_hold_ends_elsewhere_the_next_hold_works(window, qapp)


def test_hiding_the_window_returns(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    window.hide()
    qapp.processEvents()
    assert not _compare(window).held and _engine_on(window)
    window.show()
    qapp.processEvents()


def test_minimising_the_window_returns(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    window.showMinimized()
    qapp.processEvents()
    assert not _compare(window).held and _engine_on(window)
    window.showNormal()
    qapp.processEvents()


def test_stop_returns_and_leaves_the_button_disabled(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    window.stop_btn_action.trigger()
    for _ in range(200):
        if not window._processing:
            break
        QTest.qWait(10)
    qapp.processEvents()
    assert not _held(window)
    assert not _compare(window).enabled and not _find(window, "holdOriginal").isEnabled()


def test_a_lost_device_returns(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    window.engine_session._emit(RestartFailed("gone"))
    qapp.processEvents()
    assert not _held(window)
    _after_hold_ends_elsewhere_the_next_hold_works(window, qapp)
    _hold(window, qapp)
    window.shell_texts["callback_health_label"].update(text="Callbacks: stalled", status="bad")
    qapp.processEvents()
    assert not _held(window)
    assert not _compare(window).enabled


def test_voice_setup_or_test_my_sound_borrowing_the_audio_returns(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    # The dialog may never take the focus from the window; the borrow alone ends the hold.
    window.engine_session._emit(AudioBorrowed("voice setup", False))
    qapp.processEvents()
    assert not _held(window)
    _after_hold_ends_elsewhere_the_next_hold_works(window, qapp)


def test_a_dropped_scene_returns(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    _scene_stopped(window, "lost")
    qapp.processEvents()
    assert not window.eq_state.original_held and _engine_on(window)


def test_quitting_returns_before_anything_is_saved(live, qapp, monkeypatch) -> None:
    window = live
    _hold(window, qapp)
    seen: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        "mic_eq.ui.main_window.save_config",
        lambda _config: seen.append((_engine_on(window), window.configuration.current.eq.enabled)) or True,
    )
    window._shutdown()
    assert seen and seen[0] == (True, True)
    assert not window.eq_state.original_held


def test_closing_the_window_returns_on_its_way_through_shutdown(live, qapp, monkeypatch) -> None:
    window = live
    _hold(window, qapp)
    shutdown = window._shutdown
    seen: list[bool] = []

    def spy() -> bool:
        seen.append(window.eq_state.original_held)
        return shutdown()

    monkeypatch.setattr(window, "_shutdown", spy)
    window._quitting = True
    window.close()
    qapp.processEvents()
    assert seen and not window.eq_state.original_held and _engine_on(window)


def test_the_hold_leaves_no_trace_in_history_dirty_state_or_saves(live, qapp, monkeypatch) -> None:
    window = live
    configuration = window.configuration
    revision, count = configuration.history.revision, configuration.changes_since_saved
    modified, current = configuration.identity.modified, configuration.current.to_dict()
    saved: list = []
    monkeypatch.setattr(
        "mic_eq.ui.main_window.save_preset", lambda preset, **_kw: saved.append(preset) or Path("Held.json")
    )
    _hold(window, qapp)
    QTest.qWait(300)  # past the history debounce
    assert not configuration.commit_edit()
    assert configuration.history.revision == revision and configuration.changes_since_saved == count
    assert configuration.identity.modified == modified and configuration.current.to_dict() == current
    window.history_chip.refresh()
    assert window.history_chip.count == count
    preset = window._get_saveable_preset()
    assert preset is not None
    window._save_preset_file(preset, filepath=Path("Held.json"))
    assert saved and saved[0].eq.enabled
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert not _held(window)
    assert configuration.history.revision == revision


def test_an_edit_during_the_hold_ends_it_and_keeps_the_edit(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    window.eq_state.set_band_value(0, "gain_db", -4.0)
    window.eq_state.flush()
    qapp.processEvents()
    assert not _held(window) and window.eq_state.get_band(0).gain_db == pytest.approx(-4.0)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert _engine_on(window)


def test_switching_the_equalizer_off_during_the_hold_is_not_undone_by_the_release(live, qapp) -> None:
    window = live
    _hold(window, qapp)
    window.quick_bridge.model["eq"]["toggle"].click()
    qapp.processEvents()
    assert not _compare(window).held and not window.eq_state.get_eq_settings().enabled
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert not _engine_on(window) and not window.eq_state.get_eq_settings().enabled


def test_an_undo_or_a_reapplied_sound_during_the_hold_ends_it(live, qapp) -> None:
    window = live
    window.eq_state.set_band_value(1, "gain_db", 3.0)
    window.eq_state.flush()
    assert window.configuration.commit_edit()
    _hold(window, qapp)
    window.configuration.undo()
    qapp.processEvents()
    assert not _held(window) and window.eq_state.get_band(1).gain_db == pytest.approx(0.0)
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    # A load that writes the values already there is still a write the engine follows.
    _hold(window, qapp)
    window.eq_state.set_settings({"enabled": True})
    qapp.processEvents()
    assert not _held(window)


def test_disabled_while_stopped(quick_window, qapp) -> None:  # noqa: F811
    window = quick_window
    _show(window, qapp, EQUALIZER)
    compare = _compare(window)
    assert not compare.enabled and not _find(window, "holdOriginal").isEnabled()
    assert compare.toolTip == f"{STOPPED_REASON} {HOLD_TIP}"
    compare.press()
    assert not compare.held and _engine_on(window)


def test_disabled_while_the_equalizer_is_off(live, qapp) -> None:
    window = live
    window.eq_state.set_enabled(False)
    qapp.processEvents()
    compare = _compare(window)
    assert not compare.enabled and not _find(window, "holdOriginal").isEnabled()
    assert compare.toolTip == f"{EQ_OFF_REASON} {HOLD_TIP}"
    _focus(window, _graph_item(window.centralWidget().rootObject()), qapp)
    _key(window, QEvent.Type.KeyPress, Qt.Key.Key_B, "b")
    assert not compare.held
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")


def test_the_pill_says_in_words_what_is_playing(live, qapp) -> None:
    window = live
    pill = _find(window, "originalPill")
    assert PILL_TEXT == PILL and not pill.isVisible()
    _hold(window, qapp)
    assert pill.isVisible()
    words = next(item for item in _items(pill) if _kind(item) == "QQuickText")
    assert words.property("text") == PILL
    with _AccessibleScene() as scene:
        assert scene.name(pill) == PILL
        assert QAccessible.queryAccessibleInterface(pill).role() == QAccessible.Role.AlertMessage
    graph = _graph_item(window.centralWidget().rootObject())
    assert graph is not None and _rect(graph).contains(_rect(pill))
    # On the region strip, clear of the plot and of a handle at the top of its range.
    top = _rect(graph).top()
    assert _rect(pill).bottom() <= top + EQGraphGeometry.REGION_STRIP_HEIGHT
    highest = EQGraphGeometry.gain_to_y(EQGraphGeometry.GAIN_MAX_DB, graph.width(), graph.height())
    assert _rect(pill).bottom() < top + highest - EQGraphGeometry.SELECTED_HANDLE_RADIUS
    _key(window, QEvent.Type.KeyRelease, Qt.Key.Key_B, "b")
    assert not pill.isVisible()


def test_the_button_is_named_described_and_shows_the_b_keycap(live, qapp) -> None:
    window = live
    button = _find(window, "holdOriginal")
    with _AccessibleScene() as scene:
        assert scene.name(button) == "Hold to hear original"
        description = scene.description(button)
        assert "Hold this button" in description
        assert "hold B while the equalizer graph has focus" in description
        assert "Callers hear the original too" in description
        assert QAccessible.queryAccessibleInterface(button).role() == QAccessible.Role.Button
    assert _find(window, "holdKeycap").property("text") == "B"
    assert button.property("text") == "Hold to hear original"


def test_the_band_panel_describes_the_selected_band_and_follows_it(quick_window, qapp) -> None:  # noqa: F811
    window = quick_window
    _show(window, qapp, EQUALIZER)
    hint = _find(window, "bandHint")

    def expected(index: int) -> str:
        band = window.eq_state.get_band(index)
        return band_hint(band.filter_type, band.frequency_hz, band.gain_db, band_on=band.enabled)

    window.eq_state.select_band(2)
    qapp.processEvents()
    assert hint.isVisible() and hint.property("text") == expected(2)
    window.eq_state.set_band_value(2, "gain_db", -5.0)
    window.eq_state.flush()
    qapp.processEvents()
    assert "cut, so it reduces" in hint.property("text") and hint.property("text") == expected(2)
    window.eq_state.set_band(2, EQBandSettings("high_pass", 90.0, 0.0, 0.707))
    window.eq_state.flush()
    qapp.processEvents()
    assert hint.property("text") == "High-pass at 90 Hz · it removes everything below this frequency, starting with the rumble."
    window.eq_state.set_band_value(2, "enabled", False)
    window.eq_state.flush()
    qapp.processEvents()
    assert hint.property("text") == "Band off · switch it on to hear it."
    window.eq_state.set_band_value(2, "enabled", True)
    window.eq_state.set_enabled(False)
    window.eq_state.flush()
    qapp.processEvents()
    assert hint.property("text") == "The equalizer is off, so this band changes nothing."
    with _AccessibleScene() as scene:
        assert scene.name(hint) == hint.property("text")


def _rect(item):
    return item.mapRectToScene(item.boundingRect())


# Fonts on other platforms run wider than here; 60 px under the minimum stands in for them.
@pytest.mark.parametrize("width", [MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_WIDTH - 60], ids=["minimum", "wider-fonts"])
def test_the_header_and_the_band_sentence_keep_clear_at_the_minimum_width(
    segoe_font, quick_window, qapp, width  # noqa: F811
) -> None:
    window = quick_window
    window.setMinimumSize(width, MINIMUM_WINDOW_HEIGHT)
    window.resize(width, MINIMUM_WINDOW_HEIGHT)
    window.shell_page.setIndex(EQUALIZER)
    window.eq_state.set_band_value(9, "frequency_hz", 15000.5)
    window.eq_state.flush()
    assert window.configuration.commit_edit()
    window.history_chip.refresh()
    window.eq_state.select_band(9)
    qapp.processEvents()
    right = window.centralWidget().width()
    header = [item for item in _find(window, "holdOriginal").parentItem().childItems() if item.isVisible()]
    assert _find(window, "undoHistory") in header
    for index, item in enumerate(header):
        assert _rect(item).right() <= right
        for other in header[index + 1:]:
            assert not _rect(item).intersects(_rect(other)), (_kind(item), _kind(other))
    hold = _find(window, "holdOriginal")
    assert _rect(_find(window, "holdKeycap")).right() <= _rect(hold).right()
    # Short of room it says Hold beside its keycap, whole, never a cut-off letter or two.
    label = hold.property("contentItem")
    assert label.property("text") in ("Hold", "Hold to hear original") and not label.property("truncated")
    hint = _find(window, "bandHint")
    panel = _find(window, "bandPanel")
    assert panel.mapRectToScene(panel.boundingRect()).contains(_rect(hint))
    for sibling in hint.parentItem().childItems():
        if sibling is not hint and sibling.isVisible():
            assert not _rect(hint).intersects(_rect(sibling)), _kind(sibling)
    assert hint.property("lineCount") >= 1 and not hint.property("truncated")
