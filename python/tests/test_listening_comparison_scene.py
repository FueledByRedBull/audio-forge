"""The listening comparison dialogs are a Qt Quick scene: it loads, names its controls and takes the keyboard."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt, QTimer, qInstallMessageHandler
from PySide6.QtGui import QAccessible, QFont
from PySide6.QtMultimedia import QAudioDevice, QMediaDevices
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from mic_eq.analysis.listening_comparison import ComparisonRenderResult, RenderedComparisonClip
from mic_eq.ui.capture_session import LIVE_OUTPUT_MUTED_NOTICE
from mic_eq.ui import listening_comparison_dialog as module
from mic_eq.ui.listening_comparison_dialog import ListeningComparisonDialog, RawProcessedDialog
from mic_eq.ui.listening_comparison_view import ListeningComparisonView
from mic_eq.ui.theme import PALETTE
from test_first_run_setup import at_scene_minimum, choose, cut_short_texts, press, scene_item, scene_items, scene_text
from test_listening_comparison import _settings
from test_quick_shell import _AccessibleScene, _wait_until
from test_raw_processed_listening import _Processor, _Window

REJECTED, ACCEPTED = int(QDialog.DialogCode.Rejected), int(QDialog.DialogCode.Accepted)
PLAYS = {False: ["originalPlayButton", "currentPlayButton", "proposedPlayButton"],
         True: ["originalPlayButton", "currentPlayButton"]}
LONG_STATUS = "Offline comparison: current normal processing; proposed normal processing. " * 3


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


def _result() -> ComparisonRenderResult:
    def clip(label: str) -> RenderedComparisonClip:
        return RenderedComparisonClip(
            label=label, samples=np.full(480, 0.1, dtype=np.float32), actual_level_db=-20.0,
            actual_level_delta_db=0.0, playback_level_db=-20.0, peak_db=-20.0, level_match_gain_db=0.0,
            safety_gain_db=0.0, simulation_backend="rust", rendered_stages=("tone EQ",),
        )

    return ComparisonRenderResult(
        sample_rate=48_000, original=clip("original"), current=clip("current"), proposed=clip("proposed"),
        level_match=False, level_match_reference_db=-20.0, speech_detection="test", alignment_samples=0,
        alignment_ms=0.0, native_authoritative=True,
    )


@pytest.fixture
def open_dialog(qapp, monkeypatch):
    """Open a dialog with no render thread; `devices` null outputs stand in for real ones."""
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    made: list[ListeningComparisonDialog] = []

    def make(*, test_my_sound: bool = False, devices: int = 1, ready: bool = False) -> Any:
        monkeypatch.setattr(QMediaDevices, "audioOutputs", lambda: [QAudioDevice() for _ in range(devices)])
        capture = {"audio_data": np.zeros(480, dtype=np.float32), "sample_rate": 48_000}
        dialog = (
            RawProcessedDialog(settings=_settings(), **capture)
            if test_my_sound
            else ListeningComparisonDialog(current_settings=_settings(), proposed_settings=_settings(), **capture)
        )
        made.append(dialog)
        if devices:
            dialog.view_model.pickOutput(1)
        if ready:
            dialog._on_render_ready(_result())
        dialog.show()
        qapp.processEvents()
        dialog.activateWindow()
        dialog.quick_view.setFocus()
        qapp.processEvents()
        return dialog

    yield make
    for dialog in made:
        dialog.hide()
        dialog.deleteLater()
    qapp.processEvents()


def _focus_name(dialog: Any) -> str:
    return dialog.quick_view.quickWindow().activeFocusItem().objectName()


def _tab_walk(dialog: Any) -> list[str]:
    visited: list[str] = []
    for _ in range(20):
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Tab)
        if _focus_name(dialog) in visited:
            break
        visited.append(_focus_name(dialog))
    return visited


def _within_page(dialog: Any, item: Any) -> bool:
    scroll = scene_item(dialog, "contentScroll")
    top = item.mapToItem(scroll, QPointF(0, 0)).y()
    return top >= 0 and top + item.height() <= scroll.height()


@pytest.mark.parametrize("test_my_sound", [False, True])
def test_the_scene_loads_and_every_control_has_its_accessible_name(open_dialog, test_my_sound) -> None:
    dialog = open_dialog(test_my_sound=test_my_sound, ready=True)
    assert dialog.quick_view.status() == QQuickWidget.Status.Ready and dialog.quick_view.errors() == []
    title = "Test my sound" if test_my_sound else "Listening comparison"
    names = {
        "contentScroll": f"{title} page",
        "outputPicker": "Output device",
        "refreshButton": "Refresh playback devices",
        "stopButton": "Stop playback",
        "levelMatchSwitch": "Match speech level (actual level difference remains visible)",
        "progressBar": "Listening comparison progress bar",
        "rejectButton": "Close test my sound" if test_my_sound else "Reject proposed settings",
        "mutedNotice": LIVE_OUTPUT_MUTED_NOTICE,
        **{key: scene_text(dialog, key) for key in PLAYS[test_my_sound]},
    }
    if not test_my_sound:
        names["keepButton"] = "Keep proposed settings"
    with _AccessibleScene() as scene:
        assert {key: scene.name(scene_item(dialog, key)) for key in names} == names
        assert QAccessible.queryAccessibleInterface(dialog.quick_view).text(QAccessible.Text.Name) == f"{title} content"
        assert scene.description(scene_item(dialog, "refreshButton")) == "Refresh available playback devices."
        switch = QAccessible.queryAccessibleInterface(scene_item(dialog, "levelMatchSwitch"))
        assert switch.role() == QAccessible.Role.CheckBox and switch.state().checkable
    labels = ["Play Raw", "Play Processed"] if test_my_sound else [
        "Play Original recording", "Play Current settings", "Play Proposed settings"]
    assert [scene_text(dialog, key) for key in PLAYS[test_my_sound]] == labels
    shown = {i.objectName() for i in scene_items(dialog.quick_view.rootObject())}
    assert ("proposedPlayButton" in shown) is not test_my_sound
    assert scene_item(dialog, "keepButton").isVisible() is not test_my_sound
    assert scene_text(dialog, "rejectButton") == ("Close" if test_my_sound else "Reject Proposed")


def test_a_text_that_changes_is_read_with_its_new_text(open_dialog) -> None:
    dialog = open_dialog(devices=0)
    with _AccessibleScene() as scene:
        assert "No playback device is available" in scene.name(scene_item(dialog, "statusText"))
        assert scene_item(dialog, "statusText").property("color").name() == PALETTE.status_warn_text
        dialog.view_model.set_status("Moving closer helps")
        dialog._on_render_progress("Rendering current and proposed settings...", 15)
        assert scene.name(scene_item(dialog, "statusText")) == "Moving closer helps"
        assert scene.name(scene_item(dialog, "progressText")) == "Rendering current and proposed settings..."
        assert scene.description(scene_item(dialog, "progressBar")) == "15 percent"
    assert scene_item(dialog, "mutedNotice").isVisible()


@pytest.mark.parametrize("test_my_sound", [False, True])
def test_tab_visits_the_page_then_reject_and_keep_left_to_right(open_dialog, test_my_sound) -> None:
    dialog = open_dialog(test_my_sound=test_my_sound, ready=True)
    actions = ["rejectButton"] if test_my_sound else ["rejectButton", "keepButton"]
    page = ["contentScroll", "outputPicker", "refreshButton", "levelMatchSwitch", *PLAYS[test_my_sound]]
    assert _tab_walk(dialog) == [*page, *actions]
    # The actions stay out of the scrolling page, Keep on the right.
    scroll = scene_item(dialog, "contentScroll")
    for key in actions:
        assert not scroll.isAncestorOf(scene_item(dialog, key)), key
    xs = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).x() for key in actions]
    assert xs == sorted(xs)


def test_enter_presses_the_focused_button_and_escape_closes_the_dialog(open_dialog, qapp) -> None:
    dialog = open_dialog(ready=True)
    scene_item(dialog, "rejectButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert not dialog.isVisible() and dialog.result() == REJECTED and dialog.decision is False

    again = open_dialog(ready=True)
    QTest.keyClick(again.quick_view, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not again.isVisible() and again.result() == REJECTED and again.decision is False


@pytest.mark.parametrize("test_my_sound", [False, True])
def test_enter_outside_a_button_keeps_the_proposal_once_it_can_be_kept(open_dialog, qapp, test_my_sound) -> None:
    dialog = open_dialog(test_my_sound=test_my_sound)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    assert dialog.isVisible(), "nothing is rendered to keep yet"
    dialog._on_render_ready(_result())
    scene_item(dialog, "contentScroll").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Enter)
    qapp.processEvents()
    assert dialog.isVisible() is test_my_sound, "Test my sound has nothing to keep"
    if not test_my_sound:
        assert dialog.result() == ACCEPTED and dialog.decision is True


@pytest.mark.parametrize(
    ("button", "command", "args"),
    [
        ("refreshButton", "_refresh_playback_devices", ()),
        ("stopButton", "_stop_playback", ()),
        ("levelMatchSwitch", "_on_level_match_changed", (True,)),
        ("originalPlayButton", "_play_clip", ("original",)),
        ("proposedPlayButton", "_play_clip", ("proposed",)),
        ("keepButton", "_request_close", (True,)),
        ("rejectButton", "_request_close", (False,)),
    ],
)
def test_space_and_a_click_on_each_control_run_the_dialogs_command(
    open_dialog, monkeypatch, qapp, button, command, args
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(ListeningComparisonDialog, command, lambda self, *a: calls.append(a))
    dialog = open_dialog(ready=True)
    dialog.view_model.set_page(stopEnabled=True)
    calls.clear()
    item = scene_item(dialog, button)
    item.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    assert calls == [args]
    if button == "levelMatchSwitch":
        args = (False,)
        assert dialog.view_model.level_match is True and item.property("checked") is True
    press(dialog, button)
    qapp.processEvents()
    assert calls[1:] == [args]


def test_a_click_on_the_level_match_caption_flips_the_switch(open_dialog, qapp) -> None:
    dialog = open_dialog(ready=True)
    press(dialog, "levelMatchText")
    qapp.processEvents()
    assert dialog.view_model.level_match and scene_item(dialog, "levelMatchSwitch").property("checked")
    assert dialog.view_model.status_text == "Speech level matching applied to the next preview playback."


def test_the_output_picker_chooses_a_device_from_the_keyboard_and_follows_a_refresh(
    open_dialog, monkeypatch, qapp
) -> None:
    dialog = open_dialog(devices=2, ready=True)
    picker = scene_item(dialog, "outputPicker")
    dialog.view_model.pickOutput(0)
    assert not dialog.view_model.shown["playEnabled"], "the prompt is not a device"
    choose(dialog, "outputPicker", 2)
    assert dialog.view_model.output_index == 2 and dialog.view_model.shown["playEnabled"]
    monkeypatch.setattr(QMediaDevices, "audioOutputs", lambda: [QAudioDevice()])
    dialog._on_audio_outputs_changed()
    qapp.processEvents()
    assert dialog.view_model.output_index == 1
    assert (picker.property("currentIndex"), picker.property("count")) == (1, 2)


@pytest.mark.parametrize("large", [False, True])
def test_nothing_is_cut_short_or_runs_off_a_small_dialog(open_dialog, qapp, large) -> None:
    previous = qapp.font()
    if large:
        bigger = QFont(previous)
        bigger.setPointSize(14)
        qapp.setFont(bigger)
    try:
        dialog = open_dialog(ready=True)
        dialog.view_model.set_status(LONG_STATUS, "warn")
        assert dialog.minimumWidth() >= min(
            dialog.quick_view.rootObject().property("minimumWidth"), dialog.screen().availableGeometry().width() - 48
        )
        # Judged at the scene's own minimum, which a small or font-less screen cannot hold.
        at_scene_minimum(dialog, qapp)
        assert cut_short_texts(dialog) == []
        scroll, column = scene_item(dialog, "contentScroll"), scene_item(dialog, "contentColumn")
        right = scroll.mapToScene(QPointF(scroll.width(), 0)).x()
        assert [
            item.objectName() for item in scene_items(column, only_visible=True)
            if item.mapToScene(QPointF(item.width(), 0)).x() > right + 0.5
        ] == []
        assert scroll.property("contentHeight") > scroll.height()
        keep = scene_item(dialog, "keepButton")
        assert keep.mapToScene(QPointF(keep.width(), 0)).x() <= dialog.quick_view.width()
    finally:
        qapp.setFont(previous)


def test_the_keyboard_alone_scrolls_a_small_dialog_down_to_its_status_line(open_dialog, qapp) -> None:
    dialog = open_dialog(ready=True)
    dialog.resize(480, 360)
    qapp.processEvents()
    qapp.processEvents()
    scroll, status = scene_item(dialog, "contentScroll"), scene_item(dialog, "statusText")
    assert not _within_page(dialog, status)
    scroll.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    assert 0 < scroll.property("contentY") < scroll.height()
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_End)
    qapp.processEvents()
    assert scroll.property("contentY") == scroll.property("contentHeight") - scroll.height()
    assert _within_page(dialog, status) and _within_page(dialog, scene_item(dialog, "mutedNotice"))
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Home)
    assert scroll.property("contentY") == 0
    assert scene_item(dialog, "scrollFocusRing").isVisible()


def test_the_view_model_reports_changes_once_and_forwards_its_commands(qapp) -> None:
    calls: list[object] = []
    model = ListeningComparisonView(
        clips=(("original", "Raw"),),
        pick_output=lambda: calls.append("pick"),
        refresh=lambda: calls.append("refresh"),
        stop=lambda: calls.append("stop"),
        level_match=calls.append,
        play=calls.append,
        keep=lambda: calls.append("keep"),
        reject=lambda: calls.append("reject"),
    )
    changes: list[str] = []
    model.statusChanged.connect(lambda: changes.append("status"))
    model.outputsChanged.connect(lambda: changes.append("outputs"))
    model.set_status("Playing: Raw", "info")
    model.set_status("Playing: Raw", "info")
    model.set_status("Playback stopped.")
    assert (model.status_text, model.status_state) == ("Playback stopped.", "info"), "the colour is kept"
    with pytest.raises(ValueError, match="unknown state 'success'"):
        model.set_status("Odd", "success")
    assert model.status_state == "info", "a refused state changes nothing"
    model.set_outputs(["Choose", "Speakers"], 0)
    model.pickOutput(0)
    model.pickOutput(1)
    assert changes == ["status", "status", "outputs", "outputs"] and model.output_index == 1
    model.requestLevelMatch(True)
    model.play("original")
    model.refresh()
    model.stop()
    model.keep()
    model.reject()
    assert calls == ["pick", True, "original", "refresh", "stop", "keep", "reject"] and model.level_match


@pytest.mark.parametrize("test_my_sound", [False, True])
def test_a_scene_that_does_not_load_says_so_and_closes_without_rendering(
    qapp, monkeypatch, test_my_sound
) -> None:
    connected: list[object] = []
    rendered: list[object] = []
    shown: list[str] = []
    quitting = SimpleNamespace(aboutToQuit=SimpleNamespace(connect=connected.append))
    module = "mic_eq.ui.listening_comparison_dialog"
    monkeypatch.setattr(f"{module}.QApplication", SimpleNamespace(instance=lambda: quitting))
    monkeypatch.setattr(f"{module}.load_scene", lambda *_args: "no such file")
    monkeypatch.setattr(f"{module}.QMessageBox.critical", lambda _parent, title, text: shown.append(f"{title}: {text}"))
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: rendered.append(self))
    capture = {"audio_data": np.zeros(480, dtype=np.float32), "sample_rate": 48_000}
    dialog = (
        RawProcessedDialog(settings=_settings(), **capture)
        if test_my_sound
        else ListeningComparisonDialog(current_settings=_settings(), proposed_settings=_settings(), **capture)
    )
    assert dialog.load_error == "no such file"
    assert dialog.exec() == REJECTED and dialog.decision is False
    title = "Test my sound" if test_my_sound else "Before/After Listening Comparison"
    assert len(shown) == 1 and shown[0].startswith(f"{title}: ") and "no such file" in shown[0]
    assert rendered == [] and connected == []
    dialog.deleteLater()
    qapp.processEvents()


def _while_recording(qapp: Any, action: Any) -> tuple[Any, _Window]:
    """Record a Test my sound clip that stays at half way, and run `action(scene)` on its window."""
    window = _Window(_Processor())
    window.processor.progress = 0.5

    def act() -> None:
        dialog: Any = QApplication.activeModalWidget()
        quick = dialog.findChild(QQuickWidget)
        quick.setFocus()
        qapp.processEvents()
        action(SimpleNamespace(quick_view=quick, dialog=dialog))

    QTimer.singleShot(100, act)
    return _record_with_lease(window), window


def _record_with_lease(window: _Window) -> Any:
    """Record the way Test my sound does: borrow the audio, record, release it."""
    lease = window.engine_session.borrow(module._TEST_MY_SOUND_REASON)
    try:
        return module._record_test_clip(window, lease)
    finally:
        lease.release()


def test_the_recording_window_loads_names_its_parts_and_shows_the_progress(qapp) -> None:
    seen: dict[str, Any] = {}

    def look(scene: Any) -> None:
        assert scene.quick_view.status() == QQuickWidget.Status.Ready
        assert _wait_until(lambda: scene_item(scene, "progressBar").property("fraction") > 0)
        seen["focus"] = _focus_name(scene)
        with _AccessibleScene() as access:
            seen["names"] = {key: access.name(scene_item(scene, key)) for key in
                             ("promptText", "mutedNotice", "progressBar", "cancelButton")}
            seen["progress"] = access.description(scene_item(scene, "progressBar"))
            seen["view"] = QAccessible.queryAccessibleInterface(scene.quick_view).text(QAccessible.Text.Name)
        seen["fraction"] = scene_item(scene, "progressBar").property("fraction")
        scene.dialog.reject()

    audio, window = _while_recording(qapp, look)
    assert audio is None
    assert seen["names"] == {
        "promptText": "Recording. Speak normally for 5 seconds.",
        "mutedNotice": LIVE_OUTPUT_MUTED_NOTICE,
        "progressBar": "Recording progress",
        "cancelButton": "Cancel",
    }
    assert seen["view"] == "Test my sound recording" and seen["focus"] == "cancelButton"
    assert (seen["progress"], seen["fraction"]) == ("50 percent", 0.5)
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")], "the capture is released"


@pytest.mark.parametrize("how", ["space", "return", "enter", "escape", "click"])
def test_the_recording_window_cancels_by_keyboard_or_mouse(qapp, how) -> None:
    keys = {"space": Qt.Key.Key_Space, "return": Qt.Key.Key_Return, "enter": Qt.Key.Key_Enter,
            "escape": Qt.Key.Key_Escape}

    def cancel(scene: Any) -> None:
        if how == "click":
            press(scene, "cancelButton")
        else:
            QTest.keyClick(scene.quick_view, keys[how])

    audio, window = _while_recording(qapp, cancel)
    assert audio is None and QApplication.activeModalWidget() is None
    assert window.processor.calls == ["record 5 before_cleanup=True", "stop_raw_recording"]
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")]


def test_the_recording_window_fits_its_text_at_a_large_size(qapp) -> None:
    previous = qapp.font()
    bigger = QFont(previous)
    bigger.setPointSize(14)
    qapp.setFont(bigger)
    seen: dict[str, Any] = {}

    def measure(scene: Any) -> None:
        root = scene.quick_view.rootObject()
        seen["cut"] = cut_short_texts(scene)
        seen["wide"] = [i.objectName() for i in scene_items(root, only_visible=True)
                        if i.mapToScene(QPointF(i.width(), 0)).x() > scene.quick_view.width() + 0.5]
        seen["tall"] = root.property("implicitHeight") > scene.quick_view.height() + 0.5
        scene.dialog.reject()

    try:
        _while_recording(qapp, measure)
    finally:
        qapp.setFont(previous)
    assert seen == {"cut": [], "wide": [], "tall": False}


def test_a_recording_window_that_does_not_load_releases_the_capture(qapp, monkeypatch) -> None:
    monkeypatch.setattr(module, "load_scene", lambda *_args: "no such file")
    window = _Window(_Processor())
    with pytest.raises(RuntimeError, match="The recording window could not open: no such file"):
        _record_with_lease(window)
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")]
    assert QApplication.activeModalWidget() is None


def test_a_recording_window_that_does_not_load_is_reported_and_stops_the_audio_it_started(
    qapp, monkeypatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(module, "load_scene", lambda *_args: "no such file")
    monkeypatch.setattr(module.QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text))
    processor = _Processor(running=False)
    window = _Window(processor)
    module.open_test_my_sound(window)
    assert len(warnings) == 1
    assert "Recording did not work" in warnings[0]
    assert "The recording window could not open: no such file" in warnings[0]
    assert processor.calls == ["start", "record 5 before_cleanup=True", "stop_raw_recording", "stop"]
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")]
    assert QApplication.activeModalWidget() is None


@pytest.mark.parametrize("test_my_sound", [False, True])
def test_the_playback_and_listen_cards_are_named_groups(open_dialog, test_my_sound) -> None:
    dialog = open_dialog(test_my_sound=test_my_sound, ready=True)
    with _AccessibleScene() as scene:
        for key, title in (("playbackCard", "Playback"), ("listenCard", "Listen")):
            card = scene_item(dialog, key)
            assert QAccessible.queryAccessibleInterface(card).role() == QAccessible.Role.Grouping, key
            assert scene.name(card) == title


@pytest.mark.parametrize("test_my_sound", [False, True])
def test_the_stop_button_hands_the_keyboard_on_when_it_disables_itself(
    open_dialog, qapp, test_my_sound
) -> None:
    dialog = open_dialog(test_my_sound=test_my_sound, ready=True)
    dialog.view_model.set_page(stopEnabled=True)
    scene_item(dialog, "stopButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    qapp.processEvents()
    assert dialog.view_model.shown["stopEnabled"] is False
    assert _focus_name(dialog) == "levelMatchSwitch"
    # Playback that ends by itself does the same; a disabled Stop that is not focused moves nothing.
    dialog.view_model.set_page(stopEnabled=True)
    scene_item(dialog, "refreshButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    dialog.view_model.set_page(stopEnabled=False)
    assert _focus_name(dialog) == "refreshButton"
    dialog.view_model.set_page(stopEnabled=True)
    scene_item(dialog, "stopButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    dialog.view_model.set_page(stopEnabled=False)
    assert _focus_name(dialog) == "levelMatchSwitch"


def test_enter_on_the_closed_output_picker_keeps_the_proposal_but_in_its_open_list_it_chooses(
    open_dialog, qapp
) -> None:
    dialog = open_dialog(devices=2, ready=True)
    picker = scene_item(dialog, "outputPicker")
    picker.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    qapp.processEvents()
    assert picker.property("down") is True, "the list is open"
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert picker.property("down") is False
    assert dialog.isVisible() and dialog.decision is None, "Enter in the open list only chose"
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert dialog.result() == ACCEPTED and dialog.decision is True


def test_the_output_picker_is_read_with_the_device_it_shows(open_dialog) -> None:
    dialog = open_dialog(ready=True)
    dialog.view_model.set_outputs(["Headphones", "Speakers"], 1)
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "outputPicker")) == "Output device: Speakers"


def test_the_view_model_refuses_a_field_the_page_does_not_have() -> None:
    def ignore(*_args: object) -> None:
        return None

    view = ListeningComparisonView(
        clips=(("raw", "Raw"),), pick_output=ignore, refresh=ignore, stop=ignore,
        level_match=ignore, play=ignore, keep=ignore, reject=ignore,
    )
    with pytest.raises(KeyError, match="keepEnabld"):
        view.set_page(keepEnabld=True)
    assert "keepEnabld" not in view.shown
    view.set_page(keepEnabled=True)
    assert view.shown["keepEnabled"] is True
