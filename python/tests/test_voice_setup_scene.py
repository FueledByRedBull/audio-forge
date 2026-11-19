"""The Voice Setup dialog is a Qt Quick scene: it loads, names its controls and takes the keyboard."""
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
from PySide6.QtWidgets import QDialog

from capture_helpers import finish_take
from mic_eq.config import TARGET_CURVES, EQSettings, build_eq_candidate_settings
from mic_eq.ui import voice_setup_dialog
from mic_eq.ui.calibration_support import RAINBOW_PASSAGE
from mic_eq.ui.capture_session import LIVE_OUTPUT_MUTED_NOTICE
from mic_eq.ui.theme import PALETTE
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from mic_eq.ui.voice_setup_view import DYNAMICS, DYNAMICS_KEYS, VoiceSetupView
from test_auto_eq_candidate_lifecycle import _Owner
from test_first_run_setup import at_scene_minimum, choose, cut_short_texts, press, scene_item, scene_items, scene_text
from test_quick_shell import _AccessibleScene
from test_setup_level_check_flow import HOT, _Session

REJECTED = int(QDialog.DialogCode.Rejected)
EVERYTHING = {
    "recordingShown": True, "meterShown": True, "reviewShown": True, "summaryShown": True,
    "eqCompareShown": True, "retakeShown": True, "compareShown": True, "compareEnabled": True,
    "overrideShown": True, "mutedShown": True, "noiseShown": True, "voiceShown": True,
}
BUTTONS = ["cancelButton", "retakeButton", "compareButton", "overrideButton", "startButton"]
FIELDS = ["curvePicker", "dynamicsPicker", "p95Number", "peakNumber", "lufsNumber"]
LONG_WARNING = "Voice Setup could not confirm the settings are safe, so they are not offered. " * 4
FIRST_CURVE = next(iter(TARGET_CURVES.values()))


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
def open_dialog(qapp):
    made: list[tuple[VoiceSetupDialog, _Owner]] = []

    def make(*, everything: bool = False, custom: bool = False) -> Any:
        owner = _Owner()
        owner.config = SimpleNamespace(  # type: ignore[attr-defined]
            voice_setup_dynamics_intensity="custom" if custom else "balanced",
            voice_setup_custom_p95_db=3.5,
            voice_setup_custom_peak_cap_db=8.0,
        )
        dialog = VoiceSetupDialog(owner, configuration=owner.configuration)
        made.append((dialog, owner))
        if everything:
            dialog.view_model.set_page(**EVERYTHING)
        dialog.show()
        qapp.processEvents()
        dialog.activateWindow()
        dialog.quick_view.setFocus()
        qapp.processEvents()
        return dialog

    yield make
    for dialog, owner in made:
        dialog.hide()
        dialog.deleteLater()
        owner.close()
    qapp.processEvents()


def _focus_name(dialog: Any) -> str:
    return dialog.quick_view.quickWindow().activeFocusItem().objectName()


def _tab_walk(dialog: Any) -> list[str]:
    visited: list[str] = []
    for _ in range(24):
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Tab)
        if _focus_name(dialog) in visited:
            break
        visited.append(_focus_name(dialog))
    return visited


def _within_page(dialog: Any, item: Any) -> bool:
    scroll = scene_item(dialog, "contentScroll")
    top = item.mapToItem(scroll, QPointF(0, 0)).y()
    return top >= 0 and top + item.height() <= scroll.height()


def test_an_opened_dialog_gives_the_keyboard_to_the_scene(qapp) -> None:
    owner = _Owner()
    dialog = VoiceSetupDialog(owner, configuration=owner.configuration)
    try:
        dialog.show()
        qapp.processEvents()
        dialog.activateWindow()
        qapp.processEvents()
        assert dialog.focusWidget() is dialog.quick_view  # no click needed first
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Tab)
        assert _focus_name(dialog) == "contentScroll"
    finally:
        dialog.hide()
        dialog.deleteLater()
        owner.close()
        qapp.processEvents()


def test_the_scene_loads_and_every_control_has_its_accessible_name(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    assert dialog.quick_view.status() == QQuickWidget.Status.Ready and dialog.quick_view.errors() == []
    names = {
        "stepHeader": "Step 1 of 5: Target",
        "contentScroll": "Auto Voice Setup page",
        "curvePicker": f"Tone preset: {FIRST_CURVE.name}",
        "dynamicsPicker": "Compression: Balanced",
        "p95Number": "Typical compressor evening-out in decibels: 3.50 dB",
        "peakNumber": "Strongest compressor squeeze allowed in decibels: 8.00 dB",
        "lufsNumber": "Voice setup target loudness: -18.00 LUFS",
        "mutedNotice": LIVE_OUTPUT_MUTED_NOTICE,
        "progressBar": "Voice Setup progress",
        "levelMeter": "Voice Setup input level",
        "eqCompare": "EQ now compared with the proposed EQ",
        "cancelButton": "Cancel Auto Voice Setup",
        "retakeButton": "Retake Voice Setup recording",
        "compareButton": "Compare Recording",
        "overrideButton": "Continue Anyway",
        "startButton": "Start Voice Setup",
        "curveCard": "Tone preset",
        "dynamicsCard": "Dynamics",
        "noiseCard": "Capture room noise",
        "voiceCard": "Read this passage aloud",
        "recordingCard": "Recording",
        "reviewCard": "Recommended Settings",
    }
    with _AccessibleScene() as scene:
        assert {key: scene.name(scene_item(dialog, key)) for key in names} == names
        assert QAccessible.queryAccessibleInterface(dialog.quick_view).text(QAccessible.Text.Name) == (
            "Auto Voice Setup content"
        )
        assert scene.description(scene_item(dialog, "lufsNumber")) == (
            "Loudness target is independent of the selected tone curve."
        )
        assert scene.description(scene_item(dialog, "passageText")) == (
            "Read-only passage to speak during Auto Voice Setup."
        )
        for key in ("curveCard", "dynamicsCard", "noiseCard", "voiceCard", "recordingCard", "reviewCard"):
            assert QAccessible.queryAccessibleInterface(scene_item(dialog, key)).role() == QAccessible.Role.Grouping
        meter = QAccessible.queryAccessibleInterface(scene_item(dialog, "levelMeter"))
        assert meter.role() == QAccessible.Role.ProgressBar
        for key in ("p95Number", "peakNumber", "lufsNumber"):
            assert QAccessible.queryAccessibleInterface(scene_item(dialog, key)).role() == QAccessible.Role.SpinBox
    assert {scene_text(dialog, key) for key in BUTTONS} == {
        "Cancel", "Retake", "Compare Recording", "Continue Anyway", "Start Voice Setup"
    }
    # A button's name contains what the button says.
    assert all(scene_text(dialog, key).casefold() in names[key].casefold() for key in BUTTONS)


def test_a_text_that_changes_is_read_with_its_new_text(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    view = dialog.view_model
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "warningText")) == "Ready to record"
        view.set_warning("Moving closer helps", state="info", strong=True)
        view.set_page(
            phaseText="Capturing speech", timeText="Time remaining: 4s", startText="Record Voice", progress=40
        )
        view.set_chip("overall", "Good capture. All checks passed.", state="ok", tip="how well it supports the bands")
        view.set_chip("gate", "Noise gate: opens above -40 dB")
        view.set_issues(["Speech was only 1.2 seconds long."])
        assert scene.name(scene_item(dialog, "warningText")) == "Moving closer helps"
        assert scene.name(scene_item(dialog, "phaseText")) == "Capturing speech"
        assert scene.name(scene_item(dialog, "timeText")) == "Time remaining: 4s"
        assert scene.name(scene_item(dialog, "startButton")) == "Record Voice"
        assert scene.name(scene_item(dialog, "overallChip")) == "Good capture. All checks passed."
        assert scene.description(scene_item(dialog, "overallChip")) == "how well it supports the bands"
        assert scene.name(scene_item(dialog, "gateChip")) == "Noise gate: opens above -40 dB"
        assert scene.name(scene_item(dialog, "issueChip0")) == "Speech was only 1.2 seconds long."
        assert scene.description(scene_item(dialog, "progressBar")) == "40 percent"
    assert scene_item(dialog, "warningText").property("color").name() == PALETTE.status_info_text
    assert scene_item(dialog, "warningText").property("font").bold()
    view.set_warning("Fine", state="ok", strong=False)
    assert scene_item(dialog, "warningText").property("color").name() == PALETTE.status_ok_text
    assert not scene_item(dialog, "warningText").property("font").bold()
    assert scene_item(dialog, "overallChip").property("tone") == "ok"
    assert scene_item(dialog, "issueChip0").property("tone") == "warn"
    view.set_issues([])
    assert not [i for i in scene_items(dialog.quick_view.rootObject()) if i.objectName() == "issueChip0"]


def test_the_review_shows_the_five_lines_in_order_and_only_with_a_summary(open_dialog) -> None:
    dialog = open_dialog()
    view = dialog.view_model
    lines = ["overallChip", "issueChip0", "eqChip", "gateChip", "deesserChip", "compressorChip"]
    view.set_issues(["One reason."])
    view.set_page(recordingShown=True, reviewShown=True)
    assert not scene_item(dialog, "reviewCard").isVisible(), "no summary yet"
    view.set_page(summaryShown=True)
    assert scene_item(dialog, "reviewCard").isVisible()
    QTest.qWait(60)  # the layout settles on the next frame
    ys = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).y() for key in lines]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    assert [scene_text(dialog, key) for key in lines[:1] + lines[2:]] == [
        "Overall: --", "EQ: --", "Noise gate: --", "De-esser: --", "Compressor: --"
    ]
    view.set_page(reviewShown=False)
    assert not scene_item(dialog, "reviewCard").isVisible()


STEP_NAMES = ["Target", "Level", "Room noise", "Voice", "Review"]
CARDS = ["curveCard", "dynamicsCard", "noiseCard", "voiceCard", "mutedNotice", "recordingCard", "progressBar",
         "timeText", "levelMeter", "reviewCard"]
METER = {"progressBar", "timeText", "levelMeter"}


@pytest.mark.parametrize(
    ("state", "step", "shown"),
    [
        ("idle", 0, {"curveCard", "dynamicsCard", "recordingCard"}),
        ("level_check", 1, {"voiceCard", "mutedNotice", "recordingCard"} | METER),
        ("level_result", 1, {"voiceCard", "recordingCard"} | METER),
        ("noise_recording", 2, {"noiseCard", "mutedNotice", "recordingCard"} | METER),
        ("noise_ready", 3, {"voiceCard", "recordingCard"} | METER),
        ("voice_recording", 3, {"voiceCard", "mutedNotice", "recordingCard"} | METER),
        ("capture_analyzing", 3, {"voiceCard", "recordingCard"} | METER),
        ("analyzing", 3, {"voiceCard", "recordingCard"} | METER),
        ("verification_recording", 3, {"voiceCard", "mutedNotice", "recordingCard"} | METER),
        ("verification_analyzing", 3, {"voiceCard", "recordingCard"} | METER),
        ("completed", 4, {"recordingCard", "reviewCard"}),
        ("verification_ready", 4, {"recordingCard", "reviewCard"}),
        ("final_comparison_ready", 4, {"recordingCard", "reviewCard"}),
    ],
)
def test_each_state_of_the_run_shows_its_own_cards(open_dialog, state, step, shown) -> None:
    dialog = open_dialog()
    # As soon as a run has begun, and with a result to review.
    dialog.view_model.set_page(recordingShown=True, summaryShown=True)
    dialog.setup_state = state
    assert {key for key in CARDS if scene_item(dialog, key).isVisible()} == shown
    assert scene_item(dialog, "stepHeader").property("width") > 0
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "stepHeader")) == f"Step {step + 1} of 5: {STEP_NAMES[step]}"
    assert dialog.view_model.shown["recordingTitle"] == ("Review" if step == 4 else "Recording")
    assert scene_item(dialog, "compareButton").isVisible() is (step == 4)


def test_a_take_being_checked_stays_on_the_step_it_was_recorded_on(open_dialog) -> None:
    dialog = open_dialog()
    for phase, step in (("level_check", 1), ("noise_recording", 2), ("voice_recording", 3)):
        dialog._capture_phase = phase
        dialog.setup_state = "capture_analyzing"
        assert dialog.view_model.shown["step"] == step, phase


def test_a_failed_take_stays_on_its_own_step(open_dialog) -> None:
    dialog = open_dialog()
    dialog._failed_step = 2
    dialog.setup_state = "idle"
    assert dialog.view_model.shown["step"] == 2 and scene_item(dialog, "noiseCard").isVisible()
    assert not scene_item(dialog, "curveCard").isVisible()


def test_before_a_run_only_the_choices_are_shown(open_dialog) -> None:
    dialog = open_dialog()
    items = ["curveCard", "dynamicsCard", "noiseCard", "voiceCard", "mutedNotice", "recordingCard", "reviewCard",
             "compareButton", "retakeButton", "overrideButton"]
    assert {key for key in items if scene_item(dialog, key).isVisible()} == {"curveCard", "dynamicsCard"}
    assert scene_text(dialog, "startButton") == "Start Voice Setup"
    assert scene_item(dialog, "startButton").property("look") == "primary"


def test_tab_visits_the_page_and_its_fields_then_the_buttons_left_to_right(open_dialog) -> None:
    dialog = open_dialog(everything=True, custom=True)
    walk = _tab_walk(dialog)
    # A number is typed into: the focus is in the field inside it.
    assert [name.removesuffix("Input") for name in walk] == ["contentScroll", *FIELDS, *BUTTONS]
    # The actions stay out of the scrolling page.
    scroll = scene_item(dialog, "contentScroll")
    for key in BUTTONS:
        assert not scroll.isAncestorOf(scene_item(dialog, key)), key
    xs = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).x() for key in BUTTONS]
    assert xs == sorted(xs)
    ys = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).y() for key in FIELDS]
    assert ys == sorted(ys)


def test_the_custom_limits_are_left_out_of_the_tab_order_until_custom_is_chosen(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    walk = [name.removesuffix("Input") for name in _tab_walk(dialog)]
    assert walk == ["contentScroll", "curvePicker", "dynamicsPicker", "lufsNumber", *BUTTONS]


def test_enter_presses_the_focused_button_and_escape_closes_the_dialog(open_dialog, qapp) -> None:
    dialog = open_dialog()
    scene_item(dialog, "cancelButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert not dialog.isVisible() and dialog.result() == REJECTED

    again = open_dialog()
    QTest.keyClick(again.quick_view, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not again.isVisible() and again.result() == REJECTED


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_enter_outside_a_button_does_what_the_main_button_says(open_dialog, monkeypatch, key) -> None:
    calls: list[str] = []
    monkeypatch.setattr(VoiceSetupDialog, "_on_start_clicked", lambda self: calls.append("start"))
    dialog = open_dialog()
    scene_item(dialog, "contentScroll").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, key)
    assert calls == ["start"]
    dialog.view_model.set_page(startEnabled=False, startText="Recording Noise...")
    QTest.keyClick(dialog.quick_view, key)
    assert calls == ["start"], "the main button is not available"


@pytest.mark.parametrize(
    ("button", "command"),
    [
        ("cancelButton", "_on_cancel_clicked"),
        ("retakeButton", "_on_retake_clicked"),
        ("compareButton", "_compare_recording"),
        ("overrideButton", "_continue_past_level"),
        ("startButton", "_on_start_clicked"),
    ],
)
def test_space_and_a_click_on_each_button_run_the_dialogs_command(
    open_dialog, monkeypatch, qapp, button, command
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(VoiceSetupDialog, command, lambda self: calls.append(command))
    dialog = open_dialog(everything=True)
    scene_item(dialog, button).forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    assert calls == [command]
    press(dialog, button)
    qapp.processEvents()
    assert calls == [command, command]


def test_the_pickers_choose_from_the_keyboard_and_the_dialog_reads_the_choice(open_dialog) -> None:
    dialog = open_dialog()
    keys = list(TARGET_CURVES)
    assert dialog.get_selected_curve() == keys[0]
    assert scene_item(dialog, "curveDescription").property("text") == FIRST_CURVE.description
    choose(dialog, "curvePicker", 1)
    assert dialog.get_selected_curve() == keys[1]
    assert scene_item(dialog, "curveDescription").property("text") == TARGET_CURVES[keys[1]].description
    assert dialog.view_model.selected("dynamics") == "balanced"
    choose(dialog, "dynamicsPicker", DYNAMICS_KEYS.index("dense"))
    assert dialog.view_model.selected("dynamics") == "dense"
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "dynamicsPicker")) == "Compression: Dense"
    assert dialog.view_model.dynamics == [name for _key, name in DYNAMICS]


def test_the_dynamics_choice_and_the_limits_are_kept_in_the_settings(open_dialog) -> None:
    dialog = open_dialog()
    config = dialog.parent().config
    choose(dialog, "dynamicsPicker", DYNAMICS_KEYS.index("custom"))
    assert config.voice_setup_dynamics_intensity == "custom"
    assert (config.voice_setup_custom_p95_db, config.voice_setup_custom_peak_cap_db) == (3.5, 8.0)
    box = scene_item(dialog, "p95Number")
    box.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Up)
    assert dialog.view_model.number("p95") == 4.5 and config.voice_setup_custom_p95_db == 4.5
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    assert dialog.view_model.number("p95") == 2.5 and config.voice_setup_custom_p95_db == 2.5
    # The loudness target is not part of the dynamics setting.
    scene_item(dialog, "lufsNumber").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Up)
    assert dialog.view_model.number("lufs") == -17.0
    assert not hasattr(config, "voice_setup_target_lufs")


def test_a_number_is_stepped_from_the_keyboard_typed_and_held_in_its_range(open_dialog, qapp) -> None:
    dialog = open_dialog(custom=True)
    view = dialog.view_model
    box = scene_item(dialog, "peakNumber")
    box.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    for _ in range(10):
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Up)
    assert view.number("peak") == 12.0, "the top of the range"
    for _ in range(20):
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    assert view.number("peak") == 1.5, "the bottom of the range"
    # Typing replaces the value; Enter in the field is not the main button's.
    started: list[int] = []
    view._commands["start"] = lambda: started.append(1)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(dialog.quick_view, "6.25")
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Tab)
    qapp.processEvents()
    assert view.number("peak") == 6.25
    assert scene_item(dialog, "peakNumber").property("value") == 625
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "peakNumber")).endswith("6.25 dB")
    # The target loudness is held between -24 and -12 LUFS.
    view.setNumber("lufs", -40.0)
    assert view.number("lufs") == -24.0
    view.setNumber("lufs", -3.0)
    assert view.number("lufs") == -12.0
    view.setNumber("lufs", -14.0)
    assert scene_item(dialog, "lufsNumber").property("value") == -1400


def test_the_limits_are_for_the_custom_setting_and_every_field_locks_with_its_card(open_dialog) -> None:
    dialog = open_dialog()
    items = {key: scene_item(dialog, key) for key in FIELDS}
    assert {key: item.isEnabled() for key, item in items.items()} == {
        "curvePicker": True, "dynamicsPicker": True, "p95Number": False, "peakNumber": False, "lufsNumber": True,
    }
    dialog.view_model.pickDynamics(DYNAMICS_KEYS.index("custom"))
    assert all(item.isEnabled() for item in items.values())
    view = dialog.view_model
    view.set_page(curveEnabled=False)
    assert not items["curvePicker"].isEnabled() and items["dynamicsPicker"].isEnabled()
    view.set_page(curveEnabled=True, curveGroupEnabled=False)
    assert not items["curvePicker"].isEnabled() and items["dynamicsPicker"].isEnabled()
    view.set_page(curveGroupEnabled=True, targetEnabled=False)
    assert not items["lufsNumber"].isEnabled() and items["p95Number"].isEnabled()
    view.set_page(targetEnabled=True, dynamicsEnabled=False)
    assert not any(items[key].isEnabled() for key in FIELDS[1:])
    assert items["curvePicker"].isEnabled()


def test_enter_on_a_closed_picker_does_what_the_main_button_says_but_in_its_open_list_it_chooses(
    open_dialog, monkeypatch, qapp
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(VoiceSetupDialog, "_on_start_clicked", lambda self: calls.append("start"))
    dialog = open_dialog()
    picker = scene_item(dialog, "curvePicker")
    picker.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    qapp.processEvents()
    assert picker.property("down") is True, "the list is open"
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert picker.property("down") is False and calls == [], "Enter in the open list only chose"
    assert dialog.get_selected_curve() == list(TARGET_CURVES)[1]
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    assert calls == ["start"]


def test_enter_in_a_number_field_commits_what_was_typed_and_does_what_the_main_button_says(
    open_dialog, monkeypatch, qapp
) -> None:
    calls: list[float] = []
    monkeypatch.setattr(VoiceSetupDialog, "_on_start_clicked", lambda self: calls.append(self.view_model.number("p95")))
    dialog = open_dialog(custom=True)
    scene_item(dialog, "p95Number").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(dialog.quick_view, "5")
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert calls == [5.0], "the typed value is the one the run uses"


@pytest.mark.parametrize("large", [False, True])
def test_nothing_is_cut_short_or_runs_off_a_small_dialog(open_dialog, qapp, large) -> None:
    previous = qapp.font()
    if large:
        bigger = QFont(previous)
        bigger.setPointSize(14)
        qapp.setFont(bigger)
    try:
        dialog = open_dialog(everything=True, custom=True)
        dialog.view_model.set_warning(LONG_WARNING, state="warn", strong=True)
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
        # The widest rows of actions: the review's and the level check's.
        rows = [
            {"retakeShown": True, "compareShown": True, "overrideShown": False,
             "startText": "Record Verification Passage"},
            {"retakeShown": False, "compareShown": False, "overrideShown": True,
             "overrideText": "Continue Despite Clipping Risk", "startText": "Check Level Again"},
        ]
        for row in rows:
            dialog.view_model.set_page(**row)
            QTest.qWait(60)  # the layout settles on the next frame
            shown = [key for key in BUTTONS if scene_item(dialog, key).isVisible()]
            xs = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).x() for key in shown]
            edges = [scene_item(dialog, key).mapToScene(QPointF(scene_item(dialog, key).width(), 0)).x()
                     for key in shown]
            assert xs == sorted(xs) and xs[0] >= 0 and edges[-1] <= dialog.quick_view.width(), row
            assert all(edges[i] <= xs[i + 1] for i in range(len(shown) - 1)), row
            assert cut_short_texts(dialog) == [], row
    finally:
        qapp.setFont(previous)


def test_every_main_button_label_fits_beside_the_other_buttons(open_dialog, qapp) -> None:
    from mic_eq.ui.voice_setup_view import START_LABELS

    dialog = open_dialog(everything=True)
    at_scene_minimum(dialog, qapp)
    dialog.view_model.set_page(retakeShown=True, compareShown=True, overrideShown=False)
    for label in START_LABELS:
        dialog.view_model.set_page(startText=label)
        QTest.qWait(30)
        start = scene_item(dialog, "startButton")
        edge = start.mapToScene(QPointF(start.width(), 0)).x()
        left = start.mapToScene(QPointF(0, 0)).x()
        compare = scene_item(dialog, "compareButton")
        assert compare.mapToScene(QPointF(compare.width(), 0)).x() <= left, label
        assert edge <= dialog.quick_view.width(), label
    assert cut_short_texts(dialog) == []


def test_the_keyboard_alone_scrolls_a_small_dialog_down_to_its_message(open_dialog, qapp) -> None:
    dialog = open_dialog(everything=True)
    dialog.resize(500, 380)
    qapp.processEvents()
    qapp.processEvents()
    scroll, hint = scene_item(dialog, "contentScroll"), scene_item(dialog, "reviewHint")
    assert not _within_page(dialog, hint)
    scroll.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
    assert 0 < scroll.property("contentY") < scroll.height()
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_End)
    qapp.processEvents()
    assert scroll.property("contentY") == scroll.property("contentHeight") - scroll.height()
    assert _within_page(dialog, hint)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Home)
    assert scroll.property("contentY") == 0
    assert scene_item(dialog, "scrollFocusRing").isVisible()


@pytest.mark.parametrize(
    "button",
    ["startButton", "retakeButton", "compareButton", "curvePicker-locked", "dynamicsPicker-hidden",
     "lufsNumber-locked", "p95Number-locked", "peakNumber-hidden"],
)
def test_a_button_that_disables_or_hides_itself_hands_the_keyboard_to_the_page(open_dialog, qapp, button) -> None:
    dialog = open_dialog(everything=True, custom=True)
    hides = {"startButton": {"startEnabled": False}, "retakeButton": {"retakeShown": False},
             "compareButton": {"compareEnabled": False},
             # Enter on a field starts the run, which locks the fields and hides their cards.
             "curvePicker-locked": {"curveEnabled": False}, "dynamicsPicker-hidden": {"targetShown": False},
             "lufsNumber-locked": {"targetEnabled": False}, "p95Number-locked": {"dynamicsEnabled": False},
             "peakNumber-hidden": {"targetShown": False}}[button]
    button = button.split("-")[0]
    scene_item(dialog, button).forceActiveFocus(Qt.FocusReason.TabFocusReason)
    assert _focus_name(dialog).removesuffix("Input") == button
    dialog.view_model.set_page(**hides)
    qapp.processEvents()
    assert _focus_name(dialog) == "contentScroll"
    # A button that goes while it is not focused moves nothing.
    dialog.view_model.set_page(**{k: not v for k, v in hides.items()})
    scene_item(dialog, "cancelButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    dialog.view_model.set_page(**hides)
    assert _focus_name(dialog) == "cancelButton"


def test_the_level_meter_follows_the_dialogs_meter_and_reads_its_levels(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    assert dialog.level_meter is dialog.view_model.level_meter
    dialog.level_meter.set_levels(-20.0, -1.0)
    with _AccessibleScene() as scene:
        description = scene.description(scene_item(dialog, "levelMeter"))
    assert description == dialog.level_meter.property("accessibleText") and description != ""


def test_the_eq_comparison_is_described_and_painted_without_scene_messages(open_dialog, qapp) -> None:
    dialog = open_dialog(everything=True)
    flat = EQSettings(enabled=True)
    frequencies = [60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 3200.0, 6300.0, 10000.0, 16000.0]
    gains = [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    proposed = build_eq_candidate_settings(flat, frequencies, gains, [1.41] * 10, layer="correction", enabled=True)
    dialog.eq_compare.set_curves(flat, proposed, 48_000.0)
    qapp.processEvents()
    with _AccessibleScene() as scene:
        description = scene.description(scene_item(dialog, "eqCompare"))
    assert description == dialog.eq_compare.description() and description.startswith("Largest change: +")
    assert dialog.quick_view.grabFramebuffer().isNull() is False


def test_the_main_button_is_named_by_what_it_says_at_each_step(open_dialog, monkeypatch) -> None:
    monkeypatch.setattr(voice_setup_dialog, "CaptureSession", _Session)
    dialog = open_dialog()
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: _lend(dialog))
    with _AccessibleScene() as scene:
        start = scene_item(dialog, "startButton")
        assert scene.name(start) == scene_text(dialog, "startButton") == "Start Voice Setup"
        dialog._on_start_clicked()
        dialog._capture_start_timer.stop()
        assert scene.name(start) == scene_text(dialog, "startButton") == "Checking Level..."
        finish_take(dialog, HOT)
        assert scene.name(start) == scene_text(dialog, "startButton") == "Check Level Again"
        assert scene.name(scene_item(dialog, "overrideButton")) == "Continue Despite Clipping Risk"


def _lend(dialog: Any) -> bool:
    from engine_fakes import lend_audio

    return lend_audio(dialog)


def test_the_mute_notice_is_a_warning_chip_that_says_the_call_is_muted(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    notice = scene_item(dialog, "mutedNotice")
    text = notice.property("text").lower()
    assert notice.property("tone") == "warn" and "muted" in text and "call" in text


def test_the_passage_is_shown_whole_and_never_scrolled_inside_the_page(open_dialog, qapp) -> None:
    dialog = open_dialog()
    dialog.setup_state = "level_check"  # the passage is read from the level check on
    dialog.resize(520, 700)
    qapp.processEvents()
    passages = [i for i in scene_items(dialog.quick_view.rootObject()) if i.property("text") == RAINBOW_PASSAGE]
    assert len(passages) == 1
    passage = passages[0]
    assert passage.objectName() == "passageText" and passage.property("lineCount") > 1  # it wraps
    assert not passage.property("truncated")
    # Tall enough for every wrapped line, and the page is the only thing that scrolls.
    assert passage.height() >= passage.property("contentHeight") - 1
    scrolling = [i for i in scene_items(dialog.quick_view.rootObject()) if i.property("contentHeight") is not None]
    assert [i.objectName() for i in scrolling if i.property("contentY") is not None] == ["contentScroll"]


def test_the_picture_takes_no_input(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    picture = scene_item(dialog, "eqCompare")
    assert picture.acceptedMouseButtons() == Qt.MouseButton.NoButton and not picture.activeFocusOnTab()


def test_the_view_model_reports_changes_once_and_forwards_its_commands(qapp) -> None:
    calls: list[object] = []
    model = VoiceSetupView(
        curves=(("a", "A"), ("b", "B")),
        dynamics_key="custom",
        numbers={"p95": 3.5, "peak": 8.0, "lufs": -18.0},
        passage="Read.",
        curve_changed=calls.append,
        dynamics_changed=lambda: calls.append("dynamics"),
        start=lambda: calls.append("start"), override=lambda: calls.append("override"),
        cancel=lambda: calls.append("cancel"), retake=lambda: calls.append("retake"),
        compare=lambda: calls.append("compare"),
    )
    changes: list[str] = []
    model.pageChanged.connect(lambda: changes.append("page"))
    model.statusChanged.connect(lambda: changes.append("status"))
    model.chipsChanged.connect(lambda: changes.append("chips"))
    model.choicesChanged.connect(lambda: changes.append("choices"))
    assert model.selected("dynamics") == "custom" and model.dynamicsCustom is True
    model.set_page(startText="Go")
    model.set_page(startText="Go")
    model.set_warning("Hello", state="ok", strong=True)
    model.set_warning("Hello")
    with pytest.raises(ValueError, match="unknown state"):
        model.set_warning("Hello", state="success")
    with pytest.raises(ValueError, match="unknown state"):
        model.set_chip("gate", state="success")
    with pytest.raises(KeyError, match="no field startTitle"):
        model.set_page(startTitle="Go")
    with pytest.raises(KeyError):
        model.setNumber("gain", 1.0)
    assert (model.warning_text, model.warning_state, model.warning_strong) == ("Hello", "ok", True)
    model.set_chip("gate", "G", state="warn")
    model.set_chip("gate", "G")
    assert model.chip("gate") == {"key": "gate", "text": "G", "tip": "", "state": "warn"}
    model.set_issues(["One"])
    model.set_issues(["One"])
    assert model.issue_lines == ["One"]
    model.pickCurve(1)
    model.pickCurve(1)
    model.pickCurve(5)
    model.pickDynamics(0)
    model.pickDynamics(0)
    assert (model.selected("curve"), model.selected("dynamics"), model.dynamicsCustom) == ("b", "gentle", False)
    model.setNumber("p95", 4.0)
    model.setNumber("p95", 4.0)
    model.setNumber("peak", 99.0)
    model.setNumber("lufs", -20.0)
    assert (model.number("p95"), model.number("peak"), model.number("lufs")) == (4.0, 12.0, -20.0)
    assert changes == ["page", "status", "chips", "chips", "choices", "choices", "choices", "choices", "choices"]
    for command in (model.start, model.override, model.cancel, model.retake, model.compare):
        command()
    # The dynamics callback follows the choice and the two limits, not the loudness target.
    assert calls == [1, "dynamics", "dynamics", "dynamics", "start", "override", "cancel", "retake", "compare"]


def test_the_view_model_refuses_an_unknown_dynamics_intensity(qapp) -> None:
    with pytest.raises(ValueError, match="unknown dynamics intensity 'loud'"):
        VoiceSetupView(
            curves=(("a", "A"),),
            dynamics_key="loud",
            numbers={"p95": 3.5, "peak": 8.0, "lufs": -18.0},
            passage="Read.",
            curve_changed=lambda _index: None,
            dynamics_changed=lambda: None,
            start=lambda: None, override=lambda: None, cancel=lambda: None,
            retake=lambda: None, compare=lambda: None,
        )


def test_a_scene_that_does_not_load_says_so_and_closes_without_starting_anything(qapp, monkeypatch) -> None:
    connected: list[object] = []
    shown: list[str] = []
    quitting = SimpleNamespace(aboutToQuit=SimpleNamespace(connect=connected.append))
    module = "mic_eq.ui.voice_setup_dialog"
    monkeypatch.setattr(f"{module}.QApplication", SimpleNamespace(instance=lambda: quitting))
    monkeypatch.setattr(f"{module}.load_scene", lambda *_args: "no such file")
    monkeypatch.setattr(f"{module}.QMessageBox.critical", lambda _parent, title, text: shown.append(f"{title}: {text}"))
    owner = _Owner()
    dialog = VoiceSetupDialog(owner, configuration=owner.configuration)
    assert dialog.load_error == "no such file"
    assert dialog.exec() == REJECTED
    assert len(shown) == 1 and shown[0].startswith("Auto Voice Setup: ") and "no such file" in shown[0]
    assert connected == [], "no quit hook"
    assert owner.processor.is_running() is False and owner.engine_session.view.user_muted is False
    assert not dialog._capture_start_timer.isActive() and not dialog.recording_timer.isActive()
    assert dialog._audio_lease is None and dialog._analysis_workers == []
    dialog.reject()
    assert dialog.result() == REJECTED
    dialog.deleteLater()
    owner.close()
    qapp.processEvents()


def test_the_width_budget_holds_every_label_the_main_and_override_buttons_can_show() -> None:
    import ast
    import inspect

    from mic_eq.analysis.voice_setup import MAX_VERIFICATION_BAD_CAPTURES
    from mic_eq.ui.level_check import LevelVerdict, level_choices
    from mic_eq.ui.voice_setup_view import OVERRIDE_LABELS, START_LABELS

    for code in ("good", "too_hot", "too_quiet", "silent", "unreadable"):
        choices = level_choices(LevelVerdict(code, "", "ok"), "Record Room Noise")
        assert choices.primary in START_LABELS, code
        assert choices.override is None or choices.override in OVERRIDE_LABELS, code
    # The dialog counts the failed attempts up to the limit; the widest count must fit the budget.
    retry = max((label for label in START_LABELS if label.startswith("Retry Verification")), key=len)
    for attempt in range(1, MAX_VERIFICATION_BAD_CAPTURES + 1):
        assert len(f"Retry Verification ({attempt}/{MAX_VERIFICATION_BAD_CAPTURES})") <= len(retry)

    # Every literal the dialog passes as the main button's text.
    def values(node: ast.expr) -> set[str]:
        if isinstance(node, ast.IfExp):
            return values(node.body) | values(node.orelse)
        if isinstance(node, ast.Name) and node.id == "_APPLY_VERIFY_AGAIN_LABEL":
            return {voice_setup_dialog._APPLY_VERIFY_AGAIN_LABEL}
        return {node.value} if isinstance(node, ast.Constant) and isinstance(node.value, str) else set()

    literals = set()
    for node in ast.walk(ast.parse(inspect.getsource(voice_setup_dialog))):
        if isinstance(node, ast.keyword) and node.arg == "startText":
            literals |= values(node.value)
    assert literals and literals <= set(START_LABELS), literals - set(START_LABELS)
