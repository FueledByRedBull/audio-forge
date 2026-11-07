"""The Auto-EQ calibration dialog is a Qt Quick scene: it loads, names its controls and takes the keyboard."""
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
from mic_eq.ui import calibration_dialog
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.calibration_support import RAINBOW_PASSAGE
from mic_eq.ui.calibration_view import CalibrationView
from mic_eq.ui.capture_session import LIVE_OUTPUT_MUTED_NOTICE
from mic_eq.ui.theme import PALETTE
from test_auto_eq_candidate_lifecycle import _Owner
from test_setup_level_check_flow import HOT, _Session
from test_first_run_setup import at_scene_minimum, choose, cut_short_texts, press, scene_item, scene_items, scene_text
from test_quick_shell import _AccessibleScene

REJECTED = int(QDialog.DialogCode.Rejected)
EVERYTHING = {
    "recordingShown": True, "meterShown": True, "diagnosticsShown": True, "eqCompareShown": True,
    "retakeShown": True, "compareShown": True, "compareEnabled": True, "overrideShown": True,
    "mutedShown": True,
}
BUTTONS = ["cancelButton", "retakeButton", "compareButton", "overrideButton", "startButton"]
FIELDS = ["curvePicker", "modePicker", "smoothingPicker"]
LONG_WARNING = "Auto-EQ could not confirm the correction is safe, so it is not offered. " * 4
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
    made: list[tuple[CalibrationDialog, _Owner]] = []

    def make(*, everything: bool = False) -> Any:
        owner = _Owner()
        dialog = CalibrationDialog(owner, configuration=owner.configuration)
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


def test_the_scene_loads_and_every_control_has_its_accessible_name(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    assert dialog.quick_view.status() == QQuickWidget.Status.Ready and dialog.quick_view.errors() == []
    names = {
        "stepHeader": "Step 1 of 3: Set up",
        "contentScroll": "Auto-EQ page",
        "curvePicker": f"Tone preset: {FIRST_CURVE.name}",
        "modePicker": "Matching: Subtle",
        "smoothingPicker": "Smoothing: Conservative",
        "mutedNotice": LIVE_OUTPUT_MUTED_NOTICE,
        "progressBar": "Auto-EQ recording progress",
        "levelMeter": "Auto-EQ input level",
        "eqCompare": "EQ now compared with the proposed EQ",
        "cancelButton": "Cancel Auto-EQ",
        "retakeButton": "Retake Auto-EQ recording",
        "compareButton": "Compare Recording",
        "overrideButton": "Continue Anyway",
        "startButton": "Check Level",
        "curveCard": "Tone preset",
        "instructionsCard": "Read this passage aloud",
        "recordingCard": "Recording",
        "resultCard": "Result",
    }
    with _AccessibleScene() as scene:
        assert {key: scene.name(scene_item(dialog, key)) for key in names} == names
        assert QAccessible.queryAccessibleInterface(dialog.quick_view).text(QAccessible.Text.Name) == "Auto-EQ content"
        assert scene.description(scene_item(dialog, "modePicker")).startswith("Subtle aims for small tone changes")
        assert scene.description(scene_item(dialog, "smoothingPicker")).startswith("Conservative smoothing ignores")
        assert scene.description(scene_item(dialog, "compareButton")) == (
            "Listen to this same passage with current and proposed processing"
        )
        assert scene.description(scene_item(dialog, "passageText")) == (
            "Read-only passage to speak while Auto-EQ records."
        )
        for key in ("curveCard", "instructionsCard", "recordingCard", "resultCard"):
            assert QAccessible.queryAccessibleInterface(scene_item(dialog, key)).role() == QAccessible.Role.Grouping
        meter = QAccessible.queryAccessibleInterface(scene_item(dialog, "levelMeter"))
        assert meter.role() == QAccessible.Role.ProgressBar
    assert {scene_text(dialog, key) for key in BUTTONS} == {
        "Cancel", "Retake", "Compare Recording", "Continue Anyway", "Check Level"
    }
    # A button's name contains what the button says.
    assert all(scene_text(dialog, key).casefold() in names[key].casefold() for key in BUTTONS)


def test_a_text_that_changes_is_read_with_its_new_text(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    view = dialog.view_model
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "warningText")) == "Ready to record"
        view.set_warning("Moving closer helps", state="info", strong=True, tip="why: the room")
        view.set_page(timeText="Time remaining: 4s", startText="Record Again", progress=40)
        view.set_chip("confidence", "Good capture: clear.", state="ok", tip="how well it supports the bands")
        view.set_chip("gain", "Headroom check: passed.")
        assert scene.name(scene_item(dialog, "warningText")) == "Moving closer helps"
        assert scene.description(scene_item(dialog, "warningText")) == "why: the room"
        assert scene.name(scene_item(dialog, "timeText")) == "Time remaining: 4s"
        assert scene.name(scene_item(dialog, "startButton")) == "Record Again"
        assert scene.name(scene_item(dialog, "confidenceChip")) == "Good capture: clear."
        assert scene.description(scene_item(dialog, "confidenceChip")) == "how well it supports the bands"
        assert scene.name(scene_item(dialog, "gainChip")) == "Headroom check: passed."
        assert scene.description(scene_item(dialog, "progressBar")) == "40 percent"
        view.set_page(progressBusy=True)
        assert scene.description(scene_item(dialog, "progressBar")) == "Working"
    assert scene_item(dialog, "warningText").property("color").name() == PALETTE.status_info_text
    assert scene_item(dialog, "warningText").property("font").bold()
    view.set_warning("Fine", state="ok", strong=False)
    assert scene_item(dialog, "warningText").property("color").name() == PALETTE.status_ok_text
    assert not scene_item(dialog, "warningText").property("font").bold()
    assert scene_item(dialog, "confidenceChip").property("tone") == "ok"


@pytest.mark.parametrize(
    ("state", "step", "shown"),
    [
        ("idle", 0, {"curveCard", "instructionsCard", "recordingCard"}),
        ("level_check", 1, {"instructionsCard", "mutedNotice", "recordingCard", "progressBar", "timeText", "levelMeter"}),
        ("level_result", 1, {"instructionsCard", "recordingCard", "progressBar", "timeText", "levelMeter"}),
        ("recording", 1, {"instructionsCard", "mutedNotice", "recordingCard", "progressBar", "timeText", "levelMeter"}),
        ("capture_analyzing", 1, {"instructionsCard", "recordingCard", "progressBar", "timeText", "levelMeter"}),
        ("analyzing", 1, {"instructionsCard", "recordingCard", "progressBar", "timeText", "levelMeter"}),
        ("ready", 2, {"recordingCard"}),
    ],
)
def test_each_state_of_the_run_shows_its_own_cards(open_dialog, state, step, shown) -> None:
    dialog = open_dialog()
    dialog.view_model.set_page(recordingShown=True)  # as soon as a run has begun
    dialog.recording_state = state
    items = ["curveCard", "instructionsCard", "mutedNotice", "recordingCard", "progressBar", "timeText", "levelMeter"]
    assert {key for key in items if scene_item(dialog, key).isVisible()} == shown
    assert scene_item(dialog, "stepHeader").property("width") > 0
    with _AccessibleScene() as scene:
        names = ["Set up", "Record", "Review"]
        assert scene.name(scene_item(dialog, "stepHeader")) == f"Step {step + 1} of 3: {names[step]}"
    assert dialog.view_model.shown["recordingTitle"] == ("Review" if step == 2 else "Recording")
    assert scene_item(dialog, "compareButton").isVisible() is (step == 2)


def test_before_a_run_only_the_choices_and_the_passage_are_shown(open_dialog) -> None:
    dialog = open_dialog()
    items = ["curveCard", "instructionsCard", "mutedNotice", "recordingCard", "resultCard", "compareButton",
             "retakeButton", "overrideButton"]
    assert {key for key in items if scene_item(dialog, key).isVisible()} == {"curveCard", "instructionsCard"}
    assert scene_text(dialog, "startButton") == "Check Level"


def test_tab_visits_the_page_and_its_fields_then_the_buttons_left_to_right(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    assert _tab_walk(dialog) == ["contentScroll", *FIELDS, *BUTTONS]
    # The actions stay out of the scrolling page.
    scroll = scene_item(dialog, "contentScroll")
    for key in BUTTONS:
        assert not scroll.isAncestorOf(scene_item(dialog, key)), key
    xs = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).x() for key in BUTTONS]
    assert xs == sorted(xs)
    ys = [scene_item(dialog, key).mapToScene(QPointF(0, 0)).y() for key in FIELDS]
    assert ys == sorted(ys)


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
    monkeypatch.setattr(CalibrationDialog, "_on_start_clicked", lambda self: calls.append("start"))
    dialog = open_dialog()
    scene_item(dialog, "contentScroll").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, key)
    assert calls == ["start"]
    dialog.view_model.set_page(startEnabled=False, startText="Recording...")
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
    monkeypatch.setattr(CalibrationDialog, command, lambda self: calls.append(command))
    dialog = open_dialog(everything=True)
    scene_item(dialog, button).forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    assert calls == [command]
    press(dialog, button)
    qapp.processEvents()
    assert calls == [command, command]


def test_the_pickers_choose_from_the_keyboard_and_the_dialog_reads_the_choice(open_dialog, qapp) -> None:
    dialog = open_dialog()
    keys = list(TARGET_CURVES)
    assert dialog.get_selected_curve() == keys[0]
    assert scene_item(dialog, "curveDescription").property("text") == FIRST_CURVE.description
    choose(dialog, "curvePicker", 1)
    assert dialog.get_selected_curve() == keys[1]
    assert scene_item(dialog, "curveDescription").property("text") == TARGET_CURVES[keys[1]].description
    choose(dialog, "modePicker", 1)
    assert dialog.get_selected_target_mode() == "static"
    choose(dialog, "smoothingPicker", 2)
    assert dialog.get_selected_smoothing_strength() == "broad"
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "modePicker")) == "Matching: Full strength"
        assert scene.name(scene_item(dialog, "smoothingPicker")) == "Smoothing: Broad"


def test_the_pickers_are_locked_while_a_run_is_under_way(open_dialog) -> None:
    dialog = open_dialog()
    assert all(scene_item(dialog, key).isEnabled() for key in FIELDS)
    dialog.view_model.set_page(choicesEnabled=False)
    assert not any(scene_item(dialog, key).isEnabled() for key in FIELDS)


def test_enter_on_a_closed_picker_does_what_the_main_button_says_but_in_its_open_list_it_chooses(
    open_dialog, monkeypatch, qapp
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(CalibrationDialog, "_on_start_clicked", lambda self: calls.append("start"))
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


@pytest.mark.parametrize("large", [False, True])
def test_nothing_is_cut_short_or_runs_off_a_small_dialog(open_dialog, qapp, large) -> None:
    previous = qapp.font()
    if large:
        bigger = QFont(previous)
        bigger.setPointSize(14)
        qapp.setFont(bigger)
    try:
        dialog = open_dialog(everything=True)
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
             "startText": "Apply EQ Settings"},
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


def test_the_keyboard_alone_scrolls_a_small_dialog_down_to_its_message(open_dialog, qapp) -> None:
    dialog = open_dialog(everything=True)
    dialog.resize(480, 360)
    qapp.processEvents()
    qapp.processEvents()
    scroll, hint = scene_item(dialog, "contentScroll"), scene_item(dialog, "resultHint")
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


@pytest.mark.parametrize("button", ["startButton", "retakeButton", "compareButton"])
def test_a_button_that_disables_or_hides_itself_hands_the_keyboard_to_the_page(open_dialog, qapp, button) -> None:
    dialog = open_dialog(everything=True)
    hides = {"startButton": {"startEnabled": False}, "retakeButton": {"retakeShown": False},
             "compareButton": {"compareEnabled": False}}[button]
    scene_item(dialog, button).forceActiveFocus(Qt.FocusReason.TabFocusReason)
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
    monkeypatch.setattr(calibration_dialog, "CaptureSession", _Session)
    dialog = open_dialog()
    with _AccessibleScene() as scene:
        start = scene_item(dialog, "startButton")
        assert scene.name(start) == scene_text(dialog, "startButton") == "Check Level"
        dialog._on_start_clicked()
        dialog._capture_start_timer.stop()
        assert scene.name(start) == scene_text(dialog, "startButton") == "Checking Level..."
        finish_take(dialog, HOT)
        assert scene.name(start) == scene_text(dialog, "startButton") == "Check Level Again"
        assert scene.name(scene_item(dialog, "overrideButton")) == "Continue Despite Clipping Risk"


def test_the_mute_notice_is_a_warning_chip_that_says_the_call_is_muted(open_dialog) -> None:
    dialog = open_dialog(everything=True)
    notice = scene_item(dialog, "mutedNotice")
    text = notice.property("text").lower()
    assert notice.property("tone") == "warn" and "muted" in text and "call" in text


def test_the_passage_is_shown_whole_and_never_scrolled_inside_the_page(open_dialog, qapp) -> None:
    dialog = open_dialog()
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


def test_the_view_model_reports_changes_once_and_forwards_its_commands(qapp) -> None:
    calls: list[object] = []
    model = CalibrationView(
        steps=("One", "Two"),
        curves=(("a", "A"), ("b", "B")), modes=(("m", "M"),), smoothings=(("s", "S"), ("t", "T")),
        passage="Read.",
        curve_changed=calls.append,
        start=lambda: calls.append("start"), override=lambda: calls.append("override"),
        cancel=lambda: calls.append("cancel"), retake=lambda: calls.append("retake"),
        compare=lambda: calls.append("compare"),
    )
    changes: list[str] = []
    model.pageChanged.connect(lambda: changes.append("page"))
    model.statusChanged.connect(lambda: changes.append("status"))
    model.chipsChanged.connect(lambda: changes.append("chips"))
    model.choicesChanged.connect(lambda: changes.append("choices"))
    model.set_page(startText="Go")
    model.set_page(startText="Go")
    model.set_warning("Hello", state="ok", strong=True)
    model.set_warning("Hello")
    model.set_warning("Hello", state="success")
    assert (model.warning_text, model.warning_state, model.warning_strong) == ("Hello", "idle", True)
    model.set_warning(tip="why")
    model.set_chip("gain", "G", state="warn")
    model.set_chip("gain", "G")
    assert model.chip("gain") == {"key": "gain", "text": "G", "tip": "", "state": "warn"}
    model.pickCurve(1)
    model.pickCurve(1)
    model.pickCurve(5)
    model.pickMode(0)
    model.pickSmoothing(1)
    assert (model.selected("curve"), model.selected("mode"), model.selected("smoothing")) == ("b", "m", "t")
    assert changes == ["page", "status", "status", "status", "chips", "choices", "choices"]
    for command in (model.start, model.override, model.cancel, model.retake, model.compare):
        command()
    assert calls == [1, "start", "override", "cancel", "retake", "compare"]


def test_a_scene_that_does_not_load_says_so_and_closes_without_starting_anything(qapp, monkeypatch) -> None:
    connected: list[object] = []
    shown: list[str] = []
    quitting = SimpleNamespace(aboutToQuit=SimpleNamespace(connect=connected.append))
    module = "mic_eq.ui.calibration_dialog"
    monkeypatch.setattr(f"{module}.QApplication", SimpleNamespace(instance=lambda: quitting))
    monkeypatch.setattr(f"{module}.load_scene", lambda *_args: "no such file")
    monkeypatch.setattr(f"{module}.QMessageBox.critical", lambda _parent, title, text: shown.append(f"{title}: {text}"))
    owner = _Owner()
    dialog = CalibrationDialog(owner, configuration=owner.configuration)
    assert dialog.load_error == "no such file"
    assert dialog.exec() == REJECTED
    assert len(shown) == 1 and shown[0].startswith("Auto-EQ Voice Tone: ") and "no such file" in shown[0]
    assert connected == [], "no quit hook"
    assert owner.processor.is_running() is False and owner.engine_session.view.user_muted is False
    assert not dialog._capture_start_timer.isActive() and not dialog.recording_timer.isActive()
    assert dialog._audio_lease is None and dialog._analysis_workers == []
    dialog.reject()
    assert dialog.result() == REJECTED
    dialog.deleteLater()
    owner.close()
    qapp.processEvents()
