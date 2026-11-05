"""The Guided Setup dialog is a Qt Quick scene: it loads, names its controls and takes the keyboard."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PySide6.QtCore import QPointF, Qt, qInstallMessageHandler
from PySide6.QtGui import QAccessible, QFont
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QWidget

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.ui.first_run_setup_dialog import (
    ACTION_LABELS,
    MUTE_TIP,
    STEP_CONTENT,
    FirstRunSetupDialog,
    FirstRunSetupLoadError,
)
from mic_eq.ui.first_run_setup_view import FirstRunSetupView
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.theme import PALETTE
from test_first_run_setup import (
    _Owner,
    _Processor,
    at_scene_minimum,
    choose,
    cut_short_texts,
    opening_setup_dialogs,
    press,
    scene_item,
    scene_items,
    scene_text,
)
from test_quick_shell import _AccessibleScene, _wait_until

STEPS = ["devices", "route", "voice"]
ACTION = {"devices": "Use this route", "route": "Check levels", "voice": "Open Voice Setup"}
BUTTONS = ["pauseButton", "backButton", "skipButton", "actionButton"]
# What the dialog can say on its status line, the longest first.
MESSAGES = [
    "Audio stream is healthy. Confirm the microphone signal in your destination app. "
    "Check your destination app, then confirm that it receives your microphone signal.",
    "Those devices are no longer available. Return to the main window, refresh the device "
    "list, and try again. Progress is unsaved.",
    "Progress is unsaved. You can continue, but some steps may need to be repeated if "
    "settings cannot be saved.",
    "Output is muted. Turn off Mute output above, then confirm the destination signal.",
    "No latency result was saved. Retry when needed, or continue with the standard setup.",
    "Voice Setup closed without applying a chain. Retry or skip this step.",
]


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


@pytest.fixture(autouse=True)
def saved(monkeypatch):
    """Nothing here writes the settings file."""
    monkeypatch.setattr("mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True)


@pytest.fixture
def open_setup(qapp):
    yield from opening_setup_dialogs(qapp)


def _owner(step: str = "devices", *, running: bool = False) -> _Owner:
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step=step,
        first_run_setup_steps={name: "pending" for name in ("devices", "route", "latency", "voice")},
    )
    return _Owner(config, _Processor(running=running))


def _focus_name(dialog: Any) -> str:
    return dialog.quick_view.quickWindow().activeFocusItem().objectName()


def _within_page(dialog: Any, item: Any) -> bool:
    scroll = scene_item(dialog, "contentScroll")
    top = item.mapToItem(scroll, QPointF(0, 0)).y()
    return top >= 0 and top + item.height() <= scroll.height()


def _at_minimum_size(dialog: Any, qapp: Any) -> None:
    dialog.resize(dialog.minimumSize())
    qapp.processEvents()
    qapp.processEvents()


@contextmanager
def _text_size(qapp: Any, large: bool = True):
    """The application font at 14 pt, or left alone."""
    previous = qapp.font()
    if large:
        bigger = QFont(previous)
        bigger.setPointSize(14)
        qapp.setFont(bigger)
    try:
        yield
    finally:
        qapp.setFont(previous)


def _tab_walk(dialog: Any) -> list[str]:
    """The items that real Tab presses focus in turn, until the walk comes round."""
    visited: list[str] = []
    for _ in range(20):
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Tab)
        name = _focus_name(dialog)
        if name in visited:
            break
        visited.append(name)
    return visited


def _accessible_names(dialog: Any) -> list[str]:
    """What assistive technology reads from the dialog, in tree order."""
    names: list[str] = []

    def walk(interface: Any) -> None:
        names.append(interface.text(QAccessible.Text.Name))
        for index in range(interface.childCount()):
            walk(interface.child(index))

    with _AccessibleScene():
        walk(QAccessible.queryAccessibleInterface(dialog.quick_view))
    return names


def test_the_window_keeps_the_title_the_release_check_looks_for(qapp, open_setup) -> None:
    assert open_setup(_owner()).windowTitle() == "AudioForge Setup"


@pytest.mark.parametrize("step", STEPS)
def test_the_scene_loads_and_every_control_has_its_accessible_name(qapp, open_setup, step) -> None:
    dialog = open_setup(_owner(step))
    assert dialog.quick_view.status() == QQuickWidget.Status.Ready and dialog.quick_view.errors() == []
    with _AccessibleScene() as scene:
        names = {
            "contentScroll": "Audio setup page",
            "progressBar": "Setup progress",
            "muteSwitch": "Mute output during setup",
            "inputPicker": "Setup microphone: Mic",
            "outputPicker": "Setup destination: Cable",
            "latencyButton": "Open advanced latency calibration",
            "inputMeter": "Live microphone level",
            "outputMeter": "Live destination level",
            "pauseButton": "Pause setup",
            "backButton": "Back",
            "skipButton": "Skip this step",
            "actionButton": ACTION[step],
        }
        assert {key: scene.name(scene_item(dialog, key)) for key in names} == names
        interface = QAccessible.queryAccessibleInterface(dialog.quick_view)
        assert interface is not None and interface.text(QAccessible.Text.Name) == "Audio setup content"
        assert scene.description(scene_item(dialog, "progressBar")) == "0 of 3 steps completed"
        assert scene.description(scene_item(dialog, "muteSwitch")) == MUTE_TIP
        mute = QAccessible.queryAccessibleInterface(scene_item(dialog, "muteSwitch"))
        assert mute is not None and mute.role() == QAccessible.Role.CheckBox and mute.state().checkable
        for key in BUTTONS:
            button = QAccessible.queryAccessibleInterface(scene_item(dialog, key))
            assert button is not None and button.role() == QAccessible.Role.Button, key


@pytest.mark.parametrize("step", STEPS)
def test_a_button_name_contains_the_text_the_button_shows(qapp, open_setup, step) -> None:
    dialog = open_setup(_owner(step))
    with _AccessibleScene() as scene:
        for key in ("pauseButton", "backButton", "skipButton", "actionButton", "latencyButton"):
            shown = " ".join(scene_text(dialog, key).split()).rstrip(".").casefold()
            assert shown in scene.name(scene_item(dialog, key)).casefold(), key


@pytest.mark.parametrize("step", STEPS)
def test_every_text_the_dialog_shows_is_read_once_through_accessibility(qapp, open_setup, step) -> None:
    dialog = open_setup(_owner(step))
    names = _accessible_names(dialog)
    title, description, _action = STEP_CONTENT[step]
    for text in (title, description, "Ready. Progress is saved if you close this window."):
        assert names.count(text) == 1, text
    if step == "route":
        assert names.count("Input: OK") == 0, "no stream is running, so there is no reading yet"
        assert names.count("Input: --") == 1 and names.count("Output: --") == 1
    # The captions are read with their pickers, the progress text with its bar.
    assert "Microphone" not in names and "Destination" not in names
    assert "0 of 3 steps completed" not in names


def test_a_text_that_changes_is_read_with_its_new_text(qapp, open_setup) -> None:
    dialog = open_setup(_owner("route", running=True))
    with _AccessibleScene() as scene:
        dialog.view_model.set_status("Moving closer helps", "warn")
        assert scene.name(scene_item(dialog, "statusText")) == "Moving closer helps"
        owner = dialog.owner
        owner.shell_texts["output_health_label"].update(text="Output: CLIPPING", status="bad")
        dialog._update_route_feedback()
        assert scene.name(scene_item(dialog, "outputHealth")) == "Output: CLIPPING"
        owner.user_mute_checkbox.setChecked(True)
        assert scene.name(scene_item(dialog, "muteSwitch")) == "Mute output during setup"
        interface = QAccessible.queryAccessibleInterface(scene_item(dialog, "muteSwitch"))
        assert interface is not None and interface.state().checked


@pytest.mark.parametrize("state", ["ok", "warn", "bad", "info", "idle"])
def test_a_message_is_set_in_the_colour_of_its_state(qapp, open_setup, state) -> None:
    dialog = open_setup(_owner())
    dialog.view_model.set_status("A message", state)
    assert scene_item(dialog, "statusText").property("color").name() == getattr(PALETTE, f"status_{state}_text")


@pytest.mark.parametrize("state", ["error", "success", "", "OK"])
def test_a_state_the_scene_has_no_colours_for_is_shown_as_idle(qapp, open_setup, state) -> None:
    dialog = open_setup(_owner("route", running=True))
    dialog.view_model.set_status("A message", state)
    dialog.view_model.set_health("output", "Output: reading", state)
    assert dialog.view_model.status_state == "idle"
    assert scene_item(dialog, "statusText").property("color").name() == PALETTE.status_idle_text
    chip = scene_item(dialog, "outputHealth")
    assert chip.property("tone") == "idle"
    assert chip.property("color").name() == PALETTE.status_idle_surface


@pytest.mark.parametrize("state", ["ok", "warn", "bad", "info", "idle"])
def test_a_health_reading_is_set_off_in_the_colours_of_its_state(qapp, open_setup, state) -> None:
    dialog = open_setup(_owner("route", running=True))
    dialog.view_model.set_health("output", "Output: reading", state)
    chip = scene_item(dialog, "outputHealth")
    assert chip.property("text") == "Output: reading" and chip.property("tone") == state
    assert chip.property("color").name() == getattr(PALETTE, f"status_{state}_surface")
    assert chip.childItems()[0].property("color").name() == getattr(PALETTE, f"status_{state}_text")


@pytest.mark.parametrize(
    ("step", "page"),
    [
        ("devices", ["contentScroll", "muteSwitch", "inputPicker", "outputPicker", "latencyButton"]),
        ("route", ["contentScroll", "muteSwitch", "latencyButton"]),
        ("voice", ["contentScroll", "muteSwitch", "latencyButton"]),
    ],
)
def test_tab_visits_the_page_then_pause_back_skip_and_the_main_button(qapp, open_setup, step, page) -> None:
    dialog = open_setup(_owner(step))
    dialog.view_model.set_page(backEnabled=True)
    assert _tab_walk(dialog) == [*page, *BUTTONS]
    xs = [scene_item(dialog, name).mapToScene(QPointF(0, 0)).x() for name in BUTTONS]
    assert xs == sorted(xs)


def test_tab_leaves_out_what_cannot_be_used(qapp, open_setup) -> None:
    owner = _owner()
    vars(owner)["_on_latency_calibration_clicked"] = None  # an owner that cannot open it
    dialog = open_setup(owner)
    visited = _tab_walk(dialog)
    assert "backButton" not in visited, "Back is off on the first step"
    assert "latencyButton" not in visited, "there is no latency calibration to open"
    assert visited[-3:] == ["pauseButton", "skipButton", "actionButton"]

    # The advanced step has no latency button of its own.
    owner = _owner("latency")
    dialog = open_setup(owner)
    assert dialog.current_step == "latency"
    visited = _tab_walk(dialog)
    assert "latencyButton" not in visited and visited[-4:] == BUTTONS


def test_enter_presses_the_focused_button_and_escape_closes_the_dialog(qapp, open_setup, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(FirstRunSetupDialog, "_skip_step", lambda self: calls.append("skip"))
    dialog = open_setup(_owner())
    scene_item(dialog, "skipButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Enter)
    assert calls == ["skip", "skip"]

    scene_item(dialog, "pauseButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    qapp.processEvents()
    assert not dialog.isVisible() and dialog.result() == dialog.DialogCode.Rejected

    again = open_setup(_owner())
    QTest.keyClick(again.quick_view, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not again.isVisible() and again.result() == again.DialogCode.Rejected


def test_enter_outside_a_button_does_what_the_main_button_does(qapp, open_setup, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(FirstRunSetupDialog, "_run_current_step", lambda self: calls.append("act"))
    dialog = open_setup(_owner())
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    assert calls == ["act"]
    scene_item(dialog, "contentScroll").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Enter)
    assert calls == ["act", "act"]
    # On a switch it flips the switch, as on any button.
    scene_item(dialog, "muteSwitch").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    assert calls == ["act", "act"] and dialog.owner.user_muted is True
    # A held key acts once, and a main button that is off does nothing.
    dialog.view_model.set_page(actionEnabled=False)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Return)
    assert calls == ["act", "act"]


@pytest.mark.parametrize("step", STEPS)
def test_the_actions_fit_and_the_page_scrolls_only_vertically_in_a_small_dialog(qapp, open_setup, step) -> None:
    dialog = open_setup(_owner(step, running=True))
    for message in MESSAGES:
        dialog.view_model.set_status(message, "warn")
    _at_minimum_size(dialog, qapp)
    scroll, column = scene_item(dialog, "contentScroll"), scene_item(dialog, "contentColumn")
    right = scroll.mapToScene(QPointF(scroll.width(), 0)).x()
    wide = [
        (item.objectName(), item.mapToScene(QPointF(item.width(), 0)).x())
        for item in scene_items(column, only_visible=True)
        if item.mapToScene(QPointF(item.width(), 0)).x() > right + 0.5
    ]
    assert wide == [], "items run past the right edge of the page"
    if step != "voice":  # the voice step is short enough to fit on a wide screen
        assert scroll.property("contentHeight") > scroll.height()
    for key in BUTTONS:
        button = scene_item(dialog, key)
        assert button.mapToScene(QPointF(button.width(), 0)).x() <= dialog.quick_view.width(), key


def _seen_while_scrolling(dialog: Any, qapp: Any) -> set[str]:
    """The names of the items that are wholly in view at some scroll position, by Down presses."""
    scroll = scene_item(dialog, "contentScroll")
    scroll.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    QTest.keyClick(dialog.quick_view, Qt.Key.Key_Home)
    column = scene_item(dialog, "contentColumn")
    seen: set[str] = set()
    for _ in range(80):
        qapp.processEvents()
        seen |= {
            item.objectName()
            for item in scene_items(column, only_visible=True)
            if item.objectName() and _within_page(dialog, item)
        }
        before = scroll.property("contentY")
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Down)
        if scroll.property("contentY") == before:
            break
    return seen


def _show_status(dialog: Any, message: str) -> None:
    """Set a warning and wait for the layout to give the rewrapped text its new height."""
    dialog.view_model.set_status(message, "warn")
    status = scene_item(dialog, "statusText")
    assert _wait_until(lambda: status.height() == status.implicitHeight())


@pytest.mark.parametrize("step", STEPS)
def test_the_keyboard_alone_scrolls_a_small_dialog_to_every_part_of_the_page(qapp, open_setup, step) -> None:
    dialog = open_setup(_owner(step, running=True))
    _at_minimum_size(dialog, qapp)
    scroll = scene_item(dialog, "contentScroll")
    if step != "voice":  # the voice step is short enough to fit on a wide screen
        assert scroll.property("contentHeight") > scroll.height()
    expected = {"progressBar", "titleText", "descriptionText", "statusText", "muteSwitch", "latencyButton"}
    expected |= {"devicesPanel", "inputPicker", "outputPicker"} if step == "devices" else set()
    expected |= {"routePanel", "inputMeter", "outputMeter", "inputHealth", "outputHealth"} if step == "route" else set()
    for message in MESSAGES:
        _show_status(dialog, message)
        assert expected <= _seen_while_scrolling(dialog, qapp), message

    def press_key(key: Qt.Key) -> float:
        QTest.keyClick(dialog.quick_view, key)
        qapp.processEvents()
        return scroll.property("contentY")

    _show_status(dialog, MESSAGES[0])
    if scroll.property("contentHeight") <= scroll.height():
        assert step == "voice"  # a page that fits has nothing to scroll
        return
    scroll.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    assert press_key(Qt.Key.Key_Home) == 0 and press_key(Qt.Key.Key_Up) == 0
    page = press_key(Qt.Key.Key_PageDown)
    assert 0 < page <= scroll.height()
    assert press_key(Qt.Key.Key_PageUp) < page
    last = press_key(Qt.Key.Key_End)
    assert last == scroll.property("contentHeight") - scroll.height()
    assert press_key(Qt.Key.Key_Down) == last


def test_the_page_shows_a_focus_ring_when_it_has_the_keyboard(qapp, open_setup) -> None:
    dialog = open_setup(_owner())
    ring = scene_item(dialog, "scrollFocusRing")
    assert not ring.isVisible()
    scene_item(dialog, "contentScroll").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()
    assert ring.isVisible()


def test_a_control_that_takes_the_keyboard_is_scrolled_into_view(qapp, open_setup) -> None:
    dialog = open_setup(_owner("route"))
    _at_minimum_size(dialog, qapp)
    scroll, button = scene_item(dialog, "contentScroll"), scene_item(dialog, "latencyButton")
    scroll.setProperty("contentY", 0)
    qapp.processEvents()
    assert not _within_page(dialog, button)
    button.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()
    assert _within_page(dialog, button)


@pytest.mark.parametrize("large", [False, True], ids=["normal text", "14 pt"])
@pytest.mark.parametrize("step", STEPS)
def test_no_text_is_cut_short_and_nothing_overflows_at_the_smallest_size(qapp, step, large, open_setup) -> None:
    with _text_size(qapp, large):
        dialog = open_setup(_owner(step, running=True))
        # Judged at the scene's own minimum, which a small or font-less screen cannot hold.
        at_scene_minimum(dialog, qapp)
        for key in BUTTONS:  # a text that is not there cannot be seen to be cut short
            assert [i for i in scene_items(scene_item(dialog, key)) if i.property("truncated") is not None], key
        assert cut_short_texts(dialog) == []
        column = scene_item(dialog, "contentColumn")
        scroll = scene_item(dialog, "contentScroll")
        right = scroll.mapToScene(QPointF(scroll.width(), 0)).x()
        assert [
            item.objectName()
            for item in scene_items(column, only_visible=True)
            if item.mapToScene(QPointF(item.width(), 0)).x() > right + 0.5
        ] == []
        for key in BUTTONS:
            button = scene_item(dialog, key)
            assert button.mapToScene(QPointF(button.width(), 0)).x() <= dialog.quick_view.width(), key


@pytest.mark.parametrize(
    ("step", "button", "owner_method"),
    [
        ("voice", "actionButton", "_on_auto_voice_setup_clicked"),
        ("devices", "latencyButton", "_on_latency_calibration_clicked"),
        ("route", "latencyButton", "_on_latency_calibration_clicked"),
    ],
)
@pytest.mark.parametrize("how", ["mouse", "keyboard"])
def test_the_keyboard_comes_back_to_the_button_that_opened_a_sub_dialog(
    qapp, open_setup, step, button, owner_method, how
) -> None:
    owner = _owner(step)
    calls: list[str] = []
    vars(owner)[owner_method] = lambda: calls.append(owner_method) or False
    dialog = open_setup(owner)
    item = scene_item(dialog, button)
    item.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    assert _focus_name(dialog) == button
    if how == "mouse":
        press(dialog, button)
    else:
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    qapp.processEvents()
    assert calls == [owner_method] and dialog.isVisible()
    assert _focus_name(dialog) == button


def test_the_keyboard_goes_to_the_main_button_when_the_launching_button_is_gone(qapp, open_setup) -> None:
    owner = _owner("devices")
    dialog = open_setup(owner)
    latency = scene_item(dialog, "latencyButton")
    latency.forceActiveFocus(Qt.FocusReason.TabFocusReason)

    def hides_its_launcher() -> bool:
        dialog.view_model.set_page(latencyShown=False)
        return False

    vars(owner)["_on_latency_calibration_clicked"] = hides_its_launcher
    dialog._run_latency_calibration()
    qapp.processEvents()
    assert not latency.isVisible()
    assert _focus_name(dialog) == "actionButton"


def test_the_dialog_is_never_narrower_than_its_widest_main_button(qapp, open_setup) -> None:
    with _text_size(qapp):
        dialog = open_setup(_owner())
        root = dialog.quick_view.rootObject()
        widest = max(len(label) for label in ACTION_LABELS)
        assert widest == len("Open latency calibration")
        assert root.property("minimumWidth") >= root.property("actionsWidth") > 0
        assert root.property("captionWidth") >= 70
        assert dialog.minimumWidth() == min(
            max(440, int(root.property("minimumWidth"))), dialog.screen().availableGeometry().width() - 48
        )


@pytest.mark.parametrize("how", ["mouse", "keyboard"])
@pytest.mark.parametrize(
    ("button", "method", "step"),
    [
        ("pauseButton", "reject", "devices"),
        ("backButton", "_go_back", "route"),
        ("skipButton", "_skip_step", "devices"),
        ("actionButton", "_run_current_step", "devices"),
        ("latencyButton", "_run_latency_calibration", "devices"),
    ],
)
def test_pressing_a_scene_button_runs_the_dialogs_command(
    qapp, open_setup, monkeypatch, how, button, method, step
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(FirstRunSetupDialog, method, lambda self: calls.append(method))
    dialog = open_setup(_owner(step))
    item = scene_item(dialog, button)
    assert item.isEnabled()
    if how == "mouse":
        press(dialog, button)
    else:
        item.forceActiveFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)
    qapp.processEvents()
    assert calls == [method]


def test_a_disabled_scene_button_does_nothing(qapp, open_setup, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(FirstRunSetupDialog, "_go_back", lambda self: calls.append("back"))
    dialog = open_setup(_owner("devices"))
    assert not scene_item(dialog, "backButton").isEnabled()
    press(dialog, "backButton")
    assert calls == []


@pytest.mark.parametrize("how", ["mouse", "keyboard"])
def test_the_mute_switch_turns_the_output_mute_on_and_off(qapp, open_setup, how) -> None:
    owner = _owner()
    dialog = open_setup(owner)
    switch = scene_item(dialog, "muteSwitch")

    def flip() -> None:
        if how == "mouse":
            press(dialog, "muteSwitch")
        else:
            switch.forceActiveFocus(Qt.FocusReason.TabFocusReason)
            QTest.keyClick(dialog.quick_view, Qt.Key.Key_Space)

    flip()
    assert owner.user_muted is True and switch.property("checked") is True
    flip()
    assert owner.user_muted is False and switch.property("checked") is False
    assert owner.mute_toggles == [True, False]


def test_the_mute_switch_follows_the_owner_not_its_own_click(qapp, open_setup) -> None:
    owner = _owner()
    vars(owner)["user_mute_checkbox_action"] = SimpleNamespace(setChecked=lambda _checked: None)
    dialog = open_setup(owner)
    switch = scene_item(dialog, "muteSwitch")
    press(dialog, "muteSwitch")
    # The dialog shows the click at once, and the owner's own change takes it back.
    assert switch.property("checked") is True
    dialog._sync_mute_control()
    assert switch.property("checked") is False


def test_the_mute_switch_explains_itself_when_it_has_the_keyboard(qapp, open_setup) -> None:
    dialog = open_setup(_owner())
    switch = scene_item(dialog, "muteSwitch")
    tips = [child for child in switch.children() if child.metaObject().className().startswith("Tip")]
    assert len(tips) == 1 and not tips[0].property("visible")
    switch.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    assert _wait_until(lambda: tips[0].property("visible")), "the tip shows after its delay"
    assert tips[0].property("text") == MUTE_TIP
    scene_item(dialog, "skipButton").forceActiveFocus(Qt.FocusReason.TabFocusReason)
    assert _wait_until(lambda: not tips[0].property("visible")), "the tip goes when the focus does"


@pytest.mark.parametrize("how", ["mouse", "keyboard"])
def test_the_pickers_choose_the_microphone_and_the_destination(qapp, open_setup, how) -> None:
    owner = _owner()
    owner.input_choice.addItem("USB Mic", DeviceIdentity(name="USB Mic", endpoint_id="usb", direction="input"))
    owner.output_choice.addItem("Cable 2", DeviceIdentity(name="Cable 2", endpoint_id="c2", direction="output"))
    dialog = open_setup(owner)
    for picker, index in (("inputPicker", 1), ("outputPicker", 1)):
        if how == "keyboard":
            choose(dialog, picker, index)
        else:
            _choose_with_the_mouse(dialog, qapp, picker, index)
    assert (dialog.view_model.input_index, dialog.view_model.output_index) == (1, 1)
    with _AccessibleScene() as scene:
        assert scene.name(scene_item(dialog, "inputPicker")) == "Setup microphone: USB Mic"
        assert scene.name(scene_item(dialog, "outputPicker")) == "Setup destination: Cable 2"
    press(dialog, "actionButton")
    assert owner.input_choice.currentData().endpoint_id == "usb"
    assert dialog.current_step == "route"


def _choose_with_the_mouse(dialog: Any, qapp: Any, picker: str, index: int) -> None:
    press(dialog, picker)
    QTest.qWait(50)
    # The list is drawn over the page, so it is a child of the window, not of the picker.
    options = [
        item
        for item in scene_items(dialog.quick_view.quickWindow().contentItem(), only_visible=True)
        if item.property("modelData") is not None and item.property("highlighted") is not None
    ]
    assert options, "the list opens"
    option = next(item for item in options if item.property("index") == index)
    centre = option.mapToScene(option.boundingRect().center()).toPoint()
    QTest.mouseClick(dialog.quick_view, Qt.MouseButton.LeftButton, pos=centre)
    QTest.qWait(50)
    assert not [
        item for item in scene_items(dialog.quick_view.quickWindow().contentItem(), only_visible=True)
        if item.property("modelData") is not None and item.property("highlighted") is not None
    ], "the list closes"


def test_a_picker_with_no_devices_says_so_and_the_step_refuses(qapp, open_setup) -> None:
    owner = _owner()
    owner.output_choice.clear()
    dialog = open_setup(owner)
    picker = scene_item(dialog, "outputPicker")
    assert picker.property("currentIndex") == -1 and picker.property("count") == 0
    press(dialog, "actionButton")
    assert dialog.current_step == "devices"
    assert "must be available" in scene_text(dialog, "statusText")


def test_the_level_check_refuses_while_mute_output_is_on_and_runs_once_it_is_off(qapp, open_setup) -> None:
    owner = _owner("route")
    owner.user_mute_checkbox.setChecked(True)
    dialog = open_setup(owner)
    assert scene_item(dialog, "muteSwitch").property("checked") is True
    press(dialog, "actionButton")
    assert owner.start_calls == 0
    assert "Output is muted" in scene_text(dialog, "statusText")
    assert scene_item(dialog, "actionButton").isEnabled()

    press(dialog, "muteSwitch")
    assert owner.user_muted is False
    press(dialog, "actionButton")
    assert owner.start_calls == 1 and dialog._route_check_timer.isActive()
    assert "Checking the audio stream" in scene_text(dialog, "statusText")


def test_closing_the_window_during_a_level_check_stops_what_the_dialog_started(qapp, open_setup) -> None:
    owner = _owner("route")
    dialog = open_setup(owner)
    press(dialog, "actionButton")
    assert dialog._route_check_timer.isActive() and owner.processor.running
    assert dialog.close()
    qapp.processEvents()
    assert owner.stop_calls == 1 and not owner.processor.running
    assert not dialog._route_check_timer.isActive() and not dialog._route_feedback_timer.isActive()
    assert dialog.result() == dialog.DialogCode.Rejected


def test_a_rerun_opens_on_the_first_device_step(qapp, open_setup) -> None:
    config = AppConfig(
        first_run_setup_state="completed",
        first_run_setup_step="voice",
        first_run_setup_steps={name: "completed" for name in ("devices", "route", "latency", "voice")},
    )
    dialog = open_setup(_Owner(config, _Processor()), restart_completed=True)
    assert scene_text(dialog, "titleText") == "1. Choose devices"
    assert scene_item(dialog, "devicesPanel").isVisible()


@pytest.mark.parametrize("step", STEPS)
def test_each_step_shows_its_own_title_text_and_parts(qapp, open_setup, step) -> None:
    dialog = open_setup(_owner(step, running=True))
    title, description, action = STEP_CONTENT[step]
    assert scene_text(dialog, "titleText") == title
    assert scene_text(dialog, "descriptionText") == description
    assert scene_text(dialog, "actionButton") == action
    assert scene_item(dialog, "devicesPanel").isVisible() == (step == "devices")
    assert scene_item(dialog, "routePanel").isVisible() == (step == "route")
    assert scene_item(dialog, "latencyButton").isVisible()
    assert scene_text(dialog, "progressText") == "0 of 3 steps completed"


def test_the_view_model_reports_changes_once_and_forwards_its_commands(qapp) -> None:
    calls: list[object] = []
    model = FirstRunSetupView(
        pause=lambda: calls.append("pause"),
        back=lambda: calls.append("back"),
        skip=lambda: calls.append("skip"),
        act=lambda: calls.append("act"),
        open_latency=lambda: calls.append("latency"),
        request_mute=lambda muted: calls.append(("mute", muted)),
        action_labels=["Go"],
        inputs=["Mic"],
        outputs=["Cable"],
        input_index=0,
        output_index=0,
        mute_tip="tip",
    )
    changes: list[str] = []
    model.statusChanged.connect(lambda: changes.append("status"))
    model.pageChanged.connect(lambda: changes.append("page"))
    model.healthChanged.connect(lambda: changes.append("health"))
    model.mutedChanged.connect(lambda: changes.append("muted"))
    model.set_status("Ready", "info")
    model.set_status("Ready", "info")
    model.set_status("Ready", "bad")
    model.set_page(title="One")
    model.set_page(title="One")
    model.set_health("input", "Input: OK", "ok")
    model.set_health("input", "Input: OK", "ok")
    model.set_muted(False)
    model.set_muted(True)
    assert changes == ["status", "status", "page", "health", "muted"]
    page, health = cast(Any, model).page, cast(Any, model).health
    assert page["title"] == "One" and health["input"]["text"] == "Input: OK"
    assert health["output"]["state"] == "idle" and cast(Any, model).muted
    model.pickInput(0)
    model.pickOutput(0)
    model.pause()
    model.back()
    model.skip()
    model.act()
    model.openLatency()
    model.requestMute(True)
    assert calls == ["pause", "back", "skip", "act", "latency", ("mute", True)]


def test_a_scene_that_does_not_load_raises_before_progress_is_touched(qapp, monkeypatch) -> None:
    saves: list[object] = []
    monkeypatch.setattr("mic_eq.ui.first_run_setup_dialog.save_config", lambda config: saves.append(config) or True)
    monkeypatch.setattr("mic_eq.ui.first_run_setup_dialog.load_scene", lambda *_args: "no such file")
    config = AppConfig(
        first_run_setup_state="completed",
        first_run_setup_step="voice",
        first_run_setup_steps={name: "completed" for name in ("devices", "route", "latency", "voice")},
    )
    owner = _Owner(config, _Processor())
    with pytest.raises(FirstRunSetupLoadError, match="no such file"):
        FirstRunSetupDialog(owner, restart_completed=True)
    assert saves == []
    assert config.first_run_setup_state == "completed" and config.first_run_setup_step == "voice"
    assert set(config.first_run_setup_steps.values()) == {"completed"}
    assert owner.findChildren(QDialog) == []
    owner.deleteLater()
    qapp.processEvents()


def test_a_setup_that_cannot_load_tells_the_user_and_opens_nothing(monkeypatch, caplog) -> None:
    def broken(*_args, **_kwargs):
        raise FirstRunSetupLoadError("The setup view did not load: no details")

    shown: list[tuple] = []
    monkeypatch.setattr("mic_eq.ui.main_window.FirstRunSetupDialog", broken)
    monkeypatch.setattr("mic_eq.ui.main_window.QMessageBox.critical", lambda *args: shown.append(args))
    MainWindow._show_first_run_setup(cast(Any, SimpleNamespace()))
    assert len(shown) == 1 and "could not open" in shown[0][2]
    assert "no details" not in shown[0][2], "the details are for the log, not the message box"
    assert "no details" in caplog.text
    assert isinstance(shown[0][0], SimpleNamespace) or isinstance(shown[0][0], QWidget)
