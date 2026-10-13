"""Shared shell state rejects view drift and reports commands without polling."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QStatusBar

from mic_eq.ui.components import Glyph, IconButton
from mic_eq.ui.shell_state import (
    ActionControl,
    ChoiceState,
    StatusState,
    TextState,
    bind_button,
)


def test_route_choices_keep_identities_and_ignore_signal_blocked_view_edits(qapp):
    state = ChoiceState(qapp, name="Output device")
    identities = [dict(endpoint_id="a", name="Same name"), dict(endpoint_id="b", name="Same name")]
    for identity in identities:
        state.addItem(identity["name"], identity)
    view = QComboBox()
    state.bind_combo(view)
    with QSignalBlocker(view):
        view.setCurrentIndex(1)
    assert state.currentData() == identities[0]
    state.setIndex(1)
    assert state.currentData() == identities[1]
    assert view.currentData() == identities[1]
    state.setEnabled(False)
    state.setIndex(0)
    assert state.currentData() == identities[1]


def test_route_refresh_publishes_final_state_without_command_side_effects(qapp):
    state = ChoiceState(qapp)
    view = QComboBox()
    state.bind_combo(view)
    commands = []
    state.currentIndexChanged.connect(commands.append)
    state.blockSignals(True)
    state.addItem("Missing device", "wrong")
    state.addItem("Replacement", "right")
    state.setCurrentIndex(1)
    state.blockSignals(False)
    assert commands == []
    assert view.currentData() == state.currentData() == "right"


def test_rejected_choice_reads_back_and_notifies_without_a_poll(qapp):
    state = ChoiceState(qapp)
    state.addItem("Normal", "normal")
    state.addItem("Unavailable", "other")
    events = []
    state.changed.connect(lambda: events.append(state.currentData()))
    state.currentIndexChanged.connect(lambda index: state.setCurrentIndex(0) if index == 1 else None)
    state.setIndex(1)
    assert state.currentData() == "normal"
    assert events[-1] == "normal"


def test_text_state_owns_health_and_preset_metadata(qapp):
    state = TextState(qapp, text="Ready")
    view = QLabel()
    state.bind_label(view)
    view.setText("Unaccepted view text")
    assert state.text == "Ready"
    state.update(text="Running", status="ok", data={"name": "Studio", "modified": False})
    assert view.text() == "Running"
    assert view.property("health_state") == "ok"
    assert state.data == {"name": "Studio", "modified": False}


def test_action_rejection_and_status_updates_are_immediate(qapp):
    action = QAction("Enable", qapp)
    action.setCheckable(True)
    action.triggered.connect(lambda: action.setChecked(False))
    control = ActionControl(action, qapp)
    notifications = []
    control.changed.connect(lambda: notifications.append(control.checked))
    control.click()
    assert notifications[-1] is False
    assert not control.checked
    status = StatusState(qapp, "Ready")
    status.showMessage("Rejected", 500)
    assert status.text == "Rejected"
    status.showMessage("Persistent")
    assert not status._expiry.isActive()


def test_views_take_names_and_tips_from_their_state_and_follow_it(qapp):
    choice = ChoiceState(qapp, name="Mode", tip="Pick one")
    choice.addItem("A", "a")
    combo = QComboBox()
    choice.bind_combo(combo)
    text = TextState(qapp, text="Ready", name="Status line")
    label = QLabel()
    text.bind_label(label)
    status = StatusState(qapp, "Sample rate")
    bar = QStatusBar()
    status.bind_status_bar(bar)
    assert (combo.accessibleName(), combo.toolTip()) == ("Mode", "Pick one")
    assert label.accessibleName() == "Status line"
    assert bar.currentMessage() == "Sample rate"
    status.showMessage("Saved")
    assert bar.currentMessage() == "Saved"


def test_button_renders_its_action_but_an_icon_button_keeps_its_glyph(qapp):
    action = QAction("Refresh", qapp)
    action.setToolTip("Refresh devices")
    icon = IconButton(Glyph.REFRESH, "Refresh devices")
    plain = QPushButton()
    glyph = icon.text()
    triggered = []
    action.triggered.connect(lambda: triggered.append(True))
    bind_button(icon, action)
    bind_button(plain, action)
    assert plain.text() == "Refresh"
    assert icon.text() == glyph
    action.setText("Rescan")
    action.setEnabled(False)
    assert (plain.text(), plain.isEnabled()) == ("Rescan", False)
    assert (icon.text(), icon.isEnabled()) == (glyph, False)
    action.setEnabled(True)
    plain.click()
    assert triggered == [True]
