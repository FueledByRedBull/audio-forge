"""What the setup dialogs tell assistive technology, and the order the keyboard visits them."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest

from capture_helpers import _voice, finish_take
from mic_eq.config import AppConfig
from mic_eq.ui.first_run_setup_dialog import FirstRunSetupDialog
from mic_eq.ui.theme import PALETTE
from test_first_run_setup import _Owner, _Processor, press, scene_item, scene_text
from test_setup_level_check_flow import voice  # noqa: F401  (a fixture)
from test_quick_shell import _AccessibleScene
from test_voice_setup_integration import real_main_window  # noqa: F401  (a fixture)


def _name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface is not None
    return interface.text(QAccessible.Text.Name)


def _first_run(
    monkeypatch, step: str = "devices", *, microphones: bool = True
) -> tuple[FirstRunSetupDialog, _Owner]:
    monkeypatch.setattr("mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True)
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step=step,
        first_run_setup_steps={name: "pending" for name in ("devices", "route", "latency", "voice")},
    )
    owner = _Owner(config, _Processor())
    if not microphones:
        owner.input_choice.clear()
    return FirstRunSetupDialog(owner), owner


def _open(dialog, qapp) -> None:
    """Show the dialog and give its scene the keyboard."""
    dialog.show()
    qapp.processEvents()
    dialog.activateWindow()
    dialog.quick_view.setFocus()
    qapp.processEvents()


def test_the_setup_action_button_is_named_by_the_text_it_shows_at_each_step(monkeypatch, qapp) -> None:
    dialog, owner = _first_run(monkeypatch, "route")
    _open(dialog, qapp)
    with _AccessibleScene() as scene:
        action = scene_item(dialog, "actionButton")
        assert scene_text(dialog, "actionButton") == "Check levels"
        assert scene.name(action) == "Check levels"
        press(dialog, "actionButton")
        dialog._finish_route_check()
        assert scene_text(dialog, "actionButton") == "Confirm signal"
        assert scene.name(action) == "Confirm signal"
        assert scene.name(scene_item(dialog, "backButton")) == "Back"
        assert scene.name(scene_item(dialog, "skipButton")) == "Skip this step"
        press(dialog, "actionButton")
        assert dialog.current_step == "voice"
        assert scene.name(action) == "Open Voice Setup"
    dialog.hide()
    dialog.deleteLater()


def test_the_setup_status_uses_the_failure_and_success_colours(monkeypatch, qapp) -> None:
    def colour(dialog) -> str:
        return scene_item(dialog, "statusText").property("color").name()

    dialog, owner = _first_run(monkeypatch, microphones=False)
    dialog._run_current_step()
    assert "must be available" in scene_text(dialog, "statusText")
    assert colour(dialog) == PALETTE.status_bad_text != PALETTE.status_idle_text

    dialog, owner = _first_run(monkeypatch, "route")
    dialog._run_current_step()
    owner.processor.running = False
    dialog._finish_route_check()
    assert colour(dialog) == PALETTE.status_bad_text
    owner.processor.running = True
    dialog._run_current_step()
    dialog._finish_route_check()
    assert "Confirm" in scene_text(dialog, "statusText") or "confirm" in scene_text(dialog, "statusText")
    assert colour(dialog) == PALETTE.status_ok_text != PALETTE.status_idle_text


@pytest.mark.parametrize("step", ["devices", "route", "voice"])
def test_the_first_run_dialog_visits_top_to_bottom_then_its_buttons_left_to_right(
    monkeypatch, qapp, step
) -> None:
    dialog, _owner = _first_run(monkeypatch, step)
    dialog.view_model.set_page(backEnabled=True)
    _open(dialog, qapp)
    view = dialog.quick_view
    visited: list[str] = []
    for _ in range(20):
        QTest.keyClick(view, Qt.Key.Key_Tab)
        name = view.quickWindow().activeFocusItem().objectName()
        if name in visited:
            break
        visited.append(name)
    buttons = ["pauseButton", "backButton", "skipButton", "actionButton"]
    # The page is a tab stop of its own, so the keyboard can scroll it.
    page = ["contentScroll", "muteSwitch", *(["inputPicker", "outputPicker"] if step == "devices" else []),
            "latencyButton"]
    assert visited == [*page, *buttons]
    ys = [scene_item(dialog, name).mapToScene(QPointF(0, 0)).y() for name in page]
    xs = [scene_item(dialog, name).mapToScene(QPointF(0, 0)).x() for name in buttons]
    assert ys == sorted(ys) and xs == sorted(xs)
    dialog.hide()
    dialog.deleteLater()


def test_the_recording_note_does_not_stand_over_a_running_analysis(voice, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(voice, "_start_analysis", lambda: None)
    voice.setup_state = "voice_recording"
    voice.view_model.set_page(timeText="Complete!")
    finish_take(voice, _voice(0.2, seconds=10.0))
    assert voice.setup_state == "analyzing"
    assert voice.view_model.shown["phaseText"] == "Analyzing recordings"
    assert voice.view_model.shown["timeText"] == "Analysis in progress"
