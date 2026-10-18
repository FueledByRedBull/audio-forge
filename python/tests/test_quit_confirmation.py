"""Quitting while audio is live asks first; every other quit goes through as before."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import QMessageBox

from mic_eq.ui.main_window import MainWindow
from test_eq_editing_state import eq_control

_QUIT_CHOICE = MainWindow._quit_choice  # conftest replaces it for every other test


class _LiveProcessor:
    """Stands in for a running engine; everything else is the real processor."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.running = True

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False

    def __getattr__(self, name: str):
        return getattr(self._real, name)


@pytest.fixture
def live_window(isolated_main_window, monkeypatch):
    monkeypatch.setattr(MainWindow, "_quit_choice", _QUIT_CHOICE)
    window: Any = isolated_main_window
    window.processor = _LiveProcessor(window.processor)
    return window


@pytest.fixture
def answers(monkeypatch):
    """Questions asked, in order; the next live-audio box is answered by `reply`."""
    asked: list[Any] = []
    reply = {"button": "Cancel", "dont_ask": False}

    def exec_(box):
        asked.append([button.text() for button in box.buttons()])
        box.checkBox().setChecked(reply["dont_ask"])
        next(button for button in box.buttons() if button.text() == reply["button"]).click()
        return 0

    def question(_parent, title, *_args, **_kwargs):
        asked.append(title)
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    monkeypatch.setattr(QMessageBox, "question", question)
    return SimpleNamespace(asked=asked, reply=reply)


def _tray(window: Any, *, close_to_tray: bool = False) -> Mock:
    tray = Mock()
    tray.isVisible.return_value = True
    window._tray_icon = tray
    window._close_to_tray_action.setChecked(close_to_tray)
    return tray


def _close(window: Any) -> bool:
    event = QCloseEvent()
    window.closeEvent(event)
    return event.isAccepted()


def test_stopped_audio_quits_without_asking(live_window, answers):
    live_window.processor.running = False

    assert _close(live_window)
    assert answers.asked == []


def _quit_from_tray(window: Any) -> None:
    window._quit_from_tray()


@pytest.mark.parametrize("attempt", [_close, _quit_from_tray])
def test_live_audio_asks_and_cancel_keeps_everything_running(live_window, answers, attempt):
    attempt(live_window)
    assert answers.asked == [["Quit", "Cancel"]]
    assert live_window.processor.running and not live_window._quitting


def test_quit_stops_audio(live_window, answers):
    answers.reply["button"] = "Quit"

    assert _close(live_window)
    assert not live_window.processor.running


def test_keep_running_hides_to_the_tray_for_this_close_only(live_window, answers):
    tray = _tray(live_window)
    answers.reply["button"] = "Keep Running in Tray"

    assert not _close(live_window)
    assert answers.asked == [["Quit", "Keep Running in Tray", "Cancel"]]
    assert live_window.processor.running and live_window.isHidden()
    tray.showMessage.assert_called_once()
    assert not live_window._close_to_tray_action.isChecked()
    assert not live_window.config.close_to_tray


def test_close_to_tray_hides_without_asking(live_window, answers):
    _tray(live_window, close_to_tray=True)

    assert not _close(live_window)
    assert answers.asked == [] and live_window.processor.running


@pytest.mark.parametrize("button", ["Keep Running in Tray", "Cancel"])
def test_dont_ask_again_is_remembered_only_for_quit(live_window, answers, monkeypatch, button):
    saves: list[bool] = []
    monkeypatch.setattr(
        "mic_eq.ui.main_window.save_config",
        lambda config: saves.append(config.confirm_quit_when_live) or True,
    )
    _tray(live_window)
    answers.reply.update(button=button, dont_ask=True)
    _close(live_window)
    assert live_window.config.confirm_quit_when_live and saves == []

    answers.reply.update(button="Quit")
    assert _close(live_window)
    assert not live_window.config.confirm_quit_when_live and False in saves


@pytest.mark.parametrize("attempt", [_close, _quit_from_tray])
def test_dont_ask_again_waits_for_the_quit_to_go_ahead(
    live_window, answers, attempt, monkeypatch
):
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: None))
    answers.reply.update(button="Quit", dont_ask=True)
    discard = {"allowed": False}
    monkeypatch.setattr(live_window, "_confirm_discard_changes", lambda: discard["allowed"])

    attempt(live_window)
    assert live_window.config.confirm_quit_when_live and live_window.processor.running

    discard["allowed"] = True
    attempt(live_window)
    assert not live_window.config.confirm_quit_when_live


def test_dont_ask_again_suppresses_the_question(live_window, answers):
    live_window.config.confirm_quit_when_live = False

    assert _close(live_window)
    assert answers.asked == [] and not live_window.processor.running


def test_the_question_can_be_switched_back_on_from_settings(live_window, answers, monkeypatch):
    saves: list[bool] = []
    monkeypatch.setattr(
        "mic_eq.ui.main_window.save_config",
        lambda config: saves.append(config.confirm_quit_when_live) or True,
    )
    label = "Ask before quitting while audio is running"
    action = next(
        row["proxy"].action
        for card in live_window.shell_settings_cards
        for row in card["rows"]
        if row["proxy"].text == label
    )
    assert action.isChecked()

    action.setChecked(False)
    assert not live_window.config.confirm_quit_when_live and saves == [False]
    assert _close(live_window) and answers.asked == []

    live_window._quitting = False
    live_window.processor.running = True
    action.setChecked(True)
    assert live_window.config.confirm_quit_when_live and saves[-1] is True
    assert not _close(live_window) and answers.asked == [["Quit", "Cancel"]]


@pytest.mark.parametrize("close_to_tray", [False, True])
def test_a_windows_sign_out_never_asks_or_hides(live_window, answers, qapp, close_to_tray):
    _tray(live_window, close_to_tray=close_to_tray)
    qapp.commitDataRequest.emit(None)

    assert _close(live_window)
    assert answers.asked == [] and not live_window.processor.running


def test_tray_quit_with_close_to_tray_on_quits_without_asking(live_window, answers, monkeypatch):
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: None))
    _tray(live_window, close_to_tray=True)

    live_window._quit_from_tray()

    assert answers.asked == [] and not live_window.processor.running


def test_tray_quit_asks_once_then_asks_about_unsaved_changes(live_window, answers, monkeypatch):
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: None))
    answers.reply["button"] = "Quit"
    live_window._set_preset_modified(False)
    gain = eq_control(live_window, "gain", 4)
    gain.setValue(gain.value + 6.0)
    assert live_window.preset_modified

    live_window._quit_from_tray()

    assert answers.asked == [["Quit", "Cancel"], "Unsaved Sound Changes"]
    assert not live_window.processor.running


def test_tray_quit_keep_running_hides_the_window(live_window, answers):
    _tray(live_window)
    answers.reply["button"] = "Keep Running in Tray"

    live_window._quit_from_tray()

    assert live_window.isHidden() and live_window.processor.running and not live_window._quitting
