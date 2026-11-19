"""A Stop the user presses while audio is live asks first; every other stop goes through unasked."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from engine_fakes import FakeEngine
from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow

_CONFIRM_STOP = MainWindow._confirm_stop  # conftest replaces it for every other test


@pytest.fixture
def live(isolated_main_window):
    """A window whose engine is running, and the question it would ask before a Stop."""
    window: Any = isolated_main_window
    engine = FakeEngine()
    window.processor = window.engine_session.processor = engine
    engine.running = True
    window._confirm_stop = ask = Mock(return_value=True)
    window._sync_processing_controls()
    return window, engine, ask


def _action_with(window: Any, *, text: str | None = None, shortcut: str | None = None) -> QAction:
    return next(
        action
        for action in window.findChildren(QAction)
        if (text is None or action.text() == text)
        and (shortcut is None or action.shortcut() == QKeySequence(shortcut))
    )


def _button(window: Any) -> None:
    window.stop_btn_action.trigger()


def _menu(window: Any) -> None:
    _action_with(window, text="S&top Processing").trigger()


def _shortcut(window: Any) -> None:
    _action_with(window, shortcut="Ctrl+.").trigger()


USER_STOPS = [_button, _menu, _shortcut]
USER_STOP_IDS = ["button", "menu", "ctrl-period"]


@pytest.mark.parametrize("stop", USER_STOPS, ids=USER_STOP_IDS)
def test_a_user_stop_asks_first_and_confirming_stops(live, stop):
    window, engine, ask = live

    stop(window)

    ask.assert_called_once_with()
    assert engine.calls == ["stop"] and not engine.running


@pytest.mark.parametrize("stop", USER_STOPS, ids=USER_STOP_IDS)
def test_cancelling_keeps_audio_running_and_calls_nothing_on_the_engine(live, stop):
    window, engine, ask = live
    ask.return_value = False

    stop(window)

    ask.assert_called_once_with()
    assert engine.calls == [] and engine.running


@pytest.mark.parametrize("stop", USER_STOPS, ids=USER_STOP_IDS)
def test_a_stop_while_stopped_does_not_ask(live, stop):
    window, engine, ask = live
    engine.running = False

    stop(window)

    ask.assert_not_called()
    assert engine.calls == []


@pytest.mark.parametrize("stop", USER_STOPS, ids=USER_STOP_IDS)
def test_a_stop_while_the_engine_waits_to_retry_does_not_ask_and_cancels_the_retry(live, stop):
    window, engine, ask = live
    engine.running = False
    engine.restart_requested = True

    stop(window)

    ask.assert_not_called()
    assert engine.calls == ["stop"] and not engine.restart_requested


def test_a_stop_while_a_call_is_out_on_the_worker_does_not_ask(live):
    window, engine, ask = live
    window.engine_session._busy = "start"

    _button(window)

    ask.assert_not_called()
    assert engine.calls == []


def test_quitting_stops_audio_without_asking(live):
    window, engine, ask = live

    window.closeEvent(QCloseEvent())

    ask.assert_not_called()
    assert engine.calls == ["stop"]


def test_a_session_end_stops_audio_without_asking(live, qapp):
    window, engine, ask = live
    qapp.commitDataRequest.emit(None)

    window.closeEvent(QCloseEvent())

    ask.assert_not_called()
    assert engine.calls == ["stop"]


def test_the_own_microphone_stop_does_not_ask(live, monkeypatch):
    window, engine, ask = live
    monkeypatch.setattr(main_window, "records_own_microphone", lambda _input: True)
    monkeypatch.setattr(main_window, "note_own_microphone_refusal", lambda: None)

    assert window._stop_for_own_microphone() is True

    ask.assert_not_called()
    assert engine.calls == ["stop"]


def test_restart_audio_does_not_ask(live):
    window, engine, ask = live

    window._on_restart_audio_clicked()

    ask.assert_not_called()
    assert engine.calls[0] == "stop"


@pytest.mark.parametrize("recovered", [True, False], ids=["recovered", "still-gone"])
def test_device_loss_and_recovery_do_not_ask(live, recovered):
    window, engine, ask = live
    engine.running = False
    engine.restart_requested = True
    engine.recovery_result = recovered

    window.engine_session.tick()

    ask.assert_not_called()
    assert engine.calls == ["service_recovery"]


def test_audio_that_stopped_by_itself_while_the_box_was_open_is_not_stopped_twice(live):
    window, engine, ask = live

    def device_lost_then_confirmed() -> bool:
        engine.running = False
        return True

    ask.side_effect = device_lost_then_confirmed

    _button(window)

    ask.assert_called_once_with()
    # No second stop, and no "Failed to stop" box: the conftest guard refuses one.
    assert engine.calls == [] and not engine.running


def test_a_retry_pending_after_the_box_opened_is_cancelled_once(live):
    window, engine, ask = live

    def device_lost_then_confirmed() -> bool:
        engine.running = False
        engine.restart_requested = True
        return True

    ask.side_effect = device_lost_then_confirmed

    _button(window)

    assert engine.calls == ["stop"] and not engine.restart_requested


@pytest.fixture
def real_question(isolated_main_window, monkeypatch):
    """The real question; ``press`` says what a person does in the box instead of its modal loop."""
    monkeypatch.setattr(MainWindow, "_confirm_stop", _CONFIRM_STOP)
    seen: dict[str, Any] = {}

    def ask(press) -> bool:
        def exec_(box: QMessageBox) -> int:
            seen.update(
                text=box.text(),
                buttons=[button.text() for button in box.buttons()],
                default=box.defaultButton().text(),
                escape=box.escapeButton().text(),
            )
            box.show()
            press(box)
            return 0

        monkeypatch.setattr(QMessageBox, "exec", exec_)
        return isolated_main_window._confirm_stop()

    return ask, seen


def _key(key: Qt.Key):
    return lambda box: QTest.keyClick(box, key)


def test_the_question_names_the_callers_and_both_buttons(real_question):
    ask, seen = real_question

    ask(_key(Qt.Key.Key_Return))

    assert seen["text"] == "Callers will hear nothing until audio is started again."
    assert seen["buttons"] == ["Stop audio", "Keep running"]
    assert seen["default"] == "Stop audio" and seen["escape"] == "Keep running"


def test_enter_confirms_because_stop_audio_is_the_default(real_question):
    ask, _seen = real_question

    assert ask(_key(Qt.Key.Key_Return)) is True


def test_escape_cancels(real_question):
    ask, _seen = real_question

    assert ask(_key(Qt.Key.Key_Escape)) is False


def test_closing_the_box_cancels(real_question):
    ask, _seen = real_question

    assert ask(lambda box: box.close()) is False


def test_clicking_each_button_answers_for_it(real_question):
    ask, _seen = real_question

    def click(text: str):
        return lambda box: next(b for b in box.buttons() if b.text() == text).click()

    assert ask(click("Stop audio")) is True
    assert ask(click("Keep running")) is False
