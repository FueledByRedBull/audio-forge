"""When Qt Quick cannot start, the window says so instead of showing another view."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtQuick import QQuickWindow
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSystemTrayIcon,
    QWidget,
)
from shiboken6 import isValid

from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.quick_shell import QuickBridge, graphics_api
from test_login_startup_ui import startup_window  # noqa: F401  (a fixture)
from test_quick_shell import _load_config_without_devices, quick_window  # noqa: F401  (a fixture)

ROOT = Path(__file__).resolve().parents[2]


def _bridge_raises(window_class, monkeypatch, tmp_path):
    def refuse(_self, _window):
        raise RuntimeError("model could not be built")

    monkeypatch.setattr("mic_eq.ui.quick_shell.QuickBridge._build_model", refuse)
    return window_class(), "bridge_failed", "RuntimeError: model could not be built", "not_applicable"


def _scene_not_ready(window_class, monkeypatch, tmp_path):
    monkeypatch.setattr("mic_eq.ui.quick_shell.QML_DIR", tmp_path)
    return window_class(), "qml_not_ready", "Main.qml", "not_applicable"


def _scene_graph_error(window_class, monkeypatch, tmp_path):
    window = window_class()
    window.show()
    view = window.centralWidget()
    assert isinstance(view, QQuickWidget)
    api = graphics_api(window)
    # The scene's backend, not the placeholder, is what the export must keep.
    assert api == "software"
    view.sceneGraphError.emit(
        QQuickWindow.SceneGraphError.ContextNotAvailable, "no graphics adapter"
    )
    return window, "scene_graph_error", "no graphics adapter", api


@pytest.fixture(params=[_bridge_raises, _scene_not_ready, _scene_graph_error])
def failed_window(request, qapp, monkeypatch, tmp_path):
    _load_config_without_devices(monkeypatch)
    window, code, reason, api = request.param(MainWindow, monkeypatch, tmp_path)
    window.code, window.reason, window.api = code, reason, api
    return window


def _label(window, name: str) -> QLabel:
    label = window.centralWidget().findChild(QLabel, name)
    assert label is not None
    return label


def _reason(window) -> QPlainTextEdit:
    box = window.centralWidget().findChild(QPlainTextEdit, "errorReason")
    assert box is not None
    return box


def _button(window, text: str) -> QPushButton:
    buttons = [b for b in window.centralWidget().findChildren(QPushButton) if b.text() == text]
    assert len(buttons) == 1
    return buttons[0]


def test_the_window_shows_the_error_screen(failed_window) -> None:
    window = failed_window
    screen = window.centralWidget()
    assert isinstance(screen, QWidget) and not isinstance(screen, QQuickWidget)
    assert window.quick_view_failure == window.code
    assert "quick_bridge" not in vars(window)


def test_the_screen_says_what_happened_in_plain_words_and_gives_the_reason(failed_window) -> None:
    title, body = _label(failed_window, "errorTitle"), _label(failed_window, "errorBody")
    assert not title.isHidden() and not body.isHidden()
    assert "could not open its main window" in title.text()
    # It says nothing is shown, points at a button the screen really has, and says what to do after.
    assert "nothing to show" in body.text()
    assert "Export diagnostics if you want to report this" in body.text()
    assert _button(failed_window, "Export diagnostics")
    assert "quit and open AudioForge again" in body.text()
    assert _label(failed_window, "errorReasonCaption").text() == "Reason"
    assert failed_window.reason in _reason(failed_window).toPlainText()


def test_the_export_diagnostics_button_writes_a_file_that_names_the_failure(
    failed_window, monkeypatch, tmp_path
) -> None:
    target = tmp_path / "diagnostics.json"
    monkeypatch.setattr(
        "mic_eq.ui.main_window.QFileDialog.getSaveFileName", lambda *_a, **_k: (str(target), "")
    )
    shown: list[str] = []
    monkeypatch.setattr(
        "mic_eq.ui.main_window.QMessageBox.information", lambda _p, title, *_a: shown.append(title)
    )
    button = _button(failed_window, "Export diagnostics")
    assert button.isEnabled()
    button.click()
    assert shown == ["Diagnostics Exported"]
    assert json.loads(target.read_text(encoding="utf-8"))["interface"] == {
        "view": "error",
        "classic_reason": failed_window.code,
        "graphics_api": failed_window.api,
    }


def test_the_quit_button_quits(failed_window, monkeypatch) -> None:
    quits: list[bool] = []
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: quits.append(True)))
    button = _button(failed_window, "Quit")
    assert button.isEnabled()
    button.click()
    assert quits == [True] and failed_window._quitting


def test_a_scene_that_stops_rendering_is_replaced_once(quick_window) -> None:  # noqa: F811
    window = quick_window
    scene = window.centralWidget()
    assert isinstance(scene, QQuickWidget)
    scene.sceneGraphError.emit(QQuickWindow.SceneGraphError.ContextNotAvailable, "first")
    screen = window.centralWidget()
    assert screen is not scene
    # A second report from the dead scene changes nothing.
    scene.sceneGraphError.emit(QQuickWindow.SceneGraphError.ContextNotAvailable, "second")
    assert window.centralWidget() is screen
    assert _reason(window).toPlainText() == "first"
    # The dead scene is deleted, not left behind a second central widget.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(scene)


def test_enter_presses_the_focused_button(failed_window, monkeypatch) -> None:
    exports: list[bool] = []
    quits: list[bool] = []
    monkeypatch.setattr(failed_window, "_export_diagnostics", lambda: exports.append(True))
    monkeypatch.setattr(failed_window, "_quit_from_tray", lambda: quits.append(True))
    failed_window.show()
    QCoreApplication.processEvents()
    export, quit_button = _button(failed_window, "Export diagnostics"), _button(failed_window, "Quit")
    assert failed_window.focusWidget() is export
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
        QTest.keyClick(export, key)
    QTest.keyClick(quit_button, Qt.Key.Key_Return)
    assert exports == [True, True] and quits == [True]


def test_the_reason_is_read_only_selectable_and_cannot_push_the_buttons_off_screen(
    failed_window,
) -> None:
    box = _reason(failed_window)
    assert box.isReadOnly() and box.focusPolicy() != Qt.FocusPolicy.NoFocus
    flags = box.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByKeyboard
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse

    box.setPlainText("\n".join(f"Main.qml:{line}: Cannot assign to a missing property" for line in range(400)))
    failed_window.resize(900, 600)
    failed_window.show()
    QCoreApplication.processEvents()
    assert box.height() <= 200
    assert box.verticalScrollBar().maximum() > 0
    for text in ("Export diagnostics", "Quit"):
        button = _button(failed_window, text)
        assert button.isVisible()
        assert button.mapTo(failed_window, button.rect().bottomLeft()).y() < failed_window.height()


def test_a_failed_bridge_leaves_no_live_bridge_behind(qapp, monkeypatch, tmp_path) -> None:
    _load_config_without_devices(monkeypatch)
    window, *_ = _bridge_raises(MainWindow, monkeypatch, tmp_path)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert window.findChildren(QuickBridge) == []


def test_a_scene_that_does_not_load_leaves_no_live_bridge_behind(qapp, monkeypatch, tmp_path) -> None:
    _load_config_without_devices(monkeypatch)
    window, *_ = _scene_not_ready(MainWindow, monkeypatch, tmp_path)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert window.findChildren(QuickBridge) == []


def test_first_run_setup_does_not_open_over_the_error_screen(failed_window, monkeypatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(failed_window, "_show_first_run_setup", lambda **_k: opened.append(True))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUDIOFORGE_SMOKE_TEST", raising=False)
    failed_window.config.first_run_setup_state = "not_started"
    failed_window._maybe_show_first_run_setup()
    assert opened == []


def test_first_run_setup_still_opens_over_a_working_view(qapp, monkeypatch) -> None:
    _load_config_without_devices(monkeypatch)
    window = MainWindow()
    opened: list[bool] = []
    monkeypatch.setattr(window, "_show_first_run_setup", lambda **_k: opened.append(True))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUDIOFORGE_SMOKE_TEST", raising=False)
    window.config.first_run_setup_state = "not_started"
    window._maybe_show_first_run_setup()
    assert opened == [True]


@pytest.fixture
def broken_scene(monkeypatch, tmp_path):
    """The scene cannot load; request this before ``startup_window`` builds its window."""
    monkeypatch.setattr("mic_eq.ui.quick_shell.QML_DIR", tmp_path)


def test_login_launch_without_a_window_does_not_start_audio_and_says_so_in_the_tray(
    broken_scene, startup_window, monkeypatch  # noqa: F811
) -> None:
    window, clock, _ = startup_window
    assert window.quick_view_failure == "qml_not_ready"
    notices: list[tuple] = []
    monkeypatch.setattr(window._tray_icon, "showMessage", lambda *args: notices.append(args))

    window.begin_login_startup()
    clock[0] += 20
    window._service_login_startup()
    window._service_login_startup()

    window.processor.start.assert_not_called()
    assert window._login_startup_deadline is None
    assert not window.isVisible()
    assert len(notices) == 1
    title, message, icon, _ms = notices[0]
    assert "main window" in title
    assert "not started" in message and "Show AudioForge" in message and "tray" in message
    assert icon == QSystemTrayIcon.MessageIcon.Critical


def test_a_scene_that_loads_leaves_no_failure(quick_window) -> None:  # noqa: F811
    assert quick_window.quick_view_failure is None
    assert isinstance(quick_window.centralWidget(), QQuickWidget)


@pytest.mark.parametrize("failure", ["bridge", "qml"])
def test_the_packaged_smoke_test_exits_non_zero_when_the_scene_cannot_start(failure) -> None:
    script = f"""
import logging, sys
from PySide6.QtWidgets import QApplication
messages = []
class Capture(logging.Handler):
    def emit(self, record):
        messages.append(self.format(record))
logging.getLogger().addHandler(Capture())
failure = {failure!r}
if failure == "bridge":
    import mic_eq.ui.quick_shell as shell
    def refuse(_self, _window):
        raise RuntimeError("model could not be built")
    shell.QuickBridge._build_model = refuse
else:
    import mic_eq.ui.quick_shell as shell
    shell.QML_DIR = shell.QML_DIR / "missing"
from mic_eq.ui.app_bootstrap import run_smoke_test
from mic_eq.ui.main_window import MainWindow
status = run_smoke_test(MainWindow)
print("STATUS", status)
print("LOG", " | ".join(messages))
sys.exit(status)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    assert "the Qt Quick view did not load" in result.stdout
