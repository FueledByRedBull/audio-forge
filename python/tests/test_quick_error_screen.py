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
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QGuiApplication
from PySide6.QtQuick import QQuickWindow
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QLabel, QPushButton, QStatusBar, QWidget
from shiboken6 import isValid

from mic_eq.ui.main_window import MainWindow
from test_quick_shell import _load_config_without_devices, quick_window  # noqa: F401  (a fixture)

ROOT = Path(__file__).resolve().parents[2]
TITLE = "AudioForge could not open its main window"
BODY = (
    "The part that draws the controls failed to start, so there is nothing to show. "
    "Export diagnostics if you want to report this, then quit and open AudioForge again."
)


def _bridge_raises(window_class, monkeypatch, tmp_path):
    def refuse(_self, _window):
        raise RuntimeError("model could not be built")

    monkeypatch.setattr("mic_eq.ui.quick_shell.QuickBridge._build_model", refuse)
    return window_class(), "bridge_failed", "RuntimeError: model could not be built"


def _scene_not_ready(window_class, monkeypatch, tmp_path):
    monkeypatch.setattr("mic_eq.ui.quick_shell.QML_DIR", tmp_path)
    return window_class(), "qml_not_ready", "Main.qml"


def _scene_graph_error(window_class, monkeypatch, tmp_path):
    window = window_class()
    window.show()
    view = window.centralWidget()
    assert isinstance(view, QQuickWidget)
    view.sceneGraphError.emit(
        QQuickWindow.SceneGraphError.ContextNotAvailable, "no graphics adapter"
    )
    return window, "scene_graph_error", "no graphics adapter"


@pytest.fixture(params=[_bridge_raises, _scene_not_ready, _scene_graph_error])
def failed_window(request, qapp, monkeypatch, tmp_path):
    _load_config_without_devices(monkeypatch)
    window, code, reason = request.param(MainWindow, monkeypatch, tmp_path)
    window.code, window.reason = code, reason
    return window


def _label(window, name: str) -> QLabel:
    label = window.centralWidget().findChild(QLabel, name)
    assert label is not None
    return label


def _button(window, text: str) -> QPushButton:
    buttons = [b for b in window.centralWidget().findChildren(QPushButton) if b.text() == text]
    assert len(buttons) == 1
    return buttons[0]


def test_the_window_shows_the_error_screen_not_the_classic_view(failed_window) -> None:
    window = failed_window
    screen = window.centralWidget()
    assert isinstance(screen, QWidget) and not isinstance(screen, QQuickWidget)
    assert window.quick_view_failure == window.code
    assert "quick_bridge" not in vars(window)
    # The widget view is not built.
    assert window._classic_root is None
    assert window.findChildren(QStatusBar) == []


def test_the_screen_says_what_happened_in_plain_words_and_gives_the_reason(failed_window) -> None:
    assert _label(failed_window, "errorTitle").text() == TITLE
    assert _label(failed_window, "errorBody").text() == BODY
    reason = _label(failed_window, "errorReason").text()
    assert reason.startswith("Reason: ") and failed_window.reason in reason


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
        "graphics_api": "not_applicable",
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
    assert "first" in _label(window, "errorReason").text()
    # The dead scene is deleted, not left behind a second central widget.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(scene)


def test_a_scene_that_loads_leaves_no_failure(quick_window) -> None:  # noqa: F811
    assert quick_window.quick_view_failure is None
    assert isinstance(quick_window.centralWidget(), QQuickWidget)


def test_opting_out_of_qml_is_not_a_failure(qapp, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_QML", "0")
    _load_config_without_devices(monkeypatch)
    window = MainWindow()
    assert window.quick_view_failure is None
    assert window._classic_root is not None and window.centralWidget() is window._classic_root


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
