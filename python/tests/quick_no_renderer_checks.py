"""What the window does when Qt Quick starts without a renderer.

Not collected by the normal run: ``test_quick_no_renderer.py`` runs this file in a
child pytest with ``QT_QUICK_BACKEND=rhi`` on the offscreen platform. That asks for
a hardware backend the offscreen platform cannot give, so Qt logs "Failed to get a
QRhi", leaves the view Ready and emits no ``sceneGraphError``: the blank window.
The process spawn is recorded, not made.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import pytest
from PySide6.QtTest import QTest
from PySide6.QtQuickWidgets import QQuickWidget

from mic_eq.ui import desktop_integration, quick_shell
from mic_eq.ui.desktop_integration import RENDERER_RELAUNCH_ENV
from mic_eq.ui.main_window import MainWindow
from test_quick_shell import _load_config_without_devices

SETTLE_MS = 500  # several checks at the interval below


@pytest.fixture
def spawned(qapp, monkeypatch):
    _load_config_without_devices(monkeypatch)
    monkeypatch.delenv(RENDERER_RELAUNCH_ENV, raising=False)
    monkeypatch.delenv("AUDIOFORGE_SMOKE_TEST", raising=False)
    monkeypatch.setattr(quick_shell, "RENDERER_CHECK_MS", 20)
    monkeypatch.setattr(quick_shell, "RENDERER_CHECKS", 2)
    commands: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        desktop_integration.subprocess, "Popen",
        lambda command, env: commands.append((command, env)),
    )
    return commands


def _window() -> MainWindow:
    window = MainWindow()
    view = window.centralWidget()
    # The reproduction: loaded and Ready, but nothing will ever draw it.
    assert isinstance(view, QQuickWidget) and view.status() == QQuickWidget.Status.Ready
    return window


def test_a_shown_window_with_no_renderer_relaunches_in_software(spawned) -> None:
    window = _window()
    window.show()
    QTest.qWait(SETTLE_MS)

    view = window.centralWidget()
    assert isinstance(view, QQuickWidget) and not view.quickWindow().isSceneGraphInitialized()
    assert len(spawned) == 1
    environment = spawned[0][1]
    assert environment["QT_QUICK_BACKEND"] == "software"
    assert environment[RENDERER_RELAUNCH_ENV] == "1"
    assert not window.isVisible()  # this copy quit


def test_a_cancelled_quit_starts_no_copy_and_shows_the_error_screen(spawned, monkeypatch) -> None:
    window = _window()
    # The unsaved-changes prompt answered Cancel.
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: False)
    window.show()
    QTest.qWait(SETTLE_MS)

    assert spawned == [] and not window._quitting and window.isVisible()
    assert window._interface_snapshot()["classic_reason"] == "no_renderer"


def test_running_audio_is_kept_and_starts_no_copy(spawned) -> None:
    class RunningProcessor:
        def __init__(self, real) -> None:
            self._real = real

        def is_running(self) -> bool:
            return True

        def stop(self) -> None:
            pass

        def __getattr__(self, name: str):
            return getattr(self._real, name)

    window = _window()
    setattr(window, "processor", RunningProcessor(window.processor))
    window.show()
    QTest.qWait(SETTLE_MS)

    assert spawned == [] and window.isVisible()
    assert window._interface_snapshot()["classic_reason"] == "no_renderer"


def test_a_copy_that_was_already_relaunched_shows_the_error_screen(spawned, monkeypatch) -> None:
    monkeypatch.setenv(RENDERER_RELAUNCH_ENV, "1")
    window = _window()
    window.show()
    QTest.qWait(SETTLE_MS)

    assert spawned == []
    assert window._interface_snapshot() == {
        "view": "error", "classic_reason": "no_renderer", "graphics_api": "direct3d11",
    }
    assert window.isVisible()


def test_a_window_that_is_not_shown_is_not_judged(spawned) -> None:
    window = _window()
    QTest.qWait(SETTLE_MS)
    assert spawned == [] and "quick_bridge" in window.__dict__

    window.show()
    QTest.qWait(SETTLE_MS)
    assert len(spawned) == 1


def test_hiding_the_window_restarts_the_count(spawned, monkeypatch) -> None:
    monkeypatch.setattr(quick_shell, "RENDERER_CHECK_MS", 50)
    monkeypatch.setattr(quick_shell, "RENDERER_CHECKS", 4)
    window = _window()
    for _ in range(2):  # two checks shown, then hidden: never four in a row
        window.show()
        QTest.qWait(130)
        window.hide()
        QTest.qWait(130)
    assert spawned == []

    window.show()
    QTest.qWait(SETTLE_MS)
    assert len(spawned) == 1


def test_the_smoke_test_never_relaunches(spawned, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_SMOKE_TEST", "1")
    window = _window()
    window.show()
    QTest.qWait(SETTLE_MS)

    assert spawned == [] and window._interface_snapshot()["view"] == "quick"
