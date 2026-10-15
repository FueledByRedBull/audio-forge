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
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QSystemTrayIcon
from PySide6.QtQuickWidgets import QQuickWidget

from mic_eq.ui import desktop_integration, quick_shell
from mic_eq.ui.desktop_integration import RENDERER_RELAUNCH_ENV, SingleInstanceCoordinator
from mic_eq.ui.main_window import MainWindow
from test_quick_shell import _load_config_without_devices

SETTLE_MS = 500  # several checks at the interval below
SURVIVED = 7  # the exit code of an event loop that was still running after SETTLE_MS


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


def _run_until_settled(qapp, window: MainWindow) -> tuple[int, dict]:
    """Run the real event loop; a copy that is still up after SETTLE_MS exits with SURVIVED."""

    state: dict = {}

    def settled() -> None:
        # Read here: aboutToQuit, emitted as the loop ends, marks every window as quitting.
        state.update(
            visible=window.isVisible(), quitting=window._quitting,
            snapshot=window._interface_snapshot(),
        )
        qapp.exit(SURVIVED)

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(settled)
    watchdog.start(SETTLE_MS)
    try:
        return qapp.exec(), state
    finally:
        watchdog.stop()


def _record_save_and_spawn(window: MainWindow, monkeypatch, *, spawn_works: bool) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(window, "_save_ui_state", lambda: events.append("closeEvent saved") or True)

    def spawn(_command, env):
        events.append("spawn")
        if not spawn_works:
            raise OSError("blocked")

    monkeypatch.setattr(desktop_integration.subprocess, "Popen", spawn)
    return events


def _guard(window: MainWindow, tmp_path) -> tuple[SingleInstanceCoordinator, dict]:
    names = {"server_name": f"AudioForge.test.renderer.{id(window)}", "lock_path": tmp_path / "instance.lock"}
    guard = SingleInstanceCoordinator(**names)
    assert guard.acquire() == SingleInstanceCoordinator.ACQUIRED
    setattr(window, "single_instance", guard)
    return guard, names


def test_the_copy_quits_and_spawns_once_after_the_window_closed(spawned, qapp, monkeypatch, tmp_path) -> None:
    window = _window()
    events = _record_save_and_spawn(window, monkeypatch, spawn_works=True)
    _, names = _guard(window, tmp_path)
    window.show()

    code, _state = _run_until_settled(qapp, window)

    assert code == 0 and events == ["closeEvent saved", "spawn"]
    successor = SingleInstanceCoordinator(**names)
    assert successor.acquire() == SingleInstanceCoordinator.ACQUIRED  # the new copy can take the guard
    successor.close()


def test_a_spawn_that_fails_keeps_the_app_and_its_guard_and_shows_the_error_screen(
    spawned, qapp, monkeypatch, tmp_path
) -> None:
    window = _window()
    events = _record_save_and_spawn(window, monkeypatch, spawn_works=False)
    guard, names = _guard(window, tmp_path)
    tray = QSystemTrayIcon(window)
    window._tray_icon = tray
    tray.show()
    window.config.mute_hotkey = "Ctrl+Alt+M"
    registered: list[str] = []
    monkeypatch.setattr(window, "_unregister_mute_hotkey", lambda: None)
    monkeypatch.setattr(window, "_register_mute_hotkey", lambda shortcut: registered.append(shortcut) or True)
    window.show()

    code, state = _run_until_settled(qapp, window)

    assert code == SURVIVED  # nothing quit the app
    assert events == ["closeEvent saved", "spawn"]
    assert state["visible"] and not state["quitting"]
    assert state["snapshot"]["view"] == "error" and state["snapshot"]["classic_reason"] == "no_renderer"
    assert tray.isVisible() and registered == ["Ctrl+Alt+M"]
    rival = SingleInstanceCoordinator(**names)
    assert rival.acquire(activate_existing=False) == SingleInstanceCoordinator.ALREADY_RUNNING
    guard.close()
