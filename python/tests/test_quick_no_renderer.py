"""A window whose graphics backend fails to start is relaunched in software once.

Qt reads the graphics backend once, as the application starts, so the cases that
need a missing renderer (``quick_no_renderer_checks.py``) run in a child pytest.
This file also checks, in this process, the relaunch command and the working case.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PySide6.QtTest import QTest

from mic_eq.ui import desktop_integration
from mic_eq.ui.desktop_integration import (
    RENDERER_RELAUNCH_ENV,
    SingleInstanceCoordinator,
    relaunch_with_software_rendering,
)
from mic_eq.ui.quick_shell import graphics_api
from quick_child import run_checks
from test_quick_shell import quick_window  # noqa: F401  (a fixture)

CHECKS = Path(__file__).with_name("quick_no_renderer_checks.py")


def test_a_window_with_no_renderer_is_relaunched_or_says_so() -> None:
    run_checks(CHECKS, {"QT_QUICK_BACKEND": "rhi"})


@pytest.fixture(autouse=True)
def quick_checks(monkeypatch):
    monkeypatch.setattr("mic_eq.ui.quick_shell.RENDERER_CHECK_MS", 20)
    monkeypatch.setattr("mic_eq.ui.quick_shell.RENDERER_CHECKS", 2)


def _spawn_recorder(monkeypatch) -> list[tuple[list[str], dict[str, str]]]:
    commands: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        desktop_integration.subprocess, "Popen",
        lambda command, env: commands.append((command, env)),
    )
    return commands


def test_the_relaunch_from_source_repeats_the_command_without_the_login_argument(monkeypatch) -> None:
    commands = _spawn_recorder(monkeypatch)
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "mic_eq", "--login-startup", "--flag"])

    assert relaunch_with_software_rendering(None)

    command, environment = commands[0]
    assert command == [sys.executable, "-m", "mic_eq", "--flag"]
    assert environment["QT_QUICK_BACKEND"] == "software"
    assert environment["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert environment[RENDERER_RELAUNCH_ENV] == "1"


def test_the_relaunch_of_the_packaged_app_runs_the_executable(monkeypatch) -> None:
    commands = _spawn_recorder(monkeypatch)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["AudioForge.exe", "--login-startup"])

    assert relaunch_with_software_rendering(None)

    assert commands[0][0] == [sys.executable]


def test_the_relaunch_releases_the_single_instance_guard(qapp, monkeypatch, tmp_path) -> None:
    _spawn_recorder(monkeypatch)
    names = {"server_name": "AudioForge.test.relaunch", "lock_path": tmp_path / "instance.lock"}
    first = SingleInstanceCoordinator(**names)
    assert first.acquire() == SingleInstanceCoordinator.ACQUIRED

    assert relaunch_with_software_rendering(first)

    second = SingleInstanceCoordinator(**names)
    assert second.acquire() == SingleInstanceCoordinator.ACQUIRED
    second.close()


def test_a_relaunch_that_cannot_start_reports_it(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise OSError("blocked")

    monkeypatch.setattr(desktop_integration.subprocess, "Popen", refuse)
    assert not relaunch_with_software_rendering(None)


def test_a_relaunch_that_cannot_start_takes_the_single_instance_guard_back(qapp, monkeypatch, tmp_path) -> None:
    def refuse(*_args, **_kwargs):
        raise OSError("blocked")

    monkeypatch.setattr(desktop_integration.subprocess, "Popen", refuse)
    names = {"server_name": "AudioForge.test.relaunch.failed", "lock_path": tmp_path / "instance.lock"}
    first = SingleInstanceCoordinator(**names)
    assert first.acquire() == SingleInstanceCoordinator.ACQUIRED

    assert not relaunch_with_software_rendering(first)

    rival = SingleInstanceCoordinator(**names)
    assert rival.acquire(activate_existing=False) == SingleInstanceCoordinator.ALREADY_RUNNING
    first.close()  # a taken-back guard is released like any other
    assert rival.acquire() == SingleInstanceCoordinator.ACQUIRED
    rival.close()


def test_a_working_renderer_is_left_alone(monkeypatch, quick_window) -> None:  # noqa: F811
    commands = _spawn_recorder(monkeypatch)
    QTest.qWait(300)
    assert commands == [] and "quick_bridge" in quick_window.__dict__


def test_software_after_a_renderer_failure_is_reported(quick_window, monkeypatch) -> None:  # noqa: F811
    assert graphics_api(quick_window) == "software"
    monkeypatch.setenv(RENDERER_RELAUNCH_ENV, "1")
    assert graphics_api(quick_window) == "software_after_failure"
    assert quick_window._interface_snapshot()["graphics_api"] == "software_after_failure"


def test_the_bootstrap_hands_the_window_the_instance_guard(qapp, monkeypatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMainWindow

    from mic_eq.ui import app_bootstrap

    instance = SimpleNamespace(
        acquire=Mock(return_value="acquired"), close=Mock(),
        set_activation_callback=Mock(), last_error=None,
    )
    factory = Mock(return_value=instance, ACQUIRED="acquired")
    windows: list[QMainWindow] = []

    class Window(QMainWindow):
        def show(self) -> None:
            windows.append(self)
            QTimer.singleShot(0, lambda: qapp.exit(0))

    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    for name in ("configure_app_logging", "configure_windows_app_id", "configure_deepfilter_env",
                 "configure_vad_env", "apply_windows_window_icon", "apply_windows_taskbar_properties"):
        monkeypatch.setattr(app_bootstrap, name, lambda *_: None)

    assert app_bootstrap._run_qt_app(Window, smoke_test=False) == 0
    assert getattr(windows[0], "single_instance") is instance
