from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from typing import cast
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QMainWindow

from mic_eq.ui import app_bootstrap, desktop_integration


def _server_name() -> str:
    return f"AudioForge.test.{uuid4().hex}"


def test_tray_tooltip_reflects_processing_mute_health_and_route():
    tooltip = desktop_integration.build_tray_tooltip(
        processing=True,
        muted=True,
        health="Healthy",
        route="Mic -> Cable",
    )

    assert tooltip == "AudioForge: Running / Muted | Health: Healthy | Mic -> Cable"


def test_activate_window_restores_and_focuses_in_order():
    calls: list[str] = []
    window = SimpleNamespace(
        showNormal=lambda: calls.append("showNormal"),
        show=lambda: calls.append("show"),
        raise_=lambda: calls.append("raise"),
        activateWindow=lambda: calls.append("activate"),
    )

    desktop_integration.activate_window(cast(QMainWindow, window))

    assert calls == ["showNormal", "show", "raise", "activate"]


def test_second_instance_forwards_activation_without_removing_primary_server(
    qapp, tmp_path
):
    server_name = _server_name()
    lock_path = tmp_path / "audioforge.lock"
    primary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name,
        lock_path=lock_path,
    )
    secondary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name,
        lock_path=lock_path,
    )
    seen: list[bool] = []
    primary.set_activation_callback(lambda: seen.append(True))

    try:
        assert primary.acquire() == primary.ACQUIRED
        assert secondary.acquire() == secondary.FORWARDED
        for _ in range(10):
            qapp.processEvents()

        assert seen == [True]
        assert secondary.acquire() == secondary.FORWARDED
    finally:
        secondary.close()
        primary.close()


def test_second_instance_fails_closed_when_owner_has_no_activation_endpoint(
    tmp_path,
):
    from PySide6.QtCore import QLockFile

    lock_path = tmp_path / "audioforge.lock"
    owner_lock = QLockFile(str(lock_path))
    owner_lock.setStaleLockTime(0)
    assert owner_lock.tryLock(0)
    secondary = desktop_integration.SingleInstanceCoordinator(
        server_name=_server_name(),
        lock_path=lock_path,
    )

    try:
        assert secondary.acquire() == secondary.FAILED
        assert secondary.last_error
    finally:
        secondary.close()
        owner_lock.unlock()


def test_login_duplicate_does_not_connect_or_activate_existing_window(qapp, tmp_path):
    server_name = _server_name()
    lock_path = tmp_path / "quiet.lock"
    primary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name, lock_path=lock_path
    )
    duplicate = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name, lock_path=lock_path
    )
    activate = Mock()
    primary.set_activation_callback(activate)
    try:
        assert primary.acquire() == primary.ACQUIRED
        assert duplicate.acquire(activate_existing=False) == duplicate.ALREADY_RUNNING
        for _ in range(10):
            qapp.processEvents()
        activate.assert_not_called()
        assert duplicate.acquire() == duplicate.FORWARDED
        for _ in range(10):
            qapp.processEvents()
        activate.assert_called_once()
    finally:
        duplicate.close()
        primary.close()


def test_quiet_launch_does_not_mistake_unwritable_lock_path_for_existing_instance(tmp_path):
    instance = desktop_integration.SingleInstanceCoordinator(
        server_name=_server_name(), lock_path=tmp_path / "missing" / "quiet.lock"
    )
    try:
        assert instance.acquire(activate_existing=False) == instance.FAILED
        assert instance.last_error
    finally:
        instance.close()


def test_smoke_test_skips_single_instance_coordinator(monkeypatch):
    class UnexpectedCoordinator:
        def __init__(self):
            raise AssertionError("smoke tests must not acquire the real instance lock")

    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", UnexpectedCoordinator)
    monkeypatch.setattr(
        app_bootstrap,
        "_run_qt_app",
        lambda _window_cls, *, smoke_test: 0 if smoke_test else 1,
    )

    assert app_bootstrap.run_qt_app(cast(type[QMainWindow], object), smoke_test=True) == 0


def test_failed_instance_lock_shows_error_before_audio_setup(qapp, monkeypatch):
    instance = SimpleNamespace(
        acquire=lambda: "failed", last_error="owner unavailable",
    )
    factory = Mock(return_value=instance, ACQUIRED="acquired", FORWARDED="forwarded")
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    monkeypatch.setattr(app_bootstrap, "configure_app_logging", lambda: None)
    monkeypatch.setattr(app_bootstrap, "configure_windows_app_id", lambda: None)
    audio_setup = Mock()
    monkeypatch.setattr(app_bootstrap, "configure_deepfilter_env", audio_setup)
    message = Mock()
    monkeypatch.setattr(app_bootstrap.QMessageBox, "critical", message)

    assert app_bootstrap._run_qt_app(QMainWindow, smoke_test=False) == 1
    audio_setup.assert_not_called()
    assert "owner unavailable" in message.call_args.args[2]


def test_login_bootstrap_never_shows_window_or_modal_and_uses_quiet_lock(qapp, monkeypatch):
    from PySide6.QtCore import QTimer

    instance = SimpleNamespace(
        acquire=Mock(return_value="acquired"), close=Mock(),
        set_activation_callback=Mock(), last_error=None,
    )
    factory = Mock(
        return_value=instance, ACQUIRED="acquired", FORWARDED="forwarded",
        ALREADY_RUNNING="already_running",
    )
    seen = []

    class QuietWindow(QMainWindow):
        def __init__(self, *, login_startup=False):
            super().__init__()
            seen.append(login_startup)

        def begin_login_startup(self):
            seen.append(self.isVisible())
            QTimer.singleShot(0, lambda: qapp.exit(0))

        def show(self):
            raise AssertionError("Login must never show or focus the window")

    monkeypatch.setattr(app_bootstrap.sys, "argv", ["AudioForge.exe", "--login-startup"])
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    for name in ("configure_app_logging", "configure_windows_app_id",
                 "configure_deepfilter_env", "configure_vad_env",
                 "apply_windows_window_icon", "apply_windows_taskbar_properties"):
        monkeypatch.setattr(app_bootstrap, name, lambda *_: None)
    message = Mock(side_effect=AssertionError("Login must not display a modal"))
    monkeypatch.setattr(app_bootstrap.QMessageBox, "critical", message)
    previous = qapp.quitOnLastWindowClosed()
    try:
        assert app_bootstrap._run_qt_app(QuietWindow, smoke_test=False) == 0
        assert seen == [True, False]
        instance.acquire.assert_called_once_with(activate_existing=False)
    finally:
        qapp.setQuitOnLastWindowClosed(previous)


@pytest.mark.parametrize("close_to_tray", [False, True])
def test_login_lifecycle_keeps_hidden_session_alive_and_releases_lock_on_quit(
    qapp, monkeypatch, tmp_path, close_to_tray
):
    from PySide6.QtCore import QTimer
    from mic_eq.ui.main_window import MainWindow

    server_name = _server_name()
    lock_path = tmp_path / "login-lifecycle.lock"
    primary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name, lock_path=lock_path
    )
    factory = Mock(return_value=primary, ACQUIRED=primary.ACQUIRED,
                   FORWARDED=primary.FORWARDED, ALREADY_RUNNING=primary.ALREADY_RUNNING)
    events = []

    class LoginWindow(QMainWindow):
        def __init__(self, *, login_startup=False):
            super().__init__()
            self._quitting = False
            self._hidden_to_tray = True
            self.tray_visible = True
            self._tray_icon = SimpleNamespace(
                isVisible=lambda: self.tray_visible,
                hide=lambda: setattr(self, "tray_visible", False),
            )
            self._close_to_tray_action = SimpleNamespace(isChecked=lambda: close_to_tray)
            self._confirm_discard_changes = lambda: True
            self._unregister_mute_hotkey = lambda: None
            self._save_ui_state = lambda: True
            self.processor = SimpleNamespace(is_running=lambda: False)
            self.config = SimpleNamespace(window_geometry={})
            self.status_bar = SimpleNamespace(showMessage=lambda *_: None)

        def begin_login_startup(self):
            QTimer.singleShot(10, self.open_and_close)

        def open_and_close(self):
            events.append(("initially_hidden", not self.isVisible()))
            self.show()
            self.close()
            events.append(("tray_after_close", self.tray_visible))
            if close_to_tray:
                QTimer.singleShot(10, self.quit_from_tray)

        def quit_from_tray(self):
            events.append(("tray_session_alive", self.isHidden()))
            MainWindow._quit_from_tray(cast(MainWindow, self))

        def closeEvent(self, event):
            MainWindow.closeEvent(cast(MainWindow, self), event)

    monkeypatch.setattr(app_bootstrap.sys, "argv", ["AudioForge.exe", "--login-startup"])
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    for name in ("configure_app_logging", "configure_windows_app_id",
                 "configure_deepfilter_env", "configure_vad_env",
                 "apply_windows_window_icon", "apply_windows_taskbar_properties"):
        monkeypatch.setattr(app_bootstrap, name, lambda *_: None)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: qapp.exit(7))
    previous = qapp.quitOnLastWindowClosed()
    successor = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name, lock_path=lock_path
    )
    try:
        qapp.setQuitOnLastWindowClosed(True)
        watchdog.start(1000)
        assert app_bootstrap._run_qt_app(LoginWindow, smoke_test=False) == 0
        expected = [("initially_hidden", True), ("tray_after_close", close_to_tray)]
        if close_to_tray:
            expected.append(("tray_session_alive", True))
        assert events == expected
        assert successor.acquire() == successor.ACQUIRED
    finally:
        watchdog.stop()
        primary.close()
        successor.close()
        qapp.setQuitOnLastWindowClosed(previous)


def test_login_duplicate_exits_before_window_or_audio_setup(qapp, monkeypatch):
    instance = SimpleNamespace(acquire=Mock(return_value="already_running"))
    factory = Mock(
        return_value=instance, ACQUIRED="acquired", FORWARDED="forwarded",
        ALREADY_RUNNING="already_running",
    )
    monkeypatch.setattr(app_bootstrap.sys, "argv", ["AudioForge.exe", "--login-startup"])
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    monkeypatch.setattr(app_bootstrap, "configure_app_logging", lambda: None)
    monkeypatch.setattr(app_bootstrap, "configure_windows_app_id", lambda: None)
    audio_setup = Mock()
    monkeypatch.setattr(app_bootstrap, "configure_deepfilter_env", audio_setup)
    window = Mock()
    assert app_bootstrap._run_qt_app(cast(type[QMainWindow], window), smoke_test=False) == 0
    audio_setup.assert_not_called()
    window.assert_not_called()


def test_login_construction_error_is_logged_and_exits_without_exception_dialog(qapp, monkeypatch):
    instance = SimpleNamespace(
        acquire=Mock(return_value="acquired"), close=Mock(), last_error=None,
    )
    factory = Mock(
        return_value=instance, ACQUIRED="acquired", FORWARDED="forwarded",
        ALREADY_RUNNING="already_running",
    )
    monkeypatch.setattr(app_bootstrap.sys, "argv", ["AudioForge.exe", "--login-startup"])
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    for name in ("configure_app_logging", "configure_windows_app_id",
                 "configure_deepfilter_env", "configure_vad_env"):
        monkeypatch.setattr(app_bootstrap, name, lambda *_: None)
    window = Mock(side_effect=RuntimeError("saved configuration failed"))
    assert app_bootstrap._run_qt_app(cast(type[QMainWindow], window), smoke_test=False) == 1
    instance.close.assert_called_once()
