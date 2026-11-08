from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from typing import cast
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QMainWindow

from mic_eq.ui import app_bootstrap, desktop_integration
from engine_fakes import fake_session


def _server_name() -> str:
    return f"AudioForge.test.{uuid4().hex}"


def _pump_until(qapp, condition, timeout: float = 5.0) -> None:
    """Process events until the forwarded message has been handled."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)


def test_tray_tooltip_reflects_processing_mute_health_and_route():
    tooltip = desktop_integration.build_tray_tooltip(
        processing=True,
        muted=True,
        health="Healthy",
        route=("Mic", "Cable"),
    )

    assert tooltip == "AudioForge: Running / Muted | Health: Healthy | Mic -> Cable"


def test_activate_window_shows_a_hidden_or_minimized_window_at_its_last_size(qapp):
    window = QMainWindow()
    try:
        window.showMaximized()
        window.hide()
        desktop_integration.activate_window(window)
        assert window.isVisible() and window.isMaximized()

        window.showMinimized()
        desktop_integration.activate_window(window)
        assert window.isVisible() and window.isMaximized() and not window.isMinimized()

        window.showNormal()
        window.showMinimized()
        desktop_integration.activate_window(window)
        assert window.isVisible() and not window.isMinimized() and not window.isMaximized()
    finally:
        window.close()
        window.deleteLater()


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
        _pump_until(qapp, lambda: bool(seen))

        assert seen == [True]
        assert secondary.acquire() == secondary.FORWARDED
    finally:
        secondary.close()
        primary.close()


def test_a_sent_command_runs_its_callback_without_activating_the_window(qapp, tmp_path):
    server_name = _server_name()
    primary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name, lock_path=tmp_path / "audioforge.lock"
    )
    sender = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name, lock_path=tmp_path / "audioforge.lock"
    )
    activations: list[bool] = []
    mutes: list[bool] = []
    primary.set_activation_callback(lambda: activations.append(True))
    primary.set_command_callback(
        desktop_integration.TOGGLE_MUTE_COMMAND, lambda: mutes.append(True)
    )
    try:
        assert primary.acquire() == primary.ACQUIRED
        # A Windows pipe write completes only once the server reads it, so the
        # sender runs on its own thread, as a second launch would in its own process.
        sent: list[bool] = []
        thread = threading.Thread(
            target=lambda: sent.extend(
                sender.send_command(command)
                for command in (desktop_integration.TOGGLE_MUTE_COMMAND, "unknown-command")
            )
        )
        thread.start()
        deadline = time.monotonic() + 10
        while thread.is_alive() and time.monotonic() < deadline:
            qapp.processEvents()
            thread.join(0.01)
        assert sent == [True, True], sender.last_error
        for _ in range(50):
            qapp.processEvents()
            if mutes:
                break
        for _ in range(10):
            qapp.processEvents()
        assert mutes == [True]
        assert activations == []
    finally:
        sender.close()
        primary.close()


def test_sending_a_command_without_a_running_instance_fails(tmp_path):
    sender = desktop_integration.SingleInstanceCoordinator(
        server_name=_server_name(), lock_path=tmp_path / "audioforge.lock"
    )
    try:
        assert not sender.send_command(desktop_integration.TOGGLE_MUTE_COMMAND)
        assert sender.last_error == "AudioForge is not running"
    finally:
        sender.close()


def test_the_toggle_mute_command_line_names_this_copy(monkeypatch, tmp_path):
    image = tmp_path / "Audio Forge.AppImage"
    monkeypatch.setattr(desktop_integration.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop_integration.sys, "platform", "linux")
    monkeypatch.setenv("APPIMAGE", str(image))
    monkeypatch.setenv("APPDIR", str(Path(sys.executable).parent))  # the mount this copy runs from
    assert desktop_integration.toggle_mute_command() == f"'{image}' --toggle-mute"

    monkeypatch.setattr(desktop_integration.sys, "frozen", False)
    assert desktop_integration.toggle_mute_command().endswith("-m mic_eq --toggle-mute")


def test_the_instance_directory_is_the_users_runtime_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_integration.sys, "platform", "linux")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert desktop_integration._private_directory() == tmp_path

    # The XDG spec ignores a relative path, and a login without one gets a private directory at home.
    for unusable in ("run", None):
        if unusable is None:
            monkeypatch.delenv("XDG_RUNTIME_DIR")
        else:
            monkeypatch.setenv("XDG_RUNTIME_DIR", unusable)
        assert desktop_integration._private_directory() == Path.home() / ".cache" / "AudioForge"


def test_windows_keeps_naming_its_instance_endpoints_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_integration.sys, "platform", "win32")
    assert desktop_integration._private_directory() is None
    instance = desktop_integration.SingleInstanceCoordinator(
        server_name="AudioForge.test.names", lock_path=tmp_path / "audioforge.lock"
    )
    assert instance.server_name == "AudioForge.test.names"


@pytest.mark.skipif(sys.platform == "win32", reason="the runtime directory is a POSIX concept")
class TestRuntimeDirectory:
    @pytest.fixture(autouse=True)
    def runtime(self, monkeypatch):
        # Short, like /run/user/1000: a Unix socket path can't pass about 107 bytes, and pytest's tmp_path does.
        with tempfile.TemporaryDirectory(prefix="af", dir="/tmp") as directory:
            self.directory = Path(directory)
            self.directory.chmod(0o700)
            monkeypatch.setenv("XDG_RUNTIME_DIR", directory)
            yield

    def test_a_long_user_name_keeps_the_socket_under_the_unix_limit(self, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR")
        monkeypatch.setenv("HOME", "/home/" + "u" * 32)
        monkeypatch.setattr(desktop_integration, "_USER_SCOPE", "75" * 32)
        monkeypatch.setattr(desktop_integration, "SINGLE_INSTANCE_SERVER", f"AudioForge.SingleInstance.v1.{'75' * 32}")
        instance = desktop_integration.SingleInstanceCoordinator(
            server_name=desktop_integration.SINGLE_INSTANCE_SERVER, lock_path=Path(self.directory) / "lock"
        )
        assert len(instance.server_name.encode()) < 107

    def test_the_socket_and_the_lock_live_in_it_not_in_the_temp_directory(self, qapp):
        name = _server_name()
        instance = desktop_integration.SingleInstanceCoordinator(server_name=name)
        try:
            assert instance.server_name == str(self.directory / name)
            assert instance.acquire() == instance.ACQUIRED
            assert (self.directory / name).exists()
            assert list(self.directory.glob("AudioForge.SingleInstance.*.lock"))
        finally:
            instance.close()

    def test_toggle_mute_reaches_the_running_instance_whatever_the_temp_directory(
        self, qapp, monkeypatch, tmp_path
    ):
        # The app's own names, as a key binding launches them, from a shell with another $TMPDIR.
        primary = desktop_integration.SingleInstanceCoordinator()
        muted: list[bool] = []
        primary.set_command_callback(desktop_integration.TOGGLE_MUTE_COMMAND, lambda: muted.append(True))
        try:
            assert primary.acquire() == primary.ACQUIRED
            monkeypatch.setenv("TMPDIR", str(tmp_path / "elsewhere"))
            exits: list[int] = []
            thread = threading.Thread(target=lambda: exits.append(app_bootstrap.send_toggle_mute()))
            thread.start()
            deadline = time.monotonic() + 10
            while (thread.is_alive() or not muted) and time.monotonic() < deadline:
                qapp.processEvents()
                thread.join(0.01)
            assert exits == [0] and muted == [True]
        finally:
            primary.close()

    def test_a_directory_others_can_write_is_refused(self, qapp):
        self.directory.chmod(0o777)
        instance = desktop_integration.SingleInstanceCoordinator(server_name=_server_name())
        assert instance.acquire() == instance.FAILED
        assert "not private" in (instance.last_error or "")
        assert not list(self.directory.iterdir())

    def test_a_directory_another_account_owns_is_refused(self, qapp, monkeypatch):
        account = os.getuid()  # pyright: ignore[reportAttributeAccessIssue]
        monkeypatch.setattr(desktop_integration.os, "getuid", lambda: account + 1, raising=False)
        instance = desktop_integration.SingleInstanceCoordinator(server_name=_server_name())
        assert instance.acquire() == instance.FAILED
        assert "not private" in (instance.last_error or "")

    def test_without_a_runtime_directory_a_private_one_at_home_is_made(self, qapp, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR")
        instance = desktop_integration.SingleInstanceCoordinator(server_name=_server_name())
        try:
            assert instance.acquire() == instance.ACQUIRED
            made = Path.home() / ".cache" / "AudioForge"
            assert stat.S_IMODE(made.stat().st_mode) == 0o700
        finally:
            instance.close()


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
        _pump_until(qapp, lambda: activate.called)
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
        lambda _window_cls, *, smoke_test, enable_qt_log: 0 if smoke_test else 1,
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

    assert app_bootstrap._run_qt_app(QMainWindow, smoke_test=False, enable_qt_log=lambda: None) == 1
    audio_setup.assert_not_called()
    assert "owner unavailable" in message.call_args.args[2]


def test_login_bootstrap_never_shows_window_or_modal_and_uses_quiet_lock(qapp, monkeypatch):
    from PySide6.QtCore import QTimer

    instance = SimpleNamespace(
        acquire=Mock(return_value="acquired"), close=Mock(),
        set_activation_callback=Mock(), set_command_callback=Mock(), last_error=None,
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

        def _toggle_mute_from_hotkey(self):
            pass

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
        assert app_bootstrap._run_qt_app(QuietWindow, smoke_test=False, enable_qt_log=lambda: None) == 0
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
            self._quit_dont_ask = False
            self._hidden_to_tray = True
            self.tray_visible = True
            self._tray_icon = SimpleNamespace(
                isVisible=lambda: self.tray_visible,
                hide=lambda: setattr(self, "tray_visible", False),
            )
            self._close_to_tray_action = SimpleNamespace(isChecked=lambda: close_to_tray)
            self.configuration = SimpleNamespace(confirm_discard=lambda: True)
            self._quit_choice = lambda: "quit"
            self._hide_to_tray = lambda: MainWindow._hide_to_tray(cast(MainWindow, self))
            self._shutdown = lambda: MainWindow._shutdown(cast(MainWindow, self))
            self._save_on_shutdown = lambda: MainWindow._save_on_shutdown(cast(MainWindow, self))
            self._unregister_mute_hotkey = lambda: None
            self._toggle_mute_from_hotkey = lambda: None
            self._save_ui_state = lambda: True
            self.processor = SimpleNamespace(is_running=lambda: False)
            self.engine_session = fake_session(self.processor)
            self.config = SimpleNamespace(window_geometry={})
            self.status_message = SimpleNamespace(showMessage=lambda *_: None)

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
        assert app_bootstrap._run_qt_app(LoginWindow, smoke_test=False, enable_qt_log=lambda: None) == 0
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
    assert app_bootstrap._run_qt_app(cast(type[QMainWindow], window), smoke_test=False, enable_qt_log=lambda: None) == 0
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
    assert app_bootstrap._run_qt_app(cast(type[QMainWindow], window), smoke_test=False, enable_qt_log=lambda: None) == 1
    instance.close.assert_called_once()
