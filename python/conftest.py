"""Shared pytest fixtures for Python UI tests."""

import faulthandler
import os
import sys

import pytest


# Run Qt tests headlessly by default.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QFontDialog,
    QInputDialog,
    QMessageBox,
)
from PySide6.QtCore import QCoreApplication, QEvent

from mic_eq.ui.theme import apply_application_theme


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication instance for all UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    apply_application_theme(app)
    yield app
    app.processEvents()


# A hung test is killed with a traceback naming it, well inside the CI job limit.
TEST_TIMEOUT_SECONDS = 600
_report_fd = -1


def pytest_configure(config):
    """Keep a handle on the real stderr: pytest's capture swaps fd 2 while a test runs."""
    global _report_fd
    _report_fd = os.dup(sys.__stderr__.fileno())


def pytest_unconfigure(config):
    os.close(_report_fd)


@pytest.fixture(autouse=True)
def hung_test_traceback():
    faulthandler.dump_traceback_later(TEST_TIMEOUT_SECONDS, exit=True, file=_report_fd)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def isolated_settings_directory(monkeypatch, tmp_path_factory):
    """Keep every test away from the developer's real settings, logs and caches."""
    monkeypatch.setenv("APPDATA", str(tmp_path_factory.mktemp("appdata")))
    home = tmp_path_factory.mktemp("home")
    for name, path in {
        "LOCALAPPDATA": home / "local",
        "XDG_CONFIG_HOME": home / "config",
        "HOME": home,
        "USERPROFILE": home,
        "QML_DISK_CACHE_PATH": home / "qmlcache",
    }.items():
        monkeypatch.setenv(name, str(path))


@pytest.fixture(autouse=True)
def guard_violations():
    """Collect guard violations: Qt prints and swallows an exception raised in a slot or timer."""
    violations: list[str] = []
    yield violations
    assert not violations, f"test tripped a guard inside a Qt callback: {violations}"


def _refuse(violations: list[str], message: str):
    violations.append(message)
    raise AssertionError(message)


@pytest.fixture(autouse=True)
def refuse_real_user_state(monkeypatch, guard_violations):
    """Fail a test that reaches the real Startup folder or the app's single-instance names."""
    from mic_eq.ui import desktop_integration, login_startup

    def startup_folder():
        _refuse(guard_violations, "test reached the real Startup folder; patch login_startup.startup_shortcut_path")

    original_init = desktop_integration.SingleInstanceCoordinator.__init__
    default_server = desktop_integration.SINGLE_INSTANCE_SERVER

    def coordinator_init(self, *, server_name=default_server, lock_path=None):
        if lock_path is None or server_name == default_server:
            _refuse(guard_violations, "test used the app's real single-instance lock or server name; pass both")
        original_init(self, server_name=server_name, lock_path=lock_path)

    monkeypatch.setattr(login_startup, "startup_shortcut_path", startup_folder)
    monkeypatch.setattr(desktop_integration.SingleInstanceCoordinator, "__init__", coordinator_init)


@pytest.fixture
def isolated_main_window(qapp, monkeypatch):
    """A real main window with default settings, no presets and no devices."""
    from mic_eq.config import AppConfig
    from mic_eq.ui.main_window import MainWindow

    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    yield window
    try:
        window.processor.stop()
    except Exception:
        pass
    window.close()
    window.deleteLater()
    qapp.processEvents()


@pytest.fixture(autouse=True)
def discard_unsaved_test_windows(monkeypatch, guard_violations):
    """Discard test edits during window cleanup; refuse every other static modal dialog.

    Offscreen there is nobody to answer a modal, so an unpatched one would hang the
    test. A test that expects a dialog patches that entry point itself. Dialog
    instances are not guarded: tests close their own with timers, so a hang there is
    left to ``hung_test_traceback``.
    """

    def refuse(*_args, **_kwargs):
        _refuse(guard_violations, "test opened a modal dialog; patch the dialog entry point it expects")

    for owner, names in (
        (QMessageBox, ("information", "warning", "critical", "about", "aboutQt")),
        (QFileDialog, (
            "getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory",
            "getOpenFileUrl", "getOpenFileUrls", "getSaveFileUrl", "getExistingDirectoryUrl",
        )),
        (QInputDialog, ("getText", "getInt", "getDouble", "getItem", "getMultiLineText")),
        (QColorDialog, ("getColor",)),
        (QFontDialog, ("getFont",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, refuse)

    def answer(parent, title, *args, **kwargs):
        if title == "Unsaved Sound Changes":
            return QMessageBox.StandardButton.Discard
        return refuse()

    monkeypatch.setattr(QMessageBox, "question", answer)


@pytest.fixture(autouse=True)
def dispose_test_windows(monkeypatch):
    """Destroy each test's Qt parents through the event loop before Python GC."""
    app = QApplication.instance()
    existing = set(app.topLevelWidgets()) if isinstance(app, QApplication) else set()
    yield
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    created = set(app.topLevelWidgets()) - existing
    for widget in created:
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
