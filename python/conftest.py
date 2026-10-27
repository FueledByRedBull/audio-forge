"""Shared pytest fixtures for Python UI tests."""

import faulthandler
import gc
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
    home = tmp_path_factory.mktemp("home")
    appdata = home / "appdata"
    appdata.mkdir()
    for name, path in {
        "APPDATA": appdata,
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
def close_windows_without_the_quit_question(monkeypatch):
    """A window closed while a test fakes a running engine must not wait for an answer."""
    monkeypatch.setattr("mic_eq.ui.main_window.MainWindow._quit_choice", lambda _self: "quit")


@pytest.fixture(autouse=True, scope="module")
def collect_the_last_module_garbage():
    """Free what the last module left in reference cycles here, on the main thread.

    The native audio processor is unsendable: dropped on any other thread, it is reported
    as an unraisable exception. Cycles are freed by whichever thread allocates enough to
    start a collection, and the analysis tests start threads that do.
    """
    gc.collect()


_window_garbage = False


@pytest.fixture(autouse=True)
def dispose_test_windows(monkeypatch):
    """Destroy each test's Qt parents through the event loop before Python GC."""
    global _window_garbage
    if _window_garbage:
        # A main window made by the last test is Python garbage by now, in a reference
        # cycle, and the native audio processor in it can only be dropped on this thread.
        # Collect it before this test starts threads whose allocations would run the collection.
        _window_garbage = False
        gc.collect()
    from mic_eq.ui.main_window import MainWindow

    window_init = MainWindow.__init__

    def note_the_window(self, *args, **kwargs):
        global _window_garbage
        _window_garbage = True
        window_init(self, *args, **kwargs)
        # The fake engines answer at once; test_engine_tasks turns the worker back on.
        self._engine.threaded = False

    monkeypatch.setattr(MainWindow, "__init__", note_the_window)
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
    if _window_garbage:
        # Qt lets go of the window's Python callbacks one event round later. A window held by
        # a fixture is freed only once the fixtures are gone, by the collection above next time.
        app.processEvents()
        gc.collect()
