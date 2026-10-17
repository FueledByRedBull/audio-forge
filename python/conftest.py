"""Shared pytest fixtures for Python UI tests."""

import os

import pytest


# Run Qt tests headlessly by default.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
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


@pytest.fixture(autouse=True)
def isolated_settings_directory(monkeypatch, tmp_path_factory):
    """Keep every test away from the developer's real settings and logs."""
    monkeypatch.setenv("APPDATA", str(tmp_path_factory.mktemp("appdata")))


@pytest.fixture
def partial_results(monkeypatch):
    """Turn on withholding only the tone, which is off by default."""
    monkeypatch.setattr("mic_eq.analysis.voice_setup.PARTIAL_RESULTS", True)


@pytest.fixture
def compressor_fallback(monkeypatch):
    """Turn on keeping the user's compressor, which is off by default."""
    monkeypatch.setattr("mic_eq.analysis.voice_setup.COMPRESSOR_FALLBACK", True)


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
def discard_unsaved_test_windows(monkeypatch):
    """Discard test edits during window cleanup; tests can override the decision."""
    from PySide6.QtWidgets import QMessageBox

    question = QMessageBox.question

    def answer(parent, title, *args, **kwargs):
        if title == "Unsaved Sound Changes":
            return QMessageBox.StandardButton.Discard
        return question(parent, title, *args, **kwargs)

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
