"""Shared pytest fixtures for Python UI tests."""

import os

import pytest


# Run Qt tests headlessly by default.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from mic_eq.ui.theme import application_palette


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication instance for all UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setStyle("Fusion")
    app.setPalette(application_palette())
    yield app
    app.processEvents()
