"""Shared pytest fixtures for Python UI tests."""

import os

import pytest


# Run Qt tests headlessly by default.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication instance for all UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.processEvents()
