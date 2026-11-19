"""Shared sizing for dialogs."""

from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog

__all__ = ["configure_resizable_dialog"]


def configure_resizable_dialog(
    dialog: QDialog,
    *,
    preferred_width: int,
    preferred_height: int,
    minimum_width: int = 420,
    minimum_height: int = 320,
) -> None:
    """Clamp a dialog's initial and minimum size to its current screen."""

    screen = dialog.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        dialog.setMinimumSize(minimum_width, minimum_height)
        dialog.resize(preferred_width, preferred_height)
        dialog.setSizeGripEnabled(True)
        return
    available = screen.availableGeometry()
    max_width = max(320, available.width() - 48)
    max_height = max(240, available.height() - 48)
    width = min(preferred_width, max_width)
    height = min(preferred_height, max_height)
    dialog.setMinimumSize(min(minimum_width, width), min(minimum_height, height))
    dialog.resize(width, height)
    dialog.setSizeGripEnabled(True)
