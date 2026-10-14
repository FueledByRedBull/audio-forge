"""The page the main window shows when Qt Quick cannot start or stops rendering.

Plain QtWidgets on purpose: it must work when the graphics stack does not.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

TITLE = "AudioForge could not open its main window"
BODY = (
    "The part that draws the controls failed to start, so there is nothing to show. "
    "Export diagnostics if you want to report this, then quit and open AudioForge again."
)


class QuickErrorScreen(QWidget):
    """Says the window could not start, why, and offers diagnostics and Quit."""

    def __init__(self, window, reason: str) -> None:
        super().__init__(window)
        title = QLabel(TITLE)
        title.setObjectName("errorTitle")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        body = QLabel(BODY)
        body.setObjectName("errorBody")
        detail = QLabel(f"Reason: {reason}")
        detail.setObjectName("errorReason")
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for label in (title, body, detail):
            label.setWordWrap(True)
        export = QPushButton("Export diagnostics")
        export.clicked.connect(lambda: window._export_diagnostics())
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(lambda: window._quit_from_tray())

        buttons = QHBoxLayout()
        buttons.addWidget(export)
        buttons.addWidget(quit_button)
        buttons.addStretch(1)
        column = QVBoxLayout()
        column.setSpacing(14)
        for item in (title, body, detail):
            column.addWidget(item)
        column.addLayout(buttons)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 24, 48, 24)
        outer.addStretch(1)
        outer.addLayout(column)
        outer.addStretch(2)
        export.setFocus()


def show_quick_error(window, failure: str, reason: str) -> None:
    """Make the error screen the window's content and record which failure it is."""

    window.quick_view_failure = failure
    # Qt hides and deletes a replaced central widget once the event loop runs.
    window.setCentralWidget(QuickErrorScreen(window, reason))
