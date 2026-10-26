"""The page the main window shows when Qt Quick cannot start or stops rendering.

Plain QtWidgets on purpose: it must work when the graphics stack does not.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

REASON_MAX_HEIGHT = 160
TITLE = "AudioForge could not open its main window"
BODY = (
    "The part that draws the controls failed to start, so there is nothing to show. "
    "Export diagnostics if you want to report this, then quit and open AudioForge again."
)
AUDIO_RUNNING = "Audio is still running with your last settings. Quit AudioForge to stop it."
AUDIO_STOPPED = "Audio is not running."


class QuickErrorScreen(QWidget):
    """Says the window could not start, why, and offers diagnostics and Quit."""

    def __init__(self, window, reason: str) -> None:
        super().__init__(window)
        self._owner = window
        title = QLabel(TITLE)
        title.setObjectName("errorTitle")
        title.setStyleSheet("font-size: 15pt; font-weight: 600;")
        body = QLabel(BODY)
        body.setObjectName("errorBody")
        self.audio = QLabel()
        self.audio.setObjectName("errorAudio")
        caption = QLabel("Reason")
        caption.setObjectName("errorReasonCaption")
        # A long Qt error scrolls here instead of pushing the buttons off screen.
        detail = QPlainTextEdit(reason)
        detail.setObjectName("errorReason")
        detail.setAccessibleName("Reason")
        detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        detail.setTabChangesFocus(True)
        detail.setMaximumHeight(REASON_MAX_HEIGHT)
        for label in (title, body, self.audio):
            label.setWordWrap(True)
        export = QPushButton("Export diagnostics")
        export.clicked.connect(lambda: window._export_diagnostics())
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(lambda: window._quit_from_tray())
        for button in (export, quit_button):
            # Outside a dialog only an auto-default button answers Enter.
            button.setAutoDefault(True)

        buttons = QHBoxLayout()
        buttons.addWidget(export)
        buttons.addWidget(quit_button)
        buttons.addStretch(1)
        column = QVBoxLayout()
        column.setSpacing(14)
        for item in (title, body, self.audio, caption, detail):
            column.addWidget(item)
        column.addLayout(buttons)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 24, 48, 24)
        outer.addStretch(1)
        outer.addLayout(column)
        outer.addStretch(2)
        export.setFocus()
        self._show_audio_state()
        # Audio can still be started from the keyboard while this screen is up.
        window.stop_btn_action.changed.connect(self._show_audio_state)

    def _show_audio_state(self) -> None:
        self.audio.setText(AUDIO_RUNNING if self._owner.processor.is_running() else AUDIO_STOPPED)


def show_quick_error(
    window, failure: str, reason: str, graphics_api: str = "not_applicable"
) -> None:
    """Make the error screen the window's content and record which failure it is."""

    window.quick_view_failure = failure
    window.quick_view_graphics_api = graphics_api
    # Qt hides and deletes a replaced central widget once the event loop runs.
    window.setCentralWidget(QuickErrorScreen(window, reason))
