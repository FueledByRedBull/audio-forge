"""Words that change while focus stays elsewhere, told to assistive technology once."""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent

_SENTENCE_END = (".", "!", "?")


def announce(target: QObject | None, text: str) -> None:
    """Send ``text`` as an announcement from ``target``, when a screen reader is listening."""
    if text and target is not None and QAccessible.isActive():
        QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(target, text))


class Announcer(QObject):
    """Joins what one turn of the event loop changed into one announcement.

    A view model says each changed phase or status as it sets it; a dialog that has loaded its
    scene attaches the widget the announcement comes from. Until then nothing is said, so what
    a dialog starts with is not read out. The same words are not said twice in one turn.
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._target: QObject | None = None
        self._pending: list[str] = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._flush)

    def attach(self, target: QObject) -> None:
        if self._target is not None:
            self._target.destroyed.disconnect(self._detach)
        self._target = target
        target.destroyed.connect(self._detach)

    def _detach(self) -> None:
        """The widget is gone: what is waiting to be said has nowhere to come from."""
        self._timer.stop()
        self._pending = []
        self._target = None

    def say(self, text: str) -> None:
        text = text.strip()
        if text and self._target is not None and text not in self._pending:
            self._pending.append(text)
            self._timer.start()

    def _flush(self) -> None:
        parts, self._pending = self._pending, []
        announce(
            self._target,
            " ".join(part if part.endswith(_SENTENCE_END) else part + "." for part in parts),
        )
