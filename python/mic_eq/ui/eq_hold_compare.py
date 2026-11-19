"""Hold to hear original: the voice without the Equalizer stage for as long as a button or key is held.

There is one output, so callers hear the original too. The hold only turns the stage
off in the engine (``EQState.hold_original``); the settings, the undo history, the
unsaved-change count and every save stay as they were. Whatever could leave the EQ
off without a release ends the hold: the window hiding, minimising or losing focus,
audio stopping, the device going away, or any EQ edit.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, Property, Qt, Signal, Slot

from .engine_session import (
    BackgroundStopFailed,
    RestartFailed,
    Restarted,
    SessionView,
    StopFailed,
    Stopped,
)
from .eq_state import EQState
from .overview_presentation import DOT

HOLD_LABEL = "Hold to hear original"
HOLD_KEY = "B"
HOLD_TIP = (
    "Hold this button, or hold B while the equalizer graph has focus, to hear your voice "
    "without the equalizer. Callers hear the original too."
)
PILL_TEXT = f"Hearing your original voice {DOT} callers hear it too {DOT} release to return"
STOPPED_REASON = "Start processing to compare with your original voice."
EQ_OFF_REASON = "The equalizer is off, so you already hear your original voice."
DEVICE_LOST_REASON = "The audio device is not responding."
# The session events after which the engine's streams are not the ones the hold began on.
_ENDING_EVENTS = (Stopped, StopFailed, Restarted, RestartFailed, BackgroundStopFailed)


class EQHoldCompare(QObject):
    """The hold's state for the scene; ``press`` and ``release`` come from whatever is held."""

    changed = Signal()

    def __init__(
        self,
        parent: QObject,
        state: EQState,
        *,
        running: Callable[[], bool],
        device_lost: Callable[[], bool],
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._running = running
        self._device_lost = device_lost
        self._view = self._read()
        state.changed.connect(self.refresh)
        state.originalHeldChanged.connect(self.refresh)

    def _read(self) -> tuple[bool, str]:
        if not self._state.get_eq_settings().enabled:
            reason = EQ_OFF_REASON
        elif not self._running():
            reason = STOPPED_REASON
        elif self._device_lost():
            reason = DEVICE_LOST_REASON
        else:
            reason = ""
        return self._state.original_held, reason

    held = Property(bool, lambda self: self._view[0], notify=changed)
    enabled = Property(bool, lambda self: not self._view[1], notify=changed)
    toolTip = Property(
        str, lambda self: f"{self._view[1]} {HOLD_TIP}" if self._view[1] else HOLD_TIP, notify=changed
    )
    text = Property(str, lambda self: HOLD_LABEL, constant=True)
    name = Property(str, lambda self: HOLD_LABEL, constant=True)
    keycap = Property(str, lambda self: HOLD_KEY, constant=True)
    pill = Property(str, lambda self: PILL_TEXT, constant=True)

    @Slot()
    def press(self) -> None:
        self.refresh()
        if self.enabled:
            self._state.hold_original()

    @Slot()
    def release(self) -> None:
        self._state.release_original()

    @Slot()
    def refresh(self) -> None:
        view = self._read()
        if view[0] and view[1]:
            # Released here, the state's signal reads the view again.
            self._state.release_original()
            return
        if view != self._view:
            self._view = view
            self.changed.emit()

    @Slot(object)
    def session_changed(self, view: SessionView) -> None:
        if view.busy is not None or isinstance(view.event, _ENDING_EVENTS):
            self.release()
        self.refresh()

    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.Hide, QEvent.Type.WindowDeactivate) or (
            kind == QEvent.Type.WindowStateChange
            and watched.windowState() & Qt.WindowState.WindowMinimized
        ):
            self.release()
        return False
