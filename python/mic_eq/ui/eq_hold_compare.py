"""Hold to hear original: the voice without the Equalizer stage for as long as a button or key is held.

There is one output, so callers hear the original too. The hold only turns the stage
off in the engine (``EQState.hold_original``); the settings, the undo history, the
unsaved-change count and every save stay as they were. Whatever could leave the EQ
off without a release ends the hold: the window hiding, minimising or losing focus,
audio stopping or being borrowed, the device going away, or any EQ edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from PySide6.QtCore import QEvent, QObject, Property, Qt, Signal, Slot

from .announcements import announce
from .engine_session import (
    AudioBorrowed,
    BackgroundStopFailed,
    RestartFailed,
    Restarted,
    SessionView,
    StopFailed,
    Stopped,
)
from .eq_state import EQState
from .overview_presentation import DOT

if TYPE_CHECKING:
    from .quick_shell import OverviewState

HOLD_LABEL = "Hold to hear original"
# What the button says beside its keycap when the header has no room for the label.
HOLD_SHORT = "Hold"
HOLD_KEY = "B"
HOLD_TIP = (
    "Hold this button, or hold B while the equalizer graph has focus, to hear your voice "
    "without the equalizer. Callers hear the original too."
)
PILL_TEXT = f"Hearing your original voice {DOT} callers hear it too {DOT} release to return"
# For assistive technology, where holding is not possible: activating the button latches the
# comparison on, and activating it again returns.
LATCHED_DESCRIPTION = "You are hearing your original voice. Activate to return to your processed voice."
UNLATCHED_DESCRIPTION = "Activate to hear your original voice, and again to return to your processed voice."
LATCHED_ANNOUNCEMENT = "Hearing your original voice. Callers hear it too."
UNLATCHED_ANNOUNCEMENT = "Back to your processed voice."
STOPPED_REASON = "Start processing to compare with your original voice."
EQ_OFF_REASON = "The equalizer is off, so you already hear your original voice."
DEVICE_LOST_REASON = "The audio device is not responding."
# The session events after which the engine's streams are not the ones the hold began on,
# or that hand the audio to Voice Setup or Test my sound, which must hear the EQ.
_ENDING_EVENTS = (Stopped, StopFailed, Restarted, RestartFailed, BackgroundStopFailed, AudioBorrowed)
# The Overview states with no audio running to compare.
_STOPPED_STATES = ("ready", "no_mic")


class _View(NamedTuple):
    held: bool
    reason: str  # why the button cannot be used; empty when it can


class EQHoldCompare(QObject):
    """The hold's state for the scene; ``press`` and ``release`` come from whatever is held.

    Whether audio runs and whether its device was lost are read from the Overview's state.
    """

    changed = Signal()

    def __init__(
        self,
        parent: QObject,
        state: EQState,
        overview: OverviewState,
        window: QObject,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._overview = overview
        # What a latched hold announces itself from.
        self._window = window
        self._view = self._read()
        state.changed.connect(self.refresh)
        state.originalHeldChanged.connect(self.refresh)
        overview.changed.connect(self.refresh)

    def _read(self) -> _View:
        overview = self._overview.state
        if not self._state.get_eq_settings().enabled:
            reason = EQ_OFF_REASON
        elif overview in _STOPPED_STATES:
            reason = STOPPED_REASON
        elif overview == "device_lost":
            reason = DEVICE_LOST_REASON
        else:
            reason = ""
        return _View(self._state.original_held, reason)

    held = Property(bool, lambda self: self._view.held, notify=changed)
    enabled = Property(bool, lambda self: not self._view.reason, notify=changed)
    toolTip = Property(
        str, lambda self: f"{self._view.reason} {HOLD_TIP}" if self._view.reason else HOLD_TIP, notify=changed
    )
    accessibleDescription = Property(
        str,
        lambda self: self.toolTip
        if self._view.reason
        else f"{LATCHED_DESCRIPTION if self._view.held else UNLATCHED_DESCRIPTION} {HOLD_TIP}",
        notify=changed,
    )
    text = Property(str, lambda self: HOLD_LABEL, constant=True)
    shortText = Property(str, lambda self: HOLD_SHORT, constant=True)
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

    @Slot(bool)
    def announceLatch(self, latched: bool) -> None:
        """Say what an assistive technology's activation did; a held button needs no words."""
        announce(self._window, LATCHED_ANNOUNCEMENT if latched else UNLATCHED_ANNOUNCEMENT)

    @Slot()
    def refresh(self) -> None:
        view = self._read()
        if view.held and view.reason:
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
