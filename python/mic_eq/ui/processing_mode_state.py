"""The processing mode (normal, bypass, raw): what is selected, what the engine runs, and how a change fails.

Nothing here touches Qt. The mode is not preset state, and raw is never stored,
so it stays outside the six stage states. A failed change mutes the output and
keeps the audio running.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .processing_configuration import PROCESSING_MODES

_STATUS_TEXT = {
    "normal": "Processing active",
    "bypass": (
        "Voice effects bypassed; input conditioning and configured "
        "output protection remain"
    ),
    "raw": "Raw monitor enabled for this session; it is not stored in presets",
}


class ModeChoice(Protocol):
    def currentData(self) -> object: ...
    def currentIndex(self) -> int: ...
    def findData(self, data: Any, /) -> int: ...
    def setCurrentIndex(self, index: int, /) -> None: ...
    def blockSignals(self, blocked: bool, /) -> object: ...


class ProcessingModeState:
    def __init__(
        self,
        processor: Any,
        choice: ModeChoice,
        *,
        hold_mute: Callable[[bool], None],
        show_status: Callable[[str, int], None],
        on_changed: Callable[[], None],
    ) -> None:
        self._processor = processor
        self._choice = choice
        self._hold_mute = hold_mute
        self._show_status = show_status
        self._on_changed = on_changed
        self.applied = "normal"

    def selected(self) -> str:
        """The mode the choice shows, or the last applied one if it shows none."""
        mode = self._choice.currentData()
        if isinstance(mode, str) and mode in PROCESSING_MODES:
            return mode
        return self.applied

    def set(self, mode: str, *, notify: bool = False) -> None:
        """Show ``mode`` and apply it; raises RuntimeError, leaving ``applied`` unchanged, if the engine refuses."""
        target = mode if mode in PROCESSING_MODES else "normal"
        index = self._choice.findData(target)
        if index >= 0 and self._choice.currentIndex() != index:
            self._choice.blockSignals(True)
            self._choice.setCurrentIndex(index)
            self._choice.blockSignals(False)

        try:
            if hasattr(self._processor, "set_raw_monitor_enabled"):
                self._processor.set_raw_monitor_enabled(target == "raw")
            if hasattr(self._processor, "set_bypass"):
                self._processor.set_bypass(target == "bypass")
        except Exception as error:
            raise RuntimeError(f"Processing mode could not be applied: {error}") from error
        self.applied = target

        if notify:
            self._show_status(_STATUS_TEXT[target], 0)
        self._on_changed()

    def choose(self) -> bool:
        """Apply the mode the user picked; return False if it was refused.

        A refused mode restores the previous one; if that fails too the output is
        muted and audio keeps running.
        """
        previous = self.applied
        try:
            self.set(self.selected(), notify=True)
        except RuntimeError as error:
            try:
                self.set(previous)
            except RuntimeError:
                self._hold_mute(True)
                self._show_status(f"{error}; output muted because restoration failed", 0)
                return False
            self._show_status(f"{error}; previous mode restored", 6000)
            return False
        self._hold_mute(False)
        return True
