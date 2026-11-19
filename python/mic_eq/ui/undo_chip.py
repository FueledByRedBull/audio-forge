"""What the Equalizer page says beside Undo: the edits since the last save and the latest one in words."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeGuard

from PySide6.QtCore import Property, QObject, Signal, Slot

from .config_history import changed_configuration_paths
from .processing_configuration import ProcessingConfiguration

_STAGE_NAMES = {
    "gate": "Noise gate",
    "rnnoise": "Noise suppression",
    "deesser": "De-esser",
    "eq": "Equalizer",
    "compressor": "Compressor",
    "limiter": "Limiter",
}
# A band field, the words for it and its unit; the digits shown for each follow below.
_BAND_FIELDS = {
    "gain_db": ("gain", " dB"),
    "frequency_hz": ("frequency", " Hz"),
    "q": ("Q", ""),
}


def _number(value: float, field: str) -> str:
    if field == "gain_db":
        text = f"{value:.1f}"
    elif field == "q":
        text = f"{value:.2f}"
    else:
        text = f"{value:.1f}".removesuffix(".0")
    return text.lstrip("-") if float(text) == 0 else text


def _is_number(value: object) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_at(payload: Mapping[str, Any], path: list[str]) -> object:
    node: Any = payload
    for part in path:
        if isinstance(node, Mapping):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def describe_change(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str | None:
    """One line for a single EQ band value or stage switch that changed, else None."""
    paths = changed_configuration_paths(previous, current)
    if len(paths) != 1:
        return None
    path = next(iter(paths)).split(".")
    before, after = _value_at(previous, path), _value_at(current, path)
    if len(path) == 2 and path[1] == "enabled" and path[0] in _STAGE_NAMES:
        if isinstance(after, bool) and isinstance(before, bool):
            return f"{_STAGE_NAMES[path[0]]} {'on' if after else 'off'}"
        return None
    if len(path) == 4 and path[:2] == ["eq", "bands"] and path[2].isdigit() and path[3] in _BAND_FIELDS:
        if _is_number(before) and _is_number(after):
            name, unit = _BAND_FIELDS[path[3]]
            band = int(path[2]) + 1
            return f"Band {band} {name} {_number(before, path[3])} → {_number(after, path[3])}{unit}"
    return None


def format_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def changes_phrase(count: int) -> str:
    return f"{count} change{'' if count == 1 else 's'} since saved" if count > 0 else ""


class UndoChip(QObject):
    """The count of changes since saved and the caption for the latest one, for the page header."""

    changed = Signal()

    def __init__(self, parent: QObject, configuration: ProcessingConfiguration) -> None:
        super().__init__(parent)
        self._configuration = configuration
        self._count = 0
        self._text = ""
        self._caption = ""
        # The latest change in words and when it was made, kept for the history revision it was read at.
        self._read_at = -1
        self._line: str | None = None
        self._taken_at = 0.0

    count = Property(int, lambda self: self._count, notify=changed)
    text = Property(str, lambda self: self._text, notify=changed)
    caption = Property(str, lambda self: self._caption, notify=changed)
    # The chip component reads these from every proxy it is given.
    state = Property(str, lambda self: "idle", constant=True)
    name = Property(str, lambda self: "", constant=True)
    toolTip = Property(str, lambda self: "", constant=True)

    @Slot()
    def refresh(self) -> None:
        """Read the history again; the age moves on with the clock, so this also runs on a timer."""
        configuration = self._configuration
        history = configuration.history
        count = configuration.changes_since_saved
        if history.revision != self._read_at:
            self._read_at = history.revision
            before, after = history.previous, history.current
            self._line = (
                describe_change(before.payload(), after.payload()) if before and after else None
            )
            self._taken_at = after.taken_at if after else 0.0
        caption = ""
        if count > 0 and self._line is not None:
            caption = f"{self._line} · {format_age(configuration.clock() - self._taken_at)}"
        text = changes_phrase(count)
        if (count, text, caption) != (self._count, self._text, self._caption):
            self._count, self._text, self._caption = count, text, caption
            self.changed.emit()
