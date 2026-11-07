"""What the listening comparison scene shows, and the commands it sends back."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Property, QObject, Signal, Slot

from .capture_session import LIVE_OUTPUT_MUTED_NOTICE
from .components import Glyph

STATES = frozenset({"ok", "warn", "bad", "info", "idle"})
# What the page shows before the dialog has rendered anything.
EMPTY_PAGE = {
    "pageName": "Listening comparison page",
    "scope": "",
    "scopeWarning": "",
    "progressText": "Preparing comparison...",
    "progress": 0,
    "outputEnabled": True,
    "stopEnabled": False,
    "levelMatch": False,
    "playEnabled": False,
    "keepShown": True,
    "keepEnabled": False,
    "rejectText": "Reject Proposed",
    "rejectName": "Reject proposed settings",
    "rejectEnabled": True,
}


class ListeningComparisonView(QObject):
    """The view model of ``ListeningComparison.qml``; the dialog owns rendering and playback."""

    pageChanged = Signal()
    statusChanged = Signal()
    outputsChanged = Signal()
    playLabelsChanged = Signal()
    detailsChanged = Signal()

    def __init__(
        self,
        *,
        clips: tuple[tuple[str, str], ...],
        pick_output: Callable[[], None],
        refresh: Callable[[], None],
        stop: Callable[[], None],
        level_match: Callable[[bool], None],
        play: Callable[[str], None],
        keep: Callable[[], None],
        reject: Callable[[], None],
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._commands = {"pick": pick_output, "refresh": refresh, "stop": stop, "keep": keep, "reject": reject}
        self._level_match, self._play = level_match, play
        self._clips = [{"key": key, "caption": f"{label}:"} for key, label in clips]
        self._page = dict(EMPTY_PAGE)
        self.status_text, self.status_state = "", "idle"
        self._outputs: list[str] = []
        self._output_index = 0
        self._play_labels = {key: f"Play {label}" for key, label in clips}
        self._details = {key: "Waiting for render..." for key, _label in clips}

    page = Property(dict, lambda self: self._page, notify=pageChanged)
    status = Property(dict, lambda self: {"text": self.status_text, "state": self.status_state}, notify=statusChanged)
    outputs = Property(list, lambda self: self._outputs, notify=outputsChanged)
    outputIndex = Property(int, lambda self: self._output_index, notify=outputsChanged)
    clips = Property(list, lambda self: self._clips, constant=True)
    playLabels = Property(dict, lambda self: self._play_labels, notify=playLabelsChanged)
    details = Property(dict, lambda self: self._details, notify=detailsChanged)
    mutedNotice = Property(str, lambda self: LIVE_OUTPUT_MUTED_NOTICE, constant=True)
    expandGlyph = Property(str, lambda self: Glyph.EXPAND, constant=True)

    @property
    def output_index(self) -> int:
        return self._output_index

    @property
    def shown(self) -> dict[str, Any]:
        """The fields of the page as they are now."""
        return dict(self._page)

    @property
    def level_match(self) -> bool:
        return bool(self._page["levelMatch"])

    @property
    def play_labels(self) -> dict[str, str]:
        return dict(self._play_labels)

    @property
    def details_text(self) -> dict[str, str]:
        return dict(self._details)

    def set_page(self, **fields: object) -> None:
        """Change parts of the page; unchanged fields are left alone."""
        page = {**self._page, **fields}
        if page != self._page:
            self._page = page
            self.pageChanged.emit()

    def set_status(self, text: str, state: str | None = None) -> None:
        """Show `text`; without a `state` it keeps the colour it has."""
        state = self.status_state if state is None else state if state in STATES else "idle"
        if (text, state) != (self.status_text, self.status_state):
            self.status_text, self.status_state = text, state
            self.statusChanged.emit()

    def set_outputs(self, outputs: list[str], index: int) -> None:
        if (outputs, index) != (self._outputs, self._output_index):
            self._outputs, self._output_index = list(outputs), index
            self.outputsChanged.emit()

    def set_play_label(self, key: str, text: str) -> None:
        if self._play_labels[key] != text:
            self._play_labels = {**self._play_labels, key: text}
            self.playLabelsChanged.emit()

    def set_detail(self, key: str, text: str) -> None:
        if self._details[key] != text:
            self._details = {**self._details, key: text}
            self.detailsChanged.emit()

    @Slot(int)
    def pickOutput(self, index: int) -> None:
        if index != self._output_index:
            self._output_index = index
            self.outputsChanged.emit()
            self._commands["pick"]()

    @Slot(bool)
    def requestLevelMatch(self, enabled: bool) -> None:
        self.set_page(levelMatch=bool(enabled))
        self._level_match(bool(enabled))

    @Slot(str)
    def play(self, key: str) -> None:
        self._play(key)

    @Slot()
    def refresh(self) -> None:
        self._commands["refresh"]()

    @Slot()
    def stop(self) -> None:
        self._commands["stop"]()

    @Slot()
    def keep(self) -> None:
        self._commands["keep"]()

    @Slot()
    def reject(self) -> None:
        self._commands["reject"]()


class RecordingProgressView(QObject):
    """The view model of ``RecordingProgress.qml``: the progress of the Test my sound recording."""

    progressChanged = Signal()

    def __init__(self, *, prompt: str, cancel: Callable[[], None], parent: QObject | None = None):
        super().__init__(parent)
        self._prompt, self._cancel = prompt, cancel
        self._progress = 0

    prompt = Property(str, lambda self: self._prompt, constant=True)
    progress = Property(int, lambda self: self._progress, notify=progressChanged)
    mutedNotice = Property(str, lambda self: LIVE_OUTPUT_MUTED_NOTICE, constant=True)

    @property
    def progress_percent(self) -> int:
        return self._progress

    def set_progress(self, progress: int) -> None:
        if progress != self._progress:
            self._progress = progress
            self.progressChanged.emit()

    @Slot()
    def cancel(self) -> None:
        self._cancel()
