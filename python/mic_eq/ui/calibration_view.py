"""What the Auto-EQ calibration scene shows, and the commands it sends back."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Property, QObject, Signal, Slot

from .capture_session import LIVE_OUTPUT_MUTED_NOTICE
from .components import Glyph
from .eq_compare import EQComparison
from .level_meter import LevelMeter

STATES = frozenset({"ok", "warn", "bad", "info", "idle"})
# The label of the main button in each state of the dialog; the scene is wide enough for the longest.
START_LABELS = (
    "Check Level",
    "Checking Level...",
    "Check Level Again",
    "Record Passage",
    "Recording...",
    "Analyzing...",
    "Apply EQ Settings",
    "Record Again",
)
OVERRIDE_LABELS = ("Continue Anyway", "Continue Despite Clipping Risk")
CHIP_KEYS = ("confidence", "error", "gain", "target")
EMPTY_CHIPS = {
    "confidence": "Capture quality: --",
    "error": "Distance from the tone preset: --",
    "gain": "Headroom check: --",
    "target": "Tone preset: --",
}
# What the page shows before the dialog has set anything.
EMPTY_PAGE = {
    "pageName": "Auto-EQ page",
    "step": 0,
    "curveShown": True,
    "instructionsShown": True,
    "mutedShown": False,
    "recordingShown": False,
    "recordingTitle": "Recording",
    "meterShown": False,
    "choicesEnabled": True,
    "curveDescription": "",
    "timeText": "",
    "timeDone": False,
    "progress": 0,
    "progressBusy": False,
    "diagnosticsShown": False,
    "eqCompareShown": False,
    "retakeShown": False,
    "retakeEnabled": True,
    "compareShown": False,
    "compareEnabled": False,
    "overrideShown": False,
    "overrideText": OVERRIDE_LABELS[0],
    "startText": START_LABELS[0],
    "startEnabled": True,
}


class CalibrationView(QObject):
    """The view model of ``AutoEqCalibration.qml``; the dialog owns the recording and the analysis."""

    pageChanged = Signal()
    statusChanged = Signal()
    chipsChanged = Signal()
    choicesChanged = Signal()

    def __init__(
        self,
        *,
        steps: tuple[str, ...],
        curves: tuple[tuple[str, str], ...],
        modes: tuple[tuple[str, str], ...],
        smoothings: tuple[tuple[str, str], ...],
        passage: str,
        curve_changed: Callable[[int], None],
        start: Callable[[], None],
        override: Callable[[], None],
        cancel: Callable[[], None],
        retake: Callable[[], None],
        compare: Callable[[], None],
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._steps = list(steps)
        self._choices = {"curve": list(curves), "mode": list(modes), "smoothing": list(smoothings)}
        self._index = {"curve": 0, "mode": 0, "smoothing": 0}
        self._passage = passage
        self._curve_changed = curve_changed
        self._commands = {
            "start": start, "override": override, "cancel": cancel, "retake": retake, "compare": compare,
        }
        self._page = dict(EMPTY_PAGE)
        self.warning_text, self.warning_state = "Ready to record", "idle"
        self.warning_strong, self.warning_tip = False, ""
        self._chips = {
            key: {"key": key, "text": EMPTY_CHIPS[key], "tip": "", "state": "idle"} for key in CHIP_KEYS
        }
        self.level_meter = LevelMeter("Level", show_scale=True, parent=self)
        self.eq_compare = EQComparison(self)

    page = Property(dict, lambda self: self._page, notify=pageChanged)
    status = Property(
        dict,
        lambda self: {
            "text": self.warning_text, "state": self.warning_state,
            "strong": self.warning_strong, "tip": self.warning_tip,
        },
        notify=statusChanged,
    )
    chips = Property(list, lambda self: self.chip_list, notify=chipsChanged)
    steps = Property(list, lambda self: self._steps, constant=True)
    curves = Property(list, lambda self: self.curve_labels, constant=True)
    modes = Property(list, lambda self: self.mode_labels, constant=True)
    smoothings = Property(list, lambda self: self.smoothing_labels, constant=True)
    curveIndex = Property(int, lambda self: self._index["curve"], notify=choicesChanged)
    modeIndex = Property(int, lambda self: self._index["mode"], notify=choicesChanged)
    smoothingIndex = Property(int, lambda self: self._index["smoothing"], notify=choicesChanged)
    passage = Property(str, lambda self: self.passage_text, constant=True)
    startLabels = Property(list, lambda self: list(START_LABELS), constant=True)
    overrideLabels = Property(list, lambda self: list(OVERRIDE_LABELS), constant=True)
    mutedNotice = Property(str, lambda self: LIVE_OUTPUT_MUTED_NOTICE, constant=True)
    expandGlyph = Property(str, lambda self: Glyph.EXPAND, constant=True)
    meter = Property(QObject, lambda self: self.level_meter, constant=True)
    eqCompare = Property(QObject, lambda self: self.eq_compare, constant=True)

    @property
    def curve_labels(self) -> list[str]:
        return [name for _key, name in self._choices["curve"]]

    @property
    def mode_labels(self) -> list[str]:
        return [name for _key, name in self._choices["mode"]]

    @property
    def smoothing_labels(self) -> list[str]:
        return [name for _key, name in self._choices["smoothing"]]

    @property
    def passage_text(self) -> str:
        return self._passage

    @property
    def chip_list(self) -> list[dict[str, str]]:
        return [dict(chip) for chip in self._chips.values()]

    @property
    def shown(self) -> dict[str, object]:
        """The fields of the page as they are now."""
        return dict(self._page)

    def chip(self, key: str) -> dict[str, str]:
        return dict(self._chips[key])

    def selected(self, group: str) -> str:
        """The key of the choice made in `group`: curve, mode or smoothing."""
        return self._choices[group][self._index[group]][0]

    def curve_name(self) -> str:
        return self._choices["curve"][self._index["curve"]][1]

    def set_page(self, **fields: object) -> None:
        """Change parts of the page; unchanged fields are left alone."""
        page = {**self._page, **fields}
        if page != self._page:
            self._page = page
            self.pageChanged.emit()

    def set_warning(
        self,
        text: str | None = None,
        *,
        state: str | None = None,
        strong: bool | None = None,
        tip: str | None = None,
    ) -> None:
        """Change the message under the progress; what is not given keeps its value."""
        text = self.warning_text if text is None else text
        state = self.warning_state if state is None else state if state in STATES else "idle"
        strong = self.warning_strong if strong is None else strong
        tip = self.warning_tip if tip is None else tip
        if (text, state, strong, tip) != (
            self.warning_text, self.warning_state, self.warning_strong, self.warning_tip
        ):
            self.warning_text, self.warning_state = text, state
            self.warning_strong, self.warning_tip = strong, tip
            self.statusChanged.emit()

    def set_chip(
        self,
        key: str,
        text: str | None = None,
        *,
        state: str | None = None,
        tip: str | None = None,
    ) -> None:
        chip = self._chips[key]
        changed = {
            "text": chip["text"] if text is None else text,
            "state": chip["state"] if state is None else state if state in STATES else "idle",
            "tip": chip["tip"] if tip is None else tip,
        }
        if any(chip[field] != value for field, value in changed.items()):
            chip.update(changed)
            self.chipsChanged.emit()

    def _pick(self, group: str, index: int) -> bool:
        if not 0 <= index < len(self._choices[group]) or index == self._index[group]:
            return False
        self._index[group] = index
        self.choicesChanged.emit()
        return True

    @Slot(int)
    def pickCurve(self, index: int) -> None:
        if self._pick("curve", index):
            self._curve_changed(index)

    @Slot(int)
    def pickMode(self, index: int) -> None:
        self._pick("mode", index)

    @Slot(int)
    def pickSmoothing(self, index: int) -> None:
        self._pick("smoothing", index)

    @Slot()
    def start(self) -> None:
        self._commands["start"]()

    @Slot()
    def override(self) -> None:
        self._commands["override"]()

    @Slot()
    def cancel(self) -> None:
        self._commands["cancel"]()

    @Slot()
    def retake(self) -> None:
        self._commands["retake"]()

    @Slot()
    def compare(self) -> None:
        self._commands["compare"]()
