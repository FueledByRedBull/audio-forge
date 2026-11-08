"""What the Voice Setup scene shows, and the commands it sends back."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PySide6.QtCore import Property, QObject, Signal, Slot

from .capture_session import LIVE_OUTPUT_MUTED_NOTICE
from .components import Glyph
from .eq_compare import EQComparison
from .level_meter import LevelMeter

STATES = frozenset({"ok", "warn", "bad", "info", "idle"})


def _known_state(state: str) -> str:
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}")
    return state


# The label of the main button in each state of the dialog; the scene is wide enough for the longest.
START_LABELS = (
    "Start Voice Setup",
    "Checking Level...",
    "Check Level Again",
    "Record Room Noise",
    "Recording Noise...",
    "Retake Room Noise",
    "Record Voice",
    "Recording Voice...",
    "Record Voice Again",
    "Analyzing...",
    "Apply Voice Setup",
    "Apply Suggestions",
    "Record Verification Passage",
    "Recording Verification...",
    "Validating...",
    "Verify Reduced Processing",
    "Retry Verification (3/3)",
    "Compare Final Candidate",
    "Apply & Verify Again",
)
# The compression choices of the Dynamics card, and the keys they are stored under.
DYNAMICS = (("gentle", "Gentle"), ("balanced", "Balanced"), ("dense", "Dense"), ("custom", "Custom"))
DYNAMICS_KEYS = tuple(key for key, _name in DYNAMICS)
OVERRIDE_LABELS = ("Continue Anyway", "Continue Despite Clipping Risk")
CHIP_KEYS = ("overall", "eq", "gate", "deesser", "compressor")
EMPTY_CHIPS = {
    "overall": "Overall: --",
    "eq": "EQ: --",
    "gate": "Noise gate: --",
    "deesser": "De-esser: --",
    "compressor": "Compressor: --",
}
# The number fields of the Dynamics card: the key, the words around the field, its range and
# whether a change is a change of the dynamics setting (the loudness target is not).
NUMBER_FIELDS = (
    {
        "key": "p95", "caption": "Typical evening-out", "name": "Typical compressor evening-out in decibels",
        "suffix": " dB", "low": 1.0, "high": 8.0, "tip": "", "setting": True,
    },
    {
        "key": "peak", "caption": "Strongest squeeze allowed",
        "name": "Strongest compressor squeeze allowed in decibels",
        "suffix": " dB", "low": 1.5, "high": 12.0, "tip": "", "setting": True,
    },
    {
        "key": "lufs", "caption": "Target loudness", "name": "Voice setup target loudness",
        "suffix": " LUFS", "low": -24.0, "high": -12.0,
        "tip": "Loudness target is independent of the selected tone curve.", "setting": False,
    },
)
FIELDS_BY_KEY = {field["key"]: field for field in NUMBER_FIELDS}
DECIMALS = 2
# What the page shows before the dialog has set anything.
EMPTY_PAGE = {
    "pageName": "Auto Voice Setup page",
    "step": 0,
    "targetShown": True,
    "noiseShown": False,
    "voiceShown": False,
    "mutedShown": False,
    "recordingShown": False,
    "recordingTitle": "Recording",
    "meterShown": False,
    "reviewShown": False,
    "summaryShown": False,
    "curveEnabled": True,
    "curveGroupEnabled": True,
    "dynamicsEnabled": True,
    "targetEnabled": True,
    "curveDescription": "",
    "phaseText": "Ready to start setup",
    "timeText": "",
    "timeDone": False,
    "progress": 0,
    "eqCompareShown": False,
    "retakeShown": False,
    "retakeEnabled": True,
    "compareShown": False,
    "compareEnabled": False,
    "overrideShown": False,
    "overrideText": OVERRIDE_LABELS[0],
    "startText": START_LABELS[0],
    "startEnabled": True,
    "startLook": "primary",
}


class VoiceSetupView(QObject):
    """The view model of ``VoiceSetup.qml``; the dialog owns the recordings and the analysis."""

    pageChanged = Signal()
    statusChanged = Signal()
    chipsChanged = Signal()
    choicesChanged = Signal()

    def __init__(
        self,
        *,
        steps: tuple[str, ...],
        curves: tuple[tuple[str, str], ...],
        dynamics: tuple[tuple[str, str], ...],
        dynamics_key: str,
        numbers: Mapping[str, float],
        passage: str,
        curve_changed: Callable[[int], None],
        dynamics_changed: Callable[[], None],
        start: Callable[[], None],
        override: Callable[[], None],
        cancel: Callable[[], None],
        retake: Callable[[], None],
        compare: Callable[[], None],
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._steps = list(steps)
        self._choices = {"curve": list(curves), "dynamics": list(dynamics)}
        keys = [key for key, _name in dynamics]
        self._index = {"curve": 0, "dynamics": keys.index(dynamics_key) if dynamics_key in keys else 0}
        self._passage = passage
        self._curve_changed = curve_changed
        self._dynamics_changed = dynamics_changed
        self._numbers = {key: self._clamp(key, numbers[key]) for key in FIELDS_BY_KEY}
        self._commands = {
            "start": start, "override": override, "cancel": cancel, "retake": retake, "compare": compare,
        }
        self._page = dict(EMPTY_PAGE)
        self.warning_text, self.warning_state, self.warning_strong = "Ready to record", "idle", False
        self._chips = {
            key: {"key": key, "text": EMPTY_CHIPS[key], "tip": "", "state": "idle"} for key in CHIP_KEYS
        }
        self._issues: list[str] = []
        self.level_meter = LevelMeter("Level", show_scale=True, parent=self)
        self.eq_compare = EQComparison(self)

    page = Property(dict, lambda self: self._page, notify=pageChanged)
    status = Property(
        dict,
        lambda self: {"text": self.warning_text, "state": self.warning_state, "strong": self.warning_strong},
        notify=statusChanged,
    )
    chipMap = Property(dict, lambda self: {chip["key"]: chip for chip in self._chip_list()}, notify=chipsChanged)
    issues = Property(list, lambda self: self.issue_lines, notify=chipsChanged)
    steps = Property(list, lambda self: self._steps, constant=True)
    curves = Property(list, lambda self: [name for _key, name in self._choices["curve"]], constant=True)
    dynamics = Property(list, lambda self: [name for _key, name in self._choices["dynamics"]], constant=True)
    curveIndex = Property(int, lambda self: self._index["curve"], notify=choicesChanged)
    dynamicsIndex = Property(int, lambda self: self._index["dynamics"], notify=choicesChanged)
    dynamicsCustom = Property(bool, lambda self: self.selected("dynamics") == "custom", notify=choicesChanged)
    numberFields = Property(list, lambda self: [dict(field) for field in NUMBER_FIELDS], constant=True)
    numbers = Property(dict, lambda self: dict(self._numbers), notify=choicesChanged)
    decimals = Property(int, lambda self: DECIMALS, constant=True)
    passage = Property(str, lambda self: self._passage, constant=True)
    startLabels = Property(list, lambda self: list(START_LABELS), constant=True)
    overrideLabels = Property(list, lambda self: list(OVERRIDE_LABELS), constant=True)
    mutedNotice = Property(str, lambda self: LIVE_OUTPUT_MUTED_NOTICE, constant=True)
    expandGlyph = Property(str, lambda self: Glyph.EXPAND, constant=True)
    meter = Property(QObject, lambda self: self.level_meter, constant=True)
    eqCompare = Property(QObject, lambda self: self.eq_compare, constant=True)

    @property
    def issue_lines(self) -> list[str]:
        return list(self._issues)

    @property
    def shown(self) -> dict[str, Any]:
        """The fields of the page as they are now."""
        return dict(self._page)

    def _chip_list(self) -> list[dict[str, str]]:
        return [dict(chip) for chip in self._chips.values()]

    def chip(self, key: str) -> dict[str, str]:
        return dict(self._chips[key])

    def selected(self, group: str) -> str:
        """The key of the choice made in `group`: curve or dynamics."""
        return self._choices[group][self._index[group]][0]

    def number(self, key: str) -> float:
        """The value of a number field: p95, peak or lufs."""
        return self._numbers[key]

    @staticmethod
    def _clamp(key: str, value: float) -> float:
        field = FIELDS_BY_KEY[key]
        return round(min(field["high"], max(field["low"], float(value))), DECIMALS)

    def set_page(self, **fields: object) -> None:
        """Change parts of the page; unchanged fields are left alone."""
        unknown = set(fields) - set(self._page)
        if unknown:
            raise KeyError(f"the page has no field {', '.join(sorted(unknown))}")
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
    ) -> None:
        """Change the message under the progress; what is not given keeps its value."""
        text = self.warning_text if text is None else text
        state = self.warning_state if state is None else _known_state(state)
        strong = self.warning_strong if strong is None else strong
        if (text, state, strong) != (self.warning_text, self.warning_state, self.warning_strong):
            self.warning_text, self.warning_state, self.warning_strong = text, state, strong
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
            "state": chip["state"] if state is None else _known_state(state),
            "tip": chip["tip"] if tip is None else tip,
        }
        if any(chip[field] != value for field, value in changed.items()):
            chip.update(changed)
            self.chipsChanged.emit()

    def set_issues(self, lines: list[str]) -> None:
        """The lines under the first chip, one for each reason the capture was weak."""
        if lines != self._issues:
            self._issues = list(lines)
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
    def pickDynamics(self, index: int) -> None:
        if self._pick("dynamics", index):
            self._dynamics_changed()

    @Slot(str, float)
    def setNumber(self, key: str, value: float) -> None:
        value = self._clamp(key, value)
        if value == self._numbers[key]:
            return
        self._numbers[key] = value
        self.choicesChanged.emit()
        if FIELDS_BY_KEY[key]["setting"]:
            self._dynamics_changed()

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
