"""What the Guided Setup scene shows, and the commands it sends back."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Property, QObject, Signal, Slot

from .components import Glyph
from .level_meter import LevelMeter

NO_HEALTH = {"text": "", "state": "idle"}
# What the page shows before the dialog has rendered its first step.
EMPTY_PAGE = {
    "title": "",
    "description": "",
    "progressCount": 0,
    "progressTotal": 1,
    "progressText": "",
    "devicesShown": False,
    "routeShown": False,
    "latencyShown": False,
    "latencyEnabled": False,
    "actionText": "",
    "actionEnabled": True,
    "backEnabled": False,
}


class FirstRunSetupView(QObject):
    """The view model of ``FirstRunSetup.qml``; the dialog owns the flow."""

    pageChanged = Signal()
    statusChanged = Signal()
    healthChanged = Signal()
    mutedChanged = Signal()
    inputIndexChanged = Signal()
    outputIndexChanged = Signal()

    def __init__(
        self,
        *,
        pause: Callable[[], None],
        back: Callable[[], None],
        skip: Callable[[], None],
        act: Callable[[], None],
        open_latency: Callable[[], None],
        request_mute: Callable[[bool], None],
        action_labels: list[str],
        inputs: list[str],
        outputs: list[str],
        input_index: int,
        output_index: int,
        mute_tip: str,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._commands = {
            "pause": pause,
            "back": back,
            "skip": skip,
            "act": act,
            "latency": open_latency,
        }
        self._request_mute = request_mute
        self._action_labels = list(action_labels)
        self._inputs, self._outputs = list(inputs), list(outputs)
        self._input_index, self._output_index = input_index, output_index
        self._mute_tip = mute_tip
        self._page = dict(EMPTY_PAGE)
        self.status_text, self.status_state = "", "info"
        self._health = {"input": dict(NO_HEALTH), "output": dict(NO_HEALTH)}
        self._muted = False
        self.input_meter = LevelMeter("IN", show_scale=True, parent=self)
        self.output_meter = LevelMeter("OUT", show_scale=True, parent=self)

    page = Property(dict, lambda self: self._page, notify=pageChanged)
    status = Property(dict, lambda self: {"text": self.status_text, "state": self.status_state}, notify=statusChanged)
    health = Property(dict, lambda self: self._health, notify=healthChanged)
    muted = Property(bool, lambda self: self._muted, notify=mutedChanged)
    inputs = Property(list, lambda self: self._inputs, constant=True)
    outputs = Property(list, lambda self: self._outputs, constant=True)
    inputIndex = Property(int, lambda self: self._input_index, notify=inputIndexChanged)
    outputIndex = Property(int, lambda self: self._output_index, notify=outputIndexChanged)
    actionLabels = Property(list, lambda self: self._action_labels, constant=True)
    muteTip = Property(str, lambda self: self._mute_tip, constant=True)
    expandGlyph = Property(str, lambda self: Glyph.EXPAND, constant=True)
    inputMeter = Property(QObject, lambda self: self.input_meter, constant=True)
    outputMeter = Property(QObject, lambda self: self.output_meter, constant=True)

    @property
    def input_index(self) -> int:
        return self._input_index

    @property
    def output_index(self) -> int:
        return self._output_index

    @property
    def is_muted(self) -> bool:
        return self._muted

    def set_page(self, **fields: object) -> None:
        """Change the parts of the page that follow the step; unchanged fields are left alone."""
        page = {**self._page, **fields}
        if page != self._page:
            self._page = page
            self.pageChanged.emit()

    def set_status(self, text: str, state: str) -> None:
        if (text, state) != (self.status_text, self.status_state):
            self.status_text, self.status_state = text, state
            self.statusChanged.emit()

    def set_health(self, direction: str, text: str, state: str) -> None:
        if self._health[direction] != {"text": text, "state": state}:
            self._health = {**self._health, direction: {"text": text, "state": state}}
            self.healthChanged.emit()

    def set_muted(self, muted: bool) -> None:
        if muted != self._muted:
            self._muted = muted
            self.mutedChanged.emit()

    @Slot(int)
    def pickInput(self, index: int) -> None:
        if index != self._input_index:
            self._input_index = index
            self.inputIndexChanged.emit()

    @Slot(int)
    def pickOutput(self, index: int) -> None:
        if index != self._output_index:
            self._output_index = index
            self.outputIndexChanged.emit()

    @Slot(bool)
    def requestMute(self, muted: bool) -> None:
        self._request_mute(muted)

    @Slot()
    def pause(self) -> None:
        self._commands["pause"]()

    @Slot()
    def back(self) -> None:
        self._commands["back"]()

    @Slot()
    def skip(self) -> None:
        self._commands["skip"]()

    @Slot()
    def act(self) -> None:
        self._commands["act"]()

    @Slot()
    def openLatency(self) -> None:
        self._commands["latency"]()
