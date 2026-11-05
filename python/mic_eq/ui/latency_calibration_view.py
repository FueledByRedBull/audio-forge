"""What the latency calibration scene shows, and the commands it sends back."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Property, QObject, Signal, Slot

from .level_meter import LevelMeter

NO_READINGS = {
    "total": "-- ms",
    "confidence": "--",
    "roundTrip": "-- ms",
    "oneWay": "-- ms",
    "comp": "-- ms",
    "agreement": "--",
    "ambiguity": "--",
    "engine": "-- ms",
}


class LatencyCalibrationView(QObject):
    """The view model of ``LatencyCalibration.qml``; the dialog owns the work."""

    readingsChanged = Signal()
    detailsShownChanged = Signal()
    progressChanged = Signal()
    statusChanged = Signal()
    runEnabledChanged = Signal()
    acceptEnabledChanged = Signal()

    def __init__(
        self,
        *,
        run: Callable[[], None],
        accept: Callable[[], None],
        reset: Callable[[], None],
        close: Callable[[], None],
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._commands = {"run": run, "accept": accept, "reset": reset, "close": close}
        self._readings = dict(NO_READINGS)
        self._details_shown = False
        self._progress = 0
        self.status_text = "Ready"
        self._run_enabled = True
        self._accept_enabled = False
        self.level_meter = LevelMeter("Level", show_scale=True, parent=self)

    readings = Property(dict, lambda self: self._readings, notify=readingsChanged)
    detailsShown = Property(bool, lambda self: self._details_shown, notify=detailsShownChanged)
    progress = Property(int, lambda self: self._progress, notify=progressChanged)
    status = Property(str, lambda self: self.status_text, notify=statusChanged)
    runEnabled = Property(bool, lambda self: self._run_enabled, notify=runEnabledChanged)
    # Accept is the main action once there is a result; until then it is Run.
    acceptEnabled = Property(bool, lambda self: self._accept_enabled, notify=acceptEnabledChanged)
    meter = Property(QObject, lambda self: self.level_meter, constant=True)

    def set_readings(self, readings: dict[str, str]) -> None:
        if readings != self._readings:
            self._readings = dict(readings)
            self.readingsChanged.emit()

    def set_progress(self, progress: int) -> None:
        if progress != self._progress:
            self._progress = progress
            self.progressChanged.emit()

    def set_status(self, status: str) -> None:
        if status != self.status_text:
            self.status_text = status
            self.statusChanged.emit()

    def set_run_enabled(self, enabled: bool) -> None:
        if enabled != self._run_enabled:
            self._run_enabled = enabled
            self.runEnabledChanged.emit()

    def set_accept_enabled(self, enabled: bool) -> None:
        if enabled != self._accept_enabled:
            self._accept_enabled = enabled
            self.acceptEnabledChanged.emit()

    @Slot()
    def toggleDetails(self) -> None:
        self._details_shown = not self._details_shown
        self.detailsShownChanged.emit()

    @Slot()
    def run(self) -> None:
        self._commands["run"]()

    @Slot()
    def accept(self) -> None:
        self._commands["accept"]()

    @Slot()
    def reset(self) -> None:
        self._commands["reset"]()

    @Slot()
    def close(self) -> None:
        self._commands["close"]()
