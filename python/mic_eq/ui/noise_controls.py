"""QML presentation of shared noise suppression settings."""

from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QObject, Property, Signal, Slot

from .noise_suppression_state import NoiseSuppressionState


class NoiseControl(QObject):
    """Expose the existing toggle, model picker, strength and latency to QML."""

    changed = Signal()

    def __init__(
        self, state: NoiseSuppressionState, key: str, parent: QObject,
        pick_model: Callable[[int], None] | None = None,
    ):
        super().__init__(parent)
        self._state, self._key, self._pick_model = state, key, pick_model
        self._name, self._tip = {
            "enabled": (
                "Enable noise suppression",
                "Enable or disable the selected suppression backend.",
            ),
            "model": (
                "Noise suppression backend",
                "Choose the suppression backend.\n"
                "RNNoise: low latency baseline.\n"
                "DeepFilterNet LL: low latency with stronger cleanup.\n"
                "DeepFilterNet: stronger cleanup at about 40 ms.",
            ),
            "strength": (
                "Noise suppression strength",
                "Processing strength for the selected backend (0% dry, 100% fully processed).",
            ),
            "latency": ("", ""),
        }[key]
        state.changed.connect(self.changed)

    name = Property(str, lambda self: self._name, constant=True)
    toolTip = Property(str, lambda self: self._tip, constant=True)
    minimum = Property(float, lambda self: 0.0, constant=True)
    maximum = Property(float, lambda self: 100.0 if self._key == "strength" else 1.0, constant=True)
    step = Property(float, lambda self: 1.0 if self._key == "strength" else 0.0, constant=True)
    enabled = Property(bool, lambda self: True, constant=True)
    shown = Property(bool, lambda self: True, constant=True)
    checked = Property(
        bool, lambda self: self._key == "enabled" and bool(self._state.get_settings()["enabled"]),
        notify=changed,
    )
    value = Property(
        float, lambda self: float(round(float(self._state.get_settings()["strength"]) * 100))
        if self._key == "strength" else 0.0,
        notify=changed,
    )
    items = Property(
        list, lambda self: [display for _, display in self._state.models] if self._key == "model" else [],
        constant=True,
    )
    index = Property(
        int, lambda self: next((
            index for index, (model, _) in enumerate(self._state.models)
            if model == self._state.get_settings()["model"]
        ), -1) if self._key == "model" else -1,
        notify=changed,
    )

    @Property(str, notify=changed)
    def text(self) -> str:
        if self._key == "latency":
            return self._state.latency_text
        if self._key == "strength":
            return f"{round(float(self._state.get_settings()['strength']) * 100)}%"
        if self._key == "model":
            return next((
                display for model, display in self._state.models
                if model == self._state.get_settings()["model"]
            ), "")
        return ""

    @Slot()
    def click(self) -> None:
        try:
            if self._key == "enabled":
                self._state.set_value("enabled", not self.checked)
        except (ValueError, RuntimeError):
            pass
        finally:
            self.changed.emit()

    @Slot(int)
    def setIndex(self, index: int) -> None:
        try:
            if self._key == "model" and self._pick_model is not None:
                self._pick_model(index)  # the window reports the switch
        except (ValueError, RuntimeError):
            # The window reports a refused switch; the picker must read back its value.
            pass
        finally:
            self.changed.emit()

    @Slot(float)
    def setValue(self, value: float) -> None:
        try:
            if self._key == "strength" and math.isfinite(value):
                self._state.set_value("strength", round(min(100.0, max(0.0, value))) / 100)
        except (ValueError, RuntimeError):
            pass
        finally:
            self.changed.emit()

    @Slot(str)
    def setText(self, text: str) -> None:
        try:
            self.setValue(float(text.replace("%", "").replace(",", ".").strip()))
        except ValueError:
            self.changed.emit()

    @Slot()
    def released(self) -> None:
        self.changed.emit()
