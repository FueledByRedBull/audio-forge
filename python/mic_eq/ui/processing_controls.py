"""Qt Quick adapters for shared processing settings."""

from __future__ import annotations

import math

from PySide6.QtCore import QLocale, QObject, Property, Signal, Slot

from .compressor_state import CompressorState
from .control_specs import CONTROL_RANGES, control_spec
from .deesser_state import DeEsserState
from .gate_state import GateState
from .limiter_state import LimiterState


class SettingControl(QObject):
    """Present one shared processing setting to QML without reading a QWidget."""

    changed = Signal()

    def __init__(
        self,
        state: LimiterState | DeEsserState | CompressorState | GateState,
        section: str,
        key: str,
        parent: QObject | None,
    ):
        super().__init__(parent)
        self._state = state
        self._key = key
        self._section = section
        self._spec = control_spec(section, key)
        self._choices = self._spec.choices
        ranges = CONTROL_RANGES.get(section, {})
        self._numeric = key in ranges and not self._choices
        self._minimum, self._maximum = ranges.get(key, (0.0, 1.0))
        self._locale = QLocale()
        self._locale.setNumberOptions(
            self._locale.numberOptions() | QLocale.NumberOption.OmitGroupSeparator,
        )
        state.changed.connect(self.changed)

    def _presentation(self, field: str, default: str) -> str:
        if isinstance(self._state, GateState) and self._key == "auto_threshold_enabled":
            return self._state.presentation()["auto_threshold_" + field]
        return default

    name = Property(
        str, lambda self: self._presentation("name", self._spec.accessible_name),
        notify=changed,
    )
    toolTip = Property(
        str, lambda self: self._presentation("tooltip", self._spec.tooltip),
        notify=changed,
    )
    step = Property(float, lambda self: self._spec.step, constant=True)
    minimum = Property(float, lambda self: self._minimum, constant=True)
    maximum = Property(float, lambda self: self._maximum, constant=True)
    enabled = Property(
        bool, lambda self: self._state.control_enabled(self._key), notify=changed,
    )
    shown = Property(bool, lambda self: True, constant=True)
    checked = Property(
        bool, lambda self: bool(self._state.get_settings()[self._key]), notify=changed,
    )
    value = Property(
        float,
        lambda self: float(QLocale.c().toString(
            float(self._state.get_settings()[self._key]), "f", self._spec.decimals,
        )),
        notify=changed,
    )
    text = Property(
        str,
        lambda self: (
            self._choices[self.index] if self._choices else
            self._locale.toString(self.value, "f", self._spec.decimals) + self._spec.suffix
        ),
        notify=changed,
    )
    items = Property(list, lambda self: list(self._choices), constant=True)
    index = Property(
        int,
        lambda self: int(self._state.get_settings()[self._key]) if self._choices else -1,
        notify=changed,
    )

    @Slot(int)
    def setIndex(self, index: int) -> None:
        try:
            if self._choices and self.enabled and 0 <= index < len(self._choices):
                self._state.set_value(self._key, index)
        except (ValueError, RuntimeError):
            pass
        finally:
            self.changed.emit()

    @Slot(float)
    def setValue(self, value: float) -> None:
        try:
            if self._numeric and self.enabled and math.isfinite(value):
                self._state.set_value(
                    self._key, float(QLocale.c().toString(
                        min(self._maximum, max(self._minimum, value)),
                        "f", self._spec.decimals,
                    )),
                )
        except (ValueError, RuntimeError):
            # Failed writes are reported by the state; restore the displayed value.
            pass
        finally:
            self.changed.emit()

    @Slot(str)
    def setText(self, text: str) -> None:
        try:
            self.setValue(float(text.replace(self._spec.suffix.strip(), "").replace(",", ".").strip()))
        except ValueError:
            self.changed.emit()
        self.released()

    @Slot()
    def click(self) -> None:
        try:
            if self.enabled and not self._numeric and not self._choices:
                self._state.set_value(self._key, not self.checked)
        except (ValueError, RuntimeError):
            pass
        finally:
            self.changed.emit()

    @Slot()
    def released(self) -> None:
        try:
            self._state.flush()
        except RuntimeError:
            pass
        finally:
            self.changed.emit()


class GateText(QObject):
    """Derived gate status, updated with its settings or telemetry."""

    changed = Signal()

    def __init__(self, state: GateState, key: str, parent: QObject | None):
        super().__init__(parent)
        self._state, self._key = state, key
        state.changed.connect(self.changed)

    text = Property(str, lambda self: self._state.presentation()[self._key], notify=changed)
    toolTip = Property(str, lambda self: "", constant=True)
    name = Property(str, lambda self: "", constant=True)
    enabled = Property(bool, lambda self: True, constant=True)
    shown = Property(bool, lambda self: True, constant=True)
