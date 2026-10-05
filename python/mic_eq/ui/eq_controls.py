"""Signal-driven EQ controls for QML, with no processing-panel dependency."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QLocale, Property, Signal, Slot

from ..config import EQ_SLOPES_DB_PER_OCTAVE
from .eq_state import (
    EQState, EQ_FILTER_OPTIONS, GAIN_FILTER_TYPES, PASS_FILTER_TYPES, Q_FILTER_TYPES,
    _format_auto_eq_diagnostics, _format_frequency_label,
)


class EQControl(QObject):
    """Present one shared setting using the existing QML control contract."""

    changed = Signal()

    def __init__(
        self, state: EQState, key: str, index: int | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        allowed = {"enabled", "band", "layers", "diagnostics"}
        if index is not None:
            state.get_band(index)
            allowed = {"frequencyLabel", "enabled", "type", "gain", "gainLabel", "frequency", "q", "slope"}
        if key not in allowed:
            raise ValueError(f"Unknown EQ control: {key}")
        self._owner = state
        self._key = key
        self._band_index = index
        self._locale = QLocale()
        self._locale.setNumberOptions(
            self._locale.numberOptions() | QLocale.NumberOption.OmitGroupSeparator,
        )
        self._view = self._read()
        state.changed.connect(self._refresh)
        state.presentationChanged.connect(self._refresh)

    def _read(self) -> dict:
        view = {
            "text": "", "name": "", "toolTip": "", "enabled": True, "shown": True,
            "value": 0.0, "minimum": 0.0, "maximum": 1.0, "step": 0.0,
            "checked": False, "items": [], "index": -1, "state": "", "data": {},
        }
        key, index = self._key, self._band_index
        if index is None:
            if key == "enabled":
                view.update(name="Enable EQ", checked=self._owner.get_eq_settings().enabled)
            elif key == "band":
                view["index"] = self._owner.selected_band or 0
            elif key == "layers":
                view.update(name="Active EQ layers", text=self._owner.layer_status())
            else:
                diagnostics = self._owner.auto_eq_diagnostics
                text, status, tip = _format_auto_eq_diagnostics(diagnostics)
                view.update(
                    text=text, state=status,
                    toolTip=tip or "Auto-EQ diagnostics appear after calibration.",
                    data={"calibrated": diagnostics is not None},
                )
            return view

        band = self._owner.get_band(index)
        band_name = f"EQ band {index + 1}"
        if key == "enabled":
            view.update(name=f"Enable {band_name}", text="On", toolTip="Enable this EQ band", checked=band.enabled)
        elif key == "frequencyLabel":
            view.update(text=_format_frequency_label(band.frequency_hz), toolTip=f"{band_name} center frequency")
        elif key in {"gain", "gainLabel"}:
            value = int(round(band.gain_db * 10))
            view["enabled"] = band.filter_type in GAIN_FILTER_TYPES
            if key == "gain":
                view.update(name=f"{band_name} gain", value=float(value), minimum=-120.0, maximum=120.0, step=1.0)
            else:
                view["text"] = f"{value / 10:+.1f}" if value else "0"
        elif key in {"frequency", "q"}:
            frequency = key == "frequency"
            decimals = 0 if frequency else 1
            value = float(QLocale.c().toString(band.frequency_hz if frequency else band.q, "f", decimals))
            view.update(
                name=f"{band_name} center frequency" if frequency else f"{band_name} Q",
                toolTip="Center frequency in Hz" if frequency else "",
                value=value, text=self._locale.toString(value, "f", decimals) + (" Hz" if frequency else ""),
                minimum=20.0 if frequency else 0.1, maximum=20000.0 if frequency else 10.0,
                step=10.0 if frequency else 0.1,
                enabled=frequency or band.filter_type in Q_FILTER_TYPES,
            )
        elif key == "type":
            choices = [choice for _name, choice in EQ_FILTER_OPTIONS]
            view.update(
                name=f"{band_name} filter type", index=choices.index(band.filter_type),
                items=[name for name, _choice in EQ_FILTER_OPTIONS],
                text=EQ_FILTER_OPTIONS[choices.index(band.filter_type)][0],
                toolTip=(
                    "Notch: gain is ignored; Q controls rejection width" if band.filter_type == "notch"
                    else "Pass filter: gain and Q are ignored; slope controls order" if band.filter_type in PASS_FILTER_TYPES
                    else "Filter type"
                ),
            )
        else:
            choices = sorted(EQ_SLOPES_DB_PER_OCTAVE)
            view.update(
                name=f"{band_name} pass-filter slope", index=choices.index(band.slope_db_per_octave),
                items=[f"{slope} dB/oct" for slope in choices],
                text=f"{band.slope_db_per_octave} dB/oct", toolTip="Pass-filter slope in dB per octave",
                enabled=band.filter_type in PASS_FILTER_TYPES,
            )
        return view

    def _refresh(self, *, force: bool = False) -> None:
        view = self._read()
        if force or view != self._view:
            self._view = view
            self.changed.emit()

    text = Property(str, lambda self: self._view["text"], notify=changed)
    name = Property(str, lambda self: self._view["name"], notify=changed)
    toolTip = Property(str, lambda self: self._view["toolTip"], notify=changed)
    enabled = Property(bool, lambda self: self._view["enabled"], notify=changed)
    shown = Property(bool, lambda self: self._view["shown"], notify=changed)
    value = Property(float, lambda self: self._view["value"], notify=changed)
    minimum = Property(float, lambda self: self._view["minimum"], notify=changed)
    maximum = Property(float, lambda self: self._view["maximum"], notify=changed)
    step = Property(float, lambda self: self._view["step"], notify=changed)
    checked = Property(bool, lambda self: self._view["checked"], notify=changed)
    items = Property(list, lambda self: self._view["items"], notify=changed)
    index = Property(int, lambda self: self._view["index"], notify=changed)
    state = Property(str, lambda self: self._view["state"], notify=changed)
    data = Property(dict, lambda self: self._view["data"], notify=changed)

    @Slot(float)
    def setValue(self, value: float) -> None:
        try:
            if self._band_index is not None and self.enabled and math.isfinite(value):
                key = {"gain": "gain_db", "frequency": "frequency_hz", "q": "q"}.get(self._key)
                if key is not None:
                    value = min(self._view["maximum"], max(self._view["minimum"], value))
                    value = round(value) / 10.0 if key == "gain_db" else float(
                        QLocale.c().toString(value, "f", 0 if key == "frequency_hz" else 1)
                    )
                    self._owner.set_band_value(self._band_index, key, value)
        except (ValueError, RuntimeError):
            # The controller reports native failures; always read accepted values back.
            pass
        finally:
            self._refresh(force=True)

    @Slot(str)
    def setText(self, text: str) -> None:
        if self._key not in {"frequency", "q"}:
            return
        try:
            self.setValue(float(text.replace("Hz", "").replace(",", ".").strip()))
        except ValueError:
            self._refresh(force=True)
        self.released()

    @Slot(int)
    def setIndex(self, index: int) -> None:
        try:
            if self._band_index is None and self._key == "band":
                if 0 <= index < 10:
                    self._owner.select_band(index)
            elif self._band_index is not None and self.enabled and 0 <= index < len(self._view["items"]):
                if self._key == "type":
                    self._owner.set_band_value(self._band_index, "filter_type", EQ_FILTER_OPTIONS[index][1])
                elif self._key == "slope":
                    self._owner.set_band_value(self._band_index, "slope_db_per_octave", sorted(EQ_SLOPES_DB_PER_OCTAVE)[index])
        except (ValueError, RuntimeError):
            pass
        finally:
            self._refresh(force=True)

    @Slot()
    def click(self) -> None:
        try:
            if self._key == "enabled":
                if self._band_index is None:
                    self._owner.set_enabled(not self.checked)
                else:
                    self._owner.set_band_value(self._band_index, "enabled", not self.checked)
        except (ValueError, RuntimeError):
            pass
        finally:
            self._refresh(force=True)

    @Slot()
    def released(self) -> None:
        try:
            self._owner.flush()
        except RuntimeError:
            pass
        finally:
            self._refresh(force=True)
