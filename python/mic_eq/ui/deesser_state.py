"""De-esser settings and native writes shared by the widget and Quick views."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import QObject, Signal

from ..config import DeEsserSettings, PresetValidationError
from ..config_parts.validation import (
    _validate_bool,
    _validate_range,
)
from .control_specs import CONTROL_RANGES
from .rate_limiter import RateLimiter


DEESSER_RANGES = CONTROL_RANGES["deesser"]


class DeEsserState(QObject):
    """Own exact settings, cutoff normalization and successful-write history."""

    changed = Signal()
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    def __init__(self, processor, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self._settings = asdict(DeEsserSettings())
        self._applied_settings = self._settings.copy()
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._rate_limiter._timer.setParent(self)

    def get_settings(self) -> dict:
        """Return the latest desired values, including any queued edit."""
        return self._settings.copy()

    def get_applied_settings(self) -> dict:
        """Return the last complete settings accepted by the processor."""
        return self._applied_settings.copy()

    def control_enabled(self, name: str) -> bool:
        return name not in {"threshold_db", "ratio"} or not self._settings["auto_enabled"]

    def _validated(self, settings: dict, *, source: str = "low_cut_hz") -> dict:
        candidate = self._settings.copy()
        for name, value in settings.items():
            try:
                if name in {"enabled", "auto_enabled"}:
                    candidate[name] = _validate_bool(value, name, "deesser")
                elif name in DEESSER_RANGES:
                    candidate[name] = _validate_range(
                        value, *DEESSER_RANGES[name], name, "deesser", strict=True,
                    )
                else:
                    raise ValueError(f"Unknown de-esser setting: {name}")
            except PresetValidationError as error:
                raise ValueError(str(error)) from error
        if candidate["high_cut_hz"] < candidate["low_cut_hz"] + 200.0:
            if source == "high_cut_hz":
                candidate["low_cut_hz"] = candidate["high_cut_hz"] - 200.0
            else:
                candidate["high_cut_hz"] = candidate["low_cut_hz"] + 200.0
        return candidate

    def set_value(self, name: str, value: object) -> None:
        candidate = self._validated({name: value}, source=name)
        if candidate == self._settings:
            return
        self._settings = candidate
        self.changed.emit()
        self._rate_limiter.call(lambda: self._apply(candidate, edited=True))

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, retaining the existing defaults for omissions."""
        candidate = self._validated(asdict(DeEsserSettings()) | settings)
        self._rate_limiter.call_now(lambda: self._apply(candidate, edited=False))

    def flush(self) -> None:
        self._rate_limiter.flush()

    def cancel(self) -> None:
        self._rate_limiter.cancel()
        if self._settings != self._applied_settings:
            self._settings = self._applied_settings.copy()
            self.changed.emit()

    def _write(self, settings: dict) -> None:
        self.processor.set_deesser_enabled(settings["enabled"])
        self.processor.set_deesser_auto_enabled(settings["auto_enabled"])
        self.processor.set_deesser_auto_amount(settings["auto_amount"])
        # Expand first so native validation never clamps a valid new interval
        # against the old narrow range; use the native value during rollback too.
        if settings["high_cut_hz"] > self.processor.get_deesser_high_cut_hz():
            self.processor.set_deesser_high_cut_hz(settings["high_cut_hz"])
            self.processor.set_deesser_low_cut_hz(settings["low_cut_hz"])
        else:
            self.processor.set_deesser_low_cut_hz(settings["low_cut_hz"])
            self.processor.set_deesser_high_cut_hz(settings["high_cut_hz"])
        self.processor.set_deesser_threshold_db(settings["threshold_db"])
        self.processor.set_deesser_ratio(settings["ratio"])
        self.processor.set_deesser_attack_ms(settings["attack_ms"])
        self.processor.set_deesser_release_ms(settings["release_ms"])
        self.processor.set_deesser_max_reduction_db(settings["max_reduction_db"])

    def _apply(self, candidate: dict, *, edited: bool) -> None:
        previous = self._applied_settings
        try:
            self._write(candidate)
        except Exception as error:
            self.cancel()
            try:
                self._write(previous)
            except Exception as rollback_error:
                message = (
                    f"De-esser update failed: {error}; "
                    f"restoring its previous settings also failed: {rollback_error}"
                )
                self.recoveryFailed.emit(message)
                try:
                    self.processor.stop()
                except Exception as stop_error:
                    message += f"; stopping audio also failed: {stop_error}"
                self.writeFailed.emit(message)
                raise RuntimeError(message) from error
            self.writeFailed.emit(str(error))
            raise RuntimeError(str(error)) from error
        self._applied_settings = candidate.copy()
        if self._settings != candidate:
            self._settings = candidate
            self.changed.emit()
        if edited and candidate != previous:
            self.configurationEdited.emit("De-esser edit")
