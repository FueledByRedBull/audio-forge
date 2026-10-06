"""Limiter settings and native writes shared by the widget and Quick views."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import QObject, Signal

from ..config import LimiterSettings, PresetValidationError
from ..config_parts.validation import (
    _validate_bool,
    _validate_range,
)
from .control_specs import CONTROL_RANGES
from .rate_limiter import RateLimiter


LIMITER_RANGES = CONTROL_RANGES["limiter"]


class LimiterState(QObject):
    """Keep exact requested values independently of either view's formatting.

    Interactive changes are visible immediately and coalesced for native writes.
    Only successful writes produce history entries; a rejected write restores the
    last applied values and reports its error to both the caller and the views.
    """

    changed = Signal()
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    def __init__(self, processor, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self._settings = asdict(LimiterSettings())
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
        return name in self._settings

    def _validated(self, settings: dict) -> dict:
        candidate = self._settings.copy()
        for name, value in settings.items():
            try:
                if name in {"enabled", "careful_output_enabled"}:
                    candidate[name] = _validate_bool(value, name, "limiter")
                elif name in LIMITER_RANGES:
                    candidate[name] = _validate_range(
                        value, *LIMITER_RANGES[name], name, "limiter",
                        strict=True,
                    )
                else:
                    raise ValueError(f"Unknown limiter setting: {name}")
            except PresetValidationError as error:
                raise ValueError(str(error)) from error
        return candidate

    def set_value(self, name: str, value: object) -> None:
        """Apply one interactive edit without rounding other stored values."""
        candidate = self._validated({name: value})
        if candidate == self._settings:
            return
        self._settings = candidate
        self.changed.emit()
        self._rate_limiter.call(lambda: self._apply(candidate, edited=True))

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, superseding any queued interactive write."""
        candidate = self._validated(settings)
        self._rate_limiter.call_now(lambda: self._apply(candidate, edited=False))

    def flush(self) -> None:
        self._rate_limiter.flush()

    def cancel(self) -> None:
        self._rate_limiter.cancel()
        if self._settings != self._applied_settings:
            self._settings = self._applied_settings.copy()
            self.changed.emit()

    def _write(self, settings: dict) -> None:
        self.processor.set_limiter_enabled(settings["enabled"])
        self.processor.set_limiter_careful_output_enabled(
            settings["careful_output_enabled"]
        )
        self.processor.set_limiter_ceiling(settings["ceiling_db"])
        self.processor.set_limiter_release(settings["release_ms"])

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
                    f"Limiter update failed: {error}; "
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
            self.configurationEdited.emit("Limiter edit")
