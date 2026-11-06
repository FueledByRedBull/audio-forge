"""Limiter settings and native writes, independent of the view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from PySide6.QtCore import QObject

from ..config import LimiterSettings
from .control_specs import CONTROL_RANGES
from .stage_state import RateLimitedStageState, validated_settings


LIMITER_RANGES = CONTROL_RANGES["limiter"]


class LimiterState(RateLimitedStageState[dict]):
    """Keep exact requested values independently of the view's formatting.

    Interactive changes are visible immediately and coalesced for native writes.
    Only successful writes produce history entries; a rejected write restores the
    last applied values and reports its error to both the caller and the view.
    """

    _stage_name = "Limiter"

    def __init__(
        self,
        processor,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(processor, asdict(LimiterSettings()), parent, stop_audio=stop_audio)

    def get_settings(self) -> dict:
        """Return the latest desired values, including any queued edit."""
        return self._settings.copy()

    def get_applied_settings(self) -> dict:
        """Return the last complete settings accepted by the processor."""
        return self._applied_settings.copy()

    def control_enabled(self, name: str) -> bool:
        return name in self._settings

    def _validated(self, settings: dict) -> dict:
        return validated_settings(
            self._settings, settings, section="limiter", noun="limiter",
            booleans={"enabled", "careful_output_enabled"}, ranges=LIMITER_RANGES,
        )

    def set_value(self, name: str, value: object) -> None:
        """Apply one interactive edit without rounding other stored values."""
        candidate = self._validated({name: value})
        if candidate == self._settings:
            return
        self._queue(candidate, label="Limiter edit")

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, superseding any queued interactive write."""
        candidate = self._validated(settings)
        self._apply_inline(candidate)

    def _write(self, settings: dict) -> None:
        self.processor.set_limiter_enabled(settings["enabled"])
        self.processor.set_limiter_careful_output_enabled(
            settings["careful_output_enabled"]
        )
        self.processor.set_limiter_ceiling(settings["ceiling_db"])
        self.processor.set_limiter_release(settings["release_ms"])
