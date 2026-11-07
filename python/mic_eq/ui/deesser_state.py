"""De-esser settings and native writes, independent of the view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from PySide6.QtCore import QObject

from ..config import DeEsserSettings
from .control_specs import CONTROL_RANGES
from .stage_state import RateLimitedStageState, validated_settings


DEESSER_RANGES = CONTROL_RANGES["deesser"]


class DeEsserState(RateLimitedStageState[dict]):
    """Own exact settings, cutoff normalization and successful-write history."""

    _stage_name = "De-esser"

    def __init__(
        self,
        processor,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(processor, asdict(DeEsserSettings()), parent, stop_audio=stop_audio)

    def get_settings(self) -> dict:
        """Return the latest desired values, including any queued edit."""
        return self._settings.copy()

    def get_applied_settings(self) -> dict:
        """Return the last complete settings accepted by the processor."""
        return self._applied_settings.copy()

    def control_enabled(self, name: str) -> bool:
        return name not in {"threshold_db", "ratio"} or not self._settings["auto_enabled"]

    def _validated(self, settings: dict, *, source: str = "low_cut_hz") -> dict:
        candidate = validated_settings(
            self._settings, settings, section="deesser", noun="de-esser",
            booleans={"enabled", "auto_enabled"}, ranges=DEESSER_RANGES,
        )
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
        self._queue(candidate, label="De-esser edit")

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, retaining current values for omitted keys."""
        candidate = self._validated(settings)
        self._apply_inline(candidate)

    def _write(self, settings: dict) -> None:
        self.processor.apply_deesser_settings(settings)
