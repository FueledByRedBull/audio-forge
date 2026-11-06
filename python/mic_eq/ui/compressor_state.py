"""Exact compressor settings and calibration metadata independent of the view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import math

from PySide6.QtCore import QObject

from ..config import CompressorSettings
from .control_specs import CONTROL_RANGES
from .stage_state import RateLimitedStageState, validated_settings


_BOOLEAN_SETTINGS = {
    "enabled", "adaptive_release", "auto_makeup_enabled", "sidechain_highpass_enabled",
}
COMPRESSOR_RANGES = CONTROL_RANGES["compressor"]
_PARAMETER_KEYS = frozenset(asdict(CompressorSettings()))
_INTENSITY_KEYS = _PARAMETER_KEYS - {"target_lufs"}


class CompressorState(RateLimitedStageState[dict]):
    """Own write ordering, coalescing and calibration provenance."""

    _stage_name = "Compressor"

    def __init__(
        self,
        processor,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(
            processor,
            asdict(CompressorSettings()) | {
                "noise_reference_reliability": 0.0,
                "dynamics_profile": "balanced",
                "dynamics_customized": False,
            },
            parent,
            stop_audio=stop_audio,
        )
        self._calibrated_controls: dict | None = None

    def get_settings(self, *, include_calibration: bool = False) -> dict:
        """Return desired settings; interactive values may still be queued."""
        if not include_calibration:
            return {key: value for key, value in self._settings.items() if key in _PARAMETER_KEYS}
        return self._settings | {
            "dynamics_intensity": (
                "customized" if self._settings["dynamics_customized"]
                else self._settings["dynamics_profile"]
            ),
        }

    def get_applied_settings(self, *, include_calibration: bool = False) -> dict:
        """Return processor-accepted settings with the same optional metadata."""
        settings = self._applied_settings
        if not include_calibration:
            return {key: value for key, value in settings.items() if key in _PARAMETER_KEYS}
        return settings | {
            "dynamics_intensity": (
                "customized" if settings["dynamics_customized"]
                else settings["dynamics_profile"]
            ),
        }

    def control_enabled(self, name: str) -> bool:
        if name == "release_ms":
            return not self._settings["adaptive_release"]
        if name == "base_release_ms":
            return bool(self._settings["adaptive_release"])
        if name == "makeup_gain_db":
            return not self._settings["auto_makeup_enabled"]
        if name == "target_lufs":
            return bool(self._settings["auto_makeup_enabled"])
        return True

    def _validated(self, settings: dict) -> dict:
        return validated_settings(
            self._settings, settings, section="compressor", noun="compressor",
            booleans=_BOOLEAN_SETTINGS, ranges=COMPRESSOR_RANGES,
        )

    def set_value(self, name: str, value: object) -> None:
        if name not in _PARAMETER_KEYS:
            raise ValueError(f"Unknown compressor control: {name}")
        candidate = self._validated({name: value})
        if candidate == self._settings:
            return
        immediate = name in {
            "adaptive_release", "base_release_ms", "auto_makeup_enabled", "target_lufs",
        }
        if immediate:
            # A queued manual release must finish before switching modes.
            self.flush()
        baseline = self._calibrated_controls
        if baseline is not None and any(candidate[key] != baseline[key] for key in _INTENSITY_KEYS):
            candidate["dynamics_customized"] = True
        label = (
            "Adaptive release edit" if name in {"adaptive_release", "base_release_ms"}
            else "Auto makeup edit" if name in {"auto_makeup_enabled", "target_lufs"}
            else "Compressor edit"
        )
        self._queue(
            candidate, label=label, immediate=immediate,
            on_accepted=self._calibrated(baseline),
        )

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline; omission invalidates transient noise evidence."""
        # Recommendation dictionaries also contain analysis diagnostics; only
        # actual controls and supported metadata are applied.
        candidate = self._validated({
            key: value for key, value in settings.items() if key in _PARAMETER_KEYS
        })
        baseline = self._calibrated_controls
        profile = settings.get("dynamics_profile", settings.get("dynamics_intensity"))
        if profile in {"gentle", "balanced", "dense", "custom"}:
            candidate["dynamics_profile"] = profile
            candidate["dynamics_customized"] = bool(settings.get("dynamics_customized", False))
            baseline = {key: candidate[key] for key in _INTENSITY_KEYS}
        elif _INTENSITY_KEYS.intersection(settings):
            # Numeric presets carry no calibration provenance.
            candidate["dynamics_profile"] = "custom"
            candidate["dynamics_customized"] = False
            baseline = {key: candidate[key] for key in _INTENSITY_KEYS}
        try:
            reliability = float(settings.get("noise_reference_reliability", 0.0))
        except (TypeError, ValueError):
            reliability = math.nan
        if math.isfinite(reliability):
            candidate["noise_reference_reliability"] = min(1.0, max(0.0, reliability))
        self._apply_inline(candidate, on_accepted=self._calibrated(baseline))

    def _calibrated(self, baseline: dict | None) -> Callable[[], None]:
        """Adopt a calibration baseline only once the write carrying it is accepted."""
        def accept() -> None:
            self._calibrated_controls = baseline

        return accept

    def _write(self, settings: dict) -> None:
        self.processor.apply_compressor_settings(settings)
