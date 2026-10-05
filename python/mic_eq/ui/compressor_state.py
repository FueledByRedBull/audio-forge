"""Exact compressor settings and calibration metadata independent of either view."""

from __future__ import annotations

from dataclasses import asdict
import math

from PySide6.QtCore import QObject, Signal

from ..config import CompressorSettings, PresetValidationError
from ..config_parts.validation import (
    VALIDATION_RANGES,
    _validate_bool,
    _validate_range,
)
from .rate_limiter import RateLimiter


_BOOLEAN_SETTINGS = {
    "enabled", "adaptive_release", "auto_makeup_enabled", "sidechain_highpass_enabled",
}
COMPRESSOR_RANGES = {
    name: bounds for name, bounds in VALIDATION_RANGES["compressor"].items()
    if name not in _BOOLEAN_SETTINGS
} | {"base_release_ms": (20.0, 200.0)}
_PARAMETER_KEYS = frozenset(asdict(CompressorSettings()))
_INTENSITY_KEYS = _PARAMETER_KEYS - {"target_lufs"}


class CompressorState(QObject):
    """Own write ordering, coalescing and calibration provenance for both views."""

    changed = Signal()
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    def __init__(self, processor, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self._settings = asdict(CompressorSettings()) | {
            "noise_reference_reliability": 0.0,
            "dynamics_profile": "balanced",
            "dynamics_customized": False,
        }
        self._applied_settings = self._settings.copy()
        self._calibrated_controls: dict | None = None
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._rate_limiter._timer.setParent(self)

    def get_settings(self, *, include_calibration: bool = False) -> dict:
        if not include_calibration:
            return {key: value for key, value in self._settings.items() if key in _PARAMETER_KEYS}
        return self._settings | {
            "dynamics_intensity": (
                "customized" if self._settings["dynamics_customized"]
                else self._settings["dynamics_profile"]
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
        candidate = self._settings.copy()
        for name, value in settings.items():
            try:
                if name in _BOOLEAN_SETTINGS:
                    candidate[name] = _validate_bool(value, name, "compressor")
                elif name in COMPRESSOR_RANGES:
                    candidate[name] = _validate_range(
                        value, *COMPRESSOR_RANGES[name], name, "compressor", strict=True,
                    )
                else:
                    raise ValueError(f"Unknown compressor setting: {name}")
            except PresetValidationError as error:
                raise ValueError(str(error)) from error
        return candidate

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
        self._settings = candidate
        self.changed.emit()
        label = (
            "Adaptive release edit" if name in {"adaptive_release", "base_release_ms"}
            else "Auto makeup edit" if name in {"auto_makeup_enabled", "target_lufs"}
            else "Compressor edit"
        )
        def apply():
            self._apply(candidate, baseline, label=label)

        if immediate:
            self._rate_limiter.call_now(apply)
        else:
            self._rate_limiter.call(apply)

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline; omission invalidates transient noise evidence."""
        # Recommendation dictionaries also contain analysis diagnostics; as in
        # the panel API, only actual controls and supported metadata are applied.
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
        self._rate_limiter.call_now(lambda: self._apply(candidate, baseline))

    def flush(self) -> None:
        self._rate_limiter.flush()

    def cancel(self) -> None:
        self._rate_limiter.cancel()
        if self._settings != self._applied_settings:
            self._settings = self._applied_settings.copy()
            self.changed.emit()

    def _write(self, settings: dict) -> None:
        self.processor.set_compressor_enabled(settings["enabled"])
        self.processor.set_compressor_threshold(settings["threshold_db"])
        self.processor.set_compressor_ratio(settings["ratio"])
        self.processor.set_compressor_attack(settings["attack_ms"])
        self.processor.set_compressor_adaptive_release(settings["adaptive_release"])
        if settings["adaptive_release"]:
            self.processor.set_compressor_base_release(settings["base_release_ms"])
        else:
            self.processor.set_compressor_release(settings["release_ms"])
        self.processor.set_compressor_makeup_gain(settings["makeup_gain_db"])
        self.processor.set_compressor_sidechain_highpass_enabled(settings["sidechain_highpass_enabled"])
        self.processor.set_compressor_auto_makeup_enabled(settings["auto_makeup_enabled"])
        self.processor.set_compressor_target_lufs(settings["target_lufs"])
        self.processor.set_compressor_noise_reference_reliability(settings["noise_reference_reliability"])

    def _apply(self, candidate: dict, baseline: dict | None, *, label: str | None = None) -> None:
        previous = self._applied_settings
        try:
            self._write(candidate)
        except Exception as error:
            self.cancel()
            try:
                self._write(previous)
            except Exception as rollback_error:
                message = (
                    f"Compressor update failed: {error}; "
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
        self._calibrated_controls = baseline
        if self._settings != candidate:
            self._settings = candidate
            self.changed.emit()
        if label is not None and candidate != previous:
            self.configurationEdited.emit(label)
