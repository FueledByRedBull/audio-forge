"""Gate settings, native writes and status shared by both presentation layers."""

from __future__ import annotations

from dataclasses import asdict
import math

from PySide6.QtCore import QLocale, QObject, Signal

from ..config import GateSettings, PresetValidationError
from ..config_parts.validation import _validate_bool, _validate_range
from .control_specs import CONTROL_RANGES, control_spec, widget_control_label
from .rate_limiter import RateLimiter


GATE_RANGES = CONTROL_RANGES["gate"]
GATE_MODES = control_spec("gate", "gate_mode").choices


class GateState(QObject):
    """Own exact settings; views format them without becoming the stored values."""

    changed = Signal()
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    def __init__(self, processor, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self._settings = asdict(GateSettings())
        self._applied_settings = self._settings.copy()
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._rate_limiter._timer.setParent(self)
        self.vad_available = self._read_vad_available()
        self.confidence: float | None = None
        self.noise_floor_db: float | None = None

    def get_settings(self) -> dict:
        """Return the latest desired values, including any queued edit."""
        return self._settings.copy()

    def get_applied_settings(self) -> dict:
        """Return the last complete settings accepted by the processor."""
        return self._applied_settings.copy()

    def _validated(self, settings: dict) -> dict:
        candidate = self._settings.copy()
        for name, value in settings.items():
            try:
                if name in {"enabled", "auto_threshold_enabled"}:
                    candidate[name] = _validate_bool(value, name, "gate")
                elif name == "gate_mode":
                    if isinstance(value, bool) or not isinstance(value, int) or value not in range(3):
                        raise ValueError("Gate mode must be an integer between 0 and 2")
                    candidate[name] = value
                elif name in GATE_RANGES:
                    candidate[name] = _validate_range(
                        value, *GATE_RANGES[name], name, "gate", strict=True,
                    )
                else:
                    raise ValueError(f"Unknown gate setting: {name}")
            except PresetValidationError as error:
                raise ValueError(str(error)) from error
        return candidate

    def set_value(self, name: str, value: object) -> None:
        candidate = self._validated({name: value})
        self.refresh_vad_status()
        # A saved mode survives missing assets; explicitly selecting a VAD
        # mode without its backend still falls back to threshold-only mode.
        if name in {"gate_mode", "vad_threshold", "vad_hold_time_ms", "vad_pre_gain"}:
            if candidate["gate_mode"] > 0 and not self.vad_available:
                candidate["gate_mode"] = 0
        if candidate == self._settings:
            # A widget may already show the requested mode before it was
            # rejected; read back even when the corrected value is unchanged.
            self.changed.emit()
            return
        self._settings = candidate
        self.changed.emit()
        self._rate_limiter.call(lambda: self._apply(candidate, edited=True))

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, retaining omitted values and stored VAD modes."""
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
        self.processor.set_gate_enabled(settings["enabled"])
        self.processor.set_gate_threshold(settings["threshold_db"])
        self.processor.set_gate_attack(settings["attack_ms"])
        self.processor.set_gate_release(settings["release_ms"])
        self.processor.set_gate_mode(settings["gate_mode"])
        self.processor.set_vad_threshold(settings["vad_threshold"])
        self.processor.set_vad_hold_time(settings["vad_hold_time_ms"])
        self.processor.set_vad_pre_gain(settings["vad_pre_gain"])
        self.processor.set_auto_threshold(settings["auto_threshold_enabled"])
        self.processor.set_gate_margin(settings["gate_margin_db"])

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
                    f"Gate update failed: {error}; restoring its previous settings "
                    f"also failed: {rollback_error}"
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
        self.refresh_vad_status()
        if edited and candidate != previous:
            self.configurationEdited.emit("Noise gate edit")

    def _read_vad_available(self) -> bool:
        try:
            return bool(self.processor.is_vad_available())
        except Exception:
            return False

    def refresh_vad_status(self) -> None:
        available = self._read_vad_available()
        if available != self.vad_available:
            self.vad_available = available
            if not available:
                self.confidence = None
                self.noise_floor_db = None
            self.changed.emit()

    def update_vad_confidence(self, confidence: float | None) -> None:
        previous = (self.vad_available, self.confidence, self.noise_floor_db)
        self.vad_available = self._read_vad_available()
        self.confidence = None
        self.noise_floor_db = None
        if confidence is not None and math.isfinite(confidence) and self.vad_available:
            self.confidence = min(1.0, max(0.0, confidence))
            try:
                floor = float(self.processor.get_noise_floor())
                if math.isfinite(floor):
                    self.noise_floor_db = floor
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
                pass
        if previous != (self.vad_available, self.confidence, self.noise_floor_db):
            self.changed.emit()

    def control_enabled(self, name: str) -> bool:
        mode = self._settings["gate_mode"]
        vad_enabled = mode > 0 and self.vad_available
        automatic = mode == 1 and vad_enabled and self._settings["auto_threshold_enabled"]
        if name in {"vad_threshold", "vad_hold_time_ms", "vad_pre_gain", "confidence"}:
            return vad_enabled
        if name == "threshold_db":
            return mode != 2 and not automatic
        if name == "auto_threshold_enabled":
            return vad_enabled
        if name == "gate_margin_db":
            return automatic
        return True

    def presentation(self) -> dict[str, str]:
        """Derived labels are shared too; telemetry never changes stored settings."""
        settings = self._settings
        auto_threshold_spec = control_spec("gate", "auto_threshold_enabled")
        mode = settings["gate_mode"]
        # Match QDoubleSpinBox's two-decimal display before composing summaries.
        threshold = float(QLocale.c().toString(settings["threshold_db"], "f", 2))
        margin = float(QLocale.c().toString(settings["gate_margin_db"], "f", 2))
        vad_threshold = float(QLocale.c().toString(settings["vad_threshold"], "f", 2))
        automatic = mode == 1 and self.vad_available and settings["auto_threshold_enabled"]
        label = "Manual Threshold:"
        if mode == 2:
            label = "Level Fallback:"
            summary = (
                f"VAD Threshold: {vad_threshold:.2f} | "
                f"Level fallback when VAD is unavailable: {threshold:.1f} dB"
            )
        elif automatic:
            label = "Manual Threshold (fallback):"
            if self.noise_floor_db is None:
                summary = "Effective Threshold: -- (noise floor unavailable)"
            else:
                effective = max(-80.0, min(-10.0, self.noise_floor_db + margin))
                summary = (
                    f"Effective Threshold: {effective:.1f} dB "
                    f"({self.noise_floor_db:.1f} dB floor + {margin:.1f} dB margin)"
                )
        else:
            summary = f"Effective Threshold: {threshold:.1f} dB (manual)"
        if mode == 0:
            vad_info = "VAD: Threshold mode"
        elif not self.vad_available:
            vad_info = "VAD: Unavailable"
        elif mode == 2:
            vad_info = "VAD: Active | Speech confidence only"
        elif settings["auto_threshold_enabled"]:
            vad_info = "VAD: Active | Auto threshold on"
        else:
            vad_info = "VAD: Active | Manual threshold"
        return {
            "threshold_label": label,
            "threshold_status": summary,
            "noise_floor": (
                f"Noise Floor: {self.noise_floor_db:.1f} dB"
                if self.noise_floor_db is not None else "Noise Floor: --"
            ),
            "vad_info": vad_info,
            "auto_threshold_text": (
                "Track Noise Floor" if mode == 2
                else widget_control_label("gate", "auto_threshold_enabled")
            ),
            "auto_threshold_name": (
                "Track noise floor" if mode == 2 else auto_threshold_spec.accessible_name
            ),
            "auto_threshold_tooltip": (
                "Track the noise floor without changing the VAD speech threshold."
                if mode == 2
                else auto_threshold_spec.tooltip
            ),
        }
