"""Gate settings, native writes and status shared by both presentation layers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import math

from PySide6.QtCore import QLocale, QObject, Signal

from ..config import GateSettings
from .control_specs import CONTROL_RANGES, control_spec
from .stage_state import RateLimitedStageState, validated_settings


GATE_RANGES = CONTROL_RANGES["gate"]
GATE_MODES = control_spec("gate", "gate_mode").choices


def _gate_mode(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(3):
        raise ValueError("Gate mode must be an integer between 0 and 2")
    return value


class GateState(RateLimitedStageState[dict]):
    """Own exact settings; the view formats them without becoming the stored values."""

    # The confidence and noise floor move every meter tick; only the views that show them listen.
    telemetryChanged = Signal()
    _stage_name = "Gate"

    def __init__(
        self,
        processor,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(processor, asdict(GateSettings()), parent, stop_audio=stop_audio)
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
        return validated_settings(
            self._settings, settings, section="gate", noun="gate",
            booleans={"enabled", "auto_threshold_enabled"}, ranges=GATE_RANGES,
            special={"gate_mode": _gate_mode},
        )

    def set_value(self, name: str, value: object) -> None:
        candidate = self._validated({name: value})
        self.refresh_vad_status()
        # A saved mode survives missing assets; explicitly selecting a VAD
        # mode without its backend still falls back to threshold-only mode.
        if name in {"gate_mode", "vad_threshold", "vad_hold_time_ms", "vad_pre_gain"}:
            if candidate["gate_mode"] > 0 and not self.vad_available:
                candidate["gate_mode"] = 0
        if candidate == self._settings:
            # The view may already show the requested mode before it was
            # rejected; read back even when the corrected value is unchanged.
            self.changed.emit()
            return
        self._queue(candidate, label="Noise gate edit")

    def set_settings(self, settings: dict) -> None:
        """Apply a preset inline, retaining omitted values and stored VAD modes."""
        candidate = self._validated(settings)
        self._apply_inline(candidate)

    def _write(self, settings: dict) -> None:
        self.processor.apply_gate_settings(settings)

    def _after_apply(self) -> None:
        self.refresh_vad_status()

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
        if previous[0] != self.vad_available:
            self.changed.emit()
        elif previous != (self.vad_available, self.confidence, self.noise_floor_db):
            self.telemetryChanged.emit()

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
        if mode == 2:
            summary = (
                f"Voice detection threshold: {vad_threshold:.2f} | "
                f"Level fallback when voice detection is unavailable: {threshold:.1f} dB"
            )
        elif automatic:
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
            vad_info = "Voice detection: off (level threshold only)"
        elif not self.vad_available:
            vad_info = "Voice detection: unavailable"
        elif mode == 2:
            vad_info = "Voice detection: on | Speech confidence only"
        elif settings["auto_threshold_enabled"]:
            vad_info = "Voice detection: on | Auto threshold on"
        else:
            vad_info = "Voice detection: on | Manual threshold"
        return {
            "threshold_status": summary,
            "noise_floor": (
                f"Noise Floor: {self.noise_floor_db:.1f} dB"
                if self.noise_floor_db is not None else "Noise Floor: --"
            ),
            "vad_info": vad_info,
            "auto_threshold_name": (
                "Auto threshold (gate): track noise floor" if mode == 2 else auto_threshold_spec.accessible_name
            ),
            "auto_threshold_tooltip": (
                "Track the noise floor without changing the VAD speech threshold."
                if mode == 2
                else auto_threshold_spec.tooltip
            ),
        }
