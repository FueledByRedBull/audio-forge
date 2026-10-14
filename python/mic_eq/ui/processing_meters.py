"""Reusable meter models, independent of any view."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject

from .gate_state import GateState
from .level_meter import ConfidenceMeter, GainReductionMeter
from .shell_state import TextState

COMPRESSOR_GR_NAME = "Compressor gain reduction"
DEESSER_GR_NAME = "De-esser gain reduction"
CONFIDENCE_NAME = "Voice activity confidence"
CONFIDENCE_TIP = "Real-time VAD confidence (red=low, green=high)"
# Accessible name and tooltip of each compressor readout.
READOUTS = {
    "current_release": (
        "Current compressor release time", "Current release time (adaptive or manual)",
    ),
    "current_lufs": (
        "Current compressor loudness", "Current measured loudness (EBU R128 momentary)",
    ),
    "current_makeup_gain": (
        "Current automatic makeup gain", "Current auto makeup gain applied",
    ),
}


class ProcessingMeters(QObject):
    """Meter models and readout texts, with no widgets of their own.

    Without a gate state the confidence meter stays unavailable.
    """

    def __init__(self, processor, gate_state: GateState | None, parent: QObject | None = None):
        super().__init__(parent)
        self.processor = processor
        self.gate_state = gate_state
        self.compressor_gr = GainReductionMeter(COMPRESSOR_GR_NAME, parent=self)
        self.deesser_gr = GainReductionMeter(DEESSER_GR_NAME, parent=self)
        self.confidence = ConfidenceMeter(CONFIDENCE_NAME, CONFIDENCE_TIP, parent=self)
        self.text_states = {
            key: TextState(self, text="--", name=name, tip=tip)
            for key, (name, tip) in READOUTS.items()
        }
        if gate_state is not None:
            gate_state.changed.connect(self._render_gate)
            self._render_gate()

    def _render_gate(self) -> None:
        gate_state = self.gate_state
        assert gate_state is not None  # Only connected when there is a gate state.
        self.confidence.set_threshold(float(gate_state.get_settings()["vad_threshold"]))
        self.confidence.set_confidence(gate_state.confidence)
        self.confidence.set_enabled(gate_state.control_enabled("confidence"))

    def update_current_release(self) -> None:
        processor = self.processor
        try:
            active = (processor.is_running() and processor.is_compressor_enabled()
                      and not processor.is_bypass() and not processor.is_raw_monitor_enabled())
            value = processor.get_compressor_current_release() if active else None
            text = f"{value:.0f} ms" if value is not None and math.isfinite(value) else "--"
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            text = "--"
        self.text_states["current_release"].update(text=text)

    def update_auto_makeup(self, loudness: float | None, gain: float | None) -> None:
        for label, value, unit in (
            (self.text_states["current_lufs"], loudness, "LUFS"),
            (self.text_states["current_makeup_gain"], gain, "dB"),
        ):
            label.update(text=f"{value:.1f} {unit}" if value is not None and math.isfinite(value) else "--")
