"""Reusable meter views, independent of the processing control panels."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLabel, QWidget

from .gate_state import GateState
from .level_meter import ConfidenceMeter, GainReductionMeter
from .layout_constants import METER_LABEL_STYLE
from .shell_state import TextState


class ProcessingMeters(QObject):
    """Keep the existing painted meters and cadence in either frontend."""

    def __init__(self, processor, gate_state: GateState, parent: QWidget):
        super().__init__(parent)
        self.processor = processor
        self.gate_state = gate_state
        self.compressor_gr = GainReductionMeter(parent)
        self.compressor_gr.setAccessibleName("Compressor gain reduction")
        self.deesser_gr = GainReductionMeter(parent)
        self.deesser_gr.setAccessibleName("De-esser gain reduction")
        self.confidence = ConfidenceMeter(parent)
        self.confidence.setAccessibleName("Voice activity confidence")
        self.confidence.setToolTip("Real-time VAD confidence (red=low, green=high)")
        self.current_release = QLabel("--", parent)
        self.current_release.setAccessibleName("Current compressor release time")
        self.current_lufs = QLabel("--", parent)
        self.current_lufs.setAccessibleName("Current compressor loudness")
        self.current_makeup_gain = QLabel("--", parent)
        self.current_makeup_gain.setAccessibleName("Current automatic makeup gain")
        self.current_release.setToolTip("Current release time (adaptive or manual)")
        self.current_lufs.setToolTip("Current measured loudness (EBU R128 momentary)")
        self.current_makeup_gain.setToolTip("Current auto makeup gain applied")
        for label in (self.current_release, self.current_lufs, self.current_makeup_gain):
            label.setStyleSheet(METER_LABEL_STYLE)
        self.text_states = {}
        for name in ("current_release", "current_lufs", "current_makeup_gain"):
            label = getattr(self, name)
            state = TextState(self, text="--", name=label.accessibleName(), tip=label.toolTip())
            state.bind_label(label)
            self.text_states[name] = state
        for widget in (self.compressor_gr, self.deesser_gr, self.confidence,
                       self.current_release, self.current_lufs, self.current_makeup_gain):
            widget.show()
        gate_state.changed.connect(self._render_gate)
        self._render_gate()

    def _render_gate(self) -> None:
        self.confidence.set_threshold(float(self.gate_state.get_settings()["vad_threshold"]))
        self.confidence.set_confidence(self.gate_state.confidence)
        self.confidence.setEnabled(self.gate_state.control_enabled("confidence"))

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
