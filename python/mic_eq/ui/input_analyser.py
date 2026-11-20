"""Reads the engine's input spectrum into the EQ graph, which draws it behind the curves."""

from __future__ import annotations

import numpy as np

from .eq_graph import EQGraphModel


class InputAnalyser:
    """Polls ``processor.read_input_spectrum`` into one reused buffer and hands it to the graph model.

    The buffer and the band frequencies are allocated once, at the first poll.
    The graph repaints only for a new frame; a reading of the same frame, or
    none, leaves it alone.
    """

    def __init__(self, graph: EQGraphModel) -> None:
        self._graph = graph
        self._levels: np.ndarray | None = None
        self._frequencies: np.ndarray | None = None
        self._sequence = 0

    def poll(self, processor) -> None:
        """Take the latest spectrum; with none to show, or none readable, clear the graph's."""
        try:
            if self._levels is None or self._frequencies is None:
                bands = int(processor.input_spectrum_bin_count())
                frequencies = np.zeros(bands, dtype=np.float32)
                processor.input_spectrum_bin_frequencies(frequencies)
                self._levels, self._frequencies = np.zeros(bands, dtype=np.float32), frequencies
            sequence = processor.read_input_spectrum(self._levels)
        except (AttributeError, OSError, RuntimeError):
            self.clear()
            return
        if sequence == 0:  # no frame yet, or the stream is stopped
            self.clear()
        elif sequence != self._sequence:
            self._sequence = sequence
            self._graph.set_input_spectrum(self._levels, self._frequencies)

    def clear(self) -> None:
        """Stop showing the spectrum, so no stale area is left behind."""
        self._sequence = 0
        self._graph.clear_input_spectrum()
