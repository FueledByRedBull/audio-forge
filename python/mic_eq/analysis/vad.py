"""Offline VAD helpers shared by Auto-EQ and Auto Voice Setup.

``analyze_offline_vad`` and ``map_causal_vad_probabilities`` run in the native
core; these thresholds are what callers compare its posteriors against.
"""

from __future__ import annotations

from ..mic_eq_core import analyze_offline_vad, map_causal_vad_probabilities

VAD_SPEECH_EVIDENCE_THRESHOLD = 0.40
VAD_STRONG_SPEECH_THRESHOLD = 0.65
VAD_NOISE_CONTAMINATION_THRESHOLD = 0.35


__all__ = [
    "VAD_NOISE_CONTAMINATION_THRESHOLD",
    "VAD_SPEECH_EVIDENCE_THRESHOLD",
    "VAD_STRONG_SPEECH_THRESHOLD",
    "map_causal_vad_probabilities",
    "analyze_offline_vad",
]
