"""Offline VAD helpers shared by Auto-EQ and Auto Voice Setup."""

from __future__ import annotations

from typing import Any

import numpy as np

CALIBRATED_VAD_DEFAULT_THRESHOLD = 0.48
VAD_SPEECH_EVIDENCE_THRESHOLD = 0.40
VAD_STRONG_SPEECH_THRESHOLD = 0.65
VAD_NOISE_CONTAMINATION_THRESHOLD = 0.35


def analyze_offline_vad(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = CALIBRATED_VAD_DEFAULT_THRESHOLD,
) -> tuple[np.ndarray | None, str]:
    """Return native Silero posteriors, or an explicit analysis backend label.

    The native helper constructs the same stateful model used by the live
    worker. Pure-Python tests and reduced installations may not have the
    extension/model; in that case callers retain their energy analysis and
    expose ``energy_fallback`` in diagnostics instead of pretending Silero ran.
    """
    try:
        from mic_eq import CORE_AVAILABLE, analyze_vad_probabilities
    except ImportError:
        return None, "energy_fallback"

    if not CORE_AVAILABLE or not callable(analyze_vad_probabilities):
        return None, "energy_fallback"

    samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if samples.size == 0 or sample_rate <= 0:
        return None, "energy_fallback"

    try:
        raw_probabilities: Any = analyze_vad_probabilities(
            samples,
            int(sample_rate),
            float(threshold),
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        return None, "energy_fallback"

    probabilities = np.asarray(raw_probabilities, dtype=float).reshape(-1)
    if probabilities.size == 0 or not np.isfinite(probabilities).all():
        return None, "energy_fallback"
    return np.clip(probabilities, 0.0, 1.0), "silero"


__all__ = [
    "CALIBRATED_VAD_DEFAULT_THRESHOLD",
    "VAD_NOISE_CONTAMINATION_THRESHOLD",
    "VAD_SPEECH_EVIDENCE_THRESHOLD",
    "VAD_STRONG_SPEECH_THRESHOLD",
    "analyze_offline_vad",
]
