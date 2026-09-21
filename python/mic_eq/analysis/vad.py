"""Offline VAD helpers shared by Auto-EQ and Auto Voice Setup."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import resample_poly

CALIBRATED_VAD_DEFAULT_THRESHOLD = 0.48
VAD_SPEECH_EVIDENCE_THRESHOLD = 0.40
VAD_STRONG_SPEECH_THRESHOLD = 0.65
VAD_NOISE_CONTAMINATION_THRESHOLD = 0.35
SILERO_WINDOW_SAMPLES = 512
SILERO_SAMPLE_RATE = 16_000
VAD_ANALYSIS_SAMPLE_RATE = 48_000


def map_causal_vad_probabilities(
    probabilities: np.ndarray | None,
    frame_ends: np.ndarray,
    sample_rate: int,
) -> np.ndarray | None:
    """Map model posteriors without making a frame see a future window."""
    if probabilities is None or sample_rate <= 0:
        return None
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    ends = np.asarray(frame_ends, dtype=np.int64).reshape(-1)
    if values.size == 0 or ends.size == 0 or not np.isfinite(values).all():
        return None

    # Keep the model's 32 ms cadence in integer time. Rounding a 44.1 kHz
    # window to 1412 samples accumulates a sample of drift every window.
    completed = (
        ends * SILERO_SAMPLE_RATE // (int(sample_rate) * SILERO_WINDOW_SAMPLES)
    ) - 1
    mapped = np.zeros(ends.size, dtype=np.float32)
    available = completed >= 0
    if np.any(available):
        indices = np.minimum(completed[available], values.size - 1)
        mapped[available] = np.clip(values[indices], 0.0, 1.0)
    return mapped


def analyze_offline_vad(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = CALIBRATED_VAD_DEFAULT_THRESHOLD,
    pre_gain: float = 1.0,
) -> tuple[np.ndarray | None, str]:
    """Return native Silero posteriors, or an explicit analysis backend label.

    The native helper constructs the same stateful model used by the live
    worker. ``pre_gain`` is applied before inference to match the live model's
    input contract. Pure-Python tests and reduced installations may not have
    the extension/model; in that case callers retain their energy analysis and
    expose ``energy_fallback`` in diagnostics instead of pretending Silero ran.
    """
    try:
        from mic_eq import CORE_AVAILABLE, analyze_vad_probabilities
    except ImportError:
        return None, "energy_fallback"

    if not CORE_AVAILABLE or not callable(analyze_vad_probabilities):
        return None, "energy_fallback"

    try:
        gain = float(pre_gain)
    except (TypeError, ValueError):
        return None, "energy_fallback"
    if not np.isfinite(gain) or gain <= 0.0:
        return None, "energy_fallback"
    gain = max(0.1, gain)

    samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if samples.size == 0 or sample_rate <= 0:
        return None, "energy_fallback"
    inference_sample_rate = int(sample_rate)
    if (
        inference_sample_rate * SILERO_WINDOW_SAMPLES
    ) % SILERO_SAMPLE_RATE:
        divisor = int(np.gcd(inference_sample_rate, VAD_ANALYSIS_SAMPLE_RATE))
        samples = np.ascontiguousarray(
            resample_poly(
                samples.astype(np.float64, copy=False),
                VAD_ANALYSIS_SAMPLE_RATE // divisor,
                inference_sample_rate // divisor,
            ),
            dtype=np.float32,
        )
        inference_sample_rate = VAD_ANALYSIS_SAMPLE_RATE
    if gain != 1.0:
        samples = np.ascontiguousarray(samples * gain, dtype=np.float32)

    try:
        raw_probabilities: Any = analyze_vad_probabilities(
            samples,
            inference_sample_rate,
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
    "map_causal_vad_probabilities",
    "analyze_offline_vad",
]
