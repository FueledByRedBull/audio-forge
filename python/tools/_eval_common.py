"""Shared numerical primitives for offline evaluation, with explicit metric floors."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.signal import resample_poly


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
    *,
    dtype: type[np.float32] | type[np.float64],
) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=dtype)
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(audio, target_rate // divisor, source_rate // divisor),
        dtype=dtype,
    )


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    energy = float(np.dot(reference, reference))
    if energy <= 1e-12:
        return 0.0
    target = reference * (float(np.dot(estimate, reference)) / energy)
    residual = estimate - target
    return float(
        10.0
        * np.log10(
            (float(np.dot(target, target)) + 1e-12)
            / (float(np.dot(residual, residual)) + 1e-12)
        )
    )


def amplitude_ratio_db(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-15) / max(denominator, 1e-15))


def percentile_or_zero(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def percentile_or_none(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None
