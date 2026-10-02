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


_ESTOI_RATE = 10_000
_ESTOI_FRAME = 256
_ESTOI_FFT = 512
_ESTOI_SEGMENT = 30


def _estoi_frames(signal: np.ndarray) -> np.ndarray:
    hop = _ESTOI_FRAME // 2
    count = 1 + (signal.size - _ESTOI_FRAME) // hop
    index = np.arange(_ESTOI_FRAME)[None, :] + hop * np.arange(count)[:, None]
    return signal[index] * np.hanning(_ESTOI_FRAME + 2)[1:-1][None, :]


def _overlap_add(frames: np.ndarray) -> np.ndarray:
    hop = _ESTOI_FRAME // 2
    out = np.zeros((frames.shape[0] - 1) * hop + _ESTOI_FRAME)
    for index, frame in enumerate(frames):
        out[index * hop : index * hop + _ESTOI_FRAME] += frame
    return out


def _third_octave_bands() -> np.ndarray:
    freqs = np.linspace(0, _ESTOI_RATE, _ESTOI_FFT + 1)[: _ESTOI_FFT // 2 + 1]
    centers = np.arange(15)
    low = 150.0 * 2.0 ** ((2 * centers - 1) / 6)
    high = 150.0 * 2.0 ** ((2 * centers + 1) / 6)
    bands = np.zeros((15, freqs.size))
    for band in range(15):
        start = int(np.argmin(np.square(freqs - low[band])))
        stop = int(np.argmin(np.square(freqs - high[band])))
        bands[band, start:stop] = 1.0
    return bands


def estoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    """Extended STOI (Jensen & Taal, 2016) with the pystoi constants.

    Resampling to 10 kHz uses scipy's polyphase filter rather than pystoi's,
    so absolute values can differ slightly; paired differences stay valid.
    """
    reference = resample_audio(reference, sample_rate, _ESTOI_RATE, dtype=np.float64)
    estimate = resample_audio(estimate, sample_rate, _ESTOI_RATE, dtype=np.float64)
    ref_frames = _estoi_frames(reference)
    est_frames = _estoi_frames(estimate)
    energy = 20.0 * np.log10(np.linalg.norm(ref_frames, axis=1) + 1e-20)
    keep = energy > np.max(energy) - 40.0
    reference = _overlap_add(ref_frames[keep])
    estimate = _overlap_add(est_frames[keep])
    bands = _third_octave_bands()

    def band_envelopes(signal: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(_estoi_frames(signal), n=_ESTOI_FFT, axis=1).T
        return np.sqrt(bands @ np.square(np.abs(spectrum)))

    ref_bands = band_envelopes(reference)
    est_bands = band_envelopes(estimate)
    segments = ref_bands.shape[1] - _ESTOI_SEGMENT + 1
    if segments < 1:
        return float("nan")

    def normalized(envelopes: np.ndarray) -> np.ndarray:
        seg = np.stack([envelopes[:, m : m + _ESTOI_SEGMENT] for m in range(segments)])
        seg = seg - seg.mean(axis=2, keepdims=True)
        seg = seg / (np.linalg.norm(seg, axis=2, keepdims=True) + 1e-20)
        seg = seg - seg.mean(axis=1, keepdims=True)
        return seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-20)

    return float(
        np.sum(normalized(ref_bands) * normalized(est_bands))
        / (_ESTOI_SEGMENT * segments)
    )


def amplitude_ratio_db(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-15) / max(denominator, 1e-15))


def percentile_or_zero(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def percentile_or_none(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None
