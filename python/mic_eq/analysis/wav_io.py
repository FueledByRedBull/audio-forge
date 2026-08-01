"""Strict WAV conversion helpers shared by offline evaluation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


def pcm_to_float_mono(
    raw: np.ndarray,
    *,
    label: str = "audio",
    allow_stereo: bool = True,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert finite mono/stereo PCM to centered, normalized floating mono.

    Integer conversion deliberately happens before channel averaging. Averaging
    integer stereo first changes the dtype to floating point and can otherwise
    bypass full-scale normalization entirely.
    """

    audio = np.asarray(raw)
    if audio.ndim not in (1, 2):
        raise ValueError(f"{label} must contain mono or interleaved PCM audio")
    if audio.ndim == 2 and (not allow_stereo or audio.shape[1] == 0):
        expected = "mono" if not allow_stereo else "non-empty interleaved"
        raise ValueError(f"{label} must contain {expected} PCM audio")

    if np.issubdtype(audio.dtype, np.floating):
        converted = audio.astype(np.float64)
    elif np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(int(info.min)), int(info.max)))
        converted = audio.astype(np.float64) / scale
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = float(int(info.max) + 1) / 2.0
        converted = (audio.astype(np.float64) - midpoint) / midpoint
    else:
        raise ValueError(f"{label} uses unsupported WAV sample type {audio.dtype}")

    if converted.ndim == 2:
        converted = np.mean(converted, axis=1)
    if converted.size == 0 or not np.all(np.isfinite(converted)):
        raise ValueError(f"{label} must contain finite audio")
    return np.ascontiguousarray(converted, dtype=dtype)


def read_mono_wav(
    path: Path,
    *,
    allow_stereo: bool = True,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> tuple[int, np.ndarray]:
    """Read a WAV and apply :func:`pcm_to_float_mono` without silent repair."""

    sample_rate, raw = wavfile.read(path)
    return int(sample_rate), pcm_to_float_mono(
        np.asarray(raw),
        label=path.name,
        allow_stereo=allow_stereo,
        dtype=dtype,
    )
