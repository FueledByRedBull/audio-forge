"""Strict WAV conversion helpers shared by offline evaluation tools."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT_PCM = 1
_FORMAT_FLOAT = 3
_FORMAT_EXTENSIBLE = 0xFFFE
# Bytes 4..16 of the KSDATAFORMAT_SUBTYPE GUID {xxxxxxxx-0000-0010-8000-00AA00389B71}.
_SUBTYPE_GUID_TAIL = bytes.fromhex("00001000800000aa00389b71")


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a RIFF WAV as native samples: (frames,) or (frames, channels).

    The standard library's ``wave`` cannot read float WAVs or 24-bit samples.
    8-bit PCM is unsigned and 24-bit PCM is returned left-justified in int32.

    Read: little-endian RIFF/WAVE with PCM (8, 16, 24 or 32 bits), IEEE float (32 or
    64 bits) or WAVE_FORMAT_EXTENSIBLE with the standard KSDATAFORMAT_SUBTYPE GUID of
    one of those two, any channel count. Chunks before or after the data chunk are
    skipped. nAvgBytesPerSec, the extensible channel mask and valid-bit count are
    not checked.

    Rejected, never repaired: RIFX, RF64 and BW64; a data chunk longer than the rest
    of the file (a streamed 0xFFFFFFFF size too) or before the format chunk; a short
    format chunk; a sample size that is not a whole number of bytes or that differs
    from nBlockAlign; a data length that is not a whole number of frames; PCM wider
    than 32 bits; compressed formats; an extensible chunk with another sub-format
    GUID tail.

    The array may be a read-only view of the bytes read.
    """

    with open(path, "rb") as stream:
        preamble = stream.read(12)
        if preamble[:4] != b"RIFF" or preamble[8:] != b"WAVE":
            raise ValueError(f"{path.name} is not a RIFF WAV file")
        header = b""
        while True:
            chunk = stream.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{path.name} has no data chunk after its format chunk")
            chunk_id, size = struct.unpack("<4sI", chunk)
            if chunk_id == b"fmt ":
                header = stream.read(size)
            elif chunk_id == b"data":
                if not header:
                    raise ValueError(f"{path.name} has no format chunk before its data")
                payload = stream.read(size)
                if len(payload) != size:
                    raise ValueError(f"{path.name} data chunk is truncated")
                break
            else:
                stream.seek(size, 1)
            if size & 1:
                stream.seek(1, 1)

    if len(header) < 16:
        raise ValueError(f"{path.name} has a truncated format chunk")
    tag, channels, rate, _, block, bits = struct.unpack("<HHIIHH", header[:16])
    if tag == _FORMAT_EXTENSIBLE:
        if len(header) < 40 or struct.unpack("<H", header[16:18])[0] < 22:
            raise ValueError(f"{path.name} has a truncated extensible format chunk")
        if header[28:40] != _SUBTYPE_GUID_TAIL:
            raise ValueError(f"{path.name} has a non-standard extensible sub-format GUID")
        (tag,) = struct.unpack("<I", header[24:28])
    frame_bytes = channels * (bits // 8)
    if channels < 1 or bits % 8 or block != frame_bytes or len(payload) % max(frame_bytes, 1):
        raise ValueError(f"{path.name} has an inconsistent format or data length")

    if tag == _FORMAT_FLOAT and bits in (32, 64):
        audio = np.frombuffer(payload, f"<f{bits // 8}")
    elif tag == _FORMAT_PCM and bits == 8:
        audio = np.frombuffer(payload, np.uint8)
    elif tag == _FORMAT_PCM and bits in (16, 32):
        audio = np.frombuffer(payload, f"<i{bits // 8}")
    elif tag == _FORMAT_PCM and bits == 24:
        padded = np.zeros((len(payload) // 3, 4), np.uint8)
        padded[:, 1:] = np.frombuffer(payload, np.uint8).reshape(-1, 3)
        audio = padded.view("<i4")[:, 0]
    else:
        raise ValueError(f"{path.name} uses unsupported WAV format {tag} with {bits} bits")
    audio = audio.astype(audio.dtype.newbyteorder("="), copy=False)
    return int(rate), audio.reshape(-1, channels) if channels > 1 else audio


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

    sample_rate, raw = _read_wav(path)
    return int(sample_rate), pcm_to_float_mono(
        np.asarray(raw),
        label=path.name,
        allow_stereo=allow_stereo,
        dtype=dtype,
    )
