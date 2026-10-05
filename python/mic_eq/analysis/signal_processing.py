"""Small NumPy equivalents for the analysis signal operations used here."""

from __future__ import annotations

import math
import operator

import numpy as np


def resample_poly(samples: np.ndarray, up: int, down: int) -> np.ndarray:
    """Zero-phase, constant-padded Kaiser resampling for 1-D audio.

    This implements the SciPy defaults used by the application: a 10-lobe
    Kaiser-windowed sinc and centered output. It intentionally omits custom
    filters, axes, and nonconstant padding.
    """
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError("audio to resample must be one-dimensional")
    try:
        up, down = operator.index(up), operator.index(down)
    except TypeError as exc:
        raise ValueError("up and down must be positive integers") from exc
    if up < 1 or down < 1:
        raise ValueError("up and down must be positive integers")
    divisor = math.gcd(up, down)
    up //= divisor
    down //= divisor
    if up == down == 1:
        return values.copy()
    if values.size == 0:
        dtype = values.dtype if values.dtype.kind in "fc" else np.dtype(np.float64)
        return np.empty(0, dtype=dtype)

    dtype = values.dtype if values.dtype in (
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ) else np.dtype(np.float64)
    max_rate = max(up, down)
    half_length = 10 * max_rate
    positions = np.arange(2 * half_length + 1, dtype=np.float64) - half_length
    cutoff = 1.0 / max_rate
    taps = cutoff * np.sinc(cutoff * positions) * np.kaiser(positions.size, 5.0)
    taps /= taps.sum()
    taps = (taps * up).astype(dtype, copy=False)
    pad_before = down - half_length % down
    trim_before = (half_length + pad_before) // down
    taps = np.pad(taps, (pad_before, 0))

    output_size = (values.size * up + down - 1) // down
    terms = (taps.size + up - 1) // up
    output = np.empty(output_size, dtype=np.result_type(values.dtype, taps.dtype))
    padded = np.pad(values, (terms - 1, terms - 1))
    windows = np.lib.stride_tricks.sliding_window_view(padded, terms)
    for residue in range(min(up, output_size)):
        group_size = (output_size - 1 - residue) // up + 1
        first_high_rate_position = (residue + trim_before) * down
        phase = first_high_rate_position % up
        first_center = first_high_rate_position // up
        phase_taps = taps[phase::up]
        rows = windows[first_center::down][:group_size, -phase_taps.size :]
        output[residue::up] = rows @ phase_taps[::-1]
    return output


def welch_hamming_density(
    samples: np.ndarray, sample_rate: float, nperseg: int, noverlap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1-D, constant-detrended Hamming Welch density used by EQ."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("audio spectrum input must be one-dimensional")
    if sample_rate <= 0 or nperseg < 1 or values.size < nperseg:
        raise ValueError("sample rate and segment size must fit the input")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")

    frames = np.lib.stride_tricks.sliding_window_view(values, nperseg)[
        :: nperseg - noverlap
    ]
    frames = frames - frames.mean(axis=1, keepdims=True)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(nperseg) / nperseg)
    scale = sample_rate * np.sum(window * window)
    psd = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 / scale
    if nperseg % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    return np.fft.rfftfreq(nperseg, 1.0 / sample_rate), psd.mean(axis=0)


def find_peaks_distance_prominence(
    samples: np.ndarray, *, distance: int, prominence: float
) -> np.ndarray:
    """Find 1-D local maxima with SciPy's distance and prominence semantics."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("peak detection input must be one-dimensional")
    if distance < 1 or prominence < 0:
        raise ValueError("distance must be positive and prominence nonnegative")

    candidates: list[int] = []
    index = 1
    while index < values.size - 1:
        if values[index - 1] < values[index]:
            plateau_end = index
            while plateau_end + 1 < values.size and values[plateau_end + 1] == values[index]:
                plateau_end += 1
            if plateau_end < values.size - 1 and values[plateau_end] > values[plateau_end + 1]:
                candidates.append((index + plateau_end) // 2)
            index = plateau_end + 1
        else:
            index += 1

    peaks = np.asarray(candidates, dtype=np.intp)
    if peaks.size == 0:
        return peaks

    keep = np.ones(peaks.size, dtype=bool)
    priority_order = np.argsort(values[peaks])[::-1]
    for order_index, candidate_index in enumerate(priority_order):
        if not keep[candidate_index]:
            continue
        for higher_index in priority_order[:order_index]:
            if keep[higher_index] and abs(
                int(peaks[candidate_index]) - int(peaks[higher_index])
            ) < distance:
                keep[candidate_index] = False
                break
    peaks = peaks[keep]

    prominent: list[int] = []
    for peak in peaks:
        height = values[peak]
        left = int(peak)
        right = int(peak)
        while left > 0 and values[left - 1] <= height:
            left -= 1
        while right + 1 < values.size and values[right + 1] <= height:
            right += 1
        base = max(np.min(values[left : peak + 1]), np.min(values[peak : right + 1]))
        if height - base >= prominence:
            prominent.append(int(peak))
    return np.asarray(prominent, dtype=np.intp)
