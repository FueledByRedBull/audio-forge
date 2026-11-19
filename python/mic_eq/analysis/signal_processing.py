"""Analysis signal operations.

Clamping, band means, resampling and peak detection run in the native core;
the Welch density stays in NumPy.
"""

from __future__ import annotations

import numpy as np

from ..mic_eq_core import band_mean, clamp, find_peaks_distance_prominence, resample_poly


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


__all__ = [
    "band_mean",
    "clamp",
    "find_peaks_distance_prominence",
    "resample_poly",
    "welch_hamming_density",
]
