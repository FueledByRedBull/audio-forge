"""EQ response prediction for Auto-EQ optimization."""

from __future__ import annotations

import numpy as np

from .constants import SAMPLE_RATE

FILTER_PEAK = "peak"
FILTER_LOW_SHELF = "low_shelf"
FILTER_HIGH_SHELF = "high_shelf"


def _default_filter_types(num_bands: int) -> list[str]:
    if num_bands <= 0:
        return []
    return [
        FILTER_LOW_SHELF if index == 0 else FILTER_HIGH_SHELF if index == num_bands - 1 else FILTER_PEAK
        for index in range(num_bands)
    ]


def _biquad_coefficients(
    gain_db: float,
    q: float,
    fc: float,
    filter_type: str,
) -> tuple[float, float, float, float, float, float]:
    """Return RBJ biquad coefficients with un-normalized a0."""
    amplitude = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / SAMPLE_RATE
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    if filter_type == FILTER_LOW_SHELF:
        two_sqrt_a_alpha = 2.0 * np.sqrt(amplitude) * alpha
        b0 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cos_w0)
        b2 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (amplitude + 1.0) + (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cos_w0)
        a2 = (amplitude + 1.0) + (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha
    elif filter_type == FILTER_HIGH_SHELF:
        two_sqrt_a_alpha = 2.0 * np.sqrt(amplitude) * alpha
        b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_w0)
        b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_w0)
        a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha
    else:
        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / amplitude
    return b0, b1, b2, a0, a1, a2


def _predict_eq_response(
    freqs,
    gains,
    qs,
    center_freqs,
    filter_types: list[str] | tuple[str, ...] | None = None,
):
    """
    Predict how EQ settings affect frequency response.

    Simulates the combined effect of the full 10-band EQ, matching the live DSP
    topology with shelves on the first and last bands and peaking filters in between.
    """
    gains_arr = np.asarray(gains, dtype=float)
    qs_arr = np.asarray(qs, dtype=float)
    centers_arr = np.asarray(center_freqs, dtype=float)
    if not (gains_arr.size == qs_arr.size == centers_arr.size):
        raise ValueError("gain, Q, and center frequency arrays must have the same length")
    if filter_types is None:
        filter_types = _default_filter_types(gains_arr.size)
    if len(filter_types) != gains_arr.size:
        raise ValueError("filter_types length must match gains")

    response_linear = np.ones_like(freqs, dtype=float)
    w = 2 * np.pi * freqs / SAMPLE_RATE
    z_inv = np.exp(-1j * w)
    z_inv_2 = z_inv * z_inv

    for gain_db, q, fc, filter_type in zip(gains_arr, qs_arr, centers_arr, filter_types):
        if abs(gain_db) < 0.01:
            continue
        b0, b1, b2, a0, a1, a2 = _biquad_coefficients(
            float(gain_db),
            float(q),
            float(fc),
            str(filter_type),
        )
        numerator = b0 + b1 * z_inv + b2 * z_inv_2
        denominator = a0 + a1 * z_inv + a2 * z_inv_2
        response_linear *= np.abs(numerator / denominator)

    return 20 * np.log10(np.maximum(response_linear, 1e-12))
