"""Numerical contracts shared by offline evaluation tools."""

from __future__ import annotations

import numpy as np
import pytest

from _eval_common import (
    amplitude_ratio_db,
    estoi,
    percentile_or_none,
    percentile_or_zero,
    resample_audio,
    si_sdr,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resampling_retains_impulse_response_and_requested_precision(dtype):
    impulse = np.asarray([0.0, 1.0, 0.0], dtype=dtype)
    resampled = resample_audio(impulse, 16_000, 48_000, dtype=dtype)

    assert resampled.dtype == dtype
    np.testing.assert_allclose(
        resampled,
        [
            0.0, 0.4096566454, 0.8254431680, 1.0006061736, 0.8254431680,
            0.4096566454, 0.0, -0.1987897163, -0.1554842756,
        ],
        atol=1e-7,
    )
    same_rate = resample_audio(impulse, 48_000, 48_000, dtype=dtype)
    np.testing.assert_array_equal(same_rate, impulse)
    assert same_rate.dtype == dtype
    assert resample_audio(np.empty(0), 48_000, 48_000, dtype=dtype).size == 0


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(np.float32, 16.566899787259494), (np.float64, 16.56689860909591)],
)
def test_si_sdr_preserves_input_precision_and_degenerate_reference(dtype, expected):
    reference = np.asarray([1, -2, 0.5, 3, -0.3, -1], dtype=dtype)
    estimate = np.asarray([1.2, -1.7, 0.3, 2.4, -0.5, -1.1], dtype=dtype)

    tolerance = 1e-5 if dtype == np.float32 else 1e-10
    assert si_sdr(reference, estimate) == pytest.approx(expected, rel=0, abs=tolerance)
    assert si_sdr(np.ones(6, dtype=dtype), estimate) == 0.0
    assert si_sdr(reference, np.zeros(6, dtype=dtype)) == 0.0
    assert si_sdr(reference, reference) > 120.0


def test_metric_floors_and_empty_percentiles_remain_distinct():
    assert amplitude_ratio_db(0.0, 1.0) == -300.0
    assert amplitude_ratio_db(0.0, 0.0) == 0.0
    assert amplitude_ratio_db(2.0, 1.0) == pytest.approx(6.020599913279624)
    assert percentile_or_none([], 95) is None
    assert percentile_or_zero([], 95) == 0.0
    assert percentile_or_none([1.0, 3.0], 25) == 1.5
    assert percentile_or_zero([1, 3], 25) == 1.5


def test_estoi_is_one_for_identity_and_falls_with_added_noise():
    rng = np.random.default_rng(0)
    time = np.arange(48_000 * 3) / 48_000
    # Broadband carrier with a syllable-rate envelope, like speech.
    clean = rng.standard_normal(time.size) * np.maximum(np.sin(2 * np.pi * 4 * time), 0.0) ** 2
    scores = [
        estoi(clean, clean + level * rng.standard_normal(clean.size), 48_000)
        for level in (0.0, 0.1, 0.3, 1.0)
    ]
    assert scores[0] == pytest.approx(1.0)
    assert scores == sorted(scores, reverse=True)
    assert estoi(clean, 0.5 * clean, 48_000) == pytest.approx(1.0)
