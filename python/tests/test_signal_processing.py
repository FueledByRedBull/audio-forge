"""Numerical parity checks for the NumPy analysis primitives."""

import numpy as np
import pytest
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import resample_poly as scipy_resample_poly
from scipy.signal import welch as scipy_welch

from mic_eq.analysis.signal_processing import (
    find_peaks_distance_prominence,
    resample_poly,
    welch_hamming_density,
)
from mic_eq.analysis.spectrum import compute_voice_spectrum


@pytest.mark.parametrize("length", [0, 1, 2, 5, 31, 1024, 4410])
@pytest.mark.parametrize("rates", [(1, 3), (3, 2), (160, 147), (147, 160)])
def test_resample_poly_matches_scipy_length_alignment_and_short_inputs(length, rates):
    rng = np.random.default_rng(1937 + length)
    samples = rng.standard_normal(length)

    actual = resample_poly(samples, *rates)
    expected = scipy_resample_poly(samples, *rates)

    assert actual.shape == expected.shape
    assert len(actual) == (length * rates[0] + rates[1] - 1) // rates[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resample_poly_preserves_audio_precision_and_matches_scipy(dtype):
    samples = np.random.default_rng(412).standard_normal(2048).astype(dtype)

    actual = resample_poly(samples, 160, 147)
    expected = scipy_resample_poly(samples, 160, 147)

    assert actual.dtype == expected.dtype
    tolerance = 2e-6 if dtype is np.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_resample_poly_preserves_dc_and_rejects_above_nyquist_energy():
    dc = resample_poly(np.ones(48_000), 1, 3)
    np.testing.assert_allclose(dc[100:-100], 1.0, rtol=0, atol=1e-12)

    sample = np.arange(48_000)
    out_of_band = np.sin(2 * np.pi * 0.42 * sample)
    filtered = resample_poly(out_of_band, 1, 4)
    assert np.sqrt(np.mean(filtered[100:-100] ** 2)) < 1e-4


def test_resample_poly_rejects_invalid_rates_and_non_vector_input():
    with pytest.raises(ValueError, match="positive integers"):
        resample_poly(np.ones(8), 0, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        resample_poly(np.ones((2, 8)), 1, 2)


def test_resample_poly_reduced_identity_returns_an_unchanged_copy():
    samples = np.arange(12, dtype=np.float32)

    actual = resample_poly(samples, 2, 2)

    np.testing.assert_array_equal(actual, samples)
    assert actual.dtype == samples.dtype
    assert not np.shares_memory(actual, samples)


@pytest.mark.parametrize("length", [4096, 4097, 8192, 10_003])
def test_hamming_welch_matches_scipy_density(length):
    samples = np.random.default_rng(length).standard_normal(length)

    freqs, actual = welch_hamming_density(samples, 48_000, 4096, 2048)
    expected_freqs, expected = scipy_welch(
        samples,
        fs=48_000,
        window="hamming",
        nperseg=4096,
        noverlap=2048,
        detrend="constant",
    )

    np.testing.assert_array_equal(freqs, expected_freqs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)


def test_voice_spectrum_and_short_input_validation_match_scipy_reference():
    fs = 48_000
    nperseg = 1024
    samples = np.random.default_rng(123).standard_normal(8192)
    freqs, spectrum_db = compute_voice_spectrum(samples, fs=fs, nperseg=nperseg)
    expected_freqs, expected_psd = scipy_welch(
        samples,
        fs=fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )

    np.testing.assert_array_equal(freqs, expected_freqs)
    np.testing.assert_allclose(
        spectrum_db,
        10 * np.log10(expected_psd + 1e-12),
        rtol=0,
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="Audio too short for FFT"):
        compute_voice_spectrum(np.zeros(nperseg - 1), fs=fs, nperseg=nperseg)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([0, 1, 0, 2, 2, 2, 0, 1, 0, 3, 0.0]),
        np.array([0, 3, 2, 3, 0.0]),
        np.random.default_rng(772).standard_normal(257),
    ],
)
@pytest.mark.parametrize("distance", [1, 2, 3, 8])
def test_peak_detection_matches_scipy_distance_and_prominence(samples, distance):
    actual = find_peaks_distance_prominence(
        samples, distance=distance, prominence=0.75
    )
    expected, _ = scipy_find_peaks(samples, distance=distance, prominence=0.75)

    np.testing.assert_array_equal(actual, expected)
