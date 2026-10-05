"""Independent SciPy contracts for the fixed native analysis K-weight filter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter, resample_poly

from mic_eq import mic_eq_core
from mic_eq.analysis import voice_setup


def _scipy_reference(audio: np.ndarray, sample_rate: int = 48_000) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float64)
    if sample_rate != 48_000:
        divisor = int(np.gcd(sample_rate, 48_000))
        signal = resample_poly(signal, 48_000 // divisor, sample_rate // divisor)
    return np.asarray(
        lfilter(
            [1.0, -2.0, 1.0],
            [1.0, -1.99004745483398, 0.99007225036621],
            lfilter(
                [1.53512485958697, -2.69169618940638, 1.19839281085285],
                [1.0, -1.69065929318241, 0.73248077421585],
                signal,
            ),
        ),
        dtype=np.float64,
    )


def _signal_case(name: str) -> np.ndarray:
    if name == "empty":
        return np.asarray([], dtype=np.float64)
    if name == "singleton":
        return np.asarray([0.5])
    if name == "short":
        return np.asarray([0.25, -0.7, 0.01, 0.0, 0.8, -0.001, 0.0])
    if name == "impulse":
        result = np.zeros(48_000)
        result[0] = 1.0
        return result
    if name == "dc":
        return np.full(96_000, 0.3)
    if name == "quiet":
        return np.random.default_rng(4983).normal(0.0, 1e-9, 96_000)
    if name == "long_noise":
        return np.random.default_rng(735).normal(0.0, 0.15, 60 * 48_000)
    if name == "noise":
        return np.random.default_rng(3837).normal(0.0, 0.15, 96_000)
    frequencies = {"low_sine": 4.0, "voice_sine": 197.0, "high_sine": 20_000.0}
    time = np.arange(96_000, dtype=np.float64) / 48_000.0
    return 0.4 * np.sin(2.0 * np.pi * frequencies[name] * time)


def _assert_sample_parity(actual: np.ndarray, reference: np.ndarray) -> float:
    assert actual.dtype == np.float64
    assert actual.shape == reference.shape
    peak = float(np.max(np.abs(reference), initial=0.0))
    np.testing.assert_allclose(
        actual, reference, rtol=0.0, atol=1e-11 * max(1.0, peak)
    )
    return float(np.max(np.abs(actual - reference), initial=0.0))


@pytest.mark.parametrize(
    "case",
    [
        "empty", "singleton", "short", "impulse", "dc", "low_sine",
        "voice_sine", "high_sine", "noise", "quiet", "long_noise",
    ],
)
def test_native_k_weighting_matches_scipy(case: str, record_property):
    source = _signal_case(case)
    original = source.copy()

    actual = mic_eq_core._k_weighted_48k(source)

    error = _assert_sample_parity(actual, _scipy_reference(source))
    record_property("sample_max_abs_error", error)
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("stride", [1, 2, -1, -3])
def test_native_k_weighting_preserves_strided_readonly_input(stride: int):
    source = _signal_case("noise")[::stride]
    source.flags.writeable = False
    original = source.copy()

    actual = mic_eq_core._k_weighted_48k(source)

    _assert_sample_parity(actual, _scipy_reference(source))
    np.testing.assert_array_equal(source, original)


def test_native_k_weighting_starts_from_zero_on_every_call():
    source = _signal_case("short")
    first = mic_eq_core._k_weighted_48k(source)
    mic_eq_core._k_weighted_48k(_signal_case("dc"))

    repeated = mic_eq_core._k_weighted_48k(source)

    np.testing.assert_array_equal(repeated, first)
    _assert_sample_parity(repeated, _scipy_reference(source))


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 32_000, 44_100, 48_000, 96_000])
@pytest.mark.parametrize("length", [0, 1, 257, 32_007])
def test_python_k_weighting_keeps_resampling_contract(
    sample_rate: int, length: int, record_property,
):
    source = np.random.default_rng(384).normal(0.0, 0.2, length)

    actual = voice_setup._k_weighted_48k(source, sample_rate)

    error = _assert_sample_parity(actual, _scipy_reference(source, sample_rate))
    record_property("sample_max_abs_error", error)


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_python_k_weighting_preserves_nonfinite_propagation(sample_rate: int, value: float):
    source = np.asarray([0.25, value, -0.5, 0.0])
    reference = _scipy_reference(source, sample_rate)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = voice_setup._k_weighted_48k(source, sample_rate)

    assert not recorded
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(reference))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(reference))
    finite = np.isfinite(reference)
    _assert_sample_parity(actual[finite], reference[finite])


def _voice_capture(sample_rate: int, duration: float, kind: str) -> np.ndarray:
    time = np.arange(int(sample_rate * duration)) / sample_rate
    if kind == "silence":
        return np.zeros(time.size)
    if kind == "quiet":
        return np.random.default_rng(739).normal(0.0, 1e-8, time.size)
    envelope = np.where(time % 1.3 < 0.19, 0.002, 0.06 + 0.025 * np.sin(time * 4.1))
    return envelope * (
        np.sin(2 * np.pi * 173.0 * time)
        + 0.28 * np.sin(2 * np.pi * 1263.0 * time)
        + 0.07 * np.sin(2 * np.pi * 6301.0 * time)
    )


def _assert_measurement_parity(actual: Any, reference: Any) -> float:
    if isinstance(reference, Mapping):
        assert actual.keys() == reference.keys()
        return max(
            (_assert_measurement_parity(actual[key], reference[key]) for key in reference),
            default=0.0,
        )
    elif isinstance(reference, np.ndarray):
        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-8)
        if reference.dtype == bool:
            return 0.0
        return float(np.max(np.abs(actual - reference), initial=0.0))
    elif isinstance(reference, (float, np.floating)):
        assert actual == pytest.approx(reference, rel=0.0, abs=1e-8)
        return abs(float(actual) - float(reference))
    else:
        assert actual == reference
        return 0.0


@pytest.mark.parametrize(
    ("sample_rate", "duration", "kind"),
    [
        (48_000, 0.55, "voice"), (48_000, 6.4, "voice"),
        (44_100, 6.4, "voice"), (96_000, 6.4, "voice"),
        (48_000, 3.4, "quiet"), (48_000, 3.4, "silence"),
    ],
)
def test_k_weighting_preserves_windows_features_and_recommendation_decisions(
    monkeypatch, sample_rate: int, duration: float, kind: str, record_property,
):
    source = _voice_capture(sample_rate, duration, kind)
    actual = voice_setup._vad_masked_speech_features(source, sample_rate, -75.0)
    with monkeypatch.context() as context:
        context.setattr(voice_setup, "_k_weighted_48k", _scipy_reference)
        reference = voice_setup._vad_masked_speech_features(source, sample_rate, -75.0)
    feature_error = _assert_measurement_parity(actual, reference)
    record_property("feature_max_abs_error", feature_error)

    weighted = voice_setup._k_weighted_48k(source, sample_rate)
    expected_weighted = _scipy_reference(source, sample_rate)
    active = np.arange(weighted.size) % 48_000 >= 11_000
    for window, hop in ((19_200, 4_800), (144_000, 48_000)):
        actual_windows = voice_setup._active_loudness_windows(
            weighted, active, window_samples=window, hop_samples=hop,
        )
        expected_windows = voice_setup._active_loudness_windows(
            expected_weighted, active, window_samples=window, hop_samples=hop,
        )
        np.testing.assert_allclose(actual_windows, expected_windows, rtol=0.0, atol=1e-8)

    for confidence, snr in ((0.54, 20.0), (0.55, 10.0), (0.9, 9.99)):
        def recommend(features):
            return voice_setup._recommend_compressor_settings(
                target_preset="natural", speech_body_db=-25.0,
                speech_loudness_lufs=features["short_term_lufs"],
                loudness_range_db=features["loudness_range_db"], speech_snr_db=snr,
                capture_confidence=confidence, dynamics_intensity="balanced",
                custom_target_p95_db=3.5, custom_peak_cap_db=8.0,
            )

        actual_settings, actual_diagnostics = recommend(actual)
        expected_settings, expected_diagnostics = recommend(reference)
        _assert_measurement_parity(actual_settings, expected_settings)
        _assert_measurement_parity(actual_diagnostics, expected_diagnostics)
