"""Numerical parity checks for the analysis primitives."""

import math
import operator
from typing import Any

import numpy as np
import pytest
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import resample_poly as scipy_resample_poly
from scipy.signal import welch as scipy_welch

from mic_eq.analysis.signal_processing import (
    band_mean,
    clamp,
    find_peaks_distance_prominence,
    resample_poly,
    welch_hamming_density,
)
from mic_eq.analysis.spectrum import compute_voice_spectrum
from mic_eq.analysis.vad import analyze_offline_vad, map_causal_vad_probabilities
from mic_eq.mic_eq_core import analyze_vad_probabilities


# Empty input returns before the rates are used, so one rate pair covers it.
@pytest.mark.parametrize(
    ("length", "rates"),
    [
        pytest.param(length, rates, id=f"rates{index}-{length}")
        for index, rates in enumerate([(1, 3), (3, 2), (160, 147), (147, 160)])
        for length in (0, 1, 2, 5, 31, 1024, 4410)
        if length or index == 0
    ],
)
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


_PEAK_SAMPLES = (
    np.array([0, 1, 0, 2, 2, 2, 0, 1, 0, 3, 0.0]),
    np.array([0, 3, 2, 3, 0.0]),
    np.random.default_rng(772).standard_normal(257),
)


# Local maxima are at least 2 samples apart, so distances 1 and 2 select the same peaks: distance 2
# is the stricter boundary (two peaks exactly 2 apart both stay) and distance 1 only has to be
# accepted. The second sample has two peaks 2 apart, so every distance from 3 up drops the same one.
@pytest.mark.parametrize(
    ("samples", "distance"),
    [
        pytest.param(_PEAK_SAMPLES[index], distance, id=f"{distance}-samples{index}")
        for distance, index in [
            (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (8, 0), (8, 2),
        ]
    ],
)
def test_peak_detection_matches_scipy_distance_and_prominence(samples, distance):
    actual = find_peaks_distance_prominence(
        samples, distance=distance, prominence=0.75
    )
    expected, _ = scipy_find_peaks(samples, distance=distance, prominence=0.75)

    np.testing.assert_array_equal(actual, expected)


def test_clamp_and_band_mean_are_the_helpers_every_analysis_module_shares():
    from mic_eq.analysis import compressor_calibration, joint_tuning, voice_setup
    from mic_eq.analysis.auto_eq_parts import target

    assert clamp(5.0, 0.0, 3.0) == 3.0 and clamp(-1.0, 0.0, 3.0) == 0.0 and clamp(2.0, 0.0, 3.0) == 2.0
    freqs = np.array([100.0, 200.0, 400.0, 800.0])
    values = np.array([1.0, 2.0, 4.0, 8.0])
    assert band_mean(freqs, values, 150.0, 500.0) == 3.0
    assert band_mean(freqs, values, 1000.0, 2000.0) == 3.75  # no bin inside: the whole spectrum
    for module, name, shared in (
        (compressor_calibration, "_clamp", clamp),
        (joint_tuning, "_clamp", clamp),
        (voice_setup, "_clamp", clamp),
        (voice_setup, "_band_mean", band_mean),
        (target, "_band_mean", band_mean),
    ):
        assert getattr(module, name) is shared


def test_failure_detection_logs_through_the_shared_debug_switch(monkeypatch, capsys):
    from mic_eq.analysis import failure_detection
    from mic_eq.analysis.auto_eq_parts import constants

    assert failure_detection.debug_log is constants.debug_log
    monkeypatch.setattr(constants, "DEBUG", True)
    failure_detection.debug_log("[VALIDATION] shared")
    assert capsys.readouterr().out == "[VALIDATION] shared\n"


# Parity with the NumPy implementations the native core replaced. The
# _reference_* functions are that code, kept verbatim so the native versions
# are compared on the running platform rather than against recorded output.


def _reference_clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _reference_band_mean(freqs: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float(np.mean(values))
    return float(np.mean(values[mask]))


def _reference_resample_poly(samples: np.ndarray, up: int, down: int) -> np.ndarray:
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


def _reference_find_peaks(samples: np.ndarray, *, distance: int, prominence: float) -> np.ndarray:
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


def _reference_map_causal_vad_probabilities(
    probabilities: np.ndarray | None, positions: np.ndarray, sample_rate: int
) -> np.ndarray | None:
    if probabilities is None or sample_rate <= 0:
        return None
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    ends = np.asarray(positions, dtype=np.int64).reshape(-1)
    if values.size == 0 or ends.size == 0 or not np.isfinite(values).all():
        return None
    completed = (ends * 16_000 // (int(sample_rate) * 512)) - 1
    mapped = np.zeros(ends.size, dtype=np.float32)
    available = completed >= 0
    if np.any(available):
        indices = np.minimum(completed[available], values.size - 1)
        mapped[available] = np.clip(values[indices], 0.0, 1.0)
    return mapped


def _reference_analyze_offline_vad(
    audio: np.ndarray, sample_rate: int, *, threshold: float = 0.48, pre_gain: float = 1.0
) -> tuple[np.ndarray | None, str]:
    try:
        gain = float(pre_gain)
    except (TypeError, ValueError):
        return None, "energy_fallback"
    if not np.isfinite(gain) or gain <= 0.0:
        return None, "energy_fallback"
    gain = max(0.1, gain)

    samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if samples.size == 0 or sample_rate <= 0:
        return None, "energy_fallback"
    inference_sample_rate = int(sample_rate)
    if (inference_sample_rate * 512) % 16_000:
        divisor = int(np.gcd(inference_sample_rate, 48_000))
        samples = np.ascontiguousarray(
            _reference_resample_poly(
                samples.astype(np.float64, copy=False),
                48_000 // divisor,
                inference_sample_rate // divisor,
            ),
            dtype=np.float32,
        )
        inference_sample_rate = 48_000
    if gain != 1.0:
        samples = np.ascontiguousarray(samples * gain, dtype=np.float32)

    try:
        raw_probabilities = analyze_vad_probabilities(
            samples, inference_sample_rate, float(threshold)
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        return None, "energy_fallback"

    probabilities = np.asarray(raw_probabilities, dtype=float).reshape(-1)
    if probabilities.size == 0 or not np.isfinite(probabilities).all():
        return None, "energy_fallback"
    return np.clip(probabilities, 0.0, 1.0), "silero"


def _taps_use_platform_libm(up: int, down: int) -> bool:
    """Whether NumPy's sin and exp equal the C library's on this rate pair's taps.

    The native taps call the platform libm, as ``math`` does. NumPy dispatches
    its own AVX-512 sin/exp on some CPUs, which can differ by an ulp; only then
    is a tolerance allowed.
    """
    divisor = math.gcd(up, down)
    max_rate = max(up, down) // divisor
    half_length = 10 * max_rate
    count = 2 * half_length + 1
    x = np.pi * ((1.0 / max_rate) * (np.arange(count, dtype=np.float64) - half_length))
    x = np.where(x, x, np.finfo(np.float64).eps)
    alpha = (count - 1) / 2.0
    window_arguments = np.append(
        5.0 * np.sqrt(1 - ((np.arange(count) - alpha) / alpha) ** 2.0), 5.0
    )
    return np.array_equal(np.sin(x), [math.sin(value) for value in x]) and np.array_equal(
        np.exp(window_arguments), [math.exp(value) for value in window_arguments]
    )


def _assert_same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


_PARITY_RATES = [(1, 3), (3, 2), (160, 147), (147, 160), (2, 2), (48_000, 44_100), (1, 2), (5, 1)]


def _resample_parity_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261106)
    inputs: dict[str, np.ndarray] = {"empty": np.zeros(0)}
    for length in (1, 2, 5, 31, 1024, 4410):
        inputs[f"noise-{length}"] = rng.standard_normal(length)
    inputs["silence"] = np.zeros(4800)
    inputs["full-scale"] = np.where(np.arange(4800) % 96 < 48, 1.0, -1.0)
    inputs["nan"] = rng.standard_normal(1200)
    inputs["nan"][100] = np.nan
    inputs["inf"] = rng.standard_normal(1200)
    inputs["inf"][[100, 700]] = [np.inf, -np.inf]
    inputs["float32"] = rng.standard_normal(2048).astype(np.float32)
    inputs["float32-full-scale"] = inputs["full-scale"].astype(np.float32)
    inputs["int16"] = (rng.standard_normal(1000) * 8000).astype(np.int16)
    inputs["strided"] = rng.standard_normal(3000)[::3]
    return inputs


_RESAMPLE_PARITY_INPUTS = _resample_parity_inputs()


@pytest.mark.parametrize("rates", _PARITY_RATES, ids=lambda rates: f"{rates[0]}-{rates[1]}")
@pytest.mark.parametrize("name", list(_RESAMPLE_PARITY_INPUTS))
def test_native_resample_poly_matches_the_numpy_implementation(name, rates):
    samples = _RESAMPLE_PARITY_INPUTS[name]

    with np.errstate(invalid="ignore"):
        expected = _reference_resample_poly(samples, *rates)
    actual = resample_poly(samples, *rates)

    if rates[0] == rates[1] and samples.dtype == np.int16:
        # NumPy copied integer audio unchanged; the native identity copy is float64.
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
        return
    if _taps_use_platform_libm(*rates):
        _assert_same_bits(actual, expected)
        return
    # An ulp in a tap moves a float64 output by about 1e-16 of full scale and
    # can flip a float32 rounding.
    assert actual.dtype == expected.dtype
    finite = expected[np.isfinite(expected)]
    scale = max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0
    tolerance = (1e-6 if expected.dtype == np.float32 else 1e-12) * scale
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def _peak_parity_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1106)
    inputs = {
        "empty": np.zeros(0),
        "single": np.array([1.0]),
        "triangle": np.array([0.0, 1.0, 0.0]),
        "plateau": np.array([0.0, 2.0, 2.0, 2.0, 0.0]),
        "edge-plateau": np.array([0.0, 1.0, 1.0]),
        "scipy-0": _PEAK_SAMPLES[0],
        "scipy-1": _PEAK_SAMPLES[1],
        "noise-257": _PEAK_SAMPLES[2],
        "noise-4096": rng.standard_normal(4096) * 6.0,
        "spectrum": np.convolve(rng.standard_normal(4096) * 20.0, np.ones(9) / 9, mode="same"),
        "full-scale": np.where(np.arange(4800) % 96 < 48, 1.0, -1.0),
        "silence": np.zeros(512),
    }
    with_nan = rng.standard_normal(600) * 4.0
    with_nan[[3, 50, 51, 300]] = np.nan
    inputs["nan"] = with_nan
    with_inf = rng.standard_normal(600) * 4.0
    with_inf[[10, 200]] = [np.inf, -np.inf]
    inputs["inf"] = with_inf
    return inputs


_PEAK_PARITY_INPUTS = _peak_parity_inputs()


# Distances stay below the full-scale input's 96-sample period: its peaks tie,
# and NumPy's argsort orders large tied inputs by its SIMD sort.
@pytest.mark.parametrize("prominence", [0.0, 0.75, 3.0])
@pytest.mark.parametrize("distance", [1, 2, 3, 8, 50])
@pytest.mark.parametrize("name", list(_PEAK_PARITY_INPUTS))
def test_native_peak_detection_matches_the_numpy_implementation(name, distance, prominence):
    samples = _PEAK_PARITY_INPUTS[name]

    expected = _reference_find_peaks(samples, distance=distance, prominence=prominence)
    actual = find_peaks_distance_prominence(samples, distance=distance, prominence=prominence)

    _assert_same_bits(actual, expected)


def test_native_band_mean_and_clamp_match_the_numpy_implementation():
    rng = np.random.default_rng(611)
    freqs = np.fft.rfftfreq(4096, 1.0 / 48_000)
    spectra = {
        "noise": rng.standard_normal(freqs.size) * 12.0 - 40.0,
        "nan": np.where(rng.random(freqs.size) < 0.01, np.nan, rng.standard_normal(freqs.size)),
        "silence": np.full(freqs.size, -120.0),
    }
    bands = [
        (180.0, 800.0), (1200.0, 3500.0), (5500.0, 8500.0), (2500.0, 4500.0),
        (5000.0, 9000.0), (30_000.0, 40_000.0), (0.0, 24_000.0), (100.0, 100.0),
    ]
    for spectrum in spectra.values():
        for low, high in bands:
            expected = _reference_band_mean(freqs, spectrum, low, high)
            assert repr(band_mean(freqs, spectrum, low, high)) == repr(expected)

    edge_values = [-math.inf, -5.0, -0.0, 0.0, 0.5, 3.0, 5.0, math.inf, math.nan, 2, -7]
    ranges = [(0.0, 3.0), (3.0, 0.0), (-math.inf, math.inf), (0.0, 0.0), (-0.0, 0.0), (1, 2)]
    for value in edge_values:
        for low, high in ranges:
            assert repr(clamp(value, low, high)) == repr(_reference_clamp(value, low, high))


def test_native_causal_vad_mapping_matches_the_numpy_implementation():
    rng = np.random.default_rng(3)
    probabilities = {
        "float32": rng.random(700).astype(np.float32),
        "float64": rng.random(700),
        "out-of-range": rng.standard_normal(50) * 2.0,
        "single": np.array([0.4]),
        "nan": np.array([0.2, np.nan]),
        "inf": np.array([0.2, np.inf]),
        "empty": np.zeros(0),
        "matrix": rng.random((10, 7)),
    }
    positions = {
        "frame-ends": np.arange(1, 2_001, dtype=np.int64) * 441,
        "block-starts": np.arange(0, 2_000, dtype=np.int64) * 480,
        "negative-and-huge": np.array([-10_000, -1, 0, 1, 10**12], dtype=np.int64),
        "float": np.array([511.9, 512.0, 1536.7]),
        "matrix": np.arange(24, dtype=np.int64).reshape(4, 6) * 1000,
        "empty": np.zeros(0, dtype=np.int64),
    }
    for sample_rate in (8_000, 16_000, 22_050, 44_100, 48_000, 0, -1):
        for values in probabilities.values():
            for ends in positions.values():
                expected = _reference_map_causal_vad_probabilities(values, ends, sample_rate)
                actual = map_causal_vad_probabilities(values, ends, sample_rate)
                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None
                    _assert_same_bits(actual, expected)
    assert map_causal_vad_probabilities(None, positions["frame-ends"], 48_000) is None


def _offline_vad_parity_inputs() -> list[tuple[str, np.ndarray, int]]:
    rng = np.random.default_rng(48)
    time = np.arange(48_000) / 48_000
    voice_like = np.sin(2 * np.pi * 180.0 * time) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * time)) * 0.3
    with_nan = rng.standard_normal(4410).astype(np.float32) * 0.1
    with_nan[20] = np.nan
    with_inf = rng.standard_normal(4800).astype(np.float32) * 0.1
    with_inf[20] = np.inf
    full_scale = np.where(np.arange(48_000) % 96 < 48, 1.0, -1.0).astype(np.float32)
    return [
        ("empty", np.zeros(0, dtype=np.float32), 48_000),
        ("short", rng.standard_normal(100).astype(np.float32) * 0.1, 48_000),
        ("silence-48k", np.zeros(48_000, dtype=np.float32), 48_000),
        ("silence-44k1", np.zeros(44_100, dtype=np.float32), 44_100),
        ("full-scale-48k", full_scale, 48_000),
        ("voice-48k", voice_like.astype(np.float32), 48_000),
        ("voice-44k1", voice_like[:44_100], 44_100),
        ("noise-16k", rng.standard_normal(16_000).astype(np.float32) * 0.05, 16_000),
        ("nan-44k1", with_nan, 44_100),
        ("inf-48k", with_inf, 48_000),
        ("unsupported-rate", rng.standard_normal(4000).astype(np.float32), 1_000),
        ("zero-rate", np.ones(100, dtype=np.float32), 0),
    ]


# Without a provisioned model both sides fall back, which the backend check covers.
@pytest.mark.parametrize(
    ("threshold", "pre_gain"),
    [(0.48, 1.0), (0.48, 2.0), (0.3, 0.05), (0.48, 0.0), (0.48, math.nan), (1.5, 1.0)],
)
@pytest.mark.parametrize("case", _offline_vad_parity_inputs(), ids=lambda case: case[0])
def test_native_offline_vad_matches_the_numpy_implementation(case, threshold, pre_gain):
    _name, audio, sample_rate = case

    expected, expected_backend = _reference_analyze_offline_vad(
        audio, sample_rate, threshold=threshold, pre_gain=pre_gain
    )
    actual, backend = analyze_offline_vad(
        audio, sample_rate, threshold=threshold, pre_gain=pre_gain
    )

    assert backend == expected_backend
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    if sample_rate == 44_100 and not _taps_use_platform_libm(48_000, 44_100):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    else:
        _assert_same_bits(actual, expected)


def _offline_vad_argument_cases() -> list[tuple[str, Any, Any, dict[str, Any]]]:
    voice = (np.sin(2 * np.pi * 180.0 * np.arange(9_600) / 48_000) * 0.3).astype(np.float32)
    return [
        ("gain-none", voice, 48_000, {"pre_gain": None}),
        ("gain-text", voice, 48_000, {"pre_gain": "x"}),
        ("gain-numeric-text", voice, 48_000, {"pre_gain": "2.0"}),
        ("gain-list", voice, 48_000, {"pre_gain": [1.0, 2.0]}),
        ("gain-object", voice, 48_000, {"pre_gain": object()}),
        ("gain-bool", voice, 48_000, {"pre_gain": True}),
        ("gain-none-text-audio", "abc", 48_000, {"pre_gain": None}),
        ("gain-zero-text-audio", "abc", 48_000, {"pre_gain": 0.0}),
        ("threshold-none", voice, 48_000, {"threshold": None}),
        ("threshold-text", voice, 48_000, {"threshold": "x"}),
        ("threshold-numeric-text", voice, 48_000, {"threshold": "0.3"}),
        ("threshold-none-empty", np.zeros(0, dtype=np.float32), 48_000, {"threshold": None}),
        ("threshold-none-zero-rate", voice, 0, {"threshold": None}),
        ("text-audio", "abc", 48_000, {}),
        ("object-audio", [object()], 48_000, {}),
        ("none-audio", None, 48_000, {}),
        ("float-rate", voice, 48_000.0, {}),
        ("float-rate-44k1", voice[:8_820], 44_100.0, {}),
        ("fractional-rate", voice, 16_000.7, {}),
        ("numpy-rate", voice, np.int64(16_000), {}),
        ("none-rate-empty", np.zeros(0, dtype=np.float32), None, {}),
        ("none-rate", voice, None, {}),
        ("text-rate", voice, "48000", {}),
    ]


def _offline_vad_outcome(function: Any, audio: Any, sample_rate: Any, kwargs: dict[str, Any]):
    try:
        return function(audio, sample_rate, **kwargs)
    except Exception as error:  # noqa: BLE001 - the exception type is the outcome
        return type(error)


# The removed Python fell back for gains and thresholds float() rejects and
# checked the gain before converting the audio.
@pytest.mark.parametrize("case", _offline_vad_argument_cases(), ids=lambda case: case[0])
def test_native_offline_vad_handles_arguments_like_the_numpy_implementation(case):
    _name, audio, sample_rate, kwargs = case

    expected = _offline_vad_outcome(_reference_analyze_offline_vad, audio, sample_rate, kwargs)
    actual = _offline_vad_outcome(analyze_offline_vad, audio, sample_rate, kwargs)

    if isinstance(expected, type):
        assert actual is expected
        return
    assert isinstance(actual, tuple)
    assert actual[1] == expected[1]
    if expected[0] is None:
        assert actual[0] is None
    else:
        _assert_same_bits(actual[0], expected[0])
