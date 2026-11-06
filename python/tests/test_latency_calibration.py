"""Tests for latency calibration analysis utilities."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import latency_calibration_reference as reference


LATENCY_PATH = (
    Path(__file__).parent.parent / "mic_eq" / "analysis" / "latency_calibration.py"
)
spec = importlib.util.spec_from_file_location(
    "mic_eq.analysis.latency_calibration", LATENCY_PATH
)
assert spec is not None and spec.loader is not None
lat = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lat
spec.loader.exec_module(lat)


def test_generate_probe_signal_shape_and_peak():
    probe = lat.generate_probe_signal(sample_rate=48000, duration_ms=80.0)
    assert probe.ndim == 1
    assert len(probe) == int(48000 * 0.08)
    assert np.max(np.abs(probe)) <= 0.8001
    assert np.max(np.abs(probe)) > 0.2


def test_analyze_latency_recovers_known_offset():
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    offset_samples = int(0.123 * sample_rate)  # 123 ms
    rec_len = offset_samples + len(probe) + 2048

    rng = np.random.default_rng(1337)
    recording = rng.normal(0.0, 0.01, rec_len).astype(np.float32)
    recording[offset_samples : offset_samples + len(probe)] += probe

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
    )

    expected_ms = (offset_samples * 1000.0) / sample_rate
    assert result.success
    assert abs(result.measured_round_trip_ms - expected_ms) < 2.0
    assert result.confidence >= 0.25


def test_analyze_latency_low_confidence_for_noise_only():
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    rng = np.random.default_rng(2026)
    recording = rng.normal(0.0, 0.01, sample_rate).astype(np.float32)

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
    )

    assert not result.success


def test_analyze_latency_wrong_expected_window_does_not_false_pass():
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    playback_delay_samples = int(0.55 * sample_rate)
    actual_latency_samples = int(0.18 * sample_rate)
    offset_samples = playback_delay_samples + actual_latency_samples
    rec_len = offset_samples + len(probe) + 4096

    rng = np.random.default_rng(7331)
    recording = rng.normal(0.0, 0.02, rec_len).astype(np.float32)
    recording[offset_samples : offset_samples + len(probe)] += probe * 0.5

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
        expected_playback_start_ms=100.0,
        expected_playback_jitter_ms=25.0,
        expected_latency_min_ms=20.0,
        expected_latency_max_ms=150.0,
    )

    assert not result.success


@pytest.mark.parametrize(
    ("noise", "gain", "round_trip_ms", "max_search_ms", "latency_max_ms", "tolerance_ms"),
    [
        (0.025, 0.35, 190.0, 500.0, 450.0, 2.0),
        (0.02, 0.18, 140.0, 1000.0, 250.0, 5.0),
    ],
    ids=["noisy_probe", "quiet_probe"],
)
def test_analyze_latency_expected_window_handles_attenuated_probe(
    noise, gain, round_trip_ms, max_search_ms, latency_max_ms, tolerance_ms
):
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    playback_delay_ms = 550.0
    offset_samples = int(((playback_delay_ms + round_trip_ms) / 1000.0) * sample_rate)
    rec_len = offset_samples + len(probe) + 4096

    rng = np.random.default_rng(4242)
    recording = rng.normal(0.0, noise, rec_len).astype(np.float32)
    recording[offset_samples : offset_samples + len(probe)] += probe * gain

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=max_search_ms,
        expected_playback_start_ms=playback_delay_ms,
        expected_playback_jitter_ms=120.0,
        expected_latency_min_ms=20.0,
        expected_latency_max_ms=latency_max_ms,
    )

    assert result.success
    assert result.confidence >= 0.25
    assert abs(result.measured_round_trip_ms - round_trip_ms) < tolerance_ms


def test_analyze_latency_repeated_probe_selects_direct_path_with_echo_and_noise():
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    direct_offset = int(0.145 * sample_rate)
    echo_offset = direct_offset + int(0.055 * sample_rate)
    rec_len = echo_offset + len(probe) + 4096

    rng = np.random.default_rng(9090)
    recording = rng.normal(0.0, 0.012, rec_len).astype(np.float32)
    recording[direct_offset : direct_offset + len(probe)] += probe * 0.35
    recording[echo_offset : echo_offset + len(probe)] += probe * 0.18

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
    )

    expected_ms = (direct_offset * 1000.0) / sample_rate
    assert result.success
    assert result.repetition_count >= 3
    assert result.agreement_ms < 1.0
    assert abs(result.measured_round_trip_ms - expected_ms) < 2.0


def test_analyze_latency_recovers_direct_path_before_louder_echo():
    sample_rate = 48_000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    playback_start_samples = int(0.550 * sample_rate)
    direct_route_samples = int(0.145 * sample_rate)
    direct_start = playback_start_samples + direct_route_samples
    echo_start = direct_start + int(0.055 * sample_rate)
    recording = np.zeros(echo_start + len(probe) + 4096, dtype=np.float32)
    rng = np.random.default_rng(12012)
    recording += rng.normal(0.0, 0.005, recording.size).astype(np.float32)
    recording[direct_start : direct_start + len(probe)] += probe
    recording[echo_start : echo_start + len(probe)] += probe * 2.0

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        expected_playback_start_ms=550.0,
        expected_playback_jitter_ms=50.0,
    )

    expected_ms = (direct_route_samples * 1000.0) / sample_rate
    assert result.success
    assert abs(result.route_latency_ms - expected_ms) < 2.0


def test_analyze_latency_rejects_weak_direct_path_before_louder_echo_across_noise_seeds():
    sample_rate = 48_000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    playback_start_samples = int(0.550 * sample_rate)
    direct_start = playback_start_samples + int(0.145 * sample_rate)
    echo_start = direct_start + int(0.055 * sample_rate)
    recording_length = echo_start + len(probe) + 4096
    unresolved_results: list[tuple[int, bool, str]] = []

    for seed in range(12):
        recording = np.zeros(recording_length, dtype=np.float32)
        rng = np.random.default_rng(seed)
        recording += rng.normal(0.0, 0.01, recording_length).astype(np.float32)
        recording[direct_start : direct_start + len(probe)] += probe * 0.005
        recording[echo_start : echo_start + len(probe)] += probe * 2.0

        result = lat.analyze_latency(
            reference_probe=probe,
            recorded_signal=recording,
            sample_rate=sample_rate,
            expected_playback_start_ms=550.0,
            expected_playback_jitter_ms=50.0,
        )
        if result.success or "ambigu" not in result.message.lower():
            unresolved_results.append((seed, result.success, result.message))

    assert not unresolved_results, (
        f"weak earlier paths were not rejected as ambiguous: {unresolved_results}"
    )


def test_analyze_latency_rejects_earlier_path_with_a_later_echo_too():
    sample_rate = 48_000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    playback_start_samples = int(0.550 * sample_rate)
    direct_start = playback_start_samples + int(0.145 * sample_rate)
    dominant_echo_start = direct_start + int(0.055 * sample_rate)
    later_echo_start = dominant_echo_start + int(0.055 * sample_rate)
    recording_length = later_echo_start + len(probe) + 4096
    recording = np.random.default_rng(0).normal(0.0, 0.01, recording_length).astype(
        np.float32
    )
    recording[direct_start : direct_start + len(probe)] += probe * 0.005
    recording[dominant_echo_start : dominant_echo_start + len(probe)] += probe * 2.0
    recording[later_echo_start : later_echo_start + len(probe)] += probe * 0.05

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        expected_playback_start_ms=550.0,
        expected_playback_jitter_ms=50.0,
    )

    assert not result.success
    assert "ambigu" in result.message.lower()


def test_analyze_latency_accepts_exact_custom_fir_probe_without_noise():
    sample_rate = 48_000
    base_probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)
    probe = np.convolve(base_probe, [0.65, 0.35], mode="full")[:-1].astype(
        np.float32
    )

    playback_start_samples = int(0.550 * sample_rate)
    route_latency_samples = int(0.145 * sample_rate)
    probe_start = playback_start_samples + route_latency_samples
    recording = np.zeros(probe_start + len(probe) + 1024, dtype=np.float32)
    recording[probe_start : probe_start + len(probe)] = probe

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        expected_playback_start_ms=550.0,
        expected_playback_jitter_ms=50.0,
    )

    expected_ms = route_latency_samples * 1000.0 / sample_rate
    assert result.success
    assert abs(result.route_latency_ms - expected_ms) < 2.0


def test_analyze_latency_subsample_refinement_beats_integer_peak():
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    fractional_offset = 0.1234 * sample_rate
    base_offset = int(np.floor(fractional_offset))
    frac = fractional_offset - base_offset
    rec_len = base_offset + len(probe) + 4096

    rng = np.random.default_rng(5150)
    recording = rng.normal(0.0, 0.004, rec_len).astype(np.float32)
    recording[base_offset : base_offset + len(probe)] += probe * (1.0 - frac)
    recording[base_offset + 1 : base_offset + 1 + len(probe)] += probe * frac

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
    )

    integer_error = abs(float(result.peak_sample_offset) - fractional_offset)
    refined_error = abs(result.sub_sample_offset - fractional_offset)
    assert result.success
    assert refined_error < integer_error
    assert refined_error < 0.8


def test_analyze_latency_reports_ambiguity_for_strong_echo():
    sample_rate = 48000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)

    direct_offset = int(0.145 * sample_rate)
    echo_offset = direct_offset + int(0.055 * sample_rate)
    rec_len = echo_offset + len(probe) + 4096

    rng = np.random.default_rng(9091)
    recording = rng.normal(0.0, 0.012, rec_len).astype(np.float32)
    recording[direct_offset : direct_offset + len(probe)] += probe * 0.35
    recording[echo_offset : echo_offset + len(probe)] += probe * 0.27

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
    )

    assert not result.success
    assert "ambigu" in result.message.lower()
    assert result.ambiguity_score >= 0.9


def test_result_to_profile_has_expected_fields():
    result = lat.LatencyCalibrationResult(
        success=True,
        measured_round_trip_ms=42.0,
        estimated_one_way_ms=21.0,
        applied_compensation_ms=21.0,
        confidence=0.88,
        peak_sample_offset=2016,
        message="ok",
    )

    profile = lat.result_to_profile(
        result,
        sample_rate=48000,
        engine_latency_ms=12.5,
        engine_config_signature='{"limiter_enabled":true}',
    )
    assert profile["measured_round_trip_ms"] == 42.0
    assert profile["estimated_one_way_ms"] == 21.0
    assert profile["applied_compensation_ms"] == 21.0
    assert profile["confidence"] == 0.88
    assert profile["sample_rate"] == 48000
    assert profile["engine_latency_ms"] == 12.5
    assert profile["total_latency_ms"] == 54.5
    assert profile["engine_config_signature"] == '{"limiter_enabled":true}'
    assert "timestamp_utc" in profile


def test_latency_calibration_applies_measured_route_without_halving():
    sample_rate = 48_000
    probe = lat.generate_probe_signal(sample_rate=sample_rate, duration_ms=80.0)
    route_latency_samples = int(0.143 * sample_rate)
    recording = np.zeros(route_latency_samples + len(probe) + 2048, dtype=np.float32)
    recording[route_latency_samples : route_latency_samples + len(probe)] = probe

    result = lat.analyze_latency(
        reference_probe=probe,
        recorded_signal=recording,
        sample_rate=sample_rate,
        min_search_ms=5.0,
        max_search_ms=500.0,
    )

    expected_ms = route_latency_samples * 1000.0 / sample_rate
    assert result.success
    assert abs(result.route_latency_ms - expected_ms) < 2.0
    assert abs(result.applied_compensation_ms - expected_ms) < 2.0
    assert result.directional_latency_ms is None
    assert result.estimated_one_way_ms == 0.0

    profile = lat.result_to_profile(result, sample_rate=sample_rate)
    assert abs(profile["route_latency_ms"] - expected_ms) < 2.0
    assert profile["compensation_basis"] == "measured_output_to_input_route"
    assert profile["directional_latency_ms"] is None


# Parity with the NumPy analysis the native core replaced (latency_calibration_reference).
# Correlations go through realfft natively and pocketfft in NumPy, so floats
# derived from them agree to rounding only (at most 2.3e-15 on these cases on
# Windows/AVX2; 1e-12 leaves room for other CPUs' FFT kernels). The measured
# lag, the decision, the message and the counts must match exactly.
_FLOAT_TOLERANCE = 1e-12


def _probe_cases() -> list[tuple[int, float, float]]:
    return [
        (48_000, 80.0, 0.8), (44_100, 80.0, 0.8), (16_000, 80.0, 0.8), (96_000, 80.0, 0.8),
        (48_000, 5.0, 0.8), (48_000, 0.01, 0.8), (48_000, -10.0, 0.8), (48_000, 80.0, 0.0),
        (48_000, 80.0, -1.0), (48_000, 250.0, 1.0), (8_000, 30.0, 0.5),
    ]


@pytest.mark.parametrize("case", _probe_cases(), ids=lambda case: "-".join(map(str, case)))
def test_native_probe_matches_the_numpy_implementation(case):
    expected = reference.generate_probe_signal(*case)
    actual = lat.generate_probe_signal(*case)

    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _place(recording: np.ndarray, probe: np.ndarray, start: int, gain: float) -> None:
    recording[start : start + probe.size] += probe * gain


def _noisy(seed: int, sigma: float, length: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, length).astype(np.float32)


def _latency_cases() -> dict[str, tuple[Any, Any, int, dict[str, Any]]]:
    rate = 48_000
    probe = reference.generate_probe_signal(rate)
    cases: dict[str, tuple[Any, Any, int, dict[str, Any]]] = {}

    offset = int(0.123 * rate)
    recording = _noisy(1337, 0.01, offset + probe.size + 2048)
    _place(recording, probe, offset, 1.0)
    cases["known-offset"] = (probe, recording, rate, {})
    cases["noise-only"] = (probe, _noisy(2026, 0.01, rate), rate, {})

    offset = int(0.55 * rate) + int(0.18 * rate)
    recording = _noisy(7331, 0.02, offset + probe.size + 4096)
    _place(recording, probe, offset, 0.5)
    cases["wrong-window"] = (probe, recording, rate, {
        "expected_playback_start_ms": 100.0, "expected_playback_jitter_ms": 25.0,
        "expected_latency_min_ms": 20.0, "expected_latency_max_ms": 150.0,
    })
    for name, noise, gain, route_ms, max_ms, latency_max in (
        ("noisy-probe", 0.025, 0.35, 190.0, 500.0, 450.0),
        ("quiet-probe", 0.02, 0.18, 140.0, 1000.0, 250.0),
    ):
        offset = int(((550.0 + route_ms) / 1000.0) * rate)
        recording = _noisy(4242, noise, offset + probe.size + 4096)
        _place(recording, probe, offset, gain)
        cases[name] = (probe, recording, rate, {
            "max_search_ms": max_ms, "expected_playback_start_ms": 550.0,
            "expected_playback_jitter_ms": 120.0, "expected_latency_min_ms": 20.0,
            "expected_latency_max_ms": latency_max,
        })
    for name, seed, echo_gain in (("echo-and-noise", 9090, 0.18), ("strong-echo", 9091, 0.27)):
        direct = int(0.145 * rate)
        echo = direct + int(0.055 * rate)
        recording = _noisy(seed, 0.012, echo + probe.size + 4096)
        _place(recording, probe, direct, 0.35)
        _place(recording, probe, echo, echo_gain)
        cases[name] = (probe, recording, rate, {})

    window = {"expected_playback_start_ms": 550.0, "expected_playback_jitter_ms": 50.0}
    direct = int(0.550 * rate) + int(0.145 * rate)
    echo = direct + int(0.055 * rate)
    recording = _noisy(12012, 0.005, echo + probe.size + 4096)
    _place(recording, probe, direct, 1.0)
    _place(recording, probe, echo, 2.0)
    cases["louder-echo"] = (probe, recording, rate, window)
    for seed in range(12):
        recording = _noisy(seed, 0.01, echo + probe.size + 4096)
        _place(recording, probe, direct, 0.005)
        _place(recording, probe, echo, 2.0)
        cases[f"weak-direct-{seed}"] = (probe, recording, rate, window)
    later = echo + int(0.055 * rate)
    recording = _noisy(0, 0.01, later + probe.size + 4096)
    _place(recording, probe, direct, 0.005)
    _place(recording, probe, echo, 2.0)
    _place(recording, probe, later, 0.05)
    cases["earlier-and-later-echo"] = (probe, recording, rate, window)

    fir_probe = np.convolve(probe, [0.65, 0.35], mode="full")[:-1].astype(np.float32)
    recording = np.zeros(direct + fir_probe.size + 1024, dtype=np.float32)
    recording[direct : direct + fir_probe.size] = fir_probe
    cases["custom-fir-probe"] = (fir_probe, recording, rate, window)

    fractional = 0.1234 * rate
    base = int(np.floor(fractional))
    fraction = fractional - base
    recording = _noisy(5150, 0.004, base + probe.size + 4096)
    _place(recording, probe, base, 1.0 - fraction)
    _place(recording, probe, base + 1, fraction)
    cases["sub-sample"] = (probe, recording, rate, {})

    route = int(0.143 * rate)
    clean = np.zeros(route + probe.size + 2048, dtype=np.float32)
    clean[route : route + probe.size] = probe
    cases["clean-route"] = (probe, clean, rate, {})
    cases["clean-route-float64"] = (probe.astype(np.float64), clean.astype(np.float64), rate, {})
    cases["full-scale-clipped"] = (probe, np.clip(clean * 4.0, -1.0, 1.0), rate, {})
    cases["silence"] = (probe, np.zeros(rate, dtype=np.float32), rate, {})
    with_nan = clean.copy()
    with_nan[100] = np.nan
    cases["nan"] = (probe, with_nan, rate, {})
    with_inf = clean.copy()
    with_inf[100] = np.inf
    cases["inf"] = (probe, with_inf, rate, {})

    rate_441 = 44_100
    probe_441 = reference.generate_probe_signal(rate_441)
    route = int(0.087 * rate_441)
    recording = _noisy(441, 0.004, route + probe_441.size + 3000)
    _place(recording, probe_441, route, 0.6)
    cases["44k1"] = (probe_441, recording, rate_441, {})
    cases["44k1-window"] = (probe_441, recording, rate_441, {
        "expected_playback_start_ms": 20.0, "expected_playback_jitter_ms": 15.0,
    })
    cases["negative-min-search"] = (probe, clean, rate, {"min_search_ms": -400.0})
    cases["window-outside-range"] = (
        probe, clean, rate, {"min_search_ms": 300.0, "max_search_ms": 300.0}
    )
    cases["window-past-recording"] = (
        probe, clean, rate, {"min_search_ms": 2000.0, "max_search_ms": 3000.0}
    )
    cases["short-recording"] = (probe, clean[: probe.size - 1], rate, {})
    cases["short-probe"] = (probe[:15], clean, rate, {})
    cases["missing-probe"] = (None, clean, rate, {})
    cases["missing-recording"] = (probe, None, rate, {})
    cases["zero-rate"] = (probe, clean, 0, {})
    cases["unsupported-route"] = (probe, clean, rate, {"route_kind": "input_to_output"})
    cases["route-spelling"] = (probe, clean, rate, {"route_kind": "  Output_To_Input "})
    cases["empty-route"] = (probe, clean, rate, {"route_kind": ""})
    return cases


_LATENCY_CASES = _latency_cases()


@pytest.mark.parametrize("name", list(_LATENCY_CASES))
def test_native_latency_matches_the_numpy_implementation(name):
    probe, recording, sample_rate, kwargs = _LATENCY_CASES[name]

    with np.errstate(invalid="ignore", over="ignore"):
        expected = asdict(reference.analyze_latency(probe, recording, sample_rate, **kwargs))
    actual = asdict(lat.analyze_latency(probe, recording, sample_rate, **kwargs))

    assert actual.keys() == expected.keys()
    for field, value in expected.items():
        if isinstance(value, float):
            assert actual[field] == pytest.approx(
                value, rel=_FLOAT_TOLERANCE, abs=_FLOAT_TOLERANCE, nan_ok=True
            ), field
        else:
            assert actual[field] == value, field
