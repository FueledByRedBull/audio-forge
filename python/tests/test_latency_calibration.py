"""Tests for latency calibration analysis utilities."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


LATENCY_PATH = Path(__file__).parent.parent / "mic_eq" / "analysis" / "latency_calibration.py"
spec = importlib.util.spec_from_file_location("mic_eq.analysis.latency_calibration", LATENCY_PATH)
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

    profile = lat.result_to_profile(result, sample_rate=48000)
    assert profile["measured_round_trip_ms"] == 42.0
    assert profile["estimated_one_way_ms"] == 21.0
    assert profile["applied_compensation_ms"] == 21.0
    assert profile["confidence"] == 0.88
    assert profile["sample_rate"] == 48000
    assert "timestamp_utc" in profile
