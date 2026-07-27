"""Tests for explicit room-noise reference quality analysis."""

from __future__ import annotations

import numpy as np

from mic_eq.analysis.noise_reference import CaptureMetadata, analyze_noise_reference


def _stationary_noise(
    sample_rate: int,
    seconds: float = 3.0,
    amplitude: float = 0.002,
    seed: int = 4100,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.normal(size=int(sample_rate * seconds))).astype(np.float32)


def _speech_with_quiet_gaps(sample_rate: int, seconds: float = 5.0) -> np.ndarray:
    rng = np.random.default_rng(4101)
    t = np.arange(int(sample_rate * seconds), dtype=float) / sample_rate
    envelope = np.where((t % 1.0) < 0.72, 1.0, 0.015)
    voice = (
        0.08 * np.sin(2.0 * np.pi * 135.0 * t)
        + 0.035 * np.sin(2.0 * np.pi * 270.0 * t)
        + 0.015 * rng.normal(size=t.size)
    )
    return (envelope * voice).astype(np.float32)


def test_stationary_matching_reference_is_usable():
    sample_rate = 48_000
    noise = _stationary_noise(sample_rate)
    speech = _speech_with_quiet_gaps(sample_rate)
    result = analyze_noise_reference(noise, speech, sample_rate)

    assert result.status == "usable"
    assert result.usable is True
    assert result.quality_score >= 0.60
    assert result.conservative_spectrum_db.shape == result.frequencies.shape
    assert result.metrics["in_capture_noise_frame_count"] >= 3


def test_digital_silence_is_invalid_even_when_long_enough():
    sample_rate = 48_000
    result = analyze_noise_reference(
        np.zeros(sample_rate * 3, dtype=np.float32),
        _speech_with_quiet_gaps(sample_rate),
        sample_rate,
    )

    assert result.status == "invalid"
    assert result.usable is False
    assert "suspiciously silent" in " ".join(result.reasons)


def test_nonfinite_or_clipped_reference_is_invalid():
    sample_rate = 48_000
    noise = _stationary_noise(sample_rate)
    noise[100] = np.nan
    noise[200:500] = 1.0
    result = analyze_noise_reference(
        noise,
        _speech_with_quiet_gaps(sample_rate),
        sample_rate,
    )

    assert result.status == "invalid"
    assert any("non-finite" in reason for reason in result.reasons)
    assert any("clipped" in reason for reason in result.reasons)


def test_intermittent_reference_is_rejected_as_nonstationary():
    sample_rate = 48_000
    noise = _stationary_noise(sample_rate, amplitude=0.001)
    noise[sample_rate : sample_rate + sample_rate // 3] += 0.12
    result = analyze_noise_reference(
        noise,
        _speech_with_quiet_gaps(sample_rate),
        sample_rate,
    )

    assert result.status == "invalid"
    assert any(
        "changing events" in reason or "not stationary" in reason
        for reason in result.reasons
    )


def test_speech_contamination_uses_noise_vad_probability():
    sample_rate = 48_000
    noise = _stationary_noise(sample_rate)
    noise_vad = np.full(100, 0.90, dtype=float)
    result = analyze_noise_reference(
        noise,
        _speech_with_quiet_gaps(sample_rate),
        sample_rate,
        noise_vad_probabilities=noise_vad,
    )

    assert result.status == "invalid"
    assert any("speech is present" in reason for reason in result.reasons)


def test_capture_identity_and_age_mismatch_are_invalid():
    sample_rate = 48_000
    noise_meta = CaptureMetadata(
        captured_at_unix_s=100.0,
        input_device="Mic A",
        sample_rate=sample_rate,
        channel_mode="average",
        channel_count=2,
    )
    speech_meta = CaptureMetadata(
        captured_at_unix_s=900.0,
        input_device="Mic B",
        sample_rate=sample_rate,
        channel_mode="left",
        channel_count=1,
    )
    result = analyze_noise_reference(
        _stationary_noise(sample_rate),
        _speech_with_quiet_gaps(sample_rate),
        sample_rate,
        noise_metadata=noise_meta,
        speech_metadata=speech_meta,
    )

    assert result.status == "invalid"
    combined = " ".join(result.reasons)
    assert "input device changed" in combined
    assert "input channel mode changed" in combined
    assert "stale" in combined


def test_mismatched_in_capture_noise_uses_conservative_spectrum():
    sample_rate = 48_000
    explicit = _stationary_noise(sample_rate, amplitude=0.0005)
    speech = _speech_with_quiet_gaps(sample_rate)
    rng = np.random.default_rng(4102)
    speech += (0.012 * rng.normal(size=speech.size)).astype(np.float32)
    result = analyze_noise_reference(explicit, speech, sample_rate)

    assert result.status in {"questionable", "invalid"}
    assert result.in_capture_spectrum_db is not None
    assert np.all(
        result.conservative_spectrum_db
        >= result.explicit_spectrum_db - 1e-9
    )
    assert result.conservative_noise_rms_db > float(result.metrics["noise_rms_db"])


def test_short_reference_returns_actionable_guidance():
    sample_rate = 48_000
    result = analyze_noise_reference(
        _stationary_noise(sample_rate, seconds=0.5),
        None,
        sample_rate,
    )

    assert result.status == "invalid"
    assert result.guidance
    assert "at least" in result.guidance[0]
