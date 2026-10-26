"""Tests for explicit room-noise reference quality analysis."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eq.analysis.noise_reference import (
    SILENT_REFERENCE_GUIDANCE,
    SILENT_REFERENCE_REASON,
    CaptureMetadata,
    analyze_noise_reference,
    capture_is_silent,
)


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


def test_silent_reference_is_weak_but_does_not_block_setup():
    # A headset or driver that mutes itself between words records true silence.
    sample_rate = 48_000
    for silence in (
        np.zeros(sample_rate * 3, dtype=np.float32),
        _stationary_noise(sample_rate, amplitude=10.0 ** (-100.0 / 20.0)),
    ):
        result = analyze_noise_reference(silence, None, sample_rate)

        assert result.status == "questionable"
        assert result.usable is True
        assert result.reasons == [SILENT_REFERENCE_REASON]
        assert result.guidance == [SILENT_REFERENCE_GUIDANCE]


def test_capture_is_silent_below_minus_95_dbfs_whatever_the_shape_of_the_data():
    def constant(dbfs: float) -> np.ndarray:
        return np.full(1_000, 10.0 ** (dbfs / 20.0))

    assert capture_is_silent(constant(-95.5))
    assert not capture_is_silent(constant(-94.5))
    assert capture_is_silent(np.zeros(1_000))
    # Samples that are not numbers carry no signal; a number among them still does.
    assert capture_is_silent(np.full(1_000, np.nan))
    assert capture_is_silent(np.full(1_000, np.inf))
    assert capture_is_silent(np.empty(0))
    signal = _stationary_noise(48_000, seconds=1.0)
    signal[5] = np.nan
    assert not capture_is_silent(signal)
    # Channels are read as one run of samples.
    assert capture_is_silent(np.zeros((1_000, 2)))
    assert not capture_is_silent(np.column_stack([np.zeros(48_000), _stationary_noise(48_000, seconds=1.0)]))


@pytest.mark.parametrize(
    "capture",
    [np.empty(0), np.zeros(1), np.zeros(24_000), np.full(144_000, np.nan)],
    ids=["empty", "one-sample", "too-short", "all-nan"],
)
def test_a_capture_that_must_be_retaken_is_not_told_that_silence_is_normal(capture):
    capture = capture.astype(np.float32)
    # These read as silent too, but a retake is the only way on.
    assert capture_is_silent(capture)

    result = analyze_noise_reference(capture, None, 48_000)

    assert result.status == "invalid"
    assert result.must_recapture is True
    assert SILENT_REFERENCE_REASON not in result.reasons
    assert SILENT_REFERENCE_GUIDANCE not in result.guidance
    assert result.guidance


def test_silent_reference_takes_its_noise_from_pauses_in_the_voice_capture():
    sample_rate = 48_000
    speech = _speech_with_quiet_gaps(sample_rate)
    measured = analyze_noise_reference(_stationary_noise(sample_rate), speech, sample_rate)
    result = analyze_noise_reference(
        np.zeros(sample_rate * 3, dtype=np.float32), speech, sample_rate
    )

    assert result.status == "questionable"
    assert result.reasons == [SILENT_REFERENCE_REASON]
    assert result.metrics["in_capture_noise_frame_count"] >= 3
    # Nothing to disagree with, so the pauses are the whole estimate.
    assert result.metrics["in_capture_level_delta_db"] is None
    assert result.metrics["spectral_shape_distance_db"] is None
    assert result.in_capture_spectrum_db is not None
    np.testing.assert_array_equal(result.conservative_spectrum_db, result.in_capture_spectrum_db)
    np.testing.assert_array_equal(result.in_capture_spectrum_db, measured.in_capture_spectrum_db)
    assert result.conservative_noise_rms_db > -95.0


def test_only_captures_the_analysis_cannot_use_must_be_recaptured():
    sample_rate = 48_000
    # A gate that opens now and then is unsteady on every take; retaking cannot fix it.
    leaky_gate = _stationary_noise(sample_rate, seconds=2.0)
    for start in range(0, leaky_gate.size, sample_rate * 2 // 5):
        leaky_gate[start : start + sample_rate // 5] = 0.0
    unsteady = analyze_noise_reference(leaky_gate, None, sample_rate)
    assert unsteady.status == "invalid"
    assert unsteady.must_recapture is False

    clipped = _stationary_noise(sample_rate)
    clipped[::200] = 1.0
    too_short = _stationary_noise(sample_rate, seconds=0.5)
    for capture in (clipped, too_short):
        result = analyze_noise_reference(capture, None, sample_rate)
        assert result.status == "invalid"
        assert result.must_recapture is True
    assert analyze_noise_reference(_stationary_noise(sample_rate), None, sample_rate).must_recapture is False


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


def test_noise_capture_timestamp_after_voice_is_invalid():
    sample_rate = 48_000
    result = analyze_noise_reference(
        _stationary_noise(sample_rate),
        _speech_with_quiet_gaps(sample_rate),
        sample_rate,
        noise_metadata=CaptureMetadata(captured_at_unix_s=10_000.0),
        speech_metadata=CaptureMetadata(captured_at_unix_s=1_000.0),
    )

    assert result.status == "invalid"
    assert any("timestamp" in reason for reason in result.reasons)


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
