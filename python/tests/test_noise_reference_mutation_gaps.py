"""Noise-reference behaviours that a mutation run found unpinned."""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from mic_eq.analysis import noise_reference
from mic_eq.analysis.noise_reference import (
    OCTAVE_CENTERS_HZ,
    CaptureMetadata,
    analyze_noise_reference,
)

SAMPLE_RATE = 48_000
QUALITY_COMPONENTS = (
    "duration",
    "finite",
    "stationarity",
    "octave",
    "transient",
    "contamination",
    "consistency",
    "age",
)


def _white(amplitude: float, seconds: float = 3.0, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amplitude * rng.normal(size=int(SAMPLE_RATE * seconds))


def _tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return 0.01 * np.sin(2.0 * np.pi * frequency * t)


def _quality_inputs(monkeypatch, *args, **kwargs):
    """Run an analysis and return the scores it fed the quality mean, by name."""
    seen = []

    def record(components):
        seen.append(components)
        return 0.5

    monkeypatch.setattr(noise_reference, "_quality_mean", record)
    result = analyze_noise_reference(*args, **kwargs)
    (components,) = seen
    return dict(zip(QUALITY_COMPONENTS, components, strict=True)), result


def test_the_octave_bands_have_a_4_khz_centre_that_spans_2_8_to_5_7_khz():
    assert list(OCTAVE_CENTERS_HZ) == [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    four_khz = list(OCTAVE_CENTERS_HZ).index(4000.0)

    for frequency in (2900.0, 4000.0, 5500.0):
        frames = noise_reference._frame_analysis(_tone(frequency), SAMPLE_RATE)
        assert frames is not None
        assert frames.band_levels_db.shape[1] == len(OCTAVE_CENTERS_HZ)
        assert int(np.argmax(np.median(frames.band_levels_db, axis=0))) == four_khz


def _varying_bands(seconds: float = 6.0) -> np.ndarray:
    """Seven octave tones whose levels wander independently every 50 ms, some far more than others."""
    rng = np.random.default_rng(77)
    segment = int(SAMPLE_RATE * 0.05)
    count = int(SAMPLE_RATE * seconds)
    t = np.arange(count) / SAMPLE_RATE
    signal = np.zeros(count)
    for center, swing_db in zip(OCTAVE_CENTERS_HZ, (3.0, 3.0, 3.0, 6.0, 20.0, 30.0, 40.0)):
        level_db = rng.uniform(-swing_db, 0.0, size=count // segment + 1)
        amplitude = 0.01 * 10.0 ** (np.repeat(level_db, segment)[:count] / 20.0)
        signal += amplitude * np.sin(2.0 * np.pi * center * t)
    return signal


def test_octave_stability_and_spectral_flux_summarise_the_band_levels():
    frames = noise_reference._frame_analysis(_varying_bands(), SAMPLE_RATE)
    assert frames is not None
    levels = frames.band_levels_db
    count, bands = levels.shape

    # Typical band swing: the median over bands of each band's 10th-90th percentile spread.
    spreads = [
        float(np.percentile(levels[:, b], 90.0) - np.percentile(levels[:, b], 10.0))
        for b in range(bands)
    ]
    expected_stability = statistics.median(spreads)
    # Frame-to-frame change of the spectral shape (levels relative to each frame's median band),
    # median over bands, then the 95th percentile over frame pairs.
    shape = [[level - statistics.median(row) for level in row] for row in levels.tolist()]
    changes = [
        statistics.median(abs(shape[i + 1][b] - shape[i][b]) for b in range(bands))
        for i in range(count - 1)
    ]
    expected_flux = float(np.percentile(changes, 95.0))

    # The signal tells the median from the mean and the extremes, and the 95th percentile of the
    # changes from their 90th and their maximum.
    assert abs(statistics.mean(spreads) - expected_stability) > 1.0
    assert max(spreads) - expected_stability > 1.0
    assert np.percentile(changes, 90.0) < expected_flux < max(changes)
    assert frames.octave_stability_db == pytest.approx(expected_stability, abs=1e-9)
    assert frames.spectral_flux_db == pytest.approx(expected_flux, abs=1e-9)


def test_a_single_analysis_window_has_no_spectral_flux():
    frames = noise_reference._frame_analysis(
        _white(0.01, seconds=0.2), SAMPLE_RATE, frame_size=int(SAMPLE_RATE * 0.2)
    )
    assert frames is not None
    assert frames.band_levels_db.shape[0] == 1
    assert frames.spectral_flux_db == 0.0
    assert frames.octave_stability_db == 0.0


def test_vad_interpolation_holds_the_last_probability_past_the_right_edge():
    # Fewer probabilities than frames: the last frames lie past the last source point.
    out = noise_reference._interpolate_vad(np.array([0.0, 0.5]), 10)
    assert out is not None
    assert out[:2] == pytest.approx([0.0, 0.0])
    assert out[-2:] == pytest.approx([0.5, 0.5])
    out = noise_reference._interpolate_vad(np.array([1.0, 0.25]), 10)
    assert out is not None
    assert out[:2] == pytest.approx([1.0, 1.0])
    assert out[-2:] == pytest.approx([0.25, 0.25])
    # The held values are clipped to a probability like everything between them.
    out = noise_reference._interpolate_vad(np.array([-0.5, 2.0]), 10)
    assert out is not None
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert np.all((out >= 0.0) & (out <= 1.0))
    # In between, frames sit on the straight line through the probabilities' midpoints.
    out = noise_reference._interpolate_vad(np.array([0.0, 1.0]), 4)
    assert out is not None
    assert out == pytest.approx([0.0, 0.25, 0.75, 1.0])


@pytest.mark.parametrize(
    "probabilities, frame_count",
    [(None, 4), (np.array([0.5]), 0), (np.empty(0), 4)],
)
def test_vad_interpolation_needs_probabilities_and_frames(probabilities, frame_count):
    assert noise_reference._interpolate_vad(probabilities, frame_count) is None


def test_quality_mean_is_a_weighted_geometric_mean_of_scores_clipped_to_the_unit_range():
    mean = noise_reference._quality_mean
    assert mean([(0.25, 1.0), (1.0, 1.0)]) == pytest.approx(0.5)
    assert mean([(0.25, 3.0), (1.0, 1.0)]) == pytest.approx(0.25**0.75)
    assert mean([(4.0, 1.0), (0.25, 1.0)]) == pytest.approx(0.5)  # scores above 1 count as 1
    assert mean([(0.0, 1.0)]) == pytest.approx(0.02)  # and a zero score as 0.02
    assert mean([(-3.0, 1.0)]) == pytest.approx(0.02)
    assert mean([]) == 0.0
    assert mean([(0.5, 0.0), (0.5, 0.0)]) == 0.0


def test_quality_mean_treats_a_negative_weight_as_no_weight():
    mean = noise_reference._quality_mean
    assert mean([(0.5, 1.0), (0.1, -5.0)]) == pytest.approx(0.5)
    assert mean([(0.5, 1.0), (0.1, 0.0)]) == pytest.approx(0.5)
    assert mean([(0.5, -1.0), (0.1, -5.0)]) == 0.0


def test_quality_mean_weights_of_the_analysis(monkeypatch):
    weights = []
    monkeypatch.setattr(
        noise_reference, "_quality_mean", lambda c: weights.extend(w for _, w in c) or 0.5
    )
    analyze_noise_reference(_white(0.01), None, SAMPLE_RATE)
    assert weights == [0.10, 0.10, 0.18, 0.15, 0.10, 0.15, 0.17, 0.05]


@pytest.mark.parametrize("rate", [44_100, 96_000])
@pytest.mark.parametrize("side", ["noise", "voice"])
def test_a_capture_at_another_sample_rate_than_the_analysis_is_flagged_either_way(rate, side):
    noise = CaptureMetadata(sample_rate=rate if side == "noise" else None)
    voice = CaptureMetadata(sample_rate=rate if side == "voice" else None)

    reasons, _ = noise_reference._metadata_mismatches(noise, voice, SAMPLE_RATE)

    assert reasons == [f"{side} capture sample rate does not match analysis"]


@pytest.mark.parametrize("noise_rate, voice_rate", [(44_100, 48_000), (48_000, 44_100)])
def test_captures_at_different_sample_rates_are_flagged_in_either_order(noise_rate, voice_rate):
    reasons, _ = noise_reference._metadata_mismatches(
        CaptureMetadata(sample_rate=noise_rate),
        CaptureMetadata(sample_rate=voice_rate),
        SAMPLE_RATE,
    )

    assert "sample rate changed between noise and voice captures" in reasons
    matching = noise_reference._metadata_mismatches(
        CaptureMetadata(sample_rate=SAMPLE_RATE),
        CaptureMetadata(sample_rate=SAMPLE_RATE),
        SAMPLE_RATE,
    )
    assert matching == ([], None)


def test_a_sample_rate_mismatch_makes_the_reference_invalid():
    result = analyze_noise_reference(
        _white(0.01),
        None,
        SAMPLE_RATE,
        noise_metadata=CaptureMetadata(sample_rate=44_100),
    )

    assert result.status == "invalid"
    assert result.usable is False
    assert "noise capture sample rate does not match analysis" in result.reasons


@pytest.mark.parametrize(
    "crest_db, expected_status, strong_transients",
    [
        (6.0, "usable", False),
        (18.0, "usable", False),
        (22.0, "usable", False),
        (28.0, "questionable", True),
    ],
)
def test_transient_score_falls_by_a_third_per_6_db_of_crest_above_12(
    monkeypatch, crest_db, expected_status, strong_transients
):
    noise = np.where(np.random.default_rng(1).random(SAMPLE_RATE * 3) < 0.5, -0.01, 0.01)
    noise[70_000] = 0.01 * 10.0 ** (crest_db / 20.0)

    scores, result = _quality_inputs(monkeypatch, noise, None, SAMPLE_RATE)

    assert result.metrics["crest_factor_db"] == pytest.approx(crest_db, abs=0.1)
    assert scores["transient"][0] == pytest.approx(
        1.0 - max(0.0, result.metrics["crest_factor_db"] - 12.0) / 18.0
    )
    assert scores["transient"][0] == pytest.approx(
        {6.0: 1.0, 18.0: 2.0 / 3.0, 22.0: 4.0 / 9.0, 28.0: 1.0 / 9.0}[crest_db], abs=0.01
    )
    assert result.status == expected_status
    assert ("room-noise capture contains strong transients" in result.reasons) is strong_transients


@pytest.mark.parametrize(
    "active_frames, expected_status",
    [(0, "usable"), (2, "usable"), (3, "questionable"), (8, "questionable"), (9, "invalid")],
)
def test_speech_contamination_bands_are_8_and_30_percent_of_frames(
    monkeypatch, active_frames, expected_status
):
    # 3 s of noise is 29 analysis windows, so each active window is 3.4 % of them.
    vad = np.zeros(29)
    vad[:active_frames] = 0.9

    scores, result = _quality_inputs(
        monkeypatch, _white(0.01), None, SAMPLE_RATE, noise_vad_probabilities=vad
    )

    ratio = active_frames / 29
    assert result.metrics["vad_contamination_ratio"] == pytest.approx(ratio)
    assert scores["contamination"][0] == pytest.approx(max(0.0, 1.0 - ratio / 0.30))
    assert result.status == expected_status
    possible = "possible speech contamination in room-noise capture"
    present = "speech is present in the room-noise capture"
    assert (possible in result.reasons) is (expected_status == "questionable")
    assert (present in result.reasons) is (expected_status == "invalid")


@pytest.mark.parametrize(
    "age_s, score, reason",
    [
        (None, 1.0, None),
        (60.0, 0.9, None),
        (100.0, 1.0 - 100.0 / 600.0, None),
        (130.0, 1.0 - 130.0 / 600.0, "room-noise reference may be stale"),
        (300.0, 0.5, "room-noise reference may be stale"),
        (590.0, 1.0 - 590.0 / 600.0, "room-noise reference may be stale"),
        (610.0, 0.0, "room-noise reference is stale"),
    ],
)
def test_age_score_falls_to_zero_over_ten_minutes(monkeypatch, age_s, score, reason):
    noise_meta = CaptureMetadata(captured_at_unix_s=1_000.0)
    voice_meta = CaptureMetadata(captured_at_unix_s=None if age_s is None else 1_000.0 + age_s)

    scores, result = _quality_inputs(
        monkeypatch,
        _white(0.01),
        None,
        SAMPLE_RATE,
        noise_metadata=noise_meta,
        speech_metadata=voice_meta,
    )

    assert scores["age"][0] == pytest.approx(score)
    assert result.metrics["capture_age_s"] == age_s
    stale = [r for r in result.reasons if "stale" in r]
    assert stale == ([reason] if reason else [])


def _tilted_noise(tilt_db: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Seconds of white noise whose spectrum slopes by +-tilt_db between lo_hz and hi_hz."""
    noise = _white(0.01, seconds=5.0, seed=9)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(noise.size, 1.0 / SAMPLE_RATE)
    inside = (frequencies >= lo_hz) & (frequencies <= hi_hz)
    gain_db = np.zeros(frequencies.shape)
    gain_db[inside] = np.linspace(-tilt_db, tilt_db, int(inside.sum()))
    return np.fft.irfft(spectrum * 10.0 ** (gain_db / 20.0), noise.size)


def _shape_distance(voice: np.ndarray) -> float:
    result = analyze_noise_reference(
        _white(0.01), voice, SAMPLE_RATE, speech_vad_probabilities=np.zeros(40)
    )
    distance = result.metrics["spectral_shape_distance_db"]
    assert distance is not None
    return float(distance)


def test_spectral_shape_is_compared_between_80_and_8000_hz_only():
    # A +-30 dB slope across the voice band is a typical 15 dB away from the flat reference.
    assert 14.0 < _shape_distance(_tilted_noise(30.0, 80.0, 8_000.0)) < 15.5
    # The same slope above the voice band does not count.
    assert _shape_distance(_tilted_noise(30.0, 8_200.0, 24_000.0)) < 2.0
    # A slope over the whole spectrum is read over the band only: 80-8000 Hz is a third of its
    # +-30 dB, a 20 dB run whose typical distance from its middle is 5 dB.
    assert 4.0 < _shape_distance(_tilted_noise(30.0, 0.0, 24_000.0)) < 6.0


@pytest.mark.parametrize(
    "delta_db, expected",
    [(-9.0, "usable"), (-11.0, "usable"), (-13.0, "questionable"), (-19.0, "questionable"),
     (-21.0, "invalid"), (-26.0, "invalid")],
)
def test_room_noise_much_louder_than_the_pauses_is_questionable_then_invalid(
    delta_db, expected
):
    voice = _white(0.01 * 10.0 ** (delta_db / 20.0), seconds=5.0, seed=9)

    result = analyze_noise_reference(
        _white(0.01), voice, SAMPLE_RATE, speech_vad_probabilities=np.zeros(40)
    )

    assert result.metrics["in_capture_level_delta_db"] == pytest.approx(delta_db, abs=0.5)
    assert result.status == expected
    changed = "room-noise level changed substantially before the voice capture"
    louder = "room-noise reference is much louder than in-capture quiet frames"
    assert (changed in result.reasons) is (expected == "invalid")
    assert (louder in result.reasons) is (expected == "questionable")
    assert result.usable is (expected != "invalid")


@pytest.mark.parametrize("seconds", [0.0, 0.05])
def test_a_capture_too_short_to_analyse_has_a_minus_120_db_placeholder_spectrum(seconds):
    noise = _white(0.01, seconds=seconds)

    result = analyze_noise_reference(noise, None, SAMPLE_RATE)

    bins = max(2, noise.size)
    np.testing.assert_array_equal(
        result.frequencies, np.fft.rfftfreq(bins, 1.0 / SAMPLE_RATE)
    )
    np.testing.assert_array_equal(result.explicit_spectrum_db, np.full(bins // 2 + 1, -120.0))
    np.testing.assert_array_equal(result.conservative_spectrum_db, result.explicit_spectrum_db)
    assert result.conservative_spectrum_db is not result.explicit_spectrum_db


def test_a_capture_with_one_analysis_window_has_a_measured_spectrum():
    result = analyze_noise_reference(_white(0.01, seconds=0.1), None, SAMPLE_RATE)

    assert result.frequencies.size == 2049
    assert np.all(result.explicit_spectrum_db > -120.0)


def _speech_with_quiet_gaps() -> np.ndarray:
    t = np.arange(SAMPLE_RATE * 5) / SAMPLE_RATE
    envelope = np.where((t % 1.0) < 0.72, 1.0, 0.015)
    voice = (
        0.08 * np.sin(2.0 * np.pi * 135.0 * t)
        + 0.035 * np.sin(2.0 * np.pi * 270.0 * t)
        + _white(0.015, 5.0, seed=11)
    )
    return envelope * voice


@pytest.mark.parametrize(
    "case, status, conservative",
    [
        ("clean", "usable", False),
        ("clean with pauses in the voice capture", "usable", True),
        ("silent reference", "questionable", True),
        ("too short", "invalid", True),
    ],
)
def test_conservative_flag_is_set_by_doubt_or_by_in_capture_noise(case, status, conservative):
    speech = None
    noise = _white(0.002)
    if case == "clean with pauses in the voice capture":
        speech = _speech_with_quiet_gaps()
    elif case == "silent reference":
        noise = np.zeros(SAMPLE_RATE * 3)
    elif case == "too short":
        noise = _white(0.002, seconds=0.5)

    result = analyze_noise_reference(noise, speech, SAMPLE_RATE)

    assert result.status == status
    assert (result.in_capture_spectrum_db is not None) is (speech is not None)
    assert result.conservative is conservative
    assert result.diagnostics()["conservative"] is conservative
