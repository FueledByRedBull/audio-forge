"""Typed native EQ filter, slope, and compatibility contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mic_eq import (
    AudioProcessor,
    eq_magnitude_response_v2,
)
from mic_eq.mic_eq_core import (
    simulate_auto_eq_chain,
    simulate_eq_v2,
)
from mic_eq.config import EQSettings


def _typed_default_bands() -> list[tuple[str, float, float, float, int, bool]]:
    return [
        (
            band.filter_type,
            band.frequency_hz,
            band.gain_db,
            band.q,
            band.slope_db_per_octave,
            band.enabled,
        )
        for band in EQSettings().bands
    ]


def test_typed_native_eq_api_round_trips_every_runtime_field() -> None:
    processor = AudioProcessor()
    bands = _typed_default_bands()
    bands[0] = ("high_pass", 70.0, 0.0, 1.41, 36, True)
    bands[4] = ("notch", 2100.0, 0.0, 7.5, 12, True)
    bands[9] = ("high_shelf", 16_000.0, 3.0, 0.8, 12, False)

    processor.apply_eq_settings_v2(bands)

    for index, expected in enumerate(bands):
        assert processor.get_eq_band_config(index) == expected


def test_typed_native_eq_api_rejects_unknown_type_and_odd_slope() -> None:
    processor = AudioProcessor()
    bands = _typed_default_bands()
    bands[4] = ("magic", 1000.0, 0.0, 1.0, 12, True)
    with pytest.raises(ValueError, match="unsupported EQ filter type"):
        processor.apply_eq_settings_v2(bands)

    bands[4] = ("high_pass", 1000.0, 0.0, 1.0, 18, True)
    with pytest.raises(ValueError, match="expected one of"):
        processor.apply_eq_settings_v2(bands)


@pytest.mark.parametrize("filter_type", ["high_pass", "low_pass"])
@pytest.mark.parametrize("slope", [12, 24, 36, 48])
def test_typed_pass_response_is_minus_three_db_at_cutoff(
    filter_type: str,
    slope: int,
) -> None:
    bands = _typed_default_bands()
    bands[4] = (filter_type, 2000.0, 0.0, 1.0, slope, True)

    response = eq_magnitude_response_v2([2000.0], bands, 48_000.0)

    assert response[0] == pytest.approx(-20.0 * math.log10(math.sqrt(2.0)), abs=1e-8)


def test_typed_notch_response_ignores_gain_and_nulls_center() -> None:
    bands = _typed_default_bands()
    bands[4] = ("notch", 1000.0, 12.0, 8.0, 12, True)

    response = eq_magnitude_response_v2(
        [100.0, 1000.0, 10_000.0],
        bands,
        48_000.0,
    )

    assert response[1] < -150.0
    assert abs(response[0]) < 0.1
    assert abs(response[2]) < 0.1


def test_disabled_typed_band_is_flat_and_finite() -> None:
    bands = _typed_default_bands()
    bands[4] = ("high_pass", 20_000.0, 12.0, 10.0, 48, False)

    response = eq_magnitude_response_v2(
        np.geomspace(20.0, 20_000.0, 100).tolist(),
        bands,
        48_000.0,
    )

    np.testing.assert_allclose(response, 0.0, rtol=0.0, atol=1e-12)


def test_legacy_batch_api_restores_historical_filter_layout() -> None:
    processor = AudioProcessor()
    processor.set_eq_band_filter_type(4, "notch")
    processor.set_eq_band_enabled(4, False)
    legacy = [
        (band.frequency_hz, band.gain_db, band.q)
        for band in EQSettings().bands
    ]

    processor.apply_eq_settings(legacy)

    low = processor.get_eq_band_config(0)
    middle = processor.get_eq_band_config(4)
    high = processor.get_eq_band_config(9)
    assert low is not None
    assert middle is not None
    assert high is not None
    assert low[0] == "low_shelf"
    assert middle[0] == "bell"
    assert middle[5] is True
    assert high[0] == "high_shelf"


def test_native_typed_eq_simulator_preserves_default_audio_exactly() -> None:
    phase = np.arange(48_000, dtype=np.float32)
    audio = (
        0.2 * np.sin(phase * np.float32(2.0 * np.pi * 997.0 / 48_000.0))
    ).astype(np.float32)

    result = simulate_eq_v2(
        audio,
        48_000.0,
        _typed_default_bands(),
        return_output_audio=True,
    )

    np.testing.assert_array_equal(
        np.asarray(result["output_audio"], dtype=np.float32),
        audio,
    )
    assert result["algorithmic_latency_samples"] == 0
    assert result["non_finite_output"] is False
    assert result["max_response_db"] == pytest.approx(0.0, abs=1e-12)


def test_native_typed_eq_simulator_rejects_non_finite_audio() -> None:
    audio = np.asarray([0.0, np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="finite samples"):
        simulate_eq_v2(audio, 48_000.0, _typed_default_bands())


def test_native_typed_eq_simulator_handles_steep_pass_filter() -> None:
    rng = np.random.default_rng(0xA0D10)
    audio = rng.normal(0.0, 0.1, 48_000).astype(np.float32)
    bands = _typed_default_bands()
    bands[0] = ("high_pass", 80.0, 0.0, 1.41, 48, True)

    result = simulate_eq_v2(audio, 48_000.0, bands)

    assert result["sample_count"] == audio.size
    assert result["runtime_ms"] > 0.0
    assert result["algorithmic_latency_samples"] == 0
    assert result["non_finite_output"] is False
    assert math.isfinite(float(result["output_true_peak"]))


def test_typed_eq_full_chain_engages_limiter_and_respects_true_peak_ceiling() -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    audio = (0.5 * np.sin(2.0 * np.pi * 1000.0 * time)).astype(np.float32)
    typed = [
        ("bell", 1000.0 + index * 10.0, 12.0, 10.0, 12, True)
        for index in range(10)
    ]
    legacy = [
        (band.frequency_hz, band.gain_db, band.q)
        for band in EQSettings().bands
    ]

    result = simulate_auto_eq_chain(
        audio,
        float(sample_rate),
        legacy,
        {
            "eq_bands_v2": typed,
            "deesser_enabled": False,
            "compressor_enabled": False,
            "limiter_enabled": True,
            "limiter_careful_output_enabled": True,
        },
    )

    assert result["non_finite_output"] is False
    assert result["limiter_gain_reduction_db"] > 1.0
    assert result["output_true_peak_db"] <= (
        result["limiter_effective_ceiling_db"] + 0.05
    )
