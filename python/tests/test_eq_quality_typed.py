"""Typed EQ quality and response parity contracts."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eq import eq_magnitude_response_v2
from mic_eq.analysis.auto_eq_parts import response
from mic_eq.analysis.eq_quality import evaluate_eq_quality
from mic_eq.analysis.auto_eq_parts.constants import SAMPLE_RATE
from mic_eq.config import EQSettings


def test_band_zero_bell_is_not_misidentified_as_a_low_shelf() -> None:
    metrics = evaluate_eq_quality(
        [80.0, 160.0, 250.0],
        [4.0, 4.0, 0.0],
        [1.0, 1.0, 1.0],
        filter_types=["bell", "bell", "bell"],
    )

    assert metrics.shelf_peak_stacking == 0


def test_frequency_reordered_middle_band_keeps_its_filter_type() -> None:
    metrics = evaluate_eq_quality(
        [200.0, 400.0, 240.0],
        [4.0, 0.0, 4.0],
        [1.0, 1.0, 1.0],
        filter_types=["low_shelf", "bell", "bell"],
    )

    assert metrics.shelf_peak_stacking == 1
    warning = next(item for item in metrics.warnings if item.kind == "shelf_stack")
    assert warning.frequency_hz == 240.0


def test_only_enabled_real_shelf_and_bell_overlap_warn() -> None:
    arguments = (
        [100.0, 240.0, 200.0],
        [0.0, 3.0, 4.0],
        [1.0, 1.0, 1.0],
    )
    metrics = evaluate_eq_quality(
        *arguments,
        filter_types=["bell", "bell", "low_shelf"],
    )
    disabled = evaluate_eq_quality(
        *arguments,
        filter_types=["bell", "bell", "low_shelf"],
        enabled=[True, True, False],
    )

    assert metrics.shelf_peak_stacking == 1
    assert disabled.shelf_peak_stacking == 0


def test_distant_shelf_and_peak_do_not_warn_as_nearby_stacking() -> None:
    metrics = evaluate_eq_quality(
        [20.0, 320.0],
        [3.0, 3.0],
        [1.0, 1.0],
        filter_types=["low_shelf", "bell"],
    )

    assert metrics.shelf_peak_stacking == 0


def test_peak_on_low_shelf_plateau_still_warns_despite_center_distance() -> None:
    metrics = evaluate_eq_quality(
        [100.0, 20.0],
        [3.0, 3.0],
        [1.0, 1.0],
        filter_types=["low_shelf", "bell"],
    )

    assert metrics.shelf_peak_stacking == 1


def test_high_shelf_warning_follows_the_shelf_plateau_direction() -> None:
    plateau = evaluate_eq_quality(
        [7_000.0, 16_000.0],
        [3.0, 3.0],
        [1.0, 1.0],
        filter_types=["high_shelf", "bell"],
    )
    transition_side = evaluate_eq_quality(
        [16_000.0, 7_000.0],
        [3.0, 3.0],
        [1.0, 1.0],
        filter_types=["high_shelf", "bell"],
    )

    assert plateau.shelf_peak_stacking == 1
    assert transition_side.shelf_peak_stacking == 0


@pytest.mark.parametrize(
    ("middle_filter", "middle_enabled"),
    [("bell", False), ("low_pass", True)],
)
def test_non_gain_band_does_not_hide_adjacent_enabled_bell_overlap(
    middle_filter: str, middle_enabled: bool
) -> None:
    metrics = evaluate_eq_quality(
        [100.0, 105.0, 110.0],
        [4.0, 0.0, 4.0],
        [1.0, 1.0, 1.0],
        filter_types=["bell", middle_filter, "bell"],
        enabled=[True, middle_enabled, True],
    )

    assert metrics.overlapping_adjacent_bands == 1


def test_legacy_response_keeps_shelf_roles_with_original_band_order() -> None:
    centers = [
        1000.0,
        80.0,
        160.0,
        320.0,
        640.0,
        1280.0,
        2500.0,
        5000.0,
        8000.0,
        12000.0,
    ]
    gains = [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    qs = [1.0] * 10
    frequencies = np.logspace(
        np.log10(20.0), np.log10(min(20_000.0, SAMPLE_RATE / 2.0 - 1.0)), 256
    )

    expected = response._predict_eq_response(
        frequencies, gains, qs, centers, sample_rate=SAMPLE_RATE
    )
    metrics = evaluate_eq_quality(centers, gains, qs, sample_rate=SAMPLE_RATE)

    assert metrics.max_boost_db == pytest.approx(max(0.0, float(np.max(expected))))
    assert metrics.max_cut_db == pytest.approx(max(0.0, float(-np.min(expected))))


@pytest.mark.parametrize("sample_rate", [44_100.0, 48_000.0])
def test_typed_notch_and_pass_response_matches_native_near_nyquist(
    sample_rate: float,
) -> None:
    bands = [
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
    bands[3] = ("low_pass", 600.0, 0.0, 1.0, 24, True)
    bands[4] = ("notch", 1200.0, 9.0, 3.2, 12, True)
    bands[5] = ("high_pass", 2400.0, 0.0, 0.8, 48, True)
    bands[9] = ("high_shelf", sample_rate / 2.0 - 1.0, 2.0, 0.8, 12, True)
    frequencies = np.asarray(
        [0.0, 20.0, 600.0, 1200.0, 2400.0, sample_rate / 2.0 - 1.0, sample_rate / 2.0]
    )

    actual = response._predict_eq_response(
        frequencies,
        [band[2] for band in bands],
        [band[3] for band in bands],
        [band[1] for band in bands],
        filter_types=[band[0] for band in bands],
        slopes_db_per_octave=[band[4] for band in bands],
        enabled=[band[5] for band in bands],
        sample_rate=sample_rate,
    )
    expected = np.asarray(
        eq_magnitude_response_v2(frequencies.tolist(), bands, sample_rate)
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-8)
    assert np.isfinite(actual).all()


def test_typed_python_response_fallback_handles_notch_and_pass_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = [
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
    bands[4] = ("notch", 1000.0, 10.0, 5.0, 12, True)
    bands[5] = ("high_pass", 1800.0, 0.0, 1.0, 36, True)
    frequencies = np.asarray([100.0, 1000.0, 1800.0, 5000.0, 23_999.0])
    monkeypatch.setattr(response, "_native_typed_response", lambda *_args: None)

    actual = response._predict_eq_response(
        frequencies,
        [band[2] for band in bands],
        [band[3] for band in bands],
        [band[1] for band in bands],
        filter_types=[band[0] for band in bands],
        slopes_db_per_octave=[band[4] for band in bands],
        enabled=[band[5] for band in bands],
        sample_rate=48_000.0,
    )
    expected = np.asarray(
        eq_magnitude_response_v2(
            frequencies.tolist(),
            bands,
            48_000.0,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-8)


def test_headroom_fallback_uses_the_requested_high_center_coefficients() -> None:
    from mic_eq.analysis.auto_eq_parts import headroom

    actual_b, actual_a = headroom._biquad_coefficients(
        "peaking", 21_000.0, 6.0, 1.2, 44_100.0
    )
    raw = response._biquad_coefficients(
        6.0, 1.2, 21_000.0, "peak", 44_100.0
    )
    b0, b1, b2, a0, a1, a2 = raw

    np.testing.assert_allclose(
        actual_b,
        np.asarray([b0, b1, b2]) / a0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        actual_a,
        np.asarray([1.0, a1 / a0, a2 / a0]),
        rtol=0.0,
        atol=1.0e-14,
    )
