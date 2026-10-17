"""
Comprehensive pytest suite for Auto-EQ behavior.
"""

import re

import numpy as np
import pytest

import mic_eq
import mic_eq.mic_eq_core as mic_eq_core
from mic_eq import config
from mic_eq.analysis import auto_eq
from mic_eq.analysis.auto_eq_parts import headroom as headroom_module
from mic_eq.analysis.auto_eq_parts import optimizer as optimizer_module
from mic_eq.analysis.auto_eq_parts.dynamic_bands import (
    _snr_aware_gain_upper_bounds,
    _spectral_tilt_fit,
)
from mic_eq.analysis.auto_eq_parts.optimizer import _log_frequency_gain_curvature
from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.analysis.failure_detection import validate_analysis
from mic_eq.analysis.spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual
from mic_eq.ui.auto_eq_explanation import headroom_advice

calculate_eq_bands = auto_eq.calculate_eq_bands
analyze_auto_eq = auto_eq.analyze_auto_eq
get_target_curve = auto_eq.get_target_curve
evaluate_eq_quality = auto_eq.evaluate_eq_quality
apply_headroom_validation = auto_eq.apply_headroom_validation
simulate_candidate_chain = auto_eq.simulate_candidate_chain
EQ_FREQUENCIES = config.EQ_FREQUENCIES
AUTO_EQ_DEFAULT_Q = config.AUTO_EQ_DEFAULT_Q


def test_capture_validation_is_independent_of_fitting_smoothing():
    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    audio = np.asarray(sum(
        0.1 / harmonic * np.sin(2 * np.pi * 150 * harmonic * time)
        for harmonic in range(1, 26)
    ), dtype=np.float32)
    peak_counts = set()
    for smoothing in ("conservative", "balanced", "broad", "off"):
        _, validation = analyze_auto_eq(
            audio, sample_rate, "flat", smoothing_strength=smoothing,
            chain_settings={"compressor_enabled": False, "limiter_enabled": False},
        )
        assert validation.passed
        peak_counts.add(validation.details["peak_count"])
    assert len(peak_counts) == 1


@pytest.mark.parametrize("smoothing", ["conservative", "balanced", "broad", "off"])
@pytest.mark.parametrize("kind", ["silence", "noise", "tone"])
def test_unsmoothed_capture_validation_still_rejects_invalid_input(smoothing, kind):
    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    audio = {
        "silence": np.zeros(time.size),
        "noise": np.random.default_rng(4).normal(0, 0.03, time.size),
        "tone": 0.1 * np.sin(2 * np.pi * 150 * time),
    }[kind].astype(np.float32)
    with pytest.raises(ValueError, match="Recording too unclear"):
        analyze_auto_eq(
            audio, sample_rate, "flat", smoothing_strength=smoothing,
            chain_settings={"compressor_enabled": False, "limiter_enabled": False},
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("analysis_confidence", float("nan"), "analysis confidence"),
        ("phonetic_coverage", float("inf"), "phonetic coverage"),
        ("noise_reference_quality", float("nan"), "noise-reference quality"),
        ("noise_reference_status", "trusted", "unknown noise-reference status"),
    ],
)
def test_auto_eq_rejects_invalid_measurement_metadata(keyword, value, message):
    frequencies = np.geomspace(20.0, 20_000.0, 128)
    measured = np.zeros_like(frequencies)
    target = np.ones_like(frequencies)

    with pytest.raises(ValueError, match=message):
        calculate_eq_bands(
            frequencies,
            measured,
            target,
            **{keyword: value},
        )


def test_auto_eq_accepts_positive_infinite_uncertainty_as_unavailable_evidence():
    frequencies = np.geomspace(20.0, 20_000.0, 128)

    result = calculate_eq_bands(
        frequencies,
        np.zeros_like(frequencies),
        np.ones_like(frequencies),
        spectral_repeatability=np.zeros_like(frequencies),
        spectral_uncertainty_db=np.full_like(frequencies, np.inf),
    )

    assert result["spectral_uncertainty_available"] is True


@pytest.mark.parametrize("bad_value", [float("nan"), float("-inf"), -0.1])
def test_auto_eq_rejects_malformed_spectral_uncertainty(bad_value):
    frequencies = np.geomspace(20.0, 20_000.0, 128)
    uncertainty = np.full_like(frequencies, 0.3)
    uncertainty[17] = bad_value

    with pytest.raises(ValueError, match="spectral uncertainty"):
        calculate_eq_bands(
            frequencies,
            np.zeros_like(frequencies),
            np.ones_like(frequencies),
            spectral_uncertainty_db=uncertainty,
        )


@pytest.mark.parametrize(
    ("series_name", "series"),
    [
        ("frequencies", np.asarray([20.0, 100.0, 100.0, 1_000.0])),
        ("measured", np.asarray([0.0, 0.0, float("nan"), 0.0])),
        ("target", np.asarray([0.0, 0.0, float("inf"), 0.0])),
    ],
)
def test_auto_eq_rejects_invalid_spectrum_inputs(series_name, series):
    frequencies = np.asarray([20.0, 100.0, 500.0, 1_000.0])
    measured = np.zeros_like(frequencies)
    target = np.ones_like(frequencies)
    if series_name == "frequencies":
        frequencies = series
    elif series_name == "measured":
        measured = series
    else:
        target = series

    with pytest.raises(ValueError):
        calculate_eq_bands(frequencies, measured, target)


def test_constrained_gain_projection_preserves_feasible_material_correction():
    dense_freqs = np.geomspace(20.0, 20_000.0, 256)
    centers = np.asarray(
        [120, 205, 318, 355, 497, 903, 1823, 2985, 7246, 9961],
        dtype=float,
    )
    gains = np.asarray(
        [-9.73, -3.61, 0.27, 0.43, 0.50, 0.75, 1.61, 2.01, 1.16, 0.67],
        dtype=float,
    )
    qs = np.asarray([0.68, 0.62, 1.01, 4.22, 0.30, 0.30, 0.30, 0.44, 0.55, 0.67])
    target = _predict_eq_response(dense_freqs, gains, qs, centers)

    refined, success = optimizer_module._constrained_gain_refinement(
        gains,
        dense_freqs,
        np.zeros_like(dense_freqs),
        target,
        qs,
        centers,
        np.ones_like(dense_freqs),
    )
    limits = optimizer_module._adjacent_gain_limits(centers)

    assert success is True
    assert np.max(np.abs(refined)) > 5.0
    assert np.all(np.abs(np.diff(refined)) <= limits + 1.0e-6)


def test_constrained_refinement_can_recover_from_an_undersized_initial_fit():
    dense_freqs = np.geomspace(20.0, 20_000.0, 256)
    centers = np.geomspace(80.0, 16_000.0, 10)
    qs = np.full(10, 1.2)
    desired = np.asarray([0.0, 0.5, 1.5, 3.0, 5.0, 5.0, 3.0, 1.5, 0.5, 0.0])
    target = _predict_eq_response(dense_freqs, desired, qs, centers)
    undersized = desired * 0.20

    refined, success = optimizer_module._constrained_gain_refinement(
        undersized,
        dense_freqs,
        np.zeros_like(dense_freqs),
        target,
        qs,
        centers,
        np.ones_like(dense_freqs),
        np.full(10, -12.0),
        np.full(10, 12.0),
    )

    assert success is True
    assert np.max(refined) > np.max(undersized) + 2.0
    assert np.linalg.norm(refined - desired) < np.linalg.norm(undersized - desired)


@pytest.mark.parametrize("initial_gain", [0.0, 1.0e-3])
def test_constrained_refinement_recovers_nonzero_target_from_zero_or_tiny_start(
    initial_gain,
):
    dense_freqs = np.geomspace(20.0, 20_000.0, 256)
    centers = np.geomspace(80.0, 16_000.0, 10)
    qs = np.full(10, 1.2)
    desired = np.zeros(10)
    desired[4] = 4.0
    target = _predict_eq_response(dense_freqs, desired, qs, centers)
    initial = np.full(10, initial_gain)

    refined, success = optimizer_module._constrained_gain_refinement(
        initial,
        dense_freqs,
        np.zeros_like(dense_freqs),
        target,
        qs,
        centers,
        np.ones_like(dense_freqs),
        np.full(10, -12.0),
        np.full(10, 12.0),
    )

    assert success is True
    assert refined[4] > 2.0


@pytest.mark.parametrize("initial_gain", [0.0, 1.0e-3])
def test_constrained_refinement_preserves_a_flat_target_from_zero_or_tiny_start(
    initial_gain,
):
    dense_freqs = np.geomspace(20.0, 20_000.0, 256)
    centers = np.geomspace(80.0, 16_000.0, 10)
    qs = np.full(10, 1.2)
    initial = np.full(10, initial_gain)

    refined, success = optimizer_module._constrained_gain_refinement(
        initial,
        dense_freqs,
        np.zeros_like(dense_freqs),
        np.zeros_like(dense_freqs),
        qs,
        centers,
        np.ones_like(dense_freqs),
        np.full(10, -12.0),
        np.full(10, 12.0),
    )

    assert success is True
    assert np.max(np.abs(refined)) < 1.0e-4


def test_constrained_refinement_can_activate_a_band_for_a_coupled_multiband_target():
    dense_freqs = np.geomspace(20.0, 20_000.0, 256)
    centers = np.geomspace(80.0, 16_000.0, 10)
    qs = np.full(10, 1.2)
    desired = np.asarray([0.0, 0.5, 1.5, 3.0, 5.0, 5.0, 3.0, 1.5, 0.5, 0.0])
    target = _predict_eq_response(dense_freqs, desired, qs, centers)

    refined, success = optimizer_module._constrained_gain_refinement(
        np.zeros(10),
        dense_freqs,
        np.zeros_like(dense_freqs),
        target,
        qs,
        centers,
        np.ones_like(dense_freqs),
        np.full(10, -12.0),
        np.full(10, 12.0),
    )

    assert success is True
    assert refined[4] > 4.0
    assert refined[5] > 4.0


def test_validation_is_final_and_reported_metrics_match_returned_curve(monkeypatch):
    freqs = np.geomspace(20.0, 20_000.0, 512)
    log_freqs = np.log10(freqs)
    measured = np.full_like(freqs, -70.0) - 9.0 * np.exp(
        -((log_freqs - np.log10(2200.0)) ** 2) / (2 * 0.08**2)
    )
    target = get_target_curve(freqs, "flat")
    observed: dict[str, np.ndarray] = {}
    original = optimizer_module._validate_and_attenuate_solution

    def attenuate(*args, **kwargs):
        validated = original(*args, **kwargs)
        observed["validated"] = np.asarray(validated[0], dtype=float).copy()
        return validated

    monkeypatch.setattr(
        optimizer_module,
        "_validate_and_attenuate_solution",
        attenuate,
    )

    result = calculate_eq_bands(freqs, measured, target)
    returned = np.asarray(result["band_gains"], dtype=float)
    metrics = evaluate_eq_quality(
        result["band_freqs"],
        returned,
        result["band_qs"],
    ).to_dict()

    np.testing.assert_allclose(returned, observed["validated"], atol=1.0e-12)
    assert result["eq_quality"] == metrics


def test_validation_failure_abstains_instead_of_applying_a_flat_curve(monkeypatch):
    freqs = np.geomspace(20.0, 20_000.0, 256)
    measured = generate_test_spectrum(freqs, "harsh")
    target = get_target_curve(freqs, "flat")

    def reject_solution(
        _dense_freqs,
        _measured_dense_db,
        _target_dense_db,
        gains,
        qs,
        centers_hz,
        _weights,
        cancel_check=None,
        sample_rate=48_000.0,
    ):
        flat = np.zeros_like(gains)
        return (
            flat,
            1.0,
            1.0,
            0.0,
            evaluate_eq_quality(centers_hz, flat, qs).to_dict(),
        )

    monkeypatch.setattr(
        optimizer_module,
        "_validate_and_attenuate_solution",
        reject_solution,
    )

    result = calculate_eq_bands(freqs, measured, target)

    assert result["recommendation_status"] == "abstain"
    assert result["apply_recommended"] is False
    assert result["validation_gain_scale"] == 0.0
    assert result["abstention_reasons"] == [
        "no validated correction improved the target safely"
    ]
    assert np.allclose(result["band_gains"], 0.0)


def test_constraint_solver_failure_abstains_instead_of_applying_fallback(monkeypatch):
    freqs = np.geomspace(20.0, 20_000.0, 256)
    measured = generate_test_spectrum(freqs, "harsh")
    target = get_target_curve(freqs, "flat")

    def fail_refinement(gains, *_args, **_kwargs):
        return np.zeros_like(gains), False

    monkeypatch.setattr(
        optimizer_module,
        "_constrained_gain_refinement",
        fail_refinement,
    )

    result = calculate_eq_bands(freqs, measured, target)

    assert result["recommendation_status"] == "abstain"
    assert result["apply_recommended"] is False
    assert result["constraint_solver_success"] is False
    assert "constrained gain solve produced no safe correction" in result[
        "abstention_reasons"
    ]
    assert np.allclose(result["band_gains"], 0.0)


def _seed_for_response(response_type: str) -> int:
    seeds = {
        "noise": 1701,
        "quiet": 1702,
    }
    return seeds.get(response_type, 1700)


def generate_test_spectrum(freqs, response_type="flat"):
    """Generate synthetic spectra in dBFS."""
    base_level = -70.0
    rng = np.random.default_rng(_seed_for_response(response_type))

    if response_type == "flat":
        spectrum_db = np.full_like(freqs, base_level)
    elif response_type == "bassy":
        spectrum_db = base_level + 10.0 / (1 + (freqs / 200.0) ** 2)
    elif response_type == "bright":
        spectrum_db = base_level + 10.0 * (freqs / 4000.0) ** 2 / (
            1 + (freqs / 4000.0) ** 2
        )
    elif response_type == "midscooped":
        log_freq = np.log10(freqs)
        log_center = np.log10(1500.0)
        sigma = 0.18
        spectrum_db = base_level - 8.0 * np.exp(
            -((log_freq - log_center) ** 2) / (2 * sigma**2)
        )
    elif response_type == "dark":
        spectrum_db = base_level - 10.0 / (1 + (8000.0 / freqs) ** 2)
    elif response_type == "proximity":
        spectrum_db = base_level + 15.0 / (1 + (freqs / 100.0) ** 3)
    elif response_type == "harsh":
        spectrum_db = base_level + 12.0 * np.exp(
            -((freqs - 4000.0) ** 2) / (2 * 1500.0**2)
        )
    elif response_type == "noise":
        noise_level = -50.0
        spectrum_db = base_level + (noise_level - base_level) * rng.random(len(freqs))
    elif response_type == "extreme":
        spectrum_db = base_level + 20.0 * np.sin(3 * np.log10(freqs / 100.0))
    elif response_type == "quiet":
        spectrum_db = np.full_like(freqs, -85.0) + rng.normal(0.0, 2.0, len(freqs))
    else:
        spectrum_db = np.full_like(freqs, base_level)

    return np.clip(spectrum_db, -100.0, -30.0)


def _default_freqs():
    return np.logspace(np.log10(20), np.log10(20000), 1000)


def test_01_flat_response_to_flat_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "flat")
    target_db = get_target_curve(freqs, "flat")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    max_gain = max(abs(g) for g in gains)
    assert max_gain < 1.0


def test_02_bassy_mic_to_broadcast_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "bassy")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert gains[0] < -2.0 or gains[1] < -2.0


def test_03_dark_mic_to_podcast_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "dark")
    target_db = get_target_curve(freqs, "podcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert gains[7] > 2.0 or gains[8] > 2.0 or gains[9] > 2.0


def test_04_midscooped_to_streaming_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "midscooped")
    target_db = get_target_curve(freqs, "streaming")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    gains = eq["band_gains"]
    assert max(gains[3:6]) >= 1.0
    assert eq["eq_quality"]["safe_for_auto_eq"]


def test_05_proximity_effect_correction():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "proximity")
    target_db = get_target_curve(freqs, "broadcast")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    gains = eq["band_gains"]
    assert gains[0] <= -4.0
    assert eq["eq_quality"]["safe_for_auto_eq"]


def test_06_harsh_highs_correction():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "harsh")
    target_db = get_target_curve(freqs, "podcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert any(g < -2.0 for g in [gains[5], gains[6], gains[7]])


def test_07_noisy_signal():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "noise")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert all(-12.0 <= g <= 12.0 for g in gains)


def test_08_extreme_uneven_response():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "extreme")
    target_db = get_target_curve(freqs, "flat")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    # The automatic path stays inside a bounded combined response even for an
    # intentionally pathological spectrum.
    assert eq["eq_quality"]["safe_for_auto_eq"]
    assert eq["validation_after_error_db"] <= eq["validation_before_error_db"]


def test_09_very_quiet_signal():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "quiet")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert all(-12.0 <= g <= 12.0 for g in gains)


def test_10_predict_eq_response_linearity():
    center = float(EQ_FREQUENCIES[4])
    freqs = np.array([100.0, center, 10000.0])
    qs = [1.414] * 10

    gains_6db = np.zeros(10)
    gains_6db[4] = 6.0
    response_6db = _predict_eq_response(freqs, gains_6db, qs, EQ_FREQUENCIES)

    gains_12db = np.zeros(10)
    gains_12db[4] = 12.0
    response_12db = _predict_eq_response(freqs, gains_12db, qs, EQ_FREQUENCIES)

    linear_6db = 10 ** (response_6db[1] / 20.0)
    linear_12db = 10 ** (response_12db[1] / 20.0)
    ratio = linear_12db / linear_6db

    assert 1.8 <= ratio <= 2.2


def test_11_q_bounds_respected():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "extreme")
    target_db = get_target_curve(freqs, "streaming")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    qs = eq["band_qs"]
    centers = np.asarray(eq["band_freqs"], dtype=float)

    assert len(qs) == 10
    for i, q in enumerate(qs):
        assert 0.3 <= q <= 6.0
        if centers[i] < 250.0:
            assert q <= 2.5


def test_12_q_regularized_near_prior_for_flat_case():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "flat")
    target_db = get_target_curve(freqs, "flat")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    qs = np.asarray(eq["band_qs"], dtype=float)
    q_prior = np.full_like(qs, AUTO_EQ_DEFAULT_Q, dtype=float)
    q_high = np.where(np.asarray(EQ_FREQUENCIES, dtype=float) < 250.0, 2.5, 6.0)
    q_prior = np.clip(q_prior, 0.3, q_high)
    max_log_dev = np.max(np.abs(np.log(np.maximum(qs, 1e-9) / q_prior)))
    assert max_log_dev < 0.25


def test_13_dynamic_centers_are_sorted_and_valid_for_eq_roles():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "extreme")
    target_db = get_target_curve(freqs, "streaming")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    centers = np.asarray(eq["band_freqs"], dtype=float)

    assert len(centers) == 10
    assert np.all(np.isfinite(centers))
    assert np.all(np.diff(centers) > 0.0)
    assert 55.0 <= centers[0] <= 180.0
    assert np.all((centers[1:9] >= 200.0) & (centers[1:9] <= 9000.0))
    assert 9500.0 <= centers[9] <= 18000.0


def test_14_adjacent_gain_coupling_limit():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "extreme")
    target_db = get_target_curve(freqs, "flat")
    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    gains = np.asarray(eq["band_gains"], dtype=float)
    centers = np.asarray(eq["band_freqs"], dtype=float)

    assert np.all(np.abs(np.diff(gains)) <= 6.0 + 1e-9)
    slopes = np.abs(np.diff(gains)) / np.diff(np.log2(centers))
    assert np.all(slopes <= 12.0 + 1e-6)


def test_15_tilt_removal_reduces_linear_log_slope():
    freqs = _default_freqs()
    x = np.log10(freqs)
    measured_db = 4.0 * (x - np.mean(x))
    tilt, slope, _fit_r2 = _spectral_tilt_fit(freqs, measured_db)
    detrended = measured_db - tilt

    x_center = x - np.mean(x)
    residual_slope = float(np.dot(x_center, detrended) / np.dot(x_center, x_center))
    assert abs(slope) > 1.0
    assert abs(residual_slope) < 1e-3


def test_15a_tilt_removal_accepts_perfect_nonzero_intercept():
    freqs = _default_freqs()
    x = np.log10(freqs)
    measured_db = 4.0 * (x - np.mean(x)) + 10.0
    tilt, slope, _fit_r2 = _spectral_tilt_fit(freqs, measured_db)
    detrended = measured_db - tilt

    x_center = x - np.mean(x)
    residual_slope = float(np.dot(x_center, detrended) / np.dot(x_center, x_center))
    assert abs(slope - 4.0) < 1e-6
    assert abs(residual_slope) < 1e-6
    assert np.allclose(detrended, np.full_like(detrended, detrended[0]))


def test_15b_tilt_removal_rejects_flat_response():
    freqs = _default_freqs()
    measured_db = np.full_like(freqs, 6.0)

    tilt, slope, _fit_r2 = _spectral_tilt_fit(freqs, measured_db)
    detrended = measured_db - tilt

    assert slope == 0.0
    assert np.allclose(detrended, measured_db)


def test_15c_tilt_removal_accepts_noisy_tilt_above_fit_threshold():
    freqs = _default_freqs()
    x = np.log10(freqs)
    x_center = x - np.mean(x)
    rng = np.random.default_rng(1503)
    measured_db = 3.0 * x_center + 2.0 + rng.normal(0.0, 0.12, size=freqs.size)

    tilt, slope, _fit_r2 = _spectral_tilt_fit(freqs, measured_db)
    detrended = measured_db - tilt
    residual_slope = float(np.dot(x_center, detrended) / np.dot(x_center, x_center))

    assert abs(slope) > 2.0
    assert abs(residual_slope) < 0.2


def test_15d_tilt_removal_rejects_random_response_below_fit_threshold():
    freqs = _default_freqs()
    rng = np.random.default_rng(1504)
    measured_db = rng.normal(0.0, 4.0, size=freqs.size)

    tilt, slope, _fit_r2 = _spectral_tilt_fit(freqs, measured_db)
    detrended = measured_db - tilt

    assert slope == 0.0
    assert np.allclose(detrended, measured_db)


def test_15e_preserve_policy_keeps_broad_mic_response():
    freqs = _default_freqs()
    measured_db = -70.0 - 5.0 * np.log2(freqs / 1000.0)
    target_db = get_target_curve(freqs, "flat", target_mode="static")

    result = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        tilt_policy="preserve",
    )

    assert result["spectral_tilt_policy"] == "preserve"
    assert result["spectral_tilt_slope_db_per_decade"] < -10.0
    assert np.max(np.abs(np.asarray(result["band_gains"], dtype=float))) >= 2.0


@pytest.mark.parametrize("policy", ["detrend", "unknown"])
def test_15e1_unknown_tilt_policy_is_rejected(policy):
    freqs = _default_freqs()
    measured_db = np.zeros_like(freqs)
    target_db = get_target_curve(freqs, "flat", target_mode="static")

    with pytest.raises(ValueError, match="Unknown spectral tilt policy"):
        calculate_eq_bands(freqs, measured_db, target_db, tilt_policy=policy)


def test_15f_log_frequency_curvature_is_zero_for_linear_tilt():
    centers = np.asarray(
        [80.0, 210.0, 330.0, 800.0, 1450.0, 2400.0, 3900.0, 6100.0, 8500.0, 15000.0],
        dtype=float,
    )
    gains = 2.5 * np.log2(centers / 1000.0) + 1.0

    curvature = _log_frequency_gain_curvature(gains, centers)

    assert curvature.shape == (8,)
    assert np.max(np.abs(curvature)) < 1e-10


def test_15g_log_frequency_curvature_is_grid_density_stable():
    penalties = []
    for point_count in (9, 17, 33):
        log_centers = np.linspace(0.0, 4.0, point_count)
        centers = np.exp2(log_centers)
        gains = np.square(log_centers)
        curvature = _log_frequency_gain_curvature(gains, centers)
        penalties.append(float(np.sum(np.square(curvature))))

    assert np.allclose(penalties, np.full(3, 4.0), atol=1e-10)


def test_15h_unknown_evidence_cannot_authorize_narrow_q_or_large_boosts():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    measured = np.full_like(freqs, -70.0) - 12.0 * np.exp(
        -((log_freqs - np.log10(3000.0)) ** 2) / (2 * 0.03**2)
    )
    target = get_target_curve(freqs, "flat")

    eq = calculate_eq_bands(freqs, measured, target)
    gains = np.asarray(eq["band_gains"], dtype=float)
    qs = np.asarray(eq["band_qs"], dtype=float)
    active = np.abs(gains) >= 0.25

    assert np.max(gains) <= 3.0 + 1e-9
    assert np.all(qs[active] <= 2.8 + 1e-9)
    assert eq["q_confidence_binding_location"] == "joint_solver_bounds"


def test_16_snr_aware_boost_caps_are_bounded_and_monotonic():
    snr_db = np.array([-5.0, 0.0, 3.0, 8.0, 12.0, 18.0, 30.0], dtype=float)
    caps = _snr_aware_gain_upper_bounds(snr_db)

    assert np.all(caps >= 1.5)
    assert np.all(caps <= 12.0)
    assert np.all(np.diff(caps) >= -1e-9)


def test_16a_frequency_dependent_snr_caps_only_unsupported_boosts():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    measured_db = np.full_like(freqs, -70.0)
    measured_db -= 10.0 * np.exp(-((log_freqs - np.log10(6000.0)) ** 2) / (2 * 0.08**2))
    target_db = get_target_curve(freqs, "flat", target_mode="static")
    spectral_snr = np.full_like(freqs, 24.0)
    spectral_snr[(freqs >= 4500.0) & (freqs <= 8000.0)] = 0.0

    eq = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        spectral_repeatability=np.ones_like(freqs),
        spectral_snr_db=spectral_snr,
        noise_reference_source="explicit_capture",
        analysis_confidence=0.95,
    )
    centers = np.asarray(eq["band_freqs"], dtype=float)
    gains = np.asarray(eq["band_gains"], dtype=float)
    high_band = int(np.argmin(np.abs(centers - 6000.0)))

    assert eq["snr_reference_available"] is True
    assert eq["noise_reference_source"] == "explicit_capture"
    assert gains[high_band] <= 1.5


def test_16a_partial_missing_snr_stays_unknown_and_conservative():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    measured_db = np.full_like(freqs, -70.0)
    measured_db -= 10.0 * np.exp(-((log_freqs - np.log10(6000.0)) ** 2) / (2 * 0.08**2))
    target_db = get_target_curve(freqs, "flat", target_mode="static")
    spectral_snr = np.full_like(freqs, 24.0)
    spectral_snr[(freqs >= 4200.0) & (freqs <= 8500.0)] = np.nan

    eq = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        spectral_repeatability=np.ones_like(freqs),
        spectral_snr_db=spectral_snr,
        noise_reference_source="explicit_capture",
        analysis_confidence=0.95,
    )
    centers = np.asarray(eq["band_freqs"], dtype=float)
    high_band = int(np.argmin(np.abs(centers - 6000.0)))

    assert eq["snr_reference_available"] is True
    assert eq["band_snr_available"][high_band] is False
    assert eq["band_snr_db"][high_band] is None
    assert eq["band_gains"][high_band] <= 1.5 + 1e-9


def test_16b_low_quality_capture_abstains_instead_of_applying_eq():
    freqs = _default_freqs()
    measured_db = generate_test_spectrum(freqs, "harsh")
    target_db = get_target_curve(freqs, "flat", target_mode="static")

    eq = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        spectral_repeatability=np.full_like(freqs, 0.05),
        spectral_snr_db=np.full_like(freqs, -2.0),
        noise_reference_source="explicit_capture",
        voiced_window_ratio=0.08,
        analysis_confidence=0.15,
    )

    assert eq["recommendation_status"] == "abstain"
    assert eq["apply_recommended"] is False
    assert eq["abstention_reasons"]
    assert np.allclose(eq["band_gains"], 0.0)
    assert max(eq["band_confidences"]) < 0.75


def test_16b_questionable_noise_reference_records_reduced_reason():
    freqs = _default_freqs()
    measured_db = generate_test_spectrum(freqs, "harsh")
    target_db = get_target_curve(freqs, "flat", target_mode="static")

    eq = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        spectral_repeatability=np.ones_like(freqs),
        spectral_uncertainty_db=np.full_like(freqs, 0.3),
        phonetic_coverage=0.9,
        voiced_window_ratio=0.9,
        analysis_confidence=0.95,
        noise_reference_quality=0.6,
        noise_reference_status="questionable",
    )

    assert eq["recommendation_status"] == "reduced"
    assert "room-noise reference is questionable" in eq["recommendation_reasons"]


def test_16c_unsupported_band_abstains_locally_and_response_is_reprojected():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    measured_db = np.full_like(freqs, -70.0)
    measured_db -= 7.0 * np.exp(-((log_freqs - np.log10(550.0)) ** 2) / (2 * 0.10**2))
    measured_db -= 10.0 * np.exp(-((log_freqs - np.log10(6200.0)) ** 2) / (2 * 0.07**2))
    target_db = get_target_curve(freqs, "flat", target_mode="static")
    reliability = np.ones_like(freqs)
    reliability[freqs >= 4500.0] = 0.0
    spectral_snr = np.full_like(freqs, 24.0)
    spectral_snr[freqs >= 4500.0] = -3.0

    eq = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        spectral_repeatability=reliability,
        spectral_uncertainty_db=np.where(reliability > 0.0, 0.3, 8.0),
        phonetic_coverage=0.9,
        spectral_snr_db=spectral_snr,
        noise_reference_source="explicit_capture",
        analysis_confidence=0.9,
    )
    gains = np.asarray(eq["band_gains"], dtype=float)
    centers = np.asarray(eq["band_freqs"], dtype=float)
    limits = optimizer_module._adjacent_gain_limits(centers)

    assert eq["recommendation_status"] != "abstain"
    assert eq["local_abstained_band_indices"]
    assert np.any(np.abs(gains[centers < 2000.0]) >= 0.25)
    assert np.allclose(
        gains[np.asarray(eq["local_abstained_band_indices"], dtype=int)],
        0.0,
    )
    assert np.all(np.abs(np.diff(gains)) <= limits + 1.0e-9)


def test_17_dynamic_center_tracks_non_default_problem_frequency():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    spectrum_db = np.full_like(freqs, -70.0)
    spectrum_db -= 8.0 * np.exp(-((log_freqs - np.log10(2300.0)) ** 2) / (2 * 0.045**2))
    target_db = get_target_curve(freqs, "flat")

    eq = calculate_eq_bands(freqs, spectrum_db, target_db)
    centers = np.asarray(eq["band_freqs"], dtype=float)
    gains = np.asarray(eq["band_gains"], dtype=float)
    nearest = int(np.argmin(np.abs(centers - 2300.0)))

    assert abs(centers[nearest] - 2300.0) < 180.0
    assert abs(centers[nearest] - 2500.0) > 120.0
    assert gains[nearest] > 1.0


def test_18_q_is_narrower_for_narrower_spectral_issue():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    target_db = get_target_curve(freqs, "flat")

    narrow = np.full_like(freqs, -70.0) - 8.0 * np.exp(
        -((log_freqs - np.log10(2300.0)) ** 2) / (2 * 0.025**2)
    )
    broad = np.full_like(freqs, -70.0) - 8.0 * np.exp(
        -((log_freqs - np.log10(2300.0)) ** 2) / (2 * 0.12**2)
    )

    narrow_eq = calculate_eq_bands(freqs, narrow, target_db)
    broad_eq = calculate_eq_bands(freqs, broad, target_db)
    narrow_centers = np.asarray(narrow_eq["band_freqs"], dtype=float)
    broad_centers = np.asarray(broad_eq["band_freqs"], dtype=float)
    narrow_qs = np.asarray(narrow_eq["band_qs"], dtype=float)
    broad_qs = np.asarray(broad_eq["band_qs"], dtype=float)

    narrow_idx = int(np.argmin(np.abs(narrow_centers - 2300.0)))
    broad_idx = int(np.argmin(np.abs(broad_centers - 2300.0)))

    assert abs(narrow_centers[narrow_idx] - 2300.0) < 220.0
    assert abs(broad_centers[broad_idx] - 2300.0) < 450.0
    assert narrow_qs[narrow_idx] > broad_qs[broad_idx] + 0.75


def test_18a_conservative_smoothing_limits_narrow_artifact_correction():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    target_db = get_target_curve(freqs, "flat")
    base = np.full_like(freqs, -70.0)
    narrow_notch = base - 10.0 * np.exp(
        -((log_freqs - np.log10(2300.0)) ** 2) / (2 * 0.015**2)
    )
    narrow_peak = base + 10.0 * np.exp(
        -((log_freqs - np.log10(2300.0)) ** 2) / (2 * 0.015**2)
    )

    notch_eq = calculate_eq_bands(
        freqs, narrow_notch, target_db, smoothing_strength="conservative"
    )
    peak_eq = calculate_eq_bands(
        freqs, narrow_peak, target_db, smoothing_strength="conservative"
    )
    notch_gains = np.asarray(notch_eq["band_gains"], dtype=float)
    peak_gains = np.asarray(peak_eq["band_gains"], dtype=float)

    assert np.max(np.abs(notch_gains)) <= 3.0
    assert np.max(np.abs(peak_gains)) <= 3.0
    assert (
        notch_eq["residual_regularization"]["max_regularized_correction_db"]
        < notch_eq["residual_regularization"]["max_requested_correction_db"] * 0.55
    )
    assert notch_eq["residual_regularization"]["max_narrow_residual_db"] > 6.0


def test_18b_target_modes_are_explicit_and_bounded():
    freqs = _default_freqs()
    measured = generate_test_spectrum(freqs, "bassy")
    static_target = get_target_curve(
        freqs,
        "podcast",
        measured_db=measured,
        target_mode="static",
    )
    catalog_target = get_target_curve(freqs, "podcast", target_mode="static")
    adaptive_target = get_target_curve(
        freqs,
        "podcast",
        measured_db=measured,
        target_mode="adaptive",
    )

    assert np.allclose(static_target, catalog_target)
    assert np.max(np.abs(adaptive_target)) <= 2.0 + 1e-9
    assert np.max(np.abs(adaptive_target - static_target)) > 0.25


def test_18c_broadcast_target_is_labelled_as_a_house_style_not_a_standard():
    broadcast = config.TARGET_CURVES["broadcast"]

    assert broadcast.name == "Broadcast-style Voice"
    assert "house curve" in broadcast.description.lower()
    assert "compliant" not in broadcast.description.lower()


def test_19_diagnostics_and_validation_are_present_and_valid():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "harsh")
    target_db = get_target_curve(freqs, "podcast", measured_db=spectrum_db)

    eq = calculate_eq_bands(freqs, spectrum_db, target_db)

    assert len(eq["band_confidences"]) == 10
    assert 0.0 <= eq["analysis_confidence"] <= 1.0
    assert 0.0 <= eq["eq_confidence"] <= 1.0
    assert 0.0 <= eq["capture_confidence"] <= 1.0
    assert 0.0 <= eq["validation_confidence"] <= 1.0
    assert eq["low_confidence_active_bands"] <= eq["active_band_count"] <= 10
    assert eq["validation_after_error_db"] <= eq["validation_before_error_db"] * 1.05
    assert 0.0 <= eq["validation_gain_scale"] <= 1.0
    assert eq["target_profile"]
    assert eq["smoothing_strength"] == "conservative"
    assert "max_regularized_correction_db" in eq["residual_regularization"]


def test_20_low_confidence_boosts_are_capped_aggressively():
    freqs = _default_freqs()
    spectrum_db = np.full_like(freqs, -70.0)
    log_freqs = np.log10(freqs)
    spectrum_db -= 12.0 * np.exp(-((log_freqs - np.log10(3000.0)) ** 2) / (2 * 0.04**2))
    target_db = get_target_curve(freqs, "flat")
    repeatability = np.full_like(freqs, 0.05)

    eq = calculate_eq_bands(
        freqs,
        spectrum_db,
        target_db,
        spectral_repeatability=repeatability,
        voiced_window_ratio=0.10,
        analysis_confidence=0.15,
    )

    gains = np.asarray(eq["band_gains"], dtype=float)
    assert np.max(gains) <= 2.0
    assert eq["capture_confidence"] <= 0.15
    assert eq["analysis_confidence"] >= eq["capture_confidence"]


def test_21_eq_quality_detects_risky_overlap_and_flat_is_safe():
    risky = evaluate_eq_quality(
        [80.0, 160.0, 300.0, 340.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0],
        [0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 4.5, 4.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    safe = evaluate_eq_quality(EQ_FREQUENCIES, [0.0] * 10, [1.41] * 10)

    assert risky.overlapping_adjacent_bands >= 1
    assert risky.warnings
    assert not safe.warnings


def test_21a_eq_quality_reports_positive_boost_and_cut_excursions_separately():
    boost_gains = [0.0] * 10
    boost_gains[4] = 6.0
    cut_gains = [0.0] * 10
    cut_gains[4] = -6.0
    mixed_gains = [0.0] * 10
    mixed_gains[3] = 6.0
    mixed_gains[6] = -6.0

    boost_only = evaluate_eq_quality(EQ_FREQUENCIES, boost_gains, [1.41] * 10)
    cut_only = evaluate_eq_quality(EQ_FREQUENCIES, cut_gains, [1.41] * 10)
    mixed = evaluate_eq_quality(
        EQ_FREQUENCIES,
        mixed_gains,
        [1.41] * 10,
    )
    flat = evaluate_eq_quality(EQ_FREQUENCIES, [0.0] * 10, [1.41] * 10)

    assert boost_only.max_boost_db > 0.0
    assert boost_only.max_cut_db <= 1e-6
    assert cut_only.max_boost_db <= 1e-6
    assert cut_only.max_cut_db > 0.0
    assert mixed.max_boost_db > 0.0
    assert mixed.max_cut_db > 0.0
    assert flat.max_boost_db == 0.0
    assert flat.max_cut_db == 0.0


def test_optimizer_threads_capture_rate_through_validation(monkeypatch):
    sample_rate = 44_100.0
    quality_rates = []
    error_rates = []
    real_quality = optimizer_module.evaluate_eq_quality
    real_error = optimizer_module.weighted_target_error

    def record_quality(*args, **kwargs):
        quality_rates.append(kwargs.get("sample_rate"))
        return real_quality(*args, **kwargs)

    def record_error(*args, **kwargs):
        error_rates.append(kwargs.get("sample_rate"))
        return real_error(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "evaluate_eq_quality", record_quality)
    monkeypatch.setattr(optimizer_module, "weighted_target_error", record_error)

    freqs = np.geomspace(20.0, 20_000.0, 128)
    calculate_eq_bands(
        freqs,
        0.4 * np.sin(np.log(freqs)),
        np.zeros_like(freqs),
        sample_rate=sample_rate,
    )

    assert quality_rates and set(quality_rates) == {sample_rate}
    assert error_rates and set(error_rates) == {sample_rate}


def test_22_stable_speech_like_capture_has_useful_confidence():
    sample_rate = 48_000
    duration_s = 10
    rng = np.random.default_rng(4201)
    t = np.arange(sample_rate * duration_s, dtype=float) / sample_rate
    f0 = 135.0 + 14.0 * np.sin(2.0 * np.pi * 0.55 * t)
    phase = np.cumsum(2.0 * np.pi * f0 / sample_rate)
    harmonic_voice = np.zeros_like(t)
    for harmonic in range(1, 20):
        harmonic_voice += np.sin(harmonic * phase) / harmonic
    syllables = 0.55 + 0.30 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.1 * t))
    audio = 0.045 * harmonic_voice * syllables + rng.normal(0.0, 0.001, t.size)

    eq, validation = analyze_auto_eq(audio.astype(np.float32), sample_rate, "broadcast")
    low_confidence_bands = eq["low_confidence_active_bands"]

    assert validation.passed
    assert eq["analysis_confidence"] >= 0.65
    assert eq["eq_confidence"] >= 0.60
    assert eq["capture_confidence"] >= 0.65
    assert low_confidence_bands <= 3


def test_23_predict_eq_response_uses_shelves_for_edge_bands():
    freqs = np.array([80.0, 1000.0, 16000.0, 20000.0], dtype=float)
    qs = [1.414] * 10

    low_gains = np.zeros(10)
    low_gains[0] = 6.0
    low_response = _predict_eq_response(freqs, low_gains, qs, EQ_FREQUENCIES)
    assert low_response[0] > low_response[1] + 2.0

    high_gains = np.zeros(10)
    high_gains[9] = 6.0
    high_response = _predict_eq_response(freqs, high_gains, qs, EQ_FREQUENCIES)
    assert high_response[3] > high_response[1] + 2.0


def test_24_fallback_analysis_reports_explicit_fallback_diagnostics():
    sample_rate = 48_000
    duration_s = 10
    t = np.arange(sample_rate * duration_s, dtype=float) / sample_rate
    audio = np.zeros_like(t)
    for start_s in (1.0, 4.0, 7.0):
        start = int(start_s * sample_rate)
        stop = start + int(0.18 * sample_rate)
        audio[start:stop] = 0.04 * np.sin(2.0 * np.pi * 180.0 * t[: stop - start])

    spectrum_result = analyze_voice_spectrum(audio.astype(np.float32), sample_rate)
    freqs = spectrum_result.freqs
    spectrum_smoothed = smooth_spectrum_perceptual(
        freqs, spectrum_result.median_spectrum_db
    )
    target_db = get_target_curve(freqs, "broadcast", measured_db=spectrum_smoothed)
    eq = calculate_eq_bands(
        freqs,
        spectrum_smoothed,
        target_db,
        spectral_repeatability=spectrum_result.spectral_repeatability,
        voiced_window_ratio=spectrum_result.voiced_window_ratio,
        analysis_confidence=spectrum_result.residual_confidence,
        global_snr_db=spectrum_result.snr_db,
        target_profile="broadcast:fallback",
        used_spectrum_fallback=spectrum_result.used_single_spectrum_fallback,
    )
    validation = validate_analysis(eq, spectrum_smoothed, freqs)

    assert not validation.passed
    assert eq["used_spectrum_fallback"]
    assert eq["target_profile"].endswith(":fallback")


def test_25_headroom_validation_reduces_boosts_when_peak_headroom_is_insufficient():
    sample_rate = 48_000
    duration_s = 1.0
    t = np.arange(int(sample_rate * duration_s), dtype=float) / sample_rate
    audio = (0.62 * np.sin(2.0 * np.pi * 5000.0 * t)).astype(np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        "band_qs": [1.41] * 10,
        "validation_gain_scale": 1.0,
        "validation_confidence": 0.95,
        "analysis_confidence": 0.95,
        "capture_confidence": 0.95,
        "band_confidences": [0.9] * 10,
        "recommendation_status": "apply",
        "validation_after_error_db": 999.0,
        "eq_quality": {"stale": True},
    }
    chain_settings = {
        "compressor": {"enabled": False},
        "deesser": {"enabled": False},
        "limiter": {
            "enabled": True,
            "ceiling_db": -0.5,
            "careful_output_enabled": True,
        },
    }

    analysis_freqs = np.geomspace(80.0, 8_000.0, 128)
    measured_db = np.zeros_like(analysis_freqs)
    target_db = np.full_like(analysis_freqs, 3.0)
    validated = apply_headroom_validation(
        audio,
        sample_rate,
        eq_settings,
        chain_settings,
        analysis_freqs=analysis_freqs,
        measured_db=measured_db,
        target_db=target_db,
    )

    assert validated["headroom_gain_scale"] < 1.0
    assert max(validated["band_gains"]) < 9.0
    assert validated["headroom_validation"]["safe"]
    assert validated["validation_after_error_db"] != 999.0
    assert "stale" not in validated["eq_quality"]
    assert validated["recommendation_status"] in {"reduced", "abstain"}
    assert (
        validated["headroom_validation"]["after"]["pre_limiter_true_peak_headroom_db"]
        >= 1.0
    )


def test_26_headroom_validation_preserves_safe_correction():
    sample_rate = 48_000
    t = np.arange(sample_rate, dtype=float) / sample_rate
    audio = (
        0.05 * np.sin(2.0 * np.pi * 180.0 * t) + 0.02 * np.sin(2.0 * np.pi * 1200.0 * t)
    ).astype(np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0, 0.0, 0.0, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        "band_qs": [1.41] * 10,
        "validation_gain_scale": 1.0,
        "validation_confidence": 0.90,
        "analysis_confidence": 0.90,
    }

    validated = apply_headroom_validation(audio, sample_rate, eq_settings)

    assert validated["headroom_gain_scale"] == 1.0
    assert np.allclose(validated["band_gains"], eq_settings["band_gains"])
    assert validated["headroom_validation"]["safe"]


def test_headroom_validation_keeps_safe_neutral_recommendation():
    sample_rate = 48_000
    t = np.arange(sample_rate, dtype=float) / sample_rate
    audio = (0.02 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "validation_gain_scale": 0.55,
        "recommendation_status": "apply",
        "apply_recommended": True,
    }

    validated = apply_headroom_validation(audio, sample_rate, eq_settings)

    assert validated["headroom_gain_scale"] == 1.0
    assert validated["recommendation_status"] == "apply"
    assert validated["apply_recommended"] is True


def test_headroom_validation_abstains_neutral_candidate_after_actual_attenuation():
    settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.1] + [0.0] * 9,
        "band_qs": [1.41] * 10,
        "validation_gain_scale": 0.55,
        "recommendation_status": "apply",
        "apply_recommended": True,
        "headroom_gain_scale": 0.25,
    }

    headroom_module._refresh_scaled_eq_metadata(
        settings,
        analysis_freqs=None,
        measured_db=None,
        target_db=None,
    )

    assert settings["recommendation_status"] == "abstain"
    assert settings["apply_recommended"] is False


def test_headroom_metadata_threads_capture_rate(monkeypatch):
    sample_rate = 44_100.0
    quality_rates = []
    error_rates = []
    real_quality = headroom_module.evaluate_eq_quality
    real_error = headroom_module.weighted_target_error

    def record_quality(*args, **kwargs):
        quality_rates.append(kwargs.get("sample_rate"))
        return real_quality(*args, **kwargs)

    def record_error(*args, **kwargs):
        error_rates.append(kwargs.get("sample_rate"))
        return real_error(*args, **kwargs)

    monkeypatch.setattr(headroom_module, "evaluate_eq_quality", record_quality)
    monkeypatch.setattr(headroom_module, "weighted_target_error", record_error)

    settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [2.0] + [0.0] * 9,
        "band_qs": [1.41] * 10,
    }
    freqs = np.geomspace(20.0, 20_000.0, 128)
    headroom_module._refresh_scaled_eq_metadata(
        settings,
        analysis_freqs=freqs,
        measured_db=np.zeros_like(freqs),
        target_db=np.zeros_like(freqs),
        sample_rate=sample_rate,
    )

    assert quality_rates and set(quality_rates) == {sample_rate}
    assert error_rates and set(error_rates) == {sample_rate}


def test_headroom_zero_scale_clears_stale_apply_metadata():
    sample_rate = 48_000
    audio = np.ones(sample_rate // 4, dtype=np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [6.0] * 10,
        "band_qs": [1.41] * 10,
        "band_confidences": [0.95] * 10,
        "validation_gain_scale": 1.0,
        "validation_confidence": 0.95,
        "analysis_confidence": 0.95,
        "capture_confidence": 0.95,
        "recommendation_status": "apply",
        "apply_recommended": True,
    }
    freqs = np.geomspace(80.0, 8_000.0, 128)

    validated = apply_headroom_validation(
        audio,
        sample_rate,
        eq_settings,
        {
            "compressor": {"enabled": False},
            "deesser": {"enabled": False},
            "limiter": {"enabled": True, "ceiling_db": -20.0},
        },
        analysis_freqs=freqs,
        measured_db=np.zeros_like(freqs),
        target_db=np.full_like(freqs, 3.0),
    )

    assert validated["headroom_gain_scale"] == 0.0
    assert validated["band_gains"] == [0.0] * 10
    assert validated["active_band_count"] == 0
    assert validated["recommendation_status"] == "abstain"
    assert validated["apply_recommended"] is False


def test_headroom_nonzero_scale_recomputes_active_band_threshold(monkeypatch):
    def fake_simulation(_audio, _sample_rate, settings, _chain_settings):
        maximum_gain = max(abs(value) for value in settings["band_gains"])
        return {
            "simulation_backend": "rust",
            "pre_limiter_true_peak_headroom_db": 2.0 if maximum_gain <= 0.2 else 0.0,
            "limiter_gain_reduction_db": 0.0,
            "true_peak_limiter_gain_reduction_db": 0.0,
        }

    monkeypatch.setattr(headroom_module, "simulate_candidate_chain", fake_simulation)
    settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.8] + [0.0] * 9,
        "band_qs": [1.41] * 10,
        "band_confidences": [0.9] * 10,
        "validation_gain_scale": 1.0,
        "recommendation_status": "apply",
        "apply_recommended": True,
    }

    validated = apply_headroom_validation(
        np.zeros(480, dtype=np.float32),
        48_000,
        settings,
    )

    assert validated["headroom_gain_scale"] == 0.25
    assert validated["band_gains"][0] == pytest.approx(0.2)
    assert validated["active_band_count"] == 0
    assert validated["recommendation_status"] == "abstain"
    assert validated["apply_recommended"] is False


def test_27_validation_rejects_remaining_headroom_risk():
    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    freqs = np.fft.rfftfreq(time.size, 1.0 / sample_rate)
    audio = sum(
        0.1 / harmonic * np.sin(2 * np.pi * 150 * harmonic * time)
        for harmonic in range(1, 26)
    )
    spectrum_db = 10.0 * np.log10(
        np.maximum(np.square(np.abs(np.fft.rfft(audio))), 1.0e-12)
    )
    eq_settings = {
        "band_gains": [0.0] * 10,
        "headroom_validation": {"safe": False},
    }

    validation = validate_analysis(eq_settings, spectrum_db, freqs)

    assert not validation.passed
    # The lever is the gain; with no headroom figures there is no number to give.
    assert validation.reason is not None
    assert "Lower the gain" in validation.reason
    assert not re.search(r"\d+ dB", validation.reason)
    assert validation.details["headroom_safe"] is False


def _harmonic_spectrum():
    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    freqs = np.fft.rfftfreq(time.size, 1.0 / sample_rate)
    audio = sum(
        0.1 / harmonic * np.sin(2 * np.pi * 150 * harmonic * time)
        for harmonic in range(1, 26)
    )
    spectrum_db = 10.0 * np.log10(
        np.maximum(np.square(np.abs(np.fft.rfft(audio))), 1.0e-12)
    )
    return spectrum_db, freqs


def _render(headroom, limiter=0.0, true_peak=0.0):
    return {
        "pre_limiter_true_peak_headroom_db": headroom,
        "limiter_gain_reduction_db": limiter,
        "true_peak_limiter_gain_reduction_db": true_peak,
    }


@pytest.mark.parametrize(
    ("full_correction", "advice"),
    [
        # What lets the full correction through: the excess at scale 1.0, rounded up.
        (_render(0.96), "Lower the gain by about 1 dB, then record again."),
        (_render(-1.3), "Lower the gain by about 3 dB, then record again."),
        (_render(-2.0), "Lower the gain by about 3 dB, then record again."),
        (_render(0.5, limiter=3.4), "Lower the gain by about 3 dB, then record again."),
    ],
)
def test_headroom_refusal_names_the_level_and_the_number(full_correction, advice):
    spectrum_db, freqs = _harmonic_spectrum()

    validation = validate_analysis(
        {
            "band_gains": [0.0] * 10,
            "headroom_validation": {
                "safe": False,
                "authoritative": True,
                "status": "risk",
                "before": full_correction,
                "reference_excess_db": 0.2,  # the reference alone would have said 1 dB
            },
        },
        spectrum_db,
        freqs,
    )

    assert not validation.passed
    assert validation.reason == (
        "The microphone level is too high for a safe result. " + advice
    )


def test_headroom_records_the_excess_of_the_recording_alone_when_a_gentler_scale_was_needed(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=3.0)

    validation = _validate_typed(BOOST_HEAVY)["headroom_validation"]

    # Safe at 0.25 after six renders; the recording with no correction is one more.
    assert validation["status"] == "safe" and validation["gain_scale"] == 0.25
    assert len(calls) == 7
    assert validation["flat_excess_db"] == pytest.approx(-0.5)  # the true-peak term, 0.0 - 0.5
    assert validation["flat_excess_db"] == pytest.approx(_excess(calls[-1]))


def test_headroom_refusal_cuts_only_the_recordings_part_when_the_correction_adds_most():
    spectrum_db, freqs = _harmonic_spectrum()

    def refuse(flat_excess_db):
        return validate_analysis(
            {
                "band_gains": [0.0] * 10,
                "headroom_validation": {
                    "safe": False,
                    "authoritative": True,
                    "status": "risk",
                    "before": _render(-2.0),  # the full correction is 3.0 dB over
                    "flat_excess_db": flat_excess_db,
                },
            },
            spectrum_db,
            freqs,
        ).reason

    prefix = "The microphone level is too high for a safe result. "
    # 0.4 of the 3.0 dB is the recording's; the correction adds the other 2.6.
    assert refuse(0.4) == prefix + "Lower the gain by about 1 dB, then record again."
    # The recording is most of it: the figure that lets the full correction through stays.
    assert refuse(2.5) == prefix + "Lower the gain by about 3 dB, then record again."


@pytest.mark.parametrize("status", ["risk_not_worsened", "risk_worse"])
def test_headroom_that_the_dialog_explains_is_not_a_refusal(status):
    spectrum_db, freqs = _harmonic_spectrum()

    validation = validate_analysis(
        {
            "band_gains": [0.0] * 10,
            "headroom_validation": {
                "safe": False,
                "authoritative": True,
                "status": status,
                "before": _render(-0.2),
                "reference_excess_db": 1.2,
            },
        },
        spectrum_db,
        freqs,
    )

    assert all("headroom" not in reason for reason in validation.details["failures"])
    assert validation.details["headroom_safe"] is False


def _voice_spectrum():
    """A capture that passes validation on its own."""
    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    audio = np.asarray(
        sum(0.1 / h * np.sin(2 * np.pi * 150 * h * time) for h in range(1, 26)), dtype=np.float32
    )
    spectrum = analyze_voice_spectrum(audio, sample_rate)
    return spectrum.median_spectrum_db, spectrum.freqs


@pytest.mark.parametrize(
    ("gains", "headroom", "tone_only"),
    [
        ([12.0] * 6 + [0.0] * 4, {"safe": True}, True),  # clipped_gains >= 6
        ([10.8] * 10, {"safe": True}, True),  # gain_rms > 10 dB
        ([12.0] * 5 + [0.0] * 5, {"safe": True}, True),  # two soft tone failures together
        ([12.0] * 6 + [0.0] * 4, {"safe": False, "authoritative": True, "status": "risk"}, False),
        ([0.0] * 10, {"safe": False, "authoritative": True, "status": "risk"}, False),
    ],
    ids=["clipped", "rms", "soft-pair", "tone-and-headroom", "headroom-only"],
)
def test_validation_says_whether_only_the_tone_fit_failed(gains, headroom, tone_only):
    spectrum_db, freqs = _voice_spectrum()

    validation = validate_analysis(
        {"band_gains": gains, "headroom_validation": headroom}, spectrum_db, freqs
    )

    assert not validation.passed
    assert validation.details["tone_only"] is tone_only


def test_an_unclear_capture_is_not_a_tone_only_failure():
    freqs = np.linspace(0.0, 24_000.0, 4097)

    validation = validate_analysis(
        {"band_gains": [12.0] * 6 + [0.0] * 4}, np.zeros_like(freqs), freqs
    )

    assert not validation.passed
    assert validation.details["tone_only"] is False


@pytest.mark.parametrize(("gains", "tone_only"), [([12.0] * 6 + [0.0] * 4, True), ([0.0] * 10, False)])
def test_the_pipeline_raises_a_value_error_that_carries_the_classification(
    monkeypatch, gains, tone_only
):
    from mic_eq.analysis.auto_eq_parts import pipeline
    from mic_eq.analysis.failure_detection import AnalysisValidationError

    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    audio = np.asarray(
        sum(0.1 / h * np.sin(2 * np.pi * 150 * h * time) for h in range(1, 26)), dtype=np.float32
    )
    monkeypatch.setattr(
        pipeline,
        "apply_headroom_validation",
        lambda _audio, _rate, eq, **_kwargs: {
            **eq,
            "band_gains": gains,
            "headroom_validation": {"safe": True} if tone_only else {"safe": False, "status": "risk"},
        },
    )

    with pytest.raises(ValueError, match="unclear|microphone level") as raised:
        analyze_auto_eq(
            audio, sample_rate, "flat",
            chain_settings={"compressor_enabled": False, "limiter_enabled": False},
        )

    assert isinstance(raised.value, AnalysisValidationError)
    assert raised.value.tone_only is tone_only
    assert raised.value.failures == (
        ["clipped_gains (6 >= 6)"] if tone_only else ["headroom risk after downstream simulation"]
    )


def test_unavailable_headroom_is_unknown_not_a_capture_quality_failure():
    sample_rate = 48_000
    time = np.arange(sample_rate * 3) / sample_rate
    freqs = np.fft.rfftfreq(time.size, 1.0 / sample_rate)
    audio = sum(
        0.1 / harmonic * np.sin(2 * np.pi * 150 * harmonic * time)
        for harmonic in range(1, 26)
    )
    spectrum_db = 10.0 * np.log10(
        np.maximum(np.square(np.abs(np.fft.rfft(audio))), 1.0e-12)
    )

    validation = validate_analysis(
        {
            "band_gains": [0.0] * 10,
            "headroom_validation": {
                "safe": False,
                "authoritative": False,
                "status": "unavailable",
                "reason": "native simulator unavailable",
            },
        },
        spectrum_db,
        freqs,
    )

    assert all("headroom" not in reason.lower() for reason in validation.details["failures"])
    assert validation.details["headroom_safe"] is False


def test_28_unavailable_headroom_does_not_fabricate_metrics(monkeypatch):
    monkeypatch.setattr(
        headroom_module,
        "_native_simulate",
        lambda *_args, **_kwargs: (
            None,
            {"kind": "unavailable", "message": "native simulator unavailable"},
        ),
    )
    audio = np.zeros(4096, dtype=np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "validation_confidence": 0.95,
        "analysis_confidence": 0.95,
    }

    with pytest.raises(RuntimeError, match="native simulator unavailable"):
        apply_headroom_validation(
            audio,
            48_000,
            eq_settings,
            {
                "compressor": {"enabled": False},
                "deesser": {"enabled": False},
                "limiter": {"enabled": True, "careful_output_enabled": True},
            },
        )


def test_28b_native_headroom_failure_preserves_a_bounded_cause(monkeypatch):
    monkeypatch.setattr(
        headroom_module,
        "_native_simulate",
        lambda *_args, **_kwargs: (
            None,
            {"kind": "runtime_error", "message": "native simulator failed: test"},
        ),
    )
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
    }

    with pytest.raises(RuntimeError, match="native simulator failed: test"):
        apply_headroom_validation(
            np.zeros(4096, dtype=np.float32),
            48_000,
            eq_settings,
            {"compressor": {"enabled": False}, "deesser": {"enabled": False}},
        )


@pytest.mark.parametrize(
    ("failure_mode", "expected_kind"),
    [
        ("unavailable", "unavailable"),
        ("runtime", "runtime_error"),
        ("invalid", "invalid_result"),
    ],
)
def test_native_headroom_failure_classification_is_explicit(
    monkeypatch, failure_mode, expected_kind
):
    if failure_mode == "unavailable":
        monkeypatch.setattr(mic_eq, "CORE_AVAILABLE", False)
    elif failure_mode == "runtime":
        monkeypatch.setattr(
            mic_eq_core,
            "simulate_auto_eq_chain",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    else:
        monkeypatch.setattr(
            mic_eq_core,
            "simulate_auto_eq_chain",
            lambda *_args, **_kwargs: {"output_true_peak_db": -3.0},
        )

    _result, failure = headroom_module._native_simulate(
        np.zeros(64, dtype=np.float32),
        48_000,
        [(1000.0, 0.0, 1.41)] * 10,
        {},
    )

    assert failure is not None
    assert failure["kind"] == expected_kind


def test_headroom_safety_fails_closed_for_missing_metrics():
    assert headroom_module._is_headroom_safe({}) is False


def test_29_native_failure_cannot_report_risky_capture_as_safe(monkeypatch):
    monkeypatch.setattr(
        headroom_module,
        "_native_simulate",
        lambda *_args, **_kwargs: (
            None,
            {"kind": "unavailable", "message": "native simulator unavailable"},
        ),
    )
    t = np.arange(48_000, dtype=float) / 48_000.0
    audio = (0.95 * np.sin(2.0 * np.pi * 5000.0 * t)).astype(np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [9.0] * 10,
        "band_qs": [1.41] * 10,
    }

    with pytest.raises(RuntimeError, match="native simulator unavailable"):
        apply_headroom_validation(
            audio,
            48_000,
            eq_settings,
            {
                "compressor": {"enabled": False},
                "deesser": {"enabled": False},
                "limiter": {"enabled": True, "careful_output_enabled": True},
            },
        )


def test_voice_safe_flat_target_preserves_broad_speech_shape():
    freqs = _default_freqs()
    measured = -70.0 - 5.0 * np.log2(freqs / 1000.0)
    result = calculate_eq_bands(
        freqs,
        measured,
        np.zeros_like(freqs),
        tilt_policy="voice_safe",
    )

    assert result["spectral_tilt_policy"] == "voice_safe"
    assert np.max(np.abs(result["band_gains"])) < 0.1
    assert result["validation_before_error_db"] == pytest.approx(0.0)


def test_normalization_is_level_and_input_grid_stable():
    gains = []
    for point_count, level in ((128, -70.0), (1000, -90.0)):
        freqs = np.logspace(np.log10(20.0), np.log10(20_000.0), point_count)
        measured = level - 5.0 * np.log2(freqs / 1000.0)
        target = 0.7 * np.exp(
            -((np.log10(freqs) - np.log10(2500.0)) ** 2) / (2 * 0.25**2)
        )
        result = calculate_eq_bands(
            freqs,
            measured,
            target,
            tilt_policy="voice_safe",
        )
        gains.append(np.asarray(result["band_gains"], dtype=float))

    np.testing.assert_allclose(gains[0], gains[1], atol=0.01)


def test_auto_eq_pipeline_rejects_non_live_sample_rate():
    with pytest.raises(ValueError, match="48000"):
        analyze_auto_eq(np.zeros(480, dtype=np.float32), 44_100, "flat")


_FULL_CHAIN = {"full_chain": True, "input_pre_filtered": False}
_OLD_HEADROOM_KEYS = {
    "safe", "authoritative", "advisory", "meets_advisory_thresholds",
    "gain_scale", "before", "after", "status",
}
BOOST_HEAVY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0]
CUT_HEAVY = [-6.0, -5.0, -4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _fake_chain(
    monkeypatch, *, headroom, limiter=0.0, true_peak=0.0, free_db=0.0, backend: str | None = "rust"
):
    """Fake the native render: the peak follows the correction's largest boost or deepest cut.

    ``free_db`` is how much boost leaves the peak alone (the peak sits elsewhere in the
    spectrum). The limiter works, by ``limiter`` dB at the reference, only when that is given.
    """
    calls = []

    def simulate(_audio, _rate, eq, _chain=None):
        gains = [b["gain_db"] for b in (eq.get("layers") or {}).get("correction") or ()]
        boost, cut = max(gains, default=0.0), min(gains, default=0.0)
        change = 0.6 * max(0.0, boost - free_db) if boost > 0.0 else 0.6 * cut
        render = {
            "pre_limiter_true_peak_headroom_db": headroom - change,
            "limiter_gain_reduction_db": max(0.0, limiter + change) if limiter else 0.0,
            "true_peak_limiter_gain_reduction_db": true_peak,
            "output_true_peak_db": change - headroom,
        }
        if backend:
            render["simulation_backend"] = backend
        calls.append(render)
        return render

    monkeypatch.setattr(headroom_module, "simulate_candidate_chain", simulate)
    return calls


def _excess(render):
    # The released rule, spelled out: 1 dB of headroom, 1 dB of limiter, 0.5 dB of true-peak limiter.
    return max(
        1.0 - render["pre_limiter_true_peak_headroom_db"],
        render["limiter_gain_reduction_db"] - 1.0,
        render["true_peak_limiter_gain_reduction_db"] - 0.5,
    )


def _candidate_fit(gains):
    return {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": list(gains),
        "band_qs": [1.41] * 10,
        "band_confidences": [0.9] * 10,
        "validation_gain_scale": 1.0,
        "validation_confidence": 0.9,
        "analysis_confidence": 0.9,
        "recommendation_status": "apply",
        "apply_recommended": True,
    }


def _validate_typed(gains):
    return apply_headroom_validation(
        np.zeros(480, dtype=np.float32),
        48_000,
        _candidate_fit(gains),
        _FULL_CHAIN,
        candidate_base_eq_settings=config.EQSettings().to_dict(),
    )


@pytest.mark.parametrize(
    ("gains", "scale", "renders", "keys"),
    [
        # A scale under 0.70 also renders the recording with no correction, for the advice.
        (BOOST_HEAVY, 0.55, 5, _OLD_HEADROOM_KEYS | {"flat_excess_db"}),
        (CUT_HEAVY, 1.0, 1, _OLD_HEADROOM_KEYS),
    ],
)
def test_headroom_with_a_safe_reference_keeps_the_released_decision(
    monkeypatch, gains, scale, renders, keys
):
    calls = _fake_chain(monkeypatch, headroom=4.0)

    result = _validate_typed(gains)
    assert len(calls) == renders  # the current EQ is rendered only when no scale meets the rule

    validation = result["headroom_validation"]
    assert validation["status"] == "safe"
    assert validation["safe"] is True
    assert set(validation) == keys
    assert validation["gain_scale"] == scale
    assert result["headroom_safe"] is True
    assert result["validated_candidate_eq"] is not None


def test_a_failed_render_for_the_advice_only_leaves_out_the_split(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=4.0)
    expected = _validate_typed(BOOST_HEAVY)
    assert expected["headroom_validation"]["flat_excess_db"] == pytest.approx(-0.5)
    expected["headroom_validation"].pop("flat_excess_db")
    render = headroom_module.simulate_candidate_chain

    def simulate(audio, rate, eq, chain=None):
        gains = [b["gain_db"] for b in (eq.get("layers") or {}).get("correction") or ()]
        if not any(gains):  # only the render with no correction at all
            raise RuntimeError("flat render failed")
        return render(audio, rate, eq, chain)

    monkeypatch.setattr(headroom_module, "simulate_candidate_chain", simulate)

    result = _validate_typed(BOOST_HEAVY)

    assert result == expected
    assert result["validated_candidate_eq"] is not None
    assert "flat_excess_db" not in result["headroom_validation"]
    assert len(calls) == 5 + 4  # the second run renders the four scales only


def test_headroom_over_the_rule_keeps_a_candidate_that_adds_no_excess(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=0.4, free_db=3.0)

    result = _validate_typed(BOOST_HEAVY)

    validation = result["headroom_validation"]
    assert len(calls) == 8  # every scale once, then the current EQ as it is
    reference = calls[6]
    assert validation["reference_excess_db"] == pytest.approx(0.6)
    assert validation["flat_excess_db"] == pytest.approx(0.6)
    assert validation["incumbent_excess_db"] == pytest.approx(0.6)
    assert validation["status"] == "risk_not_worsened"
    assert validation["safe"] is False
    assert validation["meets_advisory_thresholds"] is False
    assert validation["authoritative"] is True
    assert validation["advisory"] is False
    assert result["headroom_safe"] is False
    # The 9 dB boost still lifts the peak at 0.40; at 0.25 it sits under the free 3 dB.
    assert validation["gain_scale"] == 0.25
    assert result["band_gains"][6] == pytest.approx(2.25)
    after = validation["after"]
    assert after is calls[5]
    assert _excess(after) <= _excess(reference)
    assert after["pre_limiter_true_peak_headroom_db"] >= reference["pre_limiter_true_peak_headroom_db"]
    assert after["limiter_gain_reduction_db"] <= reference["limiter_gain_reduction_db"]
    assert result["validated_candidate_eq"] is not None
    assert result["apply_recommended"] is True
    # Kept below full strength: the usual cap for a reduced correction, not the unsafe one.
    assert result["validation_confidence"] == pytest.approx(0.72)
    assert result["analysis_confidence"] == pytest.approx(0.9)


def test_headroom_over_the_rule_accepts_a_cut_only_candidate_whole(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=-0.5)
    gains = [-1.0, -0.8, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = _validate_typed(gains)

    validation = result["headroom_validation"]
    assert validation["gain_scale"] == 1.0
    assert validation["status"] == "risk_not_worsened"
    assert validation["safe"] is False
    assert _excess(validation["after"]) < _excess(calls[6])
    assert result["band_gains"] == gains
    assert result["validated_candidate_eq"] is not None
    assert result["recommendation_status"] == "apply"
    assert (result["validation_confidence"], result["analysis_confidence"]) == (0.9, 0.9)


@pytest.mark.parametrize(
    ("gains", "scale"),
    [(BOOST_HEAVY, 0.25), ([0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0)],
)
def test_headroom_with_the_limiter_working_two_decibels_at_the_reference(
    monkeypatch, gains, scale
):
    calls = _fake_chain(monkeypatch, headroom=-2.0, limiter=2.0, true_peak=0.2, free_db=3.0)

    result = _validate_typed(gains)

    validation = result["headroom_validation"]
    reference = calls[6]
    assert validation["reference_excess_db"] == pytest.approx(3.0)
    assert validation["gain_scale"] == scale
    assert validation["status"] == "risk_not_worsened"
    after = validation["after"]
    assert _excess(after) <= _excess(reference)
    assert after["limiter_gain_reduction_db"] <= reference["limiter_gain_reduction_db"]
    assert after["pre_limiter_true_peak_headroom_db"] >= reference["pre_limiter_true_peak_headroom_db"]


def test_headroom_over_the_rule_falls_back_to_no_correction_when_every_scale_adds_excess(
    monkeypatch,
):
    _fake_chain(monkeypatch, headroom=0.4)

    result = _validate_typed(BOOST_HEAVY)

    assert result["headroom_validation"]["gain_scale"] == 0.0
    assert result["headroom_validation"]["status"] == "risk_not_worsened"
    assert result["band_gains"] == [0.0] * 10
    assert result["recommendation_status"] == "abstain"
    assert result["apply_recommended"] is False
    assert result["validated_candidate_eq"] is not None


def _base_with_correction(gains):
    return config.build_eq_candidate_settings(
        config.EQSettings(), EQ_FREQUENCIES, gains, [1.41] * 10, layer="correction", enabled=True
    ).to_dict()


def test_headroom_accepts_what_leaves_no_more_excess_than_a_current_eq_that_is_over_the_rule(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=0.4)
    base = _base_with_correction(BOOST_HEAVY)  # the user's EQ today lifts the peak 5.4 dB
    gains = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]  # this one lifts it 2.4 dB

    result = apply_headroom_validation(
        np.zeros(480, dtype=np.float32), 48_000, _candidate_fit(gains), _FULL_CHAIN,
        candidate_base_eq_settings=base,
    )

    validation = result["headroom_validation"]
    assert len(calls) == 8
    assert validation["flat_excess_db"] == pytest.approx(0.6)
    assert validation["incumbent_excess_db"] == pytest.approx(6.0)
    assert validation["reference_excess_db"] == pytest.approx(6.0)
    assert validation["incumbent_output_true_peak_db"] == pytest.approx(5.0)
    assert validation["status"] == "risk_not_worsened"
    # Worse than no correction (3.0 against 0.6), better than what the user runs (6.0).
    assert validation["reference_excess_db"] == pytest.approx(6.0)
    assert validation["gain_scale"] == 1.0
    assert _excess(validation["after"]) == pytest.approx(3.0)
    assert headroom_module.headroom_shortfall_db(validation) == 3
    assert result["validated_candidate_eq"] is not None


@pytest.mark.parametrize(
    ("headroom", "free_db", "incumbent_excess", "flat_excess"),
    [
        # The current EQ meets the rule and no correction does not: nothing may get in.
        (0.4, 3.0, -0.5, 0.6),
        # The current EQ is over the rule, by less than no correction is.
        (-3.0, 0.0, 0.4, 4.0),
    ],
)
def test_headroom_leaves_a_current_eq_that_no_scale_matches_alone(
    monkeypatch, headroom, free_db, incumbent_excess, flat_excess
):
    calls = _fake_chain(monkeypatch, headroom=headroom, free_db=free_db)
    base = _base_with_correction(CUT_HEAVY)  # the user's EQ today drops the peak 3.6 dB

    result = apply_headroom_validation(
        np.zeros(480, dtype=np.float32), 48_000, _candidate_fit(BOOST_HEAVY), _FULL_CHAIN,
        candidate_base_eq_settings=base,
    )

    validation = result["headroom_validation"]
    assert len(calls) == 8
    assert validation["incumbent_excess_db"] == pytest.approx(incumbent_excess)
    assert validation["flat_excess_db"] == pytest.approx(flat_excess)
    assert validation["reference_excess_db"] == pytest.approx(incumbent_excess)
    assert validation["status"] == "risk_worse"
    assert validation["safe"] is False and validation["authoritative"] is True
    assert result["validated_candidate_eq"] is None
    assert result["apply_recommended"] is False
    assert result["band_gains"] == [0.0] * 10
    assert (result["validation_confidence"], result["analysis_confidence"]) == (0.42, 0.58)


def test_headroom_checks_for_cancellation_before_rendering_the_current_eq(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=0.4)
    asked = []

    def cancel():
        asked.append(1)
        return len(asked) == 8  # the question that comes after the seven scales

    with pytest.raises(AnalysisCancelled):
        apply_headroom_validation(
            np.zeros(480, dtype=np.float32), 48_000, _candidate_fit(BOOST_HEAVY), _FULL_CHAIN,
            candidate_base_eq_settings=config.EQSettings().to_dict(), cancel_check=cancel,
        )

    assert len(calls) == 7


def test_a_current_eq_render_with_unreadable_metrics_offers_nothing(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=0.4)
    render = headroom_module.simulate_candidate_chain

    def simulate(audio, rate, eq, chain=None):
        result = render(audio, rate, eq, chain)
        if len(calls) == 8:  # the current EQ is the eighth render
            result["pre_limiter_true_peak_headroom_db"] = float("nan")
        return result

    monkeypatch.setattr(headroom_module, "simulate_candidate_chain", simulate)

    result = _validate_typed(BOOST_HEAVY)

    assert result["headroom_validation"]["status"] == "risk"
    assert result["validated_candidate_eq"] is None
    assert result["apply_recommended"] is False


def test_a_failed_render_of_the_current_eq_is_unavailable_not_an_error(monkeypatch):
    calls = _fake_chain(monkeypatch, headroom=0.4)
    render = headroom_module.simulate_candidate_chain

    def simulate(audio, rate, eq, chain=None):
        if len(calls) == 7:  # the current EQ is the eighth render
            raise RuntimeError("native simulator failed: test")
        return render(audio, rate, eq, chain)

    monkeypatch.setattr(headroom_module, "simulate_candidate_chain", simulate)

    result = _validate_typed(BOOST_HEAVY)

    validation = result["headroom_validation"]
    assert validation["status"] == "unavailable"
    assert "native simulator failed: test" in validation["reason"]
    assert result["validated_candidate_eq"] is None
    assert result["apply_recommended"] is False


def test_headroom_takes_the_typed_fields_of_a_payload_shaped_like_the_eq_state():
    # EQState.get_settings(): the typed payload plus the legacy arrays.
    state_payload = config.EQSettings().to_dict() | {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
    }

    typed = headroom_module._typed_eq_payload(state_payload)

    assert set(typed) == {"schema_version", "enabled", "bands", "layers"} & set(state_payload)
    assert config.EQSettings.from_dict(typed).to_dict() == config.EQSettings().to_dict()


def test_headroom_accepts_the_eq_state_payload_as_the_incumbent(monkeypatch):
    _fake_chain(monkeypatch, headroom=4.0)
    state_payload = config.EQSettings().to_dict() | {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
    }

    result = apply_headroom_validation(
        np.zeros(480, dtype=np.float32), 48_000, _candidate_fit(CUT_HEAVY), _FULL_CHAIN,
        candidate_base_eq_settings=state_payload,
    )

    assert result["headroom_validation"]["status"] == "safe"
    assert result["validated_candidate_eq"] is not None


def test_headroom_still_takes_a_legacy_only_incumbent(monkeypatch):
    _fake_chain(monkeypatch, headroom=4.0)

    result = apply_headroom_validation(
        np.zeros(480, dtype=np.float32), 48_000, _candidate_fit(CUT_HEAVY), _FULL_CHAIN,
        candidate_base_eq_settings={
            "band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10,
        },
    )

    assert result["headroom_validation"]["status"] == "safe"
    assert result["validated_candidate_eq"] is not None


def test_headroom_advisory_render_is_not_accepted_over_the_rule(monkeypatch):
    _fake_chain(monkeypatch, headroom=0.4, free_db=3.0, backend=None)

    result = _validate_typed(BOOST_HEAVY)

    validation = result["headroom_validation"]
    assert validation["status"] == "advisory"
    assert validation["authoritative"] is False
    assert validation["advisory"] is True
    assert validation["safe"] is False
    assert result["headroom_safe"] is False
    assert result["validated_candidate_eq"] is None
    assert "no safe candidate remained after full-chain headroom validation" in (
        result["abstention_reasons"]
    )


def test_headroom_without_a_candidate_base_still_refuses_and_records_the_excess(monkeypatch):
    _fake_chain(monkeypatch, headroom=0.4, free_db=3.0)

    result = apply_headroom_validation(
        np.zeros(480, dtype=np.float32),
        48_000,
        _candidate_fit(BOOST_HEAVY),
        {"compressor": {"enabled": False}, "limiter": {"enabled": True}},
    )

    validation = result["headroom_validation"]
    assert validation["status"] == "risk"
    assert validation["safe"] is False
    assert validation["gain_scale"] == 0.0
    assert validation["reference_excess_db"] == pytest.approx(0.6)
    assert "validated_candidate_eq" not in result


def _speech_like(seconds=10):
    rng = np.random.default_rng(4201)
    t = np.arange(48_000 * seconds, dtype=float) / 48_000
    phase = np.cumsum(2.0 * np.pi * (135.0 + 14.0 * np.sin(2.0 * np.pi * 0.55 * t)) / 48_000)
    voice = sum(np.sin(h * phase) / h for h in range(1, 20))
    syllables = 0.55 + 0.30 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.1 * t))
    return (0.045 * voice * syllables + rng.normal(0.0, 0.001, t.size)).astype(np.float32)


def test_full_voice_setup_call_is_still_refused_with_the_level_to_lower(monkeypatch):
    _fake_chain(monkeypatch, headroom=0.4)

    with pytest.raises(ValueError) as refused:
        analyze_auto_eq(
            _speech_like(),
            48_000,
            "broadcast",
            chain_settings={
                "deesser": {"enabled": True},
                "compressor": {"enabled": False},
                "limiter": {"enabled": True},
            },
        )

    assert str(refused.value) == (
        "The microphone level is too high for a safe result. "
        "Lower the gain by about 1 dB, then record again."
    )


def test_eq_only_call_keeps_a_proposal_when_the_current_settings_are_over_the_rule(monkeypatch):
    _fake_chain(monkeypatch, headroom=-1.4, limiter=1.9)
    audio = _speech_like()

    eq, validation = analyze_auto_eq(
        audio,
        48_000,
        "broadcast",
        chain_settings=_FULL_CHAIN,
        headroom_candidate_base_eq_settings=config.EQSettings().to_dict(),
        headroom_audio_data=audio,
        headroom_chain_settings=_FULL_CHAIN,
    )

    assert validation.passed
    assert validation.details["headroom_safe"] is False
    assert eq["headroom_validation"]["status"] == "risk_not_worsened"
    assert eq["headroom_validation"]["reference_excess_db"] == pytest.approx(2.4)
    assert eq["validated_candidate_eq"] is not None


def test_eq_only_call_returns_a_correction_left_out_instead_of_refusing(monkeypatch):
    _fake_chain(monkeypatch, headroom=0.4, free_db=3.0)  # no correction is under the rule
    audio = _speech_like()

    eq, validation = analyze_auto_eq(
        audio, 48_000, "broadcast",
        chain_settings=_FULL_CHAIN,
        headroom_candidate_base_eq_settings=_base_with_correction(CUT_HEAVY),  # the current EQ is
        headroom_audio_data=audio,
        headroom_chain_settings=_FULL_CHAIN,
    )

    assert validation.passed
    assert eq["headroom_validation"]["status"] == "risk_worse"
    assert eq["apply_recommended"] is False
    assert eq["validated_candidate_eq"] is None


# Real renders: the native simulator on a synthetic signal, no audio device. Its peak is set by
# a 320 Hz tone, so a cut there lowers it, and a fade-in keeps the start-up transient out of it.
needs_native = pytest.mark.skipif(not mic_eq.CORE_AVAILABLE, reason="native extension is not built")


def _tones(level_db=0.0):
    t = np.arange(48_000) / 48_000
    x = np.minimum(1.0, t / 0.2) * (
        0.7 * np.sin(2 * np.pi * 320 * t)
        + 0.12 * np.sin(2 * np.pi * 80 * t + 1.0)
        + 0.1 * np.sin(2 * np.pi * 1280 * t + 2.0)
    )
    return np.ascontiguousarray(x * 10 ** (level_db / 20.0), dtype=np.float32)


def _native_chain(**stages):
    return {
        "full_chain": True, "input_pre_filtered": True, "processing_mode": "normal",
        "vad_available": False, "gate": {"enabled": False}, "suppressor": {"enabled": False},
        "deesser": {"enabled": False}, "compressor": {"enabled": False},
        "limiter": {"enabled": False}, **stages,
    }


def _native_render(audio, eq, chain):
    return headroom_module.simulate_candidate_chain(audio, 48_000, eq, chain)


@needs_native
def test_real_render_a_current_eq_that_meets_the_rule_is_not_replaced_by_a_hotter_candidate():
    audio, chain = _tones(), _native_chain()
    incumbent = _base_with_correction([0.0, 0.0, -4.0] + [0.0] * 7)  # cuts 320 Hz, where the peak is
    candidate = [1.5] + [0.0] * 9  # +1.5 dB at 80 Hz
    # As the engine renders them: the current EQ meets the rule, no correction does not, and
    # neither does the candidate (it is better than no correction, which is the old bar).
    assert _excess(_native_render(audio, incumbent, chain)) < 0
    assert _excess(_native_render(audio, _base_with_correction([0.0] * 10), chain)) > 0
    assert _excess(_native_render(audio, _base_with_correction(candidate), chain)) > 0

    result = apply_headroom_validation(
        audio, 48_000, _candidate_fit(candidate), chain, candidate_base_eq_settings=incumbent
    )

    validation = result["headroom_validation"]
    assert validation["status"] == "risk_worse"
    assert validation["incumbent_excess_db"] < 0 < validation["flat_excess_db"]
    assert result["validated_candidate_eq"] is None
    assert result["apply_recommended"] is False


@needs_native
@pytest.mark.parametrize(
    ("gains", "whole"),
    [
        ([0.0, 0.0, -1.0] + [0.0] * 7, True),  # a cut where the peak is
        ([1.5] + [0.0] * 9, True),
        ([0.0, 0.0, 0.0, 2.0] + [0.0] * 6, False),  # a boost that lifts the peak: only a part gets in
    ],
)
def test_real_render_an_accepted_candidate_is_no_worse_than_the_current_eq(gains, whole):
    audio, chain = _tones(1.0), _native_chain()  # hot enough that the current EQ is over the rule
    incumbent = config.EQSettings().to_dict()
    incumbent_excess = _excess(_native_render(audio, incumbent, chain))
    assert incumbent_excess > 0

    result = apply_headroom_validation(
        audio, 48_000, _candidate_fit(gains), chain, candidate_base_eq_settings=incumbent
    )

    validation = result["headroom_validation"]
    assert validation["status"] == "risk_not_worsened"
    assert validation["incumbent_excess_db"] == pytest.approx(incumbent_excess)
    offered = _native_render(audio, result["validated_candidate_eq"], chain)
    assert _excess(offered) <= incumbent_excess
    assert (validation["gain_scale"] == 1.0) is whole


@needs_native
def test_real_render_the_stated_decibels_let_the_full_correction_through_with_the_compressor_off():
    audio, chain = _tones(1.0), _native_chain()
    gains = [0.0, 0.0, -1.0] + [0.0] * 7
    result = apply_headroom_validation(
        audio, 48_000, _candidate_fit(gains), chain,
        candidate_base_eq_settings=config.EQSettings().to_dict(),
    )
    validation = result["headroom_validation"]
    advice = headroom_advice(validation, chain, bool(result["apply_recommended"]))

    assert validation["gain_scale"] == 1.0 and not _excess(validation["after"]) <= 0
    figure = re.search(r"Lower the microphone gain by about (\d+) dB", advice)
    assert figure and figure[1] == "1" and "to leave enough room" in advice
    assert _excess(_native_render(_tones(0.0), result["validated_candidate_eq"], chain)) <= 0


@needs_native
def test_real_render_the_advice_gives_no_figure_when_a_compressor_is_working():
    audio = _tones()
    chain = _native_chain(
        compressor={"enabled": True, "threshold_db": -20.0, "ratio": 4.0, "makeup_gain_db": 12.0},
        limiter={"enabled": True, "ceiling_db": -1.0, "careful_output_enabled": False},
    )
    result = apply_headroom_validation(
        audio, 48_000, _candidate_fit([0.0, 0.0, -1.0, 2.0] + [0.0] * 6), chain,
        candidate_base_eq_settings=config.EQSettings().to_dict(),
    )
    validation = result["headroom_validation"]
    figure = headroom_module.headroom_shortfall_db(validation)

    assert validation["status"] == "risk_not_worsened" and figure is not None
    # With level in equal to level out that figure would do; with this compressor it does not.
    lowered = _native_render(_tones(-figure), result["validated_candidate_eq"], chain)
    assert _excess(lowered) > 0
    advice = headroom_advice(validation, chain, bool(result["apply_recommended"]))
    assert "Lower the microphone gain" in advice and not re.search(r"\d+ dB", advice)
    assert "no worse than your current one" in advice and "to leave enough room" in advice
