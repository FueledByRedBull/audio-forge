"""
Comprehensive pytest suite for Auto-EQ behavior.
"""

import numpy as np
import pytest

from mic_eq import config
from mic_eq.analysis import auto_eq
from mic_eq.analysis.auto_eq_parts import headroom as headroom_module
from mic_eq.analysis.auto_eq_parts import optimizer as optimizer_module
from mic_eq.analysis.failure_detection import validate_analysis
from mic_eq.analysis.spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual

_predict_eq_response = auto_eq._predict_eq_response
calculate_eq_bands = auto_eq.calculate_eq_bands
analyze_auto_eq = auto_eq.analyze_auto_eq
get_target_curve = auto_eq.get_target_curve
_remove_spectral_tilt = auto_eq._remove_spectral_tilt
_log_frequency_gain_curvature = auto_eq._log_frequency_gain_curvature
_snr_aware_gain_upper_bounds = auto_eq._snr_aware_gain_upper_bounds
evaluate_eq_quality = auto_eq.evaluate_eq_quality
apply_headroom_validation = auto_eq.apply_headroom_validation
simulate_candidate_chain = auto_eq.simulate_candidate_chain
EQ_FREQUENCIES = config.EQ_FREQUENCIES
AUTO_EQ_DEFAULT_Q = config.AUTO_EQ_DEFAULT_Q


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
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert any(g > 2.0 for g in [gains[3], gains[4], gains[5]])


def test_05_proximity_effect_correction():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "proximity")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert gains[0] < -5.0


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
    gains = eq["band_gains"]
    # The constrained solver should still make a material correction without
    # requiring a dangerous hard-bound excursion.
    assert any(abs(g) >= 3.0 for g in gains)
    assert eq["validation_after_error_db"] < eq["validation_before_error_db"] * 0.80


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
    detrended, slope = _remove_spectral_tilt(freqs, measured_db)

    x_center = x - np.mean(x)
    residual_slope = float(np.dot(x_center, detrended) / np.dot(x_center, x_center))
    assert abs(slope) > 1.0
    assert abs(residual_slope) < 1e-3


def test_15a_tilt_removal_accepts_perfect_nonzero_intercept():
    freqs = _default_freqs()
    x = np.log10(freqs)
    measured_db = 4.0 * (x - np.mean(x)) + 10.0
    detrended, slope = _remove_spectral_tilt(freqs, measured_db)

    x_center = x - np.mean(x)
    residual_slope = float(np.dot(x_center, detrended) / np.dot(x_center, x_center))
    assert abs(slope - 4.0) < 1e-6
    assert abs(residual_slope) < 1e-6
    assert np.allclose(detrended, np.full_like(detrended, detrended[0]))


def test_15b_tilt_removal_rejects_flat_response():
    freqs = _default_freqs()
    measured_db = np.full_like(freqs, 6.0)

    detrended, slope = _remove_spectral_tilt(freqs, measured_db)

    assert slope == 0.0
    assert np.allclose(detrended, measured_db)


def test_15c_tilt_removal_accepts_noisy_tilt_above_fit_threshold():
    freqs = _default_freqs()
    x = np.log10(freqs)
    x_center = x - np.mean(x)
    rng = np.random.default_rng(1503)
    measured_db = 3.0 * x_center + 2.0 + rng.normal(0.0, 0.12, size=freqs.size)

    detrended, slope = _remove_spectral_tilt(freqs, measured_db)
    residual_slope = float(np.dot(x_center, detrended) / np.dot(x_center, x_center))

    assert abs(slope) > 2.0
    assert abs(residual_slope) < 0.2


def test_15d_tilt_removal_rejects_random_response_below_fit_threshold():
    freqs = _default_freqs()
    rng = np.random.default_rng(1504)
    measured_db = rng.normal(0.0, 4.0, size=freqs.size)

    detrended, slope = _remove_spectral_tilt(freqs, measured_db)

    assert slope == 0.0
    assert np.allclose(detrended, measured_db)


def test_15e_production_tilt_policy_preserves_broad_mic_response():
    freqs = _default_freqs()
    measured_db = -70.0 - 5.0 * np.log2(freqs / 1000.0)
    target_db = get_target_curve(freqs, "flat", target_mode="static")

    preserved = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        tilt_policy="preserve",
    )
    detrended = calculate_eq_bands(
        freqs,
        measured_db,
        target_db,
        tilt_policy="detrend",
    )

    preserved_gains = np.asarray(preserved["band_gains"], dtype=float)
    detrended_gains = np.asarray(detrended["band_gains"], dtype=float)
    assert preserved["spectral_tilt_policy"] == "preserve"
    assert detrended["spectral_tilt_policy"] == "detrend"
    assert np.max(np.abs(preserved_gains)) > np.max(np.abs(detrended_gains)) + 2.0


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
    assert np.max(np.abs(adaptive_target - static_target)) <= 2.0 + 1e-9
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
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "harsh")
    eq_settings = {
        "band_gains": [0.0] * 10,
        "headroom_validation": {"safe": False},
    }

    validation = validate_analysis(eq_settings, spectrum_db, freqs)

    assert not validation.passed
    assert validation.details["headroom_safe"] is False


def test_28_python_headroom_fallback_is_explicitly_advisory(monkeypatch):
    monkeypatch.setattr(
        headroom_module, "_native_simulate", lambda *_args, **_kwargs: None
    )
    audio = np.zeros(4096, dtype=np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "validation_confidence": 0.95,
        "analysis_confidence": 0.95,
    }

    validated = apply_headroom_validation(
        audio,
        48_000,
        eq_settings,
        {
            "compressor": {"enabled": False},
            "deesser": {"enabled": False},
            "limiter": {"enabled": True, "careful_output_enabled": True},
        },
    )
    headroom = validated["headroom_validation"]

    assert headroom["status"] == "advisory"
    assert headroom["advisory"] is True
    assert headroom["authoritative"] is False
    assert headroom["safe"] is False
    assert validated["headroom_safe"] is False
    assert validated["validation_confidence"] <= 0.42
    assert headroom["after"]["simulation_backend"] == "python"
    assert headroom["after"]["limitations"]


def test_29_fallback_cannot_report_risky_capture_as_safe(monkeypatch):
    monkeypatch.setattr(
        headroom_module, "_native_simulate", lambda *_args, **_kwargs: None
    )
    t = np.arange(48_000, dtype=float) / 48_000.0
    audio = (0.95 * np.sin(2.0 * np.pi * 5000.0 * t)).astype(np.float32)
    eq_settings = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [9.0] * 10,
        "band_qs": [1.41] * 10,
    }

    validated = apply_headroom_validation(
        audio,
        48_000,
        eq_settings,
        {
            "compressor": {"enabled": False},
            "deesser": {"enabled": False},
            "limiter": {"enabled": True, "careful_output_enabled": True},
        },
    )

    assert validated["headroom_validation"]["status"] == "advisory"
    assert validated["headroom_validation"]["safe"] is False
    assert validated["headroom_gain_scale"] < 1.0
