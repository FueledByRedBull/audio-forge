"""
Comprehensive pytest suite for Auto-EQ behavior.
"""

import numpy as np

from mic_eq import config
from mic_eq.analysis import auto_eq
from mic_eq.analysis.failure_detection import validate_analysis
from mic_eq.analysis.spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual

_predict_eq_response = auto_eq._predict_eq_response
calculate_eq_bands = auto_eq.calculate_eq_bands
analyze_auto_eq = auto_eq.analyze_auto_eq
get_target_curve = auto_eq.get_target_curve
_remove_spectral_tilt = auto_eq._remove_spectral_tilt
_snr_aware_gain_upper_bounds = auto_eq._snr_aware_gain_upper_bounds
evaluate_eq_quality = auto_eq.evaluate_eq_quality
EQ_FREQUENCIES = config.EQ_FREQUENCIES
AUTO_EQ_DEFAULT_Q = config.AUTO_EQ_DEFAULT_Q


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
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    # The solver applies a correction factor, so strong responses may not hit
    # hard +/-12 dB bounds in final output. Require a large correction instead.
    assert any(abs(g) >= 7.0 for g in gains)


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
    gains = np.asarray(calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"], dtype=float)

    assert np.all(np.abs(np.diff(gains)) <= 6.0 + 1e-9)


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


def test_16_snr_aware_boost_caps_are_bounded_and_monotonic():
    snr_db = np.array([-5.0, 0.0, 3.0, 8.0, 12.0, 18.0, 30.0], dtype=float)
    caps = _snr_aware_gain_upper_bounds(snr_db)

    assert np.all(caps >= 3.0)
    assert np.all(caps <= 12.0)
    assert np.all(np.diff(caps) >= -1e-9)


def test_17_dynamic_center_tracks_non_default_problem_frequency():
    freqs = _default_freqs()
    log_freqs = np.log10(freqs)
    spectrum_db = np.full_like(freqs, -70.0)
    spectrum_db -= 8.0 * np.exp(
        -((log_freqs - np.log10(2300.0)) ** 2) / (2 * 0.045**2)
    )
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
    assert 0.0 < eq["validation_gain_scale"] <= 1.0
    assert eq["target_profile"]


def test_20_low_confidence_boosts_are_capped_aggressively():
    freqs = _default_freqs()
    spectrum_db = np.full_like(freqs, -70.0)
    log_freqs = np.log10(freqs)
    spectrum_db -= 12.0 * np.exp(
        -((log_freqs - np.log10(3000.0)) ** 2) / (2 * 0.04**2)
    )
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
    spectrum_smoothed = smooth_spectrum_perceptual(freqs, spectrum_result.median_spectrum_db)
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
