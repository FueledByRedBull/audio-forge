"""Cross-take Auto-EQ confidence and feasibility tests."""

from __future__ import annotations

import inspect

import numpy as np

from mic_eq.analysis.auto_eq import (
    analyze_auto_eq,
    calculate_eq_bands,
    get_target_curve,
)
from mic_eq.analysis.auto_eq_parts.cross_take import cross_take_evidence


def _spectra() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.geomspace(50.0, 20_000.0, 256)
    log_frequency = np.log2(frequencies / 1_000.0)
    shape = (
        -1.5 * log_frequency
        + 2.0 * np.exp(-0.5 * np.square(log_frequency / 0.7))
    )
    uncertainty = np.full_like(frequencies, 0.3)
    return frequencies, shape, uncertainty


def test_rejected_candidate_is_not_exposed_by_product_auto_eq() -> None:
    parameters = inspect.signature(analyze_auto_eq).parameters

    assert "cross_take_audio" not in parameters
    assert "cross_take_vad_probabilities" not in parameters


def test_identical_and_level_shifted_takes_have_high_shape_agreement() -> None:
    frequencies, shape, uncertainty = _spectra()
    identical = cross_take_evidence(
        frequencies,
        shape,
        uncertainty,
        0.9,
        frequencies,
        shape,
        uncertainty,
        0.9,
    )
    shifted = cross_take_evidence(
        frequencies,
        shape,
        uncertainty,
        0.9,
        frequencies,
        shape + 12.0,
        uncertainty,
        0.9,
    )

    assert identical.global_confidence > 0.98
    assert shifted.global_confidence > 0.98
    assert shifted.voice_band_shape_rms_db < 1e-10


def test_narrow_mismatch_abstains_locally_without_poisoning_global_evidence() -> None:
    frequencies, shape, uncertainty = _spectra()
    mismatch = shape + 8.0 * np.exp(
        -0.5 * np.square(np.log2(frequencies / 4_000.0) / 0.10)
    )
    evidence = cross_take_evidence(
        frequencies,
        shape,
        uncertainty,
        0.9,
        frequencies,
        mismatch,
        uncertainty,
        0.9,
    )

    mismatch_bin = int(np.argmin(np.abs(frequencies - 4_000.0)))
    assert evidence.confidence[mismatch_bin] < 0.45
    assert evidence.global_confidence > 0.9


def test_global_shape_mismatch_has_low_cross_take_confidence() -> None:
    frequencies, shape, uncertainty = _spectra()
    tilted = shape + 5.0 * np.log2(frequencies / 1_000.0)
    evidence = cross_take_evidence(
        frequencies,
        shape,
        uncertainty,
        0.9,
        frequencies,
        tilted,
        uncertainty,
        0.9,
    )

    assert evidence.global_confidence < 0.35
    assert evidence.voice_band_shape_rms_db > 4.0


def test_precision_and_phonetic_coverage_remain_distinct() -> None:
    frequencies, shape, uncertainty = _spectra()
    evidence = cross_take_evidence(
        frequencies,
        shape,
        uncertainty,
        0.95,
        frequencies,
        shape,
        uncertainty,
        0.20,
    )
    diagnostics = evidence.diagnostics()

    assert evidence.global_confidence > 0.98
    assert diagnostics["minimum_phonetic_coverage"] == 0.20


def test_cross_take_confidence_limits_gain_inside_solver_feasibility() -> None:
    frequencies = np.geomspace(20.0, 20_000.0, 512)
    measured = np.full_like(frequencies, -70.0)
    measured -= 10.0 * np.exp(
        -0.5 * np.square(np.log2(frequencies / 4_000.0) / 0.12)
    )
    target = get_target_curve(
        frequencies,
        "flat",
        target_mode="static",
    )
    cross_take_confidence = np.ones_like(frequencies)
    cross_take_confidence[
        (frequencies >= 3_000.0) & (frequencies <= 5_500.0)
    ] = 0.1

    result = calculate_eq_bands(
        frequencies,
        measured,
        target,
        spectral_repeatability=np.ones_like(frequencies),
        spectral_uncertainty_db=np.full_like(frequencies, 0.3),
        cross_take_confidence=cross_take_confidence,
        phonetic_coverage=0.9,
        voiced_window_ratio=0.8,
        analysis_confidence=0.9,
    )
    centers = np.asarray(result["band_freqs"], dtype=float)
    gains = np.asarray(result["band_gains"], dtype=float)
    nearest = int(np.argmin(np.abs(centers - 4_000.0)))

    assert result["cross_take_confidence_available"] is True
    assert result["cross_take_gain_feasibility_scale"][nearest] == 0.02
    assert gains[nearest] == 0.0
