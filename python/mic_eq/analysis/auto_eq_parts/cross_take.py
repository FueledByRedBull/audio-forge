"""Evaluation-only cross-take Auto-EQ evidence.

The production Auto-EQ and Voice Setup paths do not import this module. It is
retained solely so the experiment and its objective retention decision remain
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VOICE_MIN_HZ = 80.0
VOICE_MAX_HZ = 12_000.0
LEVEL_REFERENCE_MIN_HZ = 100.0
LEVEL_REFERENCE_MAX_HZ = 8_000.0
PRECISION_SCALE_DB = 2.5
MISMATCH_SCALE_DB = 2.5


@dataclass(frozen=True, slots=True)
class CrossTakeEvidence:
    frequencies_hz: np.ndarray
    shape_delta_db: np.ndarray
    combined_precision_uncertainty_db: np.ndarray
    precision_confidence: np.ndarray
    agreement_confidence: np.ndarray
    confidence: np.ndarray
    global_confidence: float
    voice_band_shape_rms_db: float
    first_phonetic_coverage: float
    second_phonetic_coverage: float

    def diagnostics(self) -> dict[str, float | bool]:
        return {
            "available": True,
            "global_confidence": self.global_confidence,
            "voice_band_shape_rms_db": self.voice_band_shape_rms_db,
            "first_phonetic_coverage": self.first_phonetic_coverage,
            "second_phonetic_coverage": self.second_phonetic_coverage,
            "minimum_phonetic_coverage": min(
                self.first_phonetic_coverage,
                self.second_phonetic_coverage,
            ),
            "median_precision_uncertainty_db": float(
                np.median(
                    self.combined_precision_uncertainty_db[
                        (self.frequencies_hz >= VOICE_MIN_HZ)
                        & (self.frequencies_hz <= VOICE_MAX_HZ)
                    ]
                )
            ),
        }


def _level_normalized_shape(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
) -> np.ndarray:
    reference = (
        (frequencies_hz >= LEVEL_REFERENCE_MIN_HZ)
        & (frequencies_hz <= LEVEL_REFERENCE_MAX_HZ)
    )
    if np.count_nonzero(reference) < 8:
        raise ValueError("cross-take spectrum has insufficient voice-band bins")
    return spectrum_db - float(np.median(spectrum_db[reference]))


def _validated_uncertainty(
    values: np.ndarray | None,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if values is None:
        return np.full(expected_shape, np.inf, dtype=float)
    uncertainty = np.asarray(values, dtype=float)
    if uncertainty.shape != expected_shape:
        raise ValueError("cross-take uncertainty shape does not match its spectrum")
    return np.where(
        np.isfinite(uncertainty),
        np.clip(uncertainty, 0.0, 40.0),
        np.inf,
    )


def cross_take_evidence(
    first_frequencies_hz: np.ndarray,
    first_spectrum_db: np.ndarray,
    first_uncertainty_db: np.ndarray | None,
    first_phonetic_coverage: float,
    second_frequencies_hz: np.ndarray,
    second_spectrum_db: np.ndarray,
    second_uncertainty_db: np.ndarray | None,
    second_phonetic_coverage: float,
) -> CrossTakeEvidence:
    """Measure frequency-dependent agreement without conflating it with coverage."""
    frequencies = np.asarray(first_frequencies_hz, dtype=float)
    first = np.asarray(first_spectrum_db, dtype=float)
    second_frequencies = np.asarray(second_frequencies_hz, dtype=float)
    second = np.asarray(second_spectrum_db, dtype=float)
    if (
        frequencies.ndim != 1
        or first.shape != frequencies.shape
        or second_frequencies.ndim != 1
        or second.shape != second_frequencies.shape
        or frequencies.size < 16
        or second_frequencies.size < 16
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second_frequencies))
        or not np.all(np.isfinite(second))
        or np.any(np.diff(frequencies) <= 0.0)
        or np.any(np.diff(second_frequencies) <= 0.0)
    ):
        raise ValueError("cross-take spectra must be finite increasing 1-D arrays")

    second_aligned = np.interp(
        frequencies,
        second_frequencies,
        second,
        left=float(second[0]),
        right=float(second[-1]),
    )
    first_uncertainty = _validated_uncertainty(
        first_uncertainty_db,
        first.shape,
    )
    second_uncertainty_raw = _validated_uncertainty(
        second_uncertainty_db,
        second.shape,
    )
    second_uncertainty = np.interp(
        frequencies,
        second_frequencies,
        second_uncertainty_raw,
        left=float(second_uncertainty_raw[0]),
        right=float(second_uncertainty_raw[-1]),
    )

    first_shape = _level_normalized_shape(frequencies, first)
    second_shape = _level_normalized_shape(frequencies, second_aligned)
    delta = second_shape - first_shape
    combined_uncertainty = np.sqrt(
        np.square(first_uncertainty) + np.square(second_uncertainty)
    )
    finite_uncertainty = np.where(
        np.isfinite(combined_uncertainty),
        combined_uncertainty,
        40.0,
    )
    precision_confidence = 1.0 / (
        1.0 + np.square(finite_uncertainty / PRECISION_SCALE_DB)
    )
    mismatch_beyond_precision = np.maximum(
        0.0,
        np.abs(delta) - finite_uncertainty,
    )
    agreement_confidence = np.exp(
        -0.5 * np.square(mismatch_beyond_precision / MISMATCH_SCALE_DB)
    )
    confidence = np.sqrt(precision_confidence * agreement_confidence)
    voice = (frequencies >= VOICE_MIN_HZ) & (frequencies <= VOICE_MAX_HZ)
    if np.count_nonzero(voice) < 8:
        raise ValueError("cross-take spectra do not cover the voice band")
    return CrossTakeEvidence(
        frequencies_hz=frequencies,
        shape_delta_db=delta,
        combined_precision_uncertainty_db=combined_uncertainty,
        precision_confidence=np.clip(precision_confidence, 0.0, 1.0),
        agreement_confidence=np.clip(agreement_confidence, 0.0, 1.0),
        confidence=np.clip(confidence, 0.0, 1.0),
        global_confidence=float(np.median(confidence[voice])),
        voice_band_shape_rms_db=float(
            np.sqrt(np.mean(np.square(delta[voice])))
        ),
        first_phonetic_coverage=float(
            np.clip(first_phonetic_coverage, 0.0, 1.0)
        ),
        second_phonetic_coverage=float(
            np.clip(second_phonetic_coverage, 0.0, 1.0)
        ),
    )


__all__ = ["CrossTakeEvidence", "cross_take_evidence"]
