"""Shared EQ response quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .auto_eq_parts.response import _predict_eq_response


@dataclass(frozen=True, slots=True)
class EqInteractionWarning:
    """A localized EQ interaction warning."""

    kind: str
    frequency_hz: float
    severity: float
    message: str


@dataclass(frozen=True, slots=True)
class EqQualityMetrics:
    """Summary metrics for a combined 10-band EQ response."""

    max_boost_db: float
    max_cut_db: float
    ripple_db: float
    overlapping_adjacent_bands: int
    shelf_peak_stacking: int
    narrow_boost_risk: int
    warnings: tuple[EqInteractionWarning, ...]

    @property
    def risk_score(self) -> float:
        return (
            max(0.0, self.max_boost_db - 9.0) / 6.0
            + max(0.0, self.max_cut_db - 12.0) / 6.0
            + max(0.0, self.ripple_db - 10.0) / 8.0
            + self.overlapping_adjacent_bands * 0.4
            + self.shelf_peak_stacking * 0.45
            + self.narrow_boost_risk * 0.5
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_boost_db": self.max_boost_db,
            "max_cut_db": self.max_cut_db,
            "ripple_db": self.ripple_db,
            "overlapping_adjacent_bands": self.overlapping_adjacent_bands,
            "shelf_peak_stacking": self.shelf_peak_stacking,
            "narrow_boost_risk": self.narrow_boost_risk,
            "risk_score": self.risk_score,
            "warnings": [
                {
                    "kind": warning.kind,
                    "frequency_hz": warning.frequency_hz,
                    "severity": warning.severity,
                    "message": warning.message,
                }
                for warning in self.warnings
            ],
        }


def _as_arrays(
    freqs: Iterable[float],
    gains: Iterable[float],
    qs: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(list(freqs), dtype=float)
    gains_db = np.asarray(list(gains), dtype=float)
    q_values = np.asarray(list(qs), dtype=float)
    if not (centers.size == gains_db.size == q_values.size):
        raise ValueError("frequency, gain, and Q arrays must have the same length")
    order = np.argsort(centers)
    return centers[order], gains_db[order], q_values[order]


def evaluate_eq_quality(
    freqs: Iterable[float],
    gains: Iterable[float],
    qs: Iterable[float],
    sample_rate: float = 48_000.0,
) -> EqQualityMetrics:
    """Evaluate live EQ interaction risks from band frequency/gain/Q values."""
    centers, gains_db, q_values = _as_arrays(freqs, gains, qs)
    if centers.size == 0:
        return EqQualityMetrics(0.0, 0.0, 0.0, 0, 0, 0, ())

    grid = np.logspace(np.log10(20.0), np.log10(min(20_000.0, sample_rate / 2.0 - 1.0)), 256)
    response = _predict_eq_response(grid, gains_db, q_values, centers)
    voice_mask = (grid >= 80.0) & (grid <= 12_000.0)
    voice_response = response[voice_mask] if np.any(voice_mask) else response

    max_boost_db = float(max(0.0, np.max(response)))
    max_cut_db = float(max(0.0, -np.min(response)))
    ripple_db = float(np.percentile(voice_response, 95.0) - np.percentile(voice_response, 5.0))

    warnings: list[EqInteractionWarning] = []
    overlapping = 0
    shelf_stacking = 0
    narrow_boost = 0

    for i in range(centers.size - 1):
        if abs(gains_db[i]) < 0.5 or abs(gains_db[i + 1]) < 0.5:
            continue
        octave_gap = abs(float(np.log2(centers[i + 1] / centers[i])))
        same_sign = np.sign(gains_db[i]) == np.sign(gains_db[i + 1])
        high_q_pair = max(q_values[i], q_values[i + 1]) >= 3.0
        high_gain_pair = min(abs(gains_db[i]), abs(gains_db[i + 1])) >= 3.0
        if same_sign and octave_gap < 0.42 and (high_q_pair or high_gain_pair):
            overlapping += 1
            warnings.append(
                EqInteractionWarning(
                    "overlap",
                    float(np.sqrt(centers[i] * centers[i + 1])),
                    min(1.0, (0.42 - octave_gap) / 0.42 + 0.25),
                    "Adjacent bands are stacking",
                )
            )

    if centers.size >= 2:
        low_shelf_gain = gains_db[0]
        for i in range(1, min(4, centers.size)):
            if centers[i] <= 320.0 and np.sign(low_shelf_gain) == np.sign(gains_db[i]):
                if abs(low_shelf_gain) >= 3.0 and abs(gains_db[i]) >= 2.0:
                    shelf_stacking += 1
                    warnings.append(
                        EqInteractionWarning(
                            "shelf_stack",
                            float(centers[i]),
                            min(1.0, (abs(low_shelf_gain) + abs(gains_db[i])) / 16.0),
                            "Shelf and nearby peak are stacking",
                        )
                    )

        high_shelf_gain = gains_db[-1]
        for i in range(max(0, centers.size - 4), centers.size - 1):
            if centers[i] >= 7_000.0 and np.sign(high_shelf_gain) == np.sign(gains_db[i]):
                if abs(high_shelf_gain) >= 3.0 and abs(gains_db[i]) >= 2.0:
                    shelf_stacking += 1
                    warnings.append(
                        EqInteractionWarning(
                            "shelf_stack",
                            float(centers[i]),
                            min(1.0, (abs(high_shelf_gain) + abs(gains_db[i])) / 16.0),
                            "Shelf and nearby peak are stacking",
                        )
                    )

    for center, gain_db, q in zip(centers, gains_db, q_values):
        if gain_db > 5.0 and q > 3.5:
            narrow_boost += 1
            warnings.append(
                EqInteractionWarning(
                    "narrow_boost",
                    float(center),
                    min(1.0, ((gain_db - 5.0) / 7.0) + ((q - 3.5) / 5.0)),
                    "Narrow high-gain boost",
                )
            )

    if max_boost_db > 10.5:
        warnings.append(
            EqInteractionWarning(
                "max_boost",
                float(grid[int(np.argmax(response))]),
                min(1.0, (max_boost_db - 10.5) / 6.0),
                "Combined boost is high",
            )
        )
    if ripple_db > 11.0:
        warnings.append(
            EqInteractionWarning(
                "ripple",
                float(grid[int(np.argmax(np.abs(response)))]),
                min(1.0, (ripple_db - 11.0) / 8.0),
                "Combined response is uneven",
            )
        )

    warnings.sort(key=lambda warning: warning.severity, reverse=True)
    return EqQualityMetrics(
        max_boost_db=max_boost_db,
        max_cut_db=max_cut_db,
        ripple_db=ripple_db,
        overlapping_adjacent_bands=overlapping,
        shelf_peak_stacking=shelf_stacking,
        narrow_boost_risk=narrow_boost,
        warnings=tuple(warnings),
    )


def weighted_target_error(
    freqs: np.ndarray,
    measured_db: np.ndarray,
    target_db: np.ndarray,
    gains: Iterable[float],
    qs: Iterable[float],
    center_freqs: Iterable[float],
    weights: np.ndarray | None = None,
) -> float:
    """Return weighted RMS target error after applying an EQ curve."""
    gains_arr = np.asarray(list(gains), dtype=float)
    qs_arr = np.asarray(list(qs), dtype=float)
    centers_arr = np.asarray(list(center_freqs), dtype=float)
    response = _predict_eq_response(freqs, gains_arr, qs_arr, centers_arr)
    error = target_db - (measured_db + response)
    if weights is None:
        weights = np.ones_like(freqs, dtype=float)
    weighted = np.asarray(weights, dtype=float)
    denom = float(np.sum(weighted))
    if denom <= 0.0:
        return float(np.sqrt(np.mean(np.square(error))))
    return float(np.sqrt(np.sum(weighted * np.square(error)) / denom))
