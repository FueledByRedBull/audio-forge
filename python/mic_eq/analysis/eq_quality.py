"""Shared EQ response quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .auto_eq_parts.constants import SAMPLE_RATE
from .auto_eq_parts.response import (
    _default_filter_types,
    _is_gain_filter,
    _normalize_filter_type,
    _predict_eq_response,
)


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

    @property
    def safe_for_auto_eq(self) -> bool:
        """Whether the combined response stays inside the automatic bounds."""
        return (
            self.max_boost_db <= 6.0 + 1e-9
            and self.max_cut_db <= 6.0 + 1e-9
            and self.ripple_db <= 8.0 + 1e-9
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
            "safe_for_auto_eq": self.safe_for_auto_eq,
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
    filter_types: Iterable[str] | None,
    enabled: Iterable[bool] | None,
    slopes_db_per_octave: Iterable[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[bool], list[int]]:
    centers = np.asarray(list(freqs), dtype=float)
    gains_db = np.asarray(list(gains), dtype=float)
    q_values = np.asarray(list(qs), dtype=float)
    if not (
        centers.ndim == gains_db.ndim == q_values.ndim == 1
        and centers.size == gains_db.size == q_values.size
    ):
        raise ValueError("frequency, gain, and Q arrays must have the same length")
    size = centers.size
    raw_types = (
        _default_filter_types(size)
        if filter_types is None
        else [_normalize_filter_type(str(value)) for value in filter_types]
    )
    raw_enabled = [True] * size if enabled is None else list(enabled)
    raw_slopes = (
        [12] * size if slopes_db_per_octave is None else list(slopes_db_per_octave)
    )
    if not (len(raw_types) == len(raw_enabled) == len(raw_slopes) == size):
        raise ValueError("EQ filter metadata must match the frequency array length")
    if any(not isinstance(value, (bool, np.bool_)) for value in raw_enabled):
        raise ValueError("EQ enabled flags must be booleans")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw_slopes
    ):
        raise ValueError("EQ filter slopes must be integers")

    order = np.argsort(centers, kind="stable")
    return (
        centers[order],
        gains_db[order],
        q_values[order],
        [raw_types[index] for index in order],
        [bool(raw_enabled[index]) for index in order],
        [int(raw_slopes[index]) for index in order],
    )


def evaluate_eq_quality(
    freqs: Iterable[float],
    gains: Iterable[float],
    qs: Iterable[float],
    sample_rate: float = SAMPLE_RATE,
    *,
    filter_types: Iterable[str] | None = None,
    enabled: Iterable[bool] | None = None,
    slopes_db_per_octave: Iterable[int] | None = None,
) -> EqQualityMetrics:
    """Evaluate legacy or typed EQ interaction risks and response bounds."""
    frequency_values = list(freqs)
    gain_values = list(gains)
    q_values_in_order = list(qs)
    filter_types_in_order = None if filter_types is None else list(filter_types)
    enabled_in_order = None if enabled is None else list(enabled)
    slopes_in_order = (
        None if slopes_db_per_octave is None else list(slopes_db_per_octave)
    )
    centers, gains_db, q_values, kinds, enabled_flags, slopes = _as_arrays(
        frequency_values,
        gain_values,
        q_values_in_order,
        filter_types_in_order,
        enabled_in_order,
        slopes_in_order,
    )
    if centers.size == 0:
        return EqQualityMetrics(0.0, 0.0, 0.0, 0, 0, 0, ())

    grid = np.logspace(
        np.log10(20.0),
        np.log10(min(20_000.0, sample_rate / 2.0 - 1.0)),
        256,
    )
    response = _predict_eq_response(
        grid,
        gain_values,
        q_values_in_order,
        frequency_values,
        filter_types=filter_types_in_order,
        slopes_db_per_octave=slopes_in_order,
        enabled=enabled_in_order,
        sample_rate=sample_rate,
    )
    voice_mask = (grid >= 80.0) & (grid <= 12_000.0)
    voice_response = response[voice_mask] if np.any(voice_mask) else response

    max_boost_db = float(max(0.0, np.max(response)))
    max_cut_db = float(max(0.0, -np.min(response)))
    ripple_db = float(
        np.percentile(voice_response, 95.0) - np.percentile(voice_response, 5.0)
    )

    warnings: list[EqInteractionWarning] = []
    overlapping = 0
    shelf_stacking = 0
    narrow_boost = 0

    interaction_bands = [
        index
        for index, (filter_type, is_enabled, gain_db) in enumerate(
            zip(kinds, enabled_flags, gains_db, strict=True)
        )
        if is_enabled and _is_gain_filter(filter_type) and abs(gain_db) >= 0.5
    ]
    for i, j in zip(interaction_bands, interaction_bands[1:]):
        octave_gap = abs(float(np.log2(centers[j] / centers[i])))
        same_sign = np.sign(gains_db[i]) == np.sign(gains_db[j])
        high_q_pair = max(q_values[i], q_values[j]) >= 3.0
        high_gain_pair = min(abs(gains_db[i]), abs(gains_db[j])) >= 3.0
        if same_sign and octave_gap < 0.42 and (high_q_pair or high_gain_pair):
            overlapping += 1
            warnings.append(
                EqInteractionWarning(
                    "overlap",
                    float(np.sqrt(centers[i] * centers[j])),
                    min(1.0, (0.42 - octave_gap) / 0.42 + 0.25),
                    "Adjacent bands are stacking",
                )
            )

    nyquist = sample_rate / 2.0
    audible_centers = centers < nyquist
    for shelf_index, filter_type in enumerate(kinds):
        if (
            not enabled_flags[shelf_index]
            or filter_type not in {"low_shelf", "high_shelf"}
            or abs(gains_db[shelf_index]) < 3.0
        ):
            continue
        low_shelf = filter_type == "low_shelf"
        if (low_shelf and centers[shelf_index] > 320.0) or (
            not low_shelf and centers[shelf_index] < 7_000.0
        ):
            continue
        shelf_effects = np.zeros_like(centers)
        if centers[shelf_index] < nyquist and np.any(audible_centers):
            shelf_effects[audible_centers] = _predict_eq_response(
                centers[audible_centers],
                [gains_db[shelf_index]],
                [q_values[shelf_index]],
                [centers[shelf_index]],
                filter_types=[filter_type],
                sample_rate=sample_rate,
            )
        for bell_index, bell_type in enumerate(kinds):
            if (
                shelf_index == bell_index
                or not enabled_flags[bell_index]
                or bell_type != "peak"
                or abs(gains_db[bell_index]) < 2.0
                or np.sign(gains_db[shelf_index]) != np.sign(gains_db[bell_index])
            ):
                continue
            if (low_shelf and centers[bell_index] <= 320.0) or (
                not low_shelf and centers[bell_index] >= 7_000.0
            ):
                shelf_response_db = float(shelf_effects[bell_index])
                if (
                    abs(shelf_response_db) < 0.5
                    or np.sign(shelf_response_db) != np.sign(gains_db[bell_index])
                ):
                    continue
                shelf_stacking += 1
                warnings.append(
                    EqInteractionWarning(
                        "shelf_stack",
                        float(centers[bell_index]),
                        min(
                            1.0,
                            (abs(shelf_response_db) + abs(gains_db[bell_index]))
                            / 16.0,
                        ),
                        "Shelf response stacks with a same-direction peak",
                    )
                )

    for center, gain_db, q, filter_type, is_enabled in zip(
        centers, gains_db, q_values, kinds, enabled_flags, strict=True
    ):
        if is_enabled and _is_gain_filter(filter_type) and gain_db > 5.0 and q > 3.5:
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
    *,
    sample_rate: float = SAMPLE_RATE,
) -> float:
    """Return weighted RMS target error after applying an EQ curve."""
    gains_arr = np.asarray(list(gains), dtype=float)
    qs_arr = np.asarray(list(qs), dtype=float)
    centers_arr = np.asarray(list(center_freqs), dtype=float)
    response = _predict_eq_response(
        freqs,
        gains_arr,
        qs_arr,
        centers_arr,
        sample_rate=sample_rate,
    )
    error = target_db - (measured_db + response)
    if weights is None:
        weights = np.ones_like(freqs, dtype=float)
    weighted = np.asarray(weights, dtype=float)
    denom = float(np.sum(weighted))
    if denom <= 0.0:
        return float(np.sqrt(np.mean(np.square(error))))
    return float(np.sqrt(np.sum(weighted * np.square(error)) / denom))
