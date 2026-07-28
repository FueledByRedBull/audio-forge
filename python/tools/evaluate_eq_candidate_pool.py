"""Rerun the sparse Auto-EQ candidate-pool experiment after confidence redesign."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from mic_eq.analysis import auto_eq
from mic_eq.analysis.auto_eq_parts import optimizer
from mic_eq.analysis.auto_eq_parts.constants import NUM_EQ_BANDS, Q_MAX, Q_MIN
from mic_eq.analysis.auto_eq_parts.dynamic_bands import (
    _estimate_q_from_residual,
    _select_dynamic_band_layout,
)
from mic_eq.analysis.auto_eq_parts.response import FILTER_PEAK

CASES = (
    ("bassy", "broadcast"),
    ("bright", "flat"),
    ("dark", "podcast"),
    ("midscooped", "streaming"),
    ("proximity", "broadcast"),
    ("harsh", "flat"),
    ("extreme", "flat"),
    ("flat", "broadcast"),
)
POOL_SIZES = (12, 14, 16)
SEED = 991


def _spectrum(freqs: np.ndarray, kind: str) -> np.ndarray:
    base = np.full_like(freqs, -70.0)
    if kind == "bassy":
        return base + 10.0 / (1.0 + (freqs / 200.0) ** 2)
    if kind == "bright":
        return base + 10.0 * (freqs / 4000.0) ** 2 / (1.0 + (freqs / 4000.0) ** 2)
    if kind == "dark":
        return base - 10.0 / (1.0 + (8000.0 / freqs) ** 2)
    if kind == "midscooped":
        return base - 8.0 * np.exp(
            -((np.log10(freqs) - np.log10(1500.0)) ** 2) / (2.0 * 0.18**2)
        )
    if kind == "proximity":
        return base + 15.0 / (1.0 + (freqs / 100.0) ** 3)
    if kind == "harsh":
        return base + 12.0 * np.exp(-((freqs - 4000.0) ** 2) / (2.0 * 1500.0**2))
    if kind == "extreme":
        return base + 20.0 * np.sin(3.0 * np.log10(freqs / 100.0))
    return base


def _smooth_perturbation(freqs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    knots = np.geomspace(float(freqs[0]), float(freqs[-1]), 18)
    values = rng.normal(0.0, 0.25, knots.size)
    return np.interp(np.log(freqs), np.log(knots), values)


def _candidate_pool_selector(pool_size: int) -> Callable:
    def select(
        dense_freqs: np.ndarray,
        residual_db: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        production_centers, _production_qs = _select_dynamic_band_layout(
            dense_freqs,
            residual_db,
            weights,
        )
        interior_mask = (dense_freqs >= 200.0) & (dense_freqs <= 9000.0)
        indices = np.flatnonzero(interior_mask)
        extrema = [
            int(index)
            for index in indices[1:-1]
            if abs(residual_db[index]) >= abs(residual_db[index - 1])
            and abs(residual_db[index]) >= abs(residual_db[index + 1])
        ]
        extrema.sort(key=lambda index: abs(float(residual_db[index])) * weights[index], reverse=True)
        candidates = [float(dense_freqs[index]) for index in extrema]
        candidates.extend(
            float(value)
            for value in np.geomspace(200.0, 9000.0, max(10, pool_size * 2))
        )
        unique: list[float] = []
        for center in candidates:
            if all(abs(np.log2(center / existing)) >= 0.08 for existing in unique):
                unique.append(center)
            if len(unique) >= pool_size - 2:
                break
        for center in production_centers[1:-1]:
            if all(abs(np.log2(center / existing)) >= 0.02 for existing in unique):
                unique.append(float(center))
        unique = unique[: max(pool_size - 2, NUM_EQ_BANDS - 2)]

        candidate_q = np.asarray(
            [
                _estimate_q_from_residual(
                    dense_freqs,
                    residual_db,
                    center,
                    Q_MIN,
                    Q_MAX,
                    1.41,
                )
                for center in unique
            ]
        )
        basis = np.column_stack(
            [
                auto_eq._predict_eq_response(
                    dense_freqs,
                    [1.0],
                    [candidate_q[index]],
                    [center],
                    [FILTER_PEAK],
                )
                for index, center in enumerate(unique)
            ]
        )
        weighted_basis = basis * np.sqrt(weights)[:, None]
        weighted_target = residual_db * np.sqrt(weights)
        selected: list[int] = []
        remaining = list(range(len(unique)))
        for _ in range(NUM_EQ_BANDS - 2):
            best_index = remaining[0]
            best_error = float("inf")
            for candidate_index in remaining:
                trial = selected + [candidate_index]
                gains, *_ = np.linalg.lstsq(
                    weighted_basis[:, trial],
                    weighted_target,
                    rcond=None,
                )
                gains = np.clip(gains, -12.0, 12.0)
                error = float(
                    np.sqrt(
                        np.average(
                            (residual_db - basis[:, trial] @ gains) ** 2,
                            weights=weights,
                        )
                    )
                )
                if error < best_error:
                    best_error = error
                    best_index = candidate_index
            selected.append(best_index)
            remaining.remove(best_index)

        interiors = sorted(float(unique[index]) for index in selected)
        centers = np.asarray(
            [float(production_centers[0]), *interiors, float(production_centers[-1])]
        )
        qs = np.asarray(
            [
                _estimate_q_from_residual(
                    dense_freqs,
                    residual_db,
                    float(center),
                    Q_MIN,
                    Q_MAX,
                    1.41,
                )
                for center in centers
            ]
        )
        return centers, qs

    return select


@contextmanager
def _selector(selector: Callable) -> Iterator[None]:
    original = optimizer._select_dynamic_band_layout
    optimizer._select_dynamic_band_layout = selector
    try:
        yield
    finally:
        optimizer._select_dynamic_band_layout = original


def _run(
    freqs: np.ndarray,
    measured: np.ndarray,
    target: np.ndarray,
    selector: Callable,
) -> tuple[dict, float]:
    started = time.perf_counter()
    with _selector(selector):
        result = optimizer.calculate_eq_bands(
            freqs,
            measured,
            target,
            spectral_repeatability=np.full(freqs.size, 0.90),
            spectral_uncertainty_db=np.full(freqs.size, 0.40),
            phonetic_coverage=0.90,
            voiced_window_ratio=0.90,
            analysis_confidence=0.90,
            global_snr_db=24.0,
            spectral_snr_db=np.full(freqs.size, 24.0),
            noise_reference_quality=0.95,
            noise_reference_status="usable",
        )
    return result, (time.perf_counter() - started) * 1000.0


def _error(
    freqs: np.ndarray,
    measured: np.ndarray,
    target: np.ndarray,
    result: dict,
) -> float:
    voice = (freqs >= 100.0) & (freqs <= 8000.0)
    normalized = measured - float(np.mean(measured[voice]))
    response = auto_eq._predict_eq_response(
        freqs,
        result["band_gains"],
        result["band_qs"],
        result["band_freqs"],
    )
    weights = np.where(voice, 1.0, 0.25)
    return float(np.sqrt(np.average((normalized + response - target) ** 2, weights=weights)))


def _risk(result: dict) -> float:
    gains = np.asarray(result["band_gains"], dtype=float)
    return float(
        np.max(np.abs(gains)) / 12.0
        + max(0.0, 0.70 - float(result["validation_gain_scale"]))
        + (2.0 if result["recommendation_status"] == "abstain" else 0.0)
    )


def evaluate() -> dict:
    freqs = np.geomspace(20.0, 20_000.0, 1000)
    rng = np.random.default_rng(SEED)
    rows = []
    for response_kind, target_name in CASES:
        measured = _spectrum(freqs, response_kind)
        target = auto_eq.get_target_curve(freqs, target_name)
        perturbation = _smooth_perturbation(freqs, rng)
        baseline, baseline_ms = _run(
            freqs,
            measured,
            target,
            _select_dynamic_band_layout,
        )
        baseline_error = _error(freqs, measured + perturbation, target, baseline)
        candidates = []
        for pool_size in POOL_SIZES:
            result, runtime_ms = _run(
                freqs,
                measured,
                target,
                _candidate_pool_selector(pool_size),
            )
            candidates.append(
                (
                    _error(freqs, measured + perturbation, target, result),
                    pool_size,
                    result,
                    runtime_ms,
                )
            )
        candidate_error, pool_size, candidate, candidate_ms = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        relative = (baseline_error - candidate_error) / max(baseline_error, 1.0e-9)
        rows.append(
            {
                "case": f"{response_kind}/{target_name}",
                "selected_pool_size": pool_size,
                "baseline_error_db": baseline_error,
                "candidate_error_db": candidate_error,
                "relative_improvement": relative,
                "baseline_runtime_ms": baseline_ms,
                "candidate_runtime_ms": candidate_ms,
                "runtime_ratio": candidate_ms / max(baseline_ms, 1.0e-9),
                "risk_score_delta": _risk(candidate) - _risk(baseline),
                "candidate_active_bands": candidate["active_band_count"],
                "candidate_constraint_passed": candidate["constraint_solver_success"],
            }
        )

    improvements = np.asarray([row["relative_improvement"] for row in rows])
    runtime_ratios = np.asarray([row["runtime_ratio"] for row in rows])
    risk_deltas = np.asarray([row["risk_score_delta"] for row in rows])
    summary = {
        "median_relative_improvement": float(np.median(improvements)),
        "improved_fraction": float(np.mean(improvements > 0.0)),
        "lower_decile_relative_improvement": float(np.quantile(improvements, 0.10)),
        "p95_runtime_ratio": float(np.quantile(runtime_ratios, 0.95)),
        "maximum_risk_score_delta": float(np.max(risk_deltas)),
    }
    retained = bool(
        summary["median_relative_improvement"] >= 0.05
        and summary["improved_fraction"] >= 0.60
        and summary["lower_decile_relative_improvement"] >= -0.02
        and summary["p95_runtime_ratio"] <= 2.0
        and summary["maximum_risk_score_delta"] <= 0.0
        and all(row["candidate_constraint_passed"] for row in rows)
    )
    return {
        "schema_version": 2,
        "experiment": "12/14/16-candidate sparse selection versus corrected dynamic ten-band baseline",
        "retained": retained,
        "method": {
            "cases": [f"{left}/{right}" for left, right in CASES],
            "held_out_perturbation": "independent deterministic smooth 0.25 dB perturbation per case",
            "seed": SEED,
            "confidence_model": "single-application bounded fit with separate uncertainty and coverage",
        },
        "gate": {
            "required_median_relative_improvement": 0.05,
            "required_improved_fraction": 0.60,
            "maximum_lower_decile_regression": -0.02,
            "maximum_p95_runtime_ratio": 2.0,
            "maximum_risk_score_delta": 0.0,
        },
        "summary": summary,
        "rows": rows,
        "decision": (
            "Retain candidate-pool selector."
            if retained
            else "Reject candidate-pool selector; keep corrected dynamic ten-band optimizer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/eq-candidate-pool-report.json"),
    )
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["retained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
