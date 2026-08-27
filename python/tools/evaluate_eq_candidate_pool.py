"""Rerun the sparse Auto-EQ candidate-pool experiment after confidence redesign."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator
from release_provenance import sha256_file as _sha256

import numpy as np

from mic_eq.analysis import auto_eq
from mic_eq.analysis.auto_eq_parts import optimizer
from mic_eq.analysis.auto_eq_parts.constants import (
    GAIN_MAX_DB,
    MAX_ADJ_GAIN_DIFF_DB,
    MAX_GAIN_SLOPE_DB_PER_OCTAVE,
    NUM_EQ_BANDS,
    Q_MAX,
    Q_MIN,
)
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
RUNTIME_REPEATS = 3
REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = {
    "required_median_relative_improvement": 0.05,
    "required_improved_fraction": 0.60,
    "maximum_lower_decile_regression": -0.02,
    "maximum_p95_runtime_ratio": 2.0,
    "maximum_risk_score_delta": 0.0,
}


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


def _candidate_pool_selector(
    pool_size: int,
    audit: list[dict[str, object]] | None = None,
) -> Callable:
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
        extrema.sort(
            key=lambda index: abs(float(residual_db[index])) * weights[index],
            reverse=True,
        )
        candidates = [float(dense_freqs[index]) for index in extrema]
        candidates.extend(
            float(value)
            for value in np.geomspace(200.0, 9000.0, max(10, max(POOL_SIZES) * 2))
        )
        production_interiors = [float(value) for value in production_centers[1:-1]]
        unique = list(production_interiors)
        for center in candidates:
            if all(abs(np.log2(center / existing)) >= 0.08 for existing in unique):
                unique.append(center)
            if len(unique) >= max(POOL_SIZES) - 2:
                break
        unique = unique[: pool_size - 2]
        if audit is not None:
            audit.append(
                {
                    "baseline_contained": all(
                        any(math.isclose(center, value) for value in unique)
                        for center in production_interiors
                    ),
                    "pool_centers_hz": list(unique),
                }
            )

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

        def fit_error(indices_to_fit: list[int]) -> float:
            gains, *_ = np.linalg.lstsq(
                weighted_basis[:, indices_to_fit],
                weighted_target,
                rcond=None,
            )
            gains = np.clip(gains, -GAIN_MAX_DB, GAIN_MAX_DB)
            return float(
                np.sqrt(
                    np.average(
                        (
                            residual_db
                            - basis[:, indices_to_fit] @ gains
                        )
                        ** 2,
                        weights=weights,
                    )
                )
            )

        incumbent_indices = list(range(NUM_EQ_BANDS - 2))
        if fit_error(selected) >= fit_error(incumbent_indices):
            selected = incumbent_indices

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


def _bench_run(
    freqs: np.ndarray,
    measured: np.ndarray,
    target: np.ndarray,
    selector: Callable,
) -> tuple[dict, float]:
    _run(freqs, measured, target, selector)
    results: list[tuple[dict, float]] = [
        _run(freqs, measured, target, selector) for _ in range(RUNTIME_REPEATS)
    ]
    return results[-1][0], float(np.median([row[1] for row in results]))


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
    return float(
        np.sqrt(np.average((normalized + response - target) ** 2, weights=weights))
    )


def _risk(result: dict) -> float:
    gains = np.asarray(result["band_gains"], dtype=float)
    return float(
        np.max(np.abs(gains)) / 12.0
        + max(0.0, 0.70 - float(result["validation_gain_scale"]))
        + (2.0 if result["recommendation_status"] == "abstain" else 0.0)
    )


def _constraints_passed(result: dict) -> bool:
    gains = np.asarray(result["band_gains"], dtype=float)
    qs = np.asarray(result["band_qs"], dtype=float)
    q_upper_bounds = np.asarray(result["q_upper_bounds"], dtype=float)
    return bool(
        np.all(np.isfinite(gains))
        and np.all(np.isfinite(qs))
        and np.all(np.abs(gains) <= GAIN_MAX_DB + 1.0e-9)
        and np.all(qs >= Q_MIN - 1.0e-9)
        and np.all(qs <= q_upper_bounds + 1.0e-9)
        and float(result["max_adjacent_gain_difference_db"])
        <= MAX_ADJ_GAIN_DIFF_DB + 1.0e-9
        and float(result["max_adjacent_gain_slope_db_per_octave"])
        <= MAX_GAIN_SLOPE_DB_PER_OCTAVE + 1.0e-9
    )


def _summary(rows: list[dict]) -> dict[str, float]:
    improvements = np.asarray([row["relative_improvement"] for row in rows])
    runtime_ratios = np.asarray([row["runtime_ratio"] for row in rows])
    risk_deltas = np.asarray([row["risk_score_delta"] for row in rows])
    return {
        "median_relative_improvement": float(np.median(improvements)),
        "improved_fraction": float(np.mean(improvements > 0.0)),
        "lower_decile_relative_improvement": float(np.quantile(improvements, 0.10)),
        "p95_runtime_ratio": float(np.quantile(runtime_ratios, 0.95)),
        "maximum_risk_score_delta": float(np.max(risk_deltas)),
    }


def _gate(rows: list[dict], summary: dict[str, float]) -> dict[str, bool]:
    return {
        "median_improvement": summary["median_relative_improvement"]
        >= GATE["required_median_relative_improvement"],
        "improved_fraction": summary["improved_fraction"]
        >= GATE["required_improved_fraction"],
        "lower_tail": summary["lower_decile_relative_improvement"]
        >= GATE["maximum_lower_decile_regression"],
        "runtime": summary["p95_runtime_ratio"] <= GATE["maximum_p95_runtime_ratio"],
        "risk": summary["maximum_risk_score_delta"] <= GATE["maximum_risk_score_delta"],
        "constraints": all(row["candidate_constraint_passed"] for row in rows),
        "baseline_containment": all(row["baseline_contained"] for row in rows),
        "nested_candidate_spaces": all(
            row["nested_candidate_spaces"] for row in rows
        ),
    }


def _select_fixed_variant(variants: dict[str, dict]) -> int | None:
    passing = sorted(
        int(pool_size)
        for pool_size, result in variants.items()
        if all(result["checks"].values())
    )
    return passing[0] if passing else None


def evaluate() -> dict:
    freqs = np.geomspace(20.0, 20_000.0, 1000)
    rng = np.random.default_rng(SEED)
    rows_by_pool: dict[int, list[dict]] = {pool_size: [] for pool_size in POOL_SIZES}
    for response_kind, target_name in CASES:
        measured = _spectrum(freqs, response_kind)
        target = auto_eq.get_target_curve(freqs, target_name)
        perturbation = _smooth_perturbation(freqs, rng)
        baseline, baseline_ms = _bench_run(
            freqs,
            measured,
            target,
            _select_dynamic_band_layout,
        )
        baseline_error = _error(freqs, measured + perturbation, target, baseline)
        for pool_size in POOL_SIZES:
            audit: list[dict[str, object]] = []
            candidate, candidate_ms = _bench_run(
                freqs,
                measured,
                target,
                _candidate_pool_selector(pool_size, audit),
            )
            structure = audit[-1]
            candidate_error = _error(
                freqs,
                measured + perturbation,
                target,
                candidate,
            )
            relative = (baseline_error - candidate_error) / max(baseline_error, 1.0e-9)
            rows_by_pool[pool_size].append(
                {
                    "case": f"{response_kind}/{target_name}",
                    "pool_size": pool_size,
                    "baseline_error_db": baseline_error,
                    "candidate_error_db": candidate_error,
                    "relative_improvement": relative,
                    "baseline_runtime_median_ms": baseline_ms,
                    "candidate_runtime_median_ms": candidate_ms,
                    "runtime_ratio": candidate_ms / max(baseline_ms, 1.0e-9),
                    "risk_score_delta": _risk(candidate) - _risk(baseline),
                    "candidate_active_bands": candidate["active_band_count"],
                    "candidate_constraint_passed": _constraints_passed(candidate),
                    "solver_reported_success": candidate["constraint_solver_success"],
                    "baseline_contained": structure["baseline_contained"],
                    "pool_centers_hz": structure["pool_centers_hz"],
                }
            )
        case_rows = [rows_by_pool[size][-1] for size in POOL_SIZES]
        nested = all(
            set(case_rows[index]["pool_centers_hz"]).issubset(
                case_rows[index + 1]["pool_centers_hz"]
            )
            for index in range(len(case_rows) - 1)
        )
        for row in case_rows:
            row["nested_candidate_spaces"] = nested

    variants: dict[str, dict] = {}
    for pool_size, rows in rows_by_pool.items():
        summary = _summary(rows)
        variants[str(pool_size)] = {
            "summary": summary,
            "checks": _gate(rows, summary),
            "rows": rows,
        }
    selected_pool_size = _select_fixed_variant(variants)
    retained = selected_pool_size is not None
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "python/mic_eq/analysis/auto_eq_parts/constants.py",
        REPO_ROOT / "python/mic_eq/analysis/auto_eq_parts/dynamic_bands.py",
        REPO_ROOT / "python/mic_eq/analysis/auto_eq_parts/optimizer.py",
        REPO_ROOT / "python/mic_eq/analysis/auto_eq_parts/response.py",
    )
    source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in source_paths
    }
    worst_runtime_ratio = max(
        float(variant["summary"]["p95_runtime_ratio"])
        for variant in variants.values()
    )
    return {
        "schema_version": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audible_change": True,
        "experiment": "Nested 12/14/16-candidate sparse selectors containing the dynamic ten-band baseline",
        "retained": retained,
        "selected_pool_size": selected_pool_size,
        "method": {
            "cases": [f"{left}/{right}" for left, right in CASES],
            "held_out_perturbation": "one deterministic 0.25 dB perturbation per case, never used to choose a different pool size per case",
            "variant_selection": "smallest fixed pool size passing every aggregate gate",
            "candidate_space": "each larger pool is a prefix extension and contains all eight incumbent interior centers",
            "runtime_repeats": RUNTIME_REPEATS,
            "runtime_warmup_runs": 1,
            "seed": SEED,
            "confidence_model": "single in-solver gain feasibility with separate uncertainty and coverage",
        },
        "gate": GATE,
        "variants": variants,
        "decision": (
            f"Retain fixed {selected_pool_size}-candidate selector."
            if retained
            else "Reject the three tested nested selectors; keep the dynamic ten-band optimizer."
        ),
        "evaluation_contract": {
            "configuration": {
                "cases": [f"{left}/{right}" for left, right in CASES],
                "pool_sizes": list(POOL_SIZES),
                "runtime_repeats": RUNTIME_REPEATS,
                "runtime_warmup_runs": 1,
                "seed": SEED,
                "scope": "evaluation-only synthetic response fitting",
            },
            "asset_hashes": {
                "source": source_hashes,
                "corpus": "deterministic in-process synthetic response family",
            },
            "runtime": {
                "maximum_candidate_p95_runtime_ratio": worst_runtime_ratio,
                "max_p99_frame_seconds": None,
                "max_p99_frame_seconds_reason": (
                    "The optimizer is an offline whole-fit operation, not a realtime "
                    "audio callback; median fit times and P95 ratios are measured."
                ),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "latency": {
                "algorithmic_latency_samples": 0,
                "reason": "The experiment fits static EQ coefficients offline.",
            },
            "clean_preservation": {
                "audio_rendered": False,
                "maximum_risk_score_delta": max(
                    float(variant["summary"]["maximum_risk_score_delta"])
                    for variant in variants.values()
                ),
                "constraint_checks_recorded_per_variant": True,
            },
        },
        "provenance": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_hashes": source_hashes,
        },
        "limitations": [
            "The synthetic response family is a deterministic algorithmic regression set, not a perceptual listening panel.",
            "The same fixed variant is assessed across every case; no per-case oracle selection is permitted.",
            "This result applies only to the tested nested selectors, not every possible larger candidate-pool design.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/eq-candidate-pool-report.json"),
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-case report; the tracked report stays compact.",
    )
    args = parser.parse_args()
    report = evaluate()
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        args.details_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    for variant in report["variants"].values():
        variant.pop("rows", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "retained": report["retained"],
                "selected_pool_size": report["selected_pool_size"],
                "summaries": {
                    key: value["summary"] for key, value in report["variants"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
