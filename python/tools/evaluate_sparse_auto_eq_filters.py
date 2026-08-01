"""Evaluate sparse/type-selecting Auto-EQ without changing the product path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from mic_eq import (
    eq_magnitude_response_v2,
    simulate_auto_eq_chain,
    simulate_eq_v2,
)
from mic_eq.analysis.auto_eq import analyze_auto_eq
from mic_eq.analysis.auto_eq_parts.dynamic_bands import _voice_weights
from mic_eq.analysis.auto_eq_parts.target import get_target_curve
from mic_eq.analysis.spectrum import (
    analyze_voice_spectrum,
    smooth_spectrum_perceptual,
)
from mic_eq.analysis.vad import analyze_offline_vad
from mic_eq.analysis.wav_io import read_mono_wav


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "models" / "cross_take_eval"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "sparse-auto-eq-filter-report.json"
SAMPLE_RATE = 48_000
GRID_POINTS = 384
MIN_ACTIVE_GAIN_DB = 0.25
SECTION_PENALTY_DB = 0.04
NOTCH_PENALTY_DB = 0.08
MIN_OPERATION_IMPROVEMENT_DB = 1.0e-6
LOW_SHELF_MAX_HZ = 500.0
HIGH_SHELF_MIN_HZ = 3_000.0
NOTCH_MAX_GAIN_DB = -6.0
NOTCH_MIN_Q = 3.0
NOTCH_MIN_CONFIDENCE = 0.65
SEPARATOR_SECONDS = 0.25
TIMING_REPEATS = 5

GATES: dict[str, float | int] = {
    "min_comparable_cases": 20,
    "min_median_heldout_improvement_db": 0.0,
    "min_p10_heldout_improvement_db": -0.35,
    "max_median_stability_regression_db": 0.10,
    "max_p90_stability_regression_db": 0.25,
    "min_median_active_section_reduction": 1.0,
    "max_true_peak_regression_db": 0.50,
    "max_p95_limiter_gr_regression_db": 0.50,
    "max_full_chain_true_peak_overshoot_db": 0.05,
    "max_p95_runtime_ratio": 1.10,
    "max_candidate_p95_realtime_factor": 0.01,
}

TypedBand: TypeAlias = tuple[str, float, float, float, int, bool]


@dataclass(frozen=True, slots=True)
class AnalysisView:
    grid_hz: np.ndarray
    target_residual_db: np.ndarray
    weights: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = read_mono_wav(
        path,
        allow_stereo=False,
        dtype=np.float32,
    )
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path.name} must be native-48-kHz mono")
    if audio.size < SAMPLE_RATE:
        raise ValueError(f"{path.name} is too short or non-finite")
    return int(sample_rate), audio


def _grouped_cases(
    corpus_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["speaker_disjoint_splits"]
    split_by_speaker = {
        str(speaker): split
        for split in ("validation", "test")
        for speaker in splits[split]
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in manifest["pairs"]:
        speaker = str(pair["speaker"])
        if speaker not in split_by_speaker:
            continue
        key = (speaker, str(pair["statement_id"]))
        grouped.setdefault(key, []).append(pair)

    expected_deliveries = {"neutral-normal", "calm-normal", "calm-strong"}
    cases: list[dict[str, Any]] = []
    for (speaker, statement_id), pairs in sorted(grouped.items()):
        if {str(pair["delivery"]) for pair in pairs} != expected_deliveries:
            raise RuntimeError(
                f"{speaker}/{statement_id} lacks the three delivery conditions"
            )
        first_parts: list[np.ndarray] = []
        second_parts: list[np.ndarray] = []
        separator = np.zeros(
            int(round(SAMPLE_RATE * SEPARATOR_SECONDS)),
            dtype=np.float32,
        )
        source_ids: list[str] = []
        source_hashes: list[str] = []
        for pair in sorted(pairs, key=lambda item: str(item["delivery"])):
            first_path = corpus_root / pair["takes"]["01"]["path"]
            second_path = corpus_root / pair["takes"]["02"]["path"]
            first_rate, first_part = _read_mono(first_path)
            second_rate, second_part = _read_mono(second_path)
            if first_rate != SAMPLE_RATE or second_rate != SAMPLE_RATE:
                raise RuntimeError(f"{pair['id']} sample-rate mismatch")
            first_parts.extend((first_part, separator))
            second_parts.extend((second_part, separator))
            source_ids.append(str(pair["id"]))
            source_hashes.extend(
                (
                    str(pair["takes"]["01"]["sha256"]),
                    str(pair["takes"]["02"]["sha256"]),
                )
            )
        cases.append(
            {
                "id": f"{speaker}-statement-{statement_id}",
                "speaker": speaker,
                "speaker_sex": str(pairs[0]["speaker_sex"]),
                "statement_id": statement_id,
                "split": split_by_speaker[speaker],
                "source_pair_ids": source_ids,
                "source_hashes": source_hashes,
                "first": np.concatenate(first_parts[:-1]),
                "second": np.concatenate(second_parts[:-1]),
            }
        )
    return cases, manifest


def _analysis_view(
    audio: np.ndarray,
    vad_probabilities: np.ndarray | None,
) -> AnalysisView:
    spectrum = analyze_voice_spectrum(
        audio,
        SAMPLE_RATE,
        vad_probabilities=vad_probabilities,
    )
    measured = smooth_spectrum_perceptual(
        spectrum.freqs,
        spectrum.median_spectrum_db,
        strength="conservative",
    )
    voice = (spectrum.freqs >= 100.0) & (spectrum.freqs <= 8_000.0)
    level = float(np.mean(measured[voice])) if np.any(voice) else float(np.mean(measured))
    measured_normalized = measured - level
    target = get_target_curve(
        spectrum.freqs,
        "broadcast",
        measured_db=measured,
        target_mode="adaptive",
    )
    upper_hz = min(16_000.0, float(np.max(spectrum.freqs)))
    grid = np.geomspace(80.0, upper_hz, GRID_POINTS)
    residual = np.interp(grid, spectrum.freqs, target - measured_normalized)
    weights = _voice_weights(grid)
    return AnalysisView(grid, residual, weights)


def _typed_incumbent(settings: dict[str, Any]) -> list[TypedBand]:
    gains = np.asarray(settings["band_gains"], dtype=float)
    qs = np.asarray(settings["band_qs"], dtype=float)
    frequencies = np.asarray(settings["band_freqs"], dtype=float)
    if not (gains.size == qs.size == frequencies.size == 10):
        raise ValueError("incumbent Auto-EQ result must contain ten bands")
    bands: list[TypedBand] = []
    for index, (frequency, gain, q) in enumerate(
        zip(frequencies, gains, qs, strict=True)
    ):
        filter_type = (
            "low_shelf"
            if index == 0
            else "high_shelf"
            if index == 9
            else "bell"
        )
        bands.append(
            (
                filter_type,
                float(frequency),
                float(gain),
                float(q),
                12,
                bool(abs(gain) >= MIN_ACTIVE_GAIN_DB),
            )
        )
    return bands


def _response(grid_hz: np.ndarray, bands: list[TypedBand]) -> np.ndarray:
    response = np.asarray(
        eq_magnitude_response_v2(
            grid_hz.tolist(),
            bands,
            float(SAMPLE_RATE),
        ),
        dtype=np.float64,
    )
    if response.shape != grid_hz.shape or not np.all(np.isfinite(response)):
        raise RuntimeError("native EQ response was non-finite or malformed")
    return response


def _weighted_error(view: AnalysisView, bands: list[TypedBand]) -> float:
    error = view.target_residual_db - _response(view.grid_hz, bands)
    return float(
        np.sqrt(
            np.sum(view.weights * np.square(error))
            / max(float(np.sum(view.weights)), 1.0e-12)
        )
    )


def _active_count(bands: list[TypedBand]) -> int:
    return sum(bool(band[5]) for band in bands)


def _objective(view: AnalysisView, bands: list[TypedBand]) -> float:
    active = [band for band in bands if band[5]]
    notch_count = sum(band[0] == "notch" for band in active)
    return (
        _weighted_error(view, bands)
        + SECTION_PENALTY_DB * len(active)
        + NOTCH_PENALTY_DB * notch_count
    )


def _eligible_replacements(
    index: int,
    band: TypedBand,
    confidence: float,
) -> list[str]:
    filter_type, frequency, gain, q, _slope, enabled = band
    if not enabled:
        return []
    replacements: list[str] = []
    if frequency <= LOW_SHELF_MAX_HZ and filter_type != "low_shelf":
        replacements.append("low_shelf")
    if frequency >= HIGH_SHELF_MIN_HZ and filter_type != "high_shelf":
        replacements.append("high_shelf")
    if (
        filter_type != "notch"
        and gain <= NOTCH_MAX_GAIN_DB
        and q >= NOTCH_MIN_Q
        and confidence >= NOTCH_MIN_CONFIDENCE
    ):
        replacements.append("notch")
    return sorted(set(replacements))


def _candidate_operations(
    bands: list[TypedBand],
    confidences: np.ndarray,
) -> list[tuple[int, str]]:
    operations: list[tuple[int, str]] = []
    for index, band in enumerate(bands):
        if band[5]:
            operations.append((index, "disable"))
        for replacement in _eligible_replacements(
            index,
            band,
            float(confidences[index]),
        ):
            operations.append((index, replacement))
    return sorted(operations)


def _apply_operation(
    bands: list[TypedBand],
    operation: tuple[int, str],
) -> list[TypedBand]:
    index, action = operation
    candidate = list(bands)
    filter_type, frequency, gain, q, slope, enabled = candidate[index]
    if action == "disable":
        candidate[index] = (filter_type, frequency, gain, q, slope, False)
    else:
        candidate[index] = (action, frequency, gain, q, slope, enabled)
    return candidate


def _select_sparse_candidate(
    view: AnalysisView,
    incumbent: list[TypedBand],
    confidences: np.ndarray,
) -> tuple[list[TypedBand], list[dict[str, Any]]]:
    if confidences.shape != (10,) or not np.all(np.isfinite(confidences)):
        raise ValueError("band confidences must contain ten finite values")
    selected = list(incumbent)
    selected_objective = _objective(view, selected)
    trace: list[dict[str, Any]] = []
    while True:
        best: tuple[float, tuple[int, str], list[TypedBand]] | None = None
        for operation in _candidate_operations(selected, confidences):
            trial = _apply_operation(selected, operation)
            trial_objective = _objective(view, trial)
            improvement = selected_objective - trial_objective
            if improvement <= MIN_OPERATION_IMPROVEMENT_DB:
                continue
            proposal = (trial_objective, operation, trial)
            if best is None or proposal[:2] < best[:2]:
                best = proposal
        if best is None:
            break
        previous_objective = selected_objective
        selected_objective, operation, selected = best
        trace.append(
            {
                "band_index": operation[0],
                "operation": operation[1],
                "objective_before_db": previous_objective,
                "objective_after_db": selected_objective,
                "improvement_db": previous_objective - selected_objective,
            }
        )
    # A final native call is both a response computation and a strict config
    # validation. It prevents an evaluation-only shortcut around runtime rules.
    _response(view.grid_hz, selected)
    return selected, trace


def _response_disagreement_db(
    grid_hz: np.ndarray,
    first: list[TypedBand],
    second: list[TypedBand],
) -> float:
    difference = _response(grid_hz, first) - _response(grid_hz, second)
    return float(np.sqrt(np.mean(np.square(difference))))


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(
        20.0
        * math.log10(
            max(float(numerator), 1.0e-12)
            / max(float(denominator), 1.0e-12)
        )
    )


def _normalized_for_headroom(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1.0e-9:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio * (0.5 / peak), dtype=np.float32)


def _render_metrics(
    audio: np.ndarray,
    bands: list[TypedBand],
    legacy_bands: list[tuple[float, float, float]],
) -> dict[str, Any]:
    normalized = _normalized_for_headroom(audio)
    # Warm up one complete native call before collecting repeated timings.
    simulate_eq_v2(normalized, float(SAMPLE_RATE), bands)
    simulations = [
        simulate_eq_v2(normalized, float(SAMPLE_RATE), bands)
        for _ in range(TIMING_REPEATS)
    ]
    runtime_ms = np.asarray(
        [float(result["runtime_ms"]) for result in simulations],
        dtype=float,
    )
    representative = simulations[-1]
    chain = simulate_auto_eq_chain(
        normalized,
        float(SAMPLE_RATE),
        legacy_bands,
        {
            "eq_bands_v2": bands,
            "deesser_enabled": False,
            "compressor_enabled": False,
            "limiter_enabled": True,
            "limiter_ceiling_db": -1.0,
            "limiter_careful_output_enabled": True,
        },
    )
    duration_seconds = normalized.size / SAMPLE_RATE
    return {
        "input_peak_normalization": 0.5,
        "output_true_peak": float(representative["output_true_peak"]),
        "output_rms": float(representative["output_rms"]),
        "non_finite_output": bool(representative["non_finite_output"]),
        "algorithmic_latency_samples": int(
            representative["algorithmic_latency_samples"]
        ),
        "runtime_ms_median": float(np.median(runtime_ms)),
        "runtime_ms_p95": float(np.percentile(runtime_ms, 95)),
        "realtime_factor": float(np.median(runtime_ms))
        / max(duration_seconds * 1000.0, 1.0e-12),
        "full_chain_non_finite_output": bool(chain["non_finite_output"]),
        "full_chain_output_true_peak_db": float(chain["output_true_peak_db"]),
        "full_chain_ceiling_db": float(chain["limiter_effective_ceiling_db"]),
        "full_chain_limiter_gr_db": max(
            float(chain["limiter_gain_reduction_db"]),
            float(chain["true_peak_limiter_gain_reduction_db"]),
        ),
    }


def _settings_summary(bands: list[TypedBand]) -> list[dict[str, Any]]:
    return [
        {
            "filter_type": filter_type,
            "frequency_hz": frequency,
            "gain_db": gain,
            "q": q,
            "slope_db_per_octave": slope,
            "enabled": enabled,
        }
        for filter_type, frequency, gain, q, slope, enabled in bands
    ]


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    first = np.asarray(case["first"], dtype=np.float32)
    second = np.asarray(case["second"], dtype=np.float32)
    started = time.perf_counter()
    first_vad, first_backend = analyze_offline_vad(first, SAMPLE_RATE)
    second_vad, second_backend = analyze_offline_vad(second, SAMPLE_RATE)
    first_view = _analysis_view(first, first_vad)
    second_view = _analysis_view(second, second_vad)

    first_settings, _first_validation = analyze_auto_eq(
        first,
        SAMPLE_RATE,
        "broadcast",
        vad_probabilities=first_vad,
    )
    second_settings, _second_validation = analyze_auto_eq(
        second,
        SAMPLE_RATE,
        "broadcast",
        vad_probabilities=second_vad,
    )
    first_incumbent = _typed_incumbent(first_settings)
    second_incumbent = _typed_incumbent(second_settings)
    first_candidate, first_trace = _select_sparse_candidate(
        first_view,
        first_incumbent,
        np.asarray(first_settings["band_confidences"], dtype=float),
    )
    second_candidate, _second_trace = _select_sparse_candidate(
        second_view,
        second_incumbent,
        np.asarray(second_settings["band_confidences"], dtype=float),
    )

    incumbent_error = _weighted_error(second_view, first_incumbent)
    candidate_error = _weighted_error(second_view, first_candidate)
    legacy_bands = [
        (float(frequency), float(gain), float(q))
        for frequency, gain, q in zip(
            first_settings["band_freqs"],
            first_settings["band_gains"],
            first_settings["band_qs"],
            strict=True,
        )
    ]
    incumbent_render = _render_metrics(second, first_incumbent, legacy_bands)
    candidate_render = _render_metrics(second, first_candidate, legacy_bands)
    true_peak_delta_db = _db_ratio(
        candidate_render["output_true_peak"],
        incumbent_render["output_true_peak"],
    )
    result = {
        "id": str(case["id"]),
        "speaker": str(case["speaker"]),
        "speaker_sex": str(case["speaker_sex"]),
        "statement_id": str(case["statement_id"]),
        "split": str(case["split"]),
        "source_pair_ids": list(case["source_pair_ids"]),
        "source_hashes": list(case["source_hashes"]),
        "first_duration_seconds": first.size / SAMPLE_RATE,
        "second_duration_seconds": second.size / SAMPLE_RATE,
        "vad_backends": [first_backend, second_backend],
        "incumbent": {
            "heldout_target_error_db": incumbent_error,
            "active_sections": _active_count(first_incumbent),
            "cross_take_response_disagreement_db": _response_disagreement_db(
                first_view.grid_hz,
                first_incumbent,
                second_incumbent,
            ),
            "render": incumbent_render,
            "bands": _settings_summary(first_incumbent),
        },
        "candidate": {
            "heldout_target_error_db": candidate_error,
            "heldout_improvement_db": incumbent_error - candidate_error,
            "training_target_error_db": _weighted_error(
                first_view,
                first_candidate,
            ),
            "training_objective_db": _objective(first_view, first_candidate),
            "active_sections": _active_count(first_candidate),
            "active_section_reduction": _active_count(first_incumbent)
            - _active_count(first_candidate),
            "cross_take_response_disagreement_db": _response_disagreement_db(
                first_view.grid_hz,
                first_candidate,
                second_candidate,
            ),
            "render": candidate_render,
            "bands": _settings_summary(first_candidate),
            "selection_trace": first_trace,
            "true_peak_regression_db": true_peak_delta_db,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    result["candidate"]["stability_regression_db"] = (
        result["candidate"]["cross_take_response_disagreement_db"]
        - result["incumbent"]["cross_take_response_disagreement_db"]
    )
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = [
        float(row["candidate"]["heldout_improvement_db"]) for row in rows
    ]
    stability = [
        float(row["candidate"]["stability_regression_db"]) for row in rows
    ]
    section_reductions = [
        float(row["candidate"]["active_section_reduction"]) for row in rows
    ]
    true_peak_regressions = [
        float(row["candidate"]["true_peak_regression_db"]) for row in rows
    ]
    incumbent_limiter = [
        float(row["incumbent"]["render"]["full_chain_limiter_gr_db"])
        for row in rows
    ]
    candidate_limiter = [
        float(row["candidate"]["render"]["full_chain_limiter_gr_db"])
        for row in rows
    ]
    incumbent_rtfs = [
        float(row["incumbent"]["render"]["realtime_factor"])
        for row in rows
    ]
    candidate_rtfs = [
        float(row["candidate"]["render"]["realtime_factor"])
        for row in rows
    ]
    incumbent_p95 = _percentile(incumbent_rtfs, 95) or 0.0
    candidate_p95 = _percentile(candidate_rtfs, 95) or 0.0
    type_counts: Counter[str] = Counter()
    for row in rows:
        for band in row["candidate"]["bands"]:
            if band["enabled"]:
                type_counts[str(band["filter_type"])] += 1
    full_chain_overshoots = [
        max(
            0.0,
            float(candidate["render"]["full_chain_output_true_peak_db"])
            - float(candidate["render"]["full_chain_ceiling_db"]),
        )
        for candidate in (row["candidate"] for row in rows)
    ]
    return {
        "comparable_cases": len(rows),
        "median_heldout_improvement_db": _percentile(improvements, 50),
        "p10_heldout_improvement_db": _percentile(improvements, 10),
        "median_stability_regression_db": _percentile(stability, 50),
        "p90_stability_regression_db": _percentile(stability, 90),
        "median_active_section_reduction": _percentile(
            section_reductions,
            50,
        ),
        "max_true_peak_regression_db": max(true_peak_regressions, default=math.inf),
        "incumbent_p95_limiter_gr_db": _percentile(incumbent_limiter, 95),
        "candidate_p95_limiter_gr_db": _percentile(candidate_limiter, 95),
        "p95_limiter_gr_regression_db": (
            (_percentile(candidate_limiter, 95) or 0.0)
            - (_percentile(incumbent_limiter, 95) or 0.0)
        ),
        "incumbent_p95_realtime_factor": incumbent_p95,
        "candidate_p95_realtime_factor": candidate_p95,
        "p95_runtime_ratio": candidate_p95 / max(incumbent_p95, 1.0e-12),
        "max_full_chain_true_peak_overshoot_db": max(
            full_chain_overshoots,
            default=math.inf,
        ),
        "all_outputs_finite": all(
            not bool(row[path]["render"][key])
            for row in rows
            for path in ("incumbent", "candidate")
            for key in ("non_finite_output", "full_chain_non_finite_output")
        ),
        "latency_samples": sorted(
            {
                int(row[path]["render"]["algorithmic_latency_samples"])
                for row in rows
                for path in ("incumbent", "candidate")
            }
        ),
        "selected_filter_type_counts": dict(sorted(type_counts.items())),
    }


def _gate(aggregate: dict[str, Any]) -> dict[str, bool]:
    median_improvement = aggregate["median_heldout_improvement_db"]
    p10_improvement = aggregate["p10_heldout_improvement_db"]
    median_stability = aggregate["median_stability_regression_db"]
    p90_stability = aggregate["p90_stability_regression_db"]
    median_reduction = aggregate["median_active_section_reduction"]
    return {
        "enough_comparable_cases": aggregate["comparable_cases"]
        >= GATES["min_comparable_cases"],
        "median_heldout_noninferior": median_improvement is not None
        and median_improvement >= GATES["min_median_heldout_improvement_db"],
        "lower_decile_heldout_noninferior": p10_improvement is not None
        and p10_improvement >= GATES["min_p10_heldout_improvement_db"],
        "median_cross_take_stability": median_stability is not None
        and median_stability <= GATES["max_median_stability_regression_db"],
        "upper_tail_cross_take_stability": p90_stability is not None
        and p90_stability <= GATES["max_p90_stability_regression_db"],
        "material_sparsity": median_reduction is not None
        and median_reduction >= GATES["min_median_active_section_reduction"],
        "clean_output_finite": bool(aggregate["all_outputs_finite"]),
        "clean_true_peak": aggregate["max_true_peak_regression_db"]
        <= GATES["max_true_peak_regression_db"],
        "headroom_limiter_load": aggregate["p95_limiter_gr_regression_db"]
        <= GATES["max_p95_limiter_gr_regression_db"],
        "headroom_true_peak_ceiling": aggregate[
            "max_full_chain_true_peak_overshoot_db"
        ]
        <= GATES["max_full_chain_true_peak_overshoot_db"],
        "runtime_ratio": aggregate["p95_runtime_ratio"]
        <= GATES["max_p95_runtime_ratio"],
        "runtime_absolute": aggregate["candidate_p95_realtime_factor"]
        <= GATES["max_candidate_p95_realtime_factor"],
        "zero_added_latency": aggregate["latency_samples"] == [0],
        "native_constraints": bool(aggregate["native_constraints_valid"]),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        "docs/sparse-auto-eq-evaluation.md",
        "python/tools/evaluate_sparse_auto_eq_filters.py",
        "python/mic_eq/analysis/wav_io.py",
        "python/mic_eq/analysis/auto_eq_parts/optimizer.py",
        "python/mic_eq/analysis/auto_eq_parts/target.py",
        "rust-core/src/dsp/eq.rs",
        "rust-core/src/audio/processor/python_api.rs",
    )
    return {path: _sha256(REPO_ROOT / path) for path in paths}


def evaluate(corpus_root: Path) -> dict[str, Any]:
    corpus_root = corpus_root.resolve(strict=True)
    cases, manifest = _grouped_cases(corpus_root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        try:
            rows.append(_evaluate_case(case))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append({"id": str(case["id"]), "error": str(error)})

    aggregate = _aggregate(rows)
    aggregate["failed_cases"] = len(failures)
    aggregate["native_constraints_valid"] = not failures
    split_aggregates = {
        split: _aggregate([row for row in rows if row["split"] == split])
        for split in ("validation", "test")
    }
    checks = _gate(aggregate)
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    source_hashes = _source_hashes()
    manifest_path = corpus_root / "manifest.json"
    report = {
        "schema_version": 2,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Sparse and typed Auto-EQ selection against the incumbent",
        "candidate": {
            "scope": "evaluation_only",
            "algorithm": "deterministic greedy coordinate descent",
            "section_penalty_db": SECTION_PENALTY_DB,
            "notch_penalty_db": NOTCH_PENALTY_DB,
            "eligibility": {
                "low_shelf_max_hz": LOW_SHELF_MAX_HZ,
                "high_shelf_min_hz": HIGH_SHELF_MIN_HZ,
                "notch_max_gain_db": NOTCH_MAX_GAIN_DB,
                "notch_min_q": NOTCH_MIN_Q,
                "notch_min_confidence": NOTCH_MIN_CONFIDENCE,
            },
        },
        "incumbent": {
            "product_path": "single-take ten-band Auto-EQ",
            "filter_layout": "low shelf, eight bells, high shelf",
        },
        "retention_gates": GATES,
        "checks": checks,
        "decision": {
            "retained": not failed_checks,
            "failed_checks": failed_checks,
            "product_action": (
                "integrate candidate"
                if not failed_checks
                else "retain incumbent; candidate remains evaluation-only"
            ),
        },
        "aggregate": aggregate,
        "split_aggregates": split_aggregates,
        "failures": failures,
        "cases": rows,
        "evaluation_contract": {
            "configuration": {
                "sample_rate": SAMPLE_RATE,
                "grid_points": GRID_POINTS,
                "target": "broadcast:adaptive",
                "smoothing": "conservative",
                "separator_seconds": SEPARATOR_SECONDS,
                "timing_repeats": TIMING_REPEATS,
                "manifest_split_policy": "validation and test actors only",
                "pre_registration": "docs/sparse-auto-eq-evaluation.md",
            },
            "asset_hashes": {
                "corpus_manifest": _sha256(manifest_path),
                "source": source_hashes,
                "dataset_archive": str(manifest["archive"]["sha256"]),
            },
            "runtime": {
                "measurement": "native whole-clip simulator, warm plus five repeats",
                "incumbent_p95_realtime_factor": aggregate[
                    "incumbent_p95_realtime_factor"
                ],
                "candidate_p95_realtime_factor": aggregate[
                    "candidate_p95_realtime_factor"
                ],
                "max_p99_frame_seconds": None,
                "max_p99_frame_seconds_reason": (
                    "The native evaluation API reports whole-clip time rather than "
                    "per-callback P99; realtime factor and active sections are measured."
                ),
                "machine": platform.platform(),
                "python": platform.python_version(),
            },
            "latency": {
                "algorithmic_latency_samples": aggregate["latency_samples"],
                "sample_rate": SAMPLE_RATE,
            },
            "clean_preservation": {
                "all_outputs_finite": aggregate["all_outputs_finite"],
                "max_true_peak_regression_db": aggregate[
                    "max_true_peak_regression_db"
                ],
                "p95_limiter_gr_regression_db": aggregate[
                    "p95_limiter_gr_regression_db"
                ],
                "max_full_chain_true_peak_overshoot_db": aggregate[
                    "max_full_chain_true_peak_overshoot_db"
                ],
            },
        },
        "source_sha256": source_hashes,
        "limitations": [
            "RAVDESS is acted English speech rather than conversational multilingual audio.",
            "The repeated-take corpus measures speaker/take stability, not room or microphone diversity.",
            "No real frame/event labels are consumed by this response-selection experiment.",
            "A rejected evaluation candidate is intentionally not wired into Auto-EQ or presets.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate(args.corpus_root)
    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": _relative(report_path),
                "retained": report["decision"]["retained"],
                "failed_checks": report["decision"]["failed_checks"],
                "comparable_cases": report["aggregate"]["comparable_cases"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
