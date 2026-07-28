"""Validity-first Silero v5.1.2 versus v6.2.1 model selection.

Calibration and threshold selection use only the existing calibration split.
The fixed general held-out split and external multi-speaker child corpus are
never used to tune either model. Child uncertainty is bootstrapped by speaker,
not by utterance or frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_vad_models as core

NATIVE_CALIBRATION_A = 0.692_287_7
NATIVE_CALIBRATION_B = 0.086_123_86
BOOTSTRAP_SEED = 0x51_1E_20
BOOTSTRAP_REPETITIONS = 10_000
GENERAL_NONINFERIORITY_MARGIN = -0.01
CHILD_MACRO_F1_NONINFERIORITY_MARGIN = -0.01
CHILD_EVENT_RECALL_NONINFERIORITY_MARGIN = -0.02
MEANINGFUL_CHILD_IMPROVEMENT = 0.01


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


@dataclass(frozen=True)
class ModelCalibration:
    a: float
    b: float
    threshold: float


def _logit(probabilities: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(bounded / (1.0 - bounded))


def _invert_native_calibration(
    probabilities: np.ndarray,
    a: float = NATIVE_CALIBRATION_A,
    b: float = NATIVE_CALIBRATION_B,
) -> np.ndarray:
    if not np.isfinite(a) or a <= 0.0 or not np.isfinite(b):
        raise ValueError("native calibration must be finite with a > 0")
    return 1.0 / (1.0 + np.exp(-np.clip((_logit(probabilities) - b) / a, -30.0, 30.0)))


def _manifest_metadata(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("captures")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no captures list")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        absolute = str((path.parent / str(row["path"])).resolve())
        by_path[absolute] = row
    return payload, by_path


def _labeled_rows(
    captures: list[core.Capture],
    posteriors: dict[str, np.ndarray],
    *,
    split: str | None = None,
) -> list[tuple[core.Capture, np.ndarray, np.ndarray]]:
    rows: list[tuple[core.Capture, np.ndarray, np.ndarray]] = []
    for capture in captures:
        if split is not None and capture.split != split:
            continue
        probabilities = posteriors[str(capture.path)]
        labels = core._frame_labels(capture, probabilities.size)
        if labels is not None:
            rows.append((capture, probabilities, labels))
    return rows


def _fit_model_calibration(
    rows: list[tuple[core.Capture, np.ndarray, np.ndarray]],
) -> ModelCalibration:
    probabilities = np.concatenate([row[1] for row in rows])
    labels = np.concatenate([row[2] for row in rows])
    a, b = core._fit_logit_calibration(probabilities, labels)
    calibrated = core._apply_logit_calibration(probabilities, a, b)
    threshold, _score = core._best_balanced_threshold(calibrated, labels, 0.4)
    return ModelCalibration(a=a, b=b, threshold=threshold)


def _apply_calibration_set(
    posteriors: dict[str, np.ndarray],
    calibration: ModelCalibration,
) -> dict[str, np.ndarray]:
    return {
        path: core._apply_logit_calibration(values, calibration.a, calibration.b)
        for path, values in posteriors.items()
    }


def _roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positive_count = int(np.sum(labels))
    negative_count = int(np.sum(~labels))
    if positive_count == 0 or negative_count == 0:
        return 0.0
    order = np.argsort(probabilities, kind="mergesort")
    sorted_values = probabilities[order]
    ranks = np.empty(probabilities.size, dtype=float)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positive_count = int(np.sum(labels))
    if positive_count == 0:
        return 0.0
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order].astype(float)
    precision = np.cumsum(sorted_labels) / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / positive_count)


def _summarize_rows(
    rows: list[tuple[core.Capture, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, Any]:
    probabilities = np.concatenate([row[1] for row in rows])
    labels = np.concatenate([row[2] for row in rows])
    summary: dict[str, Any] = core._binary_metrics(
        probabilities,
        labels,
        threshold,
    )
    capture_metrics = [
        core._binary_metrics(row[1], row[2], threshold)
        for row in rows
    ]
    summary.update(
        {
            "capture_count": len(rows),
            "macro_f1": float(np.mean([row["f1"] for row in capture_metrics])),
            "macro_recall": float(
                np.mean([row["recall"] for row in capture_metrics])
            ),
            "roc_auc": _roc_auc(probabilities, labels),
            "average_precision": _average_precision(probabilities, labels),
            "events": core._event_metrics(rows, threshold),
        }
    )
    return summary


def _group_child_rows(
    rows: list[tuple[core.Capture, np.ndarray, np.ndarray]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[core.Capture, np.ndarray, np.ndarray]]]:
    grouped: dict[str, list[tuple[core.Capture, np.ndarray, np.ndarray]]] = {}
    for row in rows:
        speaker_id = str(metadata[str(row[0].path)]["speaker_id"])
        grouped.setdefault(speaker_id, []).append(row)
    return grouped


def _speaker_metrics(
    rows: list[tuple[core.Capture, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, float]:
    summary = _summarize_rows(rows, threshold)
    return {
        "macro_f1": float(summary["macro_f1"]),
        "recall": float(summary["recall"]),
        "roc_auc": float(summary["roc_auc"]),
        "event_recall": float(summary["events"]["event_recall"]),
    }


def _child_subgroup_summaries(
    rows: list[tuple[core.Capture, np.ndarray, np.ndarray]],
    metadata: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[core.Capture, np.ndarray, np.ndarray]]] = {}
    for row in rows:
        attributes = metadata[str(row[0].path)]
        keys = (
            f"age_{int(attributes['age'])}",
            f"gender_{attributes['gender']}",
            (
                "age_band_6_9"
                if int(attributes["age"]) <= 9
                else "age_band_10_13"
                if int(attributes["age"]) <= 13
                else "age_band_14_16"
            ),
        )
        for key in keys:
            grouped.setdefault(key, []).append(row)
    return {
        key: _summarize_rows(subgroup_rows, threshold)
        for key, subgroup_rows in sorted(grouped.items())
    }


def _paired_bootstrap(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    metric: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, float]:
    speaker_ids = sorted(baseline)
    if speaker_ids != sorted(candidate):
        raise ValueError("speaker sets differ")
    deltas = np.asarray(
        [candidate[speaker][metric] - baseline[speaker][metric] for speaker in speaker_ids],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, deltas.size, size=(repetitions, deltas.size))
    bootstrap_means = np.mean(deltas[sampled], axis=1)
    return {
        "speaker_count": len(speaker_ids),
        "candidate_minus_baseline_mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "candidate_better_speaker_fraction": float(np.mean(deltas > 0.0)),
        "tied_speaker_fraction": float(np.mean(deltas == 0.0)),
    }


def _hash_model(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": _portable_path(path),
        "size": path.stat().st_size,
        "sha256": digest,
    }


def _evaluate_model_raw(
    *,
    model_path: Path,
    captures: list[core.Capture],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    runtime, native_posteriors = core._evaluate_model(
        model_path=model_path,
        captures=captures,
        threshold=0.5,
        repetitions=1,
    )
    raw = {
        path: _invert_native_calibration(values)
        for path, values in native_posteriors.items()
    }
    runtime.pop("captures", None)
    runtime.pop("quality", None)
    runtime["model_path"] = _portable_path(model_path)
    return runtime, raw


def evaluate(
    *,
    calibration_manifest: Path,
    child_manifest: Path,
    models: list[tuple[str, Path]],
) -> dict[str, Any]:
    if len(models) != 2:
        raise ValueError("exactly two models are required: baseline then candidate")
    general_captures = core._load_manifest(calibration_manifest)
    child_captures = core._load_manifest(child_manifest)
    child_manifest_payload, child_metadata = _manifest_metadata(child_manifest)

    model_results: dict[str, Any] = {}
    calibrated_child_sets: dict[str, dict[str, np.ndarray]] = {}
    child_speaker_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, model_path in models:
        general_runtime, general_raw = _evaluate_model_raw(
            model_path=model_path,
            captures=general_captures,
        )
        calibration = _fit_model_calibration(
            _labeled_rows(general_captures, general_raw, split="calibration")
        )
        general_calibrated = _apply_calibration_set(general_raw, calibration)
        held_out_rows = _labeled_rows(
            general_captures,
            general_calibrated,
            split="held_out",
        )

        child_runtime, child_raw = _evaluate_model_raw(
            model_path=model_path,
            captures=child_captures,
        )
        child_calibrated = _apply_calibration_set(child_raw, calibration)
        calibrated_child_sets[model_name] = child_calibrated
        child_rows = _labeled_rows(child_captures, child_calibrated)
        grouped = _group_child_rows(child_rows, child_metadata)
        per_speaker = {
            speaker: _speaker_metrics(rows, calibration.threshold)
            for speaker, rows in grouped.items()
        }
        child_speaker_metrics[model_name] = per_speaker

        model_results[model_name] = {
            "asset": _hash_model(model_path),
            "calibration": {
                "fit_split": "general calibration only",
                "a": calibration.a,
                "b": calibration.b,
                "selected_threshold": calibration.threshold,
            },
            "general_runtime": general_runtime,
            "child_runtime": child_runtime,
            "general_held_out": _summarize_rows(
                held_out_rows,
                calibration.threshold,
            ),
            "external_child": _summarize_rows(
                child_rows,
                calibration.threshold,
            ),
            "external_child_subgroups": _child_subgroup_summaries(
                child_rows,
                child_metadata,
                calibration.threshold,
            ),
        }

    baseline_name = models[0][0]
    candidate_name = models[1][0]
    comparisons = {
        metric: _paired_bootstrap(
            child_speaker_metrics[baseline_name],
            child_speaker_metrics[candidate_name],
            metric,
        )
        for metric in ("macro_f1", "recall", "roc_auc", "event_recall")
    }
    baseline_general = model_results[baseline_name]["general_held_out"]
    candidate_general = model_results[candidate_name]["general_held_out"]
    child_f1 = comparisons["macro_f1"]
    child_event = comparisons["event_recall"]
    meaningful_child_improvement = any(
        comparisons[metric]["candidate_minus_baseline_mean"]
        >= MEANINGFUL_CHILD_IMPROVEMENT
        for metric in ("macro_f1", "recall", "roc_auc", "event_recall")
    )
    gates = {
        "general_macro_f1_noninferior": (
            candidate_general["macro_f1"] - baseline_general["macro_f1"]
            >= GENERAL_NONINFERIORITY_MARGIN
        ),
        "general_false_openings_within_five_percent_or_one": (
            candidate_general["events"]["false_opening_count"]
            <= max(
                baseline_general["events"]["false_opening_count"] * 1.05,
                baseline_general["events"]["false_opening_count"] + 1,
            )
        ),
        "child_speaker_macro_f1_noninferior": (
            child_f1["ci95_low"] >= CHILD_MACRO_F1_NONINFERIORITY_MARGIN
        ),
        "child_event_recall_noninferior": (
            child_event["ci95_low"] >= CHILD_EVENT_RECALL_NONINFERIORITY_MARGIN
        ),
        "meaningful_child_improvement": meaningful_child_improvement,
    }
    retain_candidate = all(gates.values())
    return {
        "schema_version": 1,
        "method": {
            "native_calibration_inverted": {
                "a": NATIVE_CALIBRATION_A,
                "b": NATIVE_CALIBRATION_B,
                "reason": (
                    "Recover smoothed pre-calibration model posteriors from the "
                    "currently compiled production transform before fitting "
                    "each model independently."
                ),
            },
            "threshold_selection": (
                "Each model receives its own Platt calibration and balanced-"
                "accuracy threshold fitted only on the general calibration split."
            ),
            "external_child_use": (
                "No fitting or threshold tuning. Bootstrap unit is speaker. "
                "Clip-boundary onset is descriptive, not a hard gate."
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "predefined_margins": {
                "general_macro_f1": GENERAL_NONINFERIORITY_MARGIN,
                "child_speaker_macro_f1_ci95_low": (
                    CHILD_MACRO_F1_NONINFERIORITY_MARGIN
                ),
                "child_event_recall_ci95_low": (
                    CHILD_EVENT_RECALL_NONINFERIORITY_MARGIN
                ),
                "meaningful_child_improvement": MEANINGFUL_CHILD_IMPROVEMENT,
            },
        },
        "general_manifest": _portable_path(calibration_manifest),
        "child_manifest": {
            "path": _portable_path(child_manifest),
            "source": child_manifest_payload.get("source"),
            "license": child_manifest_payload.get("license"),
            "selection": child_manifest_payload.get("selection"),
        },
        "models": model_results,
        "paired_child_bootstrap": comparisons,
        "gates": gates,
        "decision": {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "retain_candidate": retain_candidate,
            "reason": (
                "Candidate passes every independently calibrated general and "
                "multi-speaker child gate."
                if retain_candidate
                else "Candidate failed at least one predefined model-selection gate."
            ),
        },
    }


def _parse_model(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("model must use NAME=PATH")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model does not exist: {path}")
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--general-manifest",
        type=Path,
        default=Path("models/vad_eval_corpus/manifest.json"),
    )
    parser.add_argument(
        "--child-manifest",
        type=Path,
        default=Path("models/vad_child_multispeaker_corpus/manifest.json"),
    )
    parser.add_argument(
        "--model",
        type=_parse_model,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/vad-model-selection-report.json"),
    )
    args = parser.parse_args()
    report = evaluate(
        calibration_manifest=args.general_manifest.resolve(),
        child_manifest=args.child_manifest.resolve(),
        models=args.model,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
