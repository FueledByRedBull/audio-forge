"""Train and evaluate the versioned AudioForge de-esser soft-fusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from mic_eq.analysis.deesser_corpus import (
    CORPUS_CASES,
    CORPUS_LICENSE,
    CORPUS_VERSION,
    generate_deesser_case,
    labels_for_analysis_frames,
)
from mic_eq.analysis.deesser_fusion import (
    CLIP_COEFFICIENTS,
    CLIP_FEATURE_NAMES,
    CLIP_INTERCEPT,
    ENABLE_PROBABILITY_THRESHOLD,
    FRAME_COEFFICIENTS,
    FRAME_FEATURE_NAMES,
    FRAME_INTERCEPT,
    MODEL_VERSION,
)
from mic_eq.analysis.voice_setup import _vad_masked_speech_features

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "deesser-corpus-v1-report.json"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0.0
    result = np.empty(values.shape, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
    class_balanced: bool = False,
    nonnegative_coefficients: bool = True,
) -> tuple[float, np.ndarray]:
    """Fit deterministic L2 logistic regression with monotonic coefficients."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if class_balanced:
        positives = max(1, int(np.count_nonzero(y >= 0.5)))
        negatives = max(1, int(np.count_nonzero(y < 0.5)))
        sample_weights = np.where(
            y >= 0.5,
            0.5 * y.size / positives,
            0.5 * y.size / negatives,
        )
    else:
        sample_weights = np.ones(y.shape, dtype=float)
    design = np.column_stack([np.ones(x.shape[0], dtype=float), x])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss_rows = np.logaddexp(0.0, logits) - y * logits
        loss = float(np.sum(sample_weights * loss_rows))
        loss += 0.5 * regularization * float(np.dot(beta[1:], beta[1:]))
        probabilities = _sigmoid(logits)
        gradient = design.T @ (sample_weights * (probabilities - y))
        gradient[1:] += regularization * beta[1:]
        return loss, gradient

    bounds = (
        [(None, None)] + [(0.0, None)] * x.shape[1]
        if nonnegative_coefficients
        else None
    )
    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"logistic fit failed: {result.message}")
    beta = np.asarray(result.x, dtype=float)
    return float(beta[0]), beta[1:]


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    positives = max(1, int(np.count_nonzero(sorted_labels)))
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, sorted_labels.size + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & negative))
    false_negative = int(np.count_nonzero(~predicted & positive))
    true_negative = int(np.count_nonzero(~predicted & negative))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    false_positive_rate = false_positive / max(1, int(np.count_nonzero(negative)))
    brier = float(np.mean(np.square(probabilities - labels)))

    bin_edges = np.linspace(0.0, 1.0, 11)
    calibration_error = 0.0
    for low, high in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (probabilities >= low) & (
            probabilities <= high if high == 1.0 else probabilities < high
        )
        if not np.any(mask):
            continue
        calibration_error += float(np.mean(mask)) * abs(
            float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
        )

    return {
        "samples": int(labels.size),
        "positive_samples": int(np.count_nonzero(positive)),
        "negative_samples": int(np.count_nonzero(negative)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "pr_auc_average_precision": _average_precision(labels, probabilities),
        "brier_score": brier,
        "expected_calibration_error": float(calibration_error),
        "positive_probability_p10": (
            float(np.percentile(probabilities[positive], 10.0))
            if np.any(positive)
            else 0.0
        ),
        "positive_probability_median": (
            float(np.median(probabilities[positive]))
            if np.any(positive)
            else 0.0
        ),
        "negative_probability_p90": (
            float(np.percentile(probabilities[negative], 90.0))
            if np.any(negative)
            else 0.0
        ),
        "negative_probability_max": (
            float(np.max(probabilities[negative]))
            if np.any(negative)
            else 0.0
        ),
    }


def _low_false_activation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    maximum_false_positive_rate: float = 0.03,
) -> float:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.35, 0.95, 241),
                probabilities,
            ]
        )
    )
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = _classification_metrics(labels, probabilities, float(threshold))
        false_positive_rate = float(metrics["false_positive_rate"])
        if false_positive_rate > maximum_false_positive_rate:
            continue
        recall = float(metrics["recall"])
        precision = float(metrics["precision"])
        feasible.append((recall, precision, float(threshold)))
    if not feasible:
        return 0.95
    best_recall = max(item[0] for item in feasible)
    recall_matches = [
        item for item in feasible if np.isclose(item[0], best_recall, atol=1e-12)
    ]
    best_precision = max(item[1] for item in recall_matches)
    equally_valid = [
        item[2]
        for item in recall_matches
        if np.isclose(item[1], best_precision, atol=1e-12)
    ]
    # Use the centre of the equally valid interval instead of balancing one
    # floating-point step above the hardest negative fixture.
    return float(0.5 * (min(equally_valid) + max(equally_valid)))


def _extract_corpus() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame_rows: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    clips: list[dict[str, Any]] = []
    for case in CORPUS_CASES:
        generated = generate_deesser_case(case)
        noise_rms_db = float(
            10.0
            * np.log10(
                max(
                    float(np.mean(np.square(generated.noise_audio, dtype=np.float64))),
                    1e-18,
                )
            )
        )
        extracted = _vad_masked_speech_features(
            generated.speech_audio,
            case.sample_rate,
            noise_rms_db,
            vad_probabilities=generated.vad_probabilities,
            noise_audio=generated.noise_audio,
        )
        evidence = extracted["deesser_frame_evidence"]
        rows = np.asarray(evidence["frame_feature_rows"], dtype=float)
        indices = np.asarray(evidence["frame_indices"], dtype=int)
        labels = labels_for_analysis_frames(generated, indices)
        if rows.shape[0] != labels.size:
            raise RuntimeError(f"frame/label mismatch for {case.name}")
        frame_rows.append(rows)
        frame_labels.append(labels)
        clips.append(
            {
                "name": case.name,
                "label": int(case.needs_deesser),
                "condition": case.condition,
                "sample_rate": case.sample_rate,
                "temporal_score": float(evidence.get("temporal_score", 0.0)),
                "row_start": sum(part.shape[0] for part in frame_rows[:-1]),
                "row_count": rows.shape[0],
            }
        )
    return (
        np.vstack(frame_rows),
        np.concatenate(frame_labels),
        clips,
    )


def train_and_evaluate() -> dict[str, Any]:
    frame_features, frame_labels, clips = _extract_corpus()
    fold_count = 6
    clip_folds = np.zeros(len(clips), dtype=int)
    for label in (0, 1):
        label_indices = [
            index for index, clip in enumerate(clips) if int(clip["label"]) == label
        ]
        for rank, index in enumerate(label_indices):
            clip_folds[index] = rank % fold_count

    frame_folds = np.empty(frame_labels.size, dtype=int)
    for clip_index, clip in enumerate(clips):
        start = int(clip["row_start"])
        stop = start + int(clip["row_count"])
        frame_folds[start:stop] = clip_folds[clip_index]

    frame_probabilities_oof = np.zeros(frame_labels.shape, dtype=float)
    for fold in range(fold_count):
        train_mask = frame_folds != fold
        test_mask = ~train_mask
        fold_intercept, fold_coefficients = _fit_logistic(
            frame_features[train_mask],
            frame_labels[train_mask],
            regularization=2.0,
        )
        frame_probabilities_oof[test_mask] = _sigmoid(
            fold_intercept + frame_features[test_mask] @ fold_coefficients
        )
    frame_threshold = _low_false_activation_threshold(
        frame_labels,
        frame_probabilities_oof,
        maximum_false_positive_rate=0.03,
    )

    clip_features_oof: list[list[float]] = []
    clip_labels: list[int] = []
    for clip in clips:
        start = int(clip["row_start"])
        stop = start + int(clip["row_count"])
        probabilities = frame_probabilities_oof[start:stop]
        raw_features = frame_features[start:stop]
        top_count = max(1, int(np.ceil(probabilities.size * 0.10)))
        clip_features_oof.append(
            [
                float(np.percentile(probabilities, 90.0)),
                float(np.mean(np.partition(probabilities, -top_count)[-top_count:])),
                float(np.mean(probabilities)),
                float(clip["temporal_score"]),
                float(np.percentile(raw_features[:, 0], 90.0)),
                float(np.percentile(raw_features[:, 2], 90.0)),
            ]
        )
        clip_labels.append(int(clip["label"]))
    clip_features_oof_array = np.asarray(clip_features_oof, dtype=float)
    clip_labels_array = np.asarray(clip_labels, dtype=int)
    clip_probabilities_oof = np.zeros(clip_labels_array.shape, dtype=float)
    for fold in range(fold_count):
        train_mask = clip_folds != fold
        test_mask = ~train_mask
        fold_intercept, fold_coefficients = _fit_logistic(
            clip_features_oof_array[train_mask],
            clip_labels_array[train_mask],
            regularization=0.4,
        )
        clip_probabilities_oof[test_mask] = _sigmoid(
            fold_intercept
            + clip_features_oof_array[test_mask] @ fold_coefficients
        )
    clip_threshold = _low_false_activation_threshold(
        clip_labels_array,
        clip_probabilities_oof,
        maximum_false_positive_rate=0.03,
    )

    frame_intercept, frame_coefficients = _fit_logistic(
        frame_features,
        frame_labels,
        regularization=2.0,
    )
    final_frame_probabilities = _sigmoid(
        frame_intercept + frame_features @ frame_coefficients
    )
    final_clip_features: list[list[float]] = []
    for clip in clips:
        start = int(clip["row_start"])
        stop = start + int(clip["row_count"])
        probabilities = final_frame_probabilities[start:stop]
        raw_features = frame_features[start:stop]
        top_count = max(1, int(np.ceil(probabilities.size * 0.10)))
        final_clip_features.append(
            [
                float(np.percentile(probabilities, 90.0)),
                float(np.mean(np.partition(probabilities, -top_count)[-top_count:])),
                float(np.mean(probabilities)),
                float(clip["temporal_score"]),
                float(np.percentile(raw_features[:, 0], 90.0)),
                float(np.percentile(raw_features[:, 2], 90.0)),
            ]
        )
    final_clip_features_array = np.asarray(final_clip_features, dtype=float)
    clip_intercept, clip_coefficients = _fit_logistic(
        final_clip_features_array,
        clip_labels_array,
        regularization=0.4,
    )

    per_condition: dict[str, dict[str, float | int]] = {}
    for condition in sorted({str(clip["condition"]) for clip in clips}):
        mask = np.asarray(
            [clip["condition"] == condition for clip in clips],
            dtype=np.bool_,
        )
        per_condition[condition] = _classification_metrics(
            clip_labels_array[mask],
            clip_probabilities_oof[mask],
            clip_threshold,
        )

    return {
        "corpus": {
            "version": CORPUS_VERSION,
            "license": CORPUS_LICENSE,
            "generated": True,
            "clip_count": len(clips),
            "real_recordings": False,
            "perceptual_validation": False,
            "evaluation_method": "six-fold clip-grouped out-of-fold predictions",
            "limitations": [
                "Generated fixtures validate detector behavior, not listening quality.",
                "Real-speaker evaluation remains required before perceptual claims.",
            ],
        },
        "model": {
            "version": MODEL_VERSION,
            "frame_feature_names": list(FRAME_FEATURE_NAMES),
            "frame_intercept": frame_intercept,
            "frame_coefficients": frame_coefficients.tolist(),
            "frame_operating_threshold": frame_threshold,
            "clip_feature_names": list(CLIP_FEATURE_NAMES),
            "clip_intercept": clip_intercept,
            "clip_coefficients": clip_coefficients.tolist(),
            "clip_operating_threshold": clip_threshold,
            "selection_policy": (
                "maximize recall with false-positive rate <= 0.03, then "
                "precision, then use the midpoint of the equally valid interval"
            ),
        },
        "metrics": {
            "frame": _classification_metrics(
                frame_labels,
                frame_probabilities_oof,
                frame_threshold,
            ),
            "clip": _classification_metrics(
                clip_labels_array,
                clip_probabilities_oof,
                clip_threshold,
            ),
            "clip_by_condition": per_condition,
        },
    }


def _check_runtime_model(report: dict[str, Any]) -> None:
    model = report["model"]
    checks = (
        ("frame intercept", FRAME_INTERCEPT, float(model["frame_intercept"])),
        (
            "clip intercept",
            CLIP_INTERCEPT,
            float(model["clip_intercept"]),
        ),
        (
            "clip threshold",
            ENABLE_PROBABILITY_THRESHOLD,
            float(model["clip_operating_threshold"]),
        ),
    )
    failures: list[str] = []
    for label, actual, expected in checks:
        if not np.isclose(actual, expected, atol=5e-4, rtol=0.0):
            failures.append(f"{label}: runtime={actual:.8f}, fitted={expected:.8f}")
    if not np.allclose(
        FRAME_COEFFICIENTS,
        np.asarray(model["frame_coefficients"], dtype=float),
        atol=5e-4,
        rtol=0.0,
    ):
        failures.append("frame coefficients differ from fitted corpus model")
    if not np.allclose(
        CLIP_COEFFICIENTS,
        np.asarray(model["clip_coefficients"], dtype=float),
        atol=5e-4,
        rtol=0.0,
    ):
        failures.append("clip coefficients differ from fitted corpus model")
    if failures:
        raise SystemExit("\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()

    report = train_and_evaluate()
    if args.check_model:
        _check_runtime_model(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "De-esser corpus evaluation: "
        f"{report['metrics']['clip']['precision']:.3f} precision, "
        f"{report['metrics']['clip']['recall']:.3f} recall, "
        f"{report['metrics']['clip']['false_positive_rate']:.3f} false-positive rate"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
