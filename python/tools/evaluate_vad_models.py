"""Compare pinned Silero ONNX models through AudioForge's native VAD path."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


@dataclass(frozen=True)
class Capture:
    path: Path
    sample_rate: int
    audio: np.ndarray
    split: str = "unlabeled"
    condition: str = "unlabeled"
    speech_intervals_samples: tuple[tuple[int, int], ...] | None = None


def _parse_model(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("models must use NAME=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model does not exist: {path}")
    return name.strip(), path


def _load_mono_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw_audio = wavfile.read(path)
    audio = np.asarray(raw_audio)
    if audio.ndim == 2:
        audio = np.mean(audio.astype(np.float64), axis=1)
    if audio.ndim != 1:
        raise ValueError(f"{path} must contain mono or interleaved PCM audio")

    if np.issubdtype(audio.dtype, np.floating):
        normalized = audio.astype(np.float32)
    elif np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype.name)
        scale = float(max(abs(info.min), info.max))
        normalized = audio.astype(np.float32) / scale
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype.name)
        midpoint = (float(info.max) + 1.0) / 2.0
        normalized = ((audio.astype(np.float64) - midpoint) / midpoint).astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV sample type: {audio.dtype}")

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    return int(sample_rate), np.ascontiguousarray(np.clip(normalized, -1.0, 1.0))


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else 0.0


def _frame_labels(capture: Capture, frame_count: int) -> np.ndarray | None:
    if capture.speech_intervals_samples is None:
        return None
    window_size = int(np.ceil(512 * capture.sample_rate / 16000))
    labels = np.zeros(frame_count, dtype=bool)
    for frame_index in range(frame_count):
        frame_start = frame_index * window_size
        frame_end = min(frame_start + window_size, capture.audio.size)
        frame_length = max(frame_end - frame_start, 1)
        speech_overlap = 0
        for interval_start, interval_end in capture.speech_intervals_samples:
            speech_overlap += max(
                0,
                min(frame_end, interval_end) - max(frame_start, interval_start),
            )
        labels[frame_index] = speech_overlap >= frame_length * 0.5
    return labels


def _binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    if probabilities.shape != labels.shape:
        raise ValueError("posterior and label shapes differ")
    decisions = probabilities >= threshold
    positive = labels
    negative = ~labels
    true_positive = int(np.sum(decisions & positive))
    false_positive = int(np.sum(decisions & negative))
    true_negative = int(np.sum(~decisions & negative))
    false_negative = int(np.sum(~decisions & positive))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    return {
        "frame_count": int(labels.size),
        "speech_frame_count": int(np.sum(positive)),
        "noise_frame_count": int(np.sum(negative)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        ),
        "balanced_accuracy": float((recall + specificity) * 0.5),
        "brier_score": float(np.mean((probabilities - labels.astype(float)) ** 2)),
    }


def _best_balanced_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    reference_threshold: float,
) -> tuple[float, float]:
    scored = [
        (
            float(candidate),
            float(
                _binary_metrics(probabilities, labels, float(candidate))[
                    "balanced_accuracy"
                ]
            ),
        )
        for candidate in np.linspace(0.10, 0.90, 161)
    ]
    return max(
        scored,
        key=lambda item: (item[1], -abs(item[0] - reference_threshold)),
    )


def _runs(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= minimum_frames
    ]


def _event_metrics(
    rows: list[tuple[Capture, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, float | int]:
    minimum_event_frames = math.ceil(0.250 / 0.032)
    true_event_count = 0
    detected_event_count = 0
    false_opening_count = 0
    onset_delays_ms: list[float] = []
    noise_seconds = 0.0

    for capture, probabilities, labels in rows:
        predicted_runs = _runs(probabilities >= threshold, minimum_event_frames)
        true_runs = _runs(labels, 1)
        true_event_count += len(true_runs)
        window_size = int(np.ceil(512 * capture.sample_rate / 16000))
        frame_ms = 1000.0 * window_size / capture.sample_rate

        for true_start, true_end in true_runs:
            overlapping = [
                run
                for run in predicted_runs
                if run[0] < true_end and run[1] > true_start
            ]
            if overlapping:
                detected_event_count += 1
                first_start = min(run[0] for run in overlapping)
                onset_delays_ms.append(max(0.0, first_start - true_start) * frame_ms)

        for predicted_start, predicted_end in predicted_runs:
            if not any(
                predicted_start < true_end and predicted_end > true_start
                for true_start, true_end in true_runs
            ):
                false_opening_count += 1
        noise_seconds += float(np.sum(~labels) * window_size / capture.sample_rate)

    return {
        "minimum_event_duration_ms": 250,
        "true_event_count": true_event_count,
        "detected_event_count": detected_event_count,
        "event_recall": (
            float(detected_event_count / true_event_count) if true_event_count else 0.0
        ),
        "false_opening_count": false_opening_count,
        "false_openings_per_noise_minute": (
            float(false_opening_count / (noise_seconds / 60.0))
            if noise_seconds > 0.0
            else 0.0
        ),
        "median_onset_delay_ms": (
            float(np.median(onset_delays_ms)) if onset_delays_ms else 0.0
        ),
        "p95_onset_delay_ms": _quantile(np.asarray(onset_delays_ms), 0.95),
    }


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
        )
    return error


def _fit_logit_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    epsilon = 1e-6
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    targets = labels.astype(float)

    def objective(parameters: np.ndarray) -> float:
        a, b = parameters
        transformed = np.clip(a * logits + b, -30.0, 30.0)
        calibrated = 1.0 / (1.0 + np.exp(-transformed))
        calibrated = np.clip(calibrated, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(
                targets * np.log(calibrated)
                + (1.0 - targets) * np.log(1.0 - calibrated)
            )
        )

    result = minimize(
        objective,
        np.asarray([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(0.05, 10.0), (-10.0, 10.0)],
    )
    if not result.success or not np.isfinite(result.x).all():
        return 1.0, 0.0
    return float(result.x[0]), float(result.x[1])


def _apply_logit_calibration(
    probabilities: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    epsilon = 1e-6
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-np.clip(a * logits + b, -30.0, 30.0)))


def _calibration_analysis(
    labeled: list[tuple[Capture, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, Any] | None:
    calibration_rows = [row for row in labeled if row[0].split == "calibration"]
    held_out_rows = [row for row in labeled if row[0].split == "held_out"]
    if not calibration_rows or not held_out_rows:
        return None

    calibration_probabilities = np.concatenate([row[1] for row in calibration_rows])
    calibration_labels = np.concatenate([row[2] for row in calibration_rows])
    held_out_probabilities = np.concatenate([row[1] for row in held_out_rows])
    held_out_labels = np.concatenate([row[2] for row in held_out_rows])
    a, b = _fit_logit_calibration(calibration_probabilities, calibration_labels)
    held_out_calibrated = _apply_logit_calibration(held_out_probabilities, a, b)
    identity_brier = float(
        np.mean((held_out_probabilities - held_out_labels.astype(float)) ** 2)
    )
    calibrated_brier = float(
        np.mean((held_out_calibrated - held_out_labels.astype(float)) ** 2)
    )
    identity_ece = _expected_calibration_error(
        held_out_probabilities,
        held_out_labels,
    )
    calibrated_ece = _expected_calibration_error(
        held_out_calibrated,
        held_out_labels,
    )

    calibrated_calibration_probabilities = _apply_logit_calibration(
        calibration_probabilities,
        a,
        b,
    )
    selected_threshold, calibration_score = _best_balanced_threshold(
        calibrated_calibration_probabilities,
        calibration_labels,
        threshold,
    )
    held_out_default = _binary_metrics(
        held_out_calibrated,
        held_out_labels,
        threshold,
    )
    held_out_selected = _binary_metrics(
        held_out_calibrated,
        held_out_labels,
        selected_threshold,
    )
    keep_calibration = (
        calibrated_brier < identity_brier - 1e-4
        and calibrated_ece <= identity_ece + 1e-4
    )
    return {
        "a": a,
        "b": b,
        "selected_threshold": selected_threshold,
        "calibration_balanced_accuracy_at_selected_threshold": calibration_score,
        "held_out_identity_brier": identity_brier,
        "held_out_calibrated_brier": calibrated_brier,
        "held_out_identity_ece": identity_ece,
        "held_out_calibrated_ece": calibrated_ece,
        "keep_calibration": keep_calibration,
        "held_out_default_threshold": held_out_default,
        "held_out_selected_threshold": held_out_selected,
    }


def _aggregate_quality(
    captures: list[Capture],
    posteriors: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any] | None:
    labeled: list[tuple[Capture, np.ndarray, np.ndarray]] = []
    for capture in captures:
        probabilities = posteriors[str(capture.path)]
        labels = _frame_labels(capture, probabilities.size)
        if labels is not None:
            labeled.append((capture, probabilities, labels))
    if not labeled:
        return None

    def summarize(rows: list[tuple[Capture, np.ndarray, np.ndarray]]) -> dict[str, Any]:
        probabilities = np.concatenate([row[1] for row in rows])
        labels = np.concatenate([row[2] for row in rows])
        metrics: dict[str, Any] = _binary_metrics(
            probabilities,
            labels,
            threshold,
        )
        per_capture_f1 = np.asarray(
            [_binary_metrics(row[1], row[2], threshold)["f1"] for row in rows],
            dtype=float,
        )
        metrics["capture_count"] = len(rows)
        metrics["macro_f1"] = float(np.mean(per_capture_f1))
        metrics["worst_capture_f1"] = float(np.min(per_capture_f1))
        speech_probabilities = probabilities[labels]
        noise_probabilities = probabilities[~labels]
        metrics["speech_p05"] = _quantile(speech_probabilities, 0.05)
        metrics["speech_p50"] = _quantile(speech_probabilities, 0.50)
        metrics["speech_p95"] = _quantile(speech_probabilities, 0.95)
        metrics["noise_p50"] = _quantile(noise_probabilities, 0.50)
        metrics["noise_p95"] = _quantile(noise_probabilities, 0.95)
        metrics["noise_p99"] = _quantile(noise_probabilities, 0.99)
        best_threshold, best_balanced_accuracy = _best_balanced_threshold(
            probabilities,
            labels,
            threshold,
        )
        metrics["descriptive_best_balanced_threshold"] = best_threshold
        metrics["descriptive_best_balanced_accuracy"] = best_balanced_accuracy
        metrics["events"] = _event_metrics(rows, threshold)
        return metrics

    by_split = {
        split: summarize([row for row in labeled if row[0].split == split])
        for split in sorted({row[0].split for row in labeled})
    }
    by_condition = {
        condition: summarize([row for row in labeled if row[0].condition == condition])
        for condition in sorted({row[0].condition for row in labeled})
    }
    result = {
        "overall": summarize(labeled),
        "by_split": by_split,
        "by_condition": by_condition,
    }
    calibration = _calibration_analysis(labeled, threshold)
    if calibration is not None:
        result["calibration"] = calibration
    return result


def _load_manifest(path: Path) -> list[Capture]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("captures"), list):
        raise ValueError("manifest must contain a captures list")

    captures: list[Capture] = []
    for raw_capture in payload["captures"]:
        if not isinstance(raw_capture, dict):
            raise ValueError("manifest capture entries must be objects")
        capture_path = (manifest_path.parent / str(raw_capture["path"])).resolve()
        sample_rate, audio = _load_mono_audio(capture_path)
        intervals = tuple(
            (int(interval[0]), int(interval[1]))
            for interval in raw_capture.get("speech_intervals_samples", [])
        )
        if any(
            start < 0 or end <= start or end > audio.size for start, end in intervals
        ):
            raise ValueError(f"invalid speech interval in {capture_path}")
        captures.append(
            Capture(
                path=capture_path,
                sample_rate=sample_rate,
                audio=audio,
                split=str(raw_capture.get("split", "unlabeled")),
                condition=str(raw_capture.get("condition", "unlabeled")),
                speech_intervals_samples=intervals,
            )
        )
    return captures


def _evaluate_model(
    *,
    model_path: Path,
    captures: list[Capture],
    threshold: float,
    repetitions: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import mic_eq

    os.environ["VAD_MODEL_PATH"] = str(model_path)
    per_capture: dict[str, Any] = {}
    posteriors: dict[str, np.ndarray] = {}
    timed_seconds: list[float] = []
    total_audio_seconds = 0.0

    for capture in captures:
        path = capture.path
        sample_rate = capture.sample_rate
        audio = capture.audio
        probabilities: np.ndarray | None = None
        capture_times: list[float] = []
        for _ in range(repetitions + 1):
            started = time.perf_counter()
            probabilities = np.asarray(
                mic_eq.analyze_vad_probabilities(audio, sample_rate, threshold),
                dtype=np.float64,
            )
            capture_times.append(time.perf_counter() - started)

        assert probabilities is not None
        if probabilities.size == 0 or not np.isfinite(probabilities).all():
            raise RuntimeError(f"{model_path} returned invalid posteriors for {path}")

        measured_times = capture_times[1:]
        timed_seconds.extend(measured_times)
        audio_seconds = float(audio.size / sample_rate)
        total_audio_seconds += audio_seconds * repetitions
        internal_capture_key = str(path)
        report_capture_key = _portable_path(path)
        posteriors[internal_capture_key] = probabilities
        per_capture[report_capture_key] = {
            "split": capture.split,
            "condition": capture.condition,
            "sample_rate": sample_rate,
            "audio_seconds": audio_seconds,
            "frame_count": int(probabilities.size),
            "posterior_mean": float(np.mean(probabilities)),
            "posterior_p50": _quantile(probabilities, 0.50),
            "posterior_p95": _quantile(probabilities, 0.95),
            "active_ratio_035": float(np.mean(probabilities >= 0.35)),
            "active_ratio_050": float(np.mean(probabilities >= 0.50)),
            "runtime_median_seconds": float(np.median(measured_times)),
            "runtime_p95_seconds": _quantile(np.asarray(measured_times), 0.95),
        }

    total_runtime = float(sum(timed_seconds))
    summary = {
        "model_path": _portable_path(model_path),
        "capture_count": len(captures),
        "timed_repetitions": repetitions,
        "runtime_median_seconds": float(np.median(timed_seconds)),
        "runtime_p95_seconds": _quantile(np.asarray(timed_seconds), 0.95),
        "aggregate_realtime_factor": (
            total_runtime / total_audio_seconds if total_audio_seconds > 0.0 else 0.0
        ),
        "captures": per_capture,
    }
    quality = _aggregate_quality(captures, posteriors, threshold)
    if quality is not None:
        summary["quality"] = quality
    return summary, posteriors


def _compare_posteriors(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    absolute_deltas: list[np.ndarray] = []
    decision_changes_035 = 0
    decision_changes_050 = 0
    frame_count = 0

    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate captures differ")
    for capture_key in baseline:
        baseline_values = baseline[capture_key]
        candidate_values = candidate[capture_key]
        if baseline_values.shape != candidate_values.shape:
            raise ValueError(f"posterior shape differs for {capture_key}")
        deltas = np.abs(candidate_values - baseline_values)
        absolute_deltas.append(deltas)
        decision_changes_035 += int(
            np.sum((baseline_values >= 0.35) != (candidate_values >= 0.35))
        )
        decision_changes_050 += int(
            np.sum((baseline_values >= 0.50) != (candidate_values >= 0.50))
        )
        frame_count += int(baseline_values.size)

    all_deltas = (
        np.concatenate(absolute_deltas) if absolute_deltas else np.zeros(0, dtype=float)
    )
    return {
        "frame_count": frame_count,
        "mean_absolute_posterior_delta": (
            float(np.mean(all_deltas)) if all_deltas.size else 0.0
        ),
        "p95_absolute_posterior_delta": _quantile(all_deltas, 0.95),
        "max_absolute_posterior_delta": (
            float(np.max(all_deltas)) if all_deltas.size else 0.0
        ),
        "decision_changes_035": decision_changes_035,
        "decision_changes_050": decision_changes_050,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="*",
        type=Path,
        help="PCM WAV captures to evaluate",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON corpus manifest with labels; may be combined with positional audio",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        required=True,
        help="Pinned model in NAME=PATH form; first model is the baseline",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the complete JSON report to this path instead of stdout",
    )
    args = parser.parse_args()

    if len(args.model) < 1:
        parser.error("at least one --model value is required")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    if not args.audio and args.manifest is None:
        parser.error("provide at least one audio path or --manifest")

    captures: list[Capture] = []
    if args.manifest is not None:
        try:
            captures.extend(_load_manifest(args.manifest))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            parser.error(f"invalid manifest: {error}")
    for raw_path in args.audio:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            parser.error(f"audio does not exist: {path}")
        sample_rate, audio = _load_mono_audio(path)
        captures.append(Capture(path=path, sample_rate=sample_rate, audio=audio))

    evaluations: dict[str, Any] = {}
    posterior_sets: dict[str, dict[str, np.ndarray]] = {}
    for model_name, model_path in args.model:
        if model_name in evaluations:
            parser.error(f"duplicate model name: {model_name}")
        summary, posteriors = _evaluate_model(
            model_path=model_path,
            captures=captures,
            threshold=float(args.threshold),
            repetitions=int(args.repetitions),
        )
        evaluations[model_name] = summary
        posterior_sets[model_name] = posteriors

    baseline_name = args.model[0][0]
    comparisons = {
        model_name: _compare_posteriors(
            posterior_sets[baseline_name],
            posterior_sets[model_name],
        )
        for model_name, _model_path in args.model[1:]
    }
    rendered = json.dumps(
        {
            "baseline": baseline_name,
            "threshold": float(args.threshold),
            "models": evaluations,
            "comparisons": comparisons,
        },
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
