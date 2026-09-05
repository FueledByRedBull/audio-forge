"""Compare AudioForge VAD through two isolated ONNX Runtime backends."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "python" / "tools"
CPU_ORT_ARCHIVE_SHA256 = (
    "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd"
)
THRESHOLDS = (0.35, 0.36, 0.40, 0.48, 0.50, 0.65)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _asset_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": _portable_path(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _hysteresis(
    values: np.ndarray,
    threshold: float = 0.48,
    close_margin: float = 0.12,
    onset_velocity: float = 0.08,
) -> np.ndarray:
    close_threshold = max(0.02, threshold - close_margin)
    state = False
    previous = 0.0
    decisions = np.zeros(values.shape, dtype=bool)
    for index, value in enumerate(values):
        opening = value >= threshold or (
            value - previous >= onset_velocity and value >= close_threshold
        )
        state = bool(value >= close_threshold) if state else bool(opening)
        decisions[index] = state
        previous = float(value)
    return decisions


def _compare_posteriors(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, Any]:
    if not baseline:
        raise ValueError("baseline corpus produced no captures")
    if list(baseline) != list(candidate):
        raise ValueError("baseline and candidate capture keys differ")

    deltas: list[np.ndarray] = []
    threshold_changes = {f"{threshold:.2f}": 0 for threshold in THRESHOLDS}
    hysteresis_changes = 0
    frame_count = 0
    for key in baseline:
        left = baseline[key]
        right = candidate[key]
        if left.shape != right.shape:
            raise ValueError(f"posterior shape differs for {key}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"non-finite posterior for {key}")
        delta = np.abs(right - left)
        deltas.append(delta)
        frame_count += int(delta.size)
        for threshold in THRESHOLDS:
            threshold_changes[f"{threshold:.2f}"] += int(
                np.count_nonzero((left >= threshold) != (right >= threshold))
            )
        hysteresis_changes += int(
            np.count_nonzero(_hysteresis(left) != _hysteresis(right))
        )

    all_deltas = np.concatenate(deltas) if deltas else np.zeros(0, dtype=float)
    return {
        "capture_count": len(baseline),
        "frame_count": frame_count,
        "mean_abs_delta": float(np.mean(all_deltas)) if all_deltas.size else 0.0,
        "p95_abs_delta": float(np.quantile(all_deltas, 0.95))
        if all_deltas.size
        else 0.0,
        "max_abs_delta": float(np.max(all_deltas)) if all_deltas.size else 0.0,
        "exact_equal_frames": int(np.count_nonzero(all_deltas == 0.0)),
        "threshold_changes": threshold_changes,
        "hysteresis_surrogate": {
            "threshold": 0.48,
            "close_margin": 0.12,
            "onset_velocity": 0.08,
            "decision_changes": hysteresis_changes,
        },
    }


def _add_backend_paths(python_root: Path, dll_roots: list[Path]) -> list[Any]:
    for path in reversed((python_root.resolve(), REPO_ROOT / "python", TOOLS_ROOT)):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    handles: list[Any] = []
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for path in dll_roots:
            handles.append(add_dll_directory(str(path.resolve())))
    return handles


def _measure_backend(
    *,
    python_root: Path,
    dll_roots: list[Path],
    manifest: Path,
    model: Path,
    threshold: float,
    throughput_audio: Path,
    throughput_repetitions: int,
    output: Path,
) -> None:
    dll_handles = _add_backend_paths(python_root, dll_roots)
    _ = dll_handles  # Keep the directory handles alive through the imports below.
    from evaluate_vad_models import _aggregate_quality, _load_manifest, _load_mono_audio

    captures = _load_manifest(manifest)
    if not captures:
        raise ValueError("evaluation manifest contains no captures")
    os.environ["VAD_MODEL_PATH"] = str(model.resolve())
    import mic_eq

    native_module = importlib.import_module("mic_eq.mic_eq_core")
    native_file = native_module.__file__
    if native_file is None:
        raise RuntimeError("loaded native module has no file path")
    native_path = Path(native_file).resolve(strict=True)
    python_root = python_root.resolve(strict=True)
    if not native_path.is_relative_to(python_root):
        raise RuntimeError(
            f"loaded native module escaped requested backend root: {native_path}"
        )

    posteriors: dict[str, np.ndarray] = {}
    frame_count = 0
    for capture in captures:
        values = np.asarray(
            mic_eq.analyze_vad_probabilities(
                capture.audio, capture.sample_rate, threshold
            ),
            dtype=np.float64,
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid posterior output for {capture.path}")
        posteriors[str(capture.path)] = values
        frame_count += int(values.size)
    if frame_count == 0:
        raise ValueError("evaluation manifest produced no posterior frames")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), *posteriors.values())

    sample_rate, audio = _load_mono_audio(throughput_audio)
    mic_eq.analyze_vad_probabilities(audio, sample_rate, threshold)
    durations: list[float] = []
    import time

    for _ in range(throughput_repetitions):
        started = time.perf_counter()
        values = np.asarray(
            mic_eq.analyze_vad_probabilities(audio, sample_rate, threshold),
            dtype=np.float64,
        )
        durations.append(time.perf_counter() - started)
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError("invalid throughput posterior output")

    quality = _aggregate_quality(captures, posteriors, threshold)
    audio_seconds = float(audio.size / sample_rate)
    result = {
        "capture_count": len(captures),
        "frame_count": frame_count,
        "native_module": _asset_record(native_path),
        "quality_overall": quality["overall"] if quality else None,
        "throughput": {
            "audio_seconds": audio_seconds,
            "frame_count": int(values.size),
            "repetitions": throughput_repetitions,
            "median_seconds": float(np.median(durations)),
            "p95_seconds": float(np.quantile(durations, 0.95)),
            "realtime_factor": float(sum(durations) / (audio_seconds * throughput_repetitions)),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_backend(
    *,
    label: str,
    python_root: Path,
    dll_roots: list[Path],
    manifest: Path,
    model: Path,
    threshold: float,
    throughput_audio: Path,
    throughput_repetitions: int,
    temporary_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    output = temporary_root / f"{label}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--python-root",
        str(python_root.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--model",
        str(model.resolve()),
        "--throughput-audio",
        str(throughput_audio.resolve()),
        "--throughput-repetitions",
        str(throughput_repetitions),
        "--output",
        str(output),
        "--threshold",
        str(threshold),
    ]
    for dll_root in dll_roots:
        command.extend(("--dll-root", str(dll_root.resolve())))

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr)[-4000:]
        raise RuntimeError(f"{label} backend probe failed: {detail}")
    result = json.loads(output.read_text(encoding="utf-8"))
    archive = np.load(output.with_suffix(".npz"))
    posteriors = {key: np.asarray(archive[key]).copy() for key in archive.files}
    archive.close()
    return result, posteriors


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve(strict=True)
    model = args.model.resolve(strict=True)
    throughput_audio = args.throughput_audio.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="audioforge-ort-probe-") as temporary:
        temporary_root = Path(temporary)
        baseline_result, baseline_posteriors = _run_backend(
            label="baseline",
            python_root=args.baseline_python_root,
            dll_roots=args.baseline_dll_root,
            manifest=manifest,
            model=model,
            threshold=args.threshold,
            throughput_audio=throughput_audio,
            throughput_repetitions=args.throughput_repetitions,
            temporary_root=temporary_root,
        )
        candidate_result, candidate_posteriors = _run_backend(
            label="candidate",
            python_root=args.candidate_python_root,
            dll_roots=args.candidate_dll_root,
            manifest=manifest,
            model=model,
            threshold=args.threshold,
            throughput_audio=throughput_audio,
            throughput_repetitions=args.throughput_repetitions,
            temporary_root=temporary_root,
        )

    comparison = _compare_posteriors(baseline_posteriors, candidate_posteriors)
    baseline_throughput = baseline_result["throughput"]
    candidate_throughput = candidate_result["throughput"]
    median_delta = (
        100.0
        * (candidate_throughput["median_seconds"] - baseline_throughput["median_seconds"])
        / baseline_throughput["median_seconds"]
    )
    p95_delta = (
        100.0
        * (candidate_throughput["p95_seconds"] - baseline_throughput["p95_seconds"])
        / baseline_throughput["p95_seconds"]
    )
    rtf_delta = (
        100.0
        * (candidate_throughput["realtime_factor"] - baseline_throughput["realtime_factor"])
        / baseline_throughput["realtime_factor"]
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "Isolated CPU-only ONNX Runtime 1.23.2 VAD compatibility probe",
        "decision": {
            "production_adoption": "pending",
            "reason": "Software parity passed; ORT replacement and packaging remain a separately approved change.",
        },
        "method": {
            "evaluator": "python/tools/evaluate_onnxruntime_probe.py",
            "reproduction_command": (
                ".\\.venv\\Scripts\\python.exe python/tools/evaluate_onnxruntime_probe.py "
                "--baseline-python-root python --baseline-dll-root target/release "
                "--candidate-python-root target/onnxruntime-cpu-probe/candidate "
                "--candidate-dll-root target/onnxruntime-cpu-probe/candidate "
                "--manifest models/vad_eval_corpus/manifest.json "
                "--model models/silero_vad.onnx "
                "--throughput-audio models/vad_eval_silero_test.wav "
                "--throughput-repetitions 5 --threshold 0.48 "
                "--report evaluation/onnxruntime-cpu-probe.json"
            ),
            "manifest": _asset_record(manifest),
            "model": _asset_record(model),
            "throughput_fixture": _asset_record(throughput_audio),
            "threshold": args.threshold,
            "thresholds_checked": list(THRESHOLDS),
            "corpus_capture_count": baseline_result["capture_count"],
            "corpus_frame_count": baseline_result["frame_count"],
            "throughput_repetitions": args.throughput_repetitions,
            "native_extensions": {
                "baseline": baseline_result["native_module"],
                "candidate": candidate_result["native_module"],
            },
        },
        "results": {
            "posterior_parity": comparison,
            "quality_overall": {
                "baseline": baseline_result["quality_overall"],
                "candidate": candidate_result["quality_overall"],
            },
            "whole_clip_throughput": {
                "baseline": baseline_throughput,
                "candidate": candidate_throughput,
                "candidate_delta_percent": {
                    "median_seconds": median_delta,
                    "p95_seconds": p95_delta,
                    "realtime_factor": rtf_delta,
                },
            },
        },
        "gates": {
            "finite_posteriors": True,
            "max_abs_delta_le_1e-5": comparison["max_abs_delta"] <= 1e-5,
            "threshold_decisions_unchanged": all(
                value == 0 for value in comparison["threshold_changes"].values()
            ),
            "hysteresis_decisions_unchanged": comparison["hysteresis_surrogate"][
                "decision_changes"
            ]
            == 0,
            "whole_clip_throughput_measured": True,
            "distinct_native_extensions": baseline_result["native_module"]["sha256"]
            != candidate_result["native_module"]["sha256"],
            "per_frame_jitter_or_deadline_measured": False,
            "hardware_qualification_measured": False,
        },
        "limitations": [
            "Throughput is a whole-clip software measurement; it does not establish per-frame jitter or realtime callback deadlines.",
            "No microphone, output route, or hardware qualification was performed.",
            "The candidate was isolated from production and does not change the current runtime or package assets.",
            "The isolated candidate extension predates the telemetry opt-out source edit; rebuild the candidate from the final source before adopting it.",
        ],
        "source_sha256": {
            "python/tools/evaluate_onnxruntime_probe.py": _sha256(Path(__file__)),
        },
    }
    if args.cpu_archive is not None:
        archive = _asset_record(args.cpu_archive.resolve(strict=True))
        if archive["sha256"] != CPU_ORT_ARCHIVE_SHA256:
            raise ValueError(
                f"CPU ORT archive hash mismatch: expected {CPU_ORT_ARCHIVE_SHA256}, got {archive['sha256']}"
            )
        report["method"]["cpu_archive"] = archive
    if args.candidate_runtime_file:
        report["method"]["candidate_runtime_files"] = [
            _asset_record(path.resolve(strict=True))
            for path in args.candidate_runtime_file
        ]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-python-root", type=Path)
    parser.add_argument("--candidate-python-root", type=Path)
    parser.add_argument("--baseline-dll-root", type=Path, action="append", default=[])
    parser.add_argument("--candidate-dll-root", type=Path, action="append", default=[])
    parser.add_argument("--python-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dll-root", type=Path, action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--throughput-audio", type=Path, required=True)
    parser.add_argument("--throughput-repetitions", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--cpu-archive", type=Path)
    parser.add_argument("--candidate-runtime-file", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.throughput_repetitions < 1:
        raise SystemExit("--throughput-repetitions must be positive")
    if args.child:
        if args.python_root is None or args.output is None:
            raise SystemExit("child probe requires --python-root and --output")
        _measure_backend(
            python_root=args.python_root,
            dll_roots=args.dll_root,
            manifest=args.manifest,
            model=args.model,
            threshold=args.threshold,
            throughput_audio=args.throughput_audio,
            throughput_repetitions=args.throughput_repetitions,
            output=args.output,
        )
        return 0

    required = {
        "--baseline-python-root": args.baseline_python_root,
        "--candidate-python-root": args.candidate_python_root,
        "--report": args.report,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    report = evaluate(args)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
