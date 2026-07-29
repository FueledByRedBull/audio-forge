"""Retain or reject VAD-driven auto makeup on long-form real speech."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from mic_eq import analyze_vad_probabilities, simulate_auto_makeup_control


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "models" / "dpdfnet_eval_subset"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "auto-makeup-real-speech-report.json"
SAMPLE_RATE = 48_000
CONTROL_BLOCK_SIZE = 480
CONTROL_CADENCE_HZ = SAMPLE_RATE / CONTROL_BLOCK_SIZE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    audio = np.asarray(raw)
    if audio.ndim == 2:
        audio = np.mean(audio.astype(np.float64), axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        bits = audio.dtype.itemsize * 8
        full_scale = (
            2 ** (bits - 1)
            if np.issubdtype(audio.dtype, np.signedinteger)
            else 2**bits - 1
        )
        audio = audio.astype(np.float64) / float(full_scale)
    return int(sample_rate), np.asarray(audio, dtype=np.float64)


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    return np.asarray(
        resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def _pairs(corpus_root: Path, max_languages: int) -> list[tuple[Path, Path]]:
    selected: dict[str, tuple[Path, Path]] = {}
    for clean in sorted((corpus_root / "Clean").glob("*_clean.wav")):
        language = clean.name.split("_", 1)[0]
        noisy = corpus_root / "Noisy" / clean.name.replace("_clean.wav", "_noisy.wav")
        if language not in selected and noisy.is_file():
            selected[language] = (clean, noisy)
    pairs = list(selected.values())[:max_languages]
    if not pairs:
        raise RuntimeError(f"No clean/noisy pairs found under {corpus_root}")
    return pairs


def _control_probabilities(
    frame_probabilities: np.ndarray,
    sample_count: int,
    block_count: int,
) -> np.ndarray:
    if frame_probabilities.size == 0:
        return np.zeros(block_count, dtype=np.float64)
    duration = sample_count / SAMPLE_RATE
    source_times = (np.arange(frame_probabilities.size) + 0.5) * (
        duration / frame_probabilities.size
    )
    target_times = (np.arange(block_count) + 0.5) / CONTROL_CADENCE_HZ
    return np.interp(
        target_times,
        source_times,
        frame_probabilities,
        left=float(frame_probabilities[0]),
        right=float(frame_probabilities[-1]),
    )


def _block_rms_db(audio: np.ndarray) -> np.ndarray:
    blocks = [
        audio[start : start + CONTROL_BLOCK_SIZE]
        for start in range(0, audio.size, CONTROL_BLOCK_SIZE)
    ]
    return np.asarray(
        [
            20.0
            * np.log10(
                max(float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))), 1e-9)
            )
            for block in blocks
        ],
        dtype=np.float64,
    )


def _pumping_score(trace_db: np.ndarray) -> float:
    if trace_db.size < 10:
        return 0.0
    centered = trace_db - np.mean(trace_db)
    spectrum = np.fft.rfft(centered * np.hanning(centered.size))
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / CONTROL_CADENCE_HZ)
    band = (frequencies >= 2.0) & (frequencies <= 8.0)
    total = float(np.sum(np.square(np.abs(spectrum))))
    if total <= 1e-12 or not np.any(band):
        return 0.0
    return float(np.sqrt(np.sum(np.square(np.abs(spectrum[band]))) / total))


def _run_clip(
    clean_path: Path,
    noisy_path: Path,
    *,
    clip_seconds: float,
) -> dict[str, Any]:
    clean_rate, clean = _read_mono(clean_path)
    noisy_rate, noisy = _read_mono(noisy_path)
    clean = _resample(clean, clean_rate)
    noisy = _resample(noisy, noisy_rate)
    length = min(clean.size, noisy.size)
    clip_samples = int(round(clip_seconds * SAMPLE_RATE))
    if length < clip_samples:
        raise RuntimeError(
            f"{clean_path.name} is shorter than {clip_seconds:.1f} seconds"
        )
    offset = min(15 * SAMPLE_RATE, max(0, (length - clip_samples) // 2))
    clean = clean[offset : offset + clip_samples]
    noisy = noisy[offset : offset + clip_samples]
    block_count = (noisy.size + CONTROL_BLOCK_SIZE - 1) // CONTROL_BLOCK_SIZE

    clean_frames = np.asarray(
        analyze_vad_probabilities(clean, SAMPLE_RATE, 0.48), dtype=np.float64
    )
    noisy_frames = np.asarray(
        analyze_vad_probabilities(noisy, SAMPLE_RATE, 0.48), dtype=np.float64
    )
    clean_control = _control_probabilities(clean_frames, clean.size, block_count)
    noisy_control = _control_probabilities(noisy_frames, noisy.size, block_count)
    active = clean_control >= 0.48
    inactive = clean_control <= 0.20
    noisy_rms_db = _block_rms_db(noisy)
    noise_floor_db = (
        float(np.median(noisy_rms_db[inactive]))
        if np.any(inactive)
        else float(np.percentile(noisy_rms_db, 20.0))
    )

    settings = {
        "vad_reliability": 1.0,
        "adaptive_release": True,
        "return_output_audio": True,
    }
    candidate = simulate_auto_makeup_control(
        noisy,
        SAMPLE_RATE,
        noisy_control.tolist(),
        noise_floor_db,
        1.0,
        settings,
    )
    baseline = simulate_auto_makeup_control(
        noisy,
        SAMPLE_RATE,
        [],
        noise_floor_db,
        1.0,
        settings,
    )
    candidate_gain = np.asarray(candidate["makeup_gain_db"], dtype=np.float64)
    baseline_gain = np.asarray(baseline["makeup_gain_db"], dtype=np.float64)
    candidate_output = np.asarray(candidate["output_audio"], dtype=np.float64)
    baseline_output = np.asarray(baseline["output_audio"], dtype=np.float64)
    count = min(candidate_gain.size, baseline_gain.size, active.size)
    active = active[:count]
    inactive = inactive[:count]
    candidate_gain = candidate_gain[:count]
    baseline_gain = baseline_gain[:count]

    def masked_median(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.median(values[mask])) if np.any(mask) else 0.0

    boundaries = np.arange(CONTROL_BLOCK_SIZE, noisy.size, CONTROL_BLOCK_SIZE)
    input_jumps = np.abs(noisy[boundaries] - noisy[boundaries - 1])
    candidate_jumps = np.abs(
        candidate_output[boundaries] - candidate_output[boundaries - 1]
    )
    baseline_jumps = np.abs(
        baseline_output[boundaries] - baseline_output[boundaries - 1]
    )
    candidate_excess = np.maximum(candidate_jumps - input_jumps, 0.0)
    baseline_excess = np.maximum(baseline_jumps - input_jumps, 0.0)

    return {
        "id": clean_path.name.removesuffix("_mixture_clean.wav"),
        "language": clean_path.name.split("_", 1)[0],
        "clean_path": _relative(clean_path),
        "noisy_path": _relative(noisy_path),
        "source_offset_samples": int(round(offset * clean_rate / SAMPLE_RATE)),
        "duration_seconds": noisy.size / SAMPLE_RATE,
        "active_block_ratio": float(np.mean(active)),
        "inactive_block_ratio": float(np.mean(inactive)),
        "noise_floor_db": noise_floor_db,
        "candidate_active_makeup_db": masked_median(candidate_gain, active),
        "baseline_active_makeup_db": masked_median(baseline_gain, active),
        "candidate_inactive_makeup_db": masked_median(candidate_gain, inactive),
        "baseline_inactive_makeup_db": masked_median(baseline_gain, inactive),
        "candidate_pumping_score": _pumping_score(candidate_gain),
        "baseline_pumping_score": _pumping_score(baseline_gain),
        "candidate_max_transition_db": float(
            np.max(np.abs(np.diff(candidate_gain)), initial=0.0)
        ),
        "baseline_max_transition_db": float(
            np.max(np.abs(np.diff(baseline_gain)), initial=0.0)
        ),
        "candidate_p99_boundary_excess_linear": float(
            np.percentile(candidate_excess, 99.0)
        ),
        "baseline_p99_boundary_excess_linear": float(
            np.percentile(baseline_excess, 99.0)
        ),
        "candidate_max_boundary_excess_linear": float(
            np.max(candidate_excess, initial=0.0)
        ),
        "candidate_p99_block_runtime_ms": float(candidate["p99_block_runtime_ms"]),
        "baseline_p99_block_runtime_ms": float(baseline["p99_block_runtime_ms"]),
        "candidate_final_makeup_db": float(candidate_gain[-1]),
        "baseline_final_makeup_db": float(baseline_gain[-1]),
    }


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def _percentile(rows: list[dict[str, Any]], key: str, value: float) -> float:
    return float(np.percentile([float(row[key]) for row in rows], value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--max-languages", type=int, default=12)
    args = parser.parse_args()

    pairs = _pairs(args.corpus_root, args.max_languages)
    rows = [
        _run_clip(clean, noisy, clip_seconds=args.clip_seconds)
        for clean, noisy in pairs
    ]
    metrics = {
        "median_candidate_active_makeup_db": _median(
            rows, "candidate_active_makeup_db"
        ),
        "median_baseline_active_makeup_db": _median(rows, "baseline_active_makeup_db"),
        "median_candidate_inactive_makeup_db": _median(
            rows, "candidate_inactive_makeup_db"
        ),
        "median_baseline_inactive_makeup_db": _median(
            rows, "baseline_inactive_makeup_db"
        ),
        "median_candidate_pumping_score": _median(rows, "candidate_pumping_score"),
        "median_baseline_pumping_score": _median(rows, "baseline_pumping_score"),
        "p95_candidate_p99_boundary_excess_linear": _percentile(
            rows, "candidate_p99_boundary_excess_linear", 95.0
        ),
        "p95_baseline_p99_boundary_excess_linear": _percentile(
            rows, "baseline_p99_boundary_excess_linear", 95.0
        ),
        "max_candidate_p99_block_runtime_ms": max(
            float(row["candidate_p99_block_runtime_ms"]) for row in rows
        ),
        "minimum_active_block_ratio": min(
            float(row["active_block_ratio"]) for row in rows
        ),
        "minimum_inactive_block_ratio": min(
            float(row["inactive_block_ratio"]) for row in rows
        ),
    }
    gates = {
        "all_clips_contain_active_and_inactive_evidence": (
            metrics["minimum_active_block_ratio"] >= 0.05
            and metrics["minimum_inactive_block_ratio"] >= 0.02
        ),
        "speech_convergence_retained": (
            metrics["median_candidate_active_makeup_db"] >= 0.5
            and metrics["median_candidate_active_makeup_db"]
            >= metrics["median_baseline_active_makeup_db"] - 1.0
        ),
        "inactive_gain_not_worse_than_rms_fallback": (
            metrics["median_candidate_inactive_makeup_db"]
            <= metrics["median_baseline_inactive_makeup_db"] + 0.25
        ),
        "pumping_not_worse_than_rms_fallback": (
            metrics["median_candidate_pumping_score"]
            <= metrics["median_baseline_pumping_score"] + 0.05
        ),
        "boundary_discontinuity_not_worse_than_rms_fallback": (
            metrics["p95_candidate_p99_boundary_excess_linear"]
            <= max(
                0.01,
                metrics["p95_baseline_p99_boundary_excess_linear"] + 0.001,
            )
        ),
        "p99_control_work_within_10ms_deadline": (
            metrics["max_candidate_p99_block_runtime_ms"] <= 10.0
        ),
    }
    retained = all(gates.values())
    manifest = json.loads(
        (REPO_ROOT / "release-assets.json").read_text(encoding="utf-8")
    )
    silero = next(
        asset
        for asset in manifest["assets"]
        if asset["path"] == "models/silero_vad.onnx"
    )
    report = {
        "schema_version": 2,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "VAD/noise-reliability auto-makeup versus RMS-only fallback",
        "retained": retained,
        "predefined_gates": {
            "minimum_active_block_ratio": 0.05,
            "minimum_inactive_block_ratio": 0.02,
            "minimum_candidate_active_makeup_db": 0.5,
            "maximum_active_makeup_regret_db": 1.0,
            "maximum_inactive_makeup_regression_db": 0.25,
            "maximum_pumping_score_regression": 0.05,
            "maximum_p99_boundary_excess_linear": 0.01,
            "maximum_boundary_regression_linear": 0.001,
            "maximum_p99_block_runtime_ms": 10.0,
        },
        "metrics": metrics,
        "gates": gates,
        "cases": rows,
        "evaluation_contract": {
            "configuration": {
                "sample_rate": SAMPLE_RATE,
                "control_block_size": CONTROL_BLOCK_SIZE,
                "vad_threshold": 0.48,
                "compressor": {
                    "threshold_db": -24.0,
                    "ratio": 3.0,
                    "attack_ms": 10.0,
                    "release_ms": 180.0,
                    "target_lufs": -18.0,
                    "adaptive_release": True,
                },
            },
            "asset_hashes": {
                silero["path"]: silero["sha256"],
            },
            "runtime": {
                "max_p99_frame_seconds": (
                    metrics["max_candidate_p99_block_runtime_ms"] / 1000.0
                ),
                "control_deadline_seconds": CONTROL_BLOCK_SIZE / SAMPLE_RATE,
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "latency": {
                "control_block_samples": CONTROL_BLOCK_SIZE,
                "additional_audio_latency_samples": 0,
            },
            "clean_preservation": {
                "method": (
                    "The controller changes compressor makeup only; active-speech "
                    "gain retention is gated against the RMS-only fallback."
                ),
                "active_makeup_regret_db": (
                    metrics["median_baseline_active_makeup_db"]
                    - metrics["median_candidate_active_makeup_db"]
                ),
            },
            "listening_status": {
                "status": "not_run",
                "reason": (
                    "This hardening goal excludes human involvement; the VAD controller "
                    "is retained only if every predefined real-speech objective gate passes."
                ),
            },
        },
        "limitations": [
            "The supplied corpus contains simulated mixtures rather than native 48 kHz close-mic captures.",
            "Clean-reference Silero posteriors define active/inactive evaluation masks; the controller consumes noisy-mixture posteriors.",
            "The 12-language sample is one 30-second segment per language and does not replace controlled listening.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"retained": retained, "metrics": metrics}, sort_keys=True))
    print(f"Wrote {_relative(args.report)}")
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())
