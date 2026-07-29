"""Compare the shipped nnnoiseless model with pinned upstream Xiph RNNoise."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags, resample_poly, stft


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "models" / "dpdfnet_eval_subset"
DEFAULT_SHIPPED_BINARY = REPO_ROOT / "target" / "release" / "rnnoise_benchmark.exe"
DEFAULT_UPSTREAM_BINARY = (
    REPO_ROOT / "models" / "benchmarks" / "rnnoise-upstream-benchmark.exe"
)
DEFAULT_UPSTREAM_ROOT = REPO_ROOT / "models" / "benchmarks" / "upstream-rnnoise"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "rnnoise-backend-comparison.json"
SAMPLE_RATE = 48_000
FRAME_SIZE = 480
ARCHIVE_BASELINE_BYTES = 112_014_596
UPSTREAM_COMMIT = "70f1d256acd4b34a572f999a05c87bf00b67730d"
UPSTREAM_MODEL_HASH = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"


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
        return np.asarray(audio, dtype=np.float64)
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    return np.asarray(
        resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor),
        dtype=np.float64,
    )


def _paired_paths(root: Path, max_languages: int) -> list[tuple[Path, Path]]:
    selected: dict[str, tuple[Path, Path]] = {}
    for clean_path in sorted((root / "Clean").glob("*_clean.wav")):
        language = clean_path.name.split("_", 1)[0]
        noisy_path = (
            root / "Noisy" / clean_path.name.replace("_clean.wav", "_noisy.wav")
        )
        if language not in selected and noisy_path.is_file():
            selected[language] = (clean_path, noisy_path)
    pairs = list(selected.values())[:max_languages]
    if not pairs:
        raise RuntimeError(f"No paired clean/noisy WAV files found under {root}")
    return pairs


def _build_streams(
    pairs: list[tuple[Path, Path]], clip_seconds: float
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    clip_samples = int(round(clip_seconds * SAMPLE_RATE))
    separator = np.zeros(SAMPLE_RATE // 2, dtype=np.float64)
    clean_parts: list[np.ndarray] = []
    noisy_parts: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    cursor = 0
    for clean_path, noisy_path in pairs:
        clean_rate, clean = _read_mono(clean_path)
        noisy_rate, noisy = _read_mono(noisy_path)
        clean = _resample(clean, clean_rate)
        noisy = _resample(noisy, noisy_rate)
        common = min(clean.size, noisy.size)
        if common < clip_samples:
            raise RuntimeError(f"{clean_path.name} is shorter than requested clip")
        offset = min(15 * SAMPLE_RATE, max(0, (common - clip_samples) // 2))
        start = cursor
        end = start + clip_samples
        segments.append(
            {
                "id": clean_path.name.removesuffix("_mixture_clean.wav"),
                "language": clean_path.name.split("_", 1)[0],
                "clean_path": _relative(clean_path),
                "noisy_path": _relative(noisy_path),
                "stream_start": start,
                "stream_end": end,
            }
        )
        clean_parts.extend((clean[offset : offset + clip_samples], separator))
        noisy_parts.extend((noisy[offset : offset + clip_samples], separator))
        cursor = end + separator.size
    return (
        np.concatenate(clean_parts).astype(np.float32),
        np.concatenate(noisy_parts).astype(np.float32),
        segments,
    )


def _run_backend(
    binary: Path, audio: np.ndarray, work: Path, name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    input_path = work / f"{name}-input.f32"
    output_path = work / f"{name}-output.f32"
    metadata_path = work / f"{name}-metadata.json"
    np.asarray(audio, dtype="<f4").tofile(input_path)
    subprocess.run(
        [str(binary), str(input_path), str(output_path), str(metadata_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = np.fromfile(output_path, dtype="<f4").astype(np.float64)
    if output.size != audio.size:
        raise RuntimeError(
            f"{binary.name} returned {output.size} samples for {audio.size} inputs"
        )
    return output, json.loads(metadata_path.read_text(encoding="utf-8"))


def _delay_samples(reference: np.ndarray, estimate: np.ndarray) -> int:
    count = min(reference.size, estimate.size, 10 * SAMPLE_RATE)
    left = estimate[:count] - np.mean(estimate[:count])
    right = reference[:count] - np.mean(reference[:count])
    correlation = correlate(left, right, mode="full", method="fft")
    lags = correlation_lags(left.size, right.size, mode="full")
    allowed = np.abs(lags) <= 2 * FRAME_SIZE
    return int(lags[allowed][np.argmax(correlation[allowed])])


def _aligned(
    reference: np.ndarray, estimate: np.ndarray, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    if delay > 0:
        return reference[:-delay], estimate[delay:]
    if delay < 0:
        return reference[-delay:], estimate[:delay]
    return reference, estimate


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    energy = float(np.dot(reference, reference))
    if energy <= 1e-12:
        return 0.0
    target = reference * (float(np.dot(estimate, reference)) / energy)
    residual = estimate - target
    return float(
        10.0
        * np.log10(
            (float(np.dot(target, target)) + 1e-12)
            / (float(np.dot(residual, residual)) + 1e-12)
        )
    )


def _speech_lsd(reference: np.ndarray, estimate: np.ndarray) -> float:
    _, _, ref_spec = stft(
        reference,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=960,
        noverlap=480,
        boundary=cast(str, None),
        padded=False,
    )
    _, _, est_spec = stft(
        estimate,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=960,
        noverlap=480,
        boundary=cast(str, None),
        padded=False,
    )
    count = min(ref_spec.shape[1], est_spec.shape[1])
    ref_mag = np.abs(ref_spec[:, :count])
    est_mag = np.abs(est_spec[:, :count])
    frame_rms = np.sqrt(np.mean(np.square(reference.reshape(-1, FRAME_SIZE)), axis=1))
    active = frame_rms >= max(np.percentile(frame_rms, 95) * 0.03, 10 ** (-45 / 20))
    active_stft = active[1 : 1 + count]
    if active_stft.size < count:
        active_stft = np.pad(active_stft, (0, count - active_stft.size))
    floor = np.maximum(np.max(ref_mag, axis=0, keepdims=True) * 1e-4, 1e-7)
    difference_db = 20.0 * np.log10(
        np.maximum(est_mag, floor) / np.maximum(ref_mag, floor)
    )
    selected = difference_db[:, active_stft[:count]]
    return float(np.sqrt(np.mean(np.square(selected)))) if selected.size else 0.0


def _dropout_rate(reference: np.ndarray, estimate: np.ndarray) -> float:
    count = min(reference.size, estimate.size) // FRAME_SIZE
    ref_frames = reference[: count * FRAME_SIZE].reshape(count, FRAME_SIZE)
    est_frames = estimate[: count * FRAME_SIZE].reshape(count, FRAME_SIZE)
    ref_rms = np.sqrt(np.mean(np.square(ref_frames), axis=1) + 1e-12)
    est_rms = np.sqrt(np.mean(np.square(est_frames), axis=1) + 1e-12)
    active = ref_rms >= max(np.percentile(ref_rms, 95) * 0.03, 10 ** (-45 / 20))
    return (
        float(np.mean(est_rms[active] < ref_rms[active] * 0.10))
        if np.any(active)
        else 0.0
    )


def _segment_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    noisy_output: np.ndarray,
    clean_output: np.ndarray,
    segments: list[dict[str, Any]],
    noisy_delay: int,
    clean_delay: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trim = 2 * FRAME_SIZE
    for segment in segments:
        start = int(segment["stream_start"]) + trim
        end = int(segment["stream_end"]) - trim
        clean_reference = clean[start:end].astype(np.float64)
        noisy_reference = noisy[start:end].astype(np.float64)
        noisy_start = start + noisy_delay
        noisy_end = end + noisy_delay
        clean_start = start + clean_delay
        clean_end = end + clean_delay
        if (
            min(noisy_start, clean_start) < 0
            or max(noisy_end, clean_end) > noisy_output.size
        ):
            raise RuntimeError("aligned segment falls outside benchmark stream")
        enhanced = noisy_output[noisy_start:noisy_end]
        preserved = clean_output[clean_start:clean_end]
        rows.append(
            {
                "id": segment["id"],
                "language": segment["language"],
                "input_si_sdr_db": _si_sdr(clean_reference, noisy_reference),
                "enhanced_si_sdr_db": _si_sdr(clean_reference, enhanced),
                "si_sdr_improvement_db": _si_sdr(clean_reference, enhanced)
                - _si_sdr(clean_reference, noisy_reference),
                "noisy_speech_lsd_db": _speech_lsd(clean_reference, enhanced),
                "noisy_speech_dropout_rate": _dropout_rate(clean_reference, enhanced),
                "clean_si_sdr_db": _si_sdr(clean_reference, preserved),
                "clean_speech_lsd_db": _speech_lsd(clean_reference, preserved),
                "clean_speech_dropout_rate": _dropout_rate(clean_reference, preserved),
            }
        )
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def _decision(
    shipped_rows: list[dict[str, Any]],
    upstream_rows: list[dict[str, Any]],
    shipped_runtime: dict[str, Any],
    upstream_runtime: dict[str, Any],
    size_delta_bytes: int,
) -> dict[str, Any]:
    shipped = {
        key: _median(shipped_rows, key)
        for key in (
            "si_sdr_improvement_db",
            "noisy_speech_lsd_db",
            "noisy_speech_dropout_rate",
            "clean_si_sdr_db",
            "clean_speech_lsd_db",
            "clean_speech_dropout_rate",
        )
    }
    upstream = {
        key: _median(upstream_rows, key)
        for key in (
            "si_sdr_improvement_db",
            "noisy_speech_lsd_db",
            "noisy_speech_dropout_rate",
            "clean_si_sdr_db",
            "clean_speech_lsd_db",
            "clean_speech_dropout_rate",
        )
    }
    runtime_ratio = float(upstream_runtime["frame_p99_seconds"]) / max(
        float(shipped_runtime["frame_p99_seconds"]), 1e-12
    )
    gates = {
        "material_noisy_quality_win": (
            upstream["si_sdr_improvement_db"] >= shipped["si_sdr_improvement_db"] + 0.5
            and upstream["noisy_speech_lsd_db"] <= shipped["noisy_speech_lsd_db"]
        ),
        "clean_si_sdr_non_regression": (
            upstream["clean_si_sdr_db"] >= shipped["clean_si_sdr_db"] - 0.25
        ),
        "clean_lsd_non_regression": (
            upstream["clean_speech_lsd_db"] <= shipped["clean_speech_lsd_db"] + 0.25
        ),
        "clean_dropout_non_regression": (
            upstream["clean_speech_dropout_rate"]
            <= shipped["clean_speech_dropout_rate"] + 0.005
        ),
        "p99_runtime_ratio_at_most_1_5": runtime_ratio <= 1.5,
        "p99_meets_10ms_deadline": float(upstream_runtime["frame_p99_seconds"])
        <= 0.010,
        "estimated_archive_growth_at_most_5_percent": (
            size_delta_bytes / ARCHIVE_BASELINE_BYTES <= 0.05
        ),
    }
    retain = all(gates.values())
    return {
        "predefined_gates": {
            "si_sdr_improvement_over_shipped_min_db": 0.5,
            "noisy_lsd_must_not_regress": True,
            "clean_si_sdr_regression_max_db": 0.25,
            "clean_lsd_regression_max_db": 0.25,
            "clean_dropout_regression_max": 0.005,
            "p99_runtime_ratio_max": 1.5,
            "frame_deadline_seconds": 0.010,
            "estimated_archive_growth_max_ratio": 0.05,
        },
        "shipped_medians": shipped,
        "upstream_medians": upstream,
        "upstream_to_shipped_p99_runtime_ratio": runtime_ratio,
        "estimated_archive_growth_bytes": size_delta_bytes,
        "estimated_archive_growth_ratio": size_delta_bytes / ARCHIVE_BASELINE_BYTES,
        "gates": gates,
        "decision": "adopt_upstream" if retain else "retain_nnnoiseless",
    }


def evaluate(
    *,
    corpus_root: Path,
    shipped_binary: Path,
    upstream_binary: Path,
    upstream_root: Path,
    report_path: Path,
    max_languages: int,
    clip_seconds: float,
) -> dict[str, Any]:
    pairs = _paired_paths(corpus_root, max_languages)
    clean, noisy, segments = _build_streams(pairs, clip_seconds)
    with tempfile.TemporaryDirectory(prefix="audioforge-rnnoise-") as temp:
        work = Path(temp)
        shipped_noisy, shipped_noisy_runtime = _run_backend(
            shipped_binary, noisy, work, "shipped-noisy"
        )
        shipped_clean, shipped_clean_runtime = _run_backend(
            shipped_binary, clean, work, "shipped-clean"
        )
        upstream_noisy, upstream_noisy_runtime = _run_backend(
            upstream_binary, noisy, work, "upstream-noisy"
        )
        upstream_clean, upstream_clean_runtime = _run_backend(
            upstream_binary, clean, work, "upstream-clean"
        )

    delays = {
        "shipped_noisy": _delay_samples(noisy, shipped_noisy),
        "shipped_clean": _delay_samples(clean, shipped_clean),
        "upstream_noisy": _delay_samples(noisy, upstream_noisy),
        "upstream_clean": _delay_samples(clean, upstream_clean),
    }
    shipped_rows = _segment_metrics(
        clean,
        noisy,
        shipped_noisy,
        shipped_clean,
        segments,
        delays["shipped_noisy"],
        delays["shipped_clean"],
    )
    upstream_rows = _segment_metrics(
        clean,
        noisy,
        upstream_noisy,
        upstream_clean,
        segments,
        delays["upstream_noisy"],
        delays["upstream_clean"],
    )
    shipped_runtime = {
        key: max(float(shipped_noisy_runtime[key]), float(shipped_clean_runtime[key]))
        for key in shipped_noisy_runtime
        if key not in {"frames", "samples"}
    }
    upstream_runtime = {
        key: max(float(upstream_noisy_runtime[key]), float(upstream_clean_runtime[key]))
        for key in upstream_noisy_runtime
        if key not in {"frames", "samples"}
    }
    size_delta = max(0, upstream_binary.stat().st_size - shipped_binary.stat().st_size)
    decision = _decision(
        shipped_rows, upstream_rows, shipped_runtime, upstream_runtime, size_delta
    )
    current_commit = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"unexpected upstream RNNoise commit: {current_commit}")
    model_source = upstream_root / "src" / "rnnoise_data.c"
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "audible_change": False,
        "purpose": "Retain-or-reject comparison; production backend remains unchanged unless every predefined gate passes.",
        "corpus": {
            "root": _relative(corpus_root),
            "languages": len(pairs),
            "clip_seconds_per_language": clip_seconds,
            "paired_clean_noisy": True,
            "segments": segments,
        },
        "provenance": {
            "shipped": {
                "implementation": "nnnoiseless 0.5.2 (Cargo.lock)",
                "binary": _relative(shipped_binary),
                "binary_sha256": _sha256(shipped_binary),
                "compiler_profile": "cargo release, portable target defaults",
            },
            "upstream": {
                "repository": "https://github.com/xiph/rnnoise",
                "commit": UPSTREAM_COMMIT,
                "model_version_sha256": UPSTREAM_MODEL_HASH,
                "model_source": _relative(model_source),
                "model_source_sha256": _sha256(model_source),
                "binary": _relative(upstream_binary),
                "binary_sha256": _sha256(upstream_binary),
                "compiler": "Zig 0.16.0 cc -O3, portable target defaults, scalar path",
                "compiler_archive_sha256": "68659eb5f1e4eb1437a722f1dd889c5a322c9954607f5edcf337bc3684a75a7e",
            },
            "sample_rate_hz": SAMPLE_RATE,
            "frame_size": FRAME_SIZE,
            "strength": 1.0,
            "sample_transport": "raw little-endian float32; model input scaled to +/-32768",
        },
        "alignment_delay_samples": delays,
        "runtime": {
            "shipped_worst_of_clean_noisy": shipped_runtime,
            "upstream_worst_of_clean_noisy": upstream_runtime,
        },
        "rows": {"shipped": shipped_rows, "upstream": upstream_rows},
        "decision": decision,
        "listening_validation": {
            "status": "not_run",
            "reason": "No human involvement requested; objective gate must pass before listening could justify adoption.",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--shipped-binary", type=Path, default=DEFAULT_SHIPPED_BINARY)
    parser.add_argument("--upstream-binary", type=Path, default=DEFAULT_UPSTREAM_BINARY)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-languages", type=int, default=12)
    parser.add_argument("--clip-seconds", type=float, default=15.0)
    args = parser.parse_args()
    report = evaluate(
        corpus_root=args.corpus_root.resolve(),
        shipped_binary=args.shipped_binary.resolve(),
        upstream_binary=args.upstream_binary.resolve(),
        upstream_root=args.upstream_root.resolve(),
        report_path=args.report.resolve(),
        max_languages=args.max_languages,
        clip_seconds=args.clip_seconds,
    )
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
