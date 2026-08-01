"""Benchmark AudioForge's DeepFilter runtime configuration and mix alignment."""

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
from scipy.signal import correlate, correlation_lags, resample_poly, stft

from mic_eq.analysis.wav_io import read_mono_wav


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "models" / "dpdfnet_eval_subset"
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "deepfilter_benchmark.exe"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "deepfilter-hardening-report.json"
SAMPLE_RATE = 48_000
FRAME_SIZE = 480
ATTENUATION_CANDIDATES_DB = (12.0, 20.0, 30.0, 80.0)
POST_FILTER_CANDIDATES = (0.0, 0.02, 0.05)
MODELS = ("ll", "standard")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    return read_mono_wav(path, dtype=np.float64)


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float64)
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    return np.asarray(
        resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor),
        dtype=np.float64,
    )


def _paired_paths(corpus_root: Path, max_languages: int) -> list[tuple[Path, Path]]:
    corpus_root = corpus_root.resolve(strict=True)
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ValueError("corpus manifest must contain a files list")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("corpus manifest contains an invalid file record")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe corpus manifest path: {relative}")
        key = relative.as_posix()
        if key in indexed:
            raise ValueError(f"duplicate corpus manifest path: {key}")
        indexed[key] = record

    def verified(relative: Path, expected_model: str) -> Path:
        key = relative.as_posix()
        record = indexed.get(key)
        if record is None or record.get("model_name") != expected_model:
            raise ValueError(f"missing {expected_model} manifest record: {key}")
        path = (corpus_root / relative).resolve(strict=True)
        if not path.is_relative_to(corpus_root):
            raise ValueError(f"corpus path escapes root: {key}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"corpus size mismatch: {key}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"corpus hash mismatch: {key}")
        return path

    selected: dict[str, tuple[Path, Path]] = {}
    clean_records = sorted(
        (
            record
            for record in records
            if isinstance(record, dict) and record.get("model_name") == "Clean"
        ),
        key=lambda record: str(record["path"]),
    )
    for record in clean_records:
        clean_relative = Path(str(record["path"]))
        language = str(record.get("language") or clean_relative.name.split("_", 1)[0])
        noisy_relative = Path("Noisy") / clean_relative.name.replace(
            "_clean.wav", "_noisy.wav"
        )
        if language not in selected:
            selected[language] = (
                verified(clean_relative, "Clean"),
                verified(noisy_relative, "Noisy"),
            )
    pairs = list(selected.values())[:max_languages]
    if not pairs:
        raise RuntimeError(f"No paired clean/noisy WAV files found under {corpus_root}")
    return pairs


def _build_streams(
    pairs: list[tuple[Path, Path]],
    clip_seconds: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    clean_parts: list[np.ndarray] = []
    noisy_parts: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    separator = np.zeros(SAMPLE_RATE // 2, dtype=np.float64)
    cursor = 0
    clip_samples = int(round(clip_seconds * SAMPLE_RATE))

    for clean_path, noisy_path in pairs:
        clean_rate, clean = _read_mono(clean_path)
        noisy_rate, noisy = _read_mono(noisy_path)
        clean = _resample(clean, clean_rate)
        noisy = _resample(noisy, noisy_rate)
        common = min(clean.size, noisy.size)
        if common < clip_samples:
            raise RuntimeError(
                f"{clean_path.name} is shorter than {clip_seconds:.1f} seconds"
            )
        offset = min(15 * SAMPLE_RATE, max(0, (common - clip_samples) // 2))
        clean_clip = clean[offset : offset + clip_samples]
        noisy_clip = noisy[offset : offset + clip_samples]
        start = cursor
        end = start + clip_samples
        segments.append(
            {
                "id": clean_path.name.removesuffix("_mixture_clean.wav"),
                "clean_path": _relative(clean_path),
                "noisy_path": _relative(noisy_path),
                "clean_sha256": _sha256(clean_path),
                "noisy_sha256": _sha256(noisy_path),
                "source_sample_rate": clean_rate,
                "source_offset_samples": int(round(offset * clean_rate / SAMPLE_RATE)),
                "stream_start": start,
                "stream_end": end,
            }
        )
        clean_parts.extend((clean_clip, separator))
        noisy_parts.extend((noisy_clip, separator))
        cursor = end + separator.size

    return (
        np.concatenate(clean_parts).astype(np.float32),
        np.concatenate(noisy_parts).astype(np.float32),
        segments,
    )


def _write_raw(path: Path, audio: np.ndarray) -> None:
    np.asarray(audio, dtype="<f4").tofile(path)


def _read_raw(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def _run_backend(
    binary: Path,
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    attenuation_db: float,
    post_filter_beta: float,
    strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    command = [
        str(binary),
        "--model",
        model,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--library",
        str(REPO_ROOT / "df.dll"),
        "--model-dir",
        str(REPO_ROOT / "models"),
        "--attenuation-db",
        str(attenuation_db),
        "--post-filter-beta",
        str(post_filter_beta),
        "--strength",
        str(strength),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("DeepFilter benchmark emitted no JSON result")
    metadata = json.loads(lines[-1])
    return _read_raw(output_path), metadata


def _frame_rms(
    audio: np.ndarray, frame_size: int = 960, hop_size: int = 480
) -> np.ndarray:
    if audio.size < frame_size:
        return np.asarray([], dtype=np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop_size]
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _active_mask(reference: np.ndarray) -> np.ndarray:
    rms = _frame_rms(reference)
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    peak_db = 20.0 * np.log10(max(float(np.percentile(rms, 95)), 1e-9))
    threshold_db = max(-45.0, peak_db - 35.0)
    return 20.0 * np.log10(np.maximum(rms, 1e-9)) >= threshold_db


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= 1e-12:
        return 0.0
    scale = float(np.dot(estimate, reference)) / reference_energy
    target = scale * reference
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
    frequencies, _, est_spec = stft(
        estimate,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=960,
        noverlap=480,
        boundary=cast(str, None),
        padded=False,
    )
    frame_count = min(ref_spec.shape[1], est_spec.shape[1])
    active = _active_mask(reference)[:frame_count]
    band = (frequencies >= 80.0) & (frequencies <= 16_000.0)
    if not np.any(active) or not np.any(band):
        return 0.0
    ref_mag = np.abs(ref_spec[band, :frame_count][:, active])
    est_mag = np.abs(est_spec[band, :frame_count][:, active])
    floor = max(float(np.max(ref_mag)) * 1e-4, 1e-8)
    ref_db = 20.0 * np.log10(np.maximum(ref_mag, floor))
    est_db = 20.0 * np.log10(np.maximum(est_mag, floor))
    return float(np.median(np.sqrt(np.mean(np.square(est_db - ref_db), axis=0))))


def _dropout_rate(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_rms = _frame_rms(reference)
    est_rms = _frame_rms(estimate)
    count = min(ref_rms.size, est_rms.size)
    active = _active_mask(reference)[:count]
    if not np.any(active):
        return 0.0
    ratios = est_rms[:count][active] / np.maximum(ref_rms[:count][active], 1e-9)
    return float(np.mean(ratios < 0.1))


def _segment_metrics(
    clean_stream: np.ndarray,
    noisy_stream: np.ndarray,
    enhanced_clean: np.ndarray,
    enhanced_noisy: np.ndarray,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for segment in segments:
        start = int(segment["stream_start"])
        end = int(segment["stream_end"])
        clean = clean_stream[start:end].astype(np.float64)
        noisy = noisy_stream[start:end].astype(np.float64)
        clean_out = enhanced_clean[start:end]
        noisy_out = enhanced_noisy[start:end]
        metrics.append(
            {
                "id": segment["id"],
                "noisy_input_si_sdr_db": _si_sdr(clean, noisy),
                "enhanced_noisy_si_sdr_db": _si_sdr(clean, noisy_out),
                "noisy_si_sdr_improvement_db": (
                    _si_sdr(clean, noisy_out) - _si_sdr(clean, noisy)
                ),
                "noisy_speech_lsd_db": _speech_lsd(clean, noisy_out),
                "clean_si_sdr_db": _si_sdr(clean, clean_out),
                "clean_speech_lsd_db": _speech_lsd(clean, clean_out),
                "clean_dropout_rate": _dropout_rate(clean, clean_out),
            }
        )
    return metrics


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, float]:
    segment_metrics = [metric for case in cases for metric in case["segments"]]
    runtime = [case["runtime"] for case in cases]
    return {
        "median_noisy_si_sdr_improvement_db": _median(
            [metric["noisy_si_sdr_improvement_db"] for metric in segment_metrics]
        ),
        "median_noisy_speech_lsd_db": _median(
            [metric["noisy_speech_lsd_db"] for metric in segment_metrics]
        ),
        "median_clean_si_sdr_db": _median(
            [metric["clean_si_sdr_db"] for metric in segment_metrics]
        ),
        "median_clean_speech_lsd_db": _median(
            [metric["clean_speech_lsd_db"] for metric in segment_metrics]
        ),
        "median_clean_dropout_rate": _median(
            [metric["clean_dropout_rate"] for metric in segment_metrics]
        ),
        "max_realtime_factor": max(float(item["rtf"]) for item in runtime),
        "max_p99_frame_seconds": max(
            float(item["p99_frame_seconds"]) for item in runtime
        ),
        "max_frame_seconds": max(float(item["max_frame_seconds"]) for item in runtime),
    }


def _best_lag(reference: np.ndarray, output: np.ndarray, max_lag: int = 2_000) -> int:
    sample_count = min(reference.size, SAMPLE_RATE * 20)
    reference = reference[:sample_count].astype(np.float64)
    output = output[: sample_count + max_lag].astype(np.float64)
    reference -= np.mean(reference)
    output -= np.mean(output)
    values = correlate(output, reference, mode="full", method="fft")
    lags = correlation_lags(output.size, reference.size, mode="full")
    allowed = (lags >= 0) & (lags <= max_lag)
    return int(lags[allowed][int(np.argmax(np.abs(values[allowed])))])


def _evaluate_case(
    binary: Path,
    temp_root: Path,
    clean_stream: np.ndarray,
    noisy_stream: np.ndarray,
    segments: list[dict[str, Any]],
    *,
    model: str,
    attenuation_db: float,
    post_filter_beta: float,
) -> dict[str, Any]:
    clean_output, clean_runtime = _run_backend(
        binary,
        temp_root / "clean.f32",
        temp_root / f"{model}-{attenuation_db:g}-{post_filter_beta:g}-clean.f32",
        model=model,
        attenuation_db=attenuation_db,
        post_filter_beta=post_filter_beta,
        strength=1.0,
    )
    noisy_output, noisy_runtime = _run_backend(
        binary,
        temp_root / "noisy.f32",
        temp_root / f"{model}-{attenuation_db:g}-{post_filter_beta:g}-noisy.f32",
        model=model,
        attenuation_db=attenuation_db,
        post_filter_beta=post_filter_beta,
        strength=1.0,
    )
    latency = int(noisy_runtime["latency_samples"])
    aligned_clean = clean_output[latency : latency + clean_stream.size]
    aligned_noisy = noisy_output[latency : latency + noisy_stream.size]
    if (
        aligned_clean.size != clean_stream.size
        or aligned_noisy.size != noisy_stream.size
    ):
        raise RuntimeError("DeepFilter output is too short after latency alignment")
    return {
        "model": model,
        "attenuation_limit_db": attenuation_db,
        "post_filter_beta": post_filter_beta,
        "measured_wet_lag_samples": _best_lag(noisy_stream, noisy_output),
        "declared_latency_samples": latency,
        "runtime": {
            "initialization_seconds": max(
                float(clean_runtime["initialization_seconds"]),
                float(noisy_runtime["initialization_seconds"]),
            ),
            "processing_seconds": max(
                float(clean_runtime["processing_seconds"]),
                float(noisy_runtime["processing_seconds"]),
            ),
            "rtf": max(float(clean_runtime["rtf"]), float(noisy_runtime["rtf"])),
            "frame_count": min(
                int(clean_runtime["frame_count"]),
                int(noisy_runtime["frame_count"]),
            ),
            "p95_frame_seconds": max(
                float(clean_runtime["p95_frame_seconds"]),
                float(noisy_runtime["p95_frame_seconds"]),
            ),
            "p99_frame_seconds": max(
                float(clean_runtime["p99_frame_seconds"]),
                float(noisy_runtime["p99_frame_seconds"]),
            ),
            "max_frame_seconds": max(
                float(clean_runtime["max_frame_seconds"]),
                float(noisy_runtime["max_frame_seconds"]),
            ),
            "frame_deadline_seconds": FRAME_SIZE / SAMPLE_RATE,
        },
        "segments": _segment_metrics(
            clean_stream,
            noisy_stream,
            aligned_clean,
            aligned_noisy,
            segments,
        ),
    }


def _mix_contract(
    binary: Path,
    temp_root: Path,
    noisy_stream: np.ndarray,
    baseline_cases: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for model in MODELS:
        wet_output, wet_runtime = _run_backend(
            binary,
            temp_root / "noisy.f32",
            temp_root / f"{model}-mix-wet.f32",
            model=model,
            attenuation_db=80.0,
            post_filter_beta=0.0,
            strength=1.0,
        )
        dry_output, _ = _run_backend(
            binary,
            temp_root / "noisy.f32",
            temp_root / f"{model}-mix-dry.f32",
            model=model,
            attenuation_db=80.0,
            post_filter_beta=0.0,
            strength=0.0,
        )
        half_output, _ = _run_backend(
            binary,
            temp_root / "noisy.f32",
            temp_root / f"{model}-mix-half.f32",
            model=model,
            attenuation_db=80.0,
            post_filter_beta=0.0,
            strength=0.5,
        )
        latency = int(wet_runtime["latency_samples"])
        length = noisy_stream.size
        wet = wet_output[latency : latency + length]
        dry = dry_output[latency : latency + length]
        half = half_output[latency : latency + length]
        expected_half = 0.5 * wet + 0.5 * noisy_stream.astype(np.float64)
        difference = half - expected_half
        dry_difference = dry - noisy_stream
        measured_lag = int(baseline_cases[model]["measured_wet_lag_samples"])
        contracts.append(
            {
                "model": model,
                "declared_latency_samples": latency,
                "measured_wet_lag_samples": measured_lag,
                "dry_max_abs_error": float(np.max(np.abs(dry_difference))),
                "half_mix_max_abs_error": float(np.max(np.abs(difference))),
                "half_mix_rms_error": float(np.sqrt(np.mean(np.square(difference)))),
                "passes": bool(
                    measured_lag == latency
                    and np.max(np.abs(dry_difference)) <= 1e-6
                    and np.max(np.abs(difference)) <= 2e-5
                ),
            }
        )
    return contracts


def _choose_attenuation(
    cases: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    aggregates = {
        attenuation: _aggregate(
            [
                case
                for case in cases
                if case["attenuation_limit_db"] == attenuation
                and case["post_filter_beta"] == 0.0
            ]
        )
        for attenuation in ATTENUATION_CANDIDATES_DB
    }
    baseline = aggregates[80.0]
    gates: dict[str, Any] = {}
    passing: list[float] = []
    for attenuation, aggregate in aggregates.items():
        checks = {
            "clean_dropout_at_most_1_percent": (
                aggregate["median_clean_dropout_rate"] <= 0.01
            ),
            "clean_lsd_at_most_3_db": (aggregate["median_clean_speech_lsd_db"] <= 3.0),
            "rtf_at_most_0_75": aggregate["max_realtime_factor"] <= 0.75,
            "p99_frame_within_10ms_deadline": (
                aggregate["max_p99_frame_seconds"] <= FRAME_SIZE / SAMPLE_RATE
            ),
            "noisy_si_sdr_within_0_35_db_of_80_db": (
                aggregate["median_noisy_si_sdr_improvement_db"]
                >= baseline["median_noisy_si_sdr_improvement_db"] - 0.35
            ),
            "noisy_lsd_within_0_25_db_of_80_db": (
                aggregate["median_noisy_speech_lsd_db"]
                <= baseline["median_noisy_speech_lsd_db"] + 0.25
            ),
        }
        gates[str(attenuation)] = {
            "metrics": aggregate,
            "checks": checks,
            "passes": all(checks.values()),
        }
        if all(checks.values()):
            passing.append(attenuation)
    selected = min(passing) if passing else 80.0
    return selected, {
        "predefined_gates": {
            "clean_dropout_rate_max": 0.01,
            "clean_speech_lsd_db_max": 3.0,
            "max_realtime_factor": 0.75,
            "p99_frame_seconds_max": FRAME_SIZE / SAMPLE_RATE,
            "noisy_si_sdr_regret_db_max_vs_80_db": 0.35,
            "noisy_speech_lsd_regret_db_max_vs_80_db": 0.25,
        },
        "candidates": gates,
        "selected_attenuation_limit_db": selected,
        "selection_rule": "lowest attenuation limit passing every gate; otherwise retain 80 dB",
    }


def _choose_post_filter(
    cases: list[dict[str, Any]],
    attenuation_db: float,
) -> tuple[float, dict[str, Any]]:
    aggregates = {
        beta: _aggregate(
            [
                case
                for case in cases
                if case["attenuation_limit_db"] == attenuation_db
                and case["post_filter_beta"] == beta
            ]
        )
        for beta in POST_FILTER_CANDIDATES
    }
    baseline = aggregates[0.0]
    candidates: dict[str, Any] = {}
    passing: list[float] = []
    for beta, aggregate in aggregates.items():
        if beta == 0.0:
            checks = {"baseline": True}
        else:
            checks = {
                "noisy_si_sdr_improves_by_0_15_db": (
                    aggregate["median_noisy_si_sdr_improvement_db"]
                    >= baseline["median_noisy_si_sdr_improvement_db"] + 0.15
                ),
                "noisy_lsd_does_not_worsen": (
                    aggregate["median_noisy_speech_lsd_db"]
                    <= baseline["median_noisy_speech_lsd_db"]
                ),
                "clean_lsd_regression_at_most_0_15_db": (
                    aggregate["median_clean_speech_lsd_db"]
                    <= baseline["median_clean_speech_lsd_db"] + 0.15
                ),
                "clean_dropout_regression_at_most_0_2_percent": (
                    aggregate["median_clean_dropout_rate"]
                    <= baseline["median_clean_dropout_rate"] + 0.002
                ),
                "rtf_at_most_0_75": aggregate["max_realtime_factor"] <= 0.75,
                "p99_frame_within_10ms_deadline": (
                    aggregate["max_p99_frame_seconds"] <= FRAME_SIZE / SAMPLE_RATE
                ),
            }
        passed = beta > 0.0 and all(checks.values())
        candidates[str(beta)] = {
            "metrics": aggregate,
            "checks": checks,
            "passes_upgrade_gate": passed,
        }
        if passed:
            passing.append(beta)
    selected = min(passing) if passing else 0.0
    return selected, {
        "predefined_gates": {
            "noisy_si_sdr_improvement_db_min": 0.15,
            "noisy_lsd_regression_db_max": 0.0,
            "clean_lsd_regression_db_max": 0.15,
            "clean_dropout_regression_max": 0.002,
            "max_realtime_factor": 0.75,
            "p99_frame_seconds_max": FRAME_SIZE / SAMPLE_RATE,
        },
        "candidates": candidates,
        "selected_post_filter_beta": selected,
        "selection_rule": "smallest positive beta passing every upgrade gate; otherwise retain 0",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--max-languages", type=int, default=12)
    args = parser.parse_args()

    if not args.binary.is_file():
        raise RuntimeError(
            f"Missing {args.binary}; run cargo build --release -p mic_eq_core "
            "--bin deepfilter_benchmark"
        )
    pairs = _paired_paths(args.corpus_root, args.max_languages)
    clean_stream, noisy_stream, segments = _build_streams(pairs, args.clip_seconds)

    with tempfile.TemporaryDirectory(
        prefix="audioforge-deepfilter-hardening-"
    ) as temp_name:
        temp_root = Path(temp_name)
        _write_raw(temp_root / "clean.f32", clean_stream)
        _write_raw(temp_root / "noisy.f32", noisy_stream)
        cases: list[dict[str, Any]] = []

        for attenuation_db in ATTENUATION_CANDIDATES_DB:
            for model in MODELS:
                cases.append(
                    _evaluate_case(
                        args.binary,
                        temp_root,
                        clean_stream,
                        noisy_stream,
                        segments,
                        model=model,
                        attenuation_db=attenuation_db,
                        post_filter_beta=0.0,
                    )
                )

        selected_attenuation, attenuation_decision = _choose_attenuation(cases)
        for beta in POST_FILTER_CANDIDATES[1:]:
            for model in MODELS:
                cases.append(
                    _evaluate_case(
                        args.binary,
                        temp_root,
                        clean_stream,
                        noisy_stream,
                        segments,
                        model=model,
                        attenuation_db=selected_attenuation,
                        post_filter_beta=beta,
                    )
                )

        selected_beta, post_filter_decision = _choose_post_filter(
            cases, selected_attenuation
        )
        baseline_cases = {
            case["model"]: case
            for case in cases
            if case["attenuation_limit_db"] == 80.0 and case["post_filter_beta"] == 0.0
        }
        mix_contracts = _mix_contract(
            args.binary,
            temp_root,
            noisy_stream,
            baseline_cases,
        )

    manifest = json.loads(
        (REPO_ROOT / "release-assets.json").read_text(encoding="utf-8")
    )
    asset_hashes = {
        asset["path"]: asset["sha256"]
        for asset in manifest["assets"]
        if asset["path"]
        in {
            "df.dll",
            "models/DeepFilterNet3_ll_onnx.tar.gz",
            "models/DeepFilterNet3_onnx.tar.gz",
        }
    }
    source_paths = (
        "python/tools/evaluate_deepfilter_hardening.py",
        "python/mic_eq/analysis/wav_io.py",
        "rust-core/src/bin/deepfilter_benchmark.rs",
        "rust-core/src/dsp/deepfilter_ffi.rs",
    )
    source_hashes = {path: _sha256(REPO_ROOT / path) for path in source_paths}
    report = {
        "schema_version": 2,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Select AudioForge DeepFilter runtime controls and verify LL/Standard "
            "wet-dry alignment; no DPDFNet candidate is evaluated."
        ),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "binary_sha256": _sha256(args.binary),
        },
        "assets": asset_hashes,
        "source_sha256": source_hashes,
        "corpus": {
            "root": _relative(args.corpus_root),
            "pair_count": len(pairs),
            "clip_seconds_per_pair": args.clip_seconds,
            "sample_rate": SAMPLE_RATE,
            "segments": segments,
        },
        "alignment_gate": {
            "required": (
                "measured wet lag equals declared latency, 0% is delayed dry, and "
                "50% equals the samplewise mean of aligned wet and dry"
            ),
            "cases": mix_contracts,
            "passes": all(contract["passes"] for contract in mix_contracts),
        },
        "attenuation_decision": attenuation_decision,
        "post_filter_decision": post_filter_decision,
        "selected_runtime_config": {
            "attenuation_limit_db": selected_attenuation,
            "post_filter_beta": selected_beta,
        },
        "cases": cases,
    }
    selected_cases = [
        case
        for case in cases
        if case["attenuation_limit_db"] == selected_attenuation
        and case["post_filter_beta"] == selected_beta
    ]
    selected_aggregate = _aggregate(selected_cases)
    report["evaluation_contract"] = {
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "frame_size": FRAME_SIZE,
            "strength": 1.0,
            "models": list(MODELS),
            "selected_attenuation_limit_db": selected_attenuation,
            "selected_post_filter_beta": selected_beta,
        },
        "asset_hashes": {
            **asset_hashes,
            _relative(args.binary): _sha256(args.binary),
            _relative(args.corpus_root / "manifest.json"): _sha256(
                args.corpus_root / "manifest.json"
            ),
        },
        "runtime": {
            "max_realtime_factor": selected_aggregate["max_realtime_factor"],
            "max_p99_frame_seconds": selected_aggregate["max_p99_frame_seconds"],
            "max_frame_seconds": selected_aggregate["max_frame_seconds"],
            "frame_deadline_seconds": FRAME_SIZE / SAMPLE_RATE,
        },
        "latency": {
            case["model"]: {
                "declared_samples": case["declared_latency_samples"],
                "measured_samples": case["measured_wet_lag_samples"],
            }
            for case in selected_cases
        },
        "clean_preservation": {
            "median_si_sdr_db": selected_aggregate["median_clean_si_sdr_db"],
            "median_speech_lsd_db": selected_aggregate["median_clean_speech_lsd_db"],
            "median_dropout_rate": selected_aggregate["median_clean_dropout_rate"],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["selected_runtime_config"], sort_keys=True))
    print(f"Wrote {_relative(args.report)}")
    return 0 if report["alignment_gate"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
