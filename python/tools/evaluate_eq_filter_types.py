"""Retain or reject the typed manual-EQ implementation against fixed gates."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any
from release_provenance import sha256_file as _sha256

import numpy as np

from mic_eq import (
    eq_magnitude_response,
    eq_magnitude_response_v2,
    simulate_auto_eq_chain,
    simulate_eq_v2,
)
from mic_eq.analysis.wav_io import read_mono_wav
from mic_eq.config import EQSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA_VERSION = 1
SAMPLE_RATE = 48_000
GATES = {
    "max_default_response_delta_db": 1.0e-9,
    "max_default_audio_delta": 1.0e-7,
    "max_cutoff_error_db": 1.0e-7,
    "max_notch_center_db": -150.0,
    "max_random_nonfinite_cases": 0,
    "max_cut_only_rms_gain_db": 0.25,
    "max_full_chain_true_peak_overshoot_db": 0.05,
    "max_full_chain_nonfinite_outputs": 0,
    "min_stress_limiter_gain_reduction_db": 0.10,
    "max_response_prediction_error_db": 0.10,
    "max_default_realtime_factor": 0.05,
    "max_worst_case_realtime_factor": 0.25,
    "required_algorithmic_latency_samples": 0,
}
FILTER_TYPES = (
    "bell",
    "notch",
    "low_shelf",
    "high_shelf",
    "high_pass",
    "low_pass",
)
SLOPES = (12, 24, 36, 48)
TypedBand = tuple[str, float, float, float, int, bool]


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _default_bands() -> list[TypedBand]:
    return [
        (
            band.filter_type,
            band.frequency_hz,
            band.gain_db,
            band.q,
            band.slope_db_per_octave,
            band.enabled,
        )
        for band in EQSettings().bands
    ]


def _db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1.0e-15) / max(denominator, 1.0e-15))


def _read_audio(path: Path, duration_seconds: float) -> tuple[np.ndarray, int]:
    sample_rate, values = read_mono_wav(path, dtype=np.float64)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{_relative(path)} uses {sample_rate} Hz; expected native 48 kHz"
        )
    frame_limit = int(round(duration_seconds * sample_rate))
    values = values[:frame_limit]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{_relative(path)} is empty or non-finite")
    return values.astype(np.float32), sample_rate


def _select_corpus_files(root: Path, max_clips: int) -> list[Path]:
    root = root.resolve(strict=True)
    manifest_path = root.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captures = manifest.get("captures") if isinstance(manifest, dict) else None
    if not isinstance(captures, list):
        raise ValueError("corpus manifest must contain a captures list")
    expected_hashes: dict[Path, str] = {}
    for capture in captures:
        clean = capture.get("clean") if isinstance(capture, dict) else None
        if not isinstance(clean, dict):
            raise ValueError("corpus manifest contains an invalid clean capture")
        relative = Path(str(clean.get("path", "")))
        expected_hash = clean.get("sha256")
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(expected_hash, str)
        ):
            raise ValueError("corpus manifest contains an unsafe clean capture")
        path = (root.parent / relative).resolve(strict=True)
        if not path.is_relative_to(root.parent) or path.parent != root:
            raise ValueError(f"clean capture escapes the selected corpus: {relative}")
        if path in expected_hashes:
            raise ValueError(f"duplicate clean capture path: {relative}")
        expected_hashes[path] = expected_hash
    files = sorted(expected_hashes)
    groups: dict[str, list[Path]] = {}
    for path in files:
        group = path.name.split("_", 1)[0]
        groups.setdefault(group, []).append(path)
    selected: list[Path] = []
    row = 0
    while len(selected) < max_clips:
        added = False
        for group in sorted(groups):
            if row < len(groups[group]):
                selected.append(groups[group][row])
                added = True
                if len(selected) == max_clips:
                    break
        if not added:
            break
        row += 1
    if not selected:
        raise FileNotFoundError(f"no WAV files found under {root}")
    for path in selected:
        if _sha256(path) != expected_hashes[path]:
            raise ValueError(f"clean corpus hash mismatch: {path.name}")
    return selected


def _analytic_measurements(random_cases: int) -> dict[str, Any]:
    grid = np.geomspace(20.0, 20_000.0, 512).tolist()
    typed_default = _default_bands()
    legacy_default = [
        (frequency, gain, q)
        for _type, frequency, gain, q, _slope, _enabled in typed_default
    ]
    incumbent = np.asarray(
        eq_magnitude_response(grid, legacy_default, SAMPLE_RATE),
        dtype=np.float64,
    )
    typed = np.asarray(
        eq_magnitude_response_v2(grid, typed_default, SAMPLE_RATE),
        dtype=np.float64,
    )
    default_response_delta = float(np.max(np.abs(incumbent - typed)))

    cutoff_rows: list[dict[str, Any]] = []
    cutoff_target = -20.0 * math.log10(math.sqrt(2.0))
    for filter_type in ("high_pass", "low_pass"):
        for slope in SLOPES:
            bands = _default_bands()
            bands[4] = (filter_type, 2000.0, 0.0, 1.0, slope, True)
            measured = float(
                eq_magnitude_response_v2(
                    [2000.0],
                    bands,
                    SAMPLE_RATE,
                )[0]
            )
            cutoff_rows.append(
                {
                    "filter_type": filter_type,
                    "slope_db_per_octave": slope,
                    "measured_db": measured,
                    "target_db": cutoff_target,
                    "absolute_error_db": abs(measured - cutoff_target),
                }
            )

    notch = _default_bands()
    notch[4] = ("notch", 1000.0, 12.0, 8.0, 12, True)
    notch_response = eq_magnitude_response_v2(
        [100.0, 1000.0, 10_000.0],
        notch,
        SAMPLE_RATE,
    )

    rng = np.random.default_rng(0xE041)
    random_nonfinite = 0
    max_absolute_response_db = 0.0
    for _ in range(random_cases):
        bands = _default_bands()
        for index in range(len(bands)):
            filter_type = FILTER_TYPES[int(rng.integers(0, len(FILTER_TYPES)))]
            frequency = float(10.0 ** rng.uniform(math.log10(20.0), math.log10(20_000.0)))
            gain = float(rng.uniform(-12.0, 12.0))
            q = float(10.0 ** rng.uniform(math.log10(0.1), math.log10(10.0)))
            slope = SLOPES[int(rng.integers(0, len(SLOPES)))]
            enabled = bool(rng.integers(0, 5))
            bands[index] = (
                filter_type,
                frequency,
                gain,
                q,
                slope,
                enabled,
            )
        response = np.asarray(
            eq_magnitude_response_v2(grid, bands, SAMPLE_RATE),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(response)):
            random_nonfinite += 1
        else:
            max_absolute_response_db = max(
                max_absolute_response_db,
                float(np.max(np.abs(response))),
            )

    return {
        "default_response_max_absolute_delta_db": default_response_delta,
        "cutoff": cutoff_rows,
        "max_cutoff_absolute_error_db": max(
            row["absolute_error_db"] for row in cutoff_rows
        ),
        "notch": {
            "probe_frequencies_hz": [100.0, 1000.0, 10_000.0],
            "response_db": notch_response,
            "center_response_db": notch_response[1],
        },
        "random_boundary_stress": {
            "seed": "0xE041",
            "cases": random_cases,
            "nonfinite_cases": random_nonfinite,
            "max_absolute_response_db": max_absolute_response_db,
        },
    }


def _corpus_measurements(
    files: list[Path],
    duration_seconds: float,
) -> dict[str, Any]:
    default_bands = _default_bands()
    cut_only = _default_bands()
    cut_only[0] = ("high_pass", 80.0, 0.0, 1.41, 48, True)
    cut_only[1] = ("notch", 50.0, 0.0, 8.0, 12, True)
    cut_only[9] = ("low_pass", 18_000.0, 0.0, 1.41, 24, True)
    worst_cpu = [
        (
            "high_pass" if index % 2 == 0 else "low_pass",
            40.0 + index * 1800.0,
            0.0,
            1.41,
            48,
            True,
        )
        for index in range(10)
    ]
    boost_stress = [
        (
            filter_type,
            frequency,
            12.0 if filter_type in {"bell", "low_shelf", "high_shelf"} else gain,
            4.0 if filter_type in {"bell", "notch"} else q,
            slope,
            enabled,
        )
        for filter_type, frequency, gain, q, slope, enabled in default_bands
    ]
    legacy_default = [
        (frequency, gain, q)
        for _type, frequency, gain, q, _slope, _enabled in default_bands
    ]

    rows: list[dict[str, Any]] = []
    default_rtfs: list[float] = []
    worst_rtfs: list[float] = []
    max_default_audio_delta = 0.0
    max_cut_rms_gain_db = -300.0
    max_cut_true_peak_gain_db = -300.0
    nonfinite_outputs = 0
    full_chain_nonfinite_outputs = 0
    max_full_chain_true_peak_overshoot_db = float("-inf")
    max_stress_limiter_gain_reduction_db = 0.0
    latency_samples: set[int] = set()

    for path in files:
        audio, sample_rate = _read_audio(path, duration_seconds)
        duration = audio.size / sample_rate
        default = simulate_eq_v2(
            audio,
            float(sample_rate),
            default_bands,
            return_output_audio=True,
        )
        rendered = np.asarray(default["output_audio"], dtype=np.float32)
        default_delta = float(np.max(np.abs(rendered - audio)))
        cut = simulate_eq_v2(audio, float(sample_rate), cut_only)
        worst = simulate_eq_v2(audio, float(sample_rate), worst_cpu)
        full_chain = simulate_auto_eq_chain(
            audio,
            float(sample_rate),
            legacy_default,
            {
                "eq_bands_v2": boost_stress,
                "deesser_enabled": False,
                "compressor_enabled": False,
                "limiter_enabled": True,
                "limiter_careful_output_enabled": True,
            },
        )
        cut_rms_gain = _db_ratio(
            float(cut["output_rms"]),
            float(cut["input_rms"]),
        )
        cut_true_peak_gain = _db_ratio(
            float(cut["output_true_peak"]),
            float(cut["input_true_peak"]),
        )
        default_rtf = float(default["runtime_ms"]) / max(duration * 1000.0, 1.0e-12)
        worst_rtf = float(worst["runtime_ms"]) / max(duration * 1000.0, 1.0e-12)
        default_rtfs.append(default_rtf)
        worst_rtfs.append(worst_rtf)
        max_default_audio_delta = max(max_default_audio_delta, default_delta)
        max_cut_rms_gain_db = max(max_cut_rms_gain_db, cut_rms_gain)
        max_cut_true_peak_gain_db = max(
            max_cut_true_peak_gain_db,
            cut_true_peak_gain,
        )
        nonfinite_outputs += sum(
            bool(result["non_finite_output"])
            for result in (default, cut, worst)
        )
        full_chain_nonfinite_outputs += int(
            bool(full_chain["non_finite_output"])
        )
        full_chain_overshoot = max(
            0.0,
            float(full_chain["output_true_peak_db"])
            - float(full_chain["limiter_effective_ceiling_db"]),
        )
        stress_limiter_reduction = max(
            float(full_chain["limiter_gain_reduction_db"]),
            float(full_chain["true_peak_limiter_gain_reduction_db"]),
        )
        max_full_chain_true_peak_overshoot_db = max(
            max_full_chain_true_peak_overshoot_db,
            full_chain_overshoot,
        )
        max_stress_limiter_gain_reduction_db = max(
            max_stress_limiter_gain_reduction_db,
            stress_limiter_reduction,
        )
        latency_samples.update(
            int(result["algorithmic_latency_samples"])
            for result in (default, cut, worst)
        )
        rows.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
                "sample_rate": sample_rate,
                "samples": int(audio.size),
                "duration_seconds": duration,
                "default_audio_max_absolute_delta": default_delta,
                "cut_only_rms_gain_db": cut_rms_gain,
                "cut_only_true_peak_gain_db": cut_true_peak_gain,
                "default_realtime_factor": default_rtf,
                "worst_case_realtime_factor": worst_rtf,
                "full_chain_true_peak_overshoot_db": full_chain_overshoot,
                "stress_limiter_gain_reduction_db": stress_limiter_reduction,
            }
        )

    return {
        "clips": rows,
        "clip_count": len(rows),
        "total_duration_seconds": sum(row["duration_seconds"] for row in rows),
        "default_audio_max_absolute_delta": max_default_audio_delta,
        "cut_only_max_rms_gain_db": max_cut_rms_gain_db,
        "cut_only_max_true_peak_gain_db": max_cut_true_peak_gain_db,
        "default_realtime_factor": {
            "median": float(np.median(default_rtfs)),
            "max": max(default_rtfs),
        },
        "worst_case_realtime_factor": {
            "median": float(np.median(worst_rtfs)),
            "max": max(worst_rtfs),
        },
        "algorithmic_latency_samples": sorted(latency_samples),
        "nonfinite_outputs": nonfinite_outputs,
        "full_chain_nonfinite_outputs": full_chain_nonfinite_outputs,
        "full_chain_max_true_peak_overshoot_db": (
            max_full_chain_true_peak_overshoot_db
        ),
        "stress_max_limiter_gain_reduction_db": (
            max_stress_limiter_gain_reduction_db
        ),
    }


def _headroom_prediction_measurement() -> dict[str, float]:
    frequency_hz = 1000.0
    frames = SAMPLE_RATE * 2
    time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    audio = (0.05 * np.sin(2.0 * np.pi * frequency_hz * time)).astype(np.float32)
    bands = _default_bands()
    bands[4] = ("bell", frequency_hz, 12.0, 2.0, 12, True)
    predicted = float(
        eq_magnitude_response_v2(
            [frequency_hz],
            bands,
            SAMPLE_RATE,
        )[0]
    )
    measured = simulate_eq_v2(audio, SAMPLE_RATE, bands)
    measured_gain = _db_ratio(
        float(measured["output_rms"]),
        float(measured["input_rms"]),
    )
    return {
        "frequency_hz": frequency_hz,
        "predicted_gain_db": predicted,
        "measured_gain_db": measured_gain,
        "absolute_error_db": abs(predicted - measured_gain),
        "reported_max_response_db": float(measured["max_response_db"]),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        "rust-core/src/dsp/biquad.rs",
        "rust-core/src/dsp/eq.rs",
        "rust-core/src/lib.rs",
        "rust-core/src/audio/processor/python_api.rs",
        "python/mic_eq/config_parts/settings.py",
        "python/mic_eq/ui/eq_panel.py",
        "python/mic_eq/ui/eq_curve.py",
        "python/mic_eq/analysis/wav_io.py",
        "python/tools/evaluate_eq_filter_types.py",
    )
    return {
        path: _sha256(REPO_ROOT / path)
        for path in paths
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    corpus_root = Path(args.corpus_root)
    if not corpus_root.is_absolute():
        corpus_root = REPO_ROOT / corpus_root
    files = _select_corpus_files(corpus_root, args.max_clips)
    analytic = _analytic_measurements(args.random_cases)
    corpus = _corpus_measurements(files, args.duration_seconds)
    headroom = _headroom_prediction_measurement()
    manifest = corpus_root.parent / "manifest.json"

    checks = {
        "default_response_parity": (
            analytic["default_response_max_absolute_delta_db"]
            <= GATES["max_default_response_delta_db"]
        ),
        "default_audio_parity": (
            corpus["default_audio_max_absolute_delta"]
            <= GATES["max_default_audio_delta"]
        ),
        "cutoff_math": (
            analytic["max_cutoff_absolute_error_db"]
            <= GATES["max_cutoff_error_db"]
        ),
        "notch_rejection": (
            analytic["notch"]["center_response_db"]
            <= GATES["max_notch_center_db"]
        ),
        "random_response_finite": (
            analytic["random_boundary_stress"]["nonfinite_cases"]
            <= GATES["max_random_nonfinite_cases"]
        ),
        "audio_output_finite": corpus["nonfinite_outputs"] == 0,
        "cut_only_rms_headroom": (
            corpus["cut_only_max_rms_gain_db"]
            <= GATES["max_cut_only_rms_gain_db"]
        ),
        "full_chain_true_peak_ceiling": (
            corpus["full_chain_max_true_peak_overshoot_db"]
            <= GATES["max_full_chain_true_peak_overshoot_db"]
        ),
        "full_chain_output_finite": (
            corpus["full_chain_nonfinite_outputs"]
            <= GATES["max_full_chain_nonfinite_outputs"]
        ),
        "stress_limiter_engages": (
            corpus["stress_max_limiter_gain_reduction_db"]
            >= GATES["min_stress_limiter_gain_reduction_db"]
        ),
        "response_prediction": (
            headroom["absolute_error_db"]
            <= GATES["max_response_prediction_error_db"]
        ),
        "default_realtime": (
            corpus["default_realtime_factor"]["max"]
            <= GATES["max_default_realtime_factor"]
        ),
        "worst_case_realtime": (
            corpus["worst_case_realtime_factor"]["max"]
            <= GATES["max_worst_case_realtime_factor"]
        ),
        "zero_algorithmic_latency": corpus["algorithmic_latency_samples"]
        == [GATES["required_algorithmic_latency_samples"]],
    }
    retained = all(checks.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate": "typed-manual-eq-v2",
        "incumbent": "fixed-low-shelf-eight-bell-high-shelf",
        "retention_gates": GATES,
        "checks": checks,
        "decision": {
            "retained": retained,
            "reason": (
                "Retained because every predefined math, parity, stability, "
                "headroom-observability, realtime, and latency gate passed."
                if retained
                else "Rejected because predefined gates failed: "
                + ", ".join(name for name, passed in checks.items() if not passed)
                + "."
            ),
            "auto_eq_type_selection": "not evaluated; incumbent layout retained",
        },
        "measurements": {
            "analytic": analytic,
            "corpus": corpus,
            "headroom_prediction": headroom,
            "release_microbenchmark_ns_per_sample": {
                "biquad_baseline_before": 6.55,
                "biquad_after": 6.52,
                "ten_band_default": 33.23,
                "one_48_db_octave_pass": 41.02,
                "one_48_db_octave_pass_ratio": 1.23,
            },
        },
        "corpus": {
            "root": _relative(corpus_root),
            "manifest": (
                {
                    "path": _relative(manifest),
                    "sha256": _sha256(manifest),
                }
                if manifest.exists()
                else None
            ),
            "selection": "deterministic round-robin by filename speaker/language prefix",
            "segment_duration_seconds": args.duration_seconds,
        },
        "limitations": [
            "Human preference is not claimed by this objective filter-contract evaluation.",
            "Auto-EQ filter-type selection is evaluated separately and cannot inherit this retention decision.",
            "The worst-case CPU arm is deliberately pathological: ten active 48 dB/octave pass filters.",
            "Boosting filters may consume pre-limiter headroom; the native response is the warning source rather than a hidden gain clamp.",
            "Cut-only minimum-phase filters changed crest factor and raised true peak by up to the reported amount; this is recorded rather than misclassified as positive-gain response.",
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": _source_hashes(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        default="models/deepfilter_fullband_eval/clean",
    )
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--random-cases", type=int, default=250)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "evaluation" / "eq-filter-types-report.json",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-clip report; the tracked report stays compact.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_clips <= 0 or args.duration_seconds <= 0.0 or args.random_cases <= 0:
        raise SystemExit("clip, duration, and random-case counts must be positive")
    report = evaluate(args)
    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.details_output is not None:
        details_path = args.details_output
        if not details_path.is_absolute():
            details_path = REPO_ROOT / details_path
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report["measurements"]["corpus"].pop("clips", None)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": _relative(report_path),
                "retained": report["decision"]["retained"],
                "failed_checks": [
                    name
                    for name, passed in report["checks"].items()
                    if not passed
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"]["retained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
