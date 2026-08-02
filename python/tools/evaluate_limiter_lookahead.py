"""Compare limiter lookahead through the exact protected product chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mic_eq.analysis.wav_io import read_mono_wav

from mic_eq import simulate_auto_eq_chain


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "limiter-lookahead-report.json"
DEFAULT_REAL_MANIFEST = REPO_ROOT / "models/cross_take_eval/manifest.json"
SAMPLE_RATE = 48_000
LOOKAHEAD_MS = (0.5, 1.0, 2.0)
BASELINE_LOOKAHEAD_MS = 2.0
REAL_CASE_COUNT = 12
RUNTIME_REPETITIONS = 7
REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB = 3.0
REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB = 0.10
MIN_MATERIAL_LOOKAHEAD_REDUCTION_MS = 1.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cases() -> dict[str, np.ndarray]:
    """Return controlled finite fixtures retained for regression tests."""
    sample_count = SAMPLE_RATE * 4
    time = np.arange(sample_count) / SAMPLE_RATE
    sine_bursts = np.zeros(sample_count, dtype=np.float64)
    for start_s in np.arange(0.25, 3.75, 0.19):
        start = int(start_s * SAMPLE_RATE)
        length = int(0.025 * SAMPLE_RATE)
        envelope = np.hanning(length)
        sine_bursts[start : start + length] += (
            1.35
            * envelope
            * np.sin(2.0 * np.pi * 6_500.0 * time[start : start + length])
        )
    impulses = np.zeros(sample_count, dtype=np.float64)
    impulses[::997] = 1.45
    impulses[499::1553] = -1.35
    clipped_voice = (
        0.72 * np.sin(2.0 * np.pi * 180.0 * time)
        + 0.46 * np.sin(2.0 * np.pi * 2_300.0 * time)
    )
    clipped_voice *= 0.45 + 0.55 * np.sin(2.0 * np.pi * 2.1 * time) ** 2
    clipped_voice = np.clip(clipped_voice * 1.25, -1.0, 1.0)
    return {
        "controlled-sine-bursts": np.asarray(sine_bursts, dtype=np.float32),
        "controlled-impulses": np.asarray(impulses, dtype=np.float32),
        "controlled-clipped-voice": np.asarray(clipped_voice, dtype=np.float32),
    }


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    return read_mono_wav(path, allow_stereo=False, dtype=np.float32)


def _real_cases(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pairs"), list):
        raise ValueError("real transient manifest has no pair list")
    candidates = [
        pair
        for pair in manifest["pairs"]
        if pair.get("delivery") in {"neutral-normal", "calm-strong"}
    ]
    candidates.sort(
        key=lambda pair: (
            str(pair["speaker"]),
            str(pair["statement_id"]),
            str(pair["delivery"]),
        )
    )
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for pair in candidates:
        by_speaker.setdefault(str(pair["speaker"]), []).append(pair)
    split_lookup = {
        str(speaker): str(split)
        for split, speakers in manifest["speaker_disjoint_splits"].items()
        for speaker in speakers
    }
    speakers = sorted(by_speaker)
    stride = max(1, len(speakers) // REAL_CASE_COUNT)
    selected_speakers = speakers[::stride][:REAL_CASE_COUNT]
    rows: list[dict[str, Any]] = []
    for rank, speaker in enumerate(selected_speakers):
        options = by_speaker[speaker]
        pair = options[rank % len(options)]
        take_id = "01" if rank % 2 == 0 else "02"
        take = pair["takes"][take_id]
        path = manifest_path.parent / take["path"]
        if not path.is_file() or _sha256(path) != take["sha256"]:
            raise ValueError(f"real transient source hash mismatch: {pair['id']}")
        sample_rate, audio = _read_mono(path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{path.name} is not native 48 kHz")
        if float(np.max(np.abs(audio))) <= 1e-9:
            raise ValueError(f"{path.name} is silent")
        rows.append(
            {
                "id": f"real-{pair['id']}-take-{take_id}",
                "kind": "real_speech",
                "audio": audio,
                "provenance": {
                    "speaker_id": speaker,
                    "speaker_sex": pair.get("speaker_sex", "unknown"),
                    "split": split_lookup[speaker],
                    "statement_id": pair["statement_id"],
                    "delivery": pair["delivery"],
                    "source_sha256": take["sha256"],
                    "rendered_wav_sha256": _sha256(path),
                    "sample_rate": sample_rate,
                    "frames": int(audio.size),
                },
            }
        )
    if len(rows) != REAL_CASE_COUNT:
        raise RuntimeError(f"expected {REAL_CASE_COUNT} real cases, got {len(rows)}")
    return rows, {
        "dataset": manifest["dataset"],
        "dataset_page": manifest["dataset_page"],
        "license": manifest["license"],
        "redistribution": manifest["redistribution"],
        "manifest_sha256": _sha256(manifest_path),
        "selection": (
            "Twelve unique speakers across the frozen split, alternating fixed "
            "statements/deliveries and source takes before metrics are viewed."
        ),
    }


def _settings(lookahead_ms: float) -> dict[str, Any]:
    return {
        "deesser_enabled": False,
        "compressor_enabled": True,
        "compressor_threshold_db": -20.0,
        "compressor_ratio": 4.0,
        "compressor_attack_ms": 10.0,
        "compressor_release_ms": 200.0,
        "compressor_makeup_gain_db": 0.0,
        "compressor_adaptive_release": False,
        "compressor_auto_makeup_enabled": False,
        "compressor_sidechain_highpass_enabled": True,
        "limiter_enabled": True,
        "limiter_ceiling_db": -0.5,
        "limiter_release_ms": 50.0,
        "limiter_careful_output_enabled": True,
        "limiter_lookahead_ms": lookahead_ms,
        "return_output_audio": True,
    }


def _render(audio: np.ndarray, lookahead_ms: float) -> dict[str, Any]:
    bands = [(80.0 * 1.75**index, 0.0, 1.0) for index in range(10)]
    return dict(
        simulate_auto_eq_chain(
            np.ascontiguousarray(audio, dtype=np.float32),
            SAMPLE_RATE,
            bands,
            _settings(lookahead_ms),
        )
    )


def _calibrate_real_case(
    audio: np.ndarray,
    provenance: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one static gain so the protected chain actually exercises the limiter."""

    low_db = -24.0
    high_db = 72.0
    for _ in range(16):
        gain_db = 0.5 * (low_db + high_db)
        scaled = np.asarray(audio * 10.0 ** (gain_db / 20.0), dtype=np.float32)
        result = _render(scaled, BASELINE_LOOKAHEAD_MS)
        reduction = float(result["limiter_gain_reduction_db"])
        if reduction < REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB:
            low_db = gain_db
        else:
            high_db = gain_db
    gain_db = 0.5 * (low_db + high_db)
    calibrated = np.asarray(audio * 10.0 ** (gain_db / 20.0), dtype=np.float32)
    result = _render(calibrated, BASELINE_LOOKAHEAD_MS)
    reduction = float(result["limiter_gain_reduction_db"])
    if (
        abs(reduction - REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB)
        > REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB
    ):
        raise RuntimeError(
            f"real-speech limiter calibration reached {reduction:.3f} dB, "
            f"expected {REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB:.3f} dB"
        )
    return calibrated, dict(provenance) | {
        "static_pre_gain_db": gain_db,
        "target_main_limiter_gain_reduction_db": (
            REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB
        ),
        "measured_main_limiter_gain_reduction_db": reduction,
    }


def _gain_envelope_variation_db(
    reference: np.ndarray,
    aligned: np.ndarray,
) -> float:
    frame_samples = int(round(0.002 * SAMPLE_RATE))
    hop_samples = frame_samples // 2
    gains_db: list[float] = []
    for start in range(0, reference.size - frame_samples + 1, hop_samples):
        ref = np.asarray(reference[start : start + frame_samples], dtype=np.float64)
        out = np.asarray(aligned[start : start + frame_samples], dtype=np.float64)
        reference_rms = float(np.sqrt(np.mean(np.square(ref))))
        if reference_rms < 10.0 ** (-40.0 / 20.0):
            continue
        output_rms = float(np.sqrt(np.mean(np.square(out))))
        gains_db.append(
            20.0
            * math.log10(max(output_rms, 1e-12) / max(reference_rms, 1e-12))
        )
    if not gains_db:
        return 0.0
    values = np.asarray(gains_db, dtype=np.float64)
    return float(np.std(values - np.median(values)))


def _transient_indices(audio: np.ndarray, limit: int = 16) -> np.ndarray:
    derivative = np.abs(np.diff(np.asarray(audio, dtype=np.float64), prepend=0.0))
    order = np.argsort(-derivative, kind="stable")
    separation = int(round(0.006 * SAMPLE_RATE))
    selected: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        if all(abs(index - existing) >= separation for existing in selected):
            selected.append(index)
        if len(selected) == limit:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _transient_error_db(
    reference: np.ndarray, aligned: np.ndarray, indices: np.ndarray
) -> float:
    radius = int(round(0.004 * SAMPLE_RATE))
    errors: list[float] = []
    for index in indices:
        start = max(0, int(index) - radius)
        end = min(reference.size, int(index) + radius + 1)
        ref = np.asarray(reference[start:end], dtype=np.float64)
        out = np.asarray(aligned[start:end], dtype=np.float64)
        if ref.size < 8:
            continue
        # Remove one static local gain before measuring shape distortion.  The
        # limiter is expected to change level; this metric isolates transient
        # envelope/waveform deformation from that intended gain reduction.
        denominator = float(np.dot(ref, ref))
        scale = float(np.dot(ref, out) / max(denominator, 1e-12))
        error = out - scale * ref
        errors.append(
            20.0
            * math.log10(
                max(float(np.sqrt(np.mean(np.square(error)))), 1e-12)
                / max(float(np.sqrt(np.mean(np.square(out)))), 1e-12)
            )
        )
    return float(np.median(errors)) if errors else -240.0


def _case(
    case_id: str,
    audio: np.ndarray,
    lookahead_ms: float,
    *,
    kind: str = "controlled",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _render(audio, lookahead_ms)
    results = [_render(audio, lookahead_ms) for _ in range(RUNTIME_REPETITIONS)]
    result = results[-1]
    output = np.asarray(result.pop("output_audio"), dtype=np.float64)
    delay = int(round(lookahead_ms / 1000.0 * SAMPLE_RATE)) + 20
    aligned = output[delay:]
    reference = audio[: aligned.size].astype(np.float64)
    gain_variation = _gain_envelope_variation_db(reference, aligned)
    ceiling_db = float(result["limiter_effective_ceiling_db"])
    transient_indices = _transient_indices(reference)
    runtime_values = [float(item["candidate_runtime_ms"]) for item in results]
    return {
        "id": case_id,
        "kind": kind,
        "lookahead_ms": lookahead_ms,
        "pre_true_peak_overshoot_db": max(
            0.0, float(result["pre_limiter_true_peak_db"]) - ceiling_db
        ),
        "output_true_peak_overshoot_db": max(
            0.0, float(result["output_true_peak_db"]) - ceiling_db
        ),
        "main_peak_gain_reduction_db": float(result["limiter_gain_reduction_db"]),
        "true_peak_limiter_gain_reduction_db": float(
            result["true_peak_limiter_gain_reduction_db"]
        ),
        "true_peak_limited_events": int(result["true_peak_limited_events"]),
        "gain_envelope_variation_db": gain_variation,
        "transient_shape_error_db": _transient_error_db(
            reference, aligned, transient_indices
        ),
        "transient_count": int(transient_indices.size),
        "runtime_ms_median": float(np.median(runtime_values)),
        "runtime_ms_p95": float(np.percentile(runtime_values, 95.0)),
        "runtime_realtime_factor_median": float(
            np.median(runtime_values) / max(1.0, audio.size / SAMPLE_RATE * 1000.0)
        ),
        "finite_output": bool(
            not result["non_finite_output"] and np.all(np.isfinite(output))
        ),
        "processed_samples": int(result["processed_samples"]),
        "declared_alignment_samples": delay,
        "provenance": dict(provenance or {}),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "worst_pre_true_peak_overshoot_db": max(
            float(row["pre_true_peak_overshoot_db"]) for row in rows
        ),
        "worst_output_true_peak_overshoot_db": max(
            float(row["output_true_peak_overshoot_db"]) for row in rows
        ),
        "max_true_peak_limiter_gain_reduction_db": max(
            float(row["true_peak_limiter_gain_reduction_db"]) for row in rows
        ),
        "total_true_peak_limited_events": sum(
            int(row["true_peak_limited_events"]) for row in rows
        ),
        "median_gain_envelope_variation_db": float(
            np.median([row["gain_envelope_variation_db"] for row in rows])
        ),
        "minimum_pre_true_peak_overshoot_db": min(
            float(row["pre_true_peak_overshoot_db"]) for row in rows
        ),
        "minimum_main_peak_gain_reduction_db": min(
            float(row["main_peak_gain_reduction_db"]) for row in rows
        ),
        "median_transient_shape_error_db": float(
            np.median([row["transient_shape_error_db"] for row in rows])
        ),
        "p90_transient_shape_error_db": float(
            np.percentile([row["transient_shape_error_db"] for row in rows], 90.0)
        ),
        "median_runtime_ms": float(
            np.median([row["runtime_ms_median"] for row in rows])
        ),
        "p95_runtime_ms": float(np.percentile([row["runtime_ms_p95"] for row in rows], 95.0)),
        "p95_runtime_realtime_factor": float(
            np.percentile(
                [row["runtime_realtime_factor_median"] for row in rows], 95.0
            )
        ),
        "all_finite": all(bool(row["finite_output"]) for row in rows),
    }


def evaluate(real_manifest: Path) -> dict[str, Any]:
    real_rows, real_corpus = _real_cases(real_manifest)
    controlled_inputs = [
        {
            "id": case_id,
            "kind": "controlled",
            "audio": audio,
            "provenance": {
                "generator": "evaluate_limiter_lookahead.py",
                "generator_sha256": _sha256(Path(__file__).resolve()),
                "sample_rate": SAMPLE_RATE,
                "frames": int(audio.size),
            },
        }
        for case_id, audio in _cases().items()
    ]
    calibrated_real_inputs = []
    for case in real_rows:
        audio, provenance = _calibrate_real_case(
            np.asarray(case["audio"], dtype=np.float32),
            case["provenance"],
        )
        calibrated_real_inputs.append(dict(case, audio=audio, provenance=provenance))
    case_inputs = controlled_inputs + calibrated_real_inputs
    rows = [
        _case(
            str(case["id"]),
            np.asarray(case["audio"], dtype=np.float32),
            lookahead_ms,
            kind=str(case["kind"]),
            provenance=case["provenance"],
        )
        for case in case_inputs
        for lookahead_ms in LOOKAHEAD_MS
    ]
    aggregates: dict[str, dict[str, Any]] = {}
    for lookahead_ms in LOOKAHEAD_MS:
        subset = [row for row in rows if row["lookahead_ms"] == lookahead_ms]
        aggregates[str(lookahead_ms)] = {
            "all": _aggregate(subset),
            "controlled": _aggregate(
                [row for row in subset if row["kind"] == "controlled"]
            ),
            "real_speech": _aggregate(
                [row for row in subset if row["kind"] == "real_speech"]
            ),
        }
    baseline = aggregates[str(BASELINE_LOOKAHEAD_MS)]
    candidate_checks: dict[str, dict[str, Any]] = {}
    for lookahead_ms in (0.5, 1.0):
        aggregate = aggregates[str(lookahead_ms)]
        checks = {
            "pre_true_peak_overshoot_regression_at_most_0_10_db": (
                aggregate["all"]["worst_pre_true_peak_overshoot_db"]
                <= baseline["all"]["worst_pre_true_peak_overshoot_db"] + 0.10
            ),
            "output_true_peak_overshoot_at_most_0_01_db": (
                aggregate["all"]["worst_output_true_peak_overshoot_db"] <= 0.01
            ),
            "downstream_true_peak_gr_regression_at_most_0_10_db": (
                aggregate["all"]["max_true_peak_limiter_gain_reduction_db"]
                <= baseline["all"]["max_true_peak_limiter_gain_reduction_db"] + 0.10
            ),
            "downstream_true_peak_event_regression_at_most_1": (
                aggregate["all"]["total_true_peak_limited_events"]
                <= baseline["all"]["total_true_peak_limited_events"] + 1
            ),
            "gain_envelope_variation_regression_at_most_0_25_db": (
                aggregate["all"]["median_gain_envelope_variation_db"]
                <= baseline["all"]["median_gain_envelope_variation_db"] + 0.25
            ),
            "real_speech_exercises_main_limiter_within_0_10_db": (
                aggregate["real_speech"]["minimum_main_peak_gain_reduction_db"]
                >= REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB
                - REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB
            ),
            "real_transient_shape_p90_regression_at_most_1_db": (
                aggregate["real_speech"]["p90_transient_shape_error_db"]
                <= baseline["real_speech"]["p90_transient_shape_error_db"] + 1.0
            ),
            "median_runtime_ratio_at_most_1_15": (
                aggregate["all"]["median_runtime_ms"]
                <= baseline["all"]["median_runtime_ms"] * 1.15
            ),
            "p95_realtime_factor_at_most_0_25": (
                aggregate["all"]["p95_runtime_realtime_factor"] <= 0.25
            ),
            "material_latency_reduction_at_least_1_5_ms": (
                BASELINE_LOOKAHEAD_MS - lookahead_ms
                >= MIN_MATERIAL_LOOKAHEAD_REDUCTION_MS
            ),
            "finite_output": bool(aggregate["all"]["all_finite"]),
        }
        candidate_checks[str(lookahead_ms)] = {
            "checks": checks,
            "objective_passes": all(checks.values()),
        }
    passing = [
        value
        for value in (0.5, 1.0)
        if candidate_checks[str(value)]["objective_passes"]
    ]
    objective_candidate = min(passing) if passing else None
    selected = objective_candidate or BASELINE_LOOKAHEAD_MS
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "python/mic_eq/analysis/wav_io.py",
        REPO_ROOT / "rust-core/src/audio/processor/python_api.rs",
        REPO_ROOT / "rust-core/src/dsp/limiter.rs",
        REPO_ROOT / "rust-core/src/dsp/true_peak.rs",
    )
    source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in source_paths
    }
    return {
        "schema_version": 4,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "lookahead_ms": list(LOOKAHEAD_MS),
            "baseline_lookahead_ms": BASELINE_LOOKAHEAD_MS,
            "runtime_repetitions": RUNTIME_REPETITIONS,
            "chain": _settings(BASELINE_LOOKAHEAD_MS)
            | {"limiter_lookahead_ms": "varied", "return_output_audio": True},
            "alignment": "lookahead samples plus 20-sample true-peak stage delay",
        },
        "corpus": {
            "controlled": {
                "case_count": len(_cases()),
                "fixtures": sorted(_cases()),
                "generator_sha256": _sha256(Path(__file__).resolve()),
            },
            "real_speech": real_corpus | {"case_count": len(real_rows)},
        },
        "predefined_gates": {
            "pre_true_peak_overshoot_regression_db_max": 0.10,
            "output_true_peak_overshoot_db_max": 0.01,
            "downstream_true_peak_gr_regression_db_max": 0.10,
            "downstream_true_peak_event_regression_max": 1,
            "gain_envelope_variation_regression_db_max": 0.25,
            "real_main_limiter_gain_reduction_target_db": (
                REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB
            ),
            "real_main_limiter_gain_reduction_tolerance_db": (
                REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB
            ),
            "real_transient_shape_p90_regression_db_max": 1.0,
            "median_runtime_ratio_max": 1.15,
            "p95_realtime_factor_max": 0.25,
            "minimum_material_lookahead_reduction_ms": (
                MIN_MATERIAL_LOOKAHEAD_REDUCTION_MS
            ),
            "finite_output_required": True,
        },
        "aggregates": aggregates,
        "candidate_checks": candidate_checks,
        "objective_candidate_lookahead_ms": objective_candidate,
        "selected_lookahead_ms": selected,
        "decision": (
            "retain_2ms"
            if selected == BASELINE_LOOKAHEAD_MS
            else f"adopt_{selected:g}ms"
        ),
        "cases": rows,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "evaluation_contract": {
            "configuration": {
                "sample_rate": SAMPLE_RATE,
                "lookahead_ms": list(LOOKAHEAD_MS),
                "baseline_lookahead_ms": BASELINE_LOOKAHEAD_MS,
                "runtime_repetitions": RUNTIME_REPETITIONS,
                "chain": _settings(BASELINE_LOOKAHEAD_MS)
                | {"limiter_lookahead_ms": "varied"},
            },
            "asset_hashes": {
                "source": source_hashes,
                "real_manifest": real_corpus.get("manifest_sha256"),
                "real_case_source_sha256": sorted(
                    {
                        str(case["provenance"]["source_sha256"])
                        for case in real_rows
                    }
                ),
            },
            "runtime": {
                "candidate_p95_realtime_factor": max(
                    float(aggregates[str(value)]["all"]["p95_runtime_realtime_factor"])
                    for value in (0.5, 1.0)
                ),
                "max_p99_frame_seconds": None,
                "max_p99_frame_seconds_reason": (
                    "The native simulator reports whole-render timing; seven timed "
                    "renders plus warmup are recorded instead of callback P99."
                ),
                "platform": platform.platform(),
            },
            "latency": {
                "lookahead_samples": {
                    str(value): int(round(value * SAMPLE_RATE / 1000.0))
                    for value in LOOKAHEAD_MS
                },
                "downstream_true_peak_delay_samples": 20,
            },
            "clean_preservation": {
                "all_outputs_finite": all(
                    bool(value["all"]["all_finite"])
                    for value in aggregates.values()
                ),
                "candidate_checks": candidate_checks,
                "incumbent_real_transient_shape_p90_db": baseline["real_speech"][
                    "p90_transient_shape_error_db"
                ],
            },
        },
        "source_sha256": source_hashes,
        "limitations": [
            "Real speech receives one recorded static gain, calibrated through the protected chain to 3 dB of main-limiter gain reduction; the source waveform is otherwise unchanged.",
            "Transient shape error removes one local static gain and is not a perceptual score.",
            "A shorter candidate must pass every safety/quality gate and save at least 1.5 ms; a smaller latency change is not material against the measured product path.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--real-manifest", type=Path, default=DEFAULT_REAL_MANIFEST)
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-case report; the tracked report stays compact.",
    )
    args = parser.parse_args(argv)
    report = evaluate(args.real_manifest)
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        args.details_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report.pop("cases", None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "objective_candidate_lookahead_ms": report[
                    "objective_candidate_lookahead_ms"
                ],
                "selected_lookahead_ms": report["selected_lookahead_ms"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
