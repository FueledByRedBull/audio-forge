"""Evaluate gate/suppressor and de-esser/EQ ordering without changing defaults."""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from mic_eq import (
    analyze_vad_probabilities,
    simulate_auto_eq_chain,
    simulate_gate_suppressor_order,
)
from mic_eq.analysis.deesser_corpus import CORPUS_CASES, generate_deesser_case


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "models" / "dpdfnet_eval_subset"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "processing-order-report.json"
SAMPLE_RATE = 48_000
FRAME_SIZE = 480


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


def _control_probabilities(probabilities: np.ndarray, sample_count: int) -> np.ndarray:
    count = (sample_count + FRAME_SIZE - 1) // FRAME_SIZE
    if probabilities.size == 0:
        return np.zeros(count, dtype=np.float32)
    duration = sample_count / SAMPLE_RATE
    source_times = (np.arange(probabilities.size) + 0.5) * duration / probabilities.size
    target_times = (np.arange(count) + 0.5) * FRAME_SIZE / SAMPLE_RATE
    return np.asarray(
        np.interp(
            target_times,
            source_times,
            probabilities,
            left=float(probabilities[0]),
            right=float(probabilities[-1]),
        ),
        dtype=np.float32,
    )


def _block_rms(audio: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.sqrt(
                np.mean(np.square(audio[start : start + FRAME_SIZE], dtype=np.float64))
            )
            for start in range(0, audio.size, FRAME_SIZE)
        ],
        dtype=np.float64,
    )


def _pumping(trace: np.ndarray) -> float:
    if trace.size < 10:
        return 0.0
    spectrum = np.fft.rfft((trace - np.mean(trace)) * np.hanning(trace.size))
    frequencies = np.fft.rfftfreq(trace.size, FRAME_SIZE / SAMPLE_RATE)
    band = (frequencies >= 2.0) & (frequencies <= 8.0)
    total = float(np.sum(np.square(np.abs(spectrum))))
    return (
        float(np.sqrt(np.sum(np.square(np.abs(spectrum[band]))) / total))
        if total > 1e-12
        else 0.0
    )


def _paired_paths(root: Path) -> list[tuple[Path, Path]]:
    selected: dict[str, tuple[Path, Path]] = {}
    for clean in sorted((root / "Clean").glob("*_clean.wav")):
        language = clean.name.split("_", 1)[0]
        noisy = root / "Noisy" / clean.name.replace("_clean.wav", "_noisy.wav")
        if language not in selected and noisy.is_file():
            selected[language] = (clean, noisy)
    return list(selected.values())


def _gate_case(
    clean_path: Path, noisy_path: Path, clip_seconds: float
) -> dict[str, Any]:
    clean_rate, clean = _read_mono(clean_path)
    noisy_rate, noisy = _read_mono(noisy_path)
    clean = _resample(clean, clean_rate)
    noisy = _resample(noisy, noisy_rate)
    clip_samples = int(round(clip_seconds * SAMPLE_RATE))
    count = min(clean.size, noisy.size)
    offset = min(15 * SAMPLE_RATE, max(0, (count - clip_samples) // 2))
    clean = clean[offset : offset + clip_samples]
    noisy = noisy[offset : offset + clip_samples]
    probabilities = np.asarray(
        analyze_vad_probabilities(noisy, SAMPLE_RATE, 0.48), dtype=np.float32
    )
    control = _control_probabilities(probabilities, noisy.size)
    labels = _control_probabilities(
        np.asarray(
            analyze_vad_probabilities(clean, SAMPLE_RATE, 0.48), dtype=np.float32
        ),
        clean.size,
    )
    baseline = simulate_gate_suppressor_order(noisy, control.tolist(), False, 1.0)
    candidate = simulate_gate_suppressor_order(noisy, control.tolist(), True, 1.0)
    baseline_gain = np.asarray(baseline["gate_gain"], dtype=np.float64)
    candidate_gain = np.asarray(candidate["gate_gain"], dtype=np.float64)
    active = labels[: baseline_gain.size] >= 0.48
    tail = np.zeros(active.shape, dtype=bool)
    tail_frames = int(round(0.20 * SAMPLE_RATE / FRAME_SIZE))
    for index in np.flatnonzero(active[:-1] & ~active[1:]):
        tail[index + 1 : min(tail.size, index + 1 + tail_frames)] = True
    baseline_output = np.asarray(baseline["output_audio"], dtype=np.float64)
    candidate_output = np.asarray(candidate["output_audio"], dtype=np.float64)
    baseline_rms = _block_rms(baseline_output)[: active.size]
    candidate_rms = _block_rms(candidate_output)[: active.size]
    input_rms = _block_rms(noisy)[: active.size]

    def retained_ratio(output_rms: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 1.0
        return float(np.median(output_rms[mask] / np.maximum(input_rms[mask], 1e-9)))

    return {
        "id": clean_path.name.removesuffix("_mixture_clean.wav"),
        "language": clean_path.name.split("_", 1)[0],
        "active_ratio": float(np.mean(active)),
        "tail_ratio": float(np.mean(tail)),
        "baseline_false_closure_rate": float(np.mean(baseline_gain[active] < 0.1)),
        "candidate_false_closure_rate": float(np.mean(candidate_gain[active] < 0.1)),
        "baseline_active_retained_ratio": retained_ratio(baseline_rms, active),
        "candidate_active_retained_ratio": retained_ratio(candidate_rms, active),
        "baseline_tail_retained_ratio": retained_ratio(baseline_rms, tail),
        "candidate_tail_retained_ratio": retained_ratio(candidate_rms, tail),
        "baseline_pumping_score": _pumping(baseline_gain),
        "candidate_pumping_score": _pumping(candidate_gain),
        "baseline_chatter_events": int(baseline["gate_chatter_event_count"]),
        "candidate_chatter_events": int(candidate["gate_chatter_event_count"]),
        "baseline_runtime_ms": float(baseline["runtime_ms"]),
        "candidate_runtime_ms": float(candidate["runtime_ms"]),
    }


def _gate_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def median(key: str) -> float:
        return float(np.median([row[key] for row in rows]))

    baseline_chatter = sum(int(row["baseline_chatter_events"]) for row in rows)
    candidate_chatter = sum(int(row["candidate_chatter_events"]) for row in rows)
    metrics = {
        "baseline_chatter_events": baseline_chatter,
        "candidate_chatter_events": candidate_chatter,
        "median_baseline_false_closure_rate": median("baseline_false_closure_rate"),
        "median_candidate_false_closure_rate": median("candidate_false_closure_rate"),
        "median_baseline_active_retained_ratio": median(
            "baseline_active_retained_ratio"
        ),
        "median_candidate_active_retained_ratio": median(
            "candidate_active_retained_ratio"
        ),
        "median_baseline_tail_retained_ratio": median("baseline_tail_retained_ratio"),
        "median_candidate_tail_retained_ratio": median("candidate_tail_retained_ratio"),
        "median_baseline_pumping_score": median("baseline_pumping_score"),
        "median_candidate_pumping_score": median("candidate_pumping_score"),
        "median_runtime_ratio": median("candidate_runtime_ms")
        / max(median("baseline_runtime_ms"), 1e-9),
    }
    non_regression = {
        "false_closure": (
            metrics["median_candidate_false_closure_rate"]
            <= metrics["median_baseline_false_closure_rate"] + 0.005
        ),
        "active_retention": (
            metrics["median_candidate_active_retained_ratio"]
            >= metrics["median_baseline_active_retained_ratio"] * 0.98
        ),
        "tail_retention": (
            metrics["median_candidate_tail_retained_ratio"]
            >= metrics["median_baseline_tail_retained_ratio"] * 0.98
        ),
        "pumping": (
            metrics["median_candidate_pumping_score"]
            <= metrics["median_baseline_pumping_score"] + 0.03
        ),
        "runtime": metrics["median_runtime_ratio"] <= 1.10,
    }
    material_win = bool(
        candidate_chatter <= baseline_chatter * 0.90
        or metrics["median_candidate_false_closure_rate"]
        <= metrics["median_baseline_false_closure_rate"] - 0.01
        or metrics["median_candidate_tail_retained_ratio"]
        >= metrics["median_baseline_tail_retained_ratio"] * 1.05
    )
    retain_candidate = material_win and all(non_regression.values())
    return {
        "predefined_gates": {
            "material_win": ">=10% chatter reduction, >=1 point false-closure reduction, or >=5% tail retention improvement",
            "active_and_tail_retention_min_ratio_vs_baseline": 0.98,
            "pumping_regression_max": 0.03,
            "runtime_ratio_max": 1.10,
        },
        "metrics": metrics,
        "non_regression": non_regression,
        "material_win": material_win,
        "decision": "suppressor_before_gate"
        if retain_candidate
        else "retain_gate_before_suppressor",
    }


def _band_energy(audio: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    spectrum = np.fft.rfft(audio * np.hanning(audio.size))
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(np.square(np.abs(spectrum[mask]))))


def _eq_deesser_decision() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for specification in CORPUS_CASES:
        generated = generate_deesser_case(specification)
        bands = [
            (80.0, 0.0, 1.0),
            (140.0, 0.0, 1.0),
            (250.0, 0.0, 1.0),
            (450.0, 0.0, 1.0),
            (800.0, 0.0, 1.0),
            (1_500.0, 0.0, 1.0),
            (2_800.0, 1.5, 1.0),
            (5_000.0, 4.0, 1.1),
            (8_000.0, 3.0, 1.0),
            (12_000.0, 0.0, 1.0),
        ]
        common = {
            "deesser_enabled": True,
            "deesser_auto_enabled": True,
            "deesser_auto_amount": 0.75,
            "compressor_enabled": False,
            "limiter_enabled": False,
            "return_output_audio": True,
        }
        baseline = simulate_auto_eq_chain(
            generated.speech_audio,
            specification.sample_rate,
            bands,
            {**common, "eq_before_deesser": False},
        )
        candidate = simulate_auto_eq_chain(
            generated.speech_audio,
            specification.sample_rate,
            bands,
            {**common, "eq_before_deesser": True},
        )
        baseline_audio = np.asarray(baseline["output_audio"], dtype=np.float64)
        candidate_audio = np.asarray(candidate["output_audio"], dtype=np.float64)
        input_hf = _band_energy(
            generated.speech_audio.astype(np.float64),
            specification.sample_rate,
            4_000.0,
            min(10_000.0, specification.sample_rate * 0.45),
        )
        rows.append(
            {
                "id": specification.name,
                "needs_deesser": specification.needs_deesser,
                "condition": specification.condition,
                "baseline_peak_reduction_db": float(
                    baseline["deesser_gain_reduction_db"]
                ),
                "candidate_peak_reduction_db": float(
                    candidate["deesser_gain_reduction_db"]
                ),
                "baseline_hf_change_db": 10.0
                * np.log10(
                    max(
                        _band_energy(
                            baseline_audio,
                            specification.sample_rate,
                            4_000.0,
                            min(10_000.0, specification.sample_rate * 0.45),
                        ),
                        1e-18,
                    )
                    / max(input_hf, 1e-18)
                ),
                "candidate_hf_change_db": 10.0
                * np.log10(
                    max(
                        _band_energy(
                            candidate_audio,
                            specification.sample_rate,
                            4_000.0,
                            min(10_000.0, specification.sample_rate * 0.45),
                        ),
                        1e-18,
                    )
                    / max(input_hf, 1e-18)
                ),
            }
        )
    positive = [row for row in rows if row["needs_deesser"]]
    negative = [row for row in rows if not row["needs_deesser"]]

    def median(group: list[dict[str, Any]], key: str) -> float:
        return float(np.median([row[key] for row in group]))

    metrics = {
        "positive_baseline_peak_reduction_db": median(
            positive, "baseline_peak_reduction_db"
        ),
        "positive_candidate_peak_reduction_db": median(
            positive, "candidate_peak_reduction_db"
        ),
        "negative_baseline_peak_reduction_db": median(
            negative, "baseline_peak_reduction_db"
        ),
        "negative_candidate_peak_reduction_db": median(
            negative, "candidate_peak_reduction_db"
        ),
        "bright_baseline_hf_change_db": median(
            [row for row in negative if row["condition"] == "bright"],
            "baseline_hf_change_db",
        ),
        "bright_candidate_hf_change_db": median(
            [row for row in negative if row["condition"] == "bright"],
            "candidate_hf_change_db",
        ),
    }
    gates = {
        "positive_reduction_improves_by_0_25_db": (
            metrics["positive_candidate_peak_reduction_db"]
            >= metrics["positive_baseline_peak_reduction_db"] + 0.25
        ),
        "negative_reduction_regression_at_most_0_10_db": (
            metrics["negative_candidate_peak_reduction_db"]
            <= metrics["negative_baseline_peak_reduction_db"] + 0.10
        ),
        "bright_hf_attenuation_regression_at_most_0_25_db": (
            metrics["bright_candidate_hf_change_db"]
            >= metrics["bright_baseline_hf_change_db"] - 0.25
        ),
    }
    retain_candidate = all(gates.values())
    return {
        "predefined_gates": {
            "positive_reduction_improvement_db_min": 0.25,
            "negative_reduction_regression_db_max": 0.10,
            "bright_hf_attenuation_regression_db_max": 0.25,
        },
        "metrics": metrics,
        "gates": gates,
        "decision": "eq_before_deesser"
        if retain_candidate
        else "retain_deesser_before_eq",
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    args = parser.parse_args()
    gate_rows = [
        _gate_case(clean, noisy, args.clip_seconds)
        for clean, noisy in _paired_paths(args.corpus_root)
    ]
    report = {
        "schema_version": 2,
        "audible_change": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_suppressor": {**_gate_decision(gate_rows), "cases": gate_rows},
        "eq_deesser": _eq_deesser_decision(),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "listening_status": {
            "status": "not_run",
            "reason": "The autonomous hardening scope excludes human listening; ambiguous candidates retain the incumbent.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_suppressor": report["gate_suppressor"]["decision"],
                "eq_deesser": report["eq_deesser"]["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
