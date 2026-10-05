"""Evaluate de-esser/EQ ordering without changing defaults.

Gate/suppressor order is evaluated by ``evaluate_gate_corpus.py order``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mic_eq.mic_eq_core import simulate_auto_eq_chain
from mic_eq.analysis.deesser_corpus import CORPUS_CASES, generate_deesser_case


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 48_000


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-case report; the tracked report stays compact.",
    )
    args = parser.parse_args()
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "python/mic_eq/analysis/deesser_corpus.py",
        REPO_ROOT / "rust-core/src/audio/processor/block_processor.rs",
        REPO_ROOT / "rust-core/src/audio/processor/python_api.rs",
        REPO_ROOT / "rust-core/src/dsp/deesser.rs",
        REPO_ROOT / "rust-core/src/dsp/eq.rs",
    )
    eq_proxy = _eq_deesser_decision()
    source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): _source_sha256(path) for path in source_paths
    }
    report = {
        "schema_version": 5,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "product_chain_changed": eq_proxy["decision"] != "retain_deesser_before_eq",
            "eq_deesser": eq_proxy["decision"],
        },
        "eq_deesser": eq_proxy,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "provenance": {
            "source_hashes": source_hashes,
            "native_simulation_backend": "mic_eq_core",
        },
        "evaluation_contract": {
            "configuration": {
                "sample_rate": SAMPLE_RATE,
                "comparisons": ["deesser-before-EQ vs EQ-before-deesser"],
                "product_defaults_changed": False,
            },
            "asset_hashes": {"assets": {}},
            "runtime": {
                "max_p99_frame_seconds": None,
                "max_p99_frame_seconds_reason": (
                    "The native simulator reports whole-clip timing; a reordering "
                    "candidate is qualified before realtime callback measurement."
                ),
                "platform": platform.platform(),
            },
            "latency": {
                "algorithmic_latency_delta_samples": 0,
                "reason": "The comparison reorders the same processing components.",
            },
            "clean_preservation": {"eq_deesser_gates": eq_proxy["gates"]},
        },
        "limitations": [
            "The comparison uses deterministic generated de-esser proxy cases.",
        ],
    }
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        args.details_output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report["eq_deesser"].pop("cases", None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"eq_deesser": report["eq_deesser"]["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
