"""Compare 0.5, 1.0, and 2.0 ms main-limiter lookahead."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mic_eq import simulate_auto_eq_chain


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "limiter-lookahead-report.json"
SAMPLE_RATE = 48_000
LOOKAHEAD_MS = (0.5, 1.0, 2.0)


def _cases() -> dict[str, np.ndarray]:
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
    return {
        "sine_bursts": np.asarray(sine_bursts, dtype=np.float32),
        "impulses": np.asarray(impulses, dtype=np.float32),
        "clipped_voice": np.asarray(clipped_voice, dtype=np.float32),
    }


def _render(audio: np.ndarray, lookahead_ms: float) -> dict[str, Any]:
    bands = [(80.0 * 1.75**index, 0.0, 1.0) for index in range(10)]
    return simulate_auto_eq_chain(
        audio,
        SAMPLE_RATE,
        bands,
        {
            "deesser_enabled": False,
            "compressor_enabled": False,
            "limiter_enabled": True,
            "limiter_ceiling_db": -1.0,
            "limiter_release_ms": 50.0,
            "limiter_careful_output_enabled": False,
            "limiter_lookahead_ms": lookahead_ms,
            "return_output_audio": True,
        },
    )


def _case(case_id: str, audio: np.ndarray, lookahead_ms: float) -> dict[str, Any]:
    result = _render(audio, lookahead_ms)
    output = np.asarray(result.pop("output_audio"), dtype=np.float64)
    delay = int(round(lookahead_ms / 1000.0 * SAMPLE_RATE)) + 20
    aligned = output[delay:]
    reference = audio[: aligned.size].astype(np.float64)
    active = np.abs(reference) >= 0.05
    gain = aligned[active] / reference[active]
    gain_variation = float(np.std(gain)) if gain.size else 0.0
    ceiling_db = float(result["limiter_effective_ceiling_db"])
    return {
        "id": case_id,
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
        "aligned_gain_variation": gain_variation,
        "runtime_ms": float(result["candidate_runtime_ms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows = [
        _case(case_id, audio, lookahead_ms)
        for case_id, audio in _cases().items()
        for lookahead_ms in LOOKAHEAD_MS
    ]
    aggregates: dict[str, dict[str, float]] = {}
    for lookahead_ms in LOOKAHEAD_MS:
        subset = [row for row in rows if row["lookahead_ms"] == lookahead_ms]
        aggregates[str(lookahead_ms)] = {
            "worst_pre_true_peak_overshoot_db": max(
                row["pre_true_peak_overshoot_db"] for row in subset
            ),
            "worst_output_true_peak_overshoot_db": max(
                row["output_true_peak_overshoot_db"] for row in subset
            ),
            "max_true_peak_limiter_gain_reduction_db": max(
                row["true_peak_limiter_gain_reduction_db"] for row in subset
            ),
            "total_true_peak_limited_events": sum(
                row["true_peak_limited_events"] for row in subset
            ),
            "median_aligned_gain_variation": float(
                np.median([row["aligned_gain_variation"] for row in subset])
            ),
            "median_runtime_ms": float(np.median([row["runtime_ms"] for row in subset])),
        }
    baseline = aggregates["2.0"]
    candidate_checks: dict[str, dict[str, Any]] = {}
    for lookahead_ms in (0.5, 1.0):
        aggregate = aggregates[str(lookahead_ms)]
        checks = {
            "pre_true_peak_overshoot_at_most_0_10_db": (
                aggregate["worst_pre_true_peak_overshoot_db"] <= 0.10
            ),
            "output_true_peak_overshoot_at_most_0_01_db": (
                aggregate["worst_output_true_peak_overshoot_db"] <= 0.01
            ),
            "downstream_true_peak_gr_regression_at_most_0_10_db": (
                aggregate["max_true_peak_limiter_gain_reduction_db"]
                <= baseline["max_true_peak_limiter_gain_reduction_db"] + 0.10
            ),
            "gain_variation_regression_at_most_5_percent": (
                aggregate["median_aligned_gain_variation"]
                <= baseline["median_aligned_gain_variation"] * 1.05 + 1e-6
            ),
            "runtime_not_slower": (
                aggregate["median_runtime_ms"] <= baseline["median_runtime_ms"] * 1.05
            ),
        }
        candidate_checks[str(lookahead_ms)] = {
            "checks": checks,
            "passes": all(checks.values()),
        }
    passing = [
        value for value in (0.5, 1.0) if candidate_checks[str(value)]["passes"]
    ]
    selected = min(passing) if passing else 2.0
    report = {
        "schema_version": 2,
        "audible_change": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predefined_gates": {
            "pre_true_peak_overshoot_db_max": 0.10,
            "output_true_peak_overshoot_db_max": 0.01,
            "downstream_true_peak_gr_regression_db_max": 0.10,
            "gain_variation_regression_ratio_max": 1.05,
            "runtime_ratio_max": 1.05,
        },
        "aggregates": aggregates,
        "candidate_checks": candidate_checks,
        "selected_lookahead_ms": selected,
        "decision": (
            "retain_2ms"
            if selected == 2.0
            else f"adopt_{selected:g}ms_after_listening"
        ),
        "cases": rows,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "listening_status": {
            "status": "not_run",
            "reason": (
                "No human involvement is allowed; even an objective candidate remains "
                "conditional if transient transparency cannot be settled by metrics."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_lookahead_ms": selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
