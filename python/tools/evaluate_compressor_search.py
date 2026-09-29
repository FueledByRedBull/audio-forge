"""Evaluate expanded compressor search against the shipped threshold-only search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from release_provenance import sha256_file as _sha256

import numpy as np

from mic_eq.analysis.voice_setup import (
    _COMPRESSOR_OBJECTIVE_NORMALIZERS,
    _COMPRESSOR_OBJECTIVE_WEIGHTS,
    _COMPRESSOR_SEARCH_BUDGET,
    _calibrate_compressor_threshold,
)
from mic_eq.analysis.wav_io import read_mono_wav


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


CONDITIONS = {"clean_48k", "noise_5db_48k", "noise_20db_48k", "quiet"}
EQ_SETTINGS = {
    "band_freqs": [80, 160, 315, 630, 1250, 2500, 4000, 6300, 10000, 16000],
    "band_gains": [0.0] * 10,
    "band_qs": [1.41] * 10,
}
INCUMBENT = {
    "enabled": True,
    "threshold_db": -24.0,
    "ratio": 3.0,
    "attack_ms": 10.0,
    "release_ms": 180.0,
    "makeup_gain_db": 0.0,
    "adaptive_release": True,
    "base_release_ms": 80.0,
    "auto_makeup_enabled": True,
    "target_lufs": -18.0,
    "measured_short_term_lufs": -22.0,
    "sidechain_highpass_enabled": True,
}


def _load_audio(path: Path) -> tuple[int, np.ndarray]:
    return read_mono_wav(path, allow_stereo=False, dtype=np.float32)


def evaluate(manifest_path: Path, limit: int) -> dict[str, Any]:
    corpus_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captures = [
        capture
        for capture in manifest["captures"]
        if capture["split"] == "held_out"
        and capture["sample_rate"] == 48_000
        and capture["condition"] in CONDITIONS
    ][:limit]
    if len(captures) < limit:
        raise ValueError(f"needed {limit} held-out captures, found {len(captures)}")

    rows: list[dict[str, Any]] = []
    for capture in captures:
        sample_rate, audio = _load_audio(corpus_root / capture["path"])
        settings, diagnostics = _calibrate_compressor_threshold(
            speech_audio=audio,
            sample_rate=sample_rate,
            eq_settings=EQ_SETTINGS,
            deesser_settings={"enabled": False},
            compressor_settings=INCUMBENT,
            target_p95_db=3.5,
            target_median_db=1.4,
            peak_cap_db=8.0,
            allow_expanded_search=True,
        )
        baseline = _finite_float_or_none(
            diagnostics.get("threshold_only_objective")
        )
        candidate = _finite_float_or_none(diagnostics.get("total_objective"))
        improvement = (
            (baseline - candidate) / baseline
            if baseline is not None and candidate is not None and baseline > 0.0
            else None
        )
        raw_candidate_count = diagnostics.get("candidate_count")
        candidate_count = (
            raw_candidate_count
            if isinstance(raw_candidate_count, int)
            and not isinstance(raw_candidate_count, bool)
            and raw_candidate_count >= 0
            else None
        )
        rows.append(
            {
                "path": capture["path"],
                "condition": capture["condition"],
                "calibration_backend": diagnostics.get("backend"),
                "expanded_search_status": diagnostics.get("expanded_search_status"),
                "threshold_only_objective": baseline,
                "expanded_objective": candidate,
                "relative_improvement": improvement,
                "expanded_selected": diagnostics.get("expanded_search_selected"),
                "candidate_count": candidate_count,
                "search_runtime_ms": _finite_float_or_none(
                    diagnostics.get("search_runtime_ms")
                ),
                "winner": {
                    key: settings[key]
                    for key in ("threshold_db", "ratio", "attack_ms", "release_ms")
                },
                "median_gain_reduction_db": _finite_float_or_none(
                    diagnostics.get("measured_median_gain_reduction_db")
                ),
                "p95_gain_reduction_db": _finite_float_or_none(
                    diagnostics.get("measured_p95_gain_reduction_db")
                ),
                "peak_gain_reduction_db": _finite_float_or_none(
                    diagnostics.get("measured_peak_gain_reduction_db")
                ),
                "pumping_score_db": _finite_float_or_none(
                    diagnostics.get("compressor_pumping_score_db")
                ),
                "silence_level_delta_db": _finite_float_or_none(
                    diagnostics.get("silence_level_delta_db")
                ),
                "pre_limiter_true_peak_headroom_db": _finite_float_or_none(
                    diagnostics.get("pre_limiter_true_peak_headroom_db")
                ),
                "peak_cap_passed": diagnostics.get("peak_cap_passed"),
                "output_true_peak_db": _finite_float_or_none(
                    diagnostics.get("output_true_peak_db")
                ),
            }
        )

    measurable_improvements = [
        row["relative_improvement"]
        for row in rows
        if row["relative_improvement"] is not None
    ]
    median_improvement = (
        float(np.median(measurable_improvements))
        if measurable_improvements
        else None
    )
    improved_fraction = (
        float(
            sum(
                row["relative_improvement"] is not None
                and row["relative_improvement"] > 0.0
                for row in rows
            )
            / len(rows)
        )
        if rows
        else 0.0
    )
    measurable_rows = [
        row
        for row in rows
        if row["calibration_backend"] == "rust"
        and row["relative_improvement"] is not None
        and row["candidate_count"] is not None
        and row["median_gain_reduction_db"] is not None
        and row["p95_gain_reduction_db"] is not None
        and row["pumping_score_db"] is not None
        and row["silence_level_delta_db"] is not None
        and row["pre_limiter_true_peak_headroom_db"] is not None
        and row["peak_cap_passed"] is not None
        and row["output_true_peak_db"] is not None
    ]
    measurement_complete = len(measurable_rows) == len(rows)
    safety_passed = all(
        row["peak_cap_passed"] is True
        and row["candidate_count"] is not None
        and row["candidate_count"] <= _COMPRESSOR_SEARCH_BUDGET
        and row["output_true_peak_db"] is not None
        and row["median_gain_reduction_db"] is not None
        and row["p95_gain_reduction_db"] is not None
        and row["pumping_score_db"] is not None
        and abs(row["median_gain_reduction_db"] - 1.4) <= 1.5
        and abs(row["p95_gain_reduction_db"] - 3.5) <= 1.5
        and row["pumping_score_db"] <= 2.0
        and row["silence_level_delta_db"] is not None
        and row["silence_level_delta_db"] <= 0.25
        and row["pre_limiter_true_peak_headroom_db"] is not None
        and row["pre_limiter_true_peak_headroom_db"] >= 0.0
        for row in rows
    )
    retained = bool(
        measurement_complete
        and median_improvement is not None
        and median_improvement >= 0.05
        and improved_fraction >= 0.60
        and safety_passed
    )
    return {
        "schema_version": 3,
        "method": "held-out VAD corpus; exact 33-point threshold baseline versus bounded expanded search",
        "manifest": _portable_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "capture_count": len(rows),
        "candidate_budget": _COMPRESSOR_SEARCH_BUDGET,
        "objective_normalizers": _COMPRESSOR_OBJECTIVE_NORMALIZERS,
        "objective_weights": _COMPRESSOR_OBJECTIVE_WEIGHTS,
        "component_gates": {
            "median_gain_reduction_error_db_max": 1.5,
            "p95_gain_reduction_error_db_max": 1.5,
            "pumping_score_db_max": 2.0,
            "silence_level_delta_db_max": 0.25,
            "pre_limiter_true_peak_headroom_db_min": 0.0,
        },
        "median_relative_improvement": median_improvement,
        "improved_fraction": improved_fraction,
        "measurement_complete": measurement_complete,
        "measurable_capture_count": len(measurable_rows),
        "unmeasurable_capture_count": len(rows) - len(measurable_rows),
        "safety_passed": safety_passed,
        "retained": retained,
        "rows": rows,
        "limitations": [
            "The held-out speech source is isolated spoken digits, not long-form narration.",
            "The adaptive incumbent keeps Base Release fixed at 80 ms; the expanded "
            "release_ms axis is inactive, so this experiment does not qualify release tuning.",
            "Perceptual listening remains required before release.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/vad_eval_corpus/manifest.json"),
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.limit)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["retained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
