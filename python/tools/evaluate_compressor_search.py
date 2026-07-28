"""Evaluate expanded compressor search against the shipped threshold-only search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from mic_eq.analysis.voice_setup import (
    _COMPRESSOR_OBJECTIVE_NORMALIZERS,
    _COMPRESSOR_OBJECTIVE_WEIGHTS,
    _COMPRESSOR_SEARCH_BUDGET,
    _calibrate_compressor_threshold,
)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    scale = 32768.0 if np.issubdtype(raw.dtype, np.integer) else 1.0
    return int(sample_rate), np.asarray(raw, dtype=np.float32) / scale


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
        )
        baseline = float(diagnostics["threshold_only_objective"])
        candidate = float(diagnostics["total_objective"])
        improvement = (
            (baseline - candidate) / baseline
            if np.isfinite(baseline) and baseline > 0.0
            else 0.0
        )
        rows.append(
            {
                "path": capture["path"],
                "condition": capture["condition"],
                "threshold_only_objective": baseline,
                "expanded_objective": candidate,
                "relative_improvement": improvement,
                "expanded_selected": diagnostics["expanded_search_selected"],
                "candidate_count": diagnostics["candidate_count"],
                "search_runtime_ms": diagnostics["search_runtime_ms"],
                "winner": {
                    key: settings[key]
                    for key in ("threshold_db", "ratio", "attack_ms", "release_ms")
                },
                "median_gain_reduction_db": diagnostics[
                    "measured_median_gain_reduction_db"
                ],
                "p95_gain_reduction_db": diagnostics[
                    "measured_p95_gain_reduction_db"
                ],
                "peak_gain_reduction_db": diagnostics[
                    "measured_peak_gain_reduction_db"
                ],
                "pumping_score_db": diagnostics["compressor_pumping_score_db"],
                "silence_output_gain_db": diagnostics["silence_output_gain_db"],
                "pre_limiter_true_peak_headroom_db": diagnostics[
                    "pre_limiter_true_peak_headroom_db"
                ],
                "peak_cap_passed": diagnostics["peak_cap_passed"],
                "output_true_peak_db": diagnostics["output_true_peak_db"],
            }
        )

    improvements = np.asarray(
        [row["relative_improvement"] for row in rows],
        dtype=float,
    )
    median_improvement = float(np.median(improvements))
    improved_fraction = float(np.mean(improvements > 0.0))
    safety_passed = all(
        row["peak_cap_passed"]
        and row["candidate_count"] <= _COMPRESSOR_SEARCH_BUDGET
        and np.isfinite(row["output_true_peak_db"])
        and abs(row["median_gain_reduction_db"] - 1.4) <= 1.5
        and abs(row["p95_gain_reduction_db"] - 3.5) <= 1.5
        and row["pumping_score_db"] <= 2.0
        and row["silence_output_gain_db"] <= 0.25
        and row["pre_limiter_true_peak_headroom_db"] >= 0.0
        for row in rows
    )
    retained = bool(
        median_improvement >= 0.05
        and improved_fraction >= 0.60
        and safety_passed
    )
    return {
        "schema_version": 1,
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
            "silence_output_gain_db_max": 0.25,
            "pre_limiter_true_peak_headroom_db_min": 0.0,
        },
        "median_relative_improvement": median_improvement,
        "improved_fraction": improved_fraction,
        "safety_passed": safety_passed,
        "retained": retained,
        "rows": rows,
        "limitations": [
            "The held-out speech source is isolated spoken digits, not long-form narration.",
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
