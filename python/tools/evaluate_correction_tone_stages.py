"""Evaluate separate Auto-EQ correction and user-tone stages offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from mic_eq import (
    eq_magnitude_response_v2,
    simulate_auto_eq_chain,
    simulate_eq_v2,
)
from mic_eq.analysis.wav_io import read_mono_wav
from mic_eq.analysis.auto_eq import analyze_auto_eq
from mic_eq.config import EQSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "models" / "deepfilter_fullband_eval" / "clean"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "correction-tone-stage-report.json"
SAMPLE_RATE = 48_000
SEPARATOR_SECONDS = 0.25
GATES: dict[str, float | int] = {
    "min_corpus_cases": 8,
    "max_response_parity_delta_db": 1.0e-9,
    "max_true_peak_overshoot_db": 0.05,
    "max_p95_limiter_gr_db": 3.0,
    "max_candidate_p95_realtime_factor": 0.01,
    "max_p95_runtime_ratio": 2.25,
    "min_p95_runtime_improvement_fraction": 0.05,
    "required_tone_profiles": 4,
}

TypedBand: TypeAlias = tuple[str, float, float, float, int, bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _default_bands(*, enabled: bool = True) -> list[TypedBand]:
    return [
        (
            band.filter_type,
            band.frequency_hz,
            band.gain_db,
            band.q,
            band.slope_db_per_octave,
            enabled,
        )
        for band in EQSettings().bands
    ]


def _typed_correction(settings: dict[str, Any]) -> list[TypedBand]:
    frequencies = list(settings["band_freqs"])
    gains = list(settings["band_gains"])
    qs = list(settings["band_qs"])
    if not (len(frequencies) == len(gains) == len(qs) == 10):
        raise ValueError("Auto-EQ correction must contain ten bands")
    return [
        (
            "low_shelf" if index == 0 else "high_shelf" if index == 9 else "bell",
            float(frequency),
            float(gain),
            float(q),
            12,
            bool(abs(float(gain)) >= 0.25),
        )
        for index, (frequency, gain, q) in enumerate(
            zip(frequencies, gains, qs, strict=True)
        )
    ]


def _tone_profiles() -> dict[str, list[TypedBand]]:
    profiles: dict[str, list[TypedBand]] = {}
    flat = _default_bands(enabled=False)
    profiles["flat"] = flat

    presence = _default_bands(enabled=False)
    presence[5] = ("bell", 2500.0, 2.5, 1.8, 12, True)
    presence[6] = ("bell", 5000.0, 1.5, 1.4, 12, True)
    profiles["presence"] = presence

    warm = _default_bands(enabled=False)
    warm[0] = ("low_shelf", 120.0, 2.0, 0.8, 12, True)
    warm[4] = ("bell", 1300.0, -1.0, 1.2, 12, True)
    profiles["warm"] = warm

    bass_cut = _default_bands(enabled=False)
    bass_cut[0] = ("low_shelf", 100.0, -3.0, 0.8, 12, True)
    profiles["bass_cut"] = bass_cut
    return profiles


def _validate_stage(bands: list[TypedBand]) -> None:
    if len(bands) != 10:
        raise ValueError("each EQ stage must contain exactly ten bands")
    response = np.asarray(
        eq_magnitude_response_v2(
            [20.0, 1000.0, 20_000.0],
            bands,
            float(SAMPLE_RATE),
        ),
        dtype=float,
    )
    if response.shape != (3,) or not np.all(np.isfinite(response)):
        raise ValueError("stage response must be finite")


def _candidate_payload(
    correction: list[TypedBand],
    tone: list[TypedBand],
) -> dict[str, Any]:
    _validate_stage(correction)
    _validate_stage(tone)
    return {
        "schema_version": 1,
        "enabled": True,
        "correction": [list(band) for band in correction],
        "tone": [list(band) for band in tone],
    }


def _decode_candidate(payload: object) -> tuple[list[TypedBand], list[TypedBand]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "enabled",
        "correction",
        "tone",
    }:
        raise ValueError("invalid two-stage candidate schema")
    if payload["schema_version"] != 1 or payload["enabled"] is not True:
        raise ValueError("unsupported or disabled two-stage candidate")

    def parse(raw: object) -> list[TypedBand]:
        if not isinstance(raw, list) or len(raw) != 10:
            raise ValueError("each EQ stage must contain exactly ten bands")
        bands: list[TypedBand] = []
        for value in raw:
            if not isinstance(value, list) or len(value) != 6:
                raise ValueError("typed stage bands require six fields")
            filter_type, frequency, gain, q, slope, enabled = value
            if not isinstance(filter_type, str) or not isinstance(enabled, bool):
                raise ValueError("invalid typed stage band")
            bands.append(
                (
                    filter_type,
                    float(frequency),
                    float(gain),
                    float(q),
                    int(slope),
                    enabled,
                )
            )
        _validate_stage(bands)
        return bands

    return parse(payload["correction"]), parse(payload["tone"])


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _migrate_combined(combined: list[TypedBand]) -> dict[str, Any]:
    """Preserve the incumbent response by treating it as user-owned tone."""
    return _candidate_payload(_default_bands(enabled=False), combined)


def _replace_correction(
    payload: dict[str, Any],
    correction: list[TypedBand],
) -> dict[str, Any]:
    _old_correction, tone = _decode_candidate(payload)
    return _candidate_payload(correction, tone)


def _combined_response(
    frequencies: np.ndarray,
    correction: list[TypedBand],
    tone: list[TypedBand],
) -> np.ndarray:
    return np.asarray(
        eq_magnitude_response_v2(
            frequencies.tolist(),
            correction,
            float(SAMPLE_RATE),
        ),
        dtype=float,
    ) + np.asarray(
        eq_magnitude_response_v2(
            frequencies.tolist(),
            tone,
            float(SAMPLE_RATE),
        ),
        dtype=float,
    )


def _read_audio(path: Path) -> np.ndarray:
    sample_rate, audio = read_mono_wav(
        path,
        allow_stereo=False,
        dtype=np.float32,
    )
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path.name} must be native-48-kHz mono")
    return audio


def _corpus_cases(root: Path) -> list[dict[str, Any]]:
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
    paths = sorted(expected_hashes)
    if len(paths) < 16:
        raise RuntimeError("at least 16 native clean clips are required")
    separator = np.zeros(
        int(round(SEPARATOR_SECONDS * SAMPLE_RATE)),
        dtype=np.float32,
    )
    cases: list[dict[str, Any]] = []
    for index in range(0, min(len(paths), 24), 2):
        selected = paths[index : index + 2]
        if len(selected) != 2:
            continue
        for path in selected:
            if _sha256(path) != expected_hashes[path]:
                raise ValueError(f"clean corpus hash mismatch: {path.name}")
        parts = [_read_audio(path) for path in selected]
        audio = np.concatenate((parts[0], separator, parts[1]))
        cases.append(
            {
                "id": f"{selected[0].stem}+{selected[1].stem}",
                "paths": [_relative(path) for path in selected],
                "hashes": [_sha256(path) for path in selected],
                "audio": audio,
            }
        )
    return cases


def _normalized(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return np.asarray(audio * (0.5 / max(peak, 1.0e-12)), dtype=np.float32)


def _legacy_flat() -> list[tuple[float, float, float]]:
    return [(band[1], 0.0, band[3]) for band in _default_bands()]


def _render_case(
    audio: np.ndarray,
    correction: list[TypedBand],
    tone: list[TypedBand],
) -> dict[str, Any]:
    normalized = _normalized(audio)
    incumbent = simulate_eq_v2(
        normalized,
        float(SAMPLE_RATE),
        correction,
        return_output_audio=True,
    )
    correction_audio = np.asarray(incumbent["output_audio"], dtype=np.float32)
    tone_result = simulate_eq_v2(
        correction_audio,
        float(SAMPLE_RATE),
        tone,
        return_output_audio=True,
    )
    candidate_audio = np.asarray(tone_result["output_audio"], dtype=np.float32)
    chain = simulate_auto_eq_chain(
        candidate_audio,
        float(SAMPLE_RATE),
        _legacy_flat(),
        {
            "deesser_enabled": False,
            "compressor_enabled": False,
            "limiter_enabled": True,
            "limiter_ceiling_db": -1.0,
            "limiter_careful_output_enabled": True,
        },
    )
    duration_seconds = normalized.size / SAMPLE_RATE
    incumbent_runtime_ms = float(incumbent["runtime_ms"])
    candidate_runtime_ms = incumbent_runtime_ms + float(tone_result["runtime_ms"])
    return {
        "incumbent_realtime_factor": incumbent_runtime_ms
        / max(duration_seconds * 1000.0, 1.0e-12),
        "candidate_realtime_factor": candidate_runtime_ms
        / max(duration_seconds * 1000.0, 1.0e-12),
        "runtime_ratio": candidate_runtime_ms / max(incumbent_runtime_ms, 1.0e-12),
        "latency_samples": [
            int(incumbent["algorithmic_latency_samples"]),
            int(tone_result["algorithmic_latency_samples"]),
        ],
        "finite": not bool(incumbent["non_finite_output"])
        and not bool(tone_result["non_finite_output"])
        and not bool(chain["non_finite_output"]),
        "full_chain_true_peak_overshoot_db": max(
            0.0,
            float(chain["output_true_peak_db"])
            - float(chain["limiter_effective_ceiling_db"]),
        ),
        "full_chain_limiter_gr_db": max(
            float(chain["limiter_gain_reduction_db"]),
            float(chain["true_peak_limiter_gain_reduction_db"]),
        ),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    incumbent_rtfs = [float(row["render"]["incumbent_realtime_factor"]) for row in rows]
    candidate_rtfs = [float(row["render"]["candidate_realtime_factor"]) for row in rows]
    runtime_ratios = [float(row["render"]["runtime_ratio"]) for row in rows]
    limiter = [float(row["render"]["full_chain_limiter_gr_db"]) for row in rows]
    return {
        "case_count": len(rows),
        "tone_profiles": sorted({str(row["tone_profile"]) for row in rows}),
        "max_response_parity_delta_db": max(
            (float(row["response_parity_delta_db"]) for row in rows),
            default=math.inf,
        ),
        "tone_payload_preserved": all(bool(row["tone_payload_preserved"]) for row in rows),
        "schema_roundtrip": all(bool(row["schema_roundtrip"]) for row in rows),
        "all_outputs_finite": all(bool(row["render"]["finite"]) for row in rows),
        "max_true_peak_overshoot_db": max(
            (float(row["render"]["full_chain_true_peak_overshoot_db"]) for row in rows),
            default=math.inf,
        ),
        "p95_limiter_gr_db": _percentile(limiter, 95),
        "incumbent_p95_realtime_factor": _percentile(incumbent_rtfs, 95),
        "candidate_p95_realtime_factor": _percentile(candidate_rtfs, 95),
        "p95_runtime_ratio": _percentile(runtime_ratios, 95),
        "latency_samples": sorted(
            {
                int(value)
                for row in rows
                for value in row["render"]["latency_samples"]
            }
        ),
    }


def _gate(aggregate: dict[str, Any]) -> dict[str, bool]:
    return {
        "enough_cases": aggregate["case_count"] >= GATES["min_corpus_cases"],
        "all_tone_profiles": len(aggregate["tone_profiles"])
        >= GATES["required_tone_profiles"],
        "response_parity": aggregate["max_response_parity_delta_db"]
        <= GATES["max_response_parity_delta_db"],
        "tone_ownership": bool(aggregate["tone_payload_preserved"]),
        "schema_and_undo_shape": bool(aggregate["schema_roundtrip"]),
        "finite_output": bool(aggregate["all_outputs_finite"]),
        "true_peak_ceiling": aggregate["max_true_peak_overshoot_db"]
        <= GATES["max_true_peak_overshoot_db"],
        "limiter_load": aggregate["p95_limiter_gr_db"] is not None
        and aggregate["p95_limiter_gr_db"] <= GATES["max_p95_limiter_gr_db"],
        "runtime_absolute": aggregate["candidate_p95_realtime_factor"] is not None
        and aggregate["candidate_p95_realtime_factor"]
        <= GATES["max_candidate_p95_realtime_factor"],
        "runtime_ratio": aggregate["p95_runtime_ratio"] is not None
        and aggregate["p95_runtime_ratio"] <= GATES["max_p95_runtime_ratio"],
        "zero_added_latency": aggregate["latency_samples"] == [0],
        "material_objective_benefit": (
            aggregate["candidate_p95_realtime_factor"] is not None
            and aggregate["incumbent_p95_realtime_factor"] is not None
            and aggregate["candidate_p95_realtime_factor"]
            <= aggregate["incumbent_p95_realtime_factor"]
            * (1.0 - GATES["min_p95_runtime_improvement_fraction"])
        ),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        "docs/correction-tone-stage-evaluation.md",
        "python/tools/evaluate_correction_tone_stages.py",
        "python/mic_eq/analysis/wav_io.py",
        "rust-core/src/dsp/eq.rs",
        "rust-core/src/lib.rs",
    )
    return {path: _sha256(REPO_ROOT / path) for path in paths}


def evaluate(corpus_root: Path) -> dict[str, Any]:
    corpus_root = corpus_root.resolve(strict=True)
    cases = _corpus_cases(corpus_root)
    tones = _tone_profiles()
    tone_names = tuple(tones)
    grid = np.geomspace(20.0, 20_000.0, 512)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, case in enumerate(cases):
        try:
            audio = np.asarray(case["audio"], dtype=np.float32)
            settings, _validation = analyze_auto_eq(
                audio,
                SAMPLE_RATE,
                "broadcast",
            )
            correction = _typed_correction(settings)
            tone_name = tone_names[index % len(tone_names)]
            tone = tones[tone_name]
            migrated = _migrate_combined(tone)
            tone_before = json.dumps(
                migrated["tone"],
                allow_nan=False,
                separators=(",", ":"),
            )
            candidate = _replace_correction(migrated, correction)
            _decoded_correction, _decoded_tone = _decode_candidate(candidate)
            tone_after = json.dumps(
                candidate["tone"],
                allow_nan=False,
                separators=(",", ":"),
            )
            migrated_correction, migrated_tone = _decode_candidate(migrated)
            migrated_response = _combined_response(
                grid,
                migrated_correction,
                migrated_tone,
            )
            incumbent_response = np.asarray(
                eq_magnitude_response_v2(
                    grid.tolist(),
                    tone,
                    float(SAMPLE_RATE),
                ),
                dtype=float,
            )
            encoded = _canonical_payload(candidate)
            decoded = _decode_candidate(json.loads(encoded))
            rows.append(
                {
                    "id": str(case["id"]),
                    "paths": list(case["paths"]),
                    "hashes": list(case["hashes"]),
                    "duration_seconds": audio.size / SAMPLE_RATE,
                    "tone_profile": tone_name,
                    "response_parity_delta_db": float(
                        np.max(np.abs(migrated_response - incumbent_response))
                    ),
                    "tone_payload_preserved": tone_before == tone_after,
                    "schema_roundtrip": decoded == (correction, tone),
                    "render": _render_case(audio, correction, tone),
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append({"id": str(case["id"]), "error": str(error)})

    aggregate = _aggregate(rows)
    aggregate["failed_cases"] = len(failures)
    checks = _gate(aggregate)
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    manifest = corpus_root.parent / "manifest.json"
    source_hashes = _source_hashes()
    return {
        "schema_version": 2,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Separate Auto-EQ correction and user tone stages",
        "candidate": {
            "scope": "evaluation_only",
            "order": ["deesser", "correction", "tone", "compressor", "limiter"],
            "migration": "incumbent combined stage becomes tone; correction starts flat",
        },
        "incumbent": {"stages": 1, "bands": 10},
        "retention_gates": GATES,
        "checks": checks,
        "decision": {
            "retained": not failed_checks,
            "failed_checks": failed_checks,
            "product_action": (
                "integrate two stages"
                if not failed_checks
                else "retain one combined stage and reject non-combined stage tokens"
            ),
        },
        "aggregate": aggregate,
        "failures": failures,
        "cases": rows,
        "evaluation_contract": {
            "configuration": {
                "sample_rate": SAMPLE_RATE,
                "tone_profiles": list(tone_names),
                "pre_registration": "docs/correction-tone-stage-evaluation.md",
                "input_peak_normalization": 0.5,
            },
            "asset_hashes": {
                "corpus_manifest": _sha256(manifest),
                "source": source_hashes,
            },
            "runtime": {
                "incumbent_p95_realtime_factor": aggregate[
                    "incumbent_p95_realtime_factor"
                ],
                "candidate_p95_realtime_factor": aggregate[
                    "candidate_p95_realtime_factor"
                ],
                "p95_runtime_ratio": aggregate["p95_runtime_ratio"],
                "max_p99_frame_seconds": None,
                "max_p99_frame_seconds_reason": (
                    "The native offline API reports whole-clip runtime; realtime "
                    "factor is measured for both one- and two-stage paths."
                ),
                "machine": platform.platform(),
                "python": platform.python_version(),
            },
            "latency": {
                "algorithmic_latency_samples": aggregate["latency_samples"],
                "sample_rate": SAMPLE_RATE,
            },
            "clean_preservation": {
                "all_outputs_finite": aggregate["all_outputs_finite"],
                "max_true_peak_overshoot_db": aggregate[
                    "max_true_peak_overshoot_db"
                ],
                "p95_limiter_gr_db": aggregate["p95_limiter_gr_db"],
            },
        },
        "source_sha256": source_hashes,
        "limitations": [
            "The 48 kHz clean subset contains only two VoiceBank test speakers.",
            "Frozen tone profiles are representative fixtures, not preference labels.",
            "Response parity proves safety but not benefit; the candidate must therefore provide a predefined measurable runtime benefit to justify doubling the stage architecture.",
            "The candidate is not a production preset or realtime DSP path.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate(args.corpus_root)
    report_path = args.report
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": _relative(report_path),
                "retained": report["decision"]["retained"],
                "failed_checks": report["decision"]["failed_checks"],
                "case_count": report["aggregate"]["case_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
