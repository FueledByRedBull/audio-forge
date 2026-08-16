"""Evaluate cross-take Auto-EQ confidence on held-out real repeated readings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from release_provenance import sha256_file as _sha256

import numpy as np
from scipy.io import wavfile

from mic_eq.analysis.auto_eq import analyze_auto_eq, simulate_candidate_chain
from mic_eq.analysis.auto_eq_parts.cross_take import cross_take_evidence
from mic_eq.analysis.auto_eq_parts.dynamic_bands import (
    _build_dense_log_grid,
    _voice_weights,
)
from mic_eq.analysis.auto_eq_parts.headroom import apply_headroom_validation
from mic_eq.analysis.auto_eq_parts.optimizer import calculate_eq_bands
from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response
from mic_eq.analysis.auto_eq_parts.target import get_target_curve
from mic_eq.analysis.failure_detection import validate_analysis
from mic_eq.analysis.spectrum import (
    VoiceSpectrumResult,
    analyze_voice_spectrum,
    smooth_spectrum_perceptual,
)
from mic_eq.analysis.vad import analyze_offline_vad


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "models" / "cross_take_eval"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "cross-take-auto-eq-report.json"
TARGET_PRESET = "broadcast"
GATES = {
    "min_comparable_test_pairs": 10,
    "min_comparable_test_speakers": 5,
    "min_median_heldout_improvement_db": 0.0,
    "min_p10_heldout_improvement_db": -0.5,
    "max_cross_take_retry_rate": 0.15,
    "max_abstention_rate_increase": 0.10,
    "min_headroom_safe_rate": 1.0,
    "max_runtime_p95_ratio": 2.5,
    "max_runtime_p95_seconds": 4.0,
}


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    audio = np.asarray(raw)
    if int(sample_rate) != 48_000 or audio.ndim != 1:
        raise ValueError(f"{path.name} must be native-48-kHz mono")
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = float(info.max + 1) / 2.0
        audio = (audio.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(int(info.min)), int(info.max)))
        audio = audio.astype(np.float64) / scale
    converted = np.asarray(audio, dtype=np.float32)
    if converted.size < sample_rate or not np.all(np.isfinite(converted)):
        raise ValueError(f"{path.name} is too short or non-finite")
    return int(sample_rate), converted


def _read_manifest_take(
    corpus_root: Path,
    pair: dict[str, Any],
    take_id: str,
    cache: dict[Path, tuple[int, np.ndarray]],
) -> tuple[int, np.ndarray, str]:
    take = pair["takes"][take_id]
    relative_path = Path(str(take["path"]))
    path = (corpus_root / relative_path).resolve(strict=True)
    if not path.is_relative_to(corpus_root):
        raise ValueError(f"Corpus path escapes its root: {relative_path}")
    if path not in cache:
        expected_hash = str(take["sha256"])
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Corpus hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        sample_rate, audio = _read_mono(path)
        if sample_rate != int(pair["sample_rate"]):
            raise ValueError(f"Manifest sample rate mismatch for {relative_path}")
        if audio.size != int(take["frames"]):
            raise ValueError(f"Manifest frame count mismatch for {relative_path}")
        cache[path] = (sample_rate, audio)
    sample_rate, audio = cache[path]
    return sample_rate, audio, f"{pair['id']}:take-{take_id}"


def _concat_statement_takes(
    corpus_root: Path,
    pairs: list[dict[str, Any]],
    take_ids: tuple[str, ...],
    cache: dict[Path, tuple[int, np.ndarray]],
) -> tuple[int, np.ndarray, list[str]]:
    expected_deliveries = {"neutral-normal", "calm-normal", "calm-strong"}
    if {str(pair["delivery"]) for pair in pairs} != expected_deliveries:
        raise RuntimeError("Statement lacks the three delivery conditions")
    sample_rate = 48_000
    separator = np.zeros(sample_rate // 4, dtype=np.float32)
    parts: list[np.ndarray] = []
    source_take_ids: list[str] = []
    for pair in sorted(pairs, key=lambda value: str(value["delivery"])):
        for take_id in take_ids:
            take_rate, part, source_id = _read_manifest_take(
                corpus_root,
                pair,
                take_id,
                cache,
            )
            if take_rate != sample_rate:
                raise RuntimeError(f"{pair['id']} is not native 48 kHz")
            parts.extend((part, separator))
            source_take_ids.append(source_id)
    if not parts:
        raise RuntimeError("Statement has no audio")
    return sample_rate, np.concatenate(parts[:-1]), source_take_ids


def _statement_folds(statement_ids: set[str]) -> list[tuple[str, str]]:
    ordered = sorted(statement_ids)
    if len(ordered) != 2:
        raise RuntimeError(
            "Each speaker must have exactly two statements for lexical holdout"
        )
    return [(ordered[0], ordered[1]), (ordered[1], ordered[0])]


def _heldout_target_error(
    spectrum: VoiceSpectrumResult,
    eq_settings: dict[str, Any],
) -> float:
    frequencies = spectrum.freqs
    measured = smooth_spectrum_perceptual(
        frequencies,
        spectrum.median_spectrum_db,
    )
    voice = (frequencies >= 100.0) & (frequencies <= 8_000.0)
    measured_normalized = measured - float(np.mean(measured[voice]))
    target = get_target_curve(
        frequencies,
        TARGET_PRESET,
        measured_db=measured,
        target_mode="adaptive",
    )
    dense = _build_dense_log_grid(frequencies)
    measured_dense = np.interp(dense, frequencies, measured_normalized)
    target_dense = np.interp(dense, frequencies, target)
    response = _predict_eq_response(
        dense,
        eq_settings["band_gains"],
        eq_settings["band_qs"],
        eq_settings["band_freqs"],
    )
    weights = _voice_weights(dense)
    return float(
        np.sqrt(
            np.sum(weights * np.square(target_dense - measured_dense - response))
            / np.sum(weights)
        )
    )


def _headroom_safe(
    audio: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
) -> tuple[bool, dict[str, float | int | str]]:
    result = simulate_candidate_chain(
        audio,
        sample_rate,
        eq_settings,
        {
            "deesser": {"enabled": False},
            "compressor": {"enabled": False},
            "limiter": {
                "enabled": True,
                "ceiling_db": -1.0,
                "release_ms": 80.0,
                "careful_output_enabled": True,
            },
        },
    )
    true_peak = float(result.get("output_true_peak_db", 120.0))
    ceiling = float(result.get("limiter_effective_ceiling_db", -1.0))
    non_finite = bool(result.get("non_finite_output", True))
    return (
        bool(
            result.get("simulation_backend") == "rust"
            and np.isfinite(true_peak)
            and true_peak <= ceiling + 0.15
            and not non_finite
        ),
        {
            "simulation_backend": str(
                result.get("simulation_backend", "unavailable")
            ),
            "output_true_peak_db": true_peak,
            "limiter_effective_ceiling_db": ceiling,
            "non_finite_output": non_finite,
        },
    )


def _settings_summary(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep tracked evidence concise without discarding the fitted result."""
    keys = (
        "band_freqs",
        "band_gains",
        "band_qs",
        "active_band_count",
        "recommendation_status",
        "apply_recommended",
        "abstention_reasons",
        "capture_confidence",
        "eq_confidence",
        "validation_confidence",
        "target_profile",
        "target_mode",
        "smoothing_strength",
        "measurement_coverage",
        "measurement_phonetic_coverage",
        "measurement_effective_blocks",
        "measurement_vad_backend",
        "cross_take_confidence_available",
        "cross_take_band_confidences",
        "cross_take_gain_feasibility_scale",
        "cross_take_abstention_threshold",
    )
    return {key: settings.get(key) for key in keys if key in settings}


def _run_candidate(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate: int,
    *,
    use_cross_take: bool,
    first_vad: np.ndarray | None,
    second_vad: np.ndarray | None,
) -> tuple[dict[str, Any] | None, float, str | None]:
    started = time.perf_counter()
    try:
        if use_cross_take:
            settings = _analyze_cross_take_candidate(
                first,
                second,
                sample_rate,
                first_vad=first_vad,
                second_vad=second_vad,
            )
        else:
            settings, _validation = analyze_auto_eq(
                first,
                sample_rate,
                TARGET_PRESET,
                vad_probabilities=first_vad,
            )
        return settings, time.perf_counter() - started, None
    except Exception as error:
        return None, time.perf_counter() - started, str(error)


def _analyze_cross_take_candidate(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate: int,
    *,
    first_vad: np.ndarray | None,
    second_vad: np.ndarray | None,
) -> dict[str, Any]:
    """Run the rejected cross-take candidate outside the product pipeline."""
    first_spectrum = analyze_voice_spectrum(
        first,
        sample_rate,
        vad_probabilities=first_vad,
    )
    second_spectrum = analyze_voice_spectrum(
        second,
        sample_rate,
        vad_probabilities=second_vad,
    )
    first_smoothed = smooth_spectrum_perceptual(
        first_spectrum.freqs,
        first_spectrum.median_spectrum_db,
        strength="conservative",
    )
    second_smoothed = smooth_spectrum_perceptual(
        second_spectrum.freqs,
        second_spectrum.median_spectrum_db,
        strength="conservative",
    )
    evidence = cross_take_evidence(
        first_spectrum.freqs,
        first_smoothed,
        first_spectrum.measurement_uncertainty_db,
        first_spectrum.phonetic_coverage,
        second_spectrum.freqs,
        second_smoothed,
        second_spectrum.measurement_uncertainty_db,
        second_spectrum.phonetic_coverage,
    )
    target_db = get_target_curve(
        first_spectrum.freqs,
        TARGET_PRESET,
        measured_db=first_smoothed,
        target_mode="adaptive",
    )
    target_profile = (
        f"{TARGET_PRESET}:adaptive"
        if not first_spectrum.used_single_spectrum_fallback
        else f"{TARGET_PRESET}:adaptive:fallback"
    )
    settings = calculate_eq_bands(
        first_spectrum.freqs,
        first_smoothed,
        target_db,
        spectral_repeatability=first_spectrum.spectral_repeatability,
        spectral_uncertainty_db=first_spectrum.measurement_uncertainty_db,
        cross_take_confidence=evidence.confidence,
        phonetic_coverage=first_spectrum.phonetic_coverage,
        voiced_window_ratio=first_spectrum.voiced_window_ratio,
        analysis_confidence=first_spectrum.residual_confidence,
        global_snr_db=first_spectrum.snr_db,
        spectral_snr_db=first_spectrum.spectral_snr_db,
        noise_reference_source=first_spectrum.noise_reference_source,
        noise_reference_quality=1.0,
        noise_reference_status="usable",
        noise_reference_reasons=None,
        target_profile=target_profile,
        used_spectrum_fallback=first_spectrum.used_single_spectrum_fallback,
        smoothing_strength="conservative",
        tilt_policy="preserve",
    )
    settings["target_mode"] = "adaptive"
    settings["measurement_coverage"] = first_spectrum.measurement_coverage
    settings["measurement_outlier_rejection_ratio"] = (
        first_spectrum.outlier_rejection_ratio
    )
    settings["measurement_phonetic_coverage"] = (
        first_spectrum.phonetic_coverage
    )
    settings["measurement_effective_blocks"] = (
        first_spectrum.effective_measurement_blocks
    )
    settings["measurement_vad_backend"] = (
        "silero" if first_spectrum.vad_probability_used else "provided"
    )
    settings["measurement_vad_active_window_ratio"] = (
        first_spectrum.vad_active_window_ratio
    )
    settings["measurement_noise_reference_source"] = (
        first_spectrum.noise_reference_source
    )
    settings["measurement_noise_reference_quality"] = 1.0
    settings["measurement_noise_reference_status"] = "usable"
    settings["cross_take_evidence"] = evidence.diagnostics()
    settings = apply_headroom_validation(first, sample_rate, settings)
    validation = validate_analysis(
        settings,
        first_smoothed,
        first_spectrum.freqs,
    )
    if not validation.passed:
        raise ValueError(validation.reason)
    return settings


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        row for row in rows
        if row["single"]["settings"] is not None
        and row["cross_take"]["settings"] is not None
    ]
    improvements_by_speaker: dict[str, list[float]] = {}
    for row in comparable:
        improvement = float(
            row["single"]["heldout_target_error_db"]
            - row["cross_take"]["heldout_target_error_db"]
        )
        improvements_by_speaker.setdefault(str(row["speaker"]), []).append(
            improvement
        )
    speaker_improvements = {
        speaker: float(np.median(values))
        for speaker, values in sorted(improvements_by_speaker.items())
    }
    improvements = list(speaker_improvements.values())
    single_runtimes = [
        float(row["single"]["runtime_seconds"]) for row in rows
    ]
    cross_runtimes = [
        float(row["cross_take"]["runtime_seconds"]) for row in rows
    ]
    single_p95 = float(np.percentile(single_runtimes, 95)) if rows else 0.0
    cross_p95 = float(np.percentile(cross_runtimes, 95)) if rows else 0.0
    return {
        "pair_count": len(rows),
        "comparable_pair_count": len(comparable),
        "comparable_speaker_count": len(speaker_improvements),
        "speaker_median_heldout_improvements_db": speaker_improvements,
        "median_heldout_improvement_db": (
            float(np.median(improvements)) if improvements else None
        ),
        "p10_heldout_improvement_db": (
            float(np.percentile(improvements, 10)) if improvements else None
        ),
        "single_retry_rate": (
            float(np.mean([row["single"]["settings"] is None for row in rows]))
            if rows
            else 1.0
        ),
        "cross_take_retry_rate": (
            float(np.mean([row["cross_take"]["settings"] is None for row in rows]))
            if rows
            else 1.0
        ),
        "single_abstention_rate": (
            float(
                np.mean(
                    [
                        row["single"].get("recommendation_status") == "abstain"
                        for row in rows
                        if row["single"]["settings"] is not None
                    ]
                )
            )
            if any(row["single"]["settings"] is not None for row in rows)
            else 1.0
        ),
        "cross_take_abstention_rate": (
            float(
                np.mean(
                    [
                        row["cross_take"].get("recommendation_status")
                        == "abstain"
                        for row in rows
                        if row["cross_take"]["settings"] is not None
                    ]
                )
            )
            if any(row["cross_take"]["settings"] is not None for row in rows)
            else 1.0
        ),
        "cross_take_headroom_safe_rate": (
            float(
                np.mean(
                    [
                        bool(row["cross_take"].get("headroom_safe"))
                        for row in rows
                        if row["cross_take"]["settings"] is not None
                    ]
                )
            )
            if any(row["cross_take"]["settings"] is not None for row in rows)
            else 0.0
        ),
        "single_runtime_p95_seconds": single_p95,
        "cross_take_runtime_p95_seconds": cross_p95,
        "runtime_p95_ratio": cross_p95 / max(single_p95, 1e-9),
    }


def _gate(aggregate: dict[str, Any]) -> dict[str, bool]:
    median = aggregate["median_heldout_improvement_db"]
    p10 = aggregate["p10_heldout_improvement_db"]
    return {
        "enough_comparable_pairs": aggregate["comparable_pair_count"]
        >= GATES["min_comparable_test_pairs"],
        "enough_comparable_speakers": aggregate["comparable_speaker_count"]
        >= GATES["min_comparable_test_speakers"],
        "median_heldout_noninferior": median is not None
        and median >= GATES["min_median_heldout_improvement_db"],
        "lower_decile_heldout_noninferior": p10 is not None
        and p10 >= GATES["min_p10_heldout_improvement_db"],
        "retry_rate": aggregate["cross_take_retry_rate"]
        <= GATES["max_cross_take_retry_rate"],
        "abstention_rate": aggregate["cross_take_abstention_rate"]
        <= aggregate["single_abstention_rate"]
        + GATES["max_abstention_rate_increase"],
        "headroom": aggregate["cross_take_headroom_safe_rate"]
        >= GATES["min_headroom_safe_rate"],
        "runtime_ratio": aggregate["runtime_p95_ratio"]
        <= GATES["max_runtime_p95_ratio"],
        "runtime_absolute": aggregate["cross_take_runtime_p95_seconds"]
        <= GATES["max_runtime_p95_seconds"],
    }


def evaluate(corpus_root: Path) -> dict[str, Any]:
    corpus_root = corpus_root.resolve(strict=True)
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest["speaker_disjoint_splits"]
    evaluated_speakers = set(splits["validation"]) | set(splits["test"])
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "test": [],
    }
    grouped_pairs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pair in manifest["pairs"]:
        speaker = str(pair["speaker"])
        statement_id = str(pair["statement_id"])
        grouped_pairs.setdefault(speaker, {}).setdefault(
            statement_id, []
        ).append(pair)
    cache: dict[Path, tuple[int, np.ndarray]] = {}
    for speaker, statements in sorted(grouped_pairs.items()):
        if speaker not in evaluated_speakers:
            continue
        split = "validation" if speaker in splits["validation"] else "test"
        for tuning_statement_id, heldout_statement_id in _statement_folds(
            set(statements)
        ):
            tuning_pairs = statements[tuning_statement_id]
            heldout_pairs = statements[heldout_statement_id]
            sample_rate, first, first_sources = _concat_statement_takes(
                corpus_root, tuning_pairs, ("01",), cache
            )
            second_rate, second, second_sources = _concat_statement_takes(
                corpus_root, tuning_pairs, ("02",), cache
            )
            heldout_rate, heldout, heldout_sources = _concat_statement_takes(
                corpus_root, heldout_pairs, ("01", "02"), cache
            )
            if second_rate != sample_rate or heldout_rate != sample_rate:
                raise RuntimeError(f"{speaker} statement sample rates differ")
            if set(first_sources + second_sources) & set(heldout_sources):
                raise RuntimeError("Tuning and held-out source audio overlap")
            first_vad, first_vad_backend = analyze_offline_vad(
                first, sample_rate
            )
            second_vad, second_vad_backend = analyze_offline_vad(
                second, sample_rate
            )
            heldout_vad, heldout_vad_backend = analyze_offline_vad(
                heldout, sample_rate
            )
            heldout_spectrum = analyze_voice_spectrum(
                heldout,
                sample_rate,
                vad_probabilities=heldout_vad,
            )
            row: dict[str, Any] = {
                "id": (
                    f"{speaker}-tune-{tuning_statement_id}"
                    f"-holdout-{heldout_statement_id}"
                ),
                "speaker": speaker,
                "speaker_sex": tuning_pairs[0]["speaker_sex"],
                "tuning_statement_id": tuning_statement_id,
                "heldout_statement_id": heldout_statement_id,
                "tuning_source_pair_ids": sorted(
                    str(pair["id"]) for pair in tuning_pairs
                ),
                "heldout_source_pair_ids": sorted(
                    str(pair["id"]) for pair in heldout_pairs
                ),
                "tuning_first_source_take_ids": first_sources,
                "tuning_second_source_take_ids": second_sources,
                "heldout_source_take_ids": heldout_sources,
                "tuning_first_duration_seconds": first.size / sample_rate,
                "tuning_second_duration_seconds": second.size / sample_rate,
                "heldout_duration_seconds": heldout.size / sample_rate,
                "split": split,
                "vad_backends": {
                    "tuning_first": first_vad_backend,
                    "tuning_second": second_vad_backend,
                    "heldout": heldout_vad_backend,
                },
            }
            for name, use_cross_take in (
                ("single", False),
                ("cross_take", True),
            ):
                settings, runtime, error = _run_candidate(
                    first,
                    second,
                    sample_rate,
                    use_cross_take=use_cross_take,
                    first_vad=first_vad,
                    second_vad=second_vad,
                )
                candidate: dict[str, Any] = {
                    "runtime_seconds": runtime,
                    "error": error,
                    "settings": (
                        _settings_summary(settings)
                        if settings is not None
                        else None
                    ),
                }
                if settings is not None:
                    candidate["heldout_target_error_db"] = (
                        _heldout_target_error(heldout_spectrum, settings)
                    )
                    candidate["recommendation_status"] = settings[
                        "recommendation_status"
                    ]
                    safe, headroom = _headroom_safe(
                        heldout,
                        sample_rate,
                        settings,
                    )
                    candidate["headroom_safe"] = safe
                    candidate["headroom"] = headroom
                    candidate["active_band_count"] = settings[
                        "active_band_count"
                    ]
                    if use_cross_take:
                        candidate["cross_take_evidence"] = settings[
                            "cross_take_evidence"
                        ]
                row[name] = candidate
            rows_by_split[split].append(row)

    aggregates = {
        split: _aggregate(rows)
        for split, rows in rows_by_split.items()
    }
    validation_checks = _gate(aggregates["validation"])
    test_checks = _gate(aggregates["test"])
    objective_passed = all(validation_checks.values()) and all(
        test_checks.values()
    )
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT
        / "python/mic_eq/analysis/auto_eq_parts/cross_take.py",
        REPO_ROOT
        / "python/mic_eq/analysis/auto_eq_parts/optimizer.py",
        REPO_ROOT / "python/mic_eq/analysis/wav_io.py",
    )
    asset_paths = (
        manifest_path,
        REPO_ROOT / "models/silero_vad.onnx",
    )
    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "objective-passed-integration-required"
            if objective_passed
            else "rejected"
        ),
        "decision": {
            "objective_candidate_passed": objective_passed,
            "cross_take_candidate_retained": False,
            "production_call_path_absent": True,
            "evaluation_only_optimizer_hook_present": True,
            "incumbent_retained": True,
            "audible_change": True,
            "reason": (
                "The objective candidate passed independent lexical holdout, "
                "but the evaluation-only hook is not a production call path; "
                "integration and exact-chain qualification would still be required."
                if objective_passed
                else (
                    "The evaluation-only cross-take candidate failed predefined "
                    "independent lexical-holdout gates; no product call path "
                    "accepts repeated-take input and the incumbent remains."
                )
            ),
        },
        "configuration": {
            "target_preset": TARGET_PRESET,
            "evaluated_splits": ["validation", "test"],
            "precision_and_phonetic_coverage_kept_separate": True,
            "lexical_holdout": True,
            "speaker_level_lower_tail": True,
        },
        "gates": GATES,
        "checks": {
            "validation": validation_checks,
            "test": test_checks,
        },
        "aggregates": aggregates,
        "rows": rows_by_split,
        "provenance": {
            "source_hashes": {
                (
                    path.relative_to(REPO_ROOT).as_posix()
                    if path.is_relative_to(REPO_ROOT)
                    else path.name
                ): _source_sha256(path)
                for path in source_paths
            },
            "asset_hashes": {
                (
                    path.relative_to(REPO_ROOT).as_posix()
                    if path.is_relative_to(REPO_ROOT)
                    else path.name
                ): _sha256(path)
                for path in asset_paths
            },
            "corpus_license": manifest["license"],
            "corpus_redistribution": manifest["redistribution"],
            "manifest_audio_validated": True,
            "validated_input_file_count": len(cache),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "limitations": [
            *manifest["limitations"],
            "Objective held-out target error and safety gates do not establish production-path correctness when the candidate remains evaluation-only.",
            "Validation and test actors are speaker-disjoint from the unused training split.",
            "Each fold tunes on one fixed statement and scores both takes of the other statement; speaker medians prevent the two folds from being treated as independent speakers.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional full per-pair report; the tracked report stays compact.",
    )
    args = parser.parse_args()
    report = evaluate(args.corpus)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.details_output is not None:
        details_output = args.details_output.resolve()
        details_output.parent.mkdir(parents=True, exist_ok=True)
        details_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report.pop("rows", None)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Cross-take Auto-EQ evaluation status={report['status']} "
        f"test_pairs={report['aggregates']['test']['pair_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
