"""Calibrate Auto-EQ confidence cutoffs against paired real-speech stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from mic_eq import analyze_vad_probabilities
from mic_eq.analysis import auto_eq
from mic_eq.analysis.auto_eq_parts.constants import (
    GLOBAL_CAPTURE_CONFIDENCE_THRESHOLD,
    LOCAL_ABSTENTION_CONFIDENCE_THRESHOLD,
    REDUCED_RECOMMENDATION_CONFIDENCE_THRESHOLD,
)
from mic_eq.analysis.spectrum import (
    analyze_voice_spectrum,
    smooth_spectrum_perceptual,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "models" / "dpdfnet_eval_subset"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "auto-eq-confidence-calibration.json"
THRESHOLD_GRID = np.round(np.arange(0.20, 0.801, 0.025), 3)
EVALUATION_SNRS_DB = (0.0, 10.0, 30.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _read_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    audio = np.asarray(raw)
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        midpoint = float(info.max + 1) / 2.0
        audio = (audio.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(int(info.min)), int(info.max)))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    converted = np.asarray(audio, dtype=np.float32)
    if converted.ndim != 1 or not np.all(np.isfinite(converted)):
        raise ValueError(f"{path.name} must contain finite mono/stereo PCM")
    return int(sample_rate), converted


def _verified_path(corpus_root: Path, entry: dict[str, Any]) -> Path:
    relative = Path(str(entry["path"]))
    path = (corpus_root / relative).resolve(strict=True)
    if not path.is_relative_to(corpus_root):
        raise ValueError(f"Corpus path escapes its root: {relative}")
    if path.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"Corpus size mismatch for {relative}")
    if _sha256(path) != str(entry["sha256"]):
        raise ValueError(f"Corpus hash mismatch for {relative}")
    return path


def _pairs(corpus_root: Path) -> list[tuple[Path, Path, float]]:
    corpus_root = corpus_root.resolve(strict=True)
    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    entries = {str(entry["path"]): entry for entry in manifest["files"]}
    pairs: list[tuple[Path, Path, float]] = []
    clean_entries = sorted(
        (
            entry
            for entry in manifest["files"]
            if str(entry.get("model_name")) == "Clean"
        ),
        key=lambda entry: str(entry["path"]),
    )
    for clean_entry in clean_entries:
        clean_relative = str(clean_entry["path"])
        noisy_relative = clean_relative.replace("Clean/", "Noisy/", 1).replace(
            "_clean.wav", "_noisy.wav"
        )
        noisy_entry = entries.get(noisy_relative)
        if noisy_entry is None:
            raise ValueError(f"Manifest lacks paired noisy file for {clean_relative}")
        clean = _verified_path(corpus_root, clean_entry)
        noisy = _verified_path(corpus_root, noisy_entry)
        clean_rate, _clean_audio = _read_mono(clean)
        noisy_rate, _noisy_audio = _read_mono(noisy)
        expected_rate = int(clean_entry["sample_rate"])
        if clean_rate != expected_rate or noisy_rate != expected_rate:
            raise ValueError(f"Manifest sample-rate mismatch for {clean_relative}")
        pairs.append((clean, noisy, float(noisy_entry["snr_db"])))
    if not pairs:
        raise RuntimeError(f"No clean/noisy pairs found under {corpus_root}")
    return pairs


def _render_at_snr(
    clean: np.ndarray,
    residual_noise: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    clean_rms = float(np.sqrt(np.mean(np.square(clean, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(residual_noise, dtype=np.float64))))
    if clean_rms <= 1e-9 or noise_rms <= 1e-9:
        raise ValueError("Paired source has insufficient clean or noise energy")
    scale = clean_rms / (noise_rms * (10.0 ** (target_snr_db / 20.0)))
    scaled_noise = np.asarray(residual_noise * scale, dtype=np.float32)
    rendered = np.asarray(clean + scaled_noise, dtype=np.float32)
    measured_noise_rms = float(
        np.sqrt(np.mean(np.square(scaled_noise, dtype=np.float64)))
    )
    measured_snr = float(20.0 * np.log10(clean_rms / measured_noise_rms))
    if not np.all(np.isfinite(rendered)):
        raise ValueError("SNR render produced non-finite audio")
    return rendered, scaled_noise, measured_snr, float(scale)


def _solve(
    audio: np.ndarray,
    noise: np.ndarray,
    sample_rate: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    probabilities = np.asarray(
        analyze_vad_probabilities(audio, sample_rate, 0.48), dtype=np.float64
    )
    spectrum = analyze_voice_spectrum(
        audio,
        sample_rate,
        vad_probabilities=probabilities,
        noise_audio=noise,
    )
    measured = smooth_spectrum_perceptual(
        spectrum.freqs,
        spectrum.median_spectrum_db,
        strength="conservative",
    )
    target = auto_eq.get_target_curve(
        spectrum.freqs,
        "broadcast",
        measured_db=measured,
        target_mode="static",
    )
    result = auto_eq.calculate_eq_bands(
        spectrum.freqs,
        measured,
        target,
        spectral_repeatability=spectrum.spectral_repeatability,
        spectral_uncertainty_db=spectrum.measurement_uncertainty_db,
        phonetic_coverage=spectrum.phonetic_coverage,
        voiced_window_ratio=spectrum.voiced_window_ratio,
        analysis_confidence=spectrum.residual_confidence,
        global_snr_db=spectrum.snr_db,
        spectral_snr_db=spectrum.spectral_snr_db,
        noise_reference_source=spectrum.noise_reference_source,
        noise_reference_quality=1.0,
        noise_reference_status="usable",
        target_profile="broadcast:static",
        used_spectrum_fallback=spectrum.used_single_spectrum_fallback,
        smoothing_strength="conservative",
        tilt_policy="preserve",
    )
    result["measurement_confidence"] = spectrum.residual_confidence
    result["phonetic_coverage"] = spectrum.phonetic_coverage
    result["effective_blocks"] = spectrum.effective_measurement_blocks
    return result, time.perf_counter() - started


def _response(result: dict[str, Any], frequencies: np.ndarray) -> np.ndarray:
    return auto_eq._predict_eq_response(
        frequencies,
        result["pre_abstention_band_gains"],
        result["band_qs"],
        result["band_freqs"],
    )


def _case(
    clean_path: Path,
    noisy_path: Path,
    clip_seconds: float,
    *,
    target_snr_db: float,
    source_snr_db: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_rate, clean = _read_mono(clean_path)
    noisy_rate, noisy = _read_mono(noisy_path)
    if clean_rate != noisy_rate:
        raise RuntimeError(f"Sample-rate mismatch for {clean_path.name}")
    count = min(clean.size, noisy.size)
    clip_samples = int(round(clip_seconds * clean_rate))
    if count < clip_samples:
        raise RuntimeError(
            f"{clean_path.name} is shorter than {clip_seconds:.1f} seconds"
        )
    offset = min(15 * clean_rate, max(0, (count - clip_samples) // 2))
    clean = clean[offset : offset + clip_samples]
    noisy = noisy[offset : offset + clip_samples]
    residual_noise = noisy - clean
    rendered, scaled_noise, measured_snr_db, noise_scale = _render_at_snr(
        clean,
        residual_noise,
        target_snr_db,
    )
    clean_result, clean_seconds = _solve(clean, scaled_noise, clean_rate)
    noisy_result, noisy_seconds = _solve(rendered, scaled_noise, noisy_rate)
    grid = np.geomspace(80.0, min(8_000.0, clean_rate * 0.49), 256)
    clean_response = _response(clean_result, grid)
    noisy_response = _response(noisy_result, grid)
    response_error_db = float(
        np.sqrt(np.mean(np.square(clean_response - noisy_response)))
    )
    active_response = np.abs(clean_response) >= 0.25
    response_sign_agreement = (
        float(
            np.mean(
                np.sign(clean_response[active_response])
                == np.sign(noisy_response[active_response])
            )
        )
        if np.any(active_response)
        else 1.0
    )
    stable_capture = response_error_db <= 1.5 and response_sign_agreement >= 0.80
    language = clean_path.name.split("_", 1)[0]
    capture_row = {
        "id": (
            clean_path.name.removesuffix("_mixture_clean.wav")
            + f"-render-snr{target_snr_db:g}"
        ),
        "language": language,
        "split": "validation"
        if language in {"german", "korean", "spanish", "turkish"}
        else "train",
        "clean_path": _relative(clean_path),
        "noisy_path": _relative(noisy_path),
        "clean_sha256": _sha256(clean_path),
        "noisy_sha256": _sha256(noisy_path),
        "source_offset_samples": offset,
        "duration_seconds": clip_seconds,
        "source_fixture_snr_db": source_snr_db,
        "target_render_snr_db": target_snr_db,
        "measured_render_snr_db": measured_snr_db,
        "residual_noise_scale": noise_scale,
        "rendered_audio_sha256": hashlib.sha256(rendered.tobytes()).hexdigest(),
        "response_error_db": response_error_db,
        "response_sign_agreement": response_sign_agreement,
        "stable_capture": stable_capture,
        "measurement_confidence": float(noisy_result["measurement_confidence"]),
        "overall_confidence": float(noisy_result["analysis_confidence"]),
        "phonetic_coverage": float(noisy_result["phonetic_coverage"]),
        "effective_blocks": float(noisy_result["effective_blocks"]),
        "candidate_runtime_seconds": noisy_seconds,
        "reference_runtime_seconds": clean_seconds,
    }
    band_rows: list[dict[str, Any]] = []
    noisy_centers = np.asarray(noisy_result["band_freqs"], dtype=float)
    noisy_gains = np.asarray(noisy_result["pre_abstention_band_gains"], dtype=float)
    confidences = np.asarray(noisy_result["band_confidences"], dtype=float)
    clean_at_centers = np.interp(noisy_centers, grid, clean_response)
    noisy_at_centers = np.interp(noisy_centers, grid, noisy_response)
    for index, (center, gain, confidence) in enumerate(
        zip(noisy_centers, noisy_gains, confidences, strict=True)
    ):
        if abs(gain) < 0.25:
            continue
        difference = abs(float(clean_at_centers[index] - noisy_at_centers[index]))
        sign_agrees = abs(clean_at_centers[index]) < 0.25 or np.sign(
            clean_at_centers[index]
        ) == np.sign(noisy_at_centers[index])
        band_rows.append(
            {
                "capture_id": capture_row["id"],
                "language": language,
                "split": capture_row["split"],
                "target_render_snr_db": target_snr_db,
                "band_index": index,
                "center_hz": float(center),
                "pre_abstention_gain_db": float(gain),
                "confidence": float(confidence),
                "response_difference_db": difference,
                "supported": bool(difference <= 1.5 and sign_agrees),
            }
        )
    return capture_row, band_rows


def _classification(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = scores >= threshold
    true_positive = int(np.count_nonzero(predicted & labels))
    false_positive = int(np.count_nonzero(predicted & ~labels))
    false_negative = int(np.count_nonzero(~predicted & labels))
    true_negative = int(np.count_nonzero(~predicted & ~labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _calibrate(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    label_key: str,
    current_threshold: float,
) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    train_scores = np.asarray([row[score_key] for row in train], dtype=float)
    train_labels = np.asarray([row[label_key] for row in train], dtype=bool)
    candidates = [
        _classification(train_scores, train_labels, float(threshold))
        for threshold in THRESHOLD_GRID
    ]
    best = max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["precision"],
            -abs(float(item["threshold"]) - current_threshold),
        ),
    )
    validation_scores = np.asarray([row[score_key] for row in validation], dtype=float)
    validation_labels = np.asarray([row[label_key] for row in validation], dtype=bool)
    current_validation = _classification(
        validation_scores, validation_labels, current_threshold
    )
    candidate_validation = _classification(
        validation_scores, validation_labels, float(best["threshold"])
    )
    enough_classes = bool(
        np.count_nonzero(validation_labels) >= 3
        and np.count_nonzero(~validation_labels) >= 3
    )
    retain_candidate = bool(
        enough_classes
        and candidate_validation["f1"] >= current_validation["f1"] + 0.03
        and candidate_validation["precision"] >= current_validation["precision"] - 0.02
    )
    selected = float(best["threshold"]) if retain_candidate else current_threshold
    return {
        "current_threshold": current_threshold,
        "training_best_threshold": float(best["threshold"]),
        "selected_threshold": selected,
        "selection": "candidate" if retain_candidate else "current",
        "minimum_validation_class_count_met": enough_classes,
        "current_validation": current_validation,
        "candidate_validation": candidate_validation,
        "training_candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clip-seconds", type=float, default=15.0)
    args = parser.parse_args()
    captures: list[dict[str, Any]] = []
    bands: list[dict[str, Any]] = []
    for clean, noisy, source_snr_db in _pairs(args.corpus_root):
        for target_snr_db in EVALUATION_SNRS_DB:
            capture, capture_bands = _case(
                clean,
                noisy,
                args.clip_seconds,
                target_snr_db=target_snr_db,
                source_snr_db=source_snr_db,
            )
            captures.append(capture)
            bands.extend(capture_bands)

    decisions = {
        "local_abstention": _calibrate(
            bands,
            score_key="confidence",
            label_key="supported",
            current_threshold=LOCAL_ABSTENTION_CONFIDENCE_THRESHOLD,
        ),
        "global_capture_guard": _calibrate(
            captures,
            score_key="measurement_confidence",
            label_key="stable_capture",
            current_threshold=GLOBAL_CAPTURE_CONFIDENCE_THRESHOLD,
        ),
        "reduced_recommendation": _calibrate(
            captures,
            score_key="overall_confidence",
            label_key="stable_capture",
            current_threshold=REDUCED_RECOMMENDATION_CONFIDENCE_THRESHOLD,
        ),
    }
    manifest = json.loads(
        (REPO_ROOT / "release-assets.json").read_text(encoding="utf-8")
    )
    silero = next(
        asset
        for asset in manifest["assets"]
        if asset["path"] == "models/silero_vad.onnx"
    )
    runtimes = np.asarray(
        [row["candidate_runtime_seconds"] for row in captures], dtype=float
    )
    source_paths = (
        "python/tools/calibrate_auto_eq_confidence.py",
        "python/mic_eq/analysis/auto_eq.py",
        "python/mic_eq/analysis/auto_eq_parts/constants.py",
        "python/mic_eq/analysis/auto_eq_parts/optimizer.py",
        "python/mic_eq/analysis/auto_eq_parts/pipeline.py",
        "python/mic_eq/analysis/spectrum.py",
    )
    source_hashes = {
        path: _source_sha256(REPO_ROOT / path) for path in source_paths
    }
    corpus_manifest = args.corpus_root / "manifest.json"
    report = {
        "schema_version": 3,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Corpus-derived calibration of remaining Auto-EQ confidence cutoffs",
        "decision": decisions,
        "capture_count": len(captures),
        "active_band_observation_count": len(bands),
        "captures": captures,
        "band_observations": bands,
        "source_sha256": source_hashes,
        "evaluation_contract": {
            "configuration": {
                "clip_seconds": args.clip_seconds,
                "sample_rate": 16_000,
                "render_snrs_db": list(EVALUATION_SNRS_DB),
                "target": "broadcast:static",
                "smoothing": "conservative",
                "tilt_policy": "preserve",
                "validation_languages": ["german", "korean", "spanish", "turkish"],
                "stability_gate": {
                    "capture_response_rms_error_db_max": 1.5,
                    "capture_sign_agreement_min": 0.80,
                    "band_response_difference_db_max": 1.5,
                },
                "threshold_selection": (
                    "Training F1 maximum; retain candidate only when validation F1 "
                    "improves by >=0.03, precision regret <=0.02, and both validation "
                    "classes have at least three observations."
                ),
            },
            "asset_hashes": {
                silero["path"]: silero["sha256"],
                _relative(corpus_manifest): _sha256(corpus_manifest),
            },
            "runtime": {
                "max_p99_frame_seconds": float(np.percentile(runtimes, 99.0)),
                "median_case_seconds": float(np.median(runtimes)),
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "latency": {
                "status": "offline_analysis",
                "additional_realtime_audio_latency_samples": 0,
            },
            "clean_preservation": {
                "stable_capture_count": int(
                    np.count_nonzero([row["stable_capture"] for row in captures])
                ),
                "capture_count": len(captures),
                "definition": "paired clean/noisy correction response agrees within predefined component gates",
            },
        },
        "limitations": [
            "Paired SNR renders test condition stability, not repeated-take agreement.",
            "The corpus is simulated 16 kHz multilingual speech, not native product-like 48 kHz close-mic capture.",
            "Between-take agreement remains unavailable and is not inferred from these pairs.",
            "Multiple deterministic SNR renders from one source are correlated; language-disjoint validation prevents source overlap across threshold fitting and validation but does not create more speakers.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                name: decision["selected_threshold"]
                for name, decision in decisions.items()
            },
            sort_keys=True,
        )
    )
    print(f"Wrote {_relative(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
