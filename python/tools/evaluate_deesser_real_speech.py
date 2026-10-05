"""Measure the de-esser on real speech: setup-time detection and the processed audio.

Simulated users (``evaluate_voice_setup.py``) are heard through three
microphones: ordinary (unchanged), bright (a broad +6 dB shelf from 4 kHz, an
EQ matter, not sibilance) and harsh (a +8 dB resonance between 5.5 and 8 kHz
that makes sibilants piercing). Only harsh should enable the de-esser.

Detection: Voice Setup's own setup-time path (noise reference, VAD-masked
frame evidence, the fitted soft-fusion model) decides from the setup capture.
Audio: factory and recommended arms force the de-esser on; a third arm follows
the recommendation's enabled flag. The historical 5-9 kHz changes are retained,
alongside absolute residual error against uncolored clean speech, 250-2000 Hz
voice-body change and ESTOI. Render timing is mean throughput, not live p99.
Use separate invocations with --build for matched source/native comparisons.
Sibilant frames are labeled on the uncolored clean speech, where fricatives
carry most of their energy above 4 kHz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_voice_setup as harness
from scipy.signal import lfilter

from evaluate_gate_corpus import FRAME, cluster_ci, estoi

CONDITIONS = ("ordinary", "bright", "harsh")
USERS = 40
ARMS = ("recommended", "factory", "recommended_enabled")
METRICS = ("sibilant_db", "voiced_db", "sibilant_residual_db", "body_db", "estoi")


def microphone(condition: str, rng: np.random.Generator) -> list[tuple[str, float, float, float]]:
    if condition == "bright":
        return [("high_shelf", 4_000.0, 6.0, 0.707)]
    if condition == "harsh":
        return [("peak", float(rng.uniform(5_500.0, 8_000.0)), 8.0, float(rng.uniform(1.5, 3.0)))]
    return []


def _apply(audio: np.ndarray, sections: list[tuple[str, float, float, float]]) -> np.ndarray:
    """Filter audio through (filter type, center Hz, gain dB, Q) biquad sections."""
    from mic_eq.analysis.auto_eq_parts.response import _biquad_coefficients

    out = np.asarray(audio, np.float64)
    for kind, center, gain, q in sections:
        b0, b1, b2, a0, a1, a2 = _biquad_coefficients(gain, q, center, kind, harness.SAMPLE_RATE)
        out = np.asarray(lfilter([b0, b1, b2], [a0, a1, a2], out))
    return out


def _band_db(audio: np.ndarray, low: float, high: float) -> np.ndarray:
    count = audio.size // FRAME
    frames = audio[: count * FRAME].reshape(count, FRAME) * np.hanning(FRAME)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1.0 / harness.SAMPLE_RATE)
    band = (freqs >= low) & (freqs < high)
    return 10.0 * np.log10(spectrum[:, band].sum(axis=1) + 1e-20)


def sibilant_frames(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sibilant, other voiced) frame masks from clean speech."""
    high = _band_db(clean, 4_000.0, 10_000.0)
    total = _band_db(clean, 80.0, 10_000.0)
    loud = total > np.max(total) - 35.0
    sibilant = loud & (high - total > -3.0)
    return sibilant, loud & ~sibilant


def audio_metrics(capture: np.ndarray, output: np.ndarray, clean: np.ndarray,
                  sibilant: np.ndarray, voiced: np.ndarray) -> dict[str, Any]:
    """Score a complete render; missing frame support remains explicit."""
    valid = bool(capture.ndim == output.ndim == clean.ndim == 1
                 and capture.shape == output.shape == clean.shape
                 and capture.size >= FRAME
                 and all(np.isfinite(audio).all() for audio in (capture, output, clean)))
    result: dict[str, Any] = {name: None for name in METRICS}
    result["finite_output"] = valid
    if not valid:
        return result
    count = capture.size // FRAME
    if sibilant.shape != (count,) or voiced.shape != (count,):
        raise ValueError("clean frame masks must cover the complete render")
    before = _band_db(capture, 5_000.0, 9_000.0)
    after = _band_db(output, 5_000.0, 9_000.0)
    if sibilant.any():
        truth = _band_db(clean, 5_000.0, 9_000.0)
        result["sibilant_db"] = float(np.mean((after - before)[sibilant]))
        result["sibilant_residual_db"] = float(np.mean(np.abs(after - truth)[sibilant]))
    if voiced.any():
        result["voiced_db"] = float(np.mean((after - before)[voiced]))
        body_change = _band_db(output, 250.0, 2_000.0) - _band_db(capture, 250.0, 2_000.0)
        result["body_db"] = float(np.mean(body_change[voiced]))
    intelligibility = estoi(clean, output, harness.SAMPLE_RATE)
    result["estoi"] = intelligibility if np.isfinite(intelligibility) else None
    return result


def _recommendation(speech: np.ndarray, noise: np.ndarray) -> dict[str, Any]:
    from mic_eq.analysis import voice_setup as setup
    from mic_eq.analysis.noise_reference import analyze_noise_reference
    from mic_eq.analysis.spectrum import analyze_voice_spectrum, smooth_spectrum_perceptual
    from mic_eq.analysis.vad import analyze_offline_vad

    rate = harness.SAMPLE_RATE
    vad, _ = analyze_offline_vad(speech, rate)
    noise_vad, _ = analyze_offline_vad(noise, rate)
    reference = analyze_noise_reference(noise, speech, rate, noise_vad_probabilities=noise_vad,
                                        speech_vad_probabilities=vad)
    features = setup._vad_masked_speech_features(speech, rate, reference.conservative_noise_rms_db,
                                                 vad_probabilities=vad, noise_audio=noise)
    spectrum = analyze_voice_spectrum(speech, rate, vad_probabilities=vad, noise_audio=noise,
                                      noise_spectrum_override=(reference.frequencies,
                                                               reference.conservative_spectrum_db),
                                      noise_reference_source_override="validated_conservative")
    settings, diagnostics = setup._recommend_deesser_settings(
        freqs=spectrum.freqs, spectrum_db=smooth_spectrum_perceptual(spectrum.freqs, spectrum.median_spectrum_db),
        capture_confidence=1.0, noise_reference_quality=reference.quality_score,
        noise_reference_status=reference.status,
        robust_sibilance_excess_db=float(features["sibilance_excess_db"]),
        frame_evidence=features["deesser_frame_evidence"],
    )
    return settings | {"detection_probability": diagnostics["detection_probability"]}


def _render(capture: np.ndarray, settings: dict[str, Any], *, force_enabled: bool = True) -> np.ndarray:
    from mic_eq.config import EQ_FREQUENCIES
    from mic_eq.mic_eq_core import simulate_auto_eq_chain

    chain = {f"deesser_{key}": value for key, value in settings.items()
             if key in ("auto_enabled", "auto_amount", "low_cut_hz", "high_cut_hz",
                        "threshold_db", "ratio", "attack_ms", "release_ms", "max_reduction_db")}
    chain |= {"deesser_enabled": force_enabled or bool(settings["enabled"]),
              "compressor_enabled": False, "limiter_enabled": False,
              "return_output_audio": True}
    bands = [(float(f), 0.0, 1.0) for f in EQ_FREQUENCIES]
    result = simulate_auto_eq_chain(capture.astype(np.float32), float(harness.SAMPLE_RATE), bands, chain)
    return np.asarray(result["output_audio"], np.float64)


def measure(job: tuple[str, str, int, str]) -> dict[str, Any]:
    root_text, split_name, index, condition = job
    from mic_eq.config import Preset
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    root = Path(root_text)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    user = harness.user(root, manifest, split_name, index)
    sections = microphone(condition, np.random.default_rng(9_000_000 + 1_000 * harness.SPLITS.index(split_name) + index))

    def heard(audio: np.ndarray) -> np.ndarray:
        colored = _apply(audio, sections) if sections else np.asarray(audio, np.float64)
        return filtered_capture_for_analysis(colored.astype(np.float32), harness.SAMPLE_RATE)

    recommended = _recommendation(heard(user["setup_capture"]), heard(user["noise_capture"]))
    capture = heard(user["evaluation_capture"])
    clean = filtered_capture_for_analysis(user["evaluation_clean"], harness.SAMPLE_RATE).astype(np.float64)
    sibilant, voiced = sibilant_frames(user["evaluation_clean"].astype(np.float64))
    row: dict[str, Any] = {"split": split_name, "index": index, "speaker": user["speaker"],
                           "condition": condition, "enabled": bool(recommended["enabled"]),
                           "detection_probability": float(recommended["detection_probability"]),
                           "sibilant_frames": int(sibilant.sum()), "voiced_frames": int(voiced.sum())}
    for name, settings in (("recommended", recommended), ("factory", asdict(Preset().deesser)),
                           ("recommended_enabled", recommended)):
        started = time.perf_counter()
        output = _render(capture, settings, force_enabled=name != "recommended_enabled")
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        row.update({f"{name}_{metric}": value for metric, value in
                    audio_metrics(capture.astype(np.float64), output, clean, sibilant, voiced).items()})
        row[f"{name}_render_ms"] = elapsed_ms
        row[f"{name}_render_ms_per_10ms_audio"] = elapsed_ms / (capture.size / FRAME)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        clusters = np.array([r["speaker"] for r in subset])
        condition_summary: dict[str, Any] = {
            "users": len(subset),
            "enabled_rate": (cluster_ci(np.array([float(r["enabled"]) for r in subset]), clusters)
                             if subset else [None] * 3),
        }
        coverage = {}
        for arm in ARMS:
            condition_summary[f"{arm}_invalid_render_cases"] = sum(not r[f"{arm}_finite_output"] for r in subset)
            for metric in (*METRICS, "render_ms", "render_ms_per_10ms_audio"):
                key = f"{arm}_{metric}"
                values = np.array([r[key] for r in subset], dtype=float)
                coverage[key] = int(np.isfinite(values).sum())
                condition_summary[key] = (cluster_ci(values, clusters)
                                          if values.size and np.isfinite(values).all() else [None] * 3)
        condition_summary["valid_metric_cases"] = coverage
        summary[condition] = condition_summary
    return summary


def _configure_build(build: Path) -> None:
    package = (build / "python" / "mic_eq").resolve()
    sys.path.insert(0, str(package.parent))
    import mic_eq
    from mic_eq import mic_eq_core

    if (Path(str(mic_eq.__file__)).resolve().parent != package
            or Path(str(mic_eq_core.__file__)).resolve().parent != package):
        raise RuntimeError("Python package and native extension must come from the requested build")


def _measurement_identity(build: Path, root: Path) -> dict[str, Any]:
    package = build / "python" / "mic_eq"
    extensions = [p for p in package.glob("mic_eq_core*") if p.suffix in (".pyd", ".so")]
    if len(extensions) != 1:
        raise ValueError("requested build must contain exactly one native extension")
    sources = {
        "evaluator_source_sha256": (harness.REPO_ROOT, (
            "python/tools/evaluate_deesser_real_speech.py", "python/tools/evaluate_voice_setup.py",
            "python/tools/evaluate_gate_corpus.py", "python/tools/_eval_common.py")),
        "build_source_sha256": (build, (
            "rust-core/src/dsp/deesser.rs", "rust-core/src/audio/processor/python_api.rs",
            "rust-core/src/audio/processor/block_processor.rs", "python/mic_eq/analysis/voice_setup.py",
            "python/mic_eq/analysis/deesser_fusion.py", "python/mic_eq/analysis/noise_reference.py",
            "python/mic_eq/analysis/spectrum.py", "python/mic_eq/analysis/vad.py",
            "python/mic_eq/config_parts/settings.py", "python/mic_eq/ui/calibration_support.py")),
    }
    identity: dict[str, Any] = {
        "extension": extensions[0].name,
        "extension_sha256": hashlib.sha256(extensions[0].read_bytes()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        **{name: {relative: hashlib.sha256((base / relative).read_bytes().replace(b"\r\n", b"\n")
                                          .replace(b"\r", b"\n")).hexdigest() for relative in files}
           for name, (base, files) in sources.items()},
    }
    vad_path = os.environ.get("VAD_MODEL_PATH")
    if vad_path:
        identity["vad_sha256"] = hashlib.sha256(Path(vad_path).read_bytes()).hexdigest()
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=harness.DEFAULT_ROOT)
    parser.add_argument("--build", type=Path, default=harness.REPO_ROOT)
    parser.add_argument("--split", choices=harness.SPLITS, default="fit")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--users", type=int, default=USERS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path)
    args = parser.parse_args(argv)
    if args.first < 0 or args.users < 1 or not 1 <= args.workers <= 4:
        parser.error("--first must be nonnegative, --users positive, and --workers between 1 and 4")
    if args.rows_output is not None and args.rows_output.resolve() == args.output.resolve():
        parser.error("--output and --rows-output must be different files")
    args.root, args.build = args.root.resolve(), args.build.resolve()
    identity = _measurement_identity(args.build, args.root)
    jobs = [(str(args.root), args.split, index, condition)
            for index in range(args.first, args.first + args.users) for condition in CONDITIONS]
    with ProcessPoolExecutor(args.workers, initializer=_configure_build, initargs=(args.build,)) as pool:
        rows = list(pool.map(measure, jobs))
    if _measurement_identity(args.build, args.root) != identity:
        raise RuntimeError("evaluation inputs changed during the run")
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "measurement_identity": identity,
              "timing_semantics": "mean render throughput, including the Python/native call; not live p99",
              "summary": summarize(rows)}
    if args.rows_output is None:
        result["rows"] = rows
    else:
        args.rows_output.write_text(json.dumps({"schema_version": 1, "split": args.split,
                                               "measurement_identity": identity, "rows": rows},
                                              indent=1, allow_nan=False) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result["summary"], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
