"""Measure the de-esser on real speech: setup-time detection and the processed audio.

Simulated users (``evaluate_voice_setup.py``) are heard through three
microphones: ordinary (unchanged), bright (a broad +6 dB shelf from 4 kHz, an
EQ matter, not sibilance) and harsh (a +8 dB resonance between 5.5 and 8 kHz
that makes sibilants piercing). Only harsh should enable the de-esser.

Detection: Voice Setup's own setup-time path (noise reference, VAD-masked
frame evidence, the fitted soft-fusion model) decides from the setup capture.
Audio: the evaluation capture is rendered through the native de-esser twice,
with Voice Setup's recommended settings and with the factory settings, and
the 5-9 kHz level change is measured on sibilant frames and on other voiced
frames. Both renders force the de-esser on, so they show what it does when
enabled; ``enabled_rate`` says how often Voice Setup would enable it.
Sibilant frames are labeled on the uncolored clean speech, where fricatives
carry most of their energy above 4 kHz.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_voice_setup as harness
from scipy.signal import lfilter

from evaluate_gate_corpus import FRAME, cluster_ci

CONDITIONS = ("ordinary", "bright", "harsh")
USERS = 40


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


def _render(capture: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    from mic_eq.config import EQ_FREQUENCIES
    from mic_eq.mic_eq_core import simulate_auto_eq_chain

    chain = {f"deesser_{key}": value for key, value in settings.items()
             if key in ("auto_enabled", "auto_amount", "low_cut_hz", "high_cut_hz",
                        "threshold_db", "ratio", "attack_ms", "release_ms", "max_reduction_db")}
    chain |= {"deesser_enabled": True, "compressor_enabled": False, "limiter_enabled": False,
              "return_output_audio": True}
    bands = [(float(f), 0.0, 1.0) for f in EQ_FREQUENCIES]
    result = simulate_auto_eq_chain(capture.astype(np.float32), float(harness.SAMPLE_RATE), bands, chain)
    return np.asarray(result["output_audio"], np.float64)


def measure(job: tuple[str, int, str]) -> dict[str, Any]:
    split_name, index, condition = job
    from mic_eq.config import Preset
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    root = harness.DEFAULT_ROOT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    user = harness.user(root, manifest, split_name, index)
    sections = microphone(condition, np.random.default_rng(9_000_000 + 1_000 * harness.SPLITS.index(split_name) + index))

    def heard(audio: np.ndarray) -> np.ndarray:
        colored = _apply(audio, sections) if sections else np.asarray(audio, np.float64)
        return filtered_capture_for_analysis(colored.astype(np.float32), harness.SAMPLE_RATE)

    recommended = _recommendation(heard(user["setup_capture"]), heard(user["noise_capture"]))
    capture = heard(user["evaluation_capture"])
    sibilant, voiced = sibilant_frames(user["evaluation_clean"].astype(np.float64))
    before = _band_db(capture.astype(np.float64), 5_000.0, 9_000.0)
    row: dict[str, Any] = {"split": split_name, "index": index, "speaker": user["speaker"],
                           "condition": condition, "enabled": bool(recommended["enabled"]),
                           "detection_probability": float(recommended["detection_probability"]),
                           "sibilant_frames": int(sibilant.sum())}
    for name, settings in (("recommended", recommended), ("factory", asdict(Preset().deesser))):
        change = _band_db(_render(capture, settings), 5_000.0, 9_000.0)[: before.size] - before
        row[f"{name}_sibilant_db"] = float(np.mean(change[sibilant[: change.size]]))
        row[f"{name}_voiced_db"] = float(np.mean(change[voiced[: change.size]]))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        clusters = np.array([r["speaker"] for r in subset])
        summary[condition] = {
            "users": len(subset),
            "enabled_rate": cluster_ci(np.array([float(r["enabled"]) for r in subset]), clusters),
            **{f"{arm}_{frames}_db": cluster_ci(np.array([r[f"{arm}_{frames}_db"] for r in subset]), clusters)
               for arm in ("recommended", "factory") for frames in ("sibilant", "voiced")},
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=harness.SPLITS, default="fit")
    parser.add_argument("--users", type=int, default=USERS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs = [(args.split, index, condition) for index in range(args.users) for condition in CONDITIONS]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(measure, jobs))
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "summary": summarize(rows), "rows": rows}
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result["summary"], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
