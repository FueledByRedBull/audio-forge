"""Check voice-preservation regressions on the existing held-out VAD corpus.

Run before and after a native rebuild; --baseline compares the saved measurements.
This isolates gate behavior, not full-chain perceptual quality or live scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mic_eq.analysis.wav_io import read_mono_wav
from mic_eq.analysis.auto_eq import analyze_auto_eq
from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response
from mic_eq.config import EQ_FREQUENCIES, TARGET_CURVES
from mic_eq.mic_eq_core import simulate_auto_eq_chain

ROOT = Path(__file__).resolve().parents[2]
GATES = {"speech_p10_min_delta_db": -0.5, "noise_max_delta_db": 3.0}
EQ_GATES = {"neutral_max_change_db": 0.25, "style_max_change_db": 3.0}


def measure_eq(corpus: Path) -> dict:
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    speakers = set(manifest["speaker_disjoint_splits"]["test"])
    passages: dict[tuple[str, str], list[np.ndarray]] = {}
    for pair in manifest["pairs"]:
        if pair["speaker"] not in speakers:
            continue
        parts = passages.setdefault((pair["speaker"], pair["statement_id"]), [])
        for take in pair["takes"].values():
            path = corpus / take["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != take["sha256"]:
                raise ValueError(f"Corpus hash mismatch: {take['path']}")
            rate, audio = read_mono_wav(path)
            if rate != 48_000:
                raise ValueError("EQ corpus must be 48 kHz")
            parts.append(audio)
    rows = []
    freqs = np.geomspace(80.0, 16_000.0, 512)
    bands = [(float(f), 0.0, 1.0) for f in EQ_FREQUENCIES]
    for (speaker, statement), parts in sorted(passages.items()):
        audio = np.ascontiguousarray(np.concatenate(parts)[:480_000], dtype=np.float32)
        filtered = simulate_auto_eq_chain(audio, 48_000.0, bands, {
            "full_chain": True, "processing_mode": "bypass",
            "limiter_enabled": False, "return_output_audio": True,
        })
        audio = np.asarray(filtered["output_audio"], dtype=np.float32)
        for target in TARGET_CURVES:
            eq, _validation = analyze_auto_eq(audio, 48_000, target)
            response = _predict_eq_response(freqs, eq["band_gains"], eq["band_qs"], eq["band_freqs"])
            maximum = float(np.max(np.abs(response)))
            limit = EQ_GATES["neutral_max_change_db" if target == "flat" else "style_max_change_db"]
            rows.append({"speaker": speaker, "statement": statement, "target": target,
                         "max_change_db": maximum, "passed": bool(np.all(np.isfinite(response)) and maximum <= limit)})
    if len(passages) < 12:
        raise ValueError("Require all six held-out speakers and both statements")
    return {"corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "gates": EQ_GATES, "cases": rows, "passed": all(row["passed"] for row in rows),
            "limitations": "Checks bounded tonal change and neutral preservation, not microphone-response recovery or subjective quality."}


def measure(corpus: Path) -> dict:
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    bands = [(float(f), 0.0, 1.0) for f in EQ_FREQUENCIES]
    for capture in manifest["captures"]:
        if capture["split"] != "held_out" or capture["sample_rate"] != 48_000:
            continue
        path = corpus / capture["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != capture["sha256"]:
            raise ValueError(f"Corpus hash mismatch: {capture['path']}")
        rate, audio = read_mono_wav(path)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        speech = np.zeros(audio.size, dtype=bool)
        for start, end in capture["speech_intervals_samples"]:
            speech[int(start):int(end)] = True
        # Preserve identical frontend filtering in the gate-off reference.
        controls = {
            "full_chain": True, "return_output_audio": True,
            "input_cleanup_mode": "gentle", "suppressor_enabled": False,
            "eq_enabled": False, "compressor_enabled": False,
            "deesser_enabled": False, "limiter_enabled": False,
            "gate_mode": 2, "gate_vad_threshold": 0.35,
            "gate_vad_pre_gain": 2.5, "gate_vad_hold_time_ms": 200.0,
            "gate_auto_threshold_enabled": True, "gate_margin_db": 6.0,
        }
        rendered = []
        for enabled in (False, True):
            result = simulate_auto_eq_chain(
                audio, float(rate), bands, {**controls, "gate_enabled": enabled}
            )
            output = np.asarray(result["output_audio"], dtype=np.float64)
            if output.size != audio.size or not np.all(np.isfinite(output)):
                raise ValueError(f"Invalid output: {capture['path']}")
            rendered.append(output)
        mask = speech if speech.any() else np.ones(audio.size, dtype=bool)
        energies = [float(np.mean(output[mask] ** 2)) for output in rendered]
        retention = 10.0 * np.log10(max(energies[1], 1e-16) / max(energies[0], 1e-16))
        rows.append({"path": capture["path"], "speech": bool(speech.any()),
                     "retention_db": float(retention)})
    if not rows:
        raise ValueError("No held-out 48 kHz captures")
    return {
        "corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(next((ROOT / "python/mic_eq").glob("mic_eq_core*.pyd")).read_bytes()).hexdigest(),
        "gates": GATES, "cases": rows,
        "limitations": "Gate-only offline rendering; does not model asynchronous worker scheduling or measure perceptual naturalness.",
    }


def compare(current: dict, baseline: dict) -> dict:
    if current["corpus_manifest_sha256"] != baseline["corpus_manifest_sha256"]:
        raise ValueError("Baseline uses a different corpus")
    previous = {row["path"]: row for row in baseline["cases"]}
    if set(previous) != {row["path"] for row in current["cases"]}:
        raise ValueError("Baseline case set differs")
    speech, noise = [], []
    for row in current["cases"]:
        delta = row["retention_db"] - previous[row["path"]]["retention_db"]
        (speech if row["speech"] else noise).append(delta)
    if not speech or not noise:
        raise ValueError("Both speech and noise cases are required")
    p10, maximum = float(np.percentile(speech, 10)), float(max(noise))
    return {"speech_cases": len(speech), "noise_cases": len(noise),
            "speech_p10_delta_db": p10, "noise_max_delta_db": maximum,
            "passed": p10 >= GATES["speech_p10_min_delta_db"] and maximum <= GATES["noise_max_delta_db"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "models/vad_eval_corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--eq-corpus", type=Path)
    args = parser.parse_args()
    result = measure(args.corpus)
    if args.baseline:
        result["comparison"] = compare(result, json.loads(args.baseline.read_text(encoding="utf-8")))
    if args.eq_corpus:
        result["eq"] = measure_eq(args.eq_corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.get("comparison", {"baseline_cases": len(result["cases"])})))
    return 0 if result.get("comparison", {}).get("passed", True) and result.get("eq", {}).get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
