"""Evaluate Auto Voice Setup decisions on simulated users with known clean speech.

Subcommands:
  fetch   VoiceBank clean speech (the VCTK subset, CC BY 4.0) and DEMAND
          channel-1 noise (CC BY 4.0 on Zenodo) into ignored ``models/``,
          reading only the needed archive members with HTTP range requests.
  run     Simulate users from one split, run ``analyze_voice_setup`` on each
          setup capture, and render a separate evaluation capture through the
          input frontend (pre-filter, gate, noise suppression) for the current
          settings, the tuner's choice and every candidate the tuner tested.
  report  Summarize the rows written by ``run``.

A simulated user is one speaker in one synthetic room with one DEMAND
environment at one SNR and speech level. Splits are disjoint in speakers,
noise environments and room seeds: fit, calibrate and test. The test split is
reserved for final qualification. Current settings are the factory defaults
with the user's noise model.

Frontend comparison (fixed before the October 2026 pilot): arm A dominates arm
B when A is no worse than B beyond the tolerance on every metric and better by
at least the tolerance on one. Metrics, scored on the evaluation capture
against the reverberant clean speech, and tolerances: ESTOI 0.01, onset level
1 dB, pause level 1 dB, deep-pause level (at least 600 ms from speech; skipped
when no such pause exists) 1 dB, post-speech tail level 1 dB. Deep pauses were
added on 2026-10-02 after the phase 6 test run showed that reverberant tails
fill ordinary pause frames and hide what the gate does to the room noise. Per user, the tuner
improves when its choice dominates the current settings, harms when the
current settings dominate its choice, and misses an improvement when a tested
candidate dominates its choice. EQ and dynamics are outside this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter, resample_poly

import evaluate_gate_corpus as gate_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "models" / "voice_setup_eval"
DATASHARE = "https://datashare.ed.ac.uk/server/api/core/bitstreams/{}/content"
VOICEBANK = {
    "clean_trainset_28spk_wav.zip": "245452b6-6235-44b6-a6f9-e7eb19797769",
    "clean_trainset_56spk_wav.zip": "386ac199-5d34-402f-8112-2a123577dd57",
    "clean_testset_wav.zip": "dec213d3-bf57-4777-9663-c24bdce92d5e",
}
DEMAND_API = "https://zenodo.org/api/records/1227121"
DEMAND_FILES = "https://zenodo.org/records/1227121/files"
SAMPLE_RATE = gate_corpus.SAMPLE_RATE
UTTERANCES_PER_SPEAKER = 24
SPLITS = ("fit", "calibrate", "test")
NOISE_S = 2.0  # Voice Setup's room-noise recording
SPEECH_S = 10.0  # Voice Setup's voice recording
EVALUATION_S = 20.0
# Metric: (direction in which larger is better, tolerance).
TOLERANCES = {"estoi": (1.0, 0.01), "onset_db": (1.0, 1.0), "pause_db": (-1.0, 1.0),
              "deep_pause_db": (-1.0, 1.0), "tail_db": (-1.0, 1.0)}
# Undefined for both arms when a capture has no pause long enough.
OPTIONAL_METRICS = {"deep_pause_db"}


def _range_archive(url: str) -> zipfile.ZipFile:
    import io

    return zipfile.ZipFile(io.BufferedReader(gate_corpus._RangeFile(url), buffer_size=gate_corpus.RANGE_BLOCK))


def _read_wav(path: Path) -> np.ndarray:
    rate, audio = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}")
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float64) / float(np.iinfo(audio.dtype).max)
    audio = np.asarray(audio, np.float64)
    return audio[:, 0] if audio.ndim > 1 else audio


def _median_f0_hz(audio: np.ndarray) -> float:
    """Median autocorrelation pitch of voiced 40 ms frames (60-400 Hz)."""
    signal = resample_poly(audio, 1, 3)  # 16 kHz
    frames = np.lib.stride_tricks.sliding_window_view(signal, 640)[::320]
    energy = np.mean(frames**2, axis=1)
    pitches = []
    for frame in frames[energy > np.percentile(energy, 60)]:
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, "full")[639:]
        if corr[0] <= 0:
            continue
        lag = 40 + int(np.argmax(corr[40:267]))
        if corr[lag] / corr[0] > 0.5:
            pitches.append(16_000 / lag)
    return float(np.median(pitches)) if pitches else float("nan")


def fetch(root: Path) -> None:
    speakers: dict[str, dict[str, Any]] = {}
    for archive_name, uuid in VOICEBANK.items():
        url = DATASHARE.format(uuid)
        with _range_archive(url) as archive:
            members = sorted(name for name in archive.namelist() if name.endswith(".wav"))
            by_speaker: dict[str, list[str]] = {}
            for member in members:
                by_speaker.setdefault(Path(member).name.split("_")[0], []).append(member)
            for speaker, names in sorted(by_speaker.items()):
                files = []
                for member in names[:UTTERANCES_PER_SPEAKER]:
                    target = root / "speech" / speaker / Path(member).name
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
                    files.append({"path": target.relative_to(root).as_posix(),
                                  "sha256": gate_corpus._sha256(target)})
                audio = np.concatenate([_read_wav(root / entry["path"]) for entry in files])
                f0 = _median_f0_hz(audio)
                speakers[speaker] = {
                    "source": f"{url}#{archive_name}", "f0_hz": f0,
                    # Used only to balance the splits.
                    "gender": "female" if f0 > 165.0 else "male", "files": files,
                }
    import urllib.request

    with urllib.request.urlopen(DEMAND_API) as response:
        record = json.loads(response.read())
    noise = {}
    for key in sorted(f["key"] for f in record["files"] if f["key"].endswith("_48k.zip")):
        environment = key.removesuffix("_48k.zip")
        target = root / "noise" / f"{environment}_ch01.wav"
        if not target.exists():
            with _range_archive(f"{DEMAND_FILES}/{key}?download=1") as archive:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(f"{environment}/ch01.wav"))
        noise[environment] = {"path": target.relative_to(root).as_posix(),
                              "source": f"{DEMAND_FILES}/{key}#{environment}/ch01.wav",
                              "sha256": gate_corpus._sha256(target)}
    manifest = {
        "schema_version": 1,
        "description": "VoiceBank clean speech and DEMAND channel-1 noise, 48 kHz",
        "licenses": {"VoiceBank (VCTK subset)": "CC BY 4.0", "DEMAND": record["metadata"]["license"]["id"]},
        "redistribution": "local evaluation only; not a release asset",
        "speakers": speakers, "noise": noise,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def splits(manifest: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Assign speakers (gender-balanced) and noise environments round-robin."""
    speakers = sorted(manifest["speakers"], key=lambda s: (manifest["speakers"][s]["gender"], s))
    environments = sorted(manifest["noise"])
    return {name: {"speakers": speakers[index::3], "environments": environments[index::3]}
            for index, name in enumerate(SPLITS)}


def room_response(rng: np.random.Generator) -> np.ndarray:
    """Stochastic room: direct sound, five early reflections and a decaying tail.

    The tail is exponentially decaying noise with high frequencies damped, as
    air and wall absorption take them first.
    """
    rt60 = rng.uniform(0.2, 0.7)
    length = int(rt60 * SAMPLE_RATE)
    time = np.arange(length) / SAMPLE_RATE
    tail = rng.standard_normal(length) * np.exp(-6.9 * time / rt60)
    pole = np.exp(-2 * np.pi * rng.uniform(3_000.0, 8_000.0) / SAMPLE_RATE)
    tail = np.asarray(lfilter([1 - pole], [1, -pole], tail))
    tail[: int(0.002 * SAMPLE_RATE)] = 0.0
    direct_to_reverberant_db = rng.uniform(0.0, 12.0)
    tail *= 10.0 ** (-direct_to_reverberant_db / 20.0) / np.sqrt(np.sum(tail**2))
    response = tail
    response[0] = 1.0
    for delay_ms in rng.uniform(2.0, 20.0, 5):
        response[int(delay_ms * SAMPLE_RATE / 1000)] += rng.uniform(0.2, 0.6) * rng.choice((-1, 1))
    return response


def _trimmed(audio: np.ndarray) -> np.ndarray:
    levels = gate_corpus._frame_rms_db(audio)
    active = np.flatnonzero(levels > np.max(levels) - 40.0)
    return audio[active[0] * gate_corpus.FRAME : (active[-1] + 1) * gate_corpus.FRAME]


def user(root: Path, manifest: dict[str, Any], split_name: str, index: int) -> dict[str, Any]:
    """Deterministic simulated user ``index`` of a split."""
    part = splits(manifest)[split_name]
    rng = np.random.default_rng(zlib.crc32(f"{split_name}:{index}".encode()))
    speaker = part["speakers"][index % len(part["speakers"])]
    environment = str(rng.choice(part["environments"]))
    room = room_response(np.random.default_rng(1_000_000 * (SPLITS.index(split_name) + 1) + index))
    snr_db = float(rng.uniform(3.0, 30.0))
    level_dbfs = float(rng.uniform(-38.0, -18.0))
    utterances = [_trimmed(_read_wav(root / f["path"])) for f in manifest["speakers"][speaker]["files"]]
    half = len(utterances) // 2
    setup_dry, _ = gate_corpus._with_pauses(np.concatenate(utterances[:half]), rng)
    evaluation_dry, labels = gate_corpus._with_pauses(np.concatenate(utterances[half:]), rng)
    setup_dry = setup_dry[: int(SPEECH_S * SAMPLE_RATE)]
    count = int(EVALUATION_S * SAMPLE_RATE)
    evaluation_dry, labels = evaluation_dry[:count], labels[:count]
    setup_clean = fftconvolve(setup_dry, room)[: setup_dry.size]
    evaluation_clean = fftconvolve(evaluation_dry, room)[: evaluation_dry.size]
    gain = 10.0 ** ((level_dbfs - gate_corpus._active_level_db(setup_clean)) / 20.0)
    source = _read_wav(root / manifest["noise"][environment]["path"])
    source = source / np.sqrt(np.mean(source**2))
    lengths = (int(NOISE_S * SAMPLE_RATE), setup_dry.size, evaluation_dry.size)
    offset = int(rng.integers(0, source.size - sum(lengths)))
    segments = []
    for length in lengths:
        segments.append(source[offset : offset + length] * 10.0 ** ((level_dbfs - snr_db) / 20.0))
        offset += length
    captures = {
        "noise_capture": segments[0],
        "setup_capture": setup_clean * gain + segments[1],
        "evaluation_capture": evaluation_clean * gain + segments[2],
        "evaluation_clean": evaluation_clean * gain,
        "setup_clean": setup_clean * gain,
    }
    peak = max(float(np.max(np.abs(audio))) for audio in captures.values())
    scale = min(1.0, 0.95 / peak)
    return {
        "split": split_name, "index": index, "speaker": speaker, "environment": environment,
        "snr_db": snr_db, "level_dbfs": level_dbfs + 20.0 * np.log10(scale), "labels": labels,
        **{name: (audio * scale).astype(np.float32) for name, audio in captures.items()},
    }


def _arm_key(model: str, strength: float, gate: dict[str, Any]) -> str:
    return json.dumps([model, round(float(strength), 4), sorted(gate.items())], default=str)


_VAD_CACHE: dict[tuple[bytes, float, float], np.ndarray | None] = {}


def _offline_vad(capture: np.ndarray, threshold: float, pre_gain: float) -> np.ndarray | None:
    """Silero posteriors of a capture; arms that share its VAD settings reuse them."""
    from mic_eq.analysis.vad import analyze_offline_vad

    key = (hashlib.sha256(capture.tobytes()).digest(), threshold, pre_gain)
    if key not in _VAD_CACHE:
        _VAD_CACHE[key] = analyze_offline_vad(capture, SAMPLE_RATE, threshold=threshold, pre_gain=pre_gain)[0]
    return _VAD_CACHE[key]


def render_arm(capture: np.ndarray, arm: dict[str, Any], noise_floor_db: float) -> np.ndarray:
    from mic_eq.analysis.joint_tuning import (
        _align_probabilities, _candidate_settings, _load_native_simulator, _run_gate_suppressor,
    )

    gate, simulator_gate = _candidate_settings(
        arm["gate"], noise_floor_db=noise_floor_db, strength=arm["strength"],
        threshold_delta_db=0.0, release_delta_ms=0.0,
    )
    probabilities = _offline_vad(
        capture, float(gate.get("vad_threshold", 0.48)), float(gate.get("vad_pre_gain", 1.0)),
    )
    control = _align_probabilities(probabilities, capture.size, capture.size)
    simulator = _load_native_simulator()
    if simulator is None:
        raise RuntimeError("native gate/suppressor simulator is unavailable")
    result = _run_gate_suppressor(
        simulator, capture, control,
        simulator_gate | {"vad_available": probabilities is not None, "return_auto_makeup_activity": True},
        suppressor_strength=arm["strength"], noise_model=arm["model"],
    )
    return np.asarray(result["output_audio"], np.float64)


def onset_proxy_db(gated: np.ndarray, ungated: np.ndarray) -> float:
    """The gate's onset attenuation relative to the phrase body, without clean speech.

    ``gated`` and ``ungated`` are the same capture through the same suppressor
    with and without the gate, so suppression cancels and gating remains.
    Phrase starts are found on the ungated render, where noise is already
    removed: its level rises 10 dB above its own floor after at least 200 ms
    below. Each onset window starts 50 ms before that crossing, to include
    a consonant the detector would miss, and is scored as an energy ratio so
    near-silent pause frames weigh little.
    """
    def energy(audio: np.ndarray) -> np.ndarray:
        frames = audio[: audio.size // gate_corpus.FRAME * gate_corpus.FRAME].reshape(-1, gate_corpus.FRAME)
        return np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-20

    gated_energy, ungated_energy = energy(gated), energy(ungated)
    level = 10.0 * np.log10(ungated_energy)
    above = level > np.percentile(level, 20) + 10.0
    values = []
    for start in np.flatnonzero(above):
        if start < 20 or above[start - 20 : start].any() or start + 130 > above.size:
            continue
        onset, body = slice(start - 5, start + 10), slice(start + 30, start + 130)
        values.append(float(10.0 * np.log10(gated_energy[onset].sum() / ungated_energy[onset].sum())
                            - 10.0 * np.log10(gated_energy[body].sum() / ungated_energy[body].sum())))
    return float(np.median(values)) if values else float("nan")


def evaluate_user(job: tuple[str, str, int, str]) -> dict[str, Any]:
    root_text, split_name, index, model = job
    os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = "1"
    gate_corpus._configure_deepfilter([model])
    from mic_eq.analysis.voice_setup import analyze_voice_setup
    from mic_eq.config import Preset
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    root = Path(root_text)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    simulated = user(root, manifest, split_name, index)
    preset = Preset()
    current = {"model": model, "strength": 1.0, "gate": asdict(preset.gate)}
    raw_noise, raw_speech = simulated["noise_capture"], simulated["setup_capture"]
    setup = filtered_capture_for_analysis(raw_speech, SAMPLE_RATE)
    result = analyze_voice_setup(
        filtered_capture_for_analysis(raw_noise, SAMPLE_RATE), setup, SAMPLE_RATE, "broadcast",
        noise_model=model, suppressor_strength=1.0,
        incumbent_settings={"gate": dict(current["gate"]),
                            "rnnoise": {"enabled": True, "strength": 1.0, "model": model}},
        raw_noise_audio=raw_noise, raw_speech_audio=raw_speech,
    )
    tuning = result["diagnostics"]["joint_tuning"]
    chosen = {"model": result["suppressor_settings"]["model"],
              "strength": float(result["suppressor_settings"]["strength"]),
              "gate": dict(result["gate_settings"])}
    arms = {_arm_key(**current): current, _arm_key(**chosen): chosen}
    tuner_view = {}
    for candidate in tuning.get("candidates", []):
        arm = {"model": candidate["noise_model"], "strength": candidate["suppressor_strength"],
               "gate": dict(candidate["gate_settings"])}
        arms.setdefault(_arm_key(**arm), arm)
        tuner_view[_arm_key(**arm)] = {"score": candidate["score"], "passes": candidate["passes"],
                                       "metrics": candidate["metrics"]}
    capture = filtered_capture_for_analysis(simulated["evaluation_capture"], SAMPLE_RATE)
    case = {"clean": filtered_capture_for_analysis(simulated["evaluation_clean"], SAMPLE_RATE).astype(np.float64),
            "noisy": capture.astype(np.float64), "labels": simulated["labels"]}
    noise_floor = float(tuning.get("noise_floor_db", -80.0))
    scored = {key: gate_corpus.score(case, render_arm(capture, arm, noise_floor)) for key, arm in arms.items()}
    # The same suppressor with the gate off: truth for the gate's own onset
    # effect on the evaluation capture, and the proxy's reference on setup.
    ungated: dict[str, dict[str, Any]] = {}
    for key, arm in arms.items():
        off = arm | {"gate": arm["gate"] | {"enabled": False}}
        off_key = _arm_key(off["model"], off["strength"], {})
        if off_key not in ungated:
            ungated[off_key] = {"metrics": gate_corpus.score(case, render_arm(capture, off, noise_floor)),
                                "setup_render": render_arm(setup, off, noise_floor)}
        arms[key] = arm | {"ungated": off_key}
        if arm["gate"].get("enabled", True):
            scored[key]["proxy_onset_db"] = onset_proxy_db(
                render_arm(setup, arm, noise_floor), ungated[off_key]["setup_render"])
    return {
        **{k: simulated[k] for k in ("split", "index", "speaker", "environment", "snr_db", "level_dbfs")},
        "model": model, "tuner_decision": tuning.get("decision"),
        "setup_apply_recommended": bool(result["diagnostics"]["apply_recommended"]),
        "current": _arm_key(**current), "chosen": _arm_key(**chosen),
        "candidates": [_arm_key(c["noise_model"], c["suppressor_strength"], dict(c["gate_settings"]))
                       for c in tuning.get("candidates", [])],
        "arms": {key: {"settings": arms[key], "metrics": scored[key]} for key in arms},
        "ungated": {key: value["metrics"] for key, value in ungated.items()},
        "tuner_view": tuner_view,
    }


def run(function: Callable[[tuple[str, str, int, str]], dict[str, Any]], jobs: list[tuple[str, str, int, str]],
        workers: int, partial: Path) -> list[dict[str, Any]]:
    """Run jobs, appending each finished row to ``partial`` and skipping rows already there."""
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(row["split"], row["index"], row["model"]) for row in rows}
    pending = [job for job in jobs if (job[1], job[2], job[3]) not in done]
    with ProcessPoolExecutor(max(workers, 1)) as pool, partial.open("a", encoding="utf-8") as log:
        finished = (map(function, pending) if workers <= 1
                    else (future.result() for future in as_completed([pool.submit(function, job) for job in pending])))
        for row in finished:
            log.write(json.dumps(row, default=float) + "\n")
            log.flush()
            rows.append(row)
    order = {(job[1], job[2], job[3]): position for position, job in enumerate(jobs)}
    return sorted((row for row in rows if (row["split"], row["index"], row["model"]) in order),
                  key=lambda row: order[(row["split"], row["index"], row["model"])])


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    better = False
    for metric, (direction, tolerance) in TOLERANCES.items():
        if metric in OPTIONAL_METRICS and not np.isfinite(a[metric]) and not np.isfinite(b[metric]):
            continue
        gain = direction * (a[metric] - b[metric])
        if not np.isfinite(gain) or gain < -tolerance:
            return False
        better |= gain >= tolerance
    return better


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        flags: dict[str, list[float]] = {name: [] for name in ("changed", "improved", "harmed", "missed", "available")}
        for row in subset:
            metrics = {key: arm["metrics"] for key, arm in row["arms"].items()}
            current, chosen = metrics[row["current"]], metrics[row["chosen"]]
            tested = [metrics[key] for key in row["candidates"]]
            flags["changed"].append(float(row["chosen"] != row["current"]))
            flags["improved"].append(float(dominates(chosen, current)))
            flags["harmed"].append(float(dominates(current, chosen)))
            flags["missed"].append(float(any(dominates(c, chosen) for c in tested)))
            flags["available"].append(float(any(dominates(c, current) for c in tested)))
        clusters = np.array([row["speaker"] for row in subset])
        summary[model] = {"users": len(subset), "speakers": len(set(clusters)), **{
            name: gate_corpus.cluster_ci(np.array(values), clusters) for name, values in flags.items()
        }, "onset_proxy": proxy_validity(subset)}
    return summary


def proxy_validity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply GPLAN phase 3's predefined validity rule to the onset proxy."""
    from scipy.stats import rankdata

    pairs = []
    for row in rows:
        for arm in row["arms"].values():
            proxy = arm["metrics"].get("proxy_onset_db")
            if proxy is None:
                continue
            truth = arm["metrics"]["onset_db"] - row["ungated"][arm["settings"]["ungated"]]["onset_db"]
            if np.isfinite(proxy) and np.isfinite(truth):
                pairs.append((proxy, truth))
    if not pairs:
        return {"arms": 0, "valid": False}
    proxy, truth = np.array(pairs).T
    rho = float(np.corrcoef(rankdata(proxy), rankdata(truth))[0, 1])  # Spearman
    damaged, intact = truth <= -2.0, truth >= -0.5
    detected = float(np.mean(proxy[damaged] <= -1.0)) if damaged.any() else float("nan")
    cleared = float(np.mean(proxy[intact] > -1.0)) if intact.any() else float("nan")
    return {
        "arms": len(pairs), "spearman_rho": rho, "damaged_arms": int(damaged.sum()),
        "damaged_detected": detected, "intact_arms": int(intact.sum()), "intact_cleared": cleared,
        "valid": bool(rho >= 0.6 and detected >= 0.75 and cleared >= 0.90),
    }


POLICY_ARMS = ("gate_off", "vad_assisted")


def evaluate_policy(job: tuple[str, str, int, str]) -> dict[str, Any]:
    """Factory gate vs gate off vs VAD Assisted, factory settings otherwise (GPLAN phase 6)."""
    root_text, split_name, index, model = job
    os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = "1"
    gate_corpus._configure_deepfilter([model])
    from mic_eq.config import Preset
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    root = Path(root_text)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    simulated = user(root, manifest, split_name, index)
    gate = asdict(Preset().gate)
    gates = {"factory": gate, "gate_off": gate | {"enabled": False}, "vad_assisted": gate | {"gate_mode": 1}}
    capture = filtered_capture_for_analysis(simulated["evaluation_capture"], SAMPLE_RATE)
    noise = filtered_capture_for_analysis(simulated["noise_capture"], SAMPLE_RATE).astype(np.float64)
    floor = float(10.0 * np.log10(np.mean(noise**2) + 1e-20))
    case = {"clean": filtered_capture_for_analysis(simulated["evaluation_clean"], SAMPLE_RATE).astype(np.float64),
            "noisy": capture.astype(np.float64), "labels": simulated["labels"]}
    return {
        **{k: simulated[k] for k in ("split", "index", "speaker", "environment", "snr_db", "level_dbfs")},
        "model": model,
        "arms": {name: gate_corpus.score(case, render_arm(capture, {"model": model, "strength": 1.0, "gate": g}, floor))
                 for name, g in gates.items()},
    }


def policy_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        clusters = np.array([row["speaker"] for row in subset])
        arms = {}
        for name in POLICY_ARMS:
            net = np.array([float(dominates(r["arms"][name], r["arms"]["factory"]))
                            - float(dominates(r["arms"]["factory"], r["arms"][name])) for r in subset])
            estoi = np.array([r["arms"][name]["estoi"] - r["arms"]["factory"]["estoi"] for r in subset])
            net_ci, estoi_ci = gate_corpus.cluster_ci(net, clusters), gate_corpus.cluster_ci(estoi, clusters)
            arms[name] = {"net_dominance": net_ci, "estoi_difference": estoi_ci,
                          "qualifies": bool(net_ci[1] > 0.0 and estoi_ci[1] >= -0.005),
                          **{f"mean_{m}_difference": float(np.nanmean([r["arms"][name][m] - r["arms"]["factory"][m]
                                                                      for r in subset]))
                             for m in ("onset_db", "pause_db", "tail_db", "si_sdr")}}
        qualified = [name for name in POLICY_ARMS if arms[name]["qualifies"]]
        winner = max(qualified, key=lambda name: arms[name]["net_dominance"][0]) if qualified else "factory"
        summary[model] = {"users": len(subset), "arms": arms, "decision": winner}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "run", "report", "policy"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=SPLITS, default="fit")
    parser.add_argument("--first", type=int, default=0, help="run: index of the first user")
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--models", default="rnnoise,deepfilter-ll")
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--rows", type=Path, help="report: rows written by run")
    parser.add_argument("--output", type=Path, help="write the JSON result here")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.root)
        return 0
    jobs = [(str(args.root), args.split, index, model) for model in args.models.split(",")
            for index in range(args.first, args.first + args.users)]
    partial = None
    if args.command in ("run", "policy"):
        if args.output is None:
            parser.error(f"{args.command} needs --output")
        # ponytail: an interrupted run resumes its rows whatever code wrote them;
        # delete the file to start over, or record the build per row if that bites.
        partial = args.output.with_suffix(f".{args.command}.partial.jsonl")
        rows = run(evaluate_user if args.command == "run" else evaluate_policy, jobs, args.workers, partial)
        result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "split": args.split, "rows": rows}
        if args.command == "policy":
            result |= {"rule": "GPLAN phase 6", "summary": policy_report(rows)}
    else:
        result = report(json.loads(args.rows.read_text(encoding="utf-8"))["rows"])
    text = json.dumps(result, indent=2, sort_keys=True, default=float) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        if partial is not None:
            partial.unlink()  # finished: the next run to this output starts fresh
    shown = {"report": result, "policy": result.get("summary")}.get(args.command)
    sys.stdout.write(json.dumps(shown, indent=2, default=float) + "\n" if shown else f"{len(result['rows'])} rows\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
