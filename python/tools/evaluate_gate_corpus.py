"""Evaluate the noise gate on fullband EARS speech mixed with DEMAND noise.

Subcommands:
  fetch      Build the local corpus under ignored ``models/`` from the EARS and
             DEMAND releases, reading only the needed archive members with
             HTTP range requests.
  fit        Fit the speech-presence logistic on the training speakers.
  calibrate  Match each VAD mode's bias to an incumbent build's pause
             attenuation on the training speakers, per noise model; the
             adopted bias is the midpoint across models. The candidate build
             must accept the ``gate_spp_vad_assisted``/``gate_spp_vad_only``
             simulator overrides (the October 2026 speech-presence candidate,
             preserved under ``models/evaluation-details/``).
  qualify    Score an incumbent and a candidate build on the held-out speakers
             and apply the adoption rule below; writes a compact report.
  order      Render gate-first, suppressor-first and suppressor-only arms.

A build is a source root whose ``python/mic_eq`` holds a built native
extension and whose ``target/onnxruntime-cpu/lib`` holds the CPU ONNX Runtime.

The VAD control is causal: gate block j sees the last 32 ms Silero window that
ended by the block's first sample, as the live worker can deliver it.

Adoption rule, per VAD mode (fixed before the October 2026 qualification):
with speaker-clustered 95% intervals of candidate minus incumbent on the
held-out speakers, for RNNoise and DeepFilter LL, with the VAD available and
unavailable, ESTOI's lower bound is at least -0.002, pause and tail upper
bounds are below +1 dB, and SI-SDR's lower bound is above -0.1 dB; and with
the VAD available on RNNoise there is a material win (onset lower bound above
+1 dB, ESTOI lower bound above 0, or pause upper bound below -1 dB). A mode
that fails keeps the incumbent.

EARS (CC BY-NC 4.0) and DEMAND (CC BY 4.0) are evaluation inputs only and
are never release assets. EARS trims silences, so phrases are cut at their own
inter-word gaps and separated by log-uniform 0.25-2.5 s pauses; the inserted
pauses give exact speech/pause labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.optimize import minimize

from _eval_common import estoi, si_sdr

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "models" / "ears_demand_eval"
EARS_RELEASE = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset"
EARS_STATS = "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json"
DEMAND_RECORD = "https://zenodo.org/records/1227121/files"
DEMAND_ENVIRONMENTS = (
    "DKITCHEN", "DLIVING", "DWASHING", "OOFFICE",
    "OMEETING", "OHALLWAY", "PCAFETER", "STRAFFIC",
)
SPEAKERS_PER_GENDER = 12
EXCERPT_START_S = 20.0
EXCERPT_S = 30.0
SAMPLE_RATE = 48_000
FRAME = 480
SNRS_DB = (5.0, 15.0, 25.0)
SPEECH_LEVEL_DBFS = -26.0
# One Silero result per 512 samples at 16 kHz, i.e. 1536 samples at 48 kHz.
VAD_WINDOW_SAMPLES = 1536
BOOTSTRAP = 4000
GATE_SETTINGS = {
    "gate_threshold_db": -40.0, "gate_attack_ms": 10.0, "gate_release_ms": 100.0,
    "gate_vad_threshold": 0.48, "gate_vad_hold_time_ms": 200.0,
    "gate_auto_threshold_enabled": True, "gate_margin_db": 10.0,
}
METRICS = (
    "estoi", "si_sdr", "onset_db", "pause_db", "deep_pause_db", "tail_db",
    "floor_modulation_db",
)
# The adoption rule reads these; a case missing any of them is invalid.
RULE_METRICS = ("estoi", "si_sdr", "onset_db", "pause_db", "tail_db")
MODES ={1: "vad_assisted", 2: "vad_only"}
QUALIFY_MODELS = ("rnnoise", "deepfilter-ll")
RANGE_BLOCK = 1 << 20


class _RangeFile(io.RawIOBase):
    """Seekable reader over an HTTP resource using byte-range requests."""

    def __init__(self, url: str) -> None:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.url = response.geturl()
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: 0, 1: self.position, 2: self.size}[whence] + offset
        return self.position

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * RANGE_BLOCK
            end = min(self.size, start + RANGE_BLOCK) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request) as response:
                self.cache[index] = response.read()
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[index]

    def readinto(self, buffer: Any) -> int:
        size = min(len(buffer), self.size - self.position)
        written = 0
        while written < size:
            index, offset = divmod(self.position, RANGE_BLOCK)
            chunk = self._block(index)[offset : offset + size - written]
            buffer[written : written + len(chunk)] = chunk
            written += len(chunk)
            self.position += len(chunk)
        return written


def _remote_member(url: str, member: str) -> tuple[int, np.ndarray]:
    with zipfile.ZipFile(io.BufferedReader(_RangeFile(url), buffer_size=RANGE_BLOCK)) as archive:
        rate, audio = wavfile.read(io.BytesIO(archive.read(member)))
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float64) / float(np.iinfo(audio.dtype).max)
    audio = np.asarray(audio, dtype=np.float64)
    return int(rate), audio[:, 0] if audio.ndim > 1 else audio


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(root: Path) -> None:
    with urllib.request.urlopen(EARS_STATS) as response:
        stats = json.loads(response.read())
    speakers = []
    for gender in ("female", "male"):
        ids = sorted(speaker for speaker, info in stats.items() if info["gender"] == gender)
        stride = len(ids) / SPEAKERS_PER_GENDER
        speakers += [(ids[int(i * stride)], gender) for i in range(SPEAKERS_PER_GENDER)]
    files = []
    for speaker, gender in sorted(speakers):
        target = root / "speech" / f"{speaker}_freeform_01.wav"
        member = f"{speaker}/freeform_speech_01.wav"
        if not target.exists():
            rate, audio = _remote_member(f"{EARS_RELEASE}/{speaker}.zip", member)
            start = int(EXCERPT_START_S * rate)
            target.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(target, rate, audio[start : start + int(EXCERPT_S * rate)].astype(np.float32))
        files.append({
            "kind": "speech", "path": target.relative_to(root).as_posix(), "speaker": speaker,
            "gender": gender, "source": f"{EARS_RELEASE}/{speaker}.zip#{member}",
            "excerpt_start_s": EXCERPT_START_S, "sha256": _sha256(target),
        })
    for environment in DEMAND_ENVIRONMENTS:
        target = root / "noise" / f"{environment}_ch01.wav"
        member = f"{environment}/ch01.wav"
        if not target.exists():
            rate, audio = _remote_member(f"{DEMAND_RECORD}/{environment}_48k.zip?download=1", member)
            target.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(target, rate, audio.astype(np.float32))
        files.append({
            "kind": "noise", "path": target.relative_to(root).as_posix(),
            "environment": environment,
            "source": f"{DEMAND_RECORD}/{environment}_48k.zip#{member}", "sha256": _sha256(target),
        })
    manifest = {
        "schema_version": 1,
        "description": "EARS freeform speech excerpts and DEMAND channel-1 noise, 48 kHz",
        "licenses": {"EARS": "CC BY-NC 4.0", "DEMAND": "CC BY 4.0"},
        "redistribution": "local evaluation only; not a release asset",
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _frame_rms_db(audio: np.ndarray) -> np.ndarray:
    count = audio.size // FRAME
    frames = audio[: count * FRAME].reshape(count, FRAME)
    return 10.0 * np.log10(np.mean(np.square(frames), axis=1) + 1e-20)


def _active_level_db(speech: np.ndarray) -> float:
    levels = _frame_rms_db(speech)
    active = levels > np.max(levels) - 40.0
    return float(10.0 * np.log10(np.mean(10.0 ** (levels[active] / 10.0))))


def _energy_mean_db(levels_db: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(10.0 * np.log10(np.mean(10.0 ** (levels_db[mask] / 10.0))))


def _with_pauses(speech: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rel = _frame_rms_db(speech) - _active_level_db(speech)
    gaps = np.flatnonzero(rel < np.percentile(rel, 5) + 6.0)
    cuts, position = [], 0
    while True:
        later = gaps[gaps >= position + int(rng.uniform(150, 450))]
        if later.size == 0:
            break
        position = int(later[0])
        cuts.append(position * FRAME)
    fade = np.linspace(0.0, 1.0, 240)
    pieces = [np.zeros(SAMPLE_RATE // 2)]
    labels = [np.zeros(SAMPLE_RATE // 2, bool)]
    chunks = np.split(speech, cuts)
    for index, chunk in enumerate(chunks):
        chunk = chunk.copy()
        chunk[: fade.size] *= fade[: chunk.size]
        chunk[-fade.size :] *= fade[::-1][-chunk.size :]
        pause = SAMPLE_RATE if index == len(chunks) - 1 else int(
            SAMPLE_RATE * np.exp(rng.uniform(np.log(0.25), np.log(2.5)))
        )
        pieces += [chunk, np.zeros(pause)]
        labels += [np.ones(chunk.size, bool), np.zeros(pause, bool)]
    return np.concatenate(pieces), np.concatenate(labels)


def mixtures(root: Path, environments_per_speaker: int = 2) -> Iterator[dict[str, Any]]:
    """Yield each speaker x environments x three SNRs, deterministically."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    noise = {
        entry["environment"]: np.asarray(wavfile.read(root / entry["path"])[1], np.float64)
        for entry in manifest["files"] if entry["kind"] == "noise"
    }
    environments = sorted(noise)
    speech_entries = sorted(
        (entry for entry in manifest["files"] if entry["kind"] == "speech"),
        key=lambda entry: entry["speaker"],
    )
    for index, entry in enumerate(speech_entries):
        speech = np.asarray(wavfile.read(root / entry["path"])[1], np.float64)
        speech *= 10.0 ** ((SPEECH_LEVEL_DBFS - _active_level_db(speech)) / 20.0)
        clean, labels = _with_pauses(speech, np.random.default_rng(zlib.crc32(entry["speaker"].encode())))
        pair = (environments[index % 8], environments[(index + 3) % 8])
        for environment in pair[:environments_per_speaker]:
            source = noise[environment]
            rng = np.random.default_rng(zlib.crc32(f"{entry['speaker']}:{environment}".encode()))
            start = int(rng.integers(0, source.size - clean.size))
            segment = source[start : start + clean.size]
            segment = segment / np.sqrt(np.mean(np.square(segment)))
            for snr in SNRS_DB:
                mix = clean + segment * 10.0 ** ((SPEECH_LEVEL_DBFS - snr) / 20.0)
                scale = min(1.0, 0.95 / np.max(np.abs(mix)))
                yield {
                    "speaker": entry["speaker"], "gender": entry["gender"],
                    "environment": environment, "snr_db": snr,
                    "clean": clean * scale, "noisy": mix * scale, "labels": labels,
                }


def split(root: Path) -> tuple[set[str], set[str]]:
    """Speaker-disjoint, gender-balanced train/test split."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    speakers = sorted((e["gender"], e["speaker"]) for e in manifest["files"] if e["kind"] == "speech")
    train = {speaker for index, (_, speaker) in enumerate(speakers) if index % 2 == 0}
    return train, {speaker for _, speaker in speakers} - train


def causal_control(probabilities: np.ndarray, sample_count: int) -> np.ndarray:
    """Hold, for each 10 ms gate block, the last window ended by its first sample."""
    from mic_eq.analysis.vad import map_causal_vad_probabilities

    count = (sample_count + FRAME - 1) // FRAME
    control = map_causal_vad_probabilities(probabilities, np.arange(count) * FRAME, SAMPLE_RATE)
    return np.zeros(count, np.float32) if control is None else control


def vad_control(noisy: np.ndarray) -> np.ndarray:
    from mic_eq import analyze_vad_probabilities

    probabilities = np.asarray(
        analyze_vad_probabilities(noisy.astype(np.float32), SAMPLE_RATE, 0.48), np.float32
    )
    return causal_control(probabilities, noisy.size)


def _frame_labels(labels: np.ndarray) -> np.ndarray:
    count = labels.size // FRAME
    return labels[: count * FRAME].reshape(count, FRAME).any(axis=1)


def score(case: dict[str, Any], output: np.ndarray) -> dict[str, float]:
    speech = _frame_labels(case["labels"])
    count = speech.size
    since = np.full(count, 10**6)
    until = np.full(count, 10**6)
    last, nxt = -(10**6), 10**6
    for index in range(count):
        last = index if speech[index] else last
        since[index] = index - last
    for index in range(count - 1, -1, -1):
        nxt = index if speech[index] else nxt
        until[index] = nxt - index
    pause = ~speech
    tail = pause & (since <= 30)
    deep = pause & (since >= 60) & (until >= 60)
    out_db = _frame_rms_db(output)[:count]
    in_db = _frame_rms_db(case["noisy"])[:count]
    clean_db = _frame_rms_db(case["clean"])[:count]
    onset_ratio = []
    for start in np.flatnonzero(speech[1:] & ~speech[:-1]) + 1:
        onset = np.zeros(count, bool)
        body = np.zeros(count, bool)
        onset[start : start + 10] = True
        body[start + 30 : start + 130] = True
        onset_ratio.append(
            (_energy_mean_db(out_db, onset) - _energy_mean_db(clean_db, onset))
            - (_energy_mean_db(out_db, body) - _energy_mean_db(clean_db, body))
        )
    return {
        "si_sdr": si_sdr(case["clean"], output),
        "estoi": estoi(case["clean"], output, SAMPLE_RATE),
        "pause_db": _energy_mean_db(out_db, pause) - _energy_mean_db(in_db, pause),
        "deep_pause_db": _energy_mean_db(out_db, deep) - _energy_mean_db(in_db, deep),
        "tail_db": _energy_mean_db(out_db, tail) - _energy_mean_db(in_db, tail),
        "floor_modulation_db": _energy_mean_db(out_db, tail) - _energy_mean_db(out_db, deep),
        "onset_db": float(np.median(onset_ratio)),
    }


def render(case: dict[str, Any], control: np.ndarray, *, suppressor_first: bool,
           gate_enabled: bool, gate_mode: int, model: str, vad_available: bool = True,
           presence_models: dict[str, list[float]] | None = None) -> np.ndarray:
    from mic_eq.mic_eq_core import simulate_gate_suppressor_order

    settings: dict[str, Any] = GATE_SETTINGS | {
        "gate_enabled": gate_enabled, "gate_mode": gate_mode, "noise_model": model,
        "vad_available": vad_available,
    }
    if presence_models is not None:
        settings["gate_spp_vad_assisted"] = tuple(presence_models["vad_assisted"])
        settings["gate_spp_vad_only"] = tuple(presence_models["vad_only"])
    result = simulate_gate_suppressor_order(
        case["noisy"].astype(np.float32), control.tolist(), suppressor_first, 1.0, settings,
    )
    return np.asarray(result["output_audio"], np.float64)


def _configure_deepfilter(models: list[str]) -> None:
    if any(model.startswith("deepfilter") for model in models):
        from mic_eq.mic_eq_core import configure_deepfilter_runtime_paths

        configure_deepfilter_runtime_paths(str(REPO_ROOT / "df.dll"), str(REPO_ROOT / "models"))


def _extension_identity() -> dict[str, str]:
    from mic_eq import mic_eq_core

    path = Path(mic_eq_core.__file__)
    return {"extension": path.name, "extension_sha256": _sha256(path)}


def rows(root: Path, split_name: str, models: list[str], environments_per_speaker: int,
         conditions: list[tuple[int, bool]],
         presence_models: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """Per-case metrics for one build: {model: {"speaker|env|snr|mode|vad": metrics}}."""
    _configure_deepfilter(models)
    train, test = split(root)
    speakers = train if split_name == "train" else test
    result: dict[str, Any] = {"identity": _extension_identity(), "rows": {m: {} for m in models}}
    for case in mixtures(root, environments_per_speaker):
        if case["speaker"] not in speakers:
            continue
        control = vad_control(case["noisy"])
        for model in models:
            for mode, available in conditions:
                output = render(case, control, suppressor_first=False, gate_enabled=True,
                                gate_mode=mode, model=model, vad_available=available,
                                presence_models=presence_models)
                key = f"{case['speaker']}|{case['environment']}|{case['snr_db']}|{mode}|{available}"
                result["rows"][model][key] = score(case, output) | {
                    "control_sha256": hashlib.sha256(control.tobytes()).hexdigest()
                }
    return result


def rows_with_build(build: Path, root: Path, args: list[str]) -> dict[str, Any]:
    """Run ``rows`` on corpus ``root`` in a subprocess importing ``mic_eq`` from ``build``."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "rows.json"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "rows", "--build", str(build),
             "--root", str(root), "--output", str(output), *args],
            check=True, env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parent)},
        )
        return json.loads(output.read_text(encoding="utf-8"))


def _mean_pause(table: dict[str, dict[str, float]], mode: int) -> float:
    return float(np.nanmean([v["pause_db"] for k, v in table.items() if k.endswith(f"|{mode}|True")]))


def calibrate(root: Path, incumbent: Path, weights: dict[str, list[float]]) -> dict[str, Any]:
    """Bisect each mode's bias to the incumbent's training pause level, per model."""
    conditions = ["--conditions", "1:True,2:True", "--environments-per-speaker", "1", "--split", "train"]
    target = rows_with_build(incumbent, root, ["--models", ",".join(QUALIFY_MODELS), *conditions])
    fused, vad_only = weights["fused"], weights["vad_only"]

    def match(model: str, mode: int) -> float:
        low, high = -6.0, 6.0
        for _ in range(9):
            bias = 0.5 * (low + high)
            models = {"vad_assisted": [fused[0], fused[1], bias], "vad_only": [vad_only[0], 0.0, bias]}
            table = rows(root, "train", [model], 1, [(mode, True)], models)["rows"][model]
            # A higher bias opens the gate more and makes pauses louder.
            if _mean_pause(table, mode) > _mean_pause(target["rows"][model], mode):
                high = bias
            else:
                low = bias
        return 0.5 * (low + high)

    _configure_deepfilter(list(QUALIFY_MODELS))
    jobs = [(model, mode) for model in QUALIFY_MODELS for mode in MODES]
    with ThreadPoolExecutor(len(jobs)) as pool:
        biases = list(pool.map(lambda job: match(*job), jobs))
    matched: dict[str, dict[str, float]] = {model: {} for model in QUALIFY_MODELS}
    for (model, mode), bias in zip(jobs, biases):
        matched[model][MODES[mode]] = bias
    adopted = {name: float(np.mean([matched[m][name] for m in QUALIFY_MODELS])) for name in MODES.values()}
    return {"matched_bias": matched, "adopted_bias": adopted,
            "incumbent_identity": target["identity"]}


def _verdict(deltas: dict[str, dict[str, dict[str, list[float]]]],
             invalid: dict[str, dict[str, int]], mode: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for model in QUALIFY_MODELS:
        for vad in ("available", "unavailable"):
            d = deltas[model][f"{mode}_{vad}"]
            prefix = f"{model}_{vad}"
            checks[f"{prefix}_results_complete"] = invalid[model][f"{mode}_{vad}"] == 0
            checks[f"{prefix}_estoi_noninferior"] = d["estoi"][1] >= -0.002
            checks[f"{prefix}_pause_not_louder"] = d["pause_db"][2] < 1.0
            checks[f"{prefix}_tail_not_louder"] = d["tail_db"][2] < 1.0
            checks[f"{prefix}_si_sdr_noninferior"] = d["si_sdr"][1] > -0.1
    d = deltas["rnnoise"][f"{mode}_available"]
    checks["rnnoise_material_win"] = (
        d["onset_db"][1] > 1.0 or d["estoi"][1] > 0.0 or d["pause_db"][2] < -1.0
    )
    return checks | {"adopt": all(checks.values())}


def compare_rows(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Paired deltas per model and condition; invalid results are counted, not dropped.

    A case is invalid when only one build has it, the builds saw different
    VAD control, a metric the rule reads is nonfinite for either build, or
    another metric is finite for one build only. Those other metrics are
    excluded where undefined for both builds (e.g. no deep pauses).
    """
    deltas: dict[str, dict[str, dict[str, list[float]]]] = {}
    means: dict[str, dict[str, dict[str, list[float]]]] = {}
    invalid: dict[str, dict[str, int]] = {}
    for model in QUALIFY_MODELS:
        deltas[model], means[model], invalid[model] = {}, {}, {}
        old_rows, new_rows = old["rows"].get(model, {}), new["rows"].get(model, {})
        for mode, name in MODES.items():
            for available in (True, False):
                label = f"{name}_{'available' if available else 'unavailable'}"
                suffix = f"|{mode}|{available}"
                old_keys = {k for k in old_rows if k.endswith(suffix)}
                new_keys = {k for k in new_rows if k.endswith(suffix)}
                keys = sorted(old_keys & new_keys)
                bad = set(old_keys ^ new_keys)
                bad |= {k for k in keys if old_rows[k]["control_sha256"] != new_rows[k]["control_sha256"]}
                for k in keys:
                    for metric in METRICS:
                        finite = [bool(np.isfinite(rows[k][metric])) for rows in (old_rows, new_rows)]
                        if (metric in RULE_METRICS and not all(finite)) or finite[0] != finite[1]:
                            bad.add(k)
                invalid[model][label] = len(bad) if keys else max(len(bad), 1)
                keys = [k for k in keys if k not in bad]
                clusters = np.array([k.split("|")[0] for k in keys])
                deltas[model][label], means[model][label] = {}, {}
                for metric in METRICS:
                    a = np.array([old_rows[k][metric] for k in keys], dtype=np.float64)
                    b = np.array([new_rows[k][metric] for k in keys], dtype=np.float64)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if not np.any(ok):
                        deltas[model][label][metric] = [float("nan")] * 3
                        means[model][label][metric] = [float("nan")] * 2
                        continue
                    deltas[model][label][metric] = cluster_ci(b[ok] - a[ok], clusters[ok])
                    means[model][label][metric] = [float(np.mean(a[ok])), float(np.mean(b[ok]))]
    verdict = {name: _verdict(deltas, invalid, name) for name in MODES.values()}
    return {"deltas": deltas, "means": means, "invalid_cases": invalid, "verdict": verdict}


def qualify(root: Path, incumbent: Path, candidate: Path) -> dict[str, Any]:
    args = ["--models", ",".join(QUALIFY_MODELS), "--split", "test", "--environments-per-speaker", "2",
            "--conditions", "1:True,2:True,1:False,2:False"]
    with ThreadPoolExecutor(2) as pool:
        old, new = pool.map(lambda build: rows_with_build(build, root, args), (incumbent, candidate))
    comparison = compare_rows(old, new)
    deltas, means, verdict = comparison["deltas"], comparison["means"], comparison["verdict"]
    manifest = root / "manifest.json"
    tools = ("python/tools/evaluate_gate_corpus.py", "python/tools/_eval_common.py")
    candidate_sources = ("rust-core/src/dsp/gate.rs", "rust-core/src/dsp/vad.rs")
    return {
        "schema_version": 1,
        "audible_change": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {name: "adopt" if v["adopt"] else "retain_incumbent" for name, v in verdict.items()},
        "predefined_rule": "evaluate_gate_corpus.py module docstring",
        "corpus": {
            "manifest_sha256": _sha256(manifest), "speakers_train_test": [12, 12],
            "environments": list(DEMAND_ENVIRONMENTS), "snr_db": list(SNRS_DB),
            "held_out_mixtures": 72,
        },
        "evaluation_contract": {
            "configuration": GATE_SETTINGS | {
                "noise_models": list(QUALIFY_MODELS), "order": "gate_before_suppressor",
                "vad_control": "causal: last 32 ms window ended by each block's first sample",
            },
            "asset_hashes": {
                "models/silero_vad.onnx": _sha256(REPO_ROOT / "models" / "silero_vad.onnx"),
                "corpus_manifest": _sha256(manifest),
            },
            "runtime": {
                "max_p99_frame_seconds": None,
                "max_p99_frame_seconds_reason": (
                    "Offline renders through the native simulator; the gate's per-sample "
                    "work is unchanged in kind and covered by the realtime tests."
                ),
            },
            "latency": {"added_latency_samples": 0, "reason": "Gain law only; no lookahead."},
            "clean_preservation": verdict,
        },
        "incumbent_identity": old["identity"],
        "candidate_identity": new["identity"],
        "delta_candidate_minus_incumbent": deltas,
        "incumbent_and_candidate_means": means,
        "invalid_cases": comparison["invalid_cases"],
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in tools},
        "candidate_source_sha256": {path: _sha256(candidate / path) for path in candidate_sources},
        "limitations": [
            "Objective metrics on one synthetic-mixture corpus; no listening test.",
            "Pauses are inserted between EARS phrases because EARS trims silences.",
            "The incumbent build is the committed source the candidate replaces.",
        ],
    }


def cluster_ci(deltas: np.ndarray, clusters: np.ndarray, seed: int = 0) -> list[float]:
    """Mean and speaker-clustered bootstrap 95% CI."""
    groups = [deltas[clusters == key] for key in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    means = [
        np.mean(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
        for _ in range(BOOTSTRAP)
    ]
    return [float(np.mean(deltas)), *map(float, np.percentile(means, [2.5, 97.5]))]


def order(root: Path, models: list[str], environments_per_speaker: int = 2) -> dict[str, Any]:
    _configure_deepfilter(models)
    arms = {"suppressor_only": (False, False, 0)}
    for mode in (0, 1):
        arms[f"gate_first_m{mode}"] = (False, True, mode)
        arms[f"suppressor_first_m{mode}"] = (True, True, mode)
    rows = []
    for case in mixtures(root, environments_per_speaker):
        control = vad_control(case["noisy"])
        for model in models:
            for arm, (first, gate, mode) in arms.items():
                output = render(case, control, suppressor_first=first, gate_enabled=gate,
                                gate_mode=mode, model=model)
                rows.append({"speaker": case["speaker"], "environment": case["environment"],
                             "snr_db": case["snr_db"], "model": model, "arm": arm} | score(case, output))
    summary: dict[str, Any] = {}
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        means = {arm: {m: float(np.nanmean([r[m] for r in subset if r["arm"] == arm])) for m in METRICS}
                 for arm in arms}
        deltas = {}
        for mode in (0, 1):
            first = {(r["speaker"], r["environment"], r["snr_db"]): r for r in subset if r["arm"] == f"gate_first_m{mode}"}
            later = {(r["speaker"], r["environment"], r["snr_db"]): r for r in subset if r["arm"] == f"suppressor_first_m{mode}"}
            keys = sorted(first)
            clusters = np.array([key[0] for key in keys])
            deltas[f"mode_{mode}"] = {}
            for metric in METRICS:
                values = np.array([later[k][metric] - first[k][metric] for k in keys])
                ok = np.isfinite(values)
                deltas[f"mode_{mode}"][metric] = cluster_ci(values[ok], clusters[ok])
        summary[model] = {"arm_means": means, "suppressor_first_minus_gate_first": deltas}
    return summary


def fit(root: Path) -> dict[str, list[float]]:
    """Fit p = sigmoid(w . [logit(VAD), level - floor, 1]) on training frames."""
    train, _ = split(root)
    features: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for case in mixtures(root):
        if case["speaker"] not in train:
            continue
        control = vad_control(case["noisy"])
        levels = _frame_rms_db(case["noisy"])
        vad = control[: levels.size]
        floor, history, floors = -60.0, [], np.empty(levels.size)
        for index, (level, prob) in enumerate(zip(levels, vad)):
            # Mirror of VadAutoGate: 20th percentile of low-VAD frames, slewed.
            if prob < 0.3 and level > -100.0:
                history = (history + [level])[-250:]
                ordered = np.sort(np.round(np.clip(history, -80, -20)))
                delta = ordered[min(len(ordered) - 1, int(len(ordered) * 0.2))] - floor
                floor = float(np.clip(floor + (min(delta, 0.5) if delta > 0 else max(delta, -0.1)), -80, -20))
            floors[index] = floor
        p = np.clip(vad, 1e-3, 1 - 1e-3)
        features.append((np.log(p / (1 - p)), levels - floors,
                         _frame_labels(case["labels"])[: levels.size].astype(float)))
    xv, xl, y = (np.concatenate([item[i] for item in features]) for i in range(3))

    def logistic(columns: list[np.ndarray]) -> list[float]:
        design = np.column_stack(columns + [np.ones(y.size)])

        def loss(weights: np.ndarray) -> tuple[float, np.ndarray]:
            prob = 1 / (1 + np.exp(-(design @ weights)))
            value = -np.mean(y * np.log(prob + 1e-9) + (1 - y) * np.log(1 - prob + 1e-9))
            return value, design.T @ (prob - y) / y.size

        return minimize(loss, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B").x.tolist()

    return {"fused": logistic([xv, xl]), "vad_only": logistic([xv]), "level_only": logistic([xl]),
            "train_speech_fraction": [float(y.mean())]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "fit", "calibrate", "qualify", "order", "rows"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", default="rnnoise", help="comma-separated noise models")
    parser.add_argument("--environments-per-speaker", type=int, choices=(1, 2), default=2)
    parser.add_argument("--incumbent", type=Path, help="incumbent build root")
    parser.add_argument("--candidate", type=Path, default=REPO_ROOT, help="candidate build root")
    parser.add_argument("--weights", type=Path, help="fit output for calibrate")
    parser.add_argument("--build", type=Path, help="rows: import mic_eq from this build root")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--conditions", default="1:True,2:True", help="rows: mode:vad_available,...")
    parser.add_argument("--output", type=Path, help="write the JSON result here")
    args = parser.parse_args(argv)
    if args.build is not None:
        sys.path.insert(0, str(args.build / "python"))
    if args.command == "fetch":
        fetch(args.root)
        return 0
    models = args.models.split(",")
    if args.command == "rows":
        conditions = [(int(m), v == "True") for m, v in (c.split(":") for c in args.conditions.split(","))]
        result = rows(args.root, args.split, models, args.environments_per_speaker, conditions)
    elif args.command == "calibrate":
        weights = json.loads(args.weights.read_text(encoding="utf-8"))
        result = calibrate(args.root, args.incumbent, weights)
    elif args.command == "qualify":
        result = qualify(args.root, args.incumbent, args.candidate)
    elif args.command == "order":
        result = order(args.root, models, args.environments_per_speaker)
    else:
        result = fit(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
