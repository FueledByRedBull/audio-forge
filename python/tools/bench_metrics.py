"""Measures, Voice Setup driver and mix construction for ``bench_report.py``.

Every measure keeps its definition at tag v1.18.0; each docstring names the
retired tool it comes from. Those evaluators (``evaluate_gate_corpus``,
``evaluate_voice_setup_refusals``, ``_eval_common`` and the rest) are no longer
in the tree: read one with ``git show v1.18.0:python/tools/<name>.py``, and see
the Retired evaluators table in ``evaluation/README.md``. The bench clips live in
``python/tests/fixtures/bench/``: clean takes and two noise segments are
committed, every mix is built from them by ``build_mix`` with the recipe in
``manifest.json`` and a fixed seed, and no mix is stored.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter, resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "python" / "tests" / "fixtures" / "bench"
SAMPLE_RATE = 48_000
FRAME = 480  # 10 ms
NOISE_S = 2.0  # Voice Setup's room-noise recording
LIMITING_HARD_DB = (6.0, 3.0)  # limiter and true-peak limiter gain reduction the app flags
SECTIONS = ("eq_settings", "gate_settings", "suppressor_settings", "deesser_settings",
            "compressor_settings", "limiter_settings")
STEPS = {"tone": ("eq_settings",), "gate_suppression": ("gate_settings", "suppressor_settings"),
         "deesser": ("deesser_settings",), "compressor": ("compressor_settings",)}
WHOLE_ORDER = ("B1", "B2", "B3", "B4", "B5", "C1", "C2")
VETOES = ("B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2")
VETO_NAMES = {
    "B1": "unclear recording", "B2": "weak capture", "B3": "compressor target missed", "B4": "final check failed",
    "B5": "Auto-EQ abstained", "B6": "tuner kept the current gate", "C1": "verification rolled back",
    "C2": "verification could not settle",
}
CAPTURE_REFUSALS = ("raw setup captures are non-finite or clipped", "Room-noise capture was too short for setup.",
                    "Voice capture was too short for setup.")
ACTIVE_BAND_DB = 0.25  # the Auto-EQ stage's own threshold for a band that corrects anything
# The clauses of "no worse than the current settings" as the retired refusal study read them
# (``evaluate_voice_setup_refusals.RULES`` at v1.18.0): measure -> (direction in which larger is better, tolerance).
CLAUSES = {"estoi": (1.0, 0.01), "tone_error_db": (-1.0, 1.0), "sibilant_loss_db": (-1.0, 3.0),
           "deep_pause_db": (-1.0, 1.0), "tail_db": (-1.0, 1.0), "onset_db": (1.0, 1.0)}
SLACK = 1e-9
RETAKE_STARTS_S = (3.0, 5.0, 7.0)  # where, in the noise of the take, the room-noise retakes begin
# A compressor that squashes the voice: the known-bad probe of the retired refusal study (20:1 from -60 dB, 0.5 ms).
SQUASH = {"enabled": True, "threshold_db": -60.0, "ratio": 20.0, "attack_ms": 0.5, "release_ms": 200.0,
          "makeup_gain_db": 0.0, "adaptive_release": False, "auto_makeup_enabled": False}
COMPRESSOR_ARMS = ("step", "off", "squash")  # the compressor step on its own: Voice Setup's, none, the known-bad control
RETRY_COSTS = {"verification_delivery_mismatch": "D1", "verification_capture_too_short": "D2",
               "verification_capture_invalid": "D2"}
TONE_FRAME, TONE_HOP, MIN_TONE_FRAMES = 4096, 2048, 8
# (pattern in the reason, veto), first match wins; "X" is the renderer being unavailable.
REASON_VETOES = (
    ("compressor p95 gain reduction target", "B3"),
    ("Auto-EQ abstained from this capture", "B5"),
    ("offline DSP validation could not run", "X"),
    ("full-chain validation failed", "B4"),
    ("full-chain gate/suppressor silenced", "B4"),
    ("raw full-chain capture is unavailable", "B4"),
    ("too little headroom", "B4"),
    ("final full-chain headroom validation", "B4"),
    ("offline DSP validation did not pass", "B4"),
    ("VAD-active speech", "B2"),
    ("speech-to-noise ratio", "B2"),
    ("voiced frames", "B2"),
    ("capture quality", "B2"),
    ("noise", "B2"),
    ("capture", "B2"),
    ("speech", "B2"),
)


# ----------------------------------------------------------------------------------------------
# Reference measures


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float64)
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(resample_poly(audio, target_rate // divisor, source_rate // divisor), dtype=np.float64)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB; ``_eval_common.si_sdr`` at v1.18.0."""
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    energy = float(np.dot(reference, reference))
    if energy <= 1e-12:
        return 0.0
    target = reference * (float(np.dot(estimate, reference)) / energy)
    residual = estimate - target
    return float(10.0 * np.log10((float(np.dot(target, target)) + 1e-12) / (float(np.dot(residual, residual)) + 1e-12)))


_ESTOI_RATE, _ESTOI_FRAME, _ESTOI_FFT, _ESTOI_SEGMENT = 10_000, 256, 512, 30


def _estoi_frames(signal: np.ndarray) -> np.ndarray:
    hop = _ESTOI_FRAME // 2
    count = 1 + (signal.size - _ESTOI_FRAME) // hop
    index = np.arange(_ESTOI_FRAME)[None, :] + hop * np.arange(count)[:, None]
    return signal[index] * np.hanning(_ESTOI_FRAME + 2)[1:-1][None, :]


def _overlap_add(frames: np.ndarray) -> np.ndarray:
    hop = _ESTOI_FRAME // 2
    out = np.zeros((frames.shape[0] - 1) * hop + _ESTOI_FRAME)
    for index, frame in enumerate(frames):
        out[index * hop : index * hop + _ESTOI_FRAME] += frame
    return out


def _third_octave_bands() -> np.ndarray:
    freqs = np.linspace(0, _ESTOI_RATE, _ESTOI_FFT + 1)[: _ESTOI_FFT // 2 + 1]
    centers = np.arange(15)
    low = 150.0 * 2.0 ** ((2 * centers - 1) / 6)
    high = 150.0 * 2.0 ** ((2 * centers + 1) / 6)
    bands = np.zeros((15, freqs.size))
    for band in range(15):
        bands[band, int(np.argmin(np.square(freqs - low[band]))) : int(np.argmin(np.square(freqs - high[band])))] = 1.0
    return bands


class _EstoiReference(NamedTuple):
    """The reference side of ESTOI: the frames that count and the reference's normalised band envelopes."""

    keep: np.ndarray
    segments: int
    normalized: np.ndarray | None  # None when the reference is too short for one segment


def _band_envelopes(signal: np.ndarray, bands: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(_estoi_frames(signal), n=_ESTOI_FFT, axis=1).T
    return np.sqrt(bands @ np.square(np.abs(spectrum)))


def _normalized_segments(envelopes: np.ndarray, segments: int) -> np.ndarray:
    seg = np.stack([envelopes[:, m : m + _ESTOI_SEGMENT] for m in range(segments)])
    seg = seg - seg.mean(axis=2, keepdims=True)
    seg = seg / (np.linalg.norm(seg, axis=2, keepdims=True) + 1e-20)
    seg = seg - seg.mean(axis=1, keepdims=True)
    return seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-20)


def estoi_reference(reference: np.ndarray, sample_rate: int) -> _EstoiReference:
    """Everything of ``estoi`` that depends on the reference alone, so one clean take serves every render."""
    ref_frames = _estoi_frames(resample_audio(reference, sample_rate, _ESTOI_RATE))
    energy = 20.0 * np.log10(np.linalg.norm(ref_frames, axis=1) + 1e-20)
    keep = energy > np.max(energy) - 40.0
    ref_bands = _band_envelopes(_overlap_add(ref_frames[keep]), _third_octave_bands())
    segments = ref_bands.shape[1] - _ESTOI_SEGMENT + 1
    return _EstoiReference(keep, segments, _normalized_segments(ref_bands, segments) if segments >= 1 else None)


def estoi_against(reference: _EstoiReference, estimate: np.ndarray, sample_rate: int) -> float:
    """Extended STOI (Jensen & Taal, 2016) of ``estimate`` against a prepared reference."""
    if reference.normalized is None:
        return float("nan")
    est_frames = _estoi_frames(resample_audio(estimate, sample_rate, _ESTOI_RATE))
    est_bands = _band_envelopes(_overlap_add(est_frames[reference.keep]), _third_octave_bands())
    normalized = _normalized_segments(est_bands, reference.segments)
    return float(np.sum(reference.normalized * normalized) / (_ESTOI_SEGMENT * reference.segments))


def estoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    """Extended STOI (Jensen & Taal, 2016); ``_eval_common.estoi`` at v1.18.0."""
    return estoi_against(estoi_reference(reference, sample_rate), estimate, sample_rate)


def frame_rms_db(audio: np.ndarray) -> np.ndarray:
    count = audio.size // FRAME
    frames = audio[: count * FRAME].reshape(count, FRAME)
    return 10.0 * np.log10(np.mean(np.square(frames), axis=1) + 1e-20)


def active_level_db(speech: np.ndarray) -> float:
    """Energy mean of the frames within 40 dB of the loudest; ``evaluate_gate_corpus._active_level_db``."""
    levels = frame_rms_db(speech)
    return _energy_mean_db(levels, levels > np.max(levels) - 40.0)


def _energy_mean_db(levels_db: np.ndarray, where: np.ndarray | slice) -> float:
    """Energy mean of the levels a boolean mask or a slice picks; NaN when it picks none."""
    picked = levels_db[where]
    if picked.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(10.0 ** (picked / 10.0))))


def _shared(case: dict[str, Any], key: str, compute: Callable[[], Any]) -> Any:
    """A reading of the input and clean take that ``make_context`` computed once, or computed now for a bare case."""
    return case[key] if key in case else compute()


def frame_labels(labels: np.ndarray) -> np.ndarray:
    count = labels.size // FRAME
    return labels[: count * FRAME].reshape(count, FRAME).any(axis=1)


def pause_frames(speech: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pause frames within 300 ms after speech (tail) and those at least 600 ms from speech on both sides (deep)."""
    index = np.arange(speech.size)
    since = index - np.maximum.accumulate(np.where(speech, index, -(10**6)))
    until = np.minimum.accumulate(np.where(speech, index, 10**6)[::-1])[::-1] - index
    pause = ~speech
    return pause & (since <= 30), pause & (since >= 60) & (until >= 60)


def score(case: dict[str, Any], output: np.ndarray) -> dict[str, float]:
    """ESTOI, SI-SDR, onset, pause, deep-pause (pause noise) and tail readings; ``evaluate_gate_corpus.score``.

    Pauses are the frames without labelled speech; deep pauses are at least 600 ms from speech on both sides, tail
    frames within 300 ms after speech; onset is the first 100 ms of a phrase relative to its body, against the clean
    speech.
    """
    speech = frame_labels(case["labels"])
    count = speech.size
    pause = ~speech
    tail, deep = pause_frames(speech)
    out_db = frame_rms_db(output)[:count]
    in_db = _shared(case, "noisy_db", lambda: frame_rms_db(case["noisy"]))[:count]
    clean_db = _shared(case, "clean_db", lambda: frame_rms_db(case["clean"]))[:count]
    onset_ratio = []
    for start in np.flatnonzero(speech[1:] & ~speech[:-1]) + 1:
        if start + 130 > count:
            continue
        onset, body = slice(start, start + 10), slice(start + 30, start + 130)
        onset_ratio.append((_energy_mean_db(out_db, onset) - _energy_mean_db(clean_db, onset))
                           - (_energy_mean_db(out_db, body) - _energy_mean_db(clean_db, body)))
    reference = _shared(case, "estoi_reference", lambda: estoi_reference(case["clean"], SAMPLE_RATE))
    return {
        "si_sdr": si_sdr(case["clean"], output),
        "estoi": estoi_against(reference, output, SAMPLE_RATE),
        "pause_db": _energy_mean_db(out_db, pause) - _energy_mean_db(in_db, pause),
        "deep_pause_db": _energy_mean_db(out_db, deep) - _energy_mean_db(in_db, deep),
        "tail_db": _energy_mean_db(out_db, tail) - _energy_mean_db(in_db, tail),
        "onset_db": float(np.median(onset_ratio)) if onset_ratio else float("nan"),
    }


def deep_pause_seconds(labels: np.ndarray) -> float:
    """Seconds of deep-pause frames (at least 600 ms from speech) that ``score`` can read."""
    return float(np.count_nonzero(pause_frames(frame_labels(labels))[1])) * FRAME / SAMPLE_RATE


def level_matched(case: dict[str, Any], output: np.ndarray) -> np.ndarray:
    """``output`` scaled so its speech frames have the input's energy; ``evaluate_voice_setup_refusals.level_matched``."""
    speech = frame_labels(case["labels"])
    out_db, in_db = frame_rms_db(output), _shared(case, "noisy_db", lambda: frame_rms_db(case["noisy"]))
    count = min(speech.size, out_db.size, in_db.size)
    gain = _energy_mean_db(out_db[:count], speech[:count]) - _energy_mean_db(in_db[:count], speech[:count])
    return output * 10.0 ** (-gain / 20.0) if math.isfinite(gain) else output


def scored(case: dict[str, Any], output: np.ndarray) -> dict[str, float]:
    """The ``score`` readings of a render after level matching; ``evaluate_voice_setup_refusals.scored``."""
    return score(case, level_matched(case, output))


def tone_error_db(clean: np.ndarray, output: np.ndarray, labels: np.ndarray) -> float:
    """RMS over 80-12000 Hz of the median-centred long-term speech spectrum difference; refusals ``tone_error_db``."""
    from mic_eq.analysis.spectrum import _frame_spectra_db

    count = min(clean.size, output.size, labels.size)
    view = np.lib.stride_tricks.sliding_window_view
    clean_frames, output_frames = (view(np.asarray(x[:count], np.float64), TONE_FRAME)[::TONE_HOP] for x in (clean, output))
    speech = view(np.asarray(labels[:count], bool), TONE_FRAME)[::TONE_HOP].all(axis=1)
    level = 10.0 * np.log10(np.mean(np.square(clean_frames), axis=1) + 1e-20)
    keep = speech & (level >= level[speech].max() - 25.0) if speech.any() else speech
    if np.count_nonzero(keep) < MIN_TONE_FRAMES:
        return float("nan")
    spectra = [_frame_spectra_db(frames[keep], SAMPLE_RATE) for frames in (clean_frames, output_frames)]
    if spectra[0] is None or spectra[1] is None:
        return float("nan")
    delta = np.median(spectra[1][1], axis=0) - np.median(spectra[0][1], axis=0)
    band = delta[(spectra[0][0] >= 80.0) & (spectra[0][0] <= 12_000.0)]
    return float(np.sqrt(np.mean(np.square(band - np.median(band)))))


def measure(rendered: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Every reading of one render of a mix; ``evaluate_voice_setup_refusals.measure`` plus output checks.

    Sibilant loss is the dulling of 5-9 kHz in the input's sibilant frames relative to its voiced frames, as the
    product's own advice reads it (``voice_setup._sibilant_change_db``). Clipping is a sample at or above 1.0 in the
    unscaled render; limiter overload is a gain reduction of 6 dB (limiter) or 3 dB (true-peak limiter).
    """
    from mic_eq.analysis.voice_setup import _sibilant_change_db

    output = np.asarray(rendered["output_audio"], np.float64)[: ctx["clean"].size]
    change, _ = _sibilant_change_db(ctx["noisy"], output, ctx["noise"], SAMPLE_RATE)
    peak = float(np.max(np.abs(output)))
    limiter = float(rendered["limiter_gain_reduction_db"])
    true_peak = float(rendered["true_peak_limiter_gain_reduction_db"])
    return {
        **scored(ctx, output), "tone_error_db": tone_error_db(ctx["clean"], output, ctx["labels"]),
        "sibilant_loss_db": None if change is None else -float(change),
        "finite": bool(np.isfinite(output).all()), "peak": peak,
        "clipped": bool(peak >= 1.0),
        "limiter_gain_reduction_db": limiter, "true_peak_limiter_gain_reduction_db": true_peak,
        "pre_limiter_true_peak_headroom_db": float(rendered["pre_limiter_true_peak_headroom_db"]),
        "chain_latency_samples": int(rendered["chain_latency_samples"]),
        "limiter_overload": bool(limiter >= LIMITING_HARD_DB[0] or true_peak >= LIMITING_HARD_DB[1]),
    }


def clause_results(candidate: dict[str, Any], current: dict[str, Any]) -> dict[str, str]:
    """``pass``, ``fail`` or ``unmeasurable`` for each clause of a setup against the current settings.

    The overload clause fails when the candidate clips, or reaches a limiter overload, where the current settings do
    not. A render that failed has no readings, so none of its clauses can be read.
    """
    results = {}
    for name, (direction, tolerance) in CLAUSES.items():
        value, baseline = candidate.get(name, math.nan), current.get(name, math.nan)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (value, baseline)):
            results[name] = "unmeasurable"
        else:
            results[name] = "fail" if direction * (value - baseline) < -tolerance - SLACK else "pass"
    if any("error" in reading for reading in (candidate, current)):
        results["overload"] = "unmeasurable"
    else:
        worse = any(candidate[key] and not current[key] for key in ("clipped", "limiter_overload"))
        results["overload"] = "fail" if worse else "pass"
    return results


def failed_clauses(candidate: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """The clauses a setup fails against the current settings; none for a setup that was not rendered."""
    if candidate is None:
        return []
    return [name for name, result in clause_results(candidate, current).items() if result == "fail"]


# ----------------------------------------------------------------------------------------------
# Settings and chain


def current_settings(vad_available: bool) -> dict[str, Any]:
    """The factory ``Preset()`` as the six sections a chain render reads; refusals ``current_settings``."""
    from mic_eq.config import EQSettings, Preset

    preset = Preset()
    return {"eq_settings": EQSettings().to_dict(), "gate_settings": asdict(preset.gate),
            "suppressor_settings": asdict(preset.rnnoise), "deesser_settings": asdict(preset.deesser),
            "compressor_settings": asdict(preset.compressor), "limiter_settings": asdict(preset.limiter),
            "diagnostics": {"vad_available": bool(vad_available)}}


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items() if "runtime" not in str(key)}
    if isinstance(value, (list, tuple)):
        return [_without_timing(item) for item in value]
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_without_timing(value), sort_keys=True, default=_plain).encode()).hexdigest()


def applied_eq(eq_settings: dict[str, Any]) -> dict[str, Any]:
    """The typed EQ that the final check renders: candidate bands over the default EQ."""
    from mic_eq.config import EQSettings, build_eq_candidate_settings

    if "bands" in eq_settings:
        return dict(eq_settings)
    typed = build_eq_candidate_settings(
        EQSettings(), eq_settings["band_freqs"], eq_settings["band_gains"], eq_settings["band_qs"],
        layer="correction", enabled=bool(eq_settings.get("enabled", True)),
    ).to_dict()
    return {**eq_settings, **{key: typed[key] for key in ("schema_version", "enabled", "bands", "layers") if key in typed}}


def chain_of(result: dict[str, Any]) -> dict[str, Any]:
    """The full chain of a proposal, as the dialog snapshots it for verification."""
    return {
        "gate": dict(result["gate_settings"]), "rnnoise": dict(result["suppressor_settings"]),
        "deesser": dict(result["deesser_settings"]), "compressor": dict(result["compressor_settings"]),
        "limiter": dict(result["limiter_settings"]), "full_chain": True, "input_pre_filtered": False,
        "input_cleanup_mode": "off", "processing_mode": "normal",
        "vad_available": bool(result["diagnostics"]["vad_available"]),
    }


def render(settings: dict[str, Any], capture: np.ndarray, chain: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render a raw capture through the offline full chain of ``settings``; ``chain`` adds native chain keys
    (for instance ``vad_probabilities`` or ``input_cleanup_mode``)."""
    from mic_eq.analysis.auto_eq import simulate_candidate_chain

    return simulate_candidate_chain(
        np.ascontiguousarray(capture, dtype=np.float32), SAMPLE_RATE, applied_eq(settings["eq_settings"]),
        {**chain_of(settings), **(chain or {}), "return_output_audio": True},
    )


def measure_settings(settings: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Render ``settings`` once and read it; a render that fails is recorded, not hidden (a measure that fails is a fault)."""
    try:
        rendered = render(settings, ctx["raw"])
    except (RuntimeError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return measure(rendered, ctx)


def compressor_only(settings: dict[str, Any], compressor: dict[str, Any]) -> dict[str, Any]:
    """The chain with nothing on but ``compressor``: what that compression alone does to a signal."""
    off = {"enabled": False}
    return {**settings, "eq_settings": {**settings["eq_settings"], **off}, "gate_settings": {**settings["gate_settings"], **off},
            "suppressor_settings": {**settings["suppressor_settings"], **off},
            "deesser_settings": {**settings["deesser_settings"], **off}, "limiter_settings": {**settings["limiter_settings"], **off},
            "compressor_settings": compressor}


def compression_aware(ctx: dict[str, Any], compressor: dict[str, Any], vad_available: bool) -> dict[str, Any]:
    """``ctx`` with the clean take and the input as ``compressor`` alone leaves them.

    A compressor lowers the loud parts of the voice on purpose, so against the unprocessed take it narrows the
    onset-to-body difference and lifts the pauses and tails that follow speech, and the clause readings take that for
    harm. Read against this context, a render is judged for what it does beyond the compression it was asked for.
    """
    only = compressor_only(current_settings(vad_available), compressor)
    clean, noisy = (np.asarray(render(only, audio)["output_audio"], np.float64) for audio in (ctx["raw_clean"], ctx["raw"]))
    aware = {**ctx, "clean": clean[: ctx["clean"].size], "noisy": noisy[: ctx["noisy"].size]}
    aware["noisy_db"], aware["clean_db"] = frame_rms_db(aware["noisy"]), frame_rms_db(aware["clean"])
    aware["estoi_reference"] = estoi_reference(aware["clean"], SAMPLE_RATE)
    return aware


def fair_settings_readings(settings: dict[str, Any], ctx: dict[str, Any], vad_available: bool) -> dict[str, Any]:
    """The readings of ``settings`` against its own compressor's compression of the clean take and the input."""
    return measure_settings(settings, compression_aware(ctx, settings["compressor_settings"], vad_available))


def compressor_arms(current: dict[str, Any], proposed_compressor: dict[str, Any], ctx: dict[str, Any],
                    vad_available: bool) -> dict[str, dict[str, Any]]:
    """The compressor step on its own against the current settings: the proposed compressor, no compressor, and a
    squashing one as the known-bad control, each read plain and against its own compression (see ``compression_aware``)."""
    arms = {"step": proposed_compressor, "off": {**current["compressor_settings"], "enabled": False},
            "squash": {**current["compressor_settings"], **SQUASH}}
    out = {}
    for name in COMPRESSOR_ARMS:
        settings = {**current, "compressor_settings": arms[name]}
        out[name] = {"plain": measure_settings(settings, ctx), "fair": fair_settings_readings(settings, ctx, vad_available)}
    return out


def is_vacuous(withheld: dict[str, Any], current: dict[str, Any], sections: tuple[str, ...]) -> bool:
    """Whether the sections change nothing the current ones do not (a flat EQ counts as unchanged)."""
    for name in sections:
        if name == "eq_settings":
            gains = withheld[name].get("band_gains", ())
            if max((abs(float(gain)) for gain in gains), default=0.0) >= ACTIVE_BAND_DB:
                return False
        elif digest(withheld[name]) != digest(current[name]):
            return False
    return True


def ended_with(voice_setup_entry: dict[str, Any]) -> str | None:
    """Which readings are the setup the user ends with: ``final`` once applied, else the ``proposal``; None if none."""
    return "final" if voice_setup_entry.get("final") else "proposal" if voice_setup_entry.get("proposal") else None


def changed_steps(proposal: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """The steps (tone, gate_suppression, deesser, compressor) whose sections differ from the current settings."""
    return [step for step, sections in STEPS.items() if not is_vacuous(proposal, current, sections)]


# ----------------------------------------------------------------------------------------------
# Voice Setup


def verify_flow(result: dict[str, Any], captures: dict[str, Any]) -> dict[str, Any]:
    """The dialog's verification of a proposal with its automatic reductions; refusals ``verify_flow``.

    ``captures["takes"]`` are the second takes the user can record, as (filtered, raw) pairs. A ``retry`` asks for
    the next one and the third bad take stops with no verdict; ``gave_up`` is a reduction that cannot be settled.
    """
    from mic_eq.analysis.voice_setup import (
        MAX_VERIFICATION_BAD_CAPTURES,
        reduce_for_verification,
        reducible_targets,
        reduction_metrics,
        validate_voice_setup_verification,
        verification_stop,
    )

    state = deepcopy(result)
    state["eq_settings"] = applied_eq(state["eq_settings"])
    takes: list[tuple[np.ndarray, np.ndarray]] = captures["takes"]
    take, retakes, adjustments, last, first, ended = 0, [], 0, None, None, None
    while True:
        state["verification_chain_settings"] = chain_of(state)
        outcome = validate_voice_setup_verification(
            captures["noise"], captures["setup"], takes[take][0], SAMPLE_RATE, state, "broadcast",
            raw_noise_audio=captures["raw_noise"], raw_verification_speech_audio=takes[take][1])
        first = first or outcome["decision"]
        if outcome["decision"] == "retry":
            retakes.append(next((RETRY_COSTS[c] for c in outcome.get("reason_codes") or () if c in RETRY_COSTS), "unknown"))
            if len(retakes) >= MAX_VERIFICATION_BAD_CAPTURES or take + 1 >= len(takes):
                ended = "bad_takes"
                break
            take += 1
            continue
        if outcome["decision"] != "reduce":
            break
        targets = reducible_targets(outcome.get("reduction_targets"))
        metrics = reduction_metrics(outcome)
        ended = verification_stop(adjustments, targets, last, metrics)
        if ended:
            break
        before = deepcopy((state["compressor_settings"], state["deesser_settings"]))
        reduction = reduce_for_verification(state["compressor_settings"], state["deesser_settings"], targets)
        state["compressor_settings"], state["deesser_settings"] = reduction.compressor, reduction.deesser
        if metrics:
            last = metrics
        if before == (state["compressor_settings"], state["deesser_settings"]):
            ended = "no_change"
            break
        adjustments += 1
    return {"decision": "gave_up" if outcome["decision"] == "reduce" else outcome["decision"], "first_decision": first,
            "adjustments": adjustments, "ended": ended, "retakes": retakes, "takes": take + 1, "state": state}


@contextmanager
def _unclear_recording_tap() -> Iterator[list[dict[str, Any]]]:
    """Record the Auto-EQ validation verdicts and let the proposal through.

    The unclear-recording refusal (B1) throws its proposal away; the wrapper records the real verdict, then passes,
    so ``analyze_voice_setup`` goes on and the bench can still measure what the proposal would have sounded like.
    """
    from mic_eq.analysis import failure_detection

    log: list[dict[str, Any]] = []
    validate = failure_detection.validate_analysis

    def tapped(eq_settings: Any, spectrum_db: Any, freqs: Any) -> Any:
        real = validate(eq_settings, spectrum_db, freqs)
        log.append({"passed": bool(real.passed), "reason": real.reason})
        return failure_detection.ValidationResult(passed=True, reason=None, details=dict(real.details))

    failure_detection.validate_analysis = tapped
    try:
        yield log
    finally:
        failure_detection.validate_analysis = validate


def reason_veto(reason: str) -> str | None:
    for pattern, veto in REASON_VETOES:
        if pattern in reason:
            return veto
    return None


def first_whole(fired: set[str] | list[str], recommended: bool) -> str | None:
    """The veto that refuses the whole setup; C vetoes count only for a setup the analysis recommended."""
    for veto in WHOLE_ORDER:
        if veto in fired and (recommended or not veto.startswith("C")):
            return veto
    return None


def fired_vetoes(reasons: list[str], weak_capture: bool, b1: bool, tuner_decision: str | None,
                 verification: str | None) -> list[str]:
    """The vetoes of one analysis; refusals ``fired_vetoes``."""
    fired = {reason_veto(reason) for reason in reasons} - {None, "X", "B2"}
    flags = {"B1": b1, "B2": weak_capture, "B6": tuner_decision == "retain_incumbent",
             "C1": verification == "rollback", "C2": verification == "gave_up"}
    fired |= {veto for veto, flag in flags.items() if flag}
    return [veto for veto in VETOES if veto in fired]


def silero_available() -> bool:
    """Whether the offline VAD runs as Silero here (``VAD_MODEL_PATH`` or the checkout's ``models/``)."""
    from mic_eq.analysis.vad import analyze_offline_vad

    if "VAD_MODEL_PATH" not in os.environ and (REPO_ROOT / "models" / "silero_vad.onnx").is_file():
        os.environ["VAD_MODEL_PATH"] = str(REPO_ROOT / "models" / "silero_vad.onnx")
    try:
        return analyze_offline_vad(np.zeros(SAMPLE_RATE // 2, np.float32), SAMPLE_RATE)[1] == "silero"
    except Exception:  # a model that cannot load is "unavailable", whatever the reason
        return False


def make_context(mix: dict[str, Any]) -> dict[str, Any]:
    """The raw mix, the pre-filtered noisy input, clean reference and noise capture the measures read, and what
    every render of the mix shares: the frame levels of the input and the clean take, and ESTOI's reference side.

    ``retakes`` are 2 s stretches of the noise that played during the take, as the room-noise recording would be if
    the person were asked to record it again with the same sound still playing (fewer for a short take).
    """
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    def filtered(audio: np.ndarray) -> np.ndarray:
        return filtered_capture_for_analysis(audio, SAMPLE_RATE).astype(np.float64)

    during_take = mix["raw"].astype(np.float64) - mix["clean"]
    retakes = [during_take[int(s * SAMPLE_RATE) : int((s + NOISE_S) * SAMPLE_RATE)].astype(np.float32)
               for s in RETAKE_STARTS_S if int((s + NOISE_S) * SAMPLE_RATE) <= during_take.size]
    ctx = {"raw": mix["raw"], "labels": mix["labels"], "raw_noise": mix["noise_capture"], "raw_clean": mix["clean"],
           "noise": filtered(mix["noise_capture"]), "clean": filtered(mix["clean"]), "noisy": filtered(mix["raw"]),
           "retakes": [filtered(take).astype(np.float32) for take in retakes]}
    ctx["noisy_db"], ctx["clean_db"] = frame_rms_db(ctx["noisy"]), frame_rms_db(ctx["clean"])
    ctx["estoi_reference"] = estoi_reference(ctx["clean"], SAMPLE_RATE)
    return ctx


def retake_statuses(ctx: dict[str, Any], vad_available: bool) -> list[str]:
    """The room-noise status of each retake of the mix, judged against its voice capture as the analysis does."""
    from mic_eq.analysis.noise_reference import analyze_noise_reference
    from mic_eq.analysis.vad import analyze_offline_vad

    def vad(audio: np.ndarray) -> np.ndarray | None:
        return analyze_offline_vad(audio, SAMPLE_RATE)[0] if vad_available else None

    setup = ctx["noisy"].astype(np.float32)
    speech_vad = vad(setup)
    return [analyze_noise_reference(take, setup, SAMPLE_RATE, noise_vad_probabilities=vad(take),
                                    speech_vad_probabilities=speech_vad).status for take in ctx["retakes"]]


def voice_setup(ctx: dict[str, Any], vad_available: bool) -> dict[str, Any]:
    """Run ``analyze_voice_setup`` and the dialog's verification on a mix, as the app's offline flow does.

    The mix is both the speech recording and the second take, and the noise capture is the mix's own room noise.
    Returns the outcome (``applied`` or ``refused`` with the veto, ``failed`` when Auto-EQ could not run), the
    steps the proposal changes, and the readings of the proposal and of what the user ends with.
    """
    from mic_eq.analysis.voice_setup import analyze_voice_setup
    from mic_eq.config import EQSettings, Preset

    raw_noise, raw = ctx["raw_noise"], ctx["raw"]
    # The context holds the filtered captures widened to float64; narrowing them gives back the filter's own float32.
    noise, setup = ctx["noise"].astype(np.float32), ctx["noisy"].astype(np.float32)
    preset = Preset()
    started = time.process_time()
    with _unclear_recording_tap() as validation:
        try:
            result = analyze_voice_setup(
                noise, setup, SAMPLE_RATE, "broadcast", vad_available=vad_available,
                incumbent_settings={"gate": asdict(preset.gate), "rnnoise": asdict(preset.rnnoise),
                                    "eq": EQSettings().to_dict()},
                raw_noise_audio=raw_noise, raw_speech_audio=raw)
        except ValueError as error:
            if str(error) not in CAPTURE_REFUSALS:  # anything else is a fault
                raise
            return {"outcome": "refused", "veto": "capture", "vetoes": [], "reason": str(error), "steps": [],
                    "cpu_s": time.process_time() - started, "proposal": None, "final": None, "suppressor": None}
    diagnostics = result["diagnostics"]
    if result["eq_settings"] is None:
        return {"outcome": "failed", "veto": None, "vetoes": [], "reason": str(result["eq_error"]), "steps": [],
                "cpu_s": time.process_time() - started, "proposal": None, "final": None, "suppressor": None}
    b1 = any(not entry["passed"] for entry in validation)
    vad = bool(diagnostics["vad_available"])
    current = current_settings(vad)
    proposal = {name: deepcopy(result[name]) for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}
    flow = None if b1 else verify_flow(result, {"noise": noise, "setup": setup, "raw_noise": raw_noise,
                                                  "takes": [(setup, raw)]})
    tuning = diagnostics["joint_tuning"]
    fired = fired_vetoes(list(diagnostics["uncertainty_reasons"]), bool(diagnostics["weak_capture"]), b1,
                         tuning.get("decision"), None if flow is None else flow["decision"])
    recommended = bool(diagnostics["apply_recommended"])
    veto = first_whole(fired, recommended)
    unverified = flow is not None and flow["decision"] == "retry"
    outcome = "refused" if veto else "unverified" if unverified and recommended else "applied"
    final = None
    if flow is not None and outcome == "applied":
        final = {name: flow["state"][name] for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}
    cpu = time.process_time() - started
    reference = {k: diagnostics["noise_reference_quality"][k] for k in ("status", "reasons")}
    if reference["status"] != "usable":
        reference["retakes"] = retake_statuses(ctx, vad)
    ends = final or proposal
    return {
        "outcome": outcome, "veto": veto, "vetoes": fired, "reason": "; ".join(diagnostics["uncertainty_reasons"]),
        "steps": changed_steps(ends, current), "cpu_s": cpu, "noise_reference": reference,
        "tuner": tuning.get("status"),
        "settings": {"proposal": proposal, "final": final},
        "fair": {"proposal": fair_settings_readings(proposal, ctx, vad),
                 "final": None if final is None else fair_settings_readings(final, ctx, vad)},
        "compressor_step": compressor_arms(current, proposal["compressor_settings"], ctx, vad),
        "proposal": measure_settings(proposal, ctx),
        "suppressor": {k: proposal["suppressor_settings"].get(k) for k in ("enabled", "model", "strength")},
        "final": None if final is None else measure_settings(final, ctx),
        "verification": None if flow is None else {k: flow[k] for k in ("decision", "first_decision", "adjustments",
                                                                            "ended", "takes", "retakes")},
        "vad_backend": diagnostics.get("vad_analysis_backend"),
        "limiter_allowance_db": diagnostics.get("limiter_allowance_db"),
        "offline_headroom_safe": bool(diagnostics.get("offline_headroom_safe")),
    }


# ----------------------------------------------------------------------------------------------
# Bench set and mix construction


@dataclass(frozen=True)
class Bench:
    manifest: dict[str, Any]
    takes: dict[str, np.ndarray]
    noises: dict[str, np.ndarray]
    sha256: dict[str, str]  # of each audio file as read, by file name


def _decode_wav(data: bytes, name: str) -> np.ndarray:
    rate, audio = wavfile.read(io.BytesIO(data))
    if rate != SAMPLE_RATE or audio.dtype != np.int16 or audio.ndim != 1:
        raise ValueError(f"{name} must be 48 kHz mono 16-bit")
    return audio.astype(np.float64) / 32768.0


def read_wav(path: Path) -> np.ndarray:
    return _decode_wav(path.read_bytes(), path.name)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(root: Path = BENCH_DIR) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


@functools.cache
def load_bench(root: Path = BENCH_DIR) -> Bench:
    """The bench set, read once per folder: the clips are shared and read-only, and each file is hashed as it is read."""
    manifest = read_manifest(root)
    sha256: dict[str, str] = {}

    def load(entries: dict[str, Any]) -> dict[str, np.ndarray]:
        audio = {}
        for name, entry in entries.items():
            data = (root / entry["file"]).read_bytes()
            sha256[entry["file"]] = hashlib.sha256(data).hexdigest()
            audio[name] = _decode_wav(data, entry["file"])
            audio[name].setflags(write=False)
        return audio

    return Bench(manifest, load(manifest["takes"]), load(manifest["noises"]), sha256)


def speech_labels(entry: dict[str, Any], size: int) -> np.ndarray:
    labels = np.zeros(size, bool)
    for start, stop in entry["speech_samples"]:
        labels[start:stop] = True
    return labels


def unit_rms(audio: np.ndarray) -> np.ndarray:
    return audio / math.sqrt(float(np.mean(np.square(audio))))


def pink(count: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1.0 / SAMPLE_RATE)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0.0
    return unit_rms(np.fft.irfft(spectrum, count))


def fan(count: int, rng: np.random.Generator) -> np.ndarray:
    """Broadband noise shaped like a case fan (low-passed, tilted), blade tones and a 0.35 Hz level swell."""
    time_s = np.arange(count) / SAMPLE_RATE
    spectrum = np.fft.rfft(rng.standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1.0 / SAMPLE_RATE)
    spectrum *= (1.0 + freqs / 150.0) ** -0.5 / np.sqrt(1.0 + (freqs / 1_500.0) ** 4)
    body = unit_rms(np.fft.irfft(spectrum, count))
    tones = sum(level * np.sin(2 * np.pi * 120.0 * h * time_s + rng.uniform(0, 2 * np.pi))
                for h, level in ((1, 0.5), (2, 0.25), (3, 0.12)))
    swell = 1.0 + 0.3 * np.sin(2 * np.pi * 0.35 * time_s + rng.uniform(0, 2 * np.pi))
    return unit_rms((body + tones) * swell)


def keyboard(count: int, rng: np.random.Generator) -> np.ndarray:
    """Poisson key strokes (8 per second): a bright 8 ms click and a low 'thock', over a faint pink floor."""
    out = 0.03 * pink(count, rng)
    size = int(0.03 * SAMPLE_RATE)
    time_s = np.arange(size) / SAMPLE_RATE
    position = 0.0
    while True:
        position += rng.exponential(1.0 / 8.0)
        start = int(position * SAMPLE_RATE)
        if start + size > count:
            break
        click = np.diff(rng.standard_normal(size + 1)) * np.exp(-time_s / 0.0012)
        thock = 0.4 * np.sin(2 * np.pi * rng.uniform(180.0, 320.0) * time_s) * np.exp(-time_s / 0.006)
        out[start : start + size] += rng.lognormal(0.0, 0.4) * (click + thock)
    return unit_rms(out)


def hum(count: int, rng: np.random.Generator, fundamental_hz: float) -> np.ndarray:
    """Mains hum: twelve harmonics falling as 1/h^0.8 with random phases, over a faint pink floor."""
    time_s = np.arange(count) / SAMPLE_RATE
    tone = np.sum([np.sin(2 * np.pi * fundamental_hz * h * time_s + rng.uniform(0, 2 * np.pi)) / h**0.8
                   for h in range(1, 13)], axis=0)
    return unit_rms(unit_rms(tone) + 0.05 * pink(count, rng))


_NOTES = (  # chord progression as MIDI note numbers: Cmaj7, Am7, Fmaj7, G7
    (48, 55, 60, 64, 71), (45, 52, 60, 64, 67), (41, 48, 57, 60, 64), (43, 50, 59, 62, 65))


def music(count: int, rng: np.random.Generator) -> np.ndarray:
    """Plucked chords, four notes' worth of harmonics each, retriggered every 0.625 s; no recorded music."""
    out = np.zeros(count)
    pluck = int(1.6 * SAMPLE_RATE)
    time_s = np.arange(pluck) / SAMPLE_RATE
    for index, start in enumerate(range(0, count, int(0.625 * SAMPLE_RATE))):
        chord = _NOTES[(index // 4) % len(_NOTES)]
        for note in chord:
            freq = 440.0 * 2.0 ** ((note - 69) / 12.0)
            tone = sum(np.sin(2 * np.pi * freq * h * time_s + rng.uniform(0, 2 * np.pi)) / h for h in range(1, 7))
            segment = tone * np.exp(-time_s / 0.5) * np.minimum(1.0, time_s / 0.02)
            stop = min(count, start + pluck)
            out[start:stop] += segment[: stop - start]
    return unit_rms(out)


_VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (660, 1720, 2410))


def speech_like(count: int, rng: np.random.Generator) -> np.ndarray:
    """A synthetic voice: a gliding 110-150 Hz pulse train through three formants that change every ~0.2 s,
    gated into syllables at about 4 Hz with random rests. It has speech's spectrum and rhythm but no words."""
    time_s = np.arange(count) / SAMPLE_RATE
    glide = 130.0 + 20.0 * np.sin(2 * np.pi * 0.23 * time_s + rng.uniform(0, 6.28))
    source = 2.0 * np.mod(np.cumsum(glide) / SAMPLE_RATE, 1.0) - 1.0
    out = np.zeros(count)
    segment = int(0.2 * SAMPLE_RATE)
    states = [np.zeros(2) for _ in range(3)]
    for start in range(0, count, segment):
        chunk = source[start : start + segment]
        formants = _VOWELS[int(rng.integers(len(_VOWELS)))]
        for index, freq in enumerate(formants):
            radius = np.exp(-np.pi * (60.0 + 40.0 * index) / SAMPLE_RATE)
            a = [1.0, -2.0 * radius * np.cos(2 * np.pi * freq / SAMPLE_RATE), radius**2]
            chunk, states[index] = lfilter([1.0 - radius], a, chunk, zi=states[index])
        out[start : start + chunk.size] = chunk
    syllable = np.clip(np.sin(2 * np.pi * 4.0 * time_s + rng.uniform(0, 6.28)), 0.0, None) ** 0.7
    rests = np.repeat(rng.random(count // segment + 1) > 0.25, segment)[:count]
    return unit_rms(out * syllable * rests)


def tv(count: int, rng: np.random.Generator) -> np.ndarray:
    """A television in the next room: synthetic speech-like voice over a music bed 8 dB lower."""
    return unit_rms(speech_like(count, rng) + 10.0 ** (-8.0 / 20.0) * music(count, rng))


def tile(source: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Repeat ``source`` from a seeded start to ``count`` samples with 100 ms linear cross-fades at the joins."""
    fade = int(0.1 * SAMPLE_RATE)
    source = np.roll(source, -int(rng.integers(source.size)))
    ramp = np.linspace(0.0, 1.0, fade)
    out = source.copy()
    while out.size < count + fade:
        out[-fade:] = out[-fade:] * (1.0 - ramp) + source[:fade] * ramp
        out = np.concatenate([out, source[fade:]])
    return unit_rms(out[:count])


def babble(count: int, rng: np.random.Generator, voices: list[np.ndarray]) -> np.ndarray:
    """Other takes of the two speakers, each rotated by a seeded offset and levelled, summed."""
    parts = []
    for voice in voices:
        rotated = np.roll(np.tile(voice, count // voice.size + 1), int(rng.integers(voice.size)))[:count]
        parts.append(rotated * 10.0 ** (-active_level_db(rotated) / 20.0))
    return unit_rms(np.sum(parts, axis=0))


def room_impulse(rt60_s: float, drr_db: float, seed: int) -> np.ndarray:
    """Direct sound plus an exponentially decaying noise tail (60 dB in ``rt60_s``, highs damped by a 6 kHz pole),
    whose energy is ``-drr_db`` relative to the direct sound."""
    rng = np.random.default_rng(seed)
    count = int(rt60_s * SAMPLE_RATE)
    time_s = np.arange(count) / SAMPLE_RATE
    tail = rng.standard_normal(count) * np.exp(-6.908 * time_s / rt60_s)
    pole = np.exp(-2 * np.pi * 6_000.0 / SAMPLE_RATE)
    tail = np.asarray(lfilter([1.0 - pole], [1.0, -pole], tail))
    tail[: int(0.003 * SAMPLE_RATE)] = 0.0
    tail *= 10.0 ** (-drr_db / 20.0) / math.sqrt(float(np.sum(tail**2)))
    tail[0] = 1.0
    return tail


def noise_track(bench: Bench, mix_id: str, count: int, rng: np.random.Generator) -> np.ndarray:
    spec = bench.manifest["mixes"][mix_id]["noise"]
    kind = spec["kind"]
    if kind == "pink":
        return pink(count, rng)
    if kind == "fan":
        return fan(count, rng)
    if kind == "keyboard":
        return keyboard(count, rng)
    if kind == "hum":
        return hum(count, rng, float(spec["hz"]))
    if kind == "music":
        return music(count, rng)
    if kind == "tv":
        return tv(count, rng)
    if kind == "real":
        return tile(bench.noises[spec["source"]], count, rng)
    if kind == "babble":
        return babble(count, rng, [bench.takes[name] for name in spec["takes"]])
    raise ValueError(f"unknown noise kind {kind!r}")


def build_mix(bench: Bench, mix_id: str) -> dict[str, Any]:
    """One mix of the bench: ``raw`` (float32), the ``clean`` take as heard (reverberant, levelled), ``labels``
    and the room ``noise_capture``. Deterministic: the recipe's seed is the only source of randomness."""
    recipe = bench.manifest["mixes"][mix_id]
    take = bench.manifest["takes"][recipe["take"]]
    voice = bench.takes[recipe["take"]]
    labels = speech_labels(take, voice.size)
    rng = np.random.default_rng(recipe["seed"])
    if recipe.get("rir"):
        voice = fftconvolve(voice, room_impulse(recipe["rir"]["rt60_s"], recipe["rir"]["drr_db"],
                                                recipe["rir"]["seed"]))[: voice.size]
    speech = voice * 10.0 ** ((recipe["speech_dbfs"] - active_level_db(voice)) / 20.0)
    lead = int(NOISE_S * SAMPLE_RATE)
    track = noise_track(bench, mix_id, speech.size + lead, rng)
    track = track * 10.0 ** ((recipe["speech_dbfs"] - recipe["noise"]["snr_db"]) / 20.0) / math.sqrt(
        float(np.mean(np.square(track[lead:]))))
    capture, noise = track[:lead], track[lead:]
    mix = speech + noise
    post = recipe.get("post") or {}
    if "peak" in post:  # level the whole capture so its peak lands on the stated value
        gain = post["peak"] / float(np.max(np.abs(mix)))
        speech, capture, mix = speech * gain, capture * gain, mix * gain
    if post.get("clip"):  # a hard clip at full scale; the clean reference stays unclipped
        mix = np.clip(mix, -1.0, 1.0)
    return {"raw": mix.astype(np.float32), "clean": speech, "labels": labels,
            "noise_capture": capture.astype(np.float32)}
