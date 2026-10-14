"""Measure how often Voice Setup withholds a setup that was no worse than the user's current settings.

Subcommands:
  run      Simulate each study user (``evaluate_voice_setup.py``) and run ``analyze_voice_setup`` as the dialog
           does, with two taps that keep what a refusal would discard: the Auto-EQ proposal an unclear-recording
           refusal throws away, and the tone fit an abstention zeroes. Emulate verification, then render the
           evaluation capture through the current settings, each withheld setup and, for a refused setup, each
           of its steps, and measure every clause below. Refuses a spent user, and starts only when the hashes
           written by ``freeze`` still match.
  probe    ``probe --development``: the bookkeeping dry run on the 36 spent development users (fit, calibrate
           and test 0-11). It runs them with extra probes and writes reachability, clause coverage, probe and
           arithmetic checks as counts and flags. No rate and no interval is computed from spent users.
  report   Judge the rows written by ``run`` and write the tracked report. Refuses development rows.
  freeze   Write ``rules-final.md`` (this text) and ``frozen-files.json`` (hashes of this rule, the sources, the
           native extension, df.dll, the VAD model and the corpus manifest) beside the rows.

Rule (fixed before any study row existed)

Unit. A simulated user (speaker, room, DEMAND noise, SNR and level from ``evaluate_voice_setup.user``) of the
``fit`` split at indices 72-89, 102-119 and 132-149: 18 speakers x 3 realizations = 54 users, RNNoise 1.0. The
current settings are the factory ``Preset()`` with that noise model: gate, suppressor, flat EQ, de-esser off,
compressor on without makeup, limiter on. That is the main limitation, because real users' settings are unknown.
``analyze_voice_setup`` is called as the dialog calls it, with the factory gate, suppressor and flat EQ as the
incumbent and the raw captures. DeepFilter is configured in every worker as the app does and judged with the same
df.dll; a missing df.dll, DeepFilter model or VAD model stops the run.

Window. The 20 s evaluation capture, which the setup never heard, rendered whole through the full chain (pre-filter,
gate, suppressor, de-esser, EQ, compressor, limiter) and compared with the pre-filtered reverberant clean speech.
Every veto is judged on the whole 20 s. The last 10 s is a sensitivity reading for C1 and C2, whose verification
take is the first 10 s.

Clauses, each against the current settings. A difference exactly at a tolerance passes (slack 1e-9).
  estoi              ESTOI against the clean speech, not more than 0.01 lower.
  tone_error_db      RMS over 80-12000 Hz of the median-centred difference between the long-term speech spectrum
                     of the render and of the clean speech (4096-sample frames, hop 2048, frames that are speech
                     throughout with a clean level within 25 dB of the loudest; the same frames for both), not
                     more than 1.0 dB higher.
  sibilant_loss_db   The loss of 5-9 kHz level in the input's sibilant frames relative to its voiced frames
                     (``voice_setup._sibilant_change_db`` of the pre-filtered noisy capture against the render,
                     the pre-filtered noise capture as the room reference), not more than 3.0 dB higher.
                     Sensitivity: 1.5 and 6.0 dB.
  deep_pause_db      Level of pauses at least 600 ms from speech, relative to the input, not more than 1.0 dB
                     higher. This is "pause noise"; ``pause_db`` is measured and reported only.
  tail_db            Level of pause frames within 300 ms after speech, relative to the input, not more than
                     1.0 dB higher.
  onset_db           First 100 ms of each phrase relative to its body, against the clean speech, not more than
                     1.0 dB lower.
  overload           No clipping (a sample at or above 1.0 in the unscaled render) and no limiter reading of
                     6 dB (limiter) or 3 dB (true-peak limiter) gain reduction. Attributed reading, which
                     decides: it counts against a withheld setup only when it clips or reaches the reading and
                     the current settings do not. The strict reading, which counts it whenever it happens, is
                     reported beside it.
Level matching: before deep_pause_db and tail_db are read, each render is scaled by the difference between its
mean energy on speech-labelled frames and the input's on the same frames, so makeup does not look like noise.
ESTOI, onset, tone and sibilant loss do not depend on level. Clipping and limiter readings use the unscaled render.
The level check (``level_check.verdict_from_audio``) is not a veto and is not counted; every veto is also shown
by its verdict.

Verdict of a withheld setup against the current settings: ``false`` when every clause passes, ``correct`` when at
least one measurable clause fails, ``partial`` when none fails and at least one cannot be measured (a NaN, a missing
value, no deep pause, fewer than 8 sibilant frames). Headline reading, the lower bound on false refusals:
unmeasurable counts as worse, so ``partial`` is a correct refusal. Beside it, the upper bound: unmeasurable counts
as not worse, so ``partial`` is a false refusal. A withheld object that changes nothing the current settings do not
(equal sections, an EQ whose bands all stay under 0.25 dB) is ``vacuous``; one that has no object or whose render
failed is ``unjudgeable``; both are counted apart, listed, and are neither false nor correct. ``definition`` marks a
refusal that is correct by definition (B2 with under 2 s of active speech; the clip rule cannot fire in the
simulator). Soft outcomes (B2-B5) are refusals, labelled soft; hard (B1, C1, C2) and soft are reported apart.

Vetoes (the numbering of the inventory): A1-A4 are capture vetoes the simulator cannot fire (unit-tested
detectors, probed with constructed arrays); A5 is the level check; B1 the unclear-recording refusal (the
proposal is recovered with a tap); B2 a weak capture; B3 a compressor target missed; B4 the final check; B5 the
Auto-EQ abstention; B6 the tuner keeping the current gate and suppressor; C1 a verification rollback; C2 a
verification that cannot settle after reductions (three adjustments, no help, or nothing left to change); D1 and D2
are retakes, reported as costs, never judged and never counted as verified. E1 (automatic makeup withheld) and E2
(de-esser left off for invalid evidence, read from the diagnostics because the analysis does not export its flag)
are firing rates only.

Two tracks. Whole setup: one bucket per user, the first of B1, B2, B3, B4, B5, C1, C2 that fired (C only for a
setup the analysis recommended; verification of a setup that was not recommended, as if the user applied it
anyway, is emulated and reported as also fired); a user whose Auto-EQ stage failed before validation is an analysis
failure, not a refusal. A setup that B1 ends never reaches verification and is not verified. Withheld object: the
setup as proposed for B1-B5, the candidate as it ended verification for C1 and C2. Withheld step: the tone fit the
abstention zeroed (B5, the validated gains of the tap) and the tuner's own best candidate (B6: the non-incumbent
candidate with the lowest score among those that pass its gates, else the lowest overall), each applied alone to
the current settings; its failed gates and whether the margin or the holdout vetoed it are recorded. That any
tested candidate is no worse is reported as an upper bound only, from the gate and suppressor stage alone (the
tone, sibilant and overload clauses can only add failures). For each correctly refused whole setup, each single
step (tone, gate and suppression, de-esser, compressor with its gain) applied alone to the current settings is
judged too. The tuner (B6) and de-esser results are flattering: no VoiceBank speaker is untouched, and these
voices fed the tuner harness and the de-esser fit.

Rate: share of all 54 users with the speaker-clustered 95 % bootstrap of ``evaluate_gate_corpus.cluster_ci``,
per veto counted first and counted in any, for the users with at least one false first veto, by hard and soft, and
by stratum (level-check verdict, RT60 band, SNR band, gender).

Trigger (the only decision rule). A veto whose false-refusal share of all users, headline reading, has an interval
whose lower bound is above 5 % in either counting becomes a candidate for a loosening study with its own predefined
gates and fresh speakers. No veto is changed by this study.

Dry run. ``probe --development`` runs the 36 spent users (fit, calibrate and test 0-11) with six known-bad probes
(a +12 dB bell at 3 kHz and a -12 dB high shelf at 6.5 kHz, each on the EQ's nearest tone band; suppressor off;
gate threshold -15 dB with a 10 ms release; gate release 1000 ms; manual makeup +24 dB) and two known-good ones
(the current settings again, and without the limiter). A clause whose probe never trips it cannot say "worse"; a
clause unmeasurable for more than 15 % of cases is flagged before the freeze. Spent after the study: fit 72-89,
102-119 and 132-149.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_gate_corpus as gate_corpus
import evaluate_voice_setup as voice_setup_eval
import evaluate_voice_setup_loudness as loudness

REPO_ROOT = voice_setup_eval.REPO_ROOT
SAMPLE_RATE = voice_setup_eval.SAMPLE_RATE
SPLIT = "fit"
MODEL = "rnnoise"
STUDY_INDICES = (*range(72, 90), *range(102, 120), *range(132, 150))
DEFAULT_INDICES = "fit:72-89,102-119,132-149"
# No study used a user index below this in any split; AGENTS.md lists a narrower spent set.
SPENT_BELOW = 60
DEVELOPMENT_COUNT = 12
DEVELOPMENT_USERS = tuple((split, index) for split in voice_setup_eval.SPLITS for index in range(DEVELOPMENT_COUNT))
MAX_WORKERS = 4
DETAILS = REPO_ROOT / "models" / "evaluation-details" / "voice-setup-refusals-2026-10"
DEFAULT_ROWS = DETAILS / "rows.json"
DEVELOPMENT_ROWS = DETAILS / "development" / "rows.json"
DEFAULT_BOOKKEEPING = DETAILS / "development" / "bookkeeping.json"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "voice-setup-refusals-2026-10-report.json"
FROZEN = DETAILS / "frozen-files.json"
RULES_FINAL = DETAILS / "rules-final.md"
SOURCES = (
    *loudness.SOURCES,
    "python/tools/evaluate_voice_setup_refusals.py",
    "python/mic_eq/analysis/failure_detection.py",
    "python/mic_eq/analysis/spectrum.py",
    "python/mic_eq/analysis/listening_comparison.py",
    "python/mic_eq/analysis/auto_eq_parts/pipeline.py",
    "python/mic_eq/analysis/auto_eq_parts/optimizer.py",
    "python/mic_eq/ui/level_check.py",
)
SECTIONS = loudness.SECTIONS
STEPS = {"tone": ("eq_settings",), "gate_suppression": ("gate_settings", "suppressor_settings"),
         "deesser": ("deesser_settings",), "compressor": ("compressor_settings",)}
# The order in which one whole-setup bucket is chosen, and the vetoes that are hard refusals.
WHOLE_ORDER = ("B1", "B2", "B3", "B4", "B5", "C1", "C2")
VETOES = ("B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2")
HARD = ("B1", "C1", "C2")
OBJECT = {"B1": "whole", "B2": "whole", "B3": "whole", "B4": "whole", "B5": "whole", "C1": "final", "C2": "final"}
STEP_OBJECT = {"B5": "tone", "B6": "tuner"}
RETRY_COSTS = {"verification_delivery_mismatch": "D1", "verification_capture_too_short": "D2",
               "verification_capture_invalid": "D2"}
STATUSES = ("false", "correct", "partial", "definition", "vacuous", "unjudgeable")
SLACK = 1e-9
# Metric: (direction in which larger is better, tolerance).
RULES = {
    "estoi": voice_setup_eval.TOLERANCES["estoi"],
    "tone_error_db": (-1.0, 1.0),
    "sibilant_loss_db": (-1.0, 3.0),
    "deep_pause_db": voice_setup_eval.TOLERANCES["deep_pause_db"],
    "tail_db": voice_setup_eval.TOLERANCES["tail_db"],
    "onset_db": voice_setup_eval.TOLERANCES["onset_db"],
}
CLAUSES = (*RULES, "overload")
LIMITING_HARD_DB = loudness.APP_LIMITING_HARD_DB
TRIGGER_SHARE = 0.05
UNMEASURABLE_LIMIT = 0.15
TONE_FRAME, TONE_HOP, MIN_TONE_FRAMES = 4096, 2048, 8
ACTIVE_BAND_DB = 0.25  # the Auto-EQ stage's own threshold for a band that corrects anything
REDUCIBLE = ("compressor", "limiter", "deesser")
# (pattern in the reason, veto), first match wins. "X": the native renderer is unavailable, which the simulator
# cannot produce. The source-scan test fails when a reason the analysis can give matches none.
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
LIMITATIONS = (
    "Simulated users: speech in synthetic rooms with DEMAND noise, not microphone captures. The simulator makes no "
    "clipped, silent or non-finite takes, so A1-A4 cannot fire, and no user has custom settings.",
    "Current settings are always the factory Preset(); real users' current settings are unknown. This is the main "
    "limitation, and a false refusal here is a false refusal against defaults.",
    "18 speakers (48 of 54 users male), so intervals are wide; more realizations per speaker do not narrow them.",
    "Known limits unchanged: the tonal check is nearly blind below 1 kHz; Auto-EQ's Adaptive inputs saturate for real "
    "voices; in reverberant captures noise is judged by pauses of at least 600 ms from speech.",
    "A clause that cannot be measured is reported as a lower and an upper bound; the headline treats it as worse.",
    "Verification hears the first 10 s of the evaluation capture and the judge the whole 20 s, so for C1 and C2 the "
    "judge overlaps the evidence; the last 10 s is the sensitivity reading.",
    "Sibilant loss is read against the noisy input as the product's own dulling advice reads it, so it includes what "
    "the whole chain does to consonants, not only the withheld step.",
    "The tuner's upper bound renders the gate and suppressor stage alone and omits the tone, sibilant and overload "
    "clauses; it can only overstate.",
    "The tone of a B5 step is the optimizer's validated fit before the abstention and the headroom scaling.",
)
FLATTERING = (
    "No VoiceBank speaker is untouched. These speakers fed the tuner harness and all 30 fit-split speakers fit a "
    "shipped de-esser value, so the tuner (B6) and de-esser results are flattering.",
)


# ----------------------------------------------------------------------------------------------
# The judgement (pure)


def _finite(*values: Any) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def _reaches(metrics: dict[str, Any]) -> bool | None:
    limiter, true_peak = metrics.get("limiter_gain_reduction_db"), metrics.get("true_peak_limiter_gain_reduction_db")
    if limiter is None or true_peak is None or not _finite(limiter, true_peak):
        return None
    return bool(limiter >= LIMITING_HARD_DB[0] or true_peak >= LIMITING_HARD_DB[1])


def _overload(withheld: dict[str, Any], current: dict[str, Any], strict: bool) -> str:
    reaches = (_reaches(withheld), _reaches(current))
    if None in reaches or "clipped" not in withheld or "clipped" not in current:
        return "unmeasurable"
    for worse, baseline in ((withheld["clipped"], current["clipped"]), reaches):
        if worse and (strict or not baseline):
            return "fail"
    return "pass"


def clause_results(withheld: dict[str, Any], current: dict[str, Any], *, sibilant_tolerance: float = 3.0,
                   strict: bool = False) -> dict[str, str]:
    """``pass``, ``fail`` or ``unmeasurable`` for every clause of a withheld setup against the current one."""
    results = {}
    for metric, (direction, tolerance) in RULES.items():
        tolerance = sibilant_tolerance if metric == "sibilant_loss_db" else tolerance
        value, baseline = withheld.get(metric), current.get(metric)
        if value is None or baseline is None or not _finite(value, baseline):
            results[metric] = "unmeasurable"
        else:
            results[metric] = "fail" if direction * (value - baseline) < -tolerance - SLACK else "pass"
    results["overload"] = _overload(withheld, current, strict)
    return results


def judge(withheld: dict[str, Any], current: dict[str, Any], **reading: Any) -> dict[str, Any]:
    clauses = clause_results(withheld, current, **reading)
    failed = [name for name, result in clauses.items() if result == "fail"]
    unmeasurable = [name for name, result in clauses.items() if result == "unmeasurable"]
    return {"clauses": clauses, "failed": failed, "unmeasurable": unmeasurable,
            "verdict": "correct" if failed else "partial" if unmeasurable else "false"}


def no_worse(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """No measurable clause fails. Whenever ``evaluate_voice_setup.dominates(a, b)`` holds, ``no_worse(a, b)`` does."""
    return "fail" not in clause_results(candidate, baseline).values()


def false_refusal(verdict: str, bound: str) -> bool:
    """Whether a verdict is a false refusal: ``lower`` counts an unmeasurable clause as worse, ``upper`` as not worse."""
    return verdict == "false" or (bound == "upper" and verdict == "partial")


def first_whole(fired: set[str] | list[str], recommended: bool) -> str | None:
    for veto in WHOLE_ORDER:
        if veto in fired and (recommended or not veto.startswith("C")):
            return veto
    return None


def reason_veto(reason: str) -> str | None:
    for pattern, veto in REASON_VETOES:
        if pattern in reason:
            return veto
    return None


def fired_vetoes(view: dict[str, Any]) -> list[str]:
    """The vetoes of one analysis: ``reasons``, ``weak_capture``, ``b1``, ``tuner_decision`` and ``verification``."""
    fired = {reason_veto(reason) for reason in view["reasons"]}
    # B2 is the analysis' own weak-capture switch, whatever reasons accompany it.
    fired -= {None, "X", "B2"}
    flags = {"B1": view["b1"], "B2": view["weak_capture"], "B6": view["tuner_decision"] == "retain_incumbent",
             "C1": view["verification"] == "rollback", "C2": view["verification"] == "gave_up"}
    fired |= {veto for veto, flag in flags.items() if flag}
    return [veto for veto in VETOES if veto in fired]


def _candidate_key(candidate: dict[str, Any]) -> str:
    return voice_setup_eval._arm_key(
        candidate["noise_model"], candidate["suppressor_strength"], dict(candidate["gate_settings"]))


def tuner_withheld(candidates: list[dict[str, Any]], current_key: str, tuning: dict[str, Any]) -> dict[str, Any] | None:
    """The tuner's own best non-incumbent candidate: lowest score among those that pass its gates, else overall."""
    others = [c for c in candidates if _candidate_key(c) != current_key]
    if not others:
        return None
    passing = [c for c in others if c["passes"]]
    best = min(passing or others, key=lambda c: float(c["score"]))
    cause = ("no_candidate_passed" if not passing else "margin" if not tuning.get("holdout_evaluated") else "holdout")
    return {"noise_model": best["noise_model"], "suppressor_strength": float(best["suppressor_strength"]),
            "gate_settings": dict(best["gate_settings"]), "score": float(best["score"]), "passes": bool(best["passes"]),
            "failed_gates": sorted(name for name, ok in best["gates"].items() if not ok), "cause": cause,
            "tested": len(others)}


def capture_vetoes_probe() -> dict[str, bool]:
    """A1-A4 fired on hand-made arrays by the product's own detectors."""
    from mic_eq.analysis.noise_reference import analyze_noise_reference, capture_is_silent
    from mic_eq.analysis.voice_setup import analyze_voice_setup, raw_capture_is_clipped

    rng = np.random.default_rng(0)
    speech = (0.05 * rng.standard_normal(10 * SAMPLE_RATE)).astype(np.float32)
    noise = (0.01 * rng.standard_normal(2 * SAMPLE_RATE)).astype(np.float32)
    flat = speech.copy()
    flat[1000:1010] = 1.0
    short, broken = noise[: int(1.4 * SAMPLE_RATE)], noise.copy()
    broken[5] = np.nan
    try:
        analyze_voice_setup(short, speech, SAMPLE_RATE, "broadcast")
        too_short = False
    except ValueError:
        too_short = True
    return {"A1": raw_capture_is_clipped(flat, SAMPLE_RATE), "A2": capture_is_silent(np.zeros_like(speech)),
            "A3": bool(analyze_noise_reference(short, speech, SAMPLE_RATE).must_recapture
                       and analyze_noise_reference(broken, speech, SAMPLE_RATE).must_recapture),
            "A4": too_short}


# ----------------------------------------------------------------------------------------------
# Settings


def current_settings(vad_available: bool) -> dict[str, Any]:
    """The factory ``Preset()`` as the six sections a chain render reads."""
    from mic_eq.config import EQSettings, Preset

    preset = Preset()
    return {"eq_settings": EQSettings().to_dict(), "gate_settings": asdict(preset.gate),
            "suppressor_settings": asdict(preset.rnnoise), "deesser_settings": asdict(preset.deesser),
            "compressor_settings": asdict(preset.compressor), "limiter_settings": asdict(preset.limiter),
            "diagnostics": {"vad_available": bool(vad_available)}}


def hybrid(current: dict[str, Any], proposal: dict[str, Any], step: str) -> dict[str, Any]:
    """The current settings with only the sections of one step taken from the proposal."""
    settings = deepcopy(current)
    for name in STEPS[step]:
        settings[name] = deepcopy(proposal[name])
    return settings


def is_vacuous(withheld: dict[str, Any], current: dict[str, Any], sections: tuple[str, ...] = SECTIONS) -> bool:
    """Whether the withheld sections change nothing the current ones do not (a flat EQ counts as unchanged)."""
    for name in sections:
        if name == "eq_settings":
            gains = withheld[name].get("band_gains", ())
            if max((abs(float(gain)) for gain in gains), default=0.0) >= ACTIVE_BAND_DB:
                return False
        elif loudness._digest(withheld[name]) != loudness._digest(current[name]):
            return False
    return True


def shifted_tone(eq: dict[str, Any], frequency_hz: float, gain_db: float, filter_type: str | None = None) -> dict[str, Any]:
    """A typed EQ with the tone band nearest ``frequency_hz`` moved to ``frequency_hz`` and ``gain_db``, as a bell
    unless ``filter_type`` says otherwise."""
    wrong = deepcopy(eq)
    band = min(wrong["bands"], key=lambda b: abs(math.log(b["frequency_hz"] / frequency_hz)))
    band.update(gain_db=gain_db, frequency_hz=frequency_hz, filter_type=filter_type or "bell")
    return wrong


def probe_settings(current: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Known-bad probes (each with the clause it must trip) and known-good ones (target ``""``)."""
    def with_(section: str, **changes: Any) -> dict[str, Any]:
        settings = deepcopy(current)
        settings[section] = {**settings[section], **changes}
        return settings

    def eq(frequency_hz: float, gain_db: float, filter_type: str | None = None) -> dict[str, Any]:
        return {**deepcopy(current), "eq_settings": shifted_tone(current["eq_settings"], frequency_hz, gain_db, filter_type)}

    return {
        "tone_plus_12_db_3k": ("tone_error_db", eq(3_000.0, 12.0)),
        "sibilance_minus_12_db_6k5": ("sibilant_loss_db", eq(6_500.0, -12.0, "high_shelf")),
        "suppressor_off": ("deep_pause_db", with_("suppressor_settings", enabled=False)),
        "gate_threshold_minus_15": ("onset_db", with_("gate_settings", auto_threshold_enabled=False,
                                                       threshold_db=-15.0, release_ms=10.0)),
        "gate_release_1000": ("tail_db", with_("gate_settings", release_ms=1000.0)),
        "makeup_plus_24": ("overload", with_("compressor_settings", makeup_gain_db=24.0)),
        "current_again": ("", deepcopy(current)),
        "limiter_off": ("", with_("limiter_settings", enabled=False)),
    }


# ----------------------------------------------------------------------------------------------
# Taps


@contextmanager
def _taps() -> Iterator[dict[str, list[Any]]]:
    """Keep the Auto-EQ proposal that an unclear-recording refusal would discard, and the validated tone fit.

    ``validate_analysis`` is looked up inside ``analyze_auto_eq`` at every call, so the wrapper replaces its
    verdict with a pass after recording the real one; ``analyze_voice_setup`` then goes on with the proposal.
    """
    from mic_eq.analysis import failure_detection
    from mic_eq.analysis.auto_eq_parts import optimizer

    log: dict[str, list[Any]] = {"validation": [], "tone": []}
    validate, attenuate = failure_detection.validate_analysis, optimizer._validate_and_attenuate_solution

    def tapped_validate(eq_settings: Any, spectrum_db: Any, freqs: Any) -> Any:
        real = validate(eq_settings, spectrum_db, freqs)
        log["validation"].append({"passed": bool(real.passed), "reason": real.reason,
                                  "failures": [str(f) for f in real.details.get("failures", ())]})
        return failure_detection.ValidationResult(passed=True, reason=None, details=dict(real.details))

    def tapped_attenuate(*args: Any, **kwargs: Any) -> Any:
        solution = attenuate(*args, **kwargs)
        log["tone"].append([float(gain) for gain in solution[0]])
        return solution

    failure_detection.validate_analysis = tapped_validate
    optimizer._validate_and_attenuate_solution = tapped_attenuate
    try:
        yield log
    finally:
        failure_detection.validate_analysis = validate
        optimizer._validate_and_attenuate_solution = attenuate


# ----------------------------------------------------------------------------------------------
# Measurement


def level_matched(case: dict[str, Any], output: np.ndarray) -> np.ndarray:
    """``output`` scaled so that its speech frames have the input's energy."""
    speech = gate_corpus._frame_labels(case["labels"])
    out_db, in_db = gate_corpus._frame_rms_db(output), gate_corpus._frame_rms_db(case["noisy"])
    count = min(speech.size, out_db.size, in_db.size)
    gain = (gate_corpus._energy_mean_db(out_db[:count], speech[:count])
            - gate_corpus._energy_mean_db(in_db[:count], speech[:count]))
    return output * 10.0 ** (-gain / 20.0) if math.isfinite(gain) else output


def scored(case: dict[str, Any], output: np.ndarray) -> dict[str, float]:
    """The gate corpus metrics of a render, read after level matching."""
    scores = gate_corpus.score(case, level_matched(case, output))
    return {name: float(scores[name]) for name in ("estoi", "onset_db", "deep_pause_db", "tail_db", "pause_db")}


def tone_error_db(clean: np.ndarray, output: np.ndarray, labels: np.ndarray) -> float:
    """RMS over 80-12000 Hz of the median-centred difference of the two long-term speech spectra."""
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


def measure(rendered: dict[str, Any], ctx: dict[str, Any], start: int = 0) -> dict[str, Any]:
    """Every clause's reading of one render of the evaluation capture, from sample ``start`` on."""
    from mic_eq.analysis.voice_setup import _sibilant_change_db

    stop = ctx["clean"].size
    output = np.asarray(rendered["output_audio"], np.float64)[start:stop]
    case = {name: ctx[name][start:stop] for name in ("clean", "noisy", "labels")}
    change, _ = _sibilant_change_db(case["noisy"], output, ctx["noise"], SAMPLE_RATE)
    return {
        **scored(case, output), "tone_error_db": tone_error_db(case["clean"], output, case["labels"]),
        "sibilant_loss_db": None if change is None else -float(change),
        "clipped": bool(np.max(np.abs(output)) >= 1.0),
        "limiter_gain_reduction_db": float(rendered["limiter_gain_reduction_db"]),
        "true_peak_limiter_gain_reduction_db": float(rendered["true_peak_limiter_gain_reduction_db"]),
    }


def measure_settings(settings: dict[str, Any], ctx: dict[str, Any], windows: tuple[str, ...] = ("whole",)) -> dict[str, Any]:
    """Render ``settings`` once and read each window; a render that fails is recorded, not hidden."""
    try:
        rendered = loudness._render(settings, ctx["raw"])
    except (RuntimeError, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    half = ctx["raw"].size // 2
    return {window: measure(rendered, ctx, half if window == "last10" else 0) for window in windows}


# ----------------------------------------------------------------------------------------------
# Verification


def _reduction_metrics(outcome: dict[str, Any]) -> dict[str, float]:
    """The dialog's ``_reduction_metrics``."""
    values = {}
    for target, key in (("compressor", "compressor_gain_reduction_p95_db"), ("limiter", "limiter_pressure_db"),
                        ("deesser", "deesser_gain_reduction_p95_db")):
        value = outcome.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[target] = float(value)
    return values


def verify_flow(result: dict[str, Any], captures: dict[str, np.ndarray]) -> dict[str, Any]:
    """The dialog's verification of a proposal with its automatic reductions, and the candidate as it ended.

    ``gave_up`` is a reduction that cannot be settled: three adjustments used, an adjustment that did not help,
    or nothing left to change. A ``retry`` is a retake and decides nothing.
    """
    from mic_eq.analysis.voice_setup import validate_voice_setup_verification

    state = deepcopy(result)
    state["eq_settings"] = loudness._applied_eq(state["eq_settings"])
    adjustments, last, first, ended = 0, None, None, None
    while True:
        state["verification_chain_settings"] = loudness._chain(state)
        outcome = validate_voice_setup_verification(
            captures["noise"], captures["setup"], captures["second"], SAMPLE_RATE, state, "broadcast",
            raw_noise_audio=captures["raw_noise"], raw_verification_speech_audio=captures["raw_second"])
        first = first or outcome["decision"]
        if outcome["decision"] != "reduce":
            break
        targets = [t for t in outcome.get("reduction_targets") or () if t in REDUCIBLE]
        metrics = _reduction_metrics(outcome)
        comparable = [(last[t], metrics[t]) for t in targets if last and t in last and t in metrics]
        if not targets:
            ended = "no_target"
        elif adjustments >= loudness.MAX_ADJUSTMENTS:
            ended = "three_adjustments"
        elif comparable and not any(now < before - 0.05 for before, now in comparable):
            ended = "no_help"
        if ended:
            break
        before = deepcopy((state["compressor_settings"], state["deesser_settings"]))
        loudness._reduce(state, targets)
        if metrics:
            last = metrics
        if before == (state["compressor_settings"], state["deesser_settings"]):
            ended = "no_change"
            break
        adjustments += 1
    return {"decision": "gave_up" if outcome["decision"] == "reduce" else outcome["decision"], "first_decision": first,
            "adjustments": adjustments, "ended": ended, "reason_codes": list(outcome.get("reason_codes") or ()),
            "state": state}


# ----------------------------------------------------------------------------------------------
# One user


def check_assets() -> None:
    """DeepFilter and the VAD model must be there; a missing one would be skipped silently by the native code."""
    os.environ.setdefault("VAD_MODEL_PATH", str(REPO_ROOT / "models" / "silero_vad.onnx"))
    needed = [REPO_ROOT / "df.dll", REPO_ROOT / "models" / "DeepFilterNet3_ll_onnx.tar.gz",
              REPO_ROOT / "models" / "DeepFilterNet3_onnx.tar.gz", Path(os.environ["VAD_MODEL_PATH"])]
    missing = [path.name for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"study assets are missing: {', '.join(missing)}")


def configure_environment() -> None:
    """Register DeepFilter as the app does, in this process."""
    from mic_eq import configure_deepfilter_runtime_paths

    check_assets()
    configure_deepfilter_runtime_paths(str(REPO_ROOT / "df.dll"), str(REPO_ROOT / "models"))
    os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = "1"


@functools.lru_cache(maxsize=1)
def build_id() -> str:
    """The evaluator and the extension; a partial file written by another build is not resumed."""
    extension = REPO_ROOT / loudness.EXTENSION
    return hashlib.sha256((loudness._sha256(Path(__file__)) + gate_corpus._sha256(extension)).encode()).hexdigest()[:16]


def rt60_s(split: str, index: int) -> float:
    """The first draw of the user's room generator (see ``evaluate_voice_setup.user``)."""
    return float(np.random.default_rng(1_000_000 * (voice_setup_eval.SPLITS.index(split) + 1) + index).uniform(0.2, 0.7))


def _verification_row(flow: dict[str, Any]) -> dict[str, Any]:
    row = {key: flow[key] for key in ("decision", "first_decision", "adjustments", "ended", "reason_codes")}
    retry = next((RETRY_COSTS[code] for code in flow["reason_codes"] if code in RETRY_COSTS), None)
    row["retry"] = retry if flow["decision"] == "retry" else None
    return row


def evaluate_user(job: tuple[str, str, int, str], *, development: bool) -> dict[str, Any]:
    root_text, split_name, index, model = job
    configure_environment()
    from mic_eq.analysis.spectrum import analyze_voice_spectrum
    from mic_eq.analysis.voice_setup import analyze_voice_setup
    from mic_eq.config import EQSettings, Preset
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis
    from mic_eq.ui.level_check import verdict_from_audio

    root = Path(root_text)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    simulated = voice_setup_eval.user(root, manifest, split_name, index)
    raw_noise, raw_setup, raw_evaluation = (
        simulated[name] for name in ("noise_capture", "setup_capture", "evaluation_capture"))
    raw_second = raw_evaluation[: raw_setup.size]
    noise, setup, second = (
        filtered_capture_for_analysis(audio, SAMPLE_RATE) for audio in (raw_noise, raw_setup, raw_second))
    row: dict[str, Any] = {
        **{key: simulated[key] for key in ("split", "index", "speaker", "environment", "snr_db", "level_dbfs")},
        "model": model, "development": development, "build": build_id(),
        "gender": manifest["speakers"][simulated["speaker"]]["gender"], "rt60_s": rt60_s(split_name, index),
        "level_check": verdict_from_audio(raw_setup, SAMPLE_RATE).code, "fired": [],
    }
    preset = Preset()
    with _taps() as tap:
        try:
            result = analyze_voice_setup(
                noise, setup, SAMPLE_RATE, "broadcast", noise_model=model, suppressor_strength=1.0,
                incumbent_settings={"gate": asdict(preset.gate), "rnnoise": asdict(preset.rnnoise),
                                    "eq": EQSettings().to_dict()},
                raw_noise_audio=raw_noise, raw_speech_audio=raw_setup)
        except ValueError as error:  # an analysis failure (A1-A4 territory), not a refusal
            return row | {"analysis_failure": str(error)}
    if result["eq_settings"] is None:  # Auto-EQ raised before it could validate
        return row | {"analysis_failure": str(result["eq_error"])}
    diagnostics = result["diagnostics"]
    tuning = diagnostics["joint_tuning"]
    b1 = next((entry for entry in tap["validation"] if not entry["passed"]), None)
    vad = bool(diagnostics["vad_available"])
    current = current_settings(vad)
    proposal = {name: deepcopy(result[name]) for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}
    if loudness._digest(proposal["limiter_settings"]) != loudness._digest(current["limiter_settings"]):
        raise RuntimeError("the proposal's limiter differs from the factory limiter; the limiter would be a step")
    # A setup the unclear-recording refusal ends never reaches verification.
    flow = None if b1 else verify_flow(result, {"noise": noise, "setup": setup, "second": second,
                                                "raw_noise": raw_noise, "raw_second": raw_second})
    fired = fired_vetoes({"reasons": diagnostics["uncertainty_reasons"], "weak_capture": diagnostics["weak_capture"],
                          "b1": b1 is not None, "tuner_decision": tuning.get("decision"),
                          "verification": None if flow is None else flow["decision"]})
    recommended = bool(diagnostics["apply_recommended"])
    row |= {
        "fired": fired, "recommended": recommended, "reasons": list(diagnostics["uncertainty_reasons"]), "b1": b1,
        "short_speech": float(diagnostics["vad_active_duration_s"]) < 2.0,
        "verification": None if flow is None else _verification_row(flow),
        "taps": {"validation": len(tap["validation"]), "tone": len(tap["tone"])},
        "e1": not bool(diagnostics["compressor_auto_makeup_enabled"]),
        "e2": bool(not diagnostics["deesser_enabled"] and (
            float(diagnostics["deesser_detection_probability"]) == 0.0
            or diagnostics["noise_reference_quality"].get("status") == "invalid")),
    }
    ctx = {"raw": raw_evaluation, "noise": noise.astype(np.float64), "labels": simulated["labels"],
           "clean": filtered_capture_for_analysis(simulated["evaluation_clean"], SAMPLE_RATE).astype(np.float64),
           "noisy": filtered_capture_for_analysis(raw_evaluation, SAMPLE_RATE).astype(np.float64)}
    ended = flow is not None and flow["decision"] in ("rollback", "gave_up")
    objects: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {"current": measure_settings(current, ctx, ("whole", "last10") if ended else ("whole",))}

    def add(name: str, settings: dict[str, Any], sections: tuple[str, ...], windows: tuple[str, ...] = ("whole",)) -> None:
        objects[name] = {"vacuous": is_vacuous(settings, current, sections),
                         "digests": {section: loudness._digest(settings[section]) for section in sections}}
        if not objects[name]["vacuous"]:
            metrics[name] = measure_settings(settings, ctx, windows)

    add("whole", proposal, SECTIONS)
    final = None
    if ended and flow is not None:
        final = {name: flow["state"][name] for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}
        add("final", final, SECTIONS, ("whole", "last10"))
    if "B5" in fired:
        if tap["tone"]:
            tone = proposal | {"eq_settings": {**result["eq_settings"], "band_gains": tap["tone"][-1]}}
            add("tone", hybrid(current, tone, "tone"), STEPS["tone"])
        else:
            objects["tone"] = {"vacuous": False}
    candidates: list[dict[str, Any]] = list(tuning.get("candidates") or [])
    current_key = voice_setup_eval._arm_key(model, 1.0, asdict(preset.gate))
    if "B6" in fired:
        best = tuner_withheld(candidates, current_key, tuning)
        row["tuner"] = {
            "decision": tuning.get("decision"), "holdout_evaluated": bool(tuning.get("holdout_evaluated")),
            "holdout_passed": bool(tuning.get("holdout_passed")), "withheld": best,
            "candidates": [{"noise_model": c["noise_model"], "strength": float(c["suppressor_strength"]),
                            "score": float(c["score"]), "passes": bool(c["passes"])} for c in candidates]}
        if best is None:
            objects["tuner"] = {"vacuous": False}
        else:
            withheld = proposal | {"gate_settings": best["gate_settings"], "suppressor_settings": {
                "enabled": True, "strength": best["suppressor_strength"], "model": best["noise_model"], "auto_tuned": True}}
            add("tuner", hybrid(current, withheld, "gate_suppression"), STEPS["gate_suppression"])
            row["tuner"]["any_candidate_no_worse"] = _tuner_upper_bound(
                candidates, current_key, model, ctx, float(tuning.get("noise_floor_db", -80.0)))
    first = first_whole(fired, recommended)
    if first:
        primary = proposal if OBJECT[first] == "whole" else final
        shown = row | {"objects": objects, "metrics": metrics}
        if primary and outcome(shown, OBJECT[first], veto=first)["status"] in ("correct", "partial"):
            for step, sections in STEPS.items():
                add(f"step:{step}", hybrid(current, primary, step), sections)
    row |= {"objects": objects, "metrics": metrics}
    if development:
        row["probes"] = {name: measure_settings(settings, ctx) for name, (_, settings) in probe_settings(current).items()}
        row["tone_cross_check_gap_db"] = _tone_cross_check(proposal, ctx, analyze_voice_spectrum)
    return row


def _tuner_upper_bound(candidates: list[dict[str, Any]], current_key: str, model: str, ctx: dict[str, Any],
                       noise_floor_db: float) -> dict[str, int]:
    """Tested candidates no worse than the current settings on the gate and suppressor stage alone."""
    from mic_eq.config import Preset

    case = {name: ctx[name] for name in ("clean", "noisy", "labels")}
    capture = ctx["noisy"].astype(np.float32)
    current = {"model": model, "strength": 1.0, "gate": asdict(Preset().gate)}
    baseline = gate_corpus.score(case, voice_setup_eval.render_arm(capture, current, noise_floor_db))
    tested = no_worse_count = 0
    for candidate in candidates:
        if _candidate_key(candidate) == current_key:
            continue
        arm = {"model": candidate["noise_model"], "strength": candidate["suppressor_strength"],
               "gate": dict(candidate["gate_settings"])}
        tested += 1
        scores = gate_corpus.score(case, voice_setup_eval.render_arm(capture, arm, noise_floor_db))
        no_worse_count += no_worse(scores, baseline)
    return {"tested": tested, "no_worse": int(no_worse_count)}


def _tone_cross_check(proposal: dict[str, Any], ctx: dict[str, Any], analyze_voice_spectrum: Any) -> float:
    """How far ``tone_error_db`` is from the same reading made with ``analyze_voice_spectrum`` (dry run only)."""
    output = np.asarray(loudness._render(proposal, ctx["raw"])["output_audio"], np.float64)[: ctx["clean"].size]
    spectra = [analyze_voice_spectrum(audio, SAMPLE_RATE) for audio in (ctx["clean"], output)]
    delta = spectra[1].median_spectrum_db - spectra[0].median_spectrum_db
    band = delta[(spectra[0].freqs >= 80.0) & (spectra[0].freqs <= 12_000.0)]
    theirs = float(np.sqrt(np.mean(np.square(band - np.median(band)))))
    return abs(theirs - tone_error_db(ctx["clean"], output, ctx["labels"]))


# ----------------------------------------------------------------------------------------------
# Counting and the report


def outcome(row: dict[str, Any], name: str, *, veto: str | None = None, window: str = "whole",
            sibilant_tolerance: float = 3.0, strict: bool = False) -> dict[str, Any]:
    """The status of one withheld object of a user."""
    if veto == "B2" and row.get("short_speech"):
        return {"status": "definition"}
    entry = row.get("objects", {}).get(name)
    if entry is None:
        return {"status": "unjudgeable", "why": "no object"}
    if entry["vacuous"]:
        return {"status": "vacuous"}
    withheld = row.get("metrics", {}).get(name, {}).get(window)
    current = row.get("metrics", {}).get("current", {}).get(window)
    if not withheld or not current:
        return {"status": "unjudgeable", "why": "no render"}
    judged = judge(withheld, current, sibilant_tolerance=sibilant_tolerance, strict=strict)
    return {"status": judged["verdict"], "failed": judged["failed"], "unmeasurable": judged["unmeasurable"]}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """The consistency every count below relies on."""
    _check(len({(row["split"], row["index"]) for row in rows}) == len(rows), "a user appears twice")
    for row in rows:
        fired = set(row.get("fired") or ())
        label = f"{row['split']}:{row['index']}"
        _check(fired <= set(VETOES), f"{label}: unknown veto")
        _check(not (fired and row.get("analysis_failure")), f"{label}: an analysis failure fired a veto")
        decision = (row.get("verification") or {}).get("decision")
        _check(("C1" in fired) == (decision == "rollback") and ("C2" in fired) == (decision == "gave_up"),
               f"{label}: C1 and C2 need a verification that decided, never a retry")


def assess(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_rows(rows)
    users = []
    for row in rows:
        fired: set[str] = set(row.get("fired") or [])
        first = first_whole(fired, bool(row.get("recommended")))
        users.append({"row": row, "fired": fired, "first": first, "speaker": row["speaker"],
                      "bucket": "analysis_failure" if row.get("analysis_failure") else first or "none"})
    return users


def _member(user: dict[str, Any], veto: str, track: str, counting: str) -> bool:
    if track == "whole" and counting == "first":
        return user["first"] == veto
    return veto in user["fired"]


def _outcomes(users: list[dict[str, Any]], veto: str, track: str, counting: str, **reading: Any) -> list[tuple[int, dict]]:
    name = OBJECT[veto] if track == "whole" else STEP_OBJECT[veto]
    return [(i, outcome(user["row"], name, veto=veto, **reading)) for i, user in enumerate(users)
            if _member(user, veto, track, counting)]


def population(users: list[dict[str, Any]]) -> dict[str, int]:
    """Whole-track buckets, with the arithmetic that must hold for every count of the report."""
    buckets = Counter(user["bucket"] for user in users)
    _check(sum(buckets.values()) == len(users), "the buckets do not add up to the users")
    for track, vetoes in (("whole", WHOLE_ORDER), ("step", tuple(STEP_OBJECT))):
        for veto in vetoes:
            first, anywhere = (_outcomes(users, veto, track, counting) for counting in ("first", "any"))
            _check(len(first) <= len(anywhere), f"{veto}: more first than any")
            for found in (first, anywhere):
                counts = Counter(result["status"] for _, result in found)
                _check(sum(counts[status] for status in STATUSES) == len(found), f"{veto}: a status is unknown")
    return dict(buckets)


def _flags(users: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([bool(user["row"].get(key)) for user in users], float)


def _share(flags: np.ndarray, users: list[dict[str, Any]]) -> list[float]:
    return gate_corpus.cluster_ci(flags, np.array([user["speaker"] for user in users]))


def _table_row(users: list[dict[str, Any]], veto: str, track: str, counting: str, **reading: Any) -> dict[str, Any]:
    false, upper = np.zeros(len(users)), np.zeros(len(users))
    counts, clauses = Counter(), Counter()
    found = _outcomes(users, veto, track, counting, **reading)
    for i, result in found:
        counts[result["status"]] += 1
        false[i] = false_refusal(result["status"], "lower")
        upper[i] = false_refusal(result["status"], "upper")
        if result["status"] == "correct":
            clauses.update(result["failed"])
    lower = _share(false, users)
    return {"veto": veto, "track": track, "counting": counting,
            "kind": "step" if track == "step" else "hard" if veto in HARD else "soft",
            "fired": len(found), **{status: counts[status] for status in STATUSES},
            "false_share_lower_bound": lower, "false_share_upper_bound": _share(upper, users),
            "correct_by_clause": dict(sorted(clauses.items())), "candidate_for_loosening": bool(lower[1] > TRIGGER_SHARE)}


def _false_counts(users: list[dict[str, Any]], only: tuple[str, ...] | None = None, **reading: Any) -> dict[str, list[int]]:
    counts = {}
    for veto in only or WHOLE_ORDER:
        found = [result["status"] for _, result in _outcomes(users, veto, "whole", "first", **reading)]
        counts[veto] = [sum(false_refusal(s, bound) for s in found) for bound in ("lower", "upper")]
    return counts


def _band(value: float, edges: tuple[float, ...], unit: str) -> str:
    return next((f"under {edge:g} {unit}" for edge in edges if value < edge), f"{edges[-1]:g} {unit} and over")


def _strata(users: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "level_check": lambda row: row["level_check"], "gender": lambda row: row["gender"],
        "rt60": lambda row: _band(row["rt60_s"], (0.367, 0.533), "s"),
        "snr": lambda row: _band(row["snr_db"], (10.0, 20.0), "dB"),
    }
    strata: dict[str, Any] = {}
    for name, key in keys.items():
        groups: dict[str, Any] = {}
        for user in users:
            group = groups.setdefault(key(user["row"]), {"users": 0, "first": Counter(), "false_first": Counter()})
            group["users"] += 1
            group["first"][user["bucket"]] += 1
            if user["first"] and outcome(user["row"], OBJECT[user["first"]], veto=user["first"])["status"] == "false":
                group["false_first"][user["first"]] += 1
        strata[name] = {label: {"users": group["users"], "first": dict(sorted(group["first"].items())),
                                "false_first": dict(sorted(group["false_first"].items()))}
                        for label, group in sorted(groups.items())}
    return strata


def clause_coverage(users: list[dict[str, Any]]) -> dict[str, Any]:
    """For every judged object of every user, how often each clause could not be measured."""
    unmeasurable, cases = Counter(), 0
    for user in users:
        row = user["row"]
        for name in row.get("objects", {}):
            if outcome(row, name)["status"] in ("false", "correct", "partial"):
                cases += 1
                unmeasurable.update(judge(row["metrics"][name]["whole"], row["metrics"]["current"]["whole"])["unmeasurable"])
    return {"cases": cases, "unmeasurable": {clause: unmeasurable[clause] for clause in CLAUSES},
            "over_15_percent": [clause for clause in CLAUSES if unmeasurable[clause] > UNMEASURABLE_LIMIT * cases]}


def _step_breakdown(users: list[dict[str, Any]]) -> dict[str, Any]:
    """For each correctly refused whole setup, which single steps, applied alone, are worse than the current settings."""
    refused = [user for user in users if user["first"]
               and outcome(user["row"], OBJECT[user["first"]], veto=user["first"])["status"] in ("correct", "partial")]
    names = {"correct": "worse", "false": "not_worse"}
    steps = {}
    for step in STEPS:
        found = [outcome(user["row"], f"step:{step}") for user in refused]
        counts = Counter(result["status"] for result in found)
        steps[step] = {names.get(status, status): counts[status]
                       for status in ("correct", "false", "partial", "vacuous", "unjudgeable")}
        steps[step]["worse_by_clause"] = dict(sorted(Counter(
            clause for result in found if result["status"] == "correct" for clause in result["failed"]).items()))
    alone = sum(all(outcome(user["row"], f"step:{step}")["status"] != "correct" for step in STEPS) for user in refused)
    return {"refused_users": len(refused), "no_single_step_worse": alone, "steps": steps}


def refusal_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _check(not any(row.get("development") for row in rows), "development rows are spent users and give no rates")
    _check({(row["split"], row["index"]) for row in rows} == {(SPLIT, index) for index in STUDY_INDICES},
           "the rows are not the 54 study users")
    users = assess(rows)
    buckets = population(users)
    table = [_table_row(users, veto, "whole", counting) for veto in WHOLE_ORDER for counting in ("first", "any")]
    table += [_table_row(users, veto, "step", "any") for veto in STEP_OBJECT]
    flags = {name: np.zeros(len(users)) for name in ("whole", "any_track", "hard", "soft")}
    for i, user in enumerate(users):
        first = user["first"]
        whole_false = bool(first) and outcome(user["row"], OBJECT[first], veto=first)["status"] == "false"
        step_false = any(outcome(user["row"], STEP_OBJECT[veto])["status"] == "false"
                         for veto in STEP_OBJECT if veto in user["fired"])
        flags["whole"][i], flags["any_track"][i] = whole_false, whole_false or step_false
        flags["hard"][i], flags["soft"][i] = whole_false and first in HARD, whole_false and first not in HARD
    readings = {"sibilant_1_5_db": {"sibilant_tolerance": 1.5}, "sibilant_6_0_db": {"sibilant_tolerance": 6.0},
                "strict_overload": {"strict": True}}
    sensitivity: dict[str, Any] = {name: _false_counts(users, **reading) for name, reading in readings.items()}
    sensitivity["last_10_s_for_c1_c2"] = _false_counts(users, ("C1", "C2"), window="last10")
    verifications = [user["row"]["verification"] for user in users if user["row"].get("verification")]
    tuner = [user["row"]["tuner"] for user in users if user["row"].get("tuner")]
    return {
        "users": len(users), "speakers": len({user["speaker"] for user in users}), "buckets": buckets,
        "vetoes": table,
        "candidates_for_loosening": sorted({f"{r['veto']}:{r['track']}" for r in table if r["candidate_for_loosening"]}),
        "reading": {"false_share_lower_bound": "[share, lower, upper] of all users, speaker-clustered 95 % bootstrap; an "
                                               "unmeasurable clause counts as worse (the headline)",
                    "false_share_upper_bound": "the same with an unmeasurable clause counted as not worse"},
        "users_with_a_false_first_veto": {name: _share(flag, users) for name, flag in flags.items()},
        "firing_without_judgement": {
            "E1_automatic_makeup_withheld": _share(_flags(users, "e1"), users),
            "E2_deesser_left_off_for_invalid_evidence": _share(_flags(users, "e2"), users)},
        "tuner_step": {"fired": len(tuner), "no_candidate": sum(t["withheld"] is None for t in tuner),
                       "cause": dict(Counter(t["withheld"]["cause"] for t in tuner if t["withheld"])),
                       "any_candidate_no_worse_upper_bound": sum(
                           (t.get("any_candidate_no_worse") or {}).get("no_worse", 0) > 0 for t in tuner)},
        "per_step_breakdown": _step_breakdown(users), "sensitivity": sensitivity, "clause_coverage": clause_coverage(users),
        "strata": _strata(users), "retake_costs": dict(Counter(v["retry"] for v in verifications if v["retry"])),
        "verification_decisions": dict(Counter(v["decision"] for v in verifications)),
        "unreachable": {"A1-A4": "the simulator makes no clipped, silent, short or non-finite takes",
                        "never_fired": [veto for veto in VETOES if not any(veto in user["fired"] for user in users)]},
        "trigger": f"a veto is a candidate for a loosening study when the lower bound of its false-refusal share of all "
                   f"users exceeds {TRIGGER_SHARE:.0%}, counted first or in any; no veto is changed by this study",
    }


def _same_reading(a: Any, b: Any) -> bool:
    return bool(a == b or (_finite(a, b) and abs(a - b) <= 1e-6) or (a != a and b != b))


def bookkeeping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reachability, clause coverage, probes and arithmetic of the development rows: counts and flags, no rates."""
    users = assess(rows)
    buckets = population(users)
    reachability = {}
    for veto in VETOES:
        for track, objects in (("whole", OBJECT), ("step", STEP_OBJECT)):
            if veto in objects:
                found = [result["status"] for _, result in _outcomes(users, veto, track, "any")]
                reachability[veto if track == "whole" else f"{veto}_step"] = {
                    "fired": len(found), **{status: found.count(status) for status in STATUSES}}
    targets = {name: target for name, (target, _) in probe_settings(current_settings(True)).items()}
    reference = [user["row"] for user in users if "whole" in user["row"].get("metrics", {}).get("current", {})]
    judged = {name: [(row, judge(row["probes"][name]["whole"], row["metrics"]["current"]["whole"]))
                     for row in reference if "whole" in row["probes"][name]] for name in targets}
    good = Counter(clause for name, target in targets.items() if not target for _, result in judged[name]
                   for clause in result["failed"])
    probes: dict[str, Any] = {}
    for name, target in targets.items():
        results = [result for _, result in judged[name]]
        if target:
            tripped = sum(target in result["failed"] for result in results)
            measurable = sum(result["clauses"][target] != "unmeasurable" for result in results)
            probes[name] = {"clause": target, "tripped": tripped, "measurable": measurable,
                            "trips_on_most": tripped * 2 > measurable, "vacuous": tripped == 0 or good[target] > 0}
        else:
            identical = sum(all(_same_reading(row["probes"][name]["whole"][key], value)
                                for key, value in row["metrics"]["current"]["whole"].items()) for row, _ in judged[name])
            probes[name] = {"users": len(results), "identical_to_current": identical,
                            "clause_failures": dict(Counter(c for result in results for c in result["failed"]))}
    gaps = [row["tone_cross_check_gap_db"] for row in rows if "tone_cross_check_gap_db" in row]
    return {
        "development_users": len(users), "buckets": buckets, "reachability": reachability,
        "never_fired": [veto for veto in VETOES if not any(veto in user["fired"] for user in users)],
        "constructed_capture_probes": capture_vetoes_probe(), "clause_coverage": clause_coverage(users), "probes": probes,
        "tone_cross_check": {"users": len(gaps), "within_half_db": sum(gap <= 0.5 for gap in gaps)},
        "level_check": dict(Counter(row["level_check"] for row in rows)), "arithmetic": "checked",
    }


# ----------------------------------------------------------------------------------------------
# A hand-made world, for the dry run and the tests


def fabricated_rows() -> list[dict[str, Any]]:
    """The 54 study users with hand-made metrics that cover every bucket and every verdict."""
    ok = {"estoi": 0.5, "tone_error_db": 2.0, "sibilant_loss_db": 1.0, "deep_pause_db": -10.0, "tail_db": -8.0,
          "onset_db": -1.0, "pause_db": -9.0, "clipped": False, "limiter_gain_reduction_db": 0.5,
          "true_peak_limiter_gain_reduction_db": 0.1}
    readings = {"same": ok, "worse": {**ok, "tone_error_db": 4.0}, "boundary": {**ok, "tone_error_db": 3.0},
                "unknown": {**ok, "deep_pause_db": None}, "clip": {**ok, "clipped": True}}

    def user(i: int, fired: tuple[str, ...] = (), *, recommended: bool = True, decision: str | None = "accept",
             retry: str | None = None, cause: str | None = None, short: bool = False, **objects: str) -> dict[str, Any]:
        """``objects`` maps a withheld object to its kind: a reading above, ``vacuous`` or ``error``."""
        metrics: dict[str, Any] = {"current": {"whole": ok, "last10": ok}}
        for name, kind in objects.items():
            if kind != "vacuous":
                metrics[name] = ({"error": "RuntimeError: render failed"} if kind == "error"
                                 else {"whole": readings[kind], "last10": readings[kind]})
        row = {
            "split": SPLIT, "index": STUDY_INDICES[i], "speaker": f"p{i // 3}", "model": MODEL, "development": False,
            "gender": "female" if i // 3 < 2 else "male", "rt60_s": 0.2 + 0.5 * (i % 7) / 6,
            "snr_db": 3.0 + 27.0 * (i % 5) / 4, "level_check": "too_hot" if i % 9 == 0 else "good",
            "fired": list(fired), "recommended": recommended, "short_speech": short, "e1": i % 4 == 0, "e2": i % 6 == 0,
            "verification": None if decision is None else {"decision": decision, "retry": retry, "adjustments": 0},
            "objects": {name: {"vacuous": kind == "vacuous"} for name, kind in objects.items()}, "metrics": metrics,
        }
        if "B6" in fired:
            row["tuner"] = {"decision": "retain_incumbent", "withheld": {"cause": cause} if cause else None,
                            "any_candidate_no_worse": {"tested": 3, "no_worse": 1}}
        return row

    rows = [
        user(0, ("B1",), decision=None, whole="same"), user(1, ("B1",), decision=None, whole="worse"),
        user(2, ("B1", "B4"), decision=None, whole="same"),
        user(3, ("B2",), recommended=False, whole="same"), user(4, ("B2",), recommended=False, whole="unknown"),
        user(5, ("B2",), recommended=False, whole="worse"), user(6, ("B2",), recommended=False, short=True, whole="worse"),
        user(7, ("B3",), recommended=False, whole="same"), user(8, ("B4",), recommended=False, whole="clip"),
        user(9, ("B4",), recommended=False, whole="boundary"),
        user(10, ("B5",), recommended=False, whole="same", tone="same"),
        user(11, ("B5",), recommended=False, whole="vacuous", tone="vacuous"),
        user(12, ("B5",), recommended=False, whole="worse", tone="worse"),
        user(13, ("C1",), decision="rollback", whole="same", final="same"),
        user(14, ("C1",), decision="rollback", whole="same", final="worse"),
        user(15, ("C2",), decision="gave_up", whole="same", final="worse"),
        user(16, ("C2",), decision="gave_up", whole="same", final="same"),
        user(17, ("B2", "C1"), recommended=False, decision="rollback", whole="same", final="worse"),
        user(18, ("B6",), cause="margin", tuner="same"), user(19, ("B6",), cause="holdout", tuner="worse"),
        user(20, ("B6",)), user(21, (), decision="retry", retry="D1", whole="same"),
        user(22, ("B3",), recommended=False, whole="error"),
    ]
    rows += [user(i) for i in range(len(rows), len(STUDY_INDICES) - 1)]
    failure = {"analysis_failure": "Recording too unclear. Please try again.", "objects": {}, "metrics": {}}
    rows.append(user(len(STUDY_INDICES) - 1, decision=None) | failure)
    for row in rows:
        first = first_whole(row["fired"], row["recommended"])
        if first and outcome(row, OBJECT[first], veto=first)["status"] in ("correct", "partial"):
            for step in STEPS:
                row["objects"][f"step:{step}"] = {"vacuous": False}
                row["metrics"][f"step:{step}"] = {"whole": readings["worse" if step == "tone" else "same"]}
    return rows


# ----------------------------------------------------------------------------------------------
# Files and commands


def check_indices(split: str, indices: list[int], development: bool) -> None:
    """A study run never reaches a spent user; the dry run reaches only the development ones."""
    if split not in voice_setup_eval.SPLITS:
        raise SystemExit(f"unknown split {split!r}")
    if development and any(not 0 <= index < DEVELOPMENT_COUNT for index in indices):
        raise SystemExit(f"development users are indices 0-{DEVELOPMENT_COUNT - 1}")
    if not development and any(index < SPENT_BELOW for index in indices):
        raise SystemExit(f"indices below {SPENT_BELOW} are spent users; only probe --development may run them")


def parse_indices(text: str) -> tuple[str, list[int]]:
    split, _, spec = text.partition(":")
    indices: list[int] = []
    for part in spec.split(","):
        low, _, high = part.partition("-")
        indices += range(int(low), int(high or low) + 1)
    return split, indices


def worker_count(requested: int) -> int:
    return max(1, min(requested, MAX_WORKERS))


def check_resume(partial: Path, build: str) -> None:
    """A partial file resumes whatever code wrote it, so rows from another build are refused."""
    if not partial.exists():
        return
    stale = [row for row in map(json.loads, partial.read_text(encoding="utf-8").splitlines()) if row.get("build") != build]
    if stale:
        raise SystemExit(f"{partial} holds {len(stale)} rows from another build; delete it to start over")


def frozen_hashes(root: Path = voice_setup_eval.DEFAULT_ROOT) -> dict[str, str]:
    """What must not change between the freeze and the run."""
    assets = {"extension": REPO_ROOT / loudness.EXTENSION, "df.dll": REPO_ROOT / "df.dll",
              "vad_model": Path(os.environ.get("VAD_MODEL_PATH") or REPO_ROOT / "models" / "silero_vad.onnx"),
              "corpus_manifest": root / "manifest.json"}
    return {"rule": hashlib.sha256((__doc__ or "").encode()).hexdigest(),
            **{name: loudness._sha256(REPO_ROOT / name) for name in SOURCES},
            **{name: gate_corpus._sha256(path) for name, path in assets.items()}}


def check_frozen(path: Path, root: Path = voice_setup_eval.DEFAULT_ROOT) -> None:
    if not path.exists():
        raise SystemExit(f"{path} is missing; run probe --development, then freeze, before the study")
    frozen, now = json.loads(path.read_text(encoding="utf-8")), frozen_hashes(root)
    changed = sorted(name for name in now if frozen.get(name) != now[name])
    if changed:
        raise SystemExit(f"changed since the freeze: {', '.join(changed)}")


def identity(root: Path) -> dict[str, str]:
    hashes = frozen_hashes(root)
    return {"build": build_id(), "extension_sha256": hashes["extension"], "df_dll_sha256": hashes["df.dll"],
            "vad_model_sha256": hashes["vad_model"], "manifest_sha256": hashes["corpus_manifest"]}


def build_report(source: dict[str, Any]) -> dict[str, Any]:
    rows, who = source["rows"], source["identity"]
    return {
        "schema_version": 2, "audible_change": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Voice Setup refusals: how often a withheld setup was no worse than the current settings",
        "predefined_rule": "The rule in the docstring of python/tools/evaluate_voice_setup_refusals.py.",
        "rule_sha256": hashlib.sha256((__doc__ or "").encode()).hexdigest(),
        "split": SPLIT, "model": MODEL, "user_indices": {SPLIT: sorted(row["index"] for row in rows)},
        "metrics": refusal_report(rows), "limitations": list(LIMITATIONS), "flattering": list(FLATTERING),
        "source_sha256": {name: loudness._sha256(REPO_ROOT / name) for name in SOURCES},
        "implementation_sha256": {loudness.EXTENSION: who["extension_sha256"]},
        "asset_hashes": {"corpus_manifest": who["manifest_sha256"], "vad_model": who["vad_model_sha256"],
                         "df_dll": who["df_dll_sha256"]},
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, default=loudness._plain) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "probe", "report", "freeze"))
    parser.add_argument("--root", type=Path, default=voice_setup_eval.DEFAULT_ROOT, help="the corpus root")
    parser.add_argument("--indices", default=DEFAULT_INDICES, help="run: split:ranges")
    parser.add_argument("--development", action="store_true", help="probe: confirm that the 36 spent users run again")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS, help="report: rows written by run")
    parser.add_argument("--output", type=Path, help="run: rows; probe: bookkeeping; report: the tracked report")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        RULES_FINAL.parent.mkdir(parents=True, exist_ok=True)
        RULES_FINAL.write_text(__doc__ or "", encoding="utf-8", newline="\n")
        _write(FROZEN, frozen_hashes(args.root))
        return 0
    if args.command == "report":
        source = json.loads(args.rows.read_text(encoding="utf-8"))
        if source.get("development"):
            raise SystemExit("the rows are development rows: spent users give no rates")
        _write(args.output or DEFAULT_REPORT, build_report(source))
        return 0
    development = args.command == "probe"
    if development and not args.development:
        raise SystemExit("probe re-runs the 36 spent development users; pass --development")
    if development:
        jobs = [(str(args.root), split, index, MODEL) for split, index in DEVELOPMENT_USERS]
        output, rows_path = args.output or DEFAULT_BOOKKEEPING, DEVELOPMENT_ROWS
    else:
        split, indices = parse_indices(args.indices)
        check_indices(split, indices, False)
        check_frozen(FROZEN, args.root)
        jobs = [(str(args.root), split, index, MODEL) for index in indices]
        output = rows_path = args.output or DEFAULT_ROWS
    partial = rows_path.with_suffix(".run.partial.jsonl")
    check_assets()
    check_resume(partial, build_id())
    print("build:", build_id(), "| VAD_MODEL_PATH:", os.environ.get("VAD_MODEL_PATH"), flush=True)
    rows = voice_setup_eval.run(functools.partial(evaluate_user, development=development), jobs,
                                worker_count(args.workers), partial)
    source = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "development": development,
              "identity": identity(args.root), "rows": rows}
    _write(rows_path, source)
    partial.unlink()
    if development:
        from check_evaluation_hygiene import validate_report

        fabricated = rows_path.parent / "fabricated-report.json"
        _write(fabricated, build_report({"rows": fabricated_rows(), "identity": source["identity"]}))
        _write(output, bookkeeping(rows) | {"fabricated_report_hygiene_errors": validate_report(fabricated)})
    sys.stdout.write(f"{len(rows)} rows\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
