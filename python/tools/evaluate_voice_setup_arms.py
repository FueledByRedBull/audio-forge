"""Compare three Voice Setup changes, alone and together, on DAPS script 1 recorded on consumer devices.

Subcommands:
  probe    ``probe --development``: the bookkeeping dry run on the two development speakers (m2, f10; 24 recordings in
           both conditions, 48 user-conditions). It writes construction readings, parity, switch scope, determinism,
           probes, a gate self-test, reachability flags, a resume check and timing as counts and flags; no rate and no
           interval is computed from development users.
  freeze   Write ``rules-final.md`` (this text) and ``frozen-files.json`` (hashes of this rule, every ``python/mic_eq``
           module and the tools it imports, the native extension, df.dll, ONNX Runtime and its providers DLL, the
           DeepFilter and VAD models, the DAPS manifest, what ``release-assets.json`` declares for each of those
           assets, and the Python, NumPy and SciPy versions). It fails on an absent asset and does not replace a freeze
           once rows exist.
  run      One job per (speaker, version, condition), factory users first, existing next, balcony last; at most 4
           workers; rows are appended to ``rows.run.partial.jsonl`` and resumed only if they were made by this build for
           these jobs. Starts only when the hashes written by ``freeze`` still match, refuses a finished rows file, and
           stops when a row shows that the VAD did not run as Silero or that the tuner lacked a suppressor model. Every
           corpus file a user is built from is checked against ``MANIFEST.json``.
  report   Judge the rows and write the tracked report with the frozen hashes. Refuses development rows, a changed tree
           and a missing main job.
This tool imports the refusal study's helpers (``evaluate_voice_setup_refusals.py``, whose rule is frozen and whose report
is published); it never changes them.

Rule (fixed before any study row existed)

Changes. Each is a module global, off by default, read at call time: P ``voice_setup.PARTIAL_RESULTS`` (withhold only the
tone when only the tone fails); T ``voice_setup.TONAL_CHECK_COUNTS_EVERY_BAND`` (a by-band reading in verification, which
only adds rollbacks); R ``config_parts.settings.DEFAULT_SUPPRESSOR_STRENGTH`` 0.7 (1.0 off). ``arm_switches`` sets the
three globals for an arm, restores them and asserts the restoration. ``BUILTIN_PRESETS`` keeps 1.0, so every user's current
settings are built with ``Preset()`` after the globals are set, the strength is passed explicitly to the analysis, to
``incumbent_settings`` and to every render, and ``Preset().rnnoise.strength`` is asserted to equal the arm's value.

Arms (8): current; P; T; R; P+T; P+R; T+R; P+T+R. Arms that differ only by T share one analysis, because T acts only in
verification (``analysis_key``). For an existing user R is a no-op, as saved settings keep strength 1.0: R arms are
mapped to their non-R twin, and the dry run asserts that this is exact. In R arms the current settings of a factory
user have strength 0.7, so each arm is judged against its own current settings.

Data (DAPS, CC BY-NC 4.0: evaluation only, nothing shipped is fitted on it). ``models/daps_script1/<version>/
<speaker>_script1_<version>.wav``, 44.1 kHz mono; every file used is checked against ``MANIFEST.json``. Each whole file is
resampled once to 48 kHz with ``mic_eq.analysis.signal_processing.resample_poly(x, 160, 147)``. Development speakers m2 and
f10 (all 12 versions) set bookkeeping and construction thresholds only. Held-out speakers f1-f9, m1, m3-m10 (k = 0..17).
Non-balcony versions v = 0..9: ipad_office1, ipad_office2, ipad_confroom1, ipad_confroom2, ipad_livingroom1,
ipad_bedroom1, ipadflat_office1, ipadflat_confroom1, iphone_livingroom1, iphone_bedroom1. Speaker k gets versions
(k + j) mod 10 for j = 0..4: 90 users, 9 speakers per room. Two conditions per user: factory (``Preset()``, flat EQ) and
existing (factory plus tone EQ library entry (k + v) mod 3, built with ``refusals.shifted_tone`` on the nearest tone
bands: E0 +4 dB high shelf 5 kHz and -2 dB bell 300 Hz; E1 +3 dB low shelf 150 Hz and -2 dB bell 3.5 kHz; E2 +3 dB bell
2.5 kHz and -3 dB low shelf 120 Hz): N = 180 user-conditions in 18 speaker clusters. Balcony stratum: each held-out speaker
one balcony recording (ipad_balcony1 for even k, iphone_balcony1 for odd k), factory condition, all arms, reported only,
never in a gate.

Construction. Speech labels come from the clean take only: 10 ms frames, speech when the frame level is within
``SPEECH_WITHIN_DB`` of ``gate_corpus._active_level_db(clean)`` (20 dB: the specified 40 dB labels 99.9 % of every DAPS
clean take as speech, whose noise floor is about 35 dB under its active level, and leaves no pause), gaps under 250 ms
closed, islands under 50 ms dropped. Positions come from the clean labels, so rooms are paired; none is chosen from
device audio.
  Noise: noise-only runs trimmed 500 ms after speech and 100 ms before the next onset, ignoring the file's first 200 ms.
  The leading silence if it reaches 1.5 s, else the longest run (ties earliest) whose setup window fits; the
  central 2.0 s, or the whole run if it is 1.5-2.0 s. With no run of 1.5 s, the runs of 0.4 s or more are spliced in time
  order with 20 ms raised-cosine crossfades to 2.0 s and the user is flagged ``noise_spliced``. Under 2.0 s of pause
  material: excluded (counted and listed, never replaced).
  Voice: 10.0 s from 0.5 s before the first speech onset after the noise segment.
  Evaluation: 20.0 s from a pause midpoint at least 20 s after the setup window ends; the candidate with the most
  deep-pause frames (600 ms or more from speech on both sides), ties earliest. Verification takes are
  ``refusals.take_windows``, as in the refusal study.
  Alignment, on the evaluation window: band-pass 300-3400 Hz, cross-correlate clean and device over +-50 ms separately
  on the first and last 5 s; both lags within 20 ms and agreeing within 48 samples (1 ms;
  the takes drift 1-2 samples per second against the clean take, so 12 samples rejected nearly every window on the
  development speakers, and every clause reads 10 ms frames or longer), then the device window is shifted by their
  integer mean; otherwise the next candidate window, otherwise the user is excluded. Content: the Pearson
  correlation of 10 ms log-energy envelopes on speech frames must be at least ``MIN_CORRELATION`` (0.5 to start, set on
  development). A gain drift above 3 dB between the setup and evaluation windows is flagged, not excluded.
  Level: the DAPS device files are normalised near full scale, so the level check calls most takes too hot and a user
  who follows its advice lowers the gain; in the dialog the level check comes before the setup recording. So, after the
  file is resampled to 48 kHz and before any capture is cut or vetoed, the first voice (setup) take of the recording as
  it is, is judged with ``mic_eq.ui.level_check.verdict_from_audio``. If its code is ``too_hot``, the whole device
  recording is scaled by one gain that puts the peak reading of that function (its ignored-peak logic included) at
  ``level_check.target_peak_db()`` (-6 dB; -11 dB with ``LOWER_LEVEL_TARGET``). The capture vetoes and retakes below run on the scaled recording, and every
  capture of the user (noise, voice, evaluation, verification takes) comes from it; the clean take is not scaled. A
  take that still trips the clip rule after scaling carries clipping no gain change undoes, and is refused at capture as
  before. Any other verdict: levels as recorded. Each recording records the verdict of the first take before scaling
  (the row's ``level_check``), the gain in dB (0 if none) and the verdict of the chosen voice capture after. The
  scaling is part of construction, so it is identical in every arm.
  Capture vetoes now reachable: before analysis the dialog's checks are emulated: ``raw_capture_is_clipped`` on the noise
  and voice takes, ``analyze_noise_reference(noise, None).must_recapture`` and ``capture_is_silent(voice)``. A veto asks
  for the next noise segment (after a noise veto) or the next setup window, up to 3 takes; if all fail the user is refused
  at capture (A), judged on the first take with the analysis' own clip check tapped off, as B1 is.
  Existing condition: ``incumbent_settings`` carries the current EQ, and a proposal is stacked on the user's EQ (typed
  bands over it) before verification and renders; ``loudness._applied_eq`` stacks on a flat EQ, which is wrong here.

Measures: the clauses, tolerances, readings, statuses (false, correct, partial, definition, vacuous, unjudgeable) and the B1
counterfactual of the refusal study, unchanged.
  End state E, per user-condition and arm: the candidate as it ended verification when the analysis recommended it and
  verification accepts (for a P partial result, the verified rest with the current EQ); otherwise the current settings.
  Users follow the recommendation.
  False refusal: the headline counts an unmeasurable clause as worse, the other reading is reported. Whole-setup vetoes
  (A, B1-B5, C1, C2) judge the proposal (the final candidate for C1 and C2); step vetoes (B5 tone, B6 tuner, P's withheld
  tone) judge the step alone on the current settings.
  False accept: E is not vacuous and at least one measurable clause fails against the current settings; the headline
  counts an unmeasurable clause as not worse, the other reading is reported.
  P and the tap: ``refusals._taps`` would hide P's tone-only path. When it logs a failed validation in a P arm the
  analysis is run again untapped; if that run withholds only the tone it is the product's flow and the tapped run only
  supplies the withheld tone as the step object.
  Injections (BEN-T), factory users, for the analyses of T alone and of all three: +-12 dB (gated) and +-9 dB (reported) at
  the tone bands nearest 100 Hz and 200 Hz, injected into the candidate entering verification (first take), with T off
  and on. A pair counts only if verification was not a retry.

Budgets and margin rule (fixed; N = 180). Whole-setup veto 5 % (9 user-conditions); step veto 10 % (18); all whole-setup
vetoes together 10 % (18). The budget gates (BUD, S2, S3) count each user-condition's first whole-setup veto when it is
a false refusal in the headline reading (an unmeasurable clause counts as worse), each step veto when the step alone is
no worse, and all first false refusals together; the budget counts 9, 18 and 18 stay fixed whatever the number of
exclusions. Every gate passes or fails on its observed count against its threshold; equality passes. The
speaker-clustered 95 % interval (``gate_corpus.cluster_ci``, 4000 resamples, seed 0) is reported and never decides. A pass
whose interval reaches past the limit is labelled "within budget on this cohort, not confirmed at 95 %". A miss by one
is a miss. A gate whose minimum denominator is not reached fails.

Change gates. Each change is judged on K1 (the change alone against current) and K2 (all three against all but that
change); every gate must pass on both.
  FA (P, R)       false accepts with the change no more than without it. For T it is asserted (T only adds
                  rollbacks), not gated.
  BUD (P, T, R)   each whole-setup veto no more than max(9, count without); each step veto no more than max(18, count
                  without); all whole-setup vetoes together no more than max(18, count without).
  BEN-P           whole-setup false refusals together fewer with P than without.
  BEN-T           in each of the four 12 dB cells at least 30 pairs where the tonal check ran, and at least 80 % of them
                  reported as target_error.
  BEN-R           factory users. R1: among users with a measurable deep pause (30 needed), the share whose E keeps
                  deep_pause_db at -4.0 or lower is at least min(0.90, the share without R). R2: users whose E loses
                  more than 3.0 dB of sibilant level are fewer with R than without.
Shipping gates (candidate against current): S1 false accepts no more than current's; S2 each veto no more than
max(its budget, current's); S3 all whole-setup vetoes together no more than max(18, current's).

Decision. A change passes only if all its gates pass on both contrasts. No failed change: the candidate is P+T+R. Exactly
one failed change X: the pair without X. Two or more: no candidate. The candidate ships only if S1-S3 pass; otherwise the
current build stays. No other arm ever ships.

Reported, not gated: outcome buckets per arm; the refusal study's veto table per arm; clause breakdown; ESTOI; retake
costs; the sensitivity readings of the refusal study; strata (device, room, condition, gender, spliced noise, level
drift); the balcony stratum.

Dry run (``probe --development``). Construction readings per recording (lags, drift, envelope correlation, noise segment,
deep-pause frames, speech fraction, peaks, the level verdicts before and after scaling and the gain); parity of the current arm with the master code path on 4 users; T leaves the
analysis digest identical, R arms equal their twins for existing users, P differs only where the tone is withheld, every
constant restored; determinism on 4 users and the largest tuner runtime factor (``UNBOUNDED_TUNER_RUNTIME`` is decided
before the freeze: the dry run's largest factor was 0.78, above 0.5, so the runtime gate is lifted in every arm); the refusal study's seven known-bad probes plus "current again" and "limiter off", both conditions;
a gate self-test with fabricated worlds just inside and outside every threshold and the decision table for 0, 1 and 2
failed changes; reachability flags (P's tone_withheld fires, the current arm has a whole-setup false refusal, for each T
cell the tonal check ran on the injected candidate for at least half of the factory users, users refused before
verification counting in the denominator; at least half of the factory users have a measurable deep pause, as R1 needs 30
of about 85; a sibilant loss above 3 dB for a factory user, as R2 counts factory users only); resume;
timing; and a fabricated report that passes ``check_evaluation_hygiene.validate_report``.

The report states: loudspeaker playback of read speech, not a live talker; one script; the reference is the dry studio
take; device and room are confounded; labels come from clean energy; spliced-noise users; 18 clusters and budgets not
confirmed at 95 %; the current settings are the factory preset plus a hand-set EQ library; R evidence comes from factory
users only; the tone clause is nearly blind below 1 kHz, so T's benefit rests on injected errors; the B1 and A
counterfactuals; df.dll is a local build whose hash differs from release-assets.json; RNNoise 0.7 and the by-band check were
rejected before and DAPS is the new evidence; and the refusal study's finding that the compressor step is the only worse
step for most correctly refused setups bounds what any change here can show.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Container, Iterator
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

import evaluate_gate_corpus as gate_corpus
import evaluate_voice_setup as voice_setup_eval
import evaluate_voice_setup_loudness as loudness
import evaluate_voice_setup_refusals as refusals

REPO_ROOT = refusals.REPO_ROOT
SAMPLE_RATE = refusals.SAMPLE_RATE
DAPS_RATE = 44_100
FRAME = gate_corpus.FRAME  # 10 ms
SECTIONS = loudness.SECTIONS
DAPS = REPO_ROOT / "models" / "daps_script1"
DETAILS = REPO_ROOT / "models" / "evaluation-details" / "voice-setup-arms-2026-10"
DEFAULT_ROWS = DETAILS / "rows.json"
DEVELOPMENT_ROWS = DETAILS / "development" / "rows.json"
DEFAULT_BOOKKEEPING = DETAILS / "development" / "bookkeeping.json"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "voice-setup-arms-2026-10-report.json"
FROZEN = DETAILS / "frozen-files.json"
RULES_FINAL = DETAILS / "rules-final.md"
SOURCES = (*refusals.SOURCES, "python/tools/evaluate_voice_setup_arms.py")
MAX_WORKERS = refusals.MAX_WORKERS
# Decided on the development recordings before the freeze (bookkeeping item 4): True lifts the tuner's runtime gate
# in every arm, so that wall-clock load cannot change a candidate.
UNBOUNDED_TUNER_RUNTIME = True

# Arms and switches
CHANGES = ("P", "T", "R")
ARMS = ("current", "P", "T", "R", "P+T", "P+R", "T+R", "P+T+R")
R_STRENGTH = 0.7
# change: (module, name, value when on, value when off)
SWITCHES = {"P": ("mic_eq.analysis.voice_setup", "PARTIAL_RESULTS", True, False),
            "T": ("mic_eq.analysis.voice_setup", "TONAL_CHECK_COUNTS_EVERY_BAND", True, False),
            "R": ("mic_eq.config_parts.settings", "DEFAULT_SUPPRESSOR_STRENGTH", R_STRENGTH, 1.0)}

# Users
HELD_OUT = (*(f"f{i}" for i in range(1, 10)), "m1", *(f"m{i}" for i in range(3, 11)))
DEVELOPMENT_SPEAKERS = ("m2", "f10")
SPEAKERS = (*HELD_OUT, *DEVELOPMENT_SPEAKERS)  # index k; 18 and 19 are the development speakers
VERSIONS = ("ipad_office1", "ipad_office2", "ipad_confroom1", "ipad_confroom2", "ipad_livingroom1", "ipad_bedroom1",
            "ipadflat_office1", "ipadflat_confroom1", "iphone_livingroom1", "iphone_bedroom1")
BALCONY = ("ipad_balcony1", "iphone_balcony1")
ALL_VERSIONS = (*VERSIONS, *BALCONY)
VERSIONS_PER_SPEAKER = 5
CONDITIONS = ("factory", "existing", "balcony")  # balcony users are factory users, reported only
# The tone EQ library: (frequency, gain, filter type) per entry, applied with refusals.shifted_tone.
TONE_LIBRARY = (((5_000.0, 4.0, "high_shelf"), (300.0, -2.0, None)),
                ((150.0, 3.0, "low_shelf"), (3_500.0, -2.0, None)),
                ((2_500.0, 3.0, None), (120.0, -3.0, "low_shelf")))

# Construction (thresholds in frames of 10 ms unless named otherwise)
NOISE_S, SETUP_S, EVALUATION_S, ALIGN_S = 2.0, 10.0, 20.0, 5.0
SPEECH_WITHIN_DB, CLOSE_GAP, DROP_ISLAND = 20.0, 25, 5
NOISE_AFTER_SPEECH, NOISE_BEFORE_ONSET, NOISE_SKIP, NOISE_MIN_RUN, SPLICE_MIN_RUN = 50, 10, 20, 150, 40
SETUP_LEAD, SETUP_GAP, DEEP_PAUSE = 50, 2000, 60
FADE = int(0.02 * SAMPLE_RATE)
MARGIN = int(0.05 * SAMPLE_RATE)
MAX_LAG, MAX_DISAGREEMENT, MIN_CORRELATION, MAX_DRIFT_DB = int(0.02 * SAMPLE_RATE), 48, 0.5, 3.0
MAX_ATTEMPTS = 3
BAND = signal.butter(4, [300.0, 3_400.0], "bandpass", fs=SAMPLE_RATE, output="sos")

# Gates
WHOLE_VETOES = ("A", *refusals.WHOLE_ORDER)
STEP_OBJECTS = {"B5": "tone", "B6": "tuner", "P": "p_tone"}
WHOLE_BUDGET, STEP_BUDGET, COMBINED_BUDGET = 9, 18, 18
# cell name: (tone band frequency, injected gain). The 12 dB cells are gated, the 9 dB ones reported.
INJECTIONS = {f"{freq}Hz{gain:+d}dB": (float(freq), float(gain)) for size in (12, 9) for freq in (100, 200)
              for gain in (size, -size)}
MIN_PAIRS, MIN_REACHED, MIN_DEEP_USERS = 30, 0.80, 30
KEEP_DEEP_DB, DEEP_SHARE, SIBILANT_LOSS_DB = -4.0, 0.90, 3.0
FLAGS = ("fa", "fa_upper", "false_first", "false_first_upper", *(f"first:{v}" for v in WHOLE_VETOES),
         *(f"step:{v}" for v in STEP_OBJECTS))
CONTRASTS = {change: (("K1", change, "current"), ("K2", "P+T+R", "+".join(c for c in CHANGES if c != change) or "current"))
             for change in CHANGES}


# ----------------------------------------------------------------------------------------------
# Arms


def changes_of(arm: str) -> set[str]:
    return set() if arm == "current" else set(arm.split("+"))


def arm_of(changes: Container[str]) -> str:
    return "+".join(change for change in CHANGES if change in changes) or "current"


def analysis_key(arm: str) -> str:
    """Arms that differ only by T share one analysis: the tonal check acts in verification only."""
    return arm_of(changes_of(arm) - {"T"})


def existing_twin(arm: str) -> str:
    """In the existing-user condition the saved settings keep strength 1.0, so R changes nothing."""
    return arm_of(changes_of(arm) - {"R"})


def _switch_values() -> dict[str, Any]:
    return {change: getattr(import_module(module), name) for change, (module, name, _, _) in SWITCHES.items()}


@contextmanager
def arm_switches(changes: set[str]) -> Iterator[None]:
    """Set the three module globals for an arm and restore them, asserting that they are restored."""
    saved = _switch_values()
    try:
        for change, (module, name, on, off) in SWITCHES.items():
            setattr(import_module(module), name, on if change in changes else off)
        yield
    finally:
        for change, (module, name, _, _) in SWITCHES.items():
            setattr(import_module(module), name, saved[change])
        assert _switch_values() == saved, "an arm switch was not restored"


# ----------------------------------------------------------------------------------------------
# Construction of one recording: pure functions of the clean take, the device take and the clean take's labels


def read_daps(root: Path, speaker: str, version: str) -> np.ndarray:
    """A whole DAPS file (mono, 44.1 kHz) resampled once to 48 kHz."""
    from mic_eq.analysis.signal_processing import resample_poly

    rate, audio = wavfile.read(root / version / f"{speaker}_script1_{version}.wav")
    if rate != DAPS_RATE or audio.ndim != 1:
        raise ValueError(f"{speaker} {version}: expected mono {DAPS_RATE} Hz, got {rate} Hz")
    return resample_poly(audio.astype(np.float64) / float(np.iinfo(audio.dtype).max), 160, 147)


@functools.lru_cache(maxsize=4)
def read_take(root: str, speaker: str, version: str) -> np.ndarray:
    return read_daps(Path(root), speaker, version)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def speech_labels(clean: np.ndarray) -> np.ndarray:
    """10 ms frames within ``SPEECH_WITHIN_DB`` of the clean take's active level; gaps under 250 ms closed, islands under 50 ms dropped."""
    speech = gate_corpus._frame_rms_db(clean) >= gate_corpus._active_level_db(clean) - SPEECH_WITHIN_DB
    for start, stop in runs(~speech):
        if stop - start < CLOSE_GAP and start > 0 and stop < speech.size:
            speech[start:stop] = True
    for start, stop in runs(speech):
        if stop - start < DROP_ISLAND:
            speech[start:stop] = False
    return speech


def pause_runs(speech: np.ndarray) -> list[tuple[int, int]]:
    """Noise-only runs: 500 ms after speech, 100 ms before the next onset, ignoring the file's first 200 ms."""
    found = []
    for start, stop in runs(~speech):
        start = max(start + NOISE_AFTER_SPEECH if start else 0, NOISE_SKIP)
        stop = stop - NOISE_BEFORE_ONSET if stop < speech.size else stop
        if stop > start:
            found.append((start, stop))
    return found


def central(start: int, stop: int) -> tuple[int, int]:
    """The central 2.0 s of a run, or the whole run when it is shorter."""
    middle = start + max(0, stop - start - int(NOISE_S * 100)) // 2
    return middle, min(stop, middle + int(NOISE_S * 100))


def noise_candidates(speech: np.ndarray) -> list[dict[str, Any]]:
    """Room-noise captures in the order a retake takes them; a candidate is its pieces (frames) and where it ends.

    The leading silence if it reaches 1.5 s, then the other runs of 1.5 s or more, longest first (ties earliest).
    With no such run, one splice of the runs of 0.4 s or more in time order, ending as soon as 2.0 s are reached
    after the crossfades; nothing (the user is excluded) if they never reach it.
    """
    found = pause_runs(speech)
    lead = [r for r in found[:1] if r[0] == NOISE_SKIP]
    long = [r for r in found if r[1] - r[0] >= NOISE_MIN_RUN]
    order = [r for r in lead if r in long] + sorted((r for r in long if r not in lead), key=lambda r: (r[0] - r[1], r[0]))
    if order:
        return [{"pieces": [central(*r)], "end": central(*r)[1], "kind": "leading" if r in lead else "longest"} for r in order]
    chosen, total = [], 0
    for start, stop in (r for r in found if r[1] - r[0] >= SPLICE_MIN_RUN):
        chosen.append((start, stop))
        total += (stop - start) * FRAME - (FADE if len(chosen) > 1 else 0)
        if total >= int(NOISE_S * SAMPLE_RATE):
            return [{"pieces": chosen, "end": stop, "kind": "spliced"}]
    return []


def noise_audio(device: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    """The device take over a candidate's pieces; several are joined with 20 ms raised-cosine crossfades to 2.0 s."""
    audio = device[candidate["pieces"][0][0] * FRAME : candidate["pieces"][0][1] * FRAME]
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(FADE) + 0.5) / FADE)
    for start, stop in candidate["pieces"][1:]:
        piece = device[start * FRAME : stop * FRAME]
        audio = np.concatenate([audio[:-FADE], audio[-FADE:] * (1.0 - ramp) + piece[:FADE] * ramp, piece[FADE:]])
    return audio[: int(NOISE_S * SAMPLE_RATE)].astype(np.float32)


def setup_starts(speech: np.ndarray, noise_end: int) -> list[int]:
    """Setup windows in order: 10 s from 0.5 s before each speech onset after the noise segment."""
    onsets = np.flatnonzero(speech[1:] & ~speech[:-1]) + 1
    return [int(o) - SETUP_LEAD for o in onsets
            if o > noise_end and o >= SETUP_LEAD and o - SETUP_LEAD + int(SETUP_S * 100) <= speech.size]


def deep_pause_frames(speech: np.ndarray) -> int:
    """Pause frames 600 ms or more from speech on both sides, as ``gate_corpus.score`` reads them inside a window."""
    index = np.arange(speech.size)
    since = index - np.maximum.accumulate(np.where(speech, index, -10**6))
    until = np.minimum.accumulate(np.where(speech, index, 10**6)[::-1])[::-1] - index
    return int(np.count_nonzero(~speech & (since >= DEEP_PAUSE) & (until >= DEEP_PAUSE)))


def evaluation_starts(speech: np.ndarray, setup_end: int) -> list[int]:
    """20 s windows starting at a pause midpoint 20 s or more after the setup window: most deep-pause frames first."""
    size = int(EVALUATION_S * 100)
    found = [m for m in ((a + b) // 2 for a, b in runs(~speech)) if m >= setup_end + SETUP_GAP and m + size <= speech.size]
    return sorted(found, key=lambda m: (-deep_pause_frames(speech[m : m + size]), m))


def _lag(clean: np.ndarray, device: np.ndarray, start: int) -> int:
    size = int(ALIGN_S * SAMPLE_RATE)
    reference = signal.sosfiltfilt(BAND, clean[start : start + size])
    heard = signal.sosfiltfilt(BAND, device[start - MARGIN : start + size + MARGIN])
    return int(np.argmax(signal.correlate(heard, reference, mode="valid", method="fft"))) - MARGIN


def align(clean: np.ndarray, device: np.ndarray, start: int, size: int) -> dict[str, Any]:
    """The integer lag of the device take over a window (its first and last 5 s, band-passed 300-3400 Hz).

    Positive means the device is late: ``device[start + lag + n]`` matches ``clean[start + n]``. Both lags must be
    within 20 ms and agree within 48 samples.
    """
    if start < MARGIN or start + size + MARGIN > min(clean.size, device.size):
        return {"ok": False, "lags": None, "lag": 0}
    lags = [_lag(clean, device, start), _lag(clean, device, start + size - int(ALIGN_S * SAMPLE_RATE))]
    ok = max(abs(lag) for lag in lags) <= MAX_LAG and abs(lags[0] - lags[1]) <= MAX_DISAGREEMENT
    return {"ok": bool(ok), "lags": lags, "lag": int(round(sum(lags) / 2))}


def envelope_correlation(clean: np.ndarray, device: np.ndarray, speech: np.ndarray) -> float:
    """Pearson correlation of the 10 ms log-energy envelopes on speech frames."""
    levels = [gate_corpus._frame_rms_db(audio)[: speech.size][speech] for audio in (clean, device)]
    return float(np.corrcoef(levels[0], levels[1])[0, 1]) if speech.sum() > 2 else float("nan")


def gain_db(clean: np.ndarray, device: np.ndarray, speech: np.ndarray) -> float:
    """Level of the device take against the clean one on speech frames."""
    levels = [gate_corpus._frame_rms_db(audio)[: speech.size] for audio in (clean, device)]
    return gate_corpus._energy_mean_db(levels[1], speech) - gate_corpus._energy_mean_db(levels[0], speech)


def capture_vetoes(noise: np.ndarray, voice: np.ndarray) -> tuple[str | None, str | None]:
    """The dialog's checks on a room-noise take and a voice take: the code of each veto that fires, or None."""
    from mic_eq.analysis.noise_reference import analyze_noise_reference, capture_is_silent
    from mic_eq.analysis.voice_setup import raw_capture_is_clipped
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    def filtered(audio: np.ndarray) -> np.ndarray:
        return filtered_capture_for_analysis(audio, SAMPLE_RATE)

    noise_code = "noise_clipped" if raw_capture_is_clipped(noise, SAMPLE_RATE) else (
        "noise_recapture" if analyze_noise_reference(filtered(noise), None, SAMPLE_RATE).must_recapture else None)
    voice_code = "voice_clipped" if raw_capture_is_clipped(voice, SAMPLE_RATE) else (
        "voice_silent" if capture_is_silent(filtered(voice)) else None)
    return noise_code, voice_code


def construct(clean: np.ndarray, device: np.ndarray, speech: np.ndarray,
              check: Callable[[np.ndarray, np.ndarray], tuple[str | None, str | None]]) -> dict[str, Any]:
    """The captures of one recording, or the reason it is excluded (counted and listed, never replaced).

    Positions come from the clean take's labels only. ``check`` emulates the dialog's capture vetoes; a veto asks for
    the next noise segment (after a noise veto) or the next setup window, up to three takes. If the last take still
    fails the user is refused at capture (``A``) and the analysis is judged on the first take, as a counterfactual.
    The level gain is decided first, from the first voice take of the recording as it is, and the whole recording is
    scaled before any capture is cut or vetoed, as in the dialog, where the level check precedes the setup recording.
    """
    from mic_eq.ui.level_check import target_peak_db, verdict_from_audio

    candidates = noise_candidates(speech)
    options = [(i, start) for i, c in enumerate(candidates) for start in setup_starts(speech, c["end"])]
    if not options:
        return {"excluded": "noise_material" if not candidates else "no_setup_window"}
    setup_frames = int(SETUP_S * 100)
    first_voice = device[options[0][1] * FRAME : (options[0][1] + setup_frames) * FRAME].astype(np.float32)
    before = verdict_from_audio(first_voice, SAMPLE_RATE)
    level_gain_db = target_peak_db() - before.peak_db if before.peak_db is not None and before.code == "too_hot" else 0.0
    device = device * 10 ** (level_gain_db / 20)  # one gain for the whole recording, before any capture is cut or vetoed
    takes, position = [], 0
    while position is not None and len(takes) < MAX_ATTEMPTS:
        index, start = options[position]
        noise = noise_audio(device, candidates[index])
        voice = device[start * FRAME : (start + setup_frames) * FRAME].astype(np.float32)
        codes = check(noise, voice)
        takes.append({"noise": index, "setup": start, "vetoes": [c for c in codes if c], "captures": (noise, voice)})
        if not takes[-1]["vetoes"]:
            break
        position = next((p for p in range(position + 1, len(options)) if (options[p][0] != index) == bool(codes[0])), None)
    refused = bool(takes[-1]["vetoes"])
    used = takes[0] if refused else takes[-1]
    noise, voice = used["captures"]
    size, setup_end = int(EVALUATION_S * SAMPLE_RATE), used["setup"] + setup_frames
    tried = 0
    for first in evaluation_starts(speech, setup_end):
        tried += 1
        alignment = align(clean, device, first * FRAME, size)
        if not alignment["ok"]:
            continue
        window = slice(first * FRAME, first * FRAME + size)
        heard = device[window.start + alignment["lag"] : window.stop + alignment["lag"]]
        labels = speech[first : first + int(EVALUATION_S * 100)]
        correlation = envelope_correlation(clean[window], heard, labels)
        if not correlation >= MIN_CORRELATION:
            continue
        setup = slice(used["setup"] * FRAME, setup_end * FRAME)
        drift = gain_db(clean[setup], device[setup], speech[used["setup"] : setup_end]) - gain_db(clean[window], heard, labels)
        return {
            "excluded": None, "A": refused, "noise": noise, "setup": voice, "evaluation": heard.astype(np.float32),
            "clean": clean[window].astype(np.float64), "labels": np.repeat(labels, FRAME),
            "readings": {
                "noise_kind": candidates[used["noise"]]["kind"], "noise_spliced": candidates[used["noise"]]["kind"] == "spliced",
                "takes": [{key: take[key] for key in ("noise", "setup", "vetoes")} for take in takes], "A": refused,
                "lags": alignment["lags"], "lag": alignment["lag"], "envelope_correlation": correlation,
                "level": {"before": before.code, "gain_db": float(level_gain_db),
                          "after": verdict_from_audio(voice, SAMPLE_RATE).code},
                "gain_drift_db": float(drift), "drift_flag": bool(abs(drift) > MAX_DRIFT_DB), "windows_tried": tried,
                "deep_pause_frames": deep_pause_frames(labels), "speech_fraction": float(speech.mean()),
                "peaks": [float(np.max(np.abs(audio))) for audio in (noise, voice, heard)],
                "setup_start_s": used["setup"] / 100.0, "evaluation_start_s": first / 100.0}}
    return {"excluded": "no_aligned_evaluation_window" if tried else "no_evaluation_window"}


def tone_eq(k: int, version: str) -> dict[str, Any]:
    """The existing user's EQ: library entry (k + v) mod 3 on the tone bands nearest its frequencies."""
    from mic_eq.config import EQSettings

    eq = EQSettings().to_dict()
    for frequency, gain, kind in TONE_LIBRARY[(k + ALL_VERSIONS.index(version)) % len(TONE_LIBRARY)]:
        eq = refusals.shifted_tone(eq, frequency, gain, kind)
    return eq


def stacked(eq_settings: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    """A proposal's EQ as the dialog applies it: its arrays as the correction layer over the user's own EQ.

    ``loudness._applied_eq`` does the same over a flat EQ, which is wrong for a user with a tone EQ.
    """
    from mic_eq.config import EQSettings, build_eq_candidate_settings

    if "bands" in eq_settings:
        return dict(eq_settings)
    typed = build_eq_candidate_settings(
        EQSettings.from_dict(base) if base else EQSettings(), eq_settings["band_freqs"], eq_settings["band_gains"],
        eq_settings["band_qs"], layer="correction", enabled=bool(eq_settings.get("enabled", True))).to_dict()
    return {**eq_settings, **{key: typed[key] for key in ("schema_version", "enabled", "bands", "layers") if key in typed}}


def inject(eq: dict[str, Any], frequency_hz: float, gain_db: float) -> dict[str, Any]:
    """A wrong tone: the tone band nearest ``frequency_hz`` moved there at ``gain_db`` (the tone layer follows)."""
    wrong = refusals.shifted_tone(eq, frequency_hz, gain_db)
    if "layers" in wrong:
        wrong["layers"] = {**wrong["layers"], "tone": deepcopy(wrong["bands"])}
    return wrong


def check_corpus(root: Path, manifest: dict[str, Any], speaker: str, version: str) -> None:
    """The two files a recording is built from are the files the manifest names."""
    wanted = {f"{v}/{speaker}_script1_{v}.wav" for v in ("clean", version)}
    entries = [entry for entry in manifest["files"] if entry["path"] in wanted]
    differ = [e["path"] for e in entries if gate_corpus._sha256(root / e["path"]) != e["sha256"]]
    if len(entries) != len(wanted) or differ:
        raise RuntimeError(f"corpus files differ from the manifest: {', '.join(differ or sorted(wanted))}")


# ----------------------------------------------------------------------------------------------
# One user-condition: analyses, verification, end states and the objects they are judged on


@contextmanager
def capture_detector_off() -> Iterator[None]:
    """The counterfactual of a capture refusal: the analysis' own clip check does not fire (as the tap does for B1)."""
    from mic_eq.analysis import voice_setup

    saved = voice_setup.raw_capture_is_clipped
    voice_setup.raw_capture_is_clipped = lambda *args, **kwargs: False
    try:
        yield
    finally:
        voice_setup.raw_capture_is_clipped = saved


@contextmanager
def tuner_runtime() -> Iterator[None]:
    from mic_eq.analysis import joint_tuning

    saved = joint_tuning._MAX_RUNTIME_FACTOR
    if UNBOUNDED_TUNER_RUNTIME:
        joint_tuning._MAX_RUNTIME_FACTOR = math.inf
    try:
        yield
    finally:
        joint_tuning._MAX_RUNTIME_FACTOR = saved


@contextmanager
def fit_layout() -> Iterator[list[tuple[list[float], list[float]]]]:
    """The band centres and Qs of every fit the optimizer validates; ``refusals._taps`` logs only the gains."""
    from mic_eq.analysis.auto_eq_parts import optimizer

    seen: list[tuple[list[float], list[float]]] = []
    wrapped = optimizer._validate_and_attenuate_solution

    def watch(*args: Any, **kwargs: Any) -> Any:
        seen.append(([float(c) for c in args[5]], [float(q) for q in args[4]]))
        return wrapped(*args, **kwargs)

    optimizer._validate_and_attenuate_solution = watch
    try:
        yield seen
    finally:
        optimizer._validate_and_attenuate_solution = wrapped


def current_for(changes: set[str], env: dict[str, Any], vad: bool = True) -> dict[str, Any]:
    """The user's current settings under an arm's switches: ``Preset()`` is read now, so R's strength reaches it."""
    settings = refusals.current_settings(vad)
    assert settings["suppressor_settings"]["strength"] == (R_STRENGTH if "R" in changes else 1.0)
    if env["eq"]:
        settings["eq_settings"] = deepcopy(env["eq"])
    if env["existing"]:
        settings["suppressor_settings"]["strength"] = 1.0  # saved settings keep the strength they have
    return settings


def baseline(env: dict[str, Any], r_on: bool, vad: bool) -> dict[str, Any]:
    """The current settings, their own reference render and its readings, one per strength."""
    if r_on not in env["baselines"]:
        with arm_switches({"R"} if r_on else set()):
            current = current_for({"R"} if r_on else set(), env, vad)
        ctx = {**env["ctx"], "reference": None}
        metrics = refusals.measure_settings(current, ctx, ("whole", "last10"), as_reference=True)
        if "error" in metrics:  # every verdict is read against this render: a failure here is the environment's
            raise RuntimeError(f"the current settings do not render: {metrics['error']}")
        env["baselines"][r_on] = {"current": current, "ctx": ctx, "metrics": metrics, "memo": {}}
    return env["baselines"][r_on]


def analyse(key: str, env: dict[str, Any]) -> dict[str, Any]:
    """One analysis under an arm's switches, with the refusal study's taps; P's real flow replaces a tapped one."""
    from mic_eq.analysis.voice_setup import analyze_voice_setup

    changes, started = changes_of(key), time.perf_counter()
    with arm_switches(changes), tuner_runtime(), capture_detector_off() if env["A"] else nullcontext():
        current = current_for(changes, env)
        strength = current["suppressor_settings"]["strength"]

        def call() -> dict[str, Any]:
            return analyze_voice_setup(
                env["noise"], env["setup"], SAMPLE_RATE, "broadcast", noise_model=refusals.MODEL, suppressor_strength=strength,
                incumbent_settings={"gate": current["gate_settings"], "rnnoise": current["suppressor_settings"],
                                    "eq": current["eq_settings"]},
                raw_noise_audio=env["raw_noise"], raw_speech_audio=env["raw_setup"])

        try:
            with refusals._taps() as tap, fit_layout() as layout:
                result = call()
            b1 = next((entry for entry in tap["validation"] if not entry["passed"]), None)
            fit = None
            if tap["tone"] and layout:
                # The product's result may be the user's typed EQ (a withheld tone), so the fit's bands come from the optimizer.
                fit = {"band_freqs": layout[-1][0], "band_gains": tap["tone"][-1], "band_qs": layout[-1][1], "enabled": True}
            if "P" in changes and b1 and result["eq_settings"] is not None:
                # The tap hides P's tone-only path: the product's own flow is the untapped run.
                real = call()
                if real["eq_settings"] is not None and real["diagnostics"].get("tone_withheld"):
                    result, b1 = real, None
        except ValueError as error:
            if str(error) not in refusals.CAPTURE_REFUSALS:  # anything else is a fault, and stops the run
                raise
            return {"failure": str(error), "runtime_s": time.perf_counter() - started}
    if result["eq_settings"] is None:  # Auto-EQ raised before it could validate
        return {"failure": str(result["eq_error"]), "runtime_s": time.perf_counter() - started}
    loaded = refusals.check_loaded(result["diagnostics"], result["diagnostics"]["joint_tuning"])
    return {"result": result, "b1": b1, "fit": fit, "loaded": loaded, "runtime_s": time.perf_counter() - started}


def judged(memo: dict[str, Any], ctx: dict[str, Any], current: dict[str, Any], settings: dict[str, Any],
           sections: tuple[str, ...], windows: tuple[str, ...] = ("whole",)) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """An object of the study and its readings, as ``refusals.evaluate_user`` makes them; equal renders are shared."""
    entry = {"vacuous": refusals.is_vacuous(settings, current, sections)}
    if entry["vacuous"]:
        return entry, None
    key = f"{loudness._digest(settings)}{windows}"
    if key not in memo:
        memo[key] = refusals.measure_settings(settings, ctx, windows)
    metrics = dict(memo[key])
    if metrics.pop("identical", False):
        return {"vacuous": True, "identical_render": True}, None
    return entry, metrics


def arm_view(arm: str, product: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Everything the study reads of one arm of one user-condition (a view ``refusals.assess`` accepts)."""
    changes = changes_of(arm)
    started = time.perf_counter()
    r_on = "R" in changes and not env["existing"]
    view: dict[str, Any] = {"A": env["A"], "fired": [], "recommended": False, "short_speech": False, "tone_withheld": False,
                            "verification": None, "end_state": "current", "objects": {"E": {"vacuous": True}}, "metrics": {}}
    result = product.get("result")
    vad = bool(result["diagnostics"]["vad_available"]) if result else True
    base = baseline(env, r_on, vad)
    current, ctx, memo = base["current"], base["ctx"], base["memo"]
    view["metrics"] = {"current": base["metrics"], "E": base["metrics"]}
    metrics: dict[str, Any] = view["metrics"]
    if result is None:
        return view | {"analysis_failure": product["failure"], "runtime_s": {"verification": 0.0}}
    diagnostics = result["diagnostics"]
    tuning = diagnostics["joint_tuning"]
    state = {**deepcopy(result), "eq_settings": stacked(result["eq_settings"], env["eq"])}
    proposal = {name: state[name] for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}
    if loudness._digest(proposal["limiter_settings"]) != loudness._digest(current["limiter_settings"]):
        raise RuntimeError("the proposal's limiter differs from the factory limiter; the limiter would be a step")
    flow = None
    if product["b1"] is None and not env["A"]:  # a setup that B1 or the capture refusal ends is not verified
        with arm_switches(changes):
            flow = refusals.verify_flow(state, env["captures"])
    fired = refusals.fired_vetoes({
        "reasons": diagnostics["uncertainty_reasons"], "weak_capture": diagnostics["weak_capture"],
        "b1": product["b1"] is not None, "tuner_decision": tuning.get("decision"),
        "verification": None if flow is None else flow["decision"]})
    recommended = bool(diagnostics["apply_recommended"])
    objects: dict[str, Any] = view["objects"]

    def add(name: str, settings: dict[str, Any], sections: tuple[str, ...], windows: tuple[str, ...] = ("whole",)) -> None:
        objects[name], reading = judged(memo, ctx, current, settings, sections, windows)
        if reading is not None:
            metrics[name] = reading

    add("whole", proposal, SECTIONS)
    ended = flow is not None and flow["decision"] in ("rollback", "gave_up")
    if ended and flow is not None:
        add("final", {name: flow["state"][name] for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}, SECTIONS,
            ("whole", "last10"))
    if flow is not None and recommended and flow["decision"] == "accept":
        view["end_state"] = "candidate"
        add("E", {name: flow["state"][name] for name in SECTIONS} | {"diagnostics": {"vad_available": vad}}, SECTIONS)
    tone_withheld = bool(diagnostics.get("tone_withheld"))

    def tone_step(name: str) -> None:
        if product["fit"] is None:
            objects[name] = {"vacuous": False}
            return
        eq = stacked(product["fit"], env["eq"])
        add(name, refusals.hybrid(current, proposal | {"eq_settings": eq}, "tone"), refusals.STEPS["tone"])

    if "B5" in fired:
        tone_step("tone")
    if tone_withheld:
        tone_step("p_tone")
    candidates = list(tuning.get("candidates") or [])
    if "B6" in fired:
        key = voice_setup_eval._arm_key(refusals.MODEL, current["suppressor_settings"]["strength"], current["gate_settings"])
        best = refusals.tuner_withheld(candidates, key, tuning)
        if best is None:
            objects["tuner"] = {"vacuous": False}
        else:
            withheld = proposal | {"gate_settings": best["gate_settings"], "suppressor_settings": {
                "enabled": True, "strength": best["suppressor_strength"], "model": best["noise_model"], "auto_tuned": True}}
            add("tuner", refusals.hybrid(current, withheld, "gate_suppression"), refusals.STEPS["gate_suppression"])
    view |= {
        "fired": fired, "recommended": recommended, "tone_withheld": tone_withheld,
        "short_speech": float(diagnostics["vad_active_duration_s"]) < 2.0,
        "verification": None if flow is None else refusals._verification_row(flow),
        "tuner": {"decision": tuning.get("decision"), "holdout_evaluated": bool(tuning.get("holdout_evaluated")),
                  "candidates": [{"strength": float(c["suppressor_strength"]), "passes": bool(c["passes"]),
                                  "failed_gates": sorted(n for n, ok in c["gates"].items() if not ok),
                                  "runtime_factor": float(c.get("metrics", {}).get("runtime_factor", math.nan))}
                                 for c in candidates]},
        "runtime_s": {"verification": time.perf_counter() - started}, "b1": product["b1"] is not None}
    return view


def injected(key: str, product: dict[str, Any], env: dict[str, Any]) -> dict[str, Any] | None:
    """Wrong tones injected into the candidate entering verification, with the tonal check off and on (BEN-T)."""
    from mic_eq.analysis.voice_setup import validate_voice_setup_verification

    result = product.get("result")
    if result is None or product["b1"] is not None or env["A"]:
        return None
    state = {**deepcopy(result), "eq_settings": stacked(result["eq_settings"], env["eq"])}
    noise, setup, (take, raw_take) = env["captures"]["noise"], env["captures"]["setup"], env["captures"]["takes"][0]
    cells: dict[str, Any] = {}
    for cell, (frequency, gain) in INJECTIONS.items():
        wrong = {**state, "eq_settings": inject(state["eq_settings"], frequency, gain)}
        wrong["verification_chain_settings"] = loudness._chain(wrong)
        cells[cell] = {}
        for tag, changes in (("off", changes_of(key)), ("on", changes_of(key) | {"T"})):
            with arm_switches(changes):
                outcome = validate_voice_setup_verification(
                    noise, setup, take, SAMPLE_RATE, wrong, "broadcast", raw_noise_audio=env["captures"]["raw_noise"],
                    raw_verification_speech_audio=raw_take)
            cells[cell][tag] = {"ran": outcome["decision"] != "retry",
                                "target_error": "target_error" in (outcome.get("reason_codes") or ())}
    return cells


def prepare(job: tuple[str, str, int, str], *, development: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The row of a user-condition up to its construction, and what the analyses need (None when it is excluded)."""
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    root_text, condition, k, version = job
    root, speaker = Path(root_text), SPEAKERS[k]
    check_corpus(root, json.loads((root / "MANIFEST.json").read_text(encoding="utf-8")), speaker, version)
    clean, device = read_take(root_text, speaker, "clean"), read_take(root_text, speaker, version)
    speech = speech_labels(clean)
    built = construct(clean, device, speech, capture_vetoes)
    device_name, _, room = version.partition("_")
    row: dict[str, Any] = {
        "split": condition, "index": k, "model": version, "development": development, "build": build_id(root_text),
        "speaker": speaker, "version": version, "condition": condition, "device": device_name, "room": room,
        "gender": "female" if speaker.startswith("f") else "male", "excluded": built["excluded"], "arms": {}, "injections": {}}
    if built["excluded"]:
        return row | {"construction": {"speech_fraction": float(speech.mean())}}, None
    filtered = functools.partial(filtered_capture_for_analysis, sample_rate=SAMPLE_RATE)
    raw_noise, raw_setup, raw_evaluation = built["noise"], built["setup"], built["evaluation"]
    existing = condition == "existing"
    env: dict[str, Any] = {
        "existing": existing, "eq": tone_eq(k, version) if existing else None, "A": built["A"], "baselines": {},
        "noise": filtered(raw_noise), "setup": filtered(raw_setup), "raw_noise": raw_noise, "raw_setup": raw_setup,
        "ctx": {"raw": raw_evaluation, "labels": built["labels"], "clean": filtered(built["clean"]).astype(np.float64),
                "noisy": filtered(raw_evaluation).astype(np.float64), "noise": filtered(raw_noise).astype(np.float64)}}
    env["captures"] = {"noise": env["noise"], "setup": env["setup"], "raw_noise": raw_noise,
                       "takes": [(filtered(window), window) for window in refusals.take_windows(raw_evaluation, raw_setup.size)]}
    row |= {"construction": built["readings"], "level_check": built["readings"]["level"]["before"],
            "tone_entry": (k + ALL_VERSIONS.index(version)) % len(TONE_LIBRARY) if existing else None}
    return row, env


def evaluate_user(job: tuple[str, str, int, str], *, development: bool, map_r: bool = True) -> dict[str, Any]:
    """One (speaker, version, condition): construction, every arm, the injections of BEN-T. ``map_r`` is off only to
    check that R changes nothing for an existing user."""
    refusals.configure_environment()
    started = time.perf_counter()
    row, env = prepare(job, development=development)
    if env is None:
        return row
    products: dict[str, dict[str, Any]] = {}
    views: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if env["existing"] and map_r and existing_twin(arm) != arm:
            continue
        key = analysis_key(arm)
        if key not in products:
            products[key] = analyse(key, env)
        views[arm] = arm_view(arm, products[key], env)
    row["arms"] = {arm: views[arm] if arm in views else views[existing_twin(arm)] for arm in ARMS}
    row["loaded"] = next((p["loaded"] for p in products.values() if "loaded" in p), None)
    if job[1] == "factory":
        row["injections"] = {key: injected(key, products[key], env) for key in ("current", "P+R")}
    if development:
        current = baseline(env, False, True)
        row["probes"] = {name: refusals.measure_settings(settings, current["ctx"])
                         for name, (_, settings) in refusals.probe_settings(current["current"]).items()}
    row["runtime"] = {"analysis": {key: p["runtime_s"] for key, p in products.items()},
                      "verification": {arm: v["runtime_s"]["verification"] for arm, v in views.items()},
                      "user": time.perf_counter() - started}
    return row


# ----------------------------------------------------------------------------------------------
# Counting, gates and the decision


def main_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The analysed user-conditions of the two gated conditions (N = 180 when none is excluded)."""
    found = [r for r in rows if not r.get("development") and not r["excluded"] and r["condition"] in ("factory", "existing")]
    return sorted(found, key=lambda r: (r["condition"], r["index"], r["version"]))


def arm_user(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """A user-condition as the refusal study's counting reads a row (unique per arm)."""
    return {"split": f"{row['condition']}:{row['version']}", "index": row["index"], "speaker": row["speaker"],
            "model": refusals.MODEL, "development": row.get("development", False), **row["arms"][arm]}


def first_veto(view: dict[str, Any]) -> str | None:
    return None if view.get("analysis_failure") else (
        "A" if view["A"] else refusals.first_whole(set(view["fired"]), view["recommended"]))


def arm_flags(view: dict[str, Any]) -> dict[str, int]:
    """One user-condition of one arm: its false refusals (headline and upper readings), false accept and step vetoes."""
    flags = dict.fromkeys(FLAGS, 0)
    first = first_veto(view)
    if first:
        status = refusals.outcome(view, "whole" if first == "A" else refusals.OBJECT[first], veto=first)["status"]
        flags[f"first:{first}"] = flags["false_first"] = int(refusals.false_refusal(status, "lower"))
        flags["false_first_upper"] = int(refusals.false_refusal(status, "upper"))
    for veto, name in STEP_OBJECTS.items():
        if (view["tone_withheld"] if veto == "P" else veto in view["fired"]) and not view.get("analysis_failure"):
            flags[f"step:{veto}"] = int(refusals.outcome(view, name)["status"] == "false")
    status = refusals.outcome(view, "E")["status"]
    flags["fa"], flags["fa_upper"] = int(status == "correct"), int(status in ("correct", "partial"))
    return flags


def _reading(view: dict[str, Any], name: str) -> float:
    value = view["metrics"]["E"].get("whole", {}).get(name)  # a render that failed has no readings
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.nan


def gate_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything the gates read, as arrays over the analysed user-conditions."""
    users = main_rows(rows)
    views = {arm: [arm_user(row, arm) for row in users] for arm in ARMS}
    per_arm = {arm: [arm_flags(view) for view in views[arm]] for arm in ARMS}
    factory = [row for row in users if row["condition"] == "factory" and row["injections"]]
    tallies: dict[str, Any] = {}
    for key in ("current", "P+R"):
        done = [row["injections"][key] for row in factory if row["injections"].get(key)]
        tallies[key] = {cell: {tag: [sum(i[cell][tag]["ran"] for i in done),
                                     sum(i[cell][tag]["ran"] and i[cell][tag]["target_error"] for i in done)]
                               for tag in ("off", "on")} for cell in INJECTIONS}
    return {
        "speakers": np.array([row["speaker"] for row in users]),
        "factory": np.array([row["condition"] == "factory" for row in users], bool),
        "flags": {arm: {name: np.array([f[name] for f in per_arm[arm]], int) for name in FLAGS} for arm in ARMS},
        "deep": {arm: np.array([_reading(view, "deep_pause_db") for view in views[arm]]) for arm in ARMS},
        "sibilant": {arm: np.array([_reading(view, "sibilant_loss_db") for view in views[arm]]) for arm in ARMS},
        "injections": tallies}


NOT_CONFIRMED = "within budget on this cohort, not confirmed at 95 %"
LABELS = {"failed": "failed", "confirmed": "confirmed at 95 %", "not_confirmed": NOT_CONFIRMED, "passed": "passed"}


def count_gate(name: str, on: Any, off: Any, speakers: np.ndarray, *, budget: int | None = None,
               strict: bool = False) -> dict[str, Any]:
    """A count gate on observed counts: the equality passes. With a budget ``on <= max(budget, off)``; without,
    ``on <= off`` (``on < off`` when strict). The speaker-clustered interval of the same count is reported, never
    decides; a pass whose interval reaches past the limit is labelled so."""
    on, off = np.asarray(on, float), np.asarray(off, float)
    limit = max(budget, int(off.sum())) if budget is not None else int(off.sum()) - int(strict)
    deltas = on if budget is not None else on - off
    shift = 0 if budget is not None else int(off.sum())
    interval = [float(x) * on.size for x in gate_corpus.cluster_ci(deltas, speakers)]
    passed = int(on.sum()) <= limit
    label = LABELS["failed" if not passed else "confirmed" if interval[2] <= limit - shift else "not_confirmed"]
    return {"gate": name, "on": int(on.sum()), "off": int(off.sum()), "limit": limit, "pass": passed,
            "interval": interval, "label": label}


def budget_gates(on: dict[str, np.ndarray], off: dict[str, np.ndarray], speakers: np.ndarray) -> list[dict[str, Any]]:
    """BUD and S2/S3: each whole-setup veto 5 % (9), each step veto 10 % (18), all whole-setup vetoes together 10 % (18)."""
    return [count_gate(f"{veto}", on[f"first:{veto}"], off[f"first:{veto}"], speakers, budget=WHOLE_BUDGET) for veto in WHOLE_VETOES] + [
        count_gate(f"step:{veto}", on[f"step:{veto}"], off[f"step:{veto}"], speakers, budget=STEP_BUDGET) for veto in STEP_OBJECTS] + [
        count_gate("combined", on["false_first"], off["false_first"], speakers, budget=COMBINED_BUDGET)]


def deep_gate(inputs: dict[str, Any], on_arm: str, off_arm: str) -> dict[str, Any]:
    """BEN-R1: among factory users with a measurable deep pause (30 needed), the share whose end state keeps a deep
    pause at -4.0 dB or lower must be at least min(0.90, the share of the arm without the change)."""
    mask = inputs["factory"]
    counts = {}
    for tag, arm in (("on", on_arm), ("off", off_arm)):
        deep = inputs["deep"][arm][mask]
        counts[tag] = (int(np.isfinite(deep).sum()), int(np.nansum(deep <= KEEP_DEEP_DB)))
    shares = {tag: (kept / users if users else 0.0) for tag, (users, kept) in counts.items()}
    need = min(DEEP_SHARE, shares["off"])
    passed = min(users for users, _ in counts.values()) >= MIN_DEEP_USERS and shares["on"] >= need - 1e-12
    return {"gate": "BEN-R1", "users": [counts["on"][0], counts["off"][0]], "share": [shares["on"], shares["off"]],
            "limit": need, "pass": bool(passed), "label": "passed" if passed else "failed"}


def tone_gates(inputs: dict[str, Any], on_arm: str) -> list[dict[str, Any]]:
    """BEN-T: at each 12 dB cell, 30 pairs where the tonal check ran and 80 % of them reported as target_error."""
    found = []
    for cell, (_, gain) in INJECTIONS.items():
        if abs(gain) == 12.0:
            ran, hit = inputs["injections"][analysis_key(on_arm)][cell]["on"]
            passed = ran >= MIN_PAIRS and hit >= MIN_REACHED * ran - 1e-9
            found.append({"gate": f"BEN-T:{cell}", "pairs": ran, "reported": hit, "limit": MIN_REACHED, "pass": bool(passed),
                          "label": "passed" if passed else "failed"})
    return found


def change_gates(inputs: dict[str, Any], change: str) -> list[dict[str, Any]]:
    """The gates of one change on its two contrasts: K1 (the change alone against the current build) and K2 (all
    three against all but that change)."""
    gates = []
    for contrast, on_arm, off_arm in CONTRASTS[change]:
        on, off, speakers = inputs["flags"][on_arm], inputs["flags"][off_arm], inputs["speakers"]
        found = budget_gates(on, off, speakers)
        if change in "PR":
            found.append(count_gate("FA", on["fa"], off["fa"], speakers))
        if change == "T":  # asserted, not gated: the tonal check only adds rollbacks
            refusals._check(int(on["fa"].sum()) <= int(off["fa"].sum()), "T added a false accept; it only adds rollbacks")
            found += tone_gates(inputs, on_arm)
        if change == "P":
            found.append(count_gate("BEN-P", on["false_first"], off["false_first"], speakers, strict=True))
        if change == "R":
            mask = inputs["factory"]
            sibilant = [np.nan_to_num(inputs["sibilant"][arm][mask], nan=0.0) > SIBILANT_LOSS_DB for arm in (on_arm, off_arm)]
            found += [deep_gate(inputs, on_arm, off_arm), count_gate("BEN-R2", sibilant[0], sibilant[1], speakers[mask], strict=True)]
        gates += [{"contrast": contrast, "on_arm": on_arm, "off_arm": off_arm, **gate} for gate in found]
    return gates


def shipping_gates(inputs: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """S1 false accepts no more than the current build's; S2 each veto within max(its budget, the current's); S3 the
    whole-setup vetoes together within max(18, the current's)."""
    on, off, speakers = inputs["flags"][arm], inputs["flags"]["current"], inputs["speakers"]
    return [count_gate("S1", on["fa"], off["fa"], speakers)] + [
        {**gate, "gate": "S3" if gate["gate"] == "combined" else f"S2:{gate['gate']}"} for gate in budget_gates(on, off, speakers)]


def candidate_of(failed: list[str]) -> str | None:
    """No failed change: all three. Exactly one: the pair without it. Two or more: none."""
    return "P+T+R" if not failed else arm_of(set(CHANGES) - set(failed)) if len(failed) == 1 else None


def decide(inputs: dict[str, Any]) -> dict[str, Any]:
    gates = {change: change_gates(inputs, change) for change in CHANGES}
    failed = [change for change in CHANGES if not all(g["pass"] for g in gates[change])]
    candidate = candidate_of(failed)
    shipping = shipping_gates(inputs, candidate) if candidate else []
    return {"failed_changes": failed, "candidate": candidate, "ships": bool(candidate) and all(g["pass"] for g in shipping),
            "change_gates": gates, "shipping_gates": shipping}


# ----------------------------------------------------------------------------------------------
# The jobs and the report

STATEMENTS = (
    "Loudspeaker playback of read speech, one script, not a live talker: no raised voice, movement or close-microphone "
    "bass, the same delivery in every room, and room tone that includes the loudspeaker's hiss.",
    "The reference is the dry studio take. Device and room are confounded: ipadflat appears only in office1 and "
    "confroom1, iphone only in livingroom1 and bedroom1. Speech labels come from the clean take's energy.",
    "Users whose noise had to be spliced from several pauses are flagged and reported apart; users without 2.0 s of "
    "pause material, or without an aligned evaluation window, are excluded, counted and listed, never replaced.",
    "{users} analysed user-conditions in {clusters} speaker clusters, so the budgets (9, 18 and 18, fixed whatever the number "
    "of exclusions) are not confirmed at 95 %: a pass whose interval reaches past its limit is labelled 'within budget on "
    "this cohort, not confirmed at 95 %', and a miss by one is a miss.",
    "The current settings are the factory preset, or the factory preset plus one of three hand-set tone EQ library "
    "entries. The evidence for R comes from factory users only: an existing user's saved strength does not change.",
    "The tone clause is nearly blind below 1 kHz, so T's benefit rests on injected errors at 100 and 200 Hz, not on "
    "errors the recordings contain.",
    "For a recording refused at capture (A) and for an unclear recording (B1) the judged object is the proposal had "
    "the refusal not happened, a counterfactual the product discards.",
    "df.dll is a local build whose SHA-256 differs from release-assets.json; both hashes are recorded.",
    "RNNoise default strength 0.7 and the by-band tonal check were rejected before (evaluation/README.md rows 'Voice "
    "Setup verification' and 'Voice Setup verification, tonal check by band'); DAPS is the new evidence.",
    "The refusal study found that the compressor step is the only worse step for most correctly refused setups; that "
    "bounds what any change here can show.",
)
SENSITIVITY = {"sibilant_1_5_db": {"sibilant_tolerance": 1.5}, "sibilant_6_0_db": {"sibilant_tolerance": 6.0},
               "strict_overload": {"strict": True}}


def user_keys(development: bool) -> list[tuple[str, int, str]]:
    """(condition, speaker index, version): factory users first, then existing, balcony last."""
    if development:
        return [(c, k, v) for c in ("factory", "existing") for k in (18, 19) for v in ALL_VERSIONS]
    keys = [(c, k, VERSIONS[(k + j) % len(VERSIONS)]) for c in ("factory", "existing") for k in range(len(HELD_OUT))
            for j in range(VERSIONS_PER_SPEAKER)]
    return keys + [("balcony", k, BALCONY[k % 2]) for k in range(len(HELD_OUT))]


def jobs_of(root: Path, development: bool) -> list[tuple[str, str, int, str]]:
    return [(str(root), condition, k, version) for condition, k, version in user_keys(development)]


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def _share(bounds: list[float]) -> str:
    return f"{bounds[0]:.3f} [{bounds[1]:.3f}, {bounds[2]:.3f}]"


def _gate_view(gate: dict[str, Any]) -> dict[str, Any]:
    """A gate for the report: the interval as one string and the label as a key of ``LABELS``."""
    shown = {k: v for k, v in gate.items() if k not in ("contrast", "on_arm", "off_arm", "interval", "label")}
    if "interval" in gate:
        shown["interval"] = _share(gate["interval"])
    shown["label"] = next(key for key, text in LABELS.items() if text == gate["label"])
    return shown


def decision_report(decision: dict[str, Any]) -> dict[str, Any]:
    """The decision with its gates grouped by contrast; each gate is judged on its observed count."""
    grouped: dict[str, Any] = {}
    for change, gates in decision["change_gates"].items():
        contrasts: dict[str, Any] = {}
        for gate in gates:
            entry = contrasts.setdefault(gate["contrast"], {"on_arm": gate["on_arm"], "off_arm": gate["off_arm"], "gates": []})
            entry["gates"].append(_gate_view(gate))
        grouped[change] = {"passes": all(g["pass"] for g in gates), "contrasts": contrasts}
    return {**{key: decision[key] for key in ("failed_changes", "candidate", "ships")}, "labels": LABELS, "change_gates": grouped,
            "shipping_gates": [_gate_view(g) for g in decision["shipping_gates"]]}


def arm_users(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    """The refusal study's users of one arm; a capture refusal (A) is the first veto and its own bucket."""
    users = refusals.assess([arm_user(row, arm) for row in rows])
    for user in users:
        if user["bucket"] != "analysis_failure" and user["row"]["A"]:
            user["first"] = user["bucket"] = "A"
    return users


def arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    users = arm_users(rows, arm)
    flags = [arm_flags(user["row"]) for user in users]
    table = [refusals._table_row(users, veto, "whole", "first") for veto in refusals.WHOLE_ORDER]
    table += [refusals._table_row(users, veto, "step", "any") for veto in refusals.STEP_OBJECT]
    estoi = [(user["row"]["metrics"]["E"]["whole"]["estoi"], user["row"]["metrics"]["current"]["whole"]["estoi"])
            for user in users if "estoi" in user["row"]["metrics"]["E"].get("whole", {})]
    return {
        "buckets": refusals.population(users), "false_refusals": sum(f["false_first"] for f in flags),
        "false_refusals_upper": sum(f["false_first_upper"] for f in flags), "false_accepts": sum(f["fa"] for f in flags),
        "false_accepts_upper": sum(f["fa_upper"] for f in flags), "step_false": {v: sum(f[f"step:{v}"] for f in flags) for v in STEP_OBJECTS},
        "end_candidate": sum(user["row"]["end_state"] == "candidate" for user in users),
        "table": [{"veto": line["veto"], "track": line["track"], "fired": line["fired"],
                   **{status: line[status] for status in refusals.STATUSES if line[status]},
                   "false_share": _share(line["false_share_lower_bound"]), "false_share_upper": _share(line["false_share_upper_bound"]),
                   **({"correct_by_clause": line["correct_by_clause"]} if line["correct_by_clause"] else {})} for line in table],
        "clause_coverage": refusals.clause_coverage(users),
        "sensitivity": {**{name: refusals._false_counts(users, **reading) for name, reading in SENSITIVITY.items()},
                        "last_10_s_for_c1_c2": refusals._false_counts(users, ("C1", "C2"), window="last10")},
        "estoi_mean_end_state_and_current": [float(np.mean(column)) for column in zip(*estoi)] if estoi else None}


def strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per stratum: the user-conditions, and per arm [false refusals, false accepts]."""
    keys = {"device": lambda r: r["device"], "room": lambda r: r["room"], "condition": lambda r: r["condition"],
            "gender": lambda r: r["gender"], "spliced_noise": lambda r: r["construction"]["noise_spliced"],
            "level_drift": lambda r: r["construction"]["drift_flag"]}
    found: dict[str, Any] = {}
    for name, key in keys.items():
        groups: dict[str, Any] = {}
        for row in rows:
            group = groups.setdefault(str(key(row)), {"users": 0, **{arm: [0, 0] for arm in ARMS}})
            group["users"] += 1
            for arm in ARMS:
                flags = arm_flags(arm_user(row, arm))
                group[arm][0] += flags["false_first"]
                group[arm][1] += flags["fa"]
        found[name] = dict(sorted(groups.items()))
    return found


def check_rows(rows: list[dict[str, Any]]) -> None:
    refusals._check(not any(row.get("development") for row in rows), "development rows are spent users and give no rates")
    missing = set(user_keys(False)[: 2 * len(HELD_OUT) * VERSIONS_PER_SPEAKER]) - {
        (r["condition"], r["index"], r["version"]) for r in rows}
    refusals._check(not missing, f"main jobs are missing: {sorted(missing)[:3]}")


def arms_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    check_rows(rows)
    analysed = main_rows(rows)
    inputs = gate_inputs(rows)
    balcony = [r for r in rows if r["condition"] == "balcony" and not r["excluded"]]
    taken = Counter(code for row in analysed for take in row["construction"]["takes"] for code in take["vetoes"])
    return {
        "users": {"analysed": len(analysed), "speakers": len({r["speaker"] for r in analysed}),
                  "excluded": [{"condition": r["condition"], "speaker": r["speaker"], "version": r["version"], "why": r["excluded"]}
                               for r in rows if r["excluded"]],
                  "noise_spliced": sum(r["construction"]["noise_spliced"] for r in analysed),
                  "gain_drift_over_3_db": sum(r["construction"]["drift_flag"] for r in analysed),
                  "capture_refused_A": sum(r["construction"]["A"] for r in analysed)},
        "decision": decision_report(decide(inputs)), "arms": {arm: arm_summary(analysed, arm) for arm in ARMS},
        "injections": inputs["injections"], "strata": strata(analysed),
        "balcony": {"users": len(balcony), "arms": {arm: {key: sum(arm_flags(arm_user(r, arm))[key] for r in balcony)
                                                         for key in ("false_first", "fa")} for arm in ARMS}},
        "retake_costs": {"capture_vetoes": dict(taken), "users_with_a_retake": sum(len(r["construction"]["takes"]) > 1 for r in analysed),
                         "verification_retakes": dict(Counter(code for r in analysed for v in r["arms"].values()
                                                            for code in ((v["verification"] or {}).get("retakes") or ())))},
        "reading": "false refusal and false accept counts are the headline readings; the _upper counts read an "
                   "unmeasurable clause the other way",
    }


def build_report(source: dict[str, Any], frozen: dict[str, str]) -> dict[str, Any]:
    """The tracked report. Its hashes are the frozen ones, read from the freeze and not computed again."""
    release = {name: {"local": frozen[name], "declared": frozen[f"release_assets:{name}"],
                      "same": frozen[name] == frozen[f"release_assets:{name}"]} for name in refusals.RELEASE_ASSET_PATHS}
    metrics = arms_report(source["rows"])
    counts = {"users": metrics["users"]["analysed"], "clusters": metrics["users"]["speakers"]}
    return _round({
        "schema_version": 2, "audible_change": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Voice Setup changes P, T and R compared as arms on DAPS script 1 recorded on consumer devices",
        "predefined_rule": "The rule in the docstring of python/tools/evaluate_voice_setup_arms.py.",
        "rule_sha256": frozen["rule"], "build": source["identity"]["build"], "metrics": metrics,
        "limitations": [text.format(**counts) for text in STATEMENTS], "source_sha256": {name: frozen[name] for name in SOURCES},
        "implementation_sha256": {loudness.EXTENSION: frozen["extension"]},
        "asset_hashes": {"daps_manifest": frozen["daps_manifest"], "vad_model": frozen["vad_model"], "df_dll": frozen["df.dll"],
                         "onnxruntime_dll": frozen["onnxruntime.dll"],
                         "onnxruntime_providers_shared_dll": frozen["onnxruntime_providers_shared.dll"],
                         **{name: digest for name, digest in frozen.items() if name.startswith("models/")}},
        "release_assets": release, "frozen_sha256": frozen})


# ----------------------------------------------------------------------------------------------
# Hand-made worlds, for the dry run and the tests


def fabricated_inputs() -> dict[str, Any]:
    """A world of 180 user-conditions in 18 speakers, 90 of them factory, in which every change shows its benefit and
    nothing else happens: an arm without P has one false refusal (B4), an arm without R one user who loses
    sibilants; a gate test adds counts to the flags (the first ``n`` users of an arm)."""
    n = 180
    cells = {tag: [60, 60] if tag == "on" else [60, 0] for tag in ("off", "on")}
    inputs: dict[str, Any] = {
        "speakers": np.array([f"s{u % 18}" for u in range(n)]), "factory": np.arange(n) < 90,
        "flags": {arm: {name: np.zeros(n, int) for name in FLAGS} for arm in ARMS},
        "deep": {arm: np.full(n, -10.0) for arm in ARMS}, "sibilant": {arm: np.zeros(n) for arm in ARMS},
        "injections": {key: {cell: deepcopy(cells) for cell in INJECTIONS} for key in ("current", "P+R")}}
    for arm in ARMS:
        if "P" not in changes_of(arm):
            inputs["flags"][arm]["first:B4"][0] = inputs["flags"][arm]["false_first"][0] = 1
        if "R" not in changes_of(arm):
            inputs["sibilant"][arm][0] = SIBILANT_LOSS_DB + 1.0
    return inputs


def fabricated_rows() -> list[dict[str, Any]]:
    """The 180 main user-conditions and the balcony ones with hand-made readings: a false refusal by B4 in every arm for
    one user in 40, a withheld tone in the P arms (a B5 refusal in the others) for one in 40, a false accept in R."""
    ok = {"estoi": 0.5, "tone_error_db": 2.0, "sibilant_loss_db": 1.0, "deep_pause_db": -10.0, "tail_db": -8.0,
          "onset_db": -1.0, "pause_db": -9.0, "clipped": False, "limiter_gain_reduction_db": 0.5,
          "true_peak_limiter_gain_reduction_db": 0.1}
    accept = {"decision": "accept", "first_decision": "accept", "adjustments": 0, "ended": None, "reason_codes": ["accepted"],
              "takes": 1, "retakes": []}

    def view(i: int, arm: str) -> dict[str, Any]:
        refused, withheld = i % 40 == 3, i % 40 == 5
        fired = ["B4"] if refused else ["B5"] if withheld and "P" not in arm else []
        worse = i % 40 == 7 and "R" in arm
        metrics = {"current": {"whole": ok, "last10": ok}, "whole": {"whole": ok}, "E": {"whole": ok}, "tone": {"whole": ok},
                   "p_tone": {"whole": ok}}
        if worse:
            metrics["E"] = {"whole": {**ok, "tone_error_db": 4.0}}
        accepted = not fired
        return {"A": False, "fired": fired, "recommended": not fired, "short_speech": False,
                "tone_withheld": withheld and "P" in arm, "b1": False, "verification": dict(accept),
                "end_state": "candidate" if accepted else "current", "metrics": metrics,
                "objects": {"whole": {"vacuous": False}, "tone": {"vacuous": False}, "p_tone": {"vacuous": False},
                            "E": {"vacuous": not accepted}}, "runtime_s": {"verification": 0.0}}

    rows = []
    for i, (condition, k, version) in enumerate(user_keys(False)):
        device, _, room = version.partition("_")
        takes = [{"noise": 0, "setup": 0, "vetoes": []}]
        row = {"split": condition, "index": k, "model": version, "development": False, "speaker": HELD_OUT[k],
               "version": version, "condition": condition, "device": device, "room": room, "excluded": None,
               "gender": "female" if HELD_OUT[k].startswith("f") else "male", "level_check": "good",
               "construction": {"noise_spliced": i % 50 == 0, "drift_flag": i % 70 == 0, "A": False, "takes": takes},
               "arms": {arm: view(i, arm) for arm in ARMS}, "injections": {}}
        if condition == "factory":
            cell = {tag: {"ran": True, "target_error": tag == "on"} for tag in ("off", "on")}
            row["injections"] = {key: {name: cell for name in INJECTIONS} for key in ("current", "P+R")}
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# Freezing: the rule, the code and the assets


def asset_paths(root: Path = DAPS) -> dict[str, Path]:
    """Every binary, model and corpus manifest the study loads, by name; one that is absent stops the freeze."""
    paths = refusals.asset_paths(root)
    del paths["corpus_manifest"]
    return {**paths, "daps_manifest": root / "MANIFEST.json"}


def asset_hashes(root: Path = DAPS) -> dict[str, str]:
    return {name: gate_corpus._sha256(path) for name, path in asset_paths(root).items()}


def source_hashes() -> dict[str, str]:
    """Every product module the study loads, and the evaluators and tools it imports."""
    product = {path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "python" / "mic_eq").rglob("*.py")}
    return {name: loudness._sha256(REPO_ROOT / name) for name in sorted({*SOURCES, *product})}


def frozen_hashes(root: Path = DAPS) -> dict[str, str]:
    """What must not change between the freeze and the report: the rule, the code, the assets and the libraries."""
    import scipy

    return {"rule": hashlib.sha256((__doc__ or "").encode()).hexdigest(), **source_hashes(), **asset_hashes(root),
            **refusals.declared_hashes(), "runtime:python": sys.version, "runtime:numpy": np.__version__,
            "runtime:scipy": scipy.__version__}


@functools.lru_cache(maxsize=4)
def build_id(root: str) -> str:
    """Derived from every frozen hash, so a partial file written by other code is not resumed."""
    return refusals.build_of(frozen_hashes(Path(root)))


def freeze(frozen: Path, rules: Path, rows: Path, root: Path) -> None:
    """Write the rule text and the hashes. Once rows exist the freeze they were made under stays."""
    refusals.check_assets()
    if frozen.exists() and (rows.exists() or rows.with_suffix(".run.partial.jsonl").exists()):
        raise SystemExit(f"{rows} exists: the freeze it was made under stays")
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text(__doc__ or "", encoding="utf-8", newline="\n")
    refusals._write(frozen, frozen_hashes(root))


def check_frozen(path: Path, root: Path = DAPS) -> dict[str, str]:
    """The frozen hashes, once the code and the assets are shown to be what they were frozen as."""
    if not path.exists():
        raise SystemExit(f"{path} is missing; run probe --development, then freeze, before the study")
    frozen, now = json.loads(path.read_text(encoding="utf-8")), frozen_hashes(root)
    changed = sorted(name for name in {*frozen, *now} if frozen.get(name) != now.get(name))
    if changed:
        raise SystemExit(f"changed since the freeze: {', '.join(changed)}")
    return frozen


def check_unchanged(root: Path, started: str, rows: list[dict[str, Any]]) -> None:
    """A run is one build: nothing it loads changed between its start and its end."""
    if refusals.build_of(frozen_hashes(root)) != started or any(row["build"] != started for row in rows):
        raise SystemExit("the code or an asset changed during the run; its rows are not one build")


# ----------------------------------------------------------------------------------------------
# The dry run


def gate_self_test() -> list[str]:
    """Fabricated worlds just inside and just outside every gate threshold, and the decision table; the problems found.

    The intervals never decide a gate, so they are bootstrapped with few resamples here."""
    saved, gate_corpus.BOOTSTRAP = gate_corpus.BOOTSTRAP, 20
    try:
        return _gate_self_test()
    finally:
        gate_corpus.BOOTSTRAP = saved


def _gate_self_test() -> list[str]:
    problems: list[str] = []

    def check(change: str, contrast: int, gate: str, mutate: Callable[[dict[str, Any], str, str], None], expected: bool,
              label: str) -> None:
        inputs = fabricated_inputs()
        name, on_arm, off_arm = CONTRASTS[change][contrast]
        mutate(inputs, on_arm, off_arm)
        found = [g for g in change_gates(inputs, change) if g["contrast"] == name and g["gate"] == gate]
        if len(found) != 1 or found[0]["pass"] is not expected:
            problems.append(f"{change} {name} {gate} {label}")

    def flags(name: str, on_count: int, off_count: int = 0) -> Callable[[dict[str, Any], str, str], None]:
        def mutate(inputs: dict[str, Any], on_arm: str, off_arm: str) -> None:
            inputs["flags"][on_arm][name][:on_count] = 1
            inputs["flags"][off_arm][name][:off_count] = 1
        return mutate

    for change in CHANGES:
        for contrast in range(2):
            for flag, gate, budget in [*((f"first:{v}", v, WHOLE_BUDGET) for v in WHOLE_VETOES),
                                       *((f"step:{v}", f"step:{v}", STEP_BUDGET) for v in STEP_OBJECTS),
                                       ("false_first", "combined", COMBINED_BUDGET)]:
                check(change, contrast, gate, flags(flag, budget), True, "at the budget")
                check(change, contrast, gate, flags(flag, budget + 1), False, "over the budget")
                check(change, contrast, gate, flags(flag, budget + 3, budget + 3), True, "equal to the other arm")
                check(change, contrast, gate, flags(flag, budget + 4, budget + 3), False, "one over the other arm")
    for change, contrast in (("P", 0), ("P", 1), ("R", 0), ("R", 1)):
        check(change, contrast, "FA", flags("fa", 2, 2), True, "equal")
        check(change, contrast, "FA", flags("fa", 2, 1), False, "one more")
    for contrast in range(2):
        check("P", contrast, "BEN-P", flags("false_first", 1, 2), True, "one fewer")
        check("P", contrast, "BEN-P", flags("false_first", 2, 2), False, "equal")

        def pairs(ran: int, hit: int) -> Callable[[dict[str, Any], str, str], None]:
            def mutate(inputs: dict[str, Any], on_arm: str, off_arm: str) -> None:
                inputs["injections"][analysis_key(on_arm)]["100Hz+12dB"]["on"] = [ran, hit]
            return mutate

        for ran, hit, expected in ((30, 24, True), (30, 23, False), (29, 29, False), (100, 80, True), (100, 79, False)):
            check("T", contrast, "BEN-T:100Hz+12dB", pairs(ran, hit), expected, f"{hit} of {ran}")

        def deep(users: int, lost: int) -> Callable[[dict[str, Any], str, str], None]:
            def mutate(inputs: dict[str, Any], on_arm: str, off_arm: str) -> None:
                column = inputs["deep"][on_arm]
                column[users:] = np.nan
                column[:lost] = -3.0
            return mutate

        for users, lost, expected in ((90, 9, True), (90, 10, False), (30, 3, True), (29, 0, False)):
            check("R", contrast, "BEN-R1", deep(users, lost), expected, f"{users} users, {lost} lost")

        def sibilant(on_count: int, off_count: int) -> Callable[[dict[str, Any], str, str], None]:
            def mutate(inputs: dict[str, Any], on_arm: str, off_arm: str) -> None:
                inputs["sibilant"][on_arm][:on_count] = SIBILANT_LOSS_DB + 0.1
                inputs["sibilant"][off_arm][:off_count] = SIBILANT_LOSS_DB + 0.1
                inputs["sibilant"][on_arm][on_count:] = SIBILANT_LOSS_DB
            return mutate

        check("R", contrast, "BEN-R2", sibilant(1, 2), True, "one fewer")
        check("R", contrast, "BEN-R2", sibilant(2, 2), False, "equal")
    for failed, candidate in (([], "P+T+R"), (["P"], "T+R"), (["T"], "P+R"), (["R"], "P+T"), (["P", "T"], None), (["P", "T", "R"], None)):
        if candidate_of(failed) != candidate:
            problems.append(f"decision with {failed} failed")
    world = fabricated_inputs()
    if not decide(world)["ships"] or decide(world)["candidate"] != "P+T+R":
        problems.append("the neutral world does not ship")
    for name, gate, budget in (("fa", "S1", 0), ("first:B4", "S2:B4", WHOLE_BUDGET), ("step:B6", "S2:step:B6", STEP_BUDGET),
                               ("false_first", "S3", COMBINED_BUDGET)):
        for count, expected in ((budget, True), (budget + 1, False)):
            world = fabricated_inputs()
            world["flags"]["P+T+R"][name][:count] = 1
            if [g["pass"] for g in shipping_gates(world, "P+T+R") if g["gate"] == gate] != [expected]:
                problems.append(f"shipping gate {gate} at {count}")
    return problems


def tuner_runtime_factors(rows: list[dict[str, Any]]) -> list[float]:
    return [c["runtime_factor"] for row in rows if not row["excluded"] for view in row["arms"].values()
            for c in (view.get("tuner") or {}).get("candidates", ()) if math.isfinite(c["runtime_factor"])]


def reachability(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """What the owner must see before the freeze (flags, not rates)."""
    analysed = [r for r in rows if not r["excluded"]]
    factory = [r for r in analysed if r["condition"] == "factory"]
    reached = {}
    for key in ("current", "P+R"):
        for cell in INJECTIONS:  # users refused before verification get no injection: they stay in the denominator
            ran = sum(bool((r["injections"].get(key) or {}).get(cell, {}).get("on", {}).get("ran")) for r in factory)
            reached[f"T cell {key} {cell}: the tonal check ran for half of the factory users"] = bool(factory) and ran >= 0.5 * len(factory)
    measurable = sum(math.isfinite(_reading(arm_user(r, "current"), "deep_pause_db")) for r in factory)
    return {
        "P: tone_withheld fires": any(r["arms"]["P"]["tone_withheld"] for r in analysed),
        "current: a whole-setup false refusal": any(arm_flags(arm_user(r, "current"))["false_first"] for r in analysed),
        f"current: a sibilant loss above {SIBILANT_LOSS_DB:g} dB for a factory user": any(
            _reading(arm_user(r, "current"), "sibilant_loss_db") > SIBILANT_LOSS_DB for r in factory),
        "R1: a measurable deep pause for half of the factory users": bool(factory) and measurable >= 0.5 * len(factory),
        **reached}


def digest_of(result: dict[str, Any]) -> str:
    return loudness._digest(result)


def consistency(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Items 2-4 of the dry run: parity with the master code path, switch scope, determinism, restoration."""
    from mic_eq.analysis.voice_setup import analyze_voice_setup

    start = _switch_values()
    plain = [r for r in rows if not r["excluded"] and not r["construction"]["A"]
             and not r["arms"]["current"].get("analysis_failure") and not r["arms"]["current"]["b1"]]
    factory = [r for r in plain if r["condition"] == "factory"][:4]
    existing = [r for r in rows if not r["excluded"] and r["condition"] == "existing"][:4]
    report: dict[str, Any] = {"parity": [], "t_leaves_analysis": [], "determinism": [], "r_exact_for_existing": []}
    for row in factory:
        job = (str(root), row["condition"], row["index"], row["version"])
        _, env = prepare(job, development=True)
        assert env is not None
        ours = analyse("current", env)
        from mic_eq.config import EQSettings, Preset

        preset = Preset()
        master = analyze_voice_setup(  # the master code path: no switch, no tap, the module defaults
            env["noise"], env["setup"], SAMPLE_RATE, "broadcast", noise_model=refusals.MODEL, suppressor_strength=1.0,
            incumbent_settings={"gate": asdict(preset.gate), "rnnoise": asdict(preset.rnnoise), "eq": EQSettings().to_dict()},
            raw_noise_audio=env["raw_noise"], raw_speech_audio=env["raw_setup"])
        report["parity"].append(digest_of(ours["result"]) == digest_of(master))
        report["determinism"].append(digest_of(ours["result"]) == digest_of(analyse("current", env)["result"]))
        report["t_leaves_analysis"].append(digest_of(ours["result"]) == digest_of(analyse("T", env)["result"]))
    for row in existing:
        job = (str(root), row["condition"], row["index"], row["version"])
        again = evaluate_user(job, development=True, map_r=False)
        report["r_exact_for_existing"].append(all(
            loudness._digest(again["arms"][arm]) == loudness._digest(again["arms"][existing_twin(arm)])
            and loudness._digest(again["arms"][existing_twin(arm)]) == loudness._digest(row["arms"][existing_twin(arm)])
            for arm in ARMS if existing_twin(arm) != arm))
    p_scope = [loudness._digest(r["arms"]["P"]) == loudness._digest(r["arms"]["current"]) or r["arms"]["P"]["tone_withheld"]
               for r in rows if not r["excluded"]]
    report |= {"p_differs_only_with_tone_withheld": all(p_scope), "switches_restored": _switch_values() == start}
    report["all_ok"] = bool(all(all(v) for v in report.values() if isinstance(v, list)) and all(
        report[k] for k in ("p_differs_only_with_tone_withheld", "switches_restored")))
    return report


def resume_check(rows: list[dict[str, Any]], jobs: list[tuple[str, str, int, str]], scratch: Path) -> bool:
    """Stop after one row, resume, and get the rows an uninterrupted run gave, apart from timing."""
    pair = [j for j in jobs if j[1] == "existing"][:2]
    partial = scratch / "resume-check.run.partial.jsonl"
    partial.unlink(missing_ok=True)
    by_key = {(r["split"], r["index"], r["model"]): r for r in rows}
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(by_key[(pair[0][1], pair[0][2], pair[0][3])], default=float) + "\n", encoding="utf-8", newline="\n")
    build = by_key[(pair[0][1], pair[0][2], pair[0][3])]["build"]
    refusals.check_resume(partial, build, pair)
    resumed = refusals.run_jobs(functools.partial(evaluate_user, development=True), pair, 1, partial)
    partial.unlink()
    return all(loudness._digest(json.loads(json.dumps(a, default=float))) == loudness._digest(by_key[(a["split"], a["index"], a["model"])])
               for a in resumed)


def timings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    analysis = [t for r in rows if not r["excluded"] for t in r["runtime"]["analysis"].values()]
    verification = [t for r in rows if not r["excluded"] for t in r["runtime"]["verification"].values()]
    summary = lambda values: {"mean_s": float(np.mean(values)), "max_s": float(np.max(values)), "count": len(values)}  # noqa: E731
    return {"analysis": summary(analysis), "verification": summary(verification)} if analysis else {}


def bookkeeping(root: Path, rows: list[dict[str, Any]], jobs: list[tuple[str, str, int, str]], scratch: Path) -> dict[str, Any]:
    """Counts and flags of the development rows: no rate and no interval is computed from them."""
    analysed = [r for r in rows if not r["excluded"]]
    views = [{**arm_user(r, "current"), "level_check": r["level_check"], "probes": r["probes"]} for r in analysed]
    factors = tuner_runtime_factors(rows)
    flags = reachability(rows)
    checks = consistency(root, rows)
    return {
        "development_user_conditions": len(rows), "analysed": len(analysed),
        "excluded": [{"condition": r["condition"], "speaker": r["speaker"], "version": r["version"], "why": r["excluded"]}
                     for r in rows if r["excluded"]],
        "construction": {f"{r['speaker']} {r['version']}": {k: v for k, v in r["construction"].items() if k != "takes"}
                         | {"retakes": [t["vetoes"] for t in r["construction"].get("takes", ())]}
                         for r in rows if r["condition"] == "factory"},
        "consistency": checks, "gate_self_test_problems": gate_self_test(),
        "tuner": {"largest_runtime_factor": max(factors, default=None), "any_over_half": any(f > 0.5 for f in factors),
                  "runtime_gate_lifted": UNBOUNDED_TUNER_RUNTIME},
        "probes": refusals.bookkeeping(views) if views else None, "reachability": flags,
        "reachability_all": all(flags.values()), "resume_identical": resume_check(rows, jobs, scratch) if analysed else None,
        "timing": timings(rows)}


# ----------------------------------------------------------------------------------------------
# Files and commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "probe", "report", "freeze"))
    parser.add_argument("--root", type=Path, default=DAPS, help="the DAPS script 1 corpus root")
    parser.add_argument("--development", action="store_true", help="probe: confirm that the development speakers run")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS, help="report: the rows to judge")
    parser.add_argument("--output", type=Path, help="probe: the bookkeeping; report: the tracked report")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(FROZEN, RULES_FINAL, DEFAULT_ROWS, args.root)
        return 0
    if args.command == "report":
        frozen = check_frozen(FROZEN, args.root)
        source = json.loads(args.rows.read_text(encoding="utf-8"))
        if source.get("development"):
            raise SystemExit("the rows are development rows: spent speakers give no rates")
        if source["identity"]["build"] != refusals.build_of(frozen):
            raise SystemExit("the rows were made by another build than the one frozen")
        check_rows(source["rows"])
        refusals._write(args.output or DEFAULT_REPORT, build_report(source, frozen))
        return 0
    development = args.command == "probe"
    if development and not args.development:
        raise SystemExit("probe re-runs the development speakers; pass --development")
    jobs = jobs_of(args.root, development)
    if development:
        output, rows_path = args.output or DEFAULT_BOOKKEEPING, DEVELOPMENT_ROWS
    else:
        check_frozen(FROZEN, args.root)
        if args.output not in (None, DEFAULT_ROWS):
            raise SystemExit(f"the study's rows are always {DEFAULT_ROWS}; --output is for probe and report")
        output = rows_path = DEFAULT_ROWS
        refusals.check_not_finished(rows_path)
    partial = rows_path.with_suffix(".run.partial.jsonl")
    refusals.check_assets()
    started = build_id(str(args.root))
    refusals.check_resume(partial, started, jobs)
    print("build:", started, "| VAD_MODEL_PATH:", os.environ.get("VAD_MODEL_PATH"), flush=True)
    rows = refusals.run_jobs(functools.partial(evaluate_user, development=development), jobs,
                             refusals.worker_count(args.workers), partial)
    check_unchanged(args.root, started, rows)
    source = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "development": development,
              "identity": {"build": started}, "rows": rows}
    refusals._write(rows_path, source)
    partial.unlink()
    if development:
        from check_evaluation_hygiene import validate_report

        fabricated = rows_path.parent / "fabricated-report.json"
        refusals._write(fabricated, build_report({"rows": fabricated_rows(), "identity": source["identity"]}, frozen_hashes(args.root)))
        refusals._write(output, bookkeeping(args.root, rows, jobs, rows_path.parent) | {
            "fabricated_report_hygiene_errors": validate_report(fabricated)})
    sys.stdout.write(f"{len(rows)} rows\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
