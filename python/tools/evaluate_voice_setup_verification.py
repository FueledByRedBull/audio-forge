"""Qualify the bypassed tonal check of Voice Setup's verification against the released one.

Subcommands:
  run     For each simulated user (``evaluate_voice_setup.py``) and noise setting, run
          ``analyze_voice_setup`` once as the app does. Then emulate the dialog's
          verification of the second take (the start of the evaluation capture, as long
          as the setup capture), including its automatic reductions, under each arm: for
          the proposal as it is, and for four deliberately wrong EQs that add 6, 9 or
          12 dB to the correction band nearest 3 kHz or take 6 dB off the band nearest
          200 Hz (the EQ's 12 dB limit applies and the change applied is recorded). Both
          go through the full chain on the raw captures, as the app's verification does.
  report  Apply the rule to the rows written by ``run``.

Arms (they differ only through ``voice_setup.TONAL_CHECK_BYPASSES_SUPPRESSOR``):
  released   The constant off: the tonal check ("target error") measures the real
             full-chain render, so a noise suppressor that reshapes the spectrum counts
             as a tonal error.
  candidate  The constant on, as shipped: when the proposal's noise suppressor is
             enabled, the target error is computed from a second render of the same
             take and noise capture through the same chain with only the suppressor
             switched off. Every other finding of verification (delivery, limiter,
             compressor, de-esser, noise and SNR safety, dropout) comes from the real
             full-chain render. With the suppressor disabled in the proposal, or for
             downstream-only callers, verification is exactly as released. The
             suppressor dulling advice is unaffected.

Rule (fixed on 2026-10-13 04:20 local by the study's owner, before any row of the second
EARS cohort existed)

Why new speakers: the first design measured the tonal check against the suppressor's own
output. It removed every false alarm on the first EARS cohort but failed two gates: it
read up to 2.4 dB lower than the tonal check does with the suppressor bypassed, and missed
6 of 96 wrong EQs of 9 dB. The candidate above is that reference itself, so its behaviour
on the speakers seen so far is known and they are development data for it. On the first
EARS cohort (102 verified pairs over RNNoise 1.0, DeepFilterNet 1.0 and DeepFilterNet LL
1.0) the released check rolled back 35 pairs for target error and the check with the
suppressor bypassed rolled back 6; it reported a 9 dB wrong EQ at 3 kHz for 96 of 102 and
a 12 dB one for 102 of 102. On VoiceBank (54 pairs) it rolled back none and reported 17 of
18 and 18 of 18 per setting. The thresholds below were set knowing those numbers.

Users and settings: the second EARS cohort (24 speakers, none of them in the first), splits
`fit`, `calibrate` and `test`, indices 0-15: 48 users. Settings: RNNoise 1.0, DeepFilterNet
1.0 and DeepFilterNet LL 1.0. A pair is one user at one setting. A pair whose Auto-EQ stage
produced no settings, or whose verification ends in retry before any spectrum is measured,
is counted and left out of the gates that need what it lacks.

Gates (judged over the three settings together; adopted only if all hold):

  false_alarms_low     N1. With the proposal as it is, the candidate rolls back for target
                       error in at most 10 % of pairs, in at most 15 % within any one
                       setting, and in at most a third as many pairs as the released check
                       does.
  catches_wrong_eq     N2. With 9 dB added to the proposal's EQ band nearest 3 kHz the
                       candidate reports target error for at least 90 % of pairs, and with
                       12 dB (as far as the EQ's gain limit allows) for at least 97 %;
                       within any one setting at least 85 % and 94 %.
  never_stricter       N3. No pair that the released check accepts is rolled back for
                       target error by the candidate.
  nothing_else_moves   N4. Apart from the target error and its consequences for the
                       decision, every other verification finding is identical between
                       released and candidate, and the advisory fires for exactly the
                       same pairs with the same value.

Reported, not gated: the 6 dB wrong EQs at 3 kHz and 200 Hz; verification decisions per
setting under released and candidate; the extra render time per verification.

A candidate that fails stays out; there is no second attempt on these speakers.

Implementation notes. A pair is "verified" when its verification compared spectra; every
gate needs a measured spectrum, so the verified pairs form every population. A pair whose
verification measures spectra and then retries for a delivery mismatch stays in. "Rolls
back for target error" is the reason code ``target_error`` in the final outcome of the
emulated dialog (after up to three automatic reductions); "accepts" is the final decision
``accept``. N1 and N2 are tested with integer arithmetic (candidate * 10 <= pairs;
candidate * 100 <= 15 * pairs of the setting, for each setting that has pairs; 3 *
candidate <= released; reports * 100 >= percent * pairs); a denominator of zero fails a
gate. N4 compares the first-pass outcomes of the two arms field by field (NaN equal to NaN)
except the target error after the chain, the decision, the reasons and the reason codes,
so the "before" error and ``reduction_targets`` must be equal; the sets of safety and
delivery codes must be equal; where neither arm reports target error in its first pass or
final outcome, the whole emulated flow (decision, reductions, limiter pressure, findings,
codes and texts) must be equal; the advisory (code, sentence, value, abstained marker)
must be equal in the first-pass and in the final outcome of every pair. Differences in
later passes of a flow whose first pass reports target error in either arm are
consequences for the decision. The extra time is, for each verified pair, the duration of
the first verification call of each arm on the proposal (same inputs, same process, the
order of the arms alternating between pairs).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_voice_setup as voice_setup_eval
import evaluate_voice_setup_loudness as loudness

REPO_ROOT = voice_setup_eval.REPO_ROOT
SAMPLE_RATE = voice_setup_eval.SAMPLE_RATE
SETTINGS: dict[str, tuple[str, float]] = {
    "rnnoise-1.00": ("rnnoise", 1.0),
    "deepfilter-1.00": ("deepfilter", 1.0),
    "deepfilter-ll-1.00": ("deepfilter-ll", 1.0),
}
ARMS = ("released", "candidate")
SWITCH = "TONAL_CHECK_BYPASSES_SUPPRESSOR"
# (band centre in Hz, gain change in dB) of the deliberately wrong EQs
WRONG_EQS = {"plus6_3k": (3_000.0, 6.0), "plus9_3k": (3_000.0, 9.0), "plus12_3k": (3_000.0, 12.0),
             "minus6_200": (200.0, -6.0)}
EQ_GAIN_LIMIT_DB = 12.0
# Gate limits of the rule, in percent. N2: (overall, within any one setting).
N1_OVERALL_PCT, N1_SETTING_PCT, N1_RELEASED_DIVISOR = 10, 15, 3
N2_PCT = {"plus9_3k": (90, 85), "plus12_3k": (97, 94)}
# Fields the target error moves; every other field of an outcome is a finding.
MOVED = ("spectral_target_error_after_db", "decision", "reasons", "reason_codes")
SAFETY_CODES = ("verification_delivery_mismatch", "voice_gate_silencing", "noise_safety_limit", "snr_safety_limit",
                "compressor_peak_cap", "true_peak_ceiling")
ADVISORY_CODE = "suppressor_dulling"
# The renders a first verification pass makes: the take and the room noise, the frontend alone for
# the advice, and for the candidate the take and the room noise again with the suppressor off.
RENDERS = {"released": [True, True, True], "candidate": [True, True, True, False, False]}
SOURCES = (
    "python/tools/evaluate_voice_setup_verification.py",
    *loudness.SOURCES,
    "python/mic_eq/analysis/listening_comparison.py",
    "python/mic_eq/analysis/auto_eq_parts/target.py",
)
EXTENSION = loudness.EXTENSION


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_plain))


@contextmanager
def _switch(on: bool):
    """Give ``voice_setup`` the switch of one arm."""
    from mic_eq.analysis import voice_setup

    saved = getattr(voice_setup, SWITCH)
    setattr(voice_setup, SWITCH, bool(on))
    try:
        yield
    finally:
        setattr(voice_setup, SWITCH, saved)


@contextmanager
def _recording_renders(log: list[bool]):
    """Record, for every native render ``voice_setup`` makes, whether the chain had the suppressor on."""
    from mic_eq.analysis import voice_setup

    original = voice_setup.simulate_candidate_chain

    def counted(audio, rate, eq_settings, chain=None, *args, **kwargs):
        log.append(((chain or {}).get("suppressor") or {}).get("enabled", True) is not False)
        return original(audio, rate, eq_settings, chain, *args, **kwargs)

    voice_setup.simulate_candidate_chain = counted
    try:
        yield
    finally:
        voice_setup.simulate_candidate_chain = original


def _verify(result: dict[str, Any], arm: str, captures: dict[str, np.ndarray]) -> dict[str, Any]:
    """The dialog's verification of ``result`` under one arm: raw captures through the full chain, up to three
    automatic reductions. Keeps the first pass and the final outcome in full."""
    from mic_eq.analysis.voice_setup import validate_voice_setup_verification

    result = deepcopy(result)
    adjustments = 0
    first: dict[str, Any] | None = None
    calls_s: list[float] = []
    renders: list[list[bool]] = []
    while True:
        # The dialog re-reads the chain from the applied controls after every reduction.
        result["verification_chain_settings"] = loudness._chain(result)
        log: list[bool] = []
        with _switch(arm == "candidate"), _recording_renders(log):
            tick = time.perf_counter()
            outcome = validate_voice_setup_verification(
                captures["noise"], captures["setup"], captures["second"], SAMPLE_RATE, result, "broadcast",
                raw_noise_audio=captures["raw_noise"], raw_verification_speech_audio=captures["raw_second"],
            )
            calls_s.append(time.perf_counter() - tick)
        renders.append(log)
        if first is None:
            first = {key: outcome.get(key) for key in (
                "decision", "reason_codes", "spectral_target_error_before_db", "spectral_target_error_after_db")}
            first["outcome"] = _json_safe(outcome)
        targets = [str(target) for target in outcome.get("reduction_targets") or ()]
        if outcome["decision"] != "reduce" or adjustments >= loudness.MAX_ADJUSTMENTS:
            break
        before = deepcopy((result["compressor_settings"], result["deesser_settings"]))
        loudness._reduce(result, targets)
        if before == (result["compressor_settings"], result["deesser_settings"]):
            break
        adjustments += 1
    return {
        "decision": "gave_up" if outcome["decision"] == "reduce" else outcome["decision"],
        "adjustments": adjustments, "limiter_pressure": "limiter" in targets,
        "first_pass": first, "outcome": _json_safe(outcome), "calls_s": calls_s, "renders": renders,
    }


def _both(result: dict[str, Any], captures: dict[str, np.ndarray], flip: bool) -> dict[str, Any]:
    """Both arms on one proposal; ``flip`` swaps the order so that no arm always runs first."""
    order = ARMS[::-1] if flip else ARMS
    done = {arm: _verify(result, arm, captures) for arm in order}
    return {arm: done[arm] for arm in ARMS}


def _shifted_eq(eq: dict[str, Any], frequency_hz: float, delta_db: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The proposal's typed EQ with ``delta_db`` added to its correction band nearest ``frequency_hz``."""
    wrong = deepcopy(eq)
    bands = wrong["layers"]["correction"]
    index = min(range(len(bands)), key=lambda i: abs(math.log(bands[i]["frequency_hz"] / frequency_hz)))
    before = float(bands[index]["gain_db"])
    after = float(np.clip(before + delta_db, -EQ_GAIN_LIMIT_DB, EQ_GAIN_LIMIT_DB))
    bands[index]["gain_db"] = after
    if len(wrong.get("band_gains", ())) == len(bands):
        wrong["band_gains"] = [after if i == index else float(g) for i, g in enumerate(wrong["band_gains"])]
    return wrong, {"band": index, "band_frequency_hz": float(bands[index]["frequency_hz"]),
                   "gain_before_db": before, "gain_after_db": after, "applied_change_db": after - before}


def _provenance() -> dict[str, Any]:
    """Where the code of this process came from."""
    import mic_eq
    from mic_eq.analysis import voice_setup

    extension = importlib.util.find_spec("mic_eq.mic_eq_core")
    return {"mic_eq": mic_eq.__file__, "evaluator": __file__, "voice_setup": voice_setup.__file__,
            "extension": extension.origin if extension else None,
            "voice_setup_sha256": hashlib.sha256(Path(voice_setup.__file__).read_bytes()).hexdigest()}


def evaluate_pair(job: tuple[str, str, int, str]) -> dict[str, Any]:
    root_text, split_name, index, setting = job
    from mic_eq.analysis import voice_setup
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    model, strength = SETTINGS[setting]
    root = Path(root_text)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    simulated = voice_setup_eval.user(root, manifest, split_name, index)
    raw_noise, raw_setup, raw_evaluation = (
        simulated[name] for name in ("noise_capture", "setup_capture", "evaluation_capture"))
    raw_second = raw_evaluation[: raw_setup.size]
    noise, setup, second = (filtered_capture_for_analysis(audio, SAMPLE_RATE) for audio in (raw_noise, raw_setup, raw_second))
    captures = {"noise": noise, "setup": setup, "second": second, "raw_noise": raw_noise, "raw_second": raw_second}
    row: dict[str, Any] = {key: simulated[key] for key in ("split", "index", "speaker", "environment", "snr_db", "level_dbfs")}
    row["model"] = setting
    row["noise_model"], row["strength"] = model, strength
    row["provenance"] = _provenance()

    result = voice_setup.analyze_voice_setup(
        noise, setup, SAMPLE_RATE, "broadcast", noise_model=model, suppressor_strength=strength,
        incumbent_settings={"rnnoise": {"enabled": True, "strength": strength, "model": model}},
        raw_noise_audio=raw_noise, raw_speech_audio=raw_setup,
    )
    row["complete"] = result.get("eq_settings") is not None
    row["eq_error"] = result.get("eq_error")
    row["proposed_suppressor"] = {k: result["suppressor_settings"].get(k) for k in ("enabled", "strength", "model")}
    if not row["complete"]:
        return row
    result["eq_settings"] = loudness._applied_eq(result["eq_settings"])

    flip = (voice_setup_eval.SPLITS.index(split_name) + index) % 2 == 1
    verification: dict[str, Any] = {**_both(result, captures, flip), "wrong": {}}
    for name, (frequency_hz, delta_db) in WRONG_EQS.items():
        eq, change = _shifted_eq(result["eq_settings"], frequency_hz, delta_db)
        verification["wrong"][name] = {"change": change, **_both({**result, "eq_settings": eq}, captures, flip)}
    row["verification"] = verification
    return row


# ----------------------------------------------------------------------------------------------
# The rule's gates


def _dist(values: list[float]) -> dict[str, Any]:
    values = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)), "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)), "max": max(values)}


def _frac(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def verified(row: dict[str, Any]) -> bool:
    """The verification compared spectra: not a retry on a rejected (clipped or too short) second take."""
    return bool(row["complete"]) and "spectral_target_error_after_db" in row["verification"]["released"]["outcome"]


def _tgt(arm: dict[str, Any]) -> bool:
    return "target_error" in (arm["outcome"].get("reason_codes") or ())


def _first_tgt(arm: dict[str, Any]) -> bool:
    return "target_error" in (arm["first_pass"].get("reason_codes") or ())


def _pair_id(row: dict[str, Any]) -> list[Any]:
    return [row["split"], row["index"], row["model"]]


def _advisory(outcome: dict[str, Any]) -> tuple[Any, ...]:
    return (ADVISORY_CODE in (outcome.get("reason_codes") or ()), outcome.get("suppressor_dulling_db"),
            outcome.get("suppressor_dulling_abstained"), outcome.get("suppressor_dulling_advice"))


def _findings(outcome: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in outcome.items() if k not in MOVED}


def _same(a: Any, b: Any) -> bool:
    """Exact equality of two JSON-like values; a NaN equals a NaN."""
    return json.dumps(a, sort_keys=True, default=_plain) == json.dumps(b, sort_keys=True, default=_plain)


def pair_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for setting in SETTINGS:
        sub = [r for r in rows if r["model"] == setting]
        reasons: dict[str, int] = {}
        for r in sub:
            if not r["complete"]:
                key = str(r.get("eq_error"))[:90]
                reasons[key] = reasons.get(key, 0) + 1
        out[setting] = {
            "pairs": len(sub), "no_settings": sum(not r["complete"] for r in sub), "no_settings_reasons": reasons,
            "second_take_rejected": sum(r["complete"] and not verified(r) for r in sub),
            "verified": sum(verified(r) for r in sub),
            "suppressor_changed_by_analysis": sum(
                r["proposed_suppressor"]["strength"] != r["strength"] or r["proposed_suppressor"]["model"] != r["noise_model"]
                or not r["proposed_suppressor"]["enabled"] for r in sub if r["complete"]),
        }
    return out


def false_alarms_low(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    arm = lambda r, name: r["verification"][name]  # noqa: E731
    cand = sum(_tgt(arm(r, "candidate")) for r in pairs)
    rel = sum(_tgt(arm(r, "released")) for r in pairs)
    per = {}
    for setting in SETTINGS:
        sub = [r for r in pairs if r["model"] == setting]
        c, d = sum(_tgt(arm(r, "candidate")) for r in sub), sum(_tgt(arm(r, "released")) for r in sub)
        per[setting] = {"n": len(sub), "candidate": c, "released": d, "rate": _frac(c, len(sub)),
                        "passed": c * 100 <= N1_SETTING_PCT * len(sub)}
    overall = cand * 100 <= N1_OVERALL_PCT * len(pairs)
    setting_clause = all(v["passed"] for v in per.values() if v["n"])
    third = N1_RELEASED_DIVISOR * cand <= rel
    return {"passed": bool(pairs) and overall and setting_clause and third, "n": len(pairs),
            "candidate_target_error": cand, "released_target_error": rel, "rate": _frac(cand, len(pairs)),
            "clauses": {"overall_at_most_10_percent": overall, "each_setting_at_most_15_percent": setting_clause,
                        "at_most_a_third_of_released": third},
            "per_setting": per}


def catches_wrong_eq(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (overall_pct, setting_pct) in N2_PCT.items():
        report = lambda r, who: _tgt(r["verification"]["wrong"][name][who])  # noqa: E731
        hits = sum(report(r, "candidate") for r in pairs)
        per = {}
        for setting in SETTINGS:
            sub = [r for r in pairs if r["model"] == setting]
            c = sum(report(r, "candidate") for r in sub)
            per[setting] = {"n": len(sub), "candidate_reports": c, "rate": _frac(c, len(sub)),
                            "passed": c * 100 >= setting_pct * len(sub)}
        overall = hits * 100 >= overall_pct * len(pairs)
        setting_clause = all(v["passed"] for v in per.values() if v["n"])
        out[name] = {"passed": bool(pairs) and overall and setting_clause, "n": len(pairs), "candidate_reports": hits,
                     "rate": _frac(hits, len(pairs)), "limits_percent": {"overall": overall_pct, "per_setting": setting_pct},
                     "clauses": {"overall": overall, "each_setting": setting_clause}, "per_setting": per,
                     "released_reports": sum(report(r, "released") for r in pairs),
                     "applied_change_db": _dist([r["verification"]["wrong"][name]["change"]["applied_change_db"] for r in pairs])}
    return {"passed": all(v["passed"] for v in out.values()), "wrong_eqs": out}


def never_stricter(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    stricter = [r for r in pairs if r["verification"]["released"]["decision"] == "accept"
                and _tgt(r["verification"]["candidate"])]
    return {"passed": bool(pairs) and not stricter, "n": len(pairs),
            "released_accepts": sum(r["verification"]["released"]["decision"] == "accept" for r in pairs),
            "rolled_back_by_candidate": [_pair_id(r) for r in stricter]}


def nothing_else_moves(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    moved: list[dict[str, Any]] = []
    for r in pairs:
        rel, cand = r["verification"]["released"], r["verification"]["candidate"]
        problems = []
        if not _same(_findings(rel["first_pass"]["outcome"]), _findings(cand["first_pass"]["outcome"])):
            problems.append("first-pass findings differ")
        codes = lambda a: {c for c in a["first_pass"]["reason_codes"] if c in SAFETY_CODES}  # noqa: E731
        if codes(rel) != codes(cand):
            problems.append("first-pass safety or delivery codes differ")
        if not _same(_advisory(rel["first_pass"]["outcome"]), _advisory(cand["first_pass"]["outcome"])):
            problems.append("first-pass advisory differs")
        if not _same(_advisory(rel["outcome"]), _advisory(cand["outcome"])):
            problems.append("final advisory differs")
        if not (_first_tgt(rel) or _first_tgt(cand) or _tgt(rel) or _tgt(cand)):
            for key_name in ("decision", "adjustments", "limiter_pressure"):
                if rel[key_name] != cand[key_name]:
                    problems.append(f"{key_name} differs without target error")
            if (not _same(_findings(rel["outcome"]), _findings(cand["outcome"]))
                    or rel["outcome"]["reason_codes"] != cand["outcome"]["reason_codes"]
                    or rel["outcome"]["reasons"] != cand["outcome"]["reasons"]):
                problems.append("emulated flow differs without target error")
        if problems:
            moved.append({"pair": _pair_id(r), "problems": problems})
    fired = lambda who: [_pair_id(r) for r in pairs if _advisory(r["verification"][who]["outcome"])[0]]  # noqa: E731
    return {"passed": bool(pairs) and not moved, "n": len(pairs), "pairs_with_differences": len(moved),
            "examples": moved[:5],
            "advisory_fires": {who: {"count": len(fired(who)), "pairs": fired(who)} for who in ARMS}}


def decisions_by_setting(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for setting in SETTINGS:
        sub = [r for r in pairs if r["model"] == setting]
        out[setting] = {arm: {d: sum(r["verification"][arm]["decision"] == d for r in sub)
                              for d in ("accept", "gave_up", "rollback", "retry")}
                        | {"target_error": sum(_tgt(r["verification"][arm]) for r in sub)} for arm in ARMS} | {"verified": len(sub)}
    return out


def wrong_eq_reported(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("plus6_3k", "minus6_200"):
        count = lambda sub, who: sum(_tgt(r["verification"]["wrong"][name][who]) for r in sub)  # noqa: E731
        # Pairs that already roll back for target error with the proposal as it is report every wrong EQ.
        beyond = lambda who: sum(  # noqa: E731
            _tgt(r["verification"]["wrong"][name][who]) and not _tgt(r["verification"][who]) for r in pairs)
        out[name] = {who: count(pairs, who) for who in ARMS} | {"n": len(pairs)} | {
            "both": sum(_tgt(r["verification"]["wrong"][name]["candidate"]) and _tgt(r["verification"]["wrong"][name]["released"])
                        for r in pairs),
            "beyond_the_proposal": {who: beyond(who) for who in ARMS},
            "per_setting": {s: {who: count([r for r in pairs if r["model"] == s], who) for who in ARMS}
                            | {"n": sum(r["model"] == s for r in pairs)} for s in SETTINGS}}
    return out


def extra_time(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The extra time of one candidate verification over a released one: the first call of each arm on the
    proposal (same inputs, same process, order alternated)."""
    first = lambda r, arm: r["verification"][arm]["calls_s"][0]  # noqa: E731
    extra = [first(r, "candidate") - first(r, "released") for r in pairs]
    return {"pairs": len(pairs), "first_call_s": {arm: _dist([first(r, arm) for r in pairs]) for arm in ARMS},
            "first_call_extra_s": _dist(extra),
            "first_call_extra_percent_of_released": (
                100.0 * float(np.mean(extra)) / float(np.mean([first(r, "released") for r in pairs])) if pairs else None),
            "first_call_extra_by_setting_s": {
                s: _dist([first(r, "candidate") - first(r, "released") for r in pairs if r["model"] == s]) for s in SETTINGS},
            # Integrity of this harness, not a gate: the arms rendered as described above.
            "first_pass_renders_as_described": sum(
                all(r["verification"][arm]["renders"][0] == RENDERS[arm] for arm in ARMS) for r in pairs)}


def verification_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [r for r in rows if verified(r)]
    metrics: dict[str, Any] = {
        "pairs": len(rows), "verified_pairs": len(pairs),
        "users": len({(r["split"], r["index"]) for r in rows}),
        "verified_users": len({(r["split"], r["index"]) for r in pairs}),
        "speakers": len({r["speaker"] for r in rows}), "verified_speakers": len({r["speaker"] for r in pairs}),
        "accounting": pair_accounting(rows),
        "left_out": [{"pair": _pair_id(r), "speaker": r["speaker"],
                      "why": "no Auto-EQ settings: " + str(r.get("eq_error")) if not r["complete"] else "second take rejected"}
                     for r in rows if not verified(r)],
        "false_alarms_low": false_alarms_low(pairs), "catches_wrong_eq": catches_wrong_eq(pairs),
        "never_stricter": never_stricter(pairs), "nothing_else_moves": nothing_else_moves(pairs),
        "reported": {"wrong_eq_6db": wrong_eq_reported(pairs), "decisions_by_setting": decisions_by_setting(pairs),
                     "extra_time": extra_time(pairs)},
    }
    gates = {name: metrics[name]["passed"] for name in
             ("false_alarms_low", "catches_wrong_eq", "never_stricter", "nothing_else_moves")}
    return {"metrics": metrics, "gates": gates, "passed": bool(pairs) and all(gates.values())}


def _limitations(metrics: dict[str, Any]) -> list[str]:
    blind = metrics["reported"]["wrong_eq_6db"]["minus6_200"]
    return [
        "Simulated users (the corpus above): speech in synthetic rooms with DEMAND noise, not microphone captures.",
        "Verification hears the first 10 s of the 20 s evaluation capture.",
        f"{metrics['verified_users']} of {metrics['users']} users are verified, from {metrics['verified_speakers']} of "
        f"{metrics['speakers']} speakers; the other {metrics['users'] - metrics['verified_users']} get no Auto-EQ "
        "settings from the Auto-EQ stage, which this change does not touch, or a rejected second take, and are left out.",
        "EARS speech is licensed for non-commercial use and used for evaluation only; no shipped value was fitted on it. "
        "The gate thresholds were set knowing the first EARS cohort and VoiceBank, whose speakers are not in this cohort; "
        "this candidate is the second design, after the first was rejected on the first cohort.",
        f"The tonal check is nearly blind to errors below 1 kHz: a 6 dB cut at 200 Hz adds a target error report beyond "
        f"the proposal's own for {blind['beyond_the_proposal']['released']} of {blind['n']} pairs with the released check "
        f"and {blind['beyond_the_proposal']['candidate']} with the candidate. This is unchanged.",
        "The candidate renders the take and the room noise twice more, without the suppressor; the extra time was measured "
        "with four workers in parallel and other load on the machine was not controlled.",
        "Not covered: a chain with a top-level suppressor_enabled, which the native adapter would let override "
        "suppressor.enabled (neither the app nor this evaluator sets it).",
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--root", type=Path, help="run: the corpus root")
    parser.add_argument("--splits", default=",".join(voice_setup_eval.SPLITS), help="run: splits, comma-separated")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--users", type=int, default=16, help="run: users per split")
    parser.add_argument("--settings", default=",".join(SETTINGS), help="run: settings, comma-separated")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows", type=Path, help="report: rows written by run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.root is None:
            parser.error("run needs --root")
        splits, settings = args.splits.split(","), args.settings.split(",")
        jobs = [(str(args.root), split, index, setting) for split in splits
                for index in range(args.first, args.first + args.users) for setting in settings]
        vad = os.environ.get("VAD_MODEL_PATH")
        print("provenance:", json.dumps(_provenance()), "| VAD_MODEL_PATH:", vad, flush=True)
        started = datetime.now().astimezone()
        print("started", started.isoformat(timespec="seconds"), flush=True)
        partial_rows = args.output.with_suffix(".run.partial.jsonl")
        rows = voice_setup_eval.run(evaluate_pair, jobs, min(args.workers, 4), partial_rows)
        from mic_eq.analysis import voice_setup
        provenance = _provenance()
        result = {
            "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "root": str(args.root), "splits": splits, "settings": settings, "first": args.first, "users": args.users,
            "module_constants": {SWITCH: getattr(voice_setup, SWITCH),
                                 "SUPPRESSOR_DULLING_THRESHOLD_DB": voice_setup.SUPPRESSOR_DULLING_THRESHOLD_DB,
                                 "LIMITER_ALLOWANCE_DB": voice_setup.LIMITER_ALLOWANCE_DB},
            "identity": {**provenance, "extension_sha256": hashlib.sha256(Path(provenance["extension"]).read_bytes()).hexdigest(),
                         "vad_model_path": vad,
                         "vad_model_sha256": hashlib.sha256(Path(vad).read_bytes()).hexdigest() if vad else None},
            "rows": rows,
        }
        _write(args.output, result)
        partial_rows.unlink()
        sys.stdout.write(f"{len(rows)} rows\n")
        return 0
    source = json.loads(args.rows.read_text(encoding="utf-8"))
    summary = verification_report(source["rows"])
    root = Path(source["root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = source["identity"]
    rows_path = args.rows.resolve()
    result = {
        "schema_version": 2, "audible_change": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Voice Setup verification: measure the tonal check with the noise suppressor bypassed",
        "predefined_rule": "The rule in the docstring of python/tools/evaluate_voice_setup_verification.py.",
        "splits": source["splits"], "settings": source["settings"],
        "user_indices": {name: sorted({row["index"] for row in source["rows"] if row["split"] == name})
                         for name in source["splits"]},
        "retained": summary["passed"], **summary,
        "source_sha256": {name: _sha256(REPO_ROOT / name) for name in SOURCES},
        "implementation_sha256": {EXTENSION: identity["extension_sha256"]},
        "evaluation_contract": {
            "configuration": {"sample_rate": SAMPLE_RATE, "target_preset": "broadcast", "arms": ARMS,
                              "switch": SWITCH, "max_verification_adjustments": loudness.MAX_ADJUSTMENTS,
                              "wrong_eqs_hz_db": WRONG_EQS},
            "asset_hashes": {"corpus_manifest": _sha256(root / "manifest.json"), "vad_model": identity["vad_model_sha256"]},
            "runtime": {"max_p99_frame_seconds": None, "reason": "Offline setup verification; no realtime code changed."},
            "latency": {"measurement": None, "scope": "The change is to a setup check, not the signal path."},
            "clean_preservation": {"method": "None: verification only decides whether a proposal is applied and no audio "
                                             "the app processes changes. Gate nothing_else_moves holds every other "
                                             "finding identical."},
        },
        "corpus": {key: manifest[key] for key in ("description", "licenses", "redistribution")},
        "retained_evidence": {"directory": rows_path.parent.relative_to(REPO_ROOT).as_posix(),
                              "rows": rows_path.relative_to(REPO_ROOT).as_posix(), "rows_sha256": _sha256(rows_path)},
        "limitations": _limitations(summary["metrics"]),
    }
    _write(args.output, result)
    sys.stdout.write(json.dumps({key: result[key] for key in ("gates", "passed")}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
