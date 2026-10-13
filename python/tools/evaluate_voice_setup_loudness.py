"""Qualify the loudness rules of Auto Voice Setup against its final headroom check.

Subcommands:
  run     For each simulated user (``evaluate_voice_setup.py``), run
          ``analyze_voice_setup`` on the setup capture once per arm of the rule
          (``--rule``). Then emulate the dialog's verification of each proposal on
          the start of the separate evaluation capture, including its automatic
          reductions, and render the whole evaluation capture through the
          proposal. Both go through the full chain on the raw captures, as
          ``analyze_voice_setup``'s own final check does, so automatic makeup
          receives the frontend's activity evidence and backs off against the
          limiter as it does in the app.
  report  Apply the rule recorded in the rows written by ``run``; the ``narrow``
          rule also reports per split.

Arms:
  released        The requested loudness target as it is, against the check as
                  released: peaks keep 1 dB unused below the limiter.
  fitted          The target lowered by whole decibels until that check passes.
  allowed_narrow  The same, with the limiter allowed to take up to 3 dB off the
                  loudest peaks, in the final check, the fit and verification,
                  only when the limiter is enabled, the proposal uses automatic
                  makeup and the voice is not too hot (``input_too_hot`` in
                  ``analyze_voice_setup`` is false). Elsewhere it is ``fitted``.
  The arms differ only through ``voice_setup.LIMITER_ALLOWANCE_DB`` (0.0 is the
  released rule) and, for ``released``, the loudness fit being skipped. Rule
  ``fit`` compares ``released`` with ``fitted``; rule ``narrow`` runs all three
  and compares ``allowed_narrow`` with ``fitted``.

Rule ``fit``

Why: a natural voice peaks about 20 dB above its loudness. Raising it to the
default -18 LUFS with light compression leaves the limiter catching peaks, the
final headroom check fails, and verification, which lowers the target one
decibel at a time and stops after three steps, cannot recover.

Rule (fixed on 2026-10-12, before any test-split row existed). Users whose
Auto-EQ stage produced no settings in either arm are counted and left out. The
fit is adopted when all of these hold on the test split:

  fits_headroom        Every proposal the fit changes passes the existing
                       headroom check.
  not_needlessly_quiet At least 95% of the proposals the fit lowered and made
                       safe keep no more than 1.5 dB beyond the headroom the
                       check requires.
  leaves_safe_alone    Every proposal the released analysis already made safe
                       is unchanged.
  never_less_safe      No fitted proposal has more limiter gain reduction than
                       the released one (0.1 dB tolerance).
  verification_no_harm No user whose released proposal verification accepts
                       loses that with the fitted proposal, and no user needs
                       more automatic reductions.
  benefit              More fitted proposals than released ones pass the
                       headroom check, and verification needs no more
                       reductions in total.

The development split showed that a second take can press the limiter with no
makeup at all, which no loudness target can fix, so the rule does not ask for an
absolute verification pass rate.

Rule ``narrow`` (fixed on 2026-10-12 by the study's owner before any EARS row
existed)

Users: the EARS root, splits `fit`, `calibrate` and `test` (the harness partitions the 24 speakers into three groups of 8; the names are only partitions here), indices 0-15 in each: 48 users, every speaker twice in different rooms, noise and levels. Users whose Auto-EQ stage produced no settings in any arm are counted and left out. The narrow allowance is adopted only if ALL of these hold, comparing `allowed_narrow` with `fitted`:
- `intelligibility_kept`: no user's ESTOI on the evaluation capture, against the reverberant clean speech, is more than 0.01 lower than with the fitted proposal.
- `below_app_warning`: no proposal that passes the final check reaches the app's "limiting hard" warning on the evaluation capture: 6 dB of limiter or 3 dB of true-peak limiter gain reduction. The gate is about harm the candidate causes: a proposal does not count when the same user's `fitted` proposal has the identical settings digest, passes its own final check and reaches the warning too (a baseline exceedance, listed separately). The strict reading, every such proposal including those, is reported beside it.
- `never_quieter`: no proposal that passed the released check is quieter than the fitted one.
- `lowers_only_to_pass`: every proposal quieter than the fitted one passes the final check.
- `verification_no_harm`: no user whose fitted proposal verification accepts loses that.
- `strict_where_not_raised`: every user whose input is too hot, or whose proposal uses manual makeup, gets exactly the fitted proposal: identical settings and identical decisions.
- `benefit`: more proposals pass the final check than with the fitted arm, no fewer are recommended, and verification gives up on limiter pressure no more often.
If any gate fails, the candidate is rejected and the incumbent stays; there is no second attempt on these speakers.

Implementation notes. ``never_quieter`` reads "passed the released check" as the
fitted proposal passing it (the fitted arm is judged by the released check).
"Reaches the warning" is >= 6 dB of limiter or >= 3 dB of true-peak limiter gain
reduction. The identical settings digest covers the EQ, gate, suppressor,
de-esser, compressor and limiter settings without their wall-clock fields.
Gates compare the arms user by user on the users with settings in all arms. The
overall column decides; the per-split columns are descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_voice_setup as voice_setup_eval
from _eval_common import estoi

REPO_ROOT = voice_setup_eval.REPO_ROOT
SAMPLE_RATE = voice_setup_eval.SAMPLE_RATE
ARMS = ("released", "fitted", "allowed_narrow")
# The dialog's limits: _MAX_VERIFICATION_ADJUSTMENTS and _LIMITER_PRESSURE_STEP_DB.
MAX_ADJUSTMENTS = 3
LIMITER_STEP_DB = 1.0
SLACK_DB = 1.5
ESTOI_TOLERANCE = voice_setup_eval.TOLERANCES["estoi"][1]
# output_health_text in mic_eq/ui/health.py: limiter, true-peak limiter.
APP_LIMITING_HARD_DB = (6.0, 3.0)
# analyze_voice_setup's input_too_hot: a voice this loud gets no automatic makeup.
TOO_HOT_LUFS = -24.0
SECTIONS = ("eq_settings", "gate_settings", "suppressor_settings", "deesser_settings",
            "compressor_settings", "limiter_settings")
SOURCES = (
    "python/tools/evaluate_voice_setup_loudness.py",
    "python/tools/evaluate_voice_setup.py",
    "python/tools/evaluate_gate_corpus.py",
    "python/tools/_eval_common.py",
    "python/mic_eq/analysis/voice_setup.py",
    "python/mic_eq/analysis/auto_eq.py",
    "python/mic_eq/analysis/noise_reference.py",
    "python/mic_eq/analysis/joint_tuning.py",
    "python/mic_eq/analysis/compressor_calibration.py",
    "python/mic_eq/analysis/auto_eq_parts/headroom.py",
    "python/mic_eq/ui/calibration_support.py",
)
EXTENSION = "python/mic_eq/mic_eq_core.cp313-win_amd64.pyd"


@contextmanager
def _arm_rules(arm: str) -> Iterator[None]:
    """Give ``voice_setup`` the loudness rules of one arm."""
    from mic_eq.analysis import voice_setup

    names = ("_fit_loudness_to_headroom", "LIMITER_ALLOWANCE_DB")
    saved = {name: getattr(voice_setup, name) for name in names}
    if arm != "allowed_narrow":
        voice_setup.LIMITER_ALLOWANCE_DB = 0.0  # the released rule everywhere
    if arm == "released":
        voice_setup._fit_loudness_to_headroom = (
            lambda compressor, simulation, render, allowance_db=0.0: (dict(compressor), simulation))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(voice_setup, name, value)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _without_timing(value: Any) -> Any:
    """Drop wall-clock fields (keys containing "runtime"), which differ between any two runs."""
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items() if "runtime" not in str(key)}
    if isinstance(value, (list, tuple)):
        return [_without_timing(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_without_timing(value), sort_keys=True, default=_plain).encode()).hexdigest()


def _applied_eq(eq_settings: dict[str, Any]) -> dict[str, Any]:
    """The typed EQ that ``analyze_voice_setup``'s final check renders: candidate bands over the default EQ."""
    from mic_eq.config import EQSettings, build_eq_candidate_settings

    if "bands" in eq_settings:
        return dict(eq_settings)
    typed = build_eq_candidate_settings(
        EQSettings(), eq_settings["band_freqs"], eq_settings["band_gains"], eq_settings["band_qs"],
        layer="correction", enabled=bool(eq_settings.get("enabled", True)),
    ).to_dict()
    return {**eq_settings, **{key: typed[key] for key in ("schema_version", "enabled", "bands", "layers") if key in typed}}


def _chain(result: dict[str, Any]) -> dict[str, Any]:
    """The full chain of a proposal, as the dialog snapshots it for verification."""
    return {
        "gate": dict(result["gate_settings"]), "rnnoise": dict(result["suppressor_settings"]),
        "deesser": dict(result["deesser_settings"]), "compressor": dict(result["compressor_settings"]),
        "limiter": dict(result["limiter_settings"]), "full_chain": True, "input_pre_filtered": False,
        "input_cleanup_mode": "off", "processing_mode": "normal",
        "vad_available": bool(result["diagnostics"]["vad_available"]),
    }


def _reduce(result: dict[str, Any], targets: list[str]) -> None:
    """The dialog's ``_apply_verification_reduction``."""
    compressor, deesser = dict(result["compressor_settings"]), dict(result["deesser_settings"])
    if "compressor" in targets:
        compressor["threshold_db"] = min(0.0, float(compressor["threshold_db"]) + 2.0)
        compressor["ratio"] = max(1.0, float(compressor["ratio"]) * 0.9)
    if "limiter" in targets:
        if compressor.get("auto_makeup_enabled"):
            compressor["target_lufs"] = max(-24.0, float(compressor["target_lufs"]) - LIMITER_STEP_DB)
        else:
            compressor["makeup_gain_db"] = max(0.0, float(compressor["makeup_gain_db"]) - LIMITER_STEP_DB)
    if "deesser" in targets:
        deesser["auto_amount"] = float(deesser.get("auto_amount", 0.0)) * 0.85
    result["compressor_settings"], result["deesser_settings"] = compressor, deesser


def _verify(result: dict[str, Any], noise: np.ndarray, setup: np.ndarray, second: np.ndarray,
            raw_noise: np.ndarray, raw_second: np.ndarray) -> dict[str, Any]:
    """The dialog's verification: raw captures through the full chain, up to three reductions."""
    from mic_eq.analysis.voice_setup import validate_voice_setup_verification

    result = deepcopy(result)
    result["eq_settings"] = _applied_eq(result["eq_settings"])
    adjustments = 0
    while True:
        # The dialog re-reads the chain from the applied controls after every reduction.
        result["verification_chain_settings"] = _chain(result)
        outcome = validate_voice_setup_verification(
            noise, setup, second, SAMPLE_RATE, result, "broadcast",
            raw_noise_audio=raw_noise, raw_verification_speech_audio=raw_second,
        )
        targets: list[str] = [str(target) for target in outcome.get("reduction_targets") or ()]
        if outcome["decision"] != "reduce" or adjustments >= MAX_ADJUSTMENTS:
            break
        before = deepcopy((result["compressor_settings"], result["deesser_settings"]))
        _reduce(result, targets)
        if before == (result["compressor_settings"], result["deesser_settings"]):
            break
        adjustments += 1
    decision = "gave_up" if outcome["decision"] == "reduce" else outcome["decision"]
    return {"decision": decision, "adjustments": adjustments, "limiter_pressure": "limiter" in targets,
            "limiter_pressure_db": outcome.get("limiter_pressure_db"), "reason_codes": outcome.get("reason_codes"),
            "full_chain": outcome.get("full_chain")}


def _render(result: dict[str, Any], capture: np.ndarray) -> dict[str, Any]:
    from mic_eq.analysis.auto_eq import simulate_candidate_chain

    return simulate_candidate_chain(
        np.ascontiguousarray(capture, dtype=np.float32), SAMPLE_RATE, _applied_eq(result["eq_settings"]),
        {**_chain(result), "return_output_audio": True},
    )


def _listen(result: dict[str, Any], raw_evaluation: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """Score the raw evaluation capture rendered through the proposal's full chain."""
    rendered = _render(result, raw_evaluation)
    return {"estoi": estoi(clean, np.asarray(rendered["output_audio"], np.float64), SAMPLE_RATE),
            **{name: float(rendered[name]) for name in ("limiter_gain_reduction_db", "true_peak_limiter_gain_reduction_db")}}


def _final_check_gap(result: dict[str, Any], raw_setup: np.ndarray) -> float | None:
    """How far this module's render of the setup capture is from the analysis' own final check."""
    from mic_eq.analysis.auto_eq_parts.headroom import _NATIVE_SAFETY_METRICS

    recorded = result["diagnostics"].get("offline_validation") or {}
    names = (*_NATIVE_SAFETY_METRICS, "output_rms_db")
    if any(name not in recorded for name in names):
        return None
    rendered = _render(result, raw_setup)
    return max(abs(float(rendered[name]) - float(recorded[name])) for name in names)


def _arm(result: dict[str, Any], noise: np.ndarray, setup: np.ndarray, second: np.ndarray,
         evaluation: np.ndarray, clean: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    validation = diagnostics.get("offline_validation") or {}
    compressor = result["compressor_settings"]
    auto_makeup = bool(compressor.get("auto_makeup_enabled"))
    key = "target_lufs" if auto_makeup else "makeup_gain_db"
    short_term_lufs = float(diagnostics["short_term_lufs"])
    row: dict[str, Any] = {
        "complete": result.get("eq_settings") is not None, "eq_error": result.get("eq_error"),
        "auto_makeup": auto_makeup, "loudness_key": key,
        "loudness": float(compressor[key]), "lowered_db": float(diagnostics["loudness_lowered_db"]),
        "short_term_lufs": short_term_lufs,
        # analyze_voice_setup's own condition; the allowance applies only where it is false.
        "input_too_hot": bool(short_term_lufs >= TOO_HOT_LUFS or not auto_makeup),
        "limiter_enabled": bool(result["limiter_settings"].get("enabled", True)),
        "limiter_allowance_db": diagnostics.get("limiter_allowance_db"),
        "headroom_safe": bool(diagnostics.get("offline_headroom_safe")),
        "offline_validation_passed": bool(diagnostics.get("offline_validation_passed")),
        "apply_recommended": bool(diagnostics["apply_recommended"]),
        "reasons": list(diagnostics["uncertainty_reasons"]),
        "settings_sha256": {name: _digest(result.get(name)) for name in SECTIONS},
        **{name: validation.get(name) for name in (
            "pre_limiter_true_peak_headroom_db", "limiter_gain_reduction_db",
            "true_peak_limiter_gain_reduction_db", "output_true_peak_db", "output_rms_db")},
    }
    if row["complete"]:
        row["verification"] = _verify(result, noise, setup, second, raw["noise"], raw["second"])
        row["listen"] = _listen(result, raw["evaluation"], clean)
        # Integrity of this harness, not a gate: the render matches the analysis' own final check.
        row["final_check_render_gap"] = _final_check_gap(result, raw["setup"])
    return row


def evaluate_user(job: tuple[str, str, int, str], *, arms: tuple[str, ...]) -> dict[str, Any]:
    root_text, split_name, index, model = job
    import mic_eq
    from mic_eq.analysis import voice_setup
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    started = time.perf_counter()
    root = Path(root_text)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    simulated = voice_setup_eval.user(root, manifest, split_name, index)
    raw_noise, raw_setup, raw_evaluation = (
        simulated[name] for name in ("noise_capture", "setup_capture", "evaluation_capture"))
    # Verification hears a take as long as the setup one; the listener, the whole capture.
    raw_second = raw_evaluation[: raw_setup.size]
    noise, setup, second = (filtered_capture_for_analysis(audio, SAMPLE_RATE)
                            for audio in (raw_noise, raw_setup, raw_second))
    clean = filtered_capture_for_analysis(simulated["evaluation_clean"], SAMPLE_RATE).astype(np.float64)
    raw = {"noise": raw_noise, "setup": raw_setup, "second": raw_second, "evaluation": raw_evaluation}
    row: dict[str, Any] = {key: simulated[key] for key in ("split", "index", "speaker", "environment", "snr_db", "level_dbfs")}
    row["model"] = model
    row["provenance"] = {"mic_eq": mic_eq.__file__, "evaluator": __file__}
    row["elapsed_s"] = {}
    for arm in arms:
        arm_started = time.perf_counter()
        with _arm_rules(arm):
            result = voice_setup.analyze_voice_setup(
                noise, setup, SAMPLE_RATE, "broadcast", noise_model=model, suppressor_strength=1.0,
                incumbent_settings={"rnnoise": {"enabled": True, "strength": 1.0, "model": model}},
                raw_noise_audio=raw_noise, raw_speech_audio=raw_setup,
            )
            row[arm] = _arm(result, noise, setup, second, raw_evaluation, clean, raw)
        row["elapsed_s"][arm] = time.perf_counter() - arm_started
    row["elapsed_s"]["user"] = time.perf_counter() - started
    return row


def fit_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from mic_eq.analysis.auto_eq_parts.headroom import HEADROOM_TARGET_DB

    scored = [row for row in rows if row["released"]["complete"] and row["fitted"]["complete"]]
    pairs = [(row["released"], row["fitted"]) for row in scored]
    lowered_safe = [new for _, new in pairs if new["lowered_db"] > 0.0 and new["headroom_safe"]]
    tight = [new for new in lowered_safe if new["pre_limiter_true_peak_headroom_db"] <= HEADROOM_TARGET_DB + SLACK_DB]
    accepted = {arm: sum(row[arm]["verification"]["decision"] == "accept" for row in scored) for arm in ("released", "fitted")}
    gave_up = {arm: sum(row[arm]["verification"]["decision"] == "gave_up" and row[arm]["verification"]["limiter_pressure"]
                        for row in scored) for arm in ("released", "fitted")}
    reductions = {arm: sum(row[arm]["verification"]["adjustments"] for row in scored) for arm in ("released", "fitted")}
    lowered = sorted(new["lowered_db"] for _, new in pairs)
    metrics = {
        "users": len(rows), "scored_users": len(scored),
        "no_eq_settings": {arm: sum(not row[arm]["complete"] for row in rows) for arm in ("released", "fitted")},
        "headroom_safe": {arm: sum(row[arm]["headroom_safe"] for row in scored) for arm in ("released", "fitted")},
        "apply_recommended": {arm: sum(row[arm]["apply_recommended"] for row in scored) for arm in ("released", "fitted")},
        "verification_accepted": accepted, "verification_gave_up_on_limiter_pressure": gave_up,
        "verification_reductions": reductions,
        "fitted_still_unsafe": sum(not new["headroom_safe"] for _, new in pairs),
        # No makeup keeps headroom: the analysis then asks for a lower input level.
        "fitted_still_unsafe_input_too_hot": sum(
            not new["headroom_safe"] and any("too little headroom" in reason for reason in new["reasons"])
            for _, new in pairs),
        "lowered_users": sum(value > 0.0 for value in lowered),
        "lowered_db_median": float(np.median(lowered)) if lowered else None,
        "lowered_db_p90": float(np.percentile(lowered, 90)) if lowered else None,
        "lowered_db_max": lowered[-1] if lowered else None,
        "released_limiter_gain_reduction_db_median": float(np.median([old["limiter_gain_reduction_db"] for old, _ in pairs])) if pairs else None,
        "fitted_limiter_gain_reduction_db_max": max((new["limiter_gain_reduction_db"] for _, new in pairs), default=None),
    }
    gates = {
        "fits_headroom": all(new["headroom_safe"] for _, new in pairs if new["lowered_db"] > 0.0),
        "not_needlessly_quiet": len(tight) >= 0.95 * len(lowered_safe),
        "leaves_safe_alone": all(new["lowered_db"] == 0.0 and new["loudness"] == old["loudness"]
                                 for old, new in pairs if old["headroom_safe"]),
        "never_less_safe": all(new["limiter_gain_reduction_db"] <= old["limiter_gain_reduction_db"] + 0.1 for old, new in pairs),
        "verification_no_harm": all(
            row["fitted"]["verification"]["adjustments"] <= row["released"]["verification"]["adjustments"]
            and (row["fitted"]["verification"]["decision"] == "accept"
                 or row["released"]["verification"]["decision"] != "accept")
            for row in scored),
        "benefit": metrics["headroom_safe"]["fitted"] > metrics["headroom_safe"]["released"]
        and reductions["fitted"] <= reductions["released"],
    }
    return {"metrics": metrics, "gates": gates, "passed": bool(scored) and all(gates.values())}


def _counts(scored: list[dict[str, Any]], value: Any) -> dict[str, int]:
    return {arm: sum(bool(value(row[arm])) for row in scored) for arm in ARMS}


def _reaches_warning(arm: dict[str, Any]) -> bool:
    """The app's "limiting hard" warning on the evaluation capture."""
    listen = arm["listen"]
    return bool(listen["limiter_gain_reduction_db"] >= APP_LIMITING_HARD_DB[0]
                or listen["true_peak_limiter_gain_reduction_db"] >= APP_LIMITING_HARD_DB[1])


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "min": float(min(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(max(values))}


def _exceedance(row: dict[str, Any]) -> dict[str, Any]:
    return {"split": row["split"], "index": row["index"], "speaker": row["speaker"],
            **{arm: {"limiter_gain_reduction_db": row[arm]["listen"]["limiter_gain_reduction_db"],
                     "true_peak_limiter_gain_reduction_db": row[arm]["listen"]["true_peak_limiter_gain_reduction_db"],
                     "headroom_safe": row[arm]["headroom_safe"]} for arm in ("fitted", "allowed_narrow")}}


def _decisions(arm: dict[str, Any]) -> tuple[Any, ...]:
    verification = arm["verification"]
    return (arm["apply_recommended"], arm["headroom_safe"], arm["offline_validation_passed"], arm["reasons"],
            verification["decision"], verification["adjustments"], verification["limiter_pressure"],
            verification["reason_codes"])


def _narrow_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if all(row[arm]["complete"] for arm in ARMS)]
    left_out = [{"split": row["split"], "index": row["index"],
                 "arms_without_settings": [arm for arm in ARMS if not row[arm]["complete"]]}
                for row in rows if row not in scored]
    pairs = [(row["fitted"], row["allowed_narrow"]) for row in scored]
    estoi_change = [new["listen"]["estoi"] - old["listen"]["estoi"] for old, new in pairs]
    passing = [new for _, new in pairs if new["headroom_safe"]]
    newly = [new for old, new in pairs if new["headroom_safe"] and not old["headroom_safe"]]
    not_raised = [row for row in scored if row["fitted"]["input_too_hot"]]
    counts = {
        "headroom_safe": _counts(scored, lambda arm: arm["headroom_safe"]),
        "apply_recommended": _counts(scored, lambda arm: arm["apply_recommended"]),
        "verification_accepted": _counts(scored, lambda arm: arm["verification"]["decision"] == "accept"),
        "verification_gave_up_on_limiter_pressure": _counts(
            scored, lambda arm: arm["verification"]["decision"] == "gave_up" and arm["verification"]["limiter_pressure"]),
    }
    gains = sorted(new["listen"]["limiter_gain_reduction_db"] for new in newly)
    true_peak_gains = sorted(new["listen"]["true_peak_limiter_gain_reduction_db"] for new in newly)
    # Strict reading: every allowed_narrow proposal that passes its final check and reaches the warning.
    # Attributed reading (decides): not those where fitted has the identical settings, passes and reaches it too.
    over = [row for row in scored if row["allowed_narrow"]["headroom_safe"] and _reaches_warning(row["allowed_narrow"])]
    baseline = [row for row in over
                if row["fitted"]["settings_sha256"] == row["allowed_narrow"]["settings_sha256"]
                and row["fitted"]["headroom_safe"] and _reaches_warning(row["fitted"])]
    baseline_ids = {id(row) for row in baseline}
    caused = [row for row in over if id(row) not in baseline_ids]
    warning_readings = {
        "strict_passed": not over, "attributed_passed": not caused,
        "proposals_reaching_warning": len(over),
        "counted_against_the_candidate": [_exceedance(row) for row in caused],
        "baseline_exceedances": [_exceedance(row) for row in baseline],
    }
    metrics: dict[str, Any] = {
        "users": len(rows), "scored_users": len(scored), "left_out": left_out,
        "speakers": len({row["speaker"] for row in rows}),
        "scored_speakers": len({row["speaker"] for row in scored}),
        "no_eq_settings": {arm: sum(not row[arm]["complete"] for row in rows) for arm in ARMS},
        **counts,
        "allowance_applied_users": sum(bool(new["limiter_allowance_db"]) for _, new in pairs),
        "allowance_not_applied": {
            "input_too_hot_with_auto_makeup": sum(old["input_too_hot"] and old["auto_makeup"] for old, _ in pairs),
            "manual_makeup": sum(not old["auto_makeup"] for old, _ in pairs),
            "limiter_disabled": sum(not old["limiter_enabled"] for old, _ in pairs),
        },
        "below_app_warning_readings": warning_readings,
        "evaluation_limiter_gain_reduction_db_among_passing": {
            arm: _distribution([row[arm]["listen"]["limiter_gain_reduction_db"] for row in scored if row[arm]["headroom_safe"]])
            for arm in ARMS},
        "evaluation_true_peak_limiter_gain_reduction_db_among_passing": {
            arm: _distribution([row[arm]["listen"]["true_peak_limiter_gain_reduction_db"]
                                for row in scored if row[arm]["headroom_safe"]]) for arm in ARMS},
        "newly_passing_users": len(newly),
        "newly_passing_evaluation_limiter_gain_reduction_db": gains,
        "newly_passing_evaluation_true_peak_limiter_gain_reduction_db": true_peak_gains,
        "louder_users": sum(new["loudness"] > old["loudness"] for old, new in pairs),
        "quieter_users": sum(new["loudness"] < old["loudness"] for old, new in pairs),
        "estoi_change_min": min(estoi_change, default=None),
        "estoi_change_mean": float(np.mean(estoi_change)) if estoi_change else None,
        "estoi_change_max": max(estoi_change, default=None),
        "setup_limiter_gain_reduction_db_max": max((new["limiter_gain_reduction_db"] for new in passing), default=None),
        "evaluation_limiter_gain_reduction_db_max": max(
            (new["listen"]["limiter_gain_reduction_db"] for new in passing), default=None),
        "evaluation_true_peak_limiter_gain_reduction_db_max": max(
            (new["listen"]["true_peak_limiter_gain_reduction_db"] for new in passing), default=None),
        "final_check_render_gap_max": max(
            (arm["final_check_render_gap"] for row in scored for arm in (row[a] for a in ARMS)
             if arm["final_check_render_gap"] is not None), default=None),
        "strict_users": len(not_raised),
    }
    gates = {
        "intelligibility_kept": all(change >= -ESTOI_TOLERANCE for change in estoi_change),
        "below_app_warning": not caused,  # the attributed reading decides; see below_app_warning_readings
        "never_quieter": all(new["loudness"] >= old["loudness"] for old, new in pairs if old["headroom_safe"]),
        "lowers_only_to_pass": all(new["headroom_safe"] for old, new in pairs if new["loudness"] < old["loudness"]),
        "verification_no_harm": all(new["verification"]["decision"] == "accept"
                                    or old["verification"]["decision"] != "accept" for old, new in pairs),
        "strict_where_not_raised": all(
            row["fitted"]["settings_sha256"] == row["allowed_narrow"]["settings_sha256"]
            and _decisions(row["fitted"]) == _decisions(row["allowed_narrow"])
            and not row["allowed_narrow"]["limiter_allowance_db"] for row in not_raised),
        "benefit": counts["headroom_safe"]["allowed_narrow"] > counts["headroom_safe"]["fitted"]
        and counts["apply_recommended"]["allowed_narrow"] >= counts["apply_recommended"]["fitted"]
        and counts["verification_gave_up_on_limiter_pressure"]["allowed_narrow"]
        <= counts["verification_gave_up_on_limiter_pressure"]["fitted"],
    }
    return {"metrics": metrics, "gates": gates, "passed": bool(scored) and all(gates.values())}


def narrow_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [name for name in voice_setup_eval.SPLITS if any(row["split"] == name for row in rows)]
    return {**_narrow_summary(rows),
            "partitions": {name: _narrow_summary([row for row in rows if row["split"] == name]) for name in names}}


RULES: dict[str, dict[str, Any]] = {
    "fit": {
        "arms": ARMS[:2], "report": fit_report,
        "experiment": "Auto Voice Setup: fit the loudness target to the capture's headroom",
        "configuration": {"slack_db": SLACK_DB},
        "clean_preservation": "The fit only lowers makeup; limiter gain reduction is gated not to rise.",
    },
    "narrow": {
        "arms": ARMS, "report": narrow_report,
        "experiment": "Auto Voice Setup: let the limiter catch peaks, only where automatic makeup backs off",
        "configuration": {"estoi_tolerance": ESTOI_TOLERANCE, "app_limiting_hard_db": APP_LIMITING_HARD_DB,
                          "too_hot_lufs": TOO_HOT_LUFS},
        "clean_preservation": "ESTOI of each user's raw evaluation capture rendered through the proposal's full "
                              "chain, against its reverberant clean speech, with the narrow allowance against "
                              "without it.",
    },
}


def _limitations(rule: str, metrics: dict[str, Any]) -> list[str]:
    limitations = [
        "Simulated users (the corpus above): speech in synthetic rooms with DEMAND noise, not microphone captures.",
        "Verification hears the first 10 s of the evaluation capture; the listening render the whole 20 s.",
    ]
    if rule == "fit":
        return limitations + [
            "Both arms are judged by the released final check; the limiter allowance of rule 'narrow' is "
            "qualified on top of the fit, separately.",
            "The fit leaves a proposal as released when no allowed loudness passes the headroom check, which "
            "happens when automatic makeup's gain at the loudest peaks does not follow the loudness target.",
            "The rule's verification clause was revised on the development split before any test row existed.",
            "The VoiceBank test split was first scored on 2026-10-12 with the earlier evaluator, which verified "
            "downstream of the frontend; the final-check figures were identical in that run and this one.",
        ]
    return limitations + [
        f"{metrics['scored_users']} of {metrics['users']} users are scored, from {metrics['scored_speakers']} of "
        f"{metrics['speakers']} speakers; the other {len(metrics['left_out'])} get no Auto-EQ settings from the "
        "Auto-EQ stage, which this change does not touch, and are left out.",
        f"The largest limiter gain reduction among passing proposals on the evaluation capture is "
        f"{metrics['evaluation_limiter_gain_reduction_db_max']:.2f} dB, against the app's "
        f"{APP_LIMITING_HARD_DB[0]:g} dB warning.",
        "ESTOI scores intelligibility, not the timbre of limiting; limiter gain reduction is the largest value "
        "over a capture, not how often the limiter moves.",
        "The 3 dB constant was chosen earlier on the VoiceBank fit split; no shipped value was fitted on EARS speech.",
        "The compressor calibration search still judges candidates by the released headroom check.",
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_plain) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--rule", choices=tuple(RULES), help="run: the rule, which fixes the arms")
    parser.add_argument("--root", type=Path, default=voice_setup_eval.DEFAULT_ROOT, help="run: the corpus root")
    parser.add_argument("--splits", default=",".join(voice_setup_eval.SPLITS), help="run: splits, comma-separated")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--users", type=int, default=16, help="run: users per split")
    parser.add_argument("--model", default="rnnoise")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows", type=Path, help="report: rows written by run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.rule is None:
            parser.error("run needs --rule")
        arms = RULES[args.rule]["arms"]
        splits = args.splits.split(",")
        jobs = [(str(args.root), split, index, args.model)
                for split in splits for index in range(args.first, args.first + args.users)]
        origin = importlib.util.find_spec("mic_eq")
        extension = Path(origin.origin).parent / Path(EXTENSION).name if origin and origin.origin else None
        print("mic_eq:", origin.origin if origin else None, "| evaluator:", __file__, "| VAD_MODEL_PATH:",
              os.environ.get("VAD_MODEL_PATH"), flush=True)
        partial_rows = args.output.with_suffix(".run.partial.jsonl")
        rows = voice_setup_eval.run(partial(evaluate_user, arms=arms), jobs, min(args.workers, 4), partial_rows)
        vad = os.environ.get("VAD_MODEL_PATH")
        result: dict[str, Any] = {
            "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rule": args.rule, "root": str(args.root), "splits": splits, "arms": arms,
            "identity": {"mic_eq": origin.origin if origin else None, "evaluator": __file__,
                         "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()
                         if extension is not None and extension.exists() else None,
                         "vad_model_path": vad,
                         "vad_model_sha256": hashlib.sha256(Path(vad).read_bytes()).hexdigest() if vad else None},
            "rows": rows,
        }
        _write(args.output, result)
        partial_rows.unlink()
        sys.stdout.write(f"{len(rows)} rows\n")
        return 0
    source = json.loads(args.rows.read_text(encoding="utf-8"))
    rule = source["rule"]
    summary = RULES[rule]["report"](source["rows"])
    root = Path(source["root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = source["identity"]
    rows_path = args.rows.resolve()
    result = {
        "schema_version": 2, "audible_change": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": RULES[rule]["experiment"], "rule": rule,
        "predefined_rule": f"Rule '{rule}' in the docstring of python/tools/evaluate_voice_setup_loudness.py.",
        "splits": source["splits"],
        "user_indices": {name: [row["index"] for row in source["rows"] if row["split"] == name] for name in source["splits"]},
        "retained": summary["passed"], **summary,
        "source_sha256": {name: _sha256(REPO_ROOT / name) for name in SOURCES},
        "implementation_sha256": {EXTENSION: identity["extension_sha256"]},
        "evaluation_contract": {
            "configuration": {"sample_rate": SAMPLE_RATE, "target_preset": "broadcast", "requested_target_lufs": -18.0,
                              "dynamics_intensity": "balanced", "max_verification_adjustments": MAX_ADJUSTMENTS,
                              "arms": RULES[rule]["arms"], **RULES[rule]["configuration"]},
            "asset_hashes": {"corpus_manifest": _sha256(root / "manifest.json"),
                             "vad_model": identity["vad_model_sha256"]},
            "runtime": {"max_p99_frame_seconds": None, "reason": "Offline setup analysis; no realtime code changed."},
            "latency": {"measurement": None, "scope": "The change is to a recommended setting, not the signal path."},
            "clean_preservation": {"method": RULES[rule]["clean_preservation"]},
        },
        "corpus": {key: manifest[key] for key in ("description", "licenses", "redistribution")},
        "retained_evidence": {"directory": rows_path.parent.relative_to(REPO_ROOT).as_posix(),
                              "rows": rows_path.relative_to(REPO_ROOT).as_posix(), "rows_sha256": _sha256(rows_path)},
        "limitations": _limitations(rule, summary["metrics"]),
    }
    _write(args.output, result)
    sys.stdout.write(json.dumps({key: result[key] for key in ("gates", "passed")}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
