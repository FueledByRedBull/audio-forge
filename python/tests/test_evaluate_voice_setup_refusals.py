"""Tests for the Voice Setup refusal study: its judgement, taps, counting, report and guards.

The hand-made world is ``fabricated_rows()``: the 54 study users in 18 speakers, covering every bucket and
verdict. Its first-veto buckets are B1 3, B2 5, B3 2, B4 2, B5 3, C1 2, C2 2, none 34 (rows 18-21 and the
filler) and one analysis failure.
"""

from __future__ import annotations

import ast
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import check_evaluation_hygiene
from tools import evaluate_voice_setup_refusals as ev

voice_setup_eval, loudness, gate_corpus = ev.voice_setup_eval, ev.loudness, ev.gate_corpus
ANALYSIS = Path(__file__).resolve().parents[1] / "mic_eq" / "analysis"
OK = {"estoi": 0.5, "tone_error_db": 2.0, "sibilant_loss_db": 1.0, "deep_pause_db": -10.0, "tail_db": -8.0,
      "onset_db": -1.0, "pause_db": -9.0, "clipped": False, "limiter_gain_reduction_db": 0.5, "true_peak_limiter_gain_reduction_db": 0.1}
IDENTITY = {name: "0" * 64 for name in ("extension_sha256", "manifest_sha256", "vad_model_sha256", "df_dll_sha256")}


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(gate_corpus, "BOOTSTRAP", 200)


def _with(**changes: Any) -> dict[str, Any]:
    return {**OK, **changes}


# ---------------------------------------------------------------------------------------------------------------
# 1. The judgement on hand-made cases


@pytest.mark.parametrize("metric, value, expected", [
    ("estoi", 0.5 - 0.0099, "pass"), ("estoi", 0.5 - 0.0100, "pass"), ("estoi", 0.5 - 0.0101, "fail"), ("estoi", 0.9, "pass"),
    ("tone_error_db", 2.99, "pass"), ("tone_error_db", 3.00, "pass"), ("tone_error_db", 3.01, "fail"),
    ("tone_error_db", 0.0, "pass"),
    ("sibilant_loss_db", 3.99, "pass"), ("sibilant_loss_db", 4.00, "pass"), ("sibilant_loss_db", 4.01, "fail"),
    ("deep_pause_db", -9.01, "pass"), ("deep_pause_db", -9.00, "pass"), ("deep_pause_db", -8.99, "fail"),
    ("deep_pause_db", -20.0, "pass"),
    ("tail_db", -7.01, "pass"), ("tail_db", -7.00, "pass"), ("tail_db", -6.99, "fail"),
    ("onset_db", -1.99, "pass"), ("onset_db", -2.00, "pass"), ("onset_db", -2.01, "fail"), ("onset_db", 3.0, "pass"),
    ("estoi", float("nan"), "unmeasurable"), ("tone_error_db", None, "unmeasurable"),
    ("sibilant_loss_db", None, "unmeasurable"), ("deep_pause_db", float("nan"), "unmeasurable"),
])
def test_boundary_values_pass_and_directions_are_right(metric, value, expected):
    assert ev.judge(_with(**{metric: value}), OK)["clauses"][metric] == expected


@pytest.mark.parametrize("tolerance, value, expected", [
    (1.5, 2.5, "pass"), (1.5, 2.51, "fail"), (3.0, 4.0, "pass"), (6.0, 7.0, "pass"), (6.0, 7.01, "fail"),
])
def test_the_sibilant_sensitivity_readings(tolerance, value, expected):
    judged = ev.judge(_with(sibilant_loss_db=value), OK, sibilant_tolerance=tolerance)
    assert judged["clauses"]["sibilant_loss_db"] == expected


@pytest.mark.parametrize("withheld, current, strict, expected", [
    ({"clipped": True}, {}, False, "fail"),                                   # only the withheld setup clips
    ({"clipped": True}, {"clipped": True}, False, "pass"),                    # both clip: attributed reading
    ({"clipped": True}, {"clipped": True}, True, "fail"),                     # the strict reading beside it
    ({}, {"clipped": True}, False, "pass"),
    ({"limiter_gain_reduction_db": 6.0}, {}, False, "fail"),
    ({"limiter_gain_reduction_db": 5.99}, {}, False, "pass"),
    ({"true_peak_limiter_gain_reduction_db": 3.0}, {}, False, "fail"),
    ({"true_peak_limiter_gain_reduction_db": 2.99}, {}, False, "pass"),
    ({"limiter_gain_reduction_db": 7.0}, {"limiter_gain_reduction_db": 6.5}, False, "pass"),
    ({"limiter_gain_reduction_db": 7.0}, {"limiter_gain_reduction_db": 6.5}, True, "fail"),
    ({"limiter_gain_reduction_db": float("nan")}, {}, False, "unmeasurable"),
])
def test_clipping_and_limiting_count_against_a_setup_the_current_one_does_not_share(withheld, current, strict, expected):
    assert ev.judge(_with(**withheld), _with(**current), strict=strict)["clauses"]["overload"] == expected


def test_verdicts_and_the_two_bounds_of_an_unmeasurable_clause():
    assert ev.judge(OK, OK)["verdict"] == "false"
    two = ev.judge(_with(estoi=0.4, tone_error_db=5.0), OK)
    assert two["verdict"] == "correct" and two["failed"] == ["estoi", "tone_error_db"]
    mixed = ev.judge(_with(tone_error_db=5.0, deep_pause_db=None), OK)
    assert mixed["verdict"] == "correct" and mixed["failed"] == ["tone_error_db"] and mixed["unmeasurable"] == ["deep_pause_db"]
    partial = ev.judge(_with(deep_pause_db=float("nan")), OK)
    assert partial["verdict"] == "partial" and partial["failed"] == []
    assert [ev.false_refusal("partial", bound) for bound in ("lower", "upper")] == [False, True]
    assert [ev.false_refusal("correct", bound) for bound in ("lower", "upper")] == [False, False]
    assert [ev.false_refusal("false", bound) for bound in ("lower", "upper")] == [True, True]


def test_whenever_the_gate_study_says_dominates_no_worse_holds():
    rng = np.random.default_rng(3)
    keys = list(voice_setup_eval.TOLERANCES)
    dominating = 0
    for _ in range(400):
        a, b = ({key: float(rng.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]) * 0.01 ** (key == "estoi"))
                 for key in keys} for _ in range(2))
        if voice_setup_eval.dominates(a, b):
            dominating += 1
            assert ev.no_worse({**OK, **a}, {**OK, **b})
    assert dominating > 20


# ---------------------------------------------------------------------------------------------------------------
# 2. Per-step settings


def _proposal() -> dict[str, Any]:
    current = ev.current_settings(True)
    proposal = deepcopy(current)
    for name in ev.SECTIONS:
        proposal[name] = {**proposal[name], "marker": name}
    return proposal


@pytest.mark.parametrize("step, sections", list(ev.STEPS.items()))
def test_a_step_changes_only_its_own_sections(step, sections):
    current, proposal = ev.current_settings(True), _proposal()
    mixed = ev.hybrid(current, proposal, step)
    for name in ev.SECTIONS:
        source = proposal if name in sections else current
        assert loudness._digest(mixed[name]) == loudness._digest(source[name])
    assert mixed["diagnostics"] == current["diagnostics"]


def test_the_analysis_limiter_is_the_factory_limiter_so_there_is_no_limiter_step():
    from mic_eq.analysis.compressor_calibration import _normalise_limiter_settings

    factory = ev.current_settings(True)["limiter_settings"]
    assert loudness._digest(_normalise_limiter_settings(None)) == loudness._digest(factory)
    assert set(ev.STEPS) == {"tone", "gate_suppression", "deesser", "compressor"}


def test_vacuous_means_the_withheld_sections_change_nothing():
    current = ev.current_settings(True)
    flat = {**current["eq_settings"], "band_gains": [0.0, 0.1, 0.2]}
    assert ev.is_vacuous({"eq_settings": flat}, current, ("eq_settings",))
    assert not ev.is_vacuous({"eq_settings": {**flat, "band_gains": [0.0, 0.25]}}, current, ("eq_settings",))
    assert ev.is_vacuous(deepcopy(current), current, ("gate_settings", "compressor_settings"))
    assert not ev.is_vacuous(_proposal(), current, ("gate_settings",))


# ---------------------------------------------------------------------------------------------------------------
# 3. Level matching and measurement


def _case(cycles: int = 4) -> dict[str, Any]:
    """Pauses of 1.5 s and speech-like noise of 1.5 s, ``cycles`` times."""
    rng = np.random.default_rng(1)
    rate = voice_setup_eval.SAMPLE_RATE
    clean = np.concatenate([np.concatenate([np.zeros(int(1.5 * rate)), 0.1 * rng.standard_normal(int(1.5 * rate))])
                            for _ in range(cycles)])
    labels = np.concatenate([np.concatenate([np.zeros(int(1.5 * rate), bool), np.ones(int(1.5 * rate), bool)])
                             for _ in range(cycles)])
    return {"clean": clean, "noisy": clean + 0.01 * rng.standard_normal(clean.size), "labels": labels}


def test_level_matching_removes_makeup_from_the_pause_and_tail_readings_only():
    case = _case()
    quiet, loud = case["noisy"], case["noisy"] * 10.0 ** (10.0 / 20.0)
    matched_quiet, matched_loud = ev.scored(case, quiet), ev.scored(case, loud)
    for metric in matched_quiet:
        assert matched_loud[metric] == pytest.approx(matched_quiet[metric], abs=1e-6)
    raw_quiet, raw_loud = gate_corpus.score(case, quiet), gate_corpus.score(case, loud)
    for metric in ("deep_pause_db", "tail_db", "pause_db"):
        assert raw_loud[metric] - raw_quiet[metric] == pytest.approx(10.0, abs=1e-6)
    for metric in ("estoi", "onset_db"):
        assert raw_loud[metric] == pytest.approx(raw_quiet[metric], abs=1e-9)
    assert math.isfinite(matched_quiet["deep_pause_db"]) and math.isfinite(matched_quiet["onset_db"])


def test_measure_reads_every_clause_and_the_last_window():
    case = _case()
    ctx = {**case, "noise": case["noisy"][: 2 * voice_setup_eval.SAMPLE_RATE], "raw": case["noisy"]}
    output = case["noisy"].copy()
    rendered = {"output_audio": output, "limiter_gain_reduction_db": 1.5, "true_peak_limiter_gain_reduction_db": 0.25}
    whole = ev.measure(rendered, ctx)
    assert set(whole) >= set(ev.RULES) | {"pause_db", "clipped"}
    assert whole["clipped"] is False and whole["limiter_gain_reduction_db"] == 1.5
    assert whole["tone_error_db"] < 6.0
    output[100] = 1.0
    assert ev.measure(rendered, ctx)["clipped"] is True
    assert ev.measure(rendered, ctx, start=output.size // 2)["clipped"] is False


def test_a_render_that_fails_is_recorded_not_hidden(monkeypatch):
    def fail(*_args):
        raise RuntimeError("native simulator unavailable")

    monkeypatch.setattr(loudness, "_render", fail)
    result = ev.measure_settings({}, {"raw": np.zeros(10)})
    assert result == {"error": "RuntimeError: native simulator unavailable"}
    assert ev.outcome({"objects": {"whole": {"vacuous": False}}, "metrics": {"current": {"whole": OK}, "whole": result}},
                      "whole")["status"] == "unjudgeable"


def _signals(seconds: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    clean = 0.1 * rng.standard_normal(int(seconds * voice_setup_eval.SAMPLE_RATE))
    return clean, np.ones(clean.size, bool)


def test_tone_error_is_zero_for_identity_and_a_flat_gain_and_reads_a_spectral_step():
    clean, labels = _signals()
    assert ev.tone_error_db(clean, clean, labels) == pytest.approx(0.0, abs=1e-6)
    assert ev.tone_error_db(clean, clean * 2.0, labels) == pytest.approx(0.0, abs=1e-6)
    rate = voice_setup_eval.SAMPLE_RATE
    gain = np.where(np.fft.rfftfreq(clean.size, 1.0 / rate) >= 3_000.0, 10.0 ** (6.0 / 20.0), 1.0)
    stepped = np.fft.irfft(np.fft.rfft(clean) * gain, clean.size)
    bins = np.fft.rfftfreq(ev.TONE_FRAME, 1.0 / rate)
    delta = np.where(bins >= 3_000.0, 6.0, 0.0)[(bins >= 80.0) & (bins <= 12_000.0)]
    expected = float(np.sqrt(np.mean(np.square(delta - np.median(delta)))))
    assert expected == pytest.approx(6.0 * math.sqrt((3_000.0 - 80.0) / (12_000.0 - 80.0)), abs=0.03)
    assert ev.tone_error_db(clean, stepped, labels) == pytest.approx(expected, abs=0.15)


def test_tone_error_uses_only_frames_that_are_speech_in_both_signals_and_needs_enough_of_them():
    clean, labels = _signals()
    half = clean.size // 2
    other = clean.copy()
    other[:half] = np.random.default_rng(9).standard_normal(half)  # different audio where nothing is labelled speech
    labels[:half] = False
    assert ev.tone_error_db(clean, other, labels) == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(ev.tone_error_db(clean[:15_000], clean[:15_000], np.ones(15_000, bool)))
    assert math.isnan(ev.tone_error_db(clean, clean, np.zeros(clean.size, bool)))


# ---------------------------------------------------------------------------------------------------------------
# 4. Taps


def test_the_validation_tap_records_a_failure_returns_a_pass_and_restores_the_original():
    from mic_eq.analysis import failure_detection

    original = failure_detection.validate_analysis
    freqs = np.linspace(0.0, 24_000.0, 2049)
    flat = np.zeros(freqs.size)  # no formant structure: too few peaks, noise-like
    assert not original({"band_gains": [0.0] * 10}, flat, freqs).passed
    with ev._taps() as log:
        result = failure_detection.validate_analysis({"band_gains": [0.0] * 10}, flat, freqs)
        assert result.passed and result.reason is None
    assert [entry["passed"] for entry in log["validation"]] == [False]
    assert any("peak_count" in failure for failure in log["validation"][0]["failures"])
    assert log["validation"][0]["reason"] == "Recording too unclear. Please try again."
    assert failure_detection.validate_analysis is original
    with pytest.raises(RuntimeError), ev._taps():
        raise RuntimeError("boom")
    assert failure_detection.validate_analysis is original


def test_the_optimizer_tap_records_the_validated_gains_and_restores_the_original(monkeypatch):
    from mic_eq.analysis.auto_eq_parts import optimizer

    solution = (np.array([1.0, -2.0]), 0.5, 0.25, 1.0, {})
    monkeypatch.setattr(optimizer, "_validate_and_attenuate_solution", lambda *args, **kwargs: solution)
    fake = optimizer._validate_and_attenuate_solution
    with ev._taps() as log:
        assert cast(Any, optimizer)._validate_and_attenuate_solution(1, 2, cancel_check=None) is solution
    assert log["tone"] == [[1.0, -2.0]]
    assert optimizer._validate_and_attenuate_solution is fake
    with pytest.raises(RuntimeError), ev._taps():
        raise RuntimeError("boom")
    assert optimizer._validate_and_attenuate_solution is fake


# ---------------------------------------------------------------------------------------------------------------
# 5. Veto mapping and drift


def _view(**changes: Any) -> dict[str, Any]:
    return {"reasons": [], "weak_capture": False, "b1": False, "tuner_decision": "apply_candidate",
            "verification": "accept", **changes}


@pytest.mark.parametrize("reason, veto", [
    ("compressor p95 gain reduction target missed: measured 1.20 dB, target 3.50 dB (±0.40 dB)", "B3"),
    ("compressor p95 gain reduction target is unavailable", "B3"),
    ("the microphone level leaves too little headroom; lower the input gain by about 3 dB", "B4"),
    ("final full-chain headroom validation did not pass", "B4"),
    ("full-chain gate/suppressor silenced active speech", "B4"),
    ("full-chain validation failed: boom", "B4"),
    ("offline DSP validation did not pass", "B4"),
    ("raw full-chain capture is unavailable for input validation", "B4"),
    ("Auto-EQ abstained from this capture", "B5"),
    ("too little VAD-active speech", "B2"), ("speech-to-noise ratio is weak", "B2"),
    ("too few voiced frames for a stable spectrum", "B2"), ("overall capture quality is low", "B2"),
    ("room-noise capture is not stationary", "B2"),
    ("offline DSP validation could not run, so these settings are advisory", "X"),
])
def test_every_reason_maps_to_its_veto(reason, veto):
    assert ev.reason_veto(reason) == veto
    expected = [] if veto in ("X", "B2") else [veto]  # B2 is the analysis' weak_capture switch, not its wording
    assert ev.fired_vetoes(_view(reasons=[reason])) == expected
    if veto == "B2":
        assert ev.fired_vetoes(_view(reasons=[reason], weak_capture=True)) == ["B2"]


def test_the_switches_fire_their_vetoes_in_order():
    fired = ev.fired_vetoes(_view(reasons=["Auto-EQ abstained from this capture", "compressor p95 gain reduction target is unavailable"],
                                  weak_capture=True, b1=True, tuner_decision="retain_incumbent", verification="rollback"))
    assert fired == ["B1", "B2", "B3", "B5", "B6", "C1"]
    assert ev.fired_vetoes(_view(verification="gave_up")) == ["C2"]
    assert ev.fired_vetoes(_view(verification="retry")) == []


def _literals(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(str(part.value) if isinstance(part, ast.Constant) else "1.5" for part in node.values)]
    if isinstance(node, ast.BinOp):
        return ["".join(_literals(node.left) + _literals(node.right))]
    if isinstance(node, ast.IfExp):
        return _literals(node.body) + _literals(node.orelse)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [text for element in node.elts for text in _literals(element)]
    return []


def _appended(module: str, receiver: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse((ANALYSIS / module).read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("append", "extend")
                and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver):
            found += [text for arg in node.args for text in _literals(arg)]
    return found


def test_every_reason_the_analysis_can_give_has_a_veto():
    from mic_eq.analysis.noise_reference import SILENT_REFERENCE_REASON

    reasons = _appended("voice_setup.py", "uncertainty_reasons") + _appended("noise_reference.py", "reasons")
    reasons.append(SILENT_REFERENCE_REASON)
    for node in ast.walk(ast.parse((ANALYSIS / "compressor_calibration.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == "_compressor_p95_target_status":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Tuple):
                    reasons += _literals(inner.value.elts[1])
    assert len(reasons) > 25
    assert "Auto-EQ abstained from this capture" in reasons and "too little VAD-active speech" in reasons
    unmapped = sorted(reason for reason in reasons if ev.reason_veto(reason) is None)
    assert unmapped == [], f"these reasons have no veto in REASON_VETOES: {unmapped}"


# ---------------------------------------------------------------------------------------------------------------
# 6. Counting


def _users(rows=None) -> list[dict[str, Any]]:
    return ev.assess(rows or ev.fabricated_rows())


def test_first_follows_the_precedence_and_c_only_counts_for_a_recommended_setup():
    assert ev.first_whole({"B4", "B2", "C1"}, False) == "B2"
    assert ev.first_whole({"B5", "B3"}, False) == "B3"
    assert ev.first_whole({"C1", "C2"}, True) == "C1"
    assert ev.first_whole({"C1"}, False) is None
    assert ev.first_whole({"B6"}, True) is None  # B6 is the step track's


def test_the_buckets_add_up_and_each_user_has_exactly_one():
    users = _users()
    assert ev.population(users) == {"B1": 3, "B2": 5, "B3": 2, "B4": 2, "B5": 3, "C1": 2, "C2": 2, "none": 34,
                                    "analysis_failure": 1}
    assert len(users) == 54 and len({u["row"]["index"] for u in users}) == 54


def test_first_and_any_count_differently_and_the_tracks_are_independent():
    users = _users()
    first, anywhere = (ev._table_row(users, "B4", "whole", counting) for counting in ("first", "any"))
    assert (first["fired"], anywhere["fired"]) == (2, 3)  # row 2 is B1 first, B4 also
    assert (first["false"], anywhere["false"]) == (1, 2)
    c1 = ev._table_row(users, "C1", "whole", "any")
    assert (c1["fired"], c1["false"], c1["correct"]) == (3, 1, 2)  # row 17 is B2 first, C1 also fired
    assert ev._table_row(users, "C1", "whole", "first")["fired"] == 2
    tone = ev._table_row(users, "B5", "step", "any")
    assert (tone["fired"], tone["false"], tone["correct"], tone["vacuous"]) == (3, 1, 1, 1)
    tuner = ev._table_row(users, "B6", "step", "any")
    assert (tuner["fired"], tuner["false"], tuner["correct"], tuner["unjudgeable"]) == (3, 1, 1, 1)


def test_every_status_is_reachable_and_attributed():
    users = _users()
    b2 = ev._table_row(users, "B2", "whole", "first")
    assert {s: b2[s] for s in ev.STATUSES} == {"false": 2, "correct": 1, "partial": 1, "definition": 1, "vacuous": 0,
                                                "unjudgeable": 0}
    assert b2["correct_by_clause"] == {"tone_error_db": 1}
    assert ev._table_row(users, "B4", "whole", "first")["correct_by_clause"] == {"overload": 1}
    assert ev._table_row(users, "B3", "whole", "first")["unjudgeable"] == 1
    assert ev._table_row(users, "B5", "whole", "first")["vacuous"] == 1
    lower, upper = (b2[f"false_share_{bound}_bound"][0] for bound in ("lower", "upper"))
    assert (lower, upper) == (pytest.approx(2 / 54), pytest.approx(3 / 54))


def test_a_retry_decides_nothing_and_c_needs_a_decision():
    rows = ev.fabricated_rows()
    retry = rows[21]
    assert retry["verification"]["decision"] == "retry" and retry["fired"] == []
    ev.validate_rows(rows)
    broken = deepcopy(rows)
    broken[21]["fired"] = ["C1"]
    with pytest.raises(AssertionError, match="never a retry"):
        ev.validate_rows(broken)
    assert ev.refusal_report(rows)["retake_costs"] == {"D1": 1}


def test_the_arithmetic_checks_raise_on_a_corrupted_row():
    rows = ev.fabricated_rows()
    for corrupt, message in (
        (lambda r: r.append(deepcopy(r[0])), "twice"),
        (lambda r: r[0].update(fired=["B9"]), "unknown veto"),
        (lambda r: r[53].update(fired=["B3"]), "analysis failure"),
        (lambda r: r[16].update(verification={"decision": "accept"}), "never a retry"),
    ):
        broken = deepcopy(rows)
        corrupt(broken)
        with pytest.raises(AssertionError, match=message):
            ev.assess(broken)


def test_the_counting_arithmetic_raises_when_a_count_does_not_add_up(monkeypatch):
    users = _users()
    monkeypatch.setattr(ev, "outcome", lambda *args, **kwargs: {"status": "weird"})
    with pytest.raises(AssertionError, match="status is unknown"):
        ev.population(users)
    monkeypatch.undo()
    monkeypatch.setattr(ev, "_outcomes", lambda users, veto, track, counting, **kw: [(0, {"status": "false"})] * (2 if counting == "first" else 1))
    with pytest.raises(AssertionError, match="more first than any"):
        ev.population(users)


# ---------------------------------------------------------------------------------------------------------------
# 7. The tuner's withheld step and the verification flow


def _candidate(strength, gate, score, passes, **gates):
    return {"noise_model": "rnnoise", "suppressor_strength": strength, "gate_settings": gate, "score": score,
            "passes": passes, "gates": gates or {"speech": passes}}


def test_the_tuner_step_is_its_own_best_non_incumbent_candidate():
    from mic_eq.config import Preset

    gate = asdict(Preset().gate)
    key = voice_setup_eval._arm_key("rnnoise", 1.0, gate)
    incumbent = _candidate(1.0, gate, 0.1, True)
    slower = _candidate(0.8, {**gate, "release_ms": 300.0}, 0.3, True)
    better = _candidate(0.7, {**gate, "release_ms": 50.0}, 0.2, True)
    failing = _candidate(0.5, {**gate, "release_ms": 20.0}, 0.05, False, speech=False, noise=True)
    chosen = ev.tuner_withheld([incumbent, slower, better, failing], key, {"holdout_evaluated": False})
    assert chosen and chosen["score"] == 0.2 and chosen["cause"] == "margin" and chosen["tested"] == 3
    held = ev.tuner_withheld([incumbent, better], key, {"holdout_evaluated": True})
    assert held and held["cause"] == "holdout"
    none_pass = ev.tuner_withheld([incumbent, failing], key, {})
    assert none_pass and none_pass["score"] == 0.05 and none_pass["failed_gates"] == ["speech"]
    assert none_pass["cause"] == "no_candidate_passed"
    assert ev.tuner_withheld([incumbent], key, {}) is None


def _flow(monkeypatch, outcomes):
    from mic_eq.config import EQ_FREQUENCIES

    steps = iter(outcomes)
    monkeypatch.setattr("mic_eq.analysis.voice_setup.validate_voice_setup_verification", lambda *a, **k: next(steps))
    settings = ev.current_settings(True)
    result = {**{name: settings[name] for name in ev.SECTIONS}, "diagnostics": settings["diagnostics"],
              "eq_settings": {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10}}
    audio = np.zeros(10, np.float32)
    return ev.verify_flow(result, {"noise": audio, "setup": audio, "second": audio, "raw_noise": audio, "raw_second": audio})


def _reduce(p95, targets=("compressor",)):
    return {"decision": "reduce", "reduction_targets": list(targets), "reason_codes": ["compressor_excess"],
            "compressor_gain_reduction_p95_db": p95}


@pytest.mark.parametrize("outcomes, decision, adjustments, ended", [
    ([{"decision": "accept", "reason_codes": ["accepted"]}], "accept", 0, None),
    ([_reduce(5.0), {"decision": "accept", "reason_codes": ["accepted"]}], "accept", 1, None),
    ([_reduce(5.0), _reduce(4.0), _reduce(3.0), _reduce(2.0)], "gave_up", 3, "three_adjustments"),
    ([_reduce(5.0), _reduce(5.0)], "gave_up", 1, "no_help"),
    ([_reduce(5.0, ("limiter",))], "gave_up", 0, "no_change"),  # no automatic makeup to lower
    ([_reduce(5.0, ())], "gave_up", 0, "no_target"),
    ([{"decision": "rollback", "reason_codes": ["target_error"]}], "rollback", 0, None),
    ([{"decision": "retry", "reason_codes": ["verification_delivery_mismatch"]}], "retry", 0, None),
])
def test_the_verification_flow_mirrors_the_dialog(monkeypatch, outcomes, decision, adjustments, ended):
    flow = _flow(monkeypatch, outcomes)
    assert (flow["decision"], flow["adjustments"], flow["ended"]) == (decision, adjustments, ended)
    assert flow["first_decision"] == outcomes[0]["decision"]
    if adjustments:
        assert flow["state"]["compressor_settings"]["threshold_db"] == pytest.approx(-20.0 + 2.0 * adjustments)
    assert ev._verification_row(flow)["retry"] == ("D1" if decision == "retry" else None)


# ---------------------------------------------------------------------------------------------------------------
# 8. The baseline world to the report


def _walk_floats(value: Any, path: str = "$") -> list[tuple[str, float]]:
    if isinstance(value, float):
        return [(path, value)]
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _walk_floats(item, f"{path}.{key}")]
    if isinstance(value, (list, tuple)):
        return [hit for i, item in enumerate(value) for hit in _walk_floats(item, f"{path}[{i}]")]
    return []


def test_the_baseline_world_gives_exact_counts_and_finite_intervals():
    report = ev.refusal_report(ev.fabricated_rows())
    assert (report["users"], report["speakers"]) == (54, 18)
    by = {(r["veto"], r["track"], r["counting"]): r for r in report["vetoes"]}
    assert by[("B1", "whole", "first")]["kind"] == "hard" and by[("B3", "whole", "first")]["kind"] == "soft"
    assert by[("B6", "step", "any")]["kind"] == "step"
    assert report["per_step_breakdown"]["refused_users"] == 7
    assert report["per_step_breakdown"]["no_single_step_worse"] == 0
    assert report["per_step_breakdown"]["steps"]["tone"]["worse_by_clause"] == {"tone_error_db": 7}
    assert report["tuner_step"] == {"fired": 3, "no_candidate": 1, "cause": {"margin": 1, "holdout": 1},
                                    "any_candidate_no_worse_upper_bound": 3}
    assert report["verification_decisions"] == {"accept": 44, "rollback": 3, "gave_up": 2, "retry": 1}
    assert report["sensitivity"]["last_10_s_for_c1_c2"] == {"C1": [1, 1], "C2": [1, 1]}
    assert report["candidates_for_loosening"] == []
    for path, value in _walk_floats(report):
        assert math.isfinite(value), path
    for row in report["vetoes"]:
        mean, low, high = row["false_share_lower_bound"]
        assert 0.0 <= low <= mean <= high <= 1.0
        assert row["false_share_upper_bound"][0] >= mean
    assert {name: sum(group["users"] for group in groups.values()) for name, groups in report["strata"].items()} == {
        "level_check": 54, "rt60": 54, "snr": 54, "gender": 54}


def test_the_trigger_is_a_lower_bound_above_five_percent():
    rows = ev.fabricated_rows()
    for row in rows[23:41]:  # 18 users in 6 speakers whose weak-capture refusal withheld a setup that was no worse
        row.update(fired=["B3"], recommended=False, objects={"whole": {"vacuous": False}},
                   metrics={"current": {"whole": OK}, "whole": {"whole": OK}})
    report = ev.refusal_report(rows)
    assert report["candidates_for_loosening"] == ["B3:whole"]
    b3 = next(r for r in report["vetoes"] if (r["veto"], r["counting"]) == ("B3", "first"))
    assert b3["false_share_lower_bound"][1] > 0.05 and b3["candidate_for_loosening"]


def test_the_report_is_clean_small_and_states_its_rule(tmp_path):
    source = {"rows": ev.fabricated_rows(), "identity": IDENTITY}
    path = tmp_path / "report.json"
    ev._write(path, ev.build_report(source))
    assert check_evaluation_hygiene.validate_report(path) == []
    assert path.stat().st_size < 100_000
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["audible_change"] is False and report["schema_version"] == 2
    assert report["user_indices"] == {"fit": sorted(ev.STUDY_INDICES)}
    assert set(report["source_sha256"]) == set(ev.SOURCES)
    assert "tuner" in " ".join(report["flattering"]) and len(report["limitations"]) >= 5
    assert report["predefined_rule"].endswith("evaluate_voice_setup_refusals.py.")
    assert "Rule (fixed before any study row existed)" in (ev.__doc__ or "")


def test_the_unmeasurable_clauses_are_flagged_over_fifteen_percent():
    users = _users()
    coverage = ev.clause_coverage(users)
    assert coverage["unmeasurable"]["deep_pause_db"] == 1 and coverage["over_15_percent"] == []
    rows = ev.fabricated_rows()
    for row in rows:
        for entry in row["metrics"].values():
            for window in entry.values() if "error" not in entry else ():
                if isinstance(window, dict) and "estoi" in window:
                    window["estoi"] = None
    assert "estoi" in ev.clause_coverage(_users(rows))["over_15_percent"]


# ---------------------------------------------------------------------------------------------------------------
# 9. Guards, the dry run and the commands


def test_the_study_indices_are_unspent_and_map_to_the_intended_18_speakers():
    assert len(ev.STUDY_INDICES) == 54 and min(ev.STUDY_INDICES) >= ev.SPENT_BELOW
    assert {i % 30 for i in ev.STUDY_INDICES} == set(range(12, 30))
    manifest = {"speakers": {f"p{n:03d}": {"gender": "female" if n % 3 else "male"} for n in range(88)},
                "noise": {f"E{n:02d}": {} for n in range(18)}}
    fit = voice_setup_eval.splits(manifest)["fit"]["speakers"]
    chosen = [fit[i % len(fit)] for i in ev.STUDY_INDICES]
    assert len(fit) == 30 and set(chosen) == set(fit[12:]) and all(chosen.count(s) == 3 for s in set(chosen))
    assert ev.parse_indices(ev.DEFAULT_INDICES) == ("fit", list(ev.STUDY_INDICES))


def test_run_refuses_a_spent_user_and_probe_only_the_development_ones():
    ev.check_indices("fit", [60, 149], development=False)
    with pytest.raises(SystemExit, match="spent"):
        ev.check_indices("fit", [59], development=False)
    with pytest.raises(SystemExit, match="spent"):
        ev.check_indices("test", [0], development=False)
    ev.check_indices("calibrate", [0, 11], development=True)
    with pytest.raises(SystemExit, match="development"):
        ev.check_indices("fit", [12], development=True)
    with pytest.raises(SystemExit, match="spent"):
        ev.main(["run", "--indices", "fit:0-3"])
    with pytest.raises(SystemExit, match="--development"):
        ev.main(["probe"])


def test_report_refuses_development_rows(tmp_path):
    rows = ev.fabricated_rows()
    rows[0]["development"] = True
    with pytest.raises(AssertionError, match="spent users"):
        ev.refusal_report(rows)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"development": True, "identity": IDENTITY, "rows": rows}), encoding="utf-8")
    with pytest.raises(SystemExit, match="development"):
        ev.main(["report", "--rows", str(path), "--output", str(tmp_path / "report.json")])
    with pytest.raises(AssertionError, match="54 study users"):
        ev.refusal_report(ev.fabricated_rows()[:53])


def test_report_writes_the_report_where_it_is_told(tmp_path):
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"development": False, "identity": IDENTITY, "rows": ev.fabricated_rows()}), encoding="utf-8")
    assert ev.main(["report", "--rows", str(rows), "--output", str(tmp_path / "out.json")]) == 0
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["metrics"]["users"] == 54


def test_defaults_are_the_new_files_never_a_historical_report():
    assert ev.DEFAULT_REPORT == ev.REPO_ROOT / "evaluation" / "voice-setup-refusals-2026-10-report.json"
    details = ev.REPO_ROOT / "models" / "evaluation-details" / "voice-setup-refusals-2026-10"
    assert details in ev.DEFAULT_ROWS.parents and details in ev.DEFAULT_BOOKKEEPING.parents
    assert details in ev.FROZEN.parents and details in ev.DEVELOPMENT_ROWS.parents


def test_workers_are_at_most_four_and_a_partial_file_resumes_only_its_own_build(tmp_path):
    assert [ev.worker_count(n) for n in (9, 4, 2, 0, -1)] == [4, 4, 2, 1, 1]
    partial = tmp_path / "rows.run.partial.jsonl"
    ev.check_resume(partial, "build-a")  # nothing to resume
    partial.write_text(json.dumps({"split": "fit", "index": 72, "model": "rnnoise", "build": "build-a"}) + "\n", encoding="utf-8")
    ev.check_resume(partial, "build-a")
    with pytest.raises(SystemExit, match="another build"):
        ev.check_resume(partial, "build-b")


def test_the_study_starts_only_when_the_frozen_hashes_still_match(tmp_path, monkeypatch):
    frozen = {"rule": "a", "df.dll": "b"}
    monkeypatch.setattr(ev, "frozen_hashes", lambda root=None: dict(frozen))
    path = tmp_path / "frozen-files.json"
    with pytest.raises(SystemExit, match="freeze"):
        ev.check_frozen(path)
    path.write_text(json.dumps(frozen), encoding="utf-8")
    ev.check_frozen(path)
    path.write_text(json.dumps({**frozen, "df.dll": "changed"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="df.dll"):
        ev.check_frozen(path)


def test_the_capture_vetoes_a1_to_a4_fire_on_hand_made_arrays():
    assert ev.capture_vetoes_probe() == {"A1": True, "A2": True, "A3": True, "A4": True}


def _development_rows() -> list[dict[str, Any]]:
    """The hand-made world as if it were development rows, each with the probes a real run would carry."""
    bad = {"tone_plus_12_db_3k": {"tone_error_db": 9.0}, "sibilance_minus_12_db_6k5": {"sibilant_loss_db": 12.0},
           "suppressor_off": {"deep_pause_db": 0.0}, "gate_threshold_minus_15": {"onset_db": -9.0},
           "gate_release_1000": {"tail_db": 0.0}, "makeup_plus_24": {"clipped": True}}
    rows = ev.fabricated_rows()
    for row in rows:
        row["development"] = True
        row["probes"] = {name: {"whole": _with(**changes)} for name, changes in bad.items()} | {
            "current_again": {"whole": dict(OK)}, "limiter_off": {"whole": dict(OK)}}
        row["tone_cross_check_gap_db"] = 0.2
    return rows


def test_the_dry_run_output_has_counts_and_flags_but_no_rates():
    rows = _development_rows()
    result = ev.bookkeeping(rows)
    assert _walk_floats(result) == []
    text = json.dumps(result).lower()
    for word in ("rate", "share", "interval", "bootstrap", "ci\""):
        assert word not in text
    assert result["development_users"] == 54 and result["arithmetic"] == "checked"
    assert result["constructed_capture_probes"] == {"A1": True, "A2": True, "A3": True, "A4": True}
    assert result["reachability"]["B1"]["fired"] == 3 and result["never_fired"] == []
    assert result["reachability"]["B5_step"]["vacuous"] == 1
    assert result["tone_cross_check"] == {"users": 54, "within_half_db": 54}
    for name, probe in result["probes"].items():
        if "clause" in probe:
            assert probe["tripped"] == probe["measurable"] == 53 and probe["trips_on_most"] and not probe["vacuous"], name
    assert result["probes"]["current_again"] == {"users": 53, "identical_to_current": 53, "clause_failures": {}}
    with pytest.raises(AssertionError, match="spent users"):
        ev.refusal_report(rows)


def test_a_probe_that_never_trips_or_that_trips_for_the_known_good_one_too_is_vacuous():
    rows = _development_rows()
    for row in rows:
        row["probes"]["suppressor_off"] = {"whole": dict(OK)}
        row["probes"]["limiter_off"] = {"whole": _with(tail_db=0.0)}
    probes = ev.bookkeeping(rows)["probes"]
    assert probes["suppressor_off"]["tripped"] == 0 and probes["suppressor_off"]["vacuous"]
    assert probes["tone_plus_12_db_3k"]["vacuous"] is False
    assert probes["gate_release_1000"]["vacuous"] is True
    assert probes["limiter_off"]["identical_to_current"] == 0


def test_the_fabricated_world_is_the_study_users_and_covers_every_veto():
    rows = ev.fabricated_rows()
    assert [row["index"] for row in rows] == list(ev.STUDY_INDICES)
    assert {veto for row in rows for veto in row["fired"]} == set(ev.VETOES)
    assert not any(row["development"] for row in rows)


# ---------------------------------------------------------------------------------------------------------------
# 10. One user through the worker, with the analysis and the renders stubbed


def _stub_worker(monkeypatch, tmp_path, *, b1):
    """Stub the analysis, the native renders and the corpus; return the job for ``evaluate_user``."""
    from mic_eq.analysis import failure_detection, voice_setup
    from mic_eq.analysis.auto_eq_parts import optimizer
    from mic_eq.config import EQ_FREQUENCIES, Preset

    rate = voice_setup_eval.SAMPLE_RATE
    case = _case(2)
    rng = np.random.default_rng(5)
    (tmp_path / "manifest.json").write_text(json.dumps({"speakers": {"p001": {"gender": "male"}}}), encoding="utf-8")
    simulated = {
        "split": "fit", "index": 72, "speaker": "p001", "environment": "DKITCHEN", "snr_db": 12.0, "level_dbfs": -28.0,
        "noise_capture": (0.01 * rng.standard_normal(2 * rate)).astype(np.float32),
        "setup_capture": case["noisy"].astype(np.float32),
        "evaluation_capture": case["noisy"].astype(np.float32), "evaluation_clean": case["clean"].astype(np.float32),
        "labels": case["labels"],
    }
    gate = asdict(Preset().gate)
    candidates = [
        {"noise_model": "rnnoise", "suppressor_strength": 1.0, "gate_settings": gate, "score": 0.1, "passes": True,
         "gates": {"speech": True}},
        {"noise_model": "deepfilter-ll", "suppressor_strength": 0.8, "gate_settings": {**gate, "release_ms": 60.0},
         "score": 0.05, "passes": False, "gates": {"speech": False}},
    ]
    current = ev.current_settings(True)

    def analyze(*_args, **_kwargs):
        if b1:
            failure_detection.validate_analysis({"band_gains": [0.0] * 10}, np.zeros(2049), np.linspace(0.0, 24_000.0, 2049))
        cast(Any, optimizer)._validate_and_attenuate_solution()
        return {
            **{name: deepcopy(current[name]) for name in ev.SECTIONS},
            "compressor_settings": {**current["compressor_settings"], "threshold_db": -25.0},
            "eq_settings": {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10,
                            "enabled": True}, "eq_error": None,
            "diagnostics": {
                "vad_available": True, "weak_capture": False, "apply_recommended": False, "vad_active_duration_s": 5.0,
                "uncertainty_reasons": ["Auto-EQ abstained from this capture"],
                "compressor_auto_makeup_enabled": False, "deesser_enabled": False, "deesser_detection_probability": 0.0,
                "noise_reference_quality": {"status": "usable"},
                "joint_tuning": {"decision": "retain_incumbent", "holdout_evaluated": False, "noise_floor_db": -60.0,
                                 "candidates": candidates}},
        }

    def render(settings, capture):
        output = np.asarray(capture, np.float64)
        if settings["compressor_settings"]["threshold_db"] != -20.0:
            output = np.clip(30.0 * output, -2.0, 2.0)  # the proposed compressor clips; nothing else does
        return {"output_audio": output, "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0}

    monkeypatch.setattr(ev, "configure_environment", lambda: None)
    monkeypatch.setattr(ev, "build_id", lambda: "test")
    monkeypatch.setattr(voice_setup_eval, "user", lambda *args: simulated)
    monkeypatch.setattr(voice_setup_eval, "render_arm", lambda capture, arm, floor: np.asarray(capture, np.float64))
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", analyze)
    monkeypatch.setattr(optimizer, "_validate_and_attenuate_solution",
                        lambda *a, **k: (np.array([3.0, -2.0] + [0.0] * 8), 0.0, 0.0, 1.0, {}))
    monkeypatch.setattr(voice_setup, "validate_voice_setup_verification", lambda *a, **k: {
        "decision": "rollback", "reason_codes": ["target_error"], "reasons": [], "reduction_targets": []})
    monkeypatch.setattr(loudness, "_render", render)
    return str(tmp_path), "fit", 72, "rnnoise"


def test_a_user_goes_through_the_worker_into_a_row_the_report_can_judge(monkeypatch, tmp_path):
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False), development=False)
    json.dumps(row)  # plain JSON types only
    assert row["fired"] == ["B5", "B6", "C1"] and row["b1"] is None and row["recommended"] is False
    assert row["verification"]["decision"] == "rollback"
    assert set(row["objects"]) >= {"whole", "final", "tone", "tuner"} | {f"step:{s}" for s in ev.STEPS}
    assert set(row["metrics"]["current"]) == {"whole", "last10"} and set(row["metrics"]["final"]) == {"whole", "last10"}
    assert row["tuner"]["withheld"]["noise_model"] == "deepfilter-ll" and row["tuner"]["withheld"]["cause"] == "no_candidate_passed"
    assert row["tuner"]["any_candidate_no_worse"] == {"tested": 1, "no_worse": 1}
    assert 0.2 <= row["rt60_s"] <= 0.7 and row["gender"] == "male" and row["level_check"] in ("good", "too_hot", "too_quiet")
    assert row["taps"] == {"validation": 0, "tone": 1} and row["objects"]["tone"]["digests"].keys() == {"eq_settings"}
    status = {name: ev.outcome(row, name)["status"] for name in ("whole", "final", "tone", "tuner")}
    assert status == {"whole": "correct", "final": "correct", "tone": "false", "tuner": "false"}
    assert "overload" in ev.outcome(row, "whole")["failed"]
    # The stub changed only the compressor; a step that changes nothing is vacuous.
    assert [ev.outcome(row, f"step:{s}")["status"] for s in ev.STEPS] == ["vacuous", "vacuous", "vacuous", "correct"]
    assert ev.assess([row])[0]["first"] == "B5"


def test_an_unclear_recording_refusal_keeps_its_proposal_and_never_reaches_verification(monkeypatch, tmp_path):
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=True), development=False)
    assert row["fired"] == ["B1", "B5", "B6"] and row["verification"] is None and "final" not in row["objects"]
    assert row["b1"]["reason"] == "Recording too unclear. Please try again." and ev.assess([row])[0]["first"] == "B1"
    assert ev.outcome(row, "whole", veto="B1")["status"] == "correct"


def test_a_user_whose_auto_eq_failed_before_validation_is_an_analysis_failure(monkeypatch, tmp_path):
    from mic_eq.analysis import voice_setup

    job = _stub_worker(monkeypatch, tmp_path, b1=False)
    failing = {**cast(Any, voice_setup).analyze_voice_setup(), "eq_settings": None, "eq_error": "boom"}
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: failing)
    row = ev.evaluate_user(job, development=False)
    assert row["analysis_failure"] == "boom" and row["fired"] == [] and ev.assess([row])[0]["bucket"] == "analysis_failure"
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: (_ for _ in ()).throw(ValueError("too short")))
    assert ev.evaluate_user(job, development=False)["analysis_failure"] == "too short"


# ---------------------------------------------------------------------------------------------------------------
# 11. The native renders of the current settings and the probes, on a short synthetic signal


def test_the_current_settings_and_every_probe_render_natively_and_the_known_good_ones_change_nothing():
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    rate = voice_setup_eval.SAMPLE_RATE
    case = _case(1)
    rng = np.random.default_rng(4)
    raw = case["noisy"].astype(np.float32)
    ctx = {"raw": raw, "labels": case["labels"], "clean": filtered_capture_for_analysis(case["clean"], rate).astype(np.float64),
           "noisy": filtered_capture_for_analysis(raw, rate).astype(np.float64),
           "noise": filtered_capture_for_analysis((0.01 * rng.standard_normal(2 * rate)).astype(np.float32), rate).astype(np.float64)}
    current = ev.current_settings(True)
    base = ev.measure_settings(current, ctx, ("whole", "last10"))
    assert set(base) == {"whole", "last10"} and base["whole"]["clipped"] is False
    judged = {}
    for name, (target, settings) in ev.probe_settings(current).items():
        reading = ev.measure_settings(settings, ctx)
        assert "error" not in reading, (name, reading)
        judged[name] = ev.judge(reading["whole"], base["whole"])
        if not target:
            assert judged[name]["failed"] == [] and ev._same_reading(reading["whole"]["estoi"], base["whole"]["estoi"]), name
    assert "tone_error_db" in judged["tone_plus_12_db_3k"]["failed"]
    assert "overload" in judged["makeup_plus_24"]["failed"]


def test_the_taps_keep_a_refused_proposal_on_the_real_auto_eq():
    from mic_eq.analysis.auto_eq import analyze_auto_eq

    rate = voice_setup_eval.SAMPLE_RATE
    flat = (0.05 * np.random.default_rng(1).standard_normal(6 * rate)).astype(np.float32)  # no voice structure
    with pytest.raises(ValueError, match="too unclear"):
        analyze_auto_eq(flat, rate, "broadcast")
    with ev._taps() as log:
        eq, _ = analyze_auto_eq(flat, rate, "broadcast")
    assert log["validation"][0]["passed"] is False and "flatness" in " ".join(log["validation"][0]["failures"])
    assert len(log["tone"]) == 1 and len(log["tone"][0]) == len(eq["band_gains"]) == 10


def test_the_verification_flow_runs_against_the_real_validator():
    from mic_eq.config import EQ_FREQUENCIES
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    rate = voice_setup_eval.SAMPLE_RATE
    rng = np.random.default_rng(1)
    speech = np.concatenate([np.concatenate([np.zeros(rate // 2), 0.1 * rng.standard_normal(rate)]) for _ in range(3)])
    noise = (0.01 * rng.standard_normal(2 * rate)).astype(np.float32)
    take = (speech + 0.01 * rng.standard_normal(speech.size)).astype(np.float32)
    settings = ev.current_settings(True)
    result = {**{name: settings[name] for name in ev.SECTIONS}, "diagnostics": settings["diagnostics"],
              "eq_settings": {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10}}
    filtered = [filtered_capture_for_analysis(audio, rate) for audio in (noise, take)]
    flow = ev.verify_flow(result, {"noise": filtered[0], "setup": filtered[1], "second": filtered[1],
                                   "raw_noise": noise, "raw_second": take})
    assert flow["decision"] in ("accept", "rollback", "gave_up", "retry") and flow["reason_codes"]
    assert flow["first_decision"] and "bands" in flow["state"]["eq_settings"]


def test_a_development_user_carries_the_probes_and_the_cross_check_and_gives_counts_only(monkeypatch, tmp_path):
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False), development=True)
    assert row["development"] is True
    assert set(row["probes"]) == set(ev.probe_settings(ev.current_settings(True)))
    assert math.isfinite(row["tone_cross_check_gap_db"]) and row["tone_cross_check_gap_db"] < 0.5
    result = ev.bookkeeping([row])
    assert _walk_floats(result) == [] and result["development_users"] == 1
    assert result["reachability"]["B5"]["fired"] == 1 and result["tone_cross_check"] == {"users": 1, "within_half_db": 1}
    assert result["probes"]["current_again"]["identical_to_current"] == 1
