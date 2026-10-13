"""Tests for the rule that qualifies the bypassed tonal check of Voice Setup's verification.

A baseline world of 48 users x 3 settings passes every gate with margin on each clause that is
not under test: 34 users per setting are verified, 11 have no Auto-EQ settings and 3 have a
rejected second take (retry before any spectrum). Limits over its 102 verified pairs, 34 per
setting: N1 at most 10 candidate rollbacks overall and 5 per setting, and 3 * candidate <=
released; N2 at 9 dB at least 92 reports overall and 29 per setting, at 12 dB at least 99 and 32.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tools import evaluate_voice_setup_verification as evaluator

SETTINGS = tuple(evaluator.SETTINGS)
SPLITS = ("fit", "calibrate", "test")
VERIFIED, WITHOUT_SETTINGS = 34, 11
RELEASED_BAD, CANDIDATE_BAD = (15, 10, 8), 3
SENTENCE = "noise suppression is dulling consonants by about 9 dB; lower its strength or use DeepFilterNet LL"
ALL_PASS = {
    "false_alarms_low": True, "catches_wrong_eq": True, "never_stricter": True, "nothing_else_moves": True,
    "plus9_3k": True, "plus12_3k": True,
}


def _outcome(after, *, decision="accept", codes=("accepted",), advisory=None) -> dict[str, Any]:
    """A full-chain verification outcome carrying every key the report reads."""
    outcome = {
        "decision": decision,
        "reasons": ["repeatability and full-chain DSP constraints passed"] if decision == "accept" else ["target error worsened"],
        "reason_codes": list(codes), "reduction_targets": [], "full_chain": True, "full_chain_voice_dropout": False,
        "spectral_target_error_before_db": 0.75, "spectral_target_error_after_db": after,
        "snr_change_db": 2.0, "limiter_pressure_db": 0.3, "compressor_gain_reduction_p95_db": 3.0,
        "suppressor_dulling_db": None, "suppressor_dulling_abstained": True,
    }
    if advisory is not None:
        outcome["suppressor_dulling_db"], outcome["suppressor_dulling_abstained"] = advisory, False
        if advisory > 6.0:
            outcome["reason_codes"].append("suppressor_dulling")
            outcome["reasons"].append(SENTENCE)
    return outcome


def _arm(outcome, *, renders, calls=(6.0,)) -> dict[str, Any]:
    return {
        "decision": "gave_up" if outcome["decision"] == "reduce" else outcome["decision"],
        "adjustments": 0, "limiter_pressure": False,
        "first_pass": {"decision": outcome["decision"], "reason_codes": list(outcome["reason_codes"]),
                       "spectral_target_error_before_db": outcome.get("spectral_target_error_before_db"),
                       "spectral_target_error_after_db": outcome.get("spectral_target_error_after_db"),
                       "outcome": copy.deepcopy(outcome)},
        "outcome": outcome, "calls_s": list(calls), "renders": [list(renders)],
    }


def _rejected_take() -> dict[str, Any]:
    retry = {"decision": "retry", "reasons": ["verification passage was non-finite or clipped; lower input gain and repeat"],
             "reason_codes": ["verification_capture_invalid"], "reduction_targets": []}
    return {"decision": "retry", "adjustments": 0, "limiter_pressure": False,
            "first_pass": {"decision": "retry", "reason_codes": ["verification_capture_invalid"], "outcome": retry},
            "outcome": retry, "calls_s": [0.1], "renders": [[]]}


def _row(setting, user, **fields) -> dict[str, Any]:
    model, strength = evaluator.SETTINGS[setting]
    return {"split": SPLITS[user // 16 % 3], "index": user % 16, "speaker": f"p{user}", "model": setting,
            "noise_model": model, "strength": strength, "eq_error": None, "complete": True,
            "proposed_suppressor": {"enabled": True, "strength": strength, "model": model}, **fields}


def _verified_row(setting, user, advisory) -> dict[str, Any]:
    wrong = {name: {"change": {"applied_change_db": delta}, "released": _arm(_outcome(0.9, advisory=advisory), renders=(True,) * 3),
                    "candidate": _arm(_outcome(0.85, advisory=advisory), renders=(True,) * 3 + (False,) * 2, calls=(7.0,))}
             for name, (_, delta) in evaluator.WRONG_EQS.items()}
    return _row(setting, user, verification={
        "released": _arm(_outcome(0.8, advisory=advisory), renders=(True,) * 3),
        "candidate": _arm(_outcome(0.7, advisory=advisory), renders=(True,) * 3 + (False,) * 2, calls=(7.0,)),
        "wrong": wrong})


def _set_target_error(arm, rolled_back=True) -> None:
    """Make an arm roll back for target error (or accept) in its first pass and final outcome.
    The advisory, if it fired, stays on whatever the decision."""
    for outcome in (arm["outcome"], arm["first_pass"]["outcome"]):
        fired = (outcome.get("suppressor_dulling_db") or 0.0) > 6.0
        outcome["decision"] = "rollback" if rolled_back else "accept"
        outcome["reason_codes"] = ["target_error" if rolled_back else "accepted"] + (["suppressor_dulling"] if fired else [])
        outcome["reasons"] = ["target error worsened" if rolled_back else "repeatability and full-chain DSP constraints passed"] \
            + ([SENTENCE] if fired else [])
    arm["decision"] = arm["first_pass"]["decision"] = "rollback" if rolled_back else "accept"
    arm["first_pass"]["reason_codes"] = list(arm["first_pass"]["outcome"]["reason_codes"])


def _verified(rows, setting=None) -> list[dict[str, Any]]:
    return [row for row in rows if evaluator.verified(row) and (setting is None or row["model"] == setting)]


def _roll_back(rows, arm_name, counts) -> None:
    """Make exactly ``counts[setting]`` verified pairs of each setting roll back for target error under one arm
    (the first ones, so that the candidate's rollbacks lie inside the released check's)."""
    for setting, wanted in counts.items():
        for position, row in enumerate(_verified(rows, setting)):
            _set_target_error(row["verification"][arm_name], position < wanted)


def _report_wrong_eq(rows, name, counts) -> None:
    for setting, wanted in counts.items():
        for position, row in enumerate(_verified(rows, setting)):
            _set_target_error(row["verification"]["wrong"][name]["candidate"], position < wanted)


def _baseline() -> list[dict[str, Any]]:
    """Users 0-33 verified, 34-44 without settings, 45-47 with a rejected second take."""
    rows = []
    for setting in SETTINGS:
        for user in range(48):
            if user < VERIFIED:
                advisory = 9.0 if user in (4, 5) and setting == SETTINGS[0] else 2.0 if user % 3 else None
                rows.append(_verified_row(setting, user, advisory))
            elif user < VERIFIED + WITHOUT_SETTINGS:
                rows.append(_row(setting, user, complete=False,
                                 eq_error="Recording too unclear" if user % 2 else "proposed processing exceeds safe headroom"))
            else:
                rejected = {"released": _rejected_take(), "candidate": _rejected_take(),
                            "wrong": {name: {"change": {"applied_change_db": 0.0}, "released": _rejected_take(),
                                             "candidate": _rejected_take()} for name in evaluator.WRONG_EQS}}
                rows.append(_row(setting, user, verification=rejected))
    _roll_back(rows, "candidate", {setting: CANDIDATE_BAD for setting in SETTINGS})
    _roll_back(rows, "released", dict(zip(SETTINGS, RELEASED_BAD)))
    for name, wanted in (("plus9_3k", 33), ("plus12_3k", 34)):
        _report_wrong_eq(rows, name, {setting: wanted for setting in SETTINGS})
        for row in _verified(rows):
            _set_target_error(row["verification"]["wrong"][name]["released"])
    return rows


def _flags(rows) -> dict[str, bool]:
    report = evaluator.verification_report(rows)
    wrong = report["metrics"]["catches_wrong_eq"]["wrong_eqs"]
    return {**report["gates"], **{name: wrong[name]["passed"] for name in wrong}}


def _clauses(rows, gate) -> dict[str, bool]:
    return evaluator.verification_report(rows)["metrics"][gate]["clauses"]


def test_the_baseline_passes_every_gate_and_leaves_out_the_pairs_it_cannot_judge():
    rows = _baseline()
    report = evaluator.verification_report(rows)
    metrics = report["metrics"]

    assert _flags(rows) == ALL_PASS and report["passed"] is True
    assert (metrics["pairs"], metrics["verified_pairs"], metrics["users"], metrics["verified_users"]) == (144, 102, 48, 34)
    assert all(
        (value["verified"], value["no_settings"], value["second_take_rejected"]) == (34, 11, 3)
        for value in metrics["accounting"].values()
    )
    assert len(metrics["left_out"]) == 42
    assert metrics["false_alarms_low"]["n"] == 102
    assert metrics["false_alarms_low"]["candidate_target_error"] == 9 and metrics["false_alarms_low"]["released_target_error"] == 33


def test_pairs_without_settings_or_with_a_rejected_take_enter_no_gate():
    rows = _baseline()
    for row in rows:
        if not row["complete"]:
            row["verification"] = {"released": {"decision": "accept"}}
        elif not evaluator.verified(row):
            row["verification"]["candidate"]["outcome"].update(decision="rollback", reason_codes=["target_error"])

    assert _flags(rows) == ALL_PASS


def test_without_a_verified_pair_every_gate_fails():
    rows = _baseline()
    for row in rows:
        row.update(complete=False, eq_error="Recording too unclear")
    report = evaluator.verification_report(rows)

    assert report["passed"] is False and not any(report["gates"].values())


@pytest.mark.parametrize(
    ("candidate", "released", "failing"),
    [
        ((4, 3, 3), None, None),
        ((4, 4, 3), None, "overall_at_most_10_percent"),
        ((5, 2, 2), None, None),
        ((0, 6, 0), None, "each_setting_at_most_15_percent"),
        ((3, 3, 3), (10, 10, 7), None),
        ((3, 3, 3), (10, 10, 6), "at_most_a_third_of_released"),
        ((0, 0, 0), (0, 0, 0), None),
        ((1, 0, 0), (2, 0, 0), "at_most_a_third_of_released"),
    ],
    ids=["10-of-102", "11-of-102", "5-of-34", "6-of-34", "third-27", "third-26", "none", "1-against-2"],
)
def test_false_alarms_low_is_decided_by_its_three_limits(candidate, released, failing):
    rows = _baseline()
    _roll_back(rows, "candidate", dict(zip(SETTINGS, candidate)))
    if released is not None:
        _roll_back(rows, "released", dict(zip(SETTINGS, released)))

    clauses = _clauses(rows, "false_alarms_low")
    assert [name for name, held in clauses.items() if not held] == ([failing] if failing else [])
    assert _flags(rows) == {**ALL_PASS, "false_alarms_low": failing is None}


@pytest.mark.parametrize(
    ("name", "counts", "held", "failing"),
    [
        ("plus9_3k", (32, 30, 30), True, None),
        ("plus9_3k", (31, 30, 30), False, "overall"),
        ("plus9_3k", (34, 34, 29), True, None),
        ("plus9_3k", (34, 34, 28), False, "each_setting"),
        ("plus12_3k", (33, 33, 33), True, None),
        ("plus12_3k", (33, 33, 32), False, "overall"),
        ("plus12_3k", (34, 34, 32), True, None),
        ("plus12_3k", (34, 34, 31), False, "each_setting"),
    ],
    ids=["9-92-of-102", "9-91-of-102", "9-29-of-34", "9-28-of-34", "12-99-of-102", "12-98-of-102", "12-32-of-34", "12-31-of-34"],
)
def test_catches_wrong_eq_is_decided_by_its_limits_at_each_gain(name, counts, held, failing):
    rows = _baseline()
    _report_wrong_eq(rows, name, dict(zip(SETTINGS, counts)))

    clauses = evaluator.verification_report(rows)["metrics"]["catches_wrong_eq"]["wrong_eqs"][name]["clauses"]
    assert [clause for clause, ok in clauses.items() if not ok] == ([failing] if failing else [])
    assert _flags(rows) == {**ALL_PASS, "catches_wrong_eq": held, name: held}


def test_the_per_setting_limits_are_the_rules_percentages():
    # 34 pairs per setting cannot tell 15 % from 16 %; 100 pairs can.
    def per_setting_clauses(candidate_bad, plus9, plus12):
        rows = [_verified_row(SETTINGS[0], user, None) for user in range(100)]
        _roll_back(rows, "candidate", {SETTINGS[0]: candidate_bad})
        _report_wrong_eq(rows, "plus9_3k", {SETTINGS[0]: plus9})
        _report_wrong_eq(rows, "plus12_3k", {SETTINGS[0]: plus12})
        metrics = evaluator.verification_report(rows)["metrics"]
        wrong = metrics["catches_wrong_eq"]["wrong_eqs"]
        return (
            metrics["false_alarms_low"]["per_setting"][SETTINGS[0]]["passed"],
            wrong["plus9_3k"]["per_setting"][SETTINGS[0]]["passed"],
            wrong["plus12_3k"]["per_setting"][SETTINGS[0]]["passed"],
        )

    assert per_setting_clauses(15, 85, 94) == (True, True, True)
    assert per_setting_clauses(16, 84, 93) == (False, False, False)


def test_the_released_arm_and_the_6_dB_wrong_eqs_never_decide_catches_wrong_eq():
    rows = _baseline()
    for row in _verified(rows):
        _set_target_error(row["verification"]["wrong"]["plus9_3k"]["released"], False)
        _set_target_error(row["verification"]["wrong"]["plus6_3k"]["candidate"], False)

    assert _flags(rows) == ALL_PASS


def test_the_6_dB_wrong_eqs_are_reported_beyond_what_the_proposal_already_shows():
    rows = _baseline()
    _set_target_error(rows[0]["verification"]["wrong"]["minus6_200"]["released"])  # rolls back with the proposal itself
    _set_target_error(rows[20]["verification"]["wrong"]["minus6_200"]["released"])  # does not
    reported = evaluator.verification_report(rows)["metrics"]["reported"]["wrong_eq_6db"]["minus6_200"]

    assert reported["released"] == 2 and reported["candidate"] == 0
    assert reported["beyond_the_proposal"] == {"released": 1, "candidate": 0}


def _accepted_by_both(rows, setting=SETTINGS[1]) -> dict[str, Any]:
    return _verified(rows, setting)[-1]


def test_a_pair_the_released_check_accepts_but_the_candidate_rolls_back_fails_never_stricter():
    rows = _baseline()
    stricter = _accepted_by_both(rows)
    _set_target_error(stricter["verification"]["candidate"])
    gate = evaluator.verification_report(rows)["metrics"]["never_stricter"]

    assert gate["rolled_back_by_candidate"] == [[stricter["split"], stricter["index"], stricter["model"]]]
    assert _flags(rows)["never_stricter"] is False


def test_a_candidate_target_error_on_top_of_a_released_rollback_for_another_reason_is_not_stricter():
    rows = _baseline()
    row = _accepted_by_both(rows)
    for arm_name, codes in (("released", ["noise_safety_limit"]), ("candidate", ["target_error", "noise_safety_limit"])):
        arm = row["verification"][arm_name]
        for outcome in (arm["outcome"], arm["first_pass"]["outcome"]):
            outcome.update(decision="rollback", reason_codes=list(codes))
        arm["decision"] = arm["first_pass"]["decision"] = "rollback"
        arm["first_pass"]["reason_codes"] = list(codes)

    assert _flags(rows) == ALL_PASS


def _advised(rows, setting=SETTINGS[0]) -> dict[str, Any]:
    return next(row for row in _verified(rows, setting) if row["verification"]["released"]["outcome"]["suppressor_dulling_db"] == 9.0)


def _first_pass_finding(key, value, arm="candidate", final=False):
    def change(rows):
        outcome = _accepted_by_both(rows)["verification"][arm]
        (outcome["outcome"] if final else outcome["first_pass"]["outcome"])[key] = value

    return change


def _first_pass_codes(rows):
    _accepted_by_both(rows)["verification"]["released"]["first_pass"]["reason_codes"] = ["noise_safety_limit"]


def _advisory_value(rows):
    _advised(rows)["verification"]["candidate"]["outcome"]["suppressor_dulling_db"] = 9.0 + 1e-9


def _first_pass_advisory_value(rows):
    _advised(rows)["verification"]["candidate"]["first_pass"]["outcome"]["suppressor_dulling_db"] = 9.0 + 1e-9


def _advisory_abstained(rows):
    row = next(r for r in _verified(rows, SETTINGS[0]) if r["verification"]["released"]["outcome"]["suppressor_dulling_abstained"])
    row["verification"]["candidate"]["outcome"]["suppressor_dulling_abstained"] = False


def _advisory_sentence(rows):
    outcome = _advised(rows)["verification"]["candidate"]["outcome"]
    outcome["reasons"] = [t.replace("9 dB", "8 dB") if t.startswith(evaluator.ADVISORY_PREFIX) else t for t in outcome["reasons"]]


def _decision_differs(rows):
    _accepted_by_both(rows)["verification"]["released"]["decision"] = "gave_up"


def _reductions_differ(rows):
    _accepted_by_both(rows)["verification"]["candidate"]["adjustments"] = 1


@pytest.mark.parametrize(
    "change",
    [
        _first_pass_finding("snr_change_db", 2.0 + 1e-9),
        _first_pass_finding("reduction_targets", ["compressor"]),
        _first_pass_finding("spectral_target_error_before_db", 0.75 + 1e-9),
        _first_pass_codes,
        _first_pass_finding("reason_codes", ["accepted", "suppressor_dulling"], final=True),
        _advisory_value,
        _first_pass_advisory_value,
        _advisory_abstained,
        _advisory_sentence,
        _decision_differs,
        _reductions_differ,
    ],
    ids=["finding", "reduction-targets", "before-error", "safety-codes", "advisory-fires", "advisory-value",
         "first-pass-advisory-value", "advisory-abstained", "advisory-sentence", "decision", "reductions"],
)
def test_any_difference_outside_the_target_error_fails_nothing_else_moves(change):
    rows = _baseline()
    change(rows)

    assert _flags(rows) == {**ALL_PASS, "nothing_else_moves": False}


def test_the_same_nan_in_both_arms_is_equal():
    rows = _baseline()
    for arm in ("released", "candidate"):
        _first_pass_finding("snr_change_db", float("nan"), arm)(rows)
        _first_pass_finding("snr_change_db", float("nan"), arm, final=True)(rows)

    assert _flags(rows) == ALL_PASS


def test_the_flow_after_a_first_pass_with_target_error_may_differ():
    rows = _baseline()
    row = next(r for r in _verified(rows, SETTINGS[0])
               if evaluator._tgt(r["verification"]["released"]) and not evaluator._tgt(r["verification"]["candidate"]))
    candidate = row["verification"]["candidate"]
    candidate.update(adjustments=2, decision="gave_up")
    candidate["outcome"].update(compressor_gain_reduction_p95_db=2.0, decision="reduce", reason_codes=["compressor_excess"])

    assert _flags(rows) == ALL_PASS


def test_a_setting_without_pairs_is_skipped_in_the_per_setting_clauses():
    def without_last_setting(candidate):
        rows = _baseline()
        for row in rows:
            if row["model"] == SETTINGS[2]:
                row.update(complete=False, eq_error="Recording too unclear")
        _roll_back(rows, "candidate", dict(zip(SETTINGS, candidate)))
        return rows

    inside, outside = without_last_setting((3, 3, 0)), without_last_setting((4, 3, 0))  # 6 and 7 of 68 pairs

    assert evaluator.verification_report(inside)["metrics"]["false_alarms_low"]["n"] == 68
    assert _flags(inside) == ALL_PASS
    assert _clauses(outside, "false_alarms_low")["overall_at_most_10_percent"] is False
    assert _flags(outside) == {**ALL_PASS, "false_alarms_low": False}


def test_the_limitations_state_what_was_scored():
    metrics = evaluator.verification_report(_baseline())["metrics"]
    limitations = evaluator._limitations(metrics)

    assert any("34 of 48 users are verified, from 34 of 48 speakers" in text for text in limitations)
    assert any("adds a target error report beyond the proposal's own for 0 of 102 pairs" in text for text in limitations)
