"""Tests for the rules that qualify Auto Voice Setup's loudness proposals."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from tools.evaluate_voice_setup_loudness import _limitations, fit_report, narrow_report


def _arm(
    *,
    complete: bool = True,
    safe: bool = True,
    lowered: float = 0.0,
    loudness: float = -18.0,
    headroom: float = 2.0,
    limiter: float = 0.0,
    decision: str = "accept",
    adjustments: int = 0,
    pressure: bool = False,
    reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not complete:
        return {"complete": False}
    return {
        "complete": True,
        "headroom_safe": safe,
        "reasons": list(reasons),
        "apply_recommended": safe,
        "lowered_db": lowered,
        "loudness": loudness,
        "pre_limiter_true_peak_headroom_db": headroom,
        "limiter_gain_reduction_db": limiter,
        "verification": {
            "decision": decision,
            "adjustments": adjustments,
            "limiter_pressure": pressure,
        },
    }


def _rows() -> list[dict[str, Any]]:
    return [
        # Released pressed the limiter and gave up; the fit lowers it just enough.
        {
            "index": 0,
            "released": _arm(
                safe=False, headroom=-2.0, limiter=2.0, decision="gave_up", adjustments=3, pressure=True
            ),
            "fitted": _arm(lowered=3.0, loudness=-21.0, headroom=1.5),
        },
        # Released was safe already and stays as it was.
        {"index": 1, "released": _arm(), "fitted": _arm()},
    ]


def _failed(rows: list[dict[str, Any]]) -> set[str]:
    return {name for name, held in fit_report(rows)["gates"].items() if not held}


def test_a_fit_that_helps_and_changes_nothing_else_passes_every_gate():
    summary = fit_report(_rows())

    assert all(summary["gates"].values())
    assert summary["passed"] is True
    assert summary["metrics"]["headroom_safe"] == {"released": 1, "fitted": 2}
    assert summary["metrics"]["verification_reductions"] == {"released": 3, "fitted": 0}


def test_users_without_eq_settings_are_counted_and_left_out():
    rows = _rows() + [
        {"index": 2, "released": _arm(complete=False), "fitted": _arm(complete=False)},
        # One arm is enough to leave a user out.
        {"index": 3, "released": _arm(), "fitted": _arm(complete=False)},
    ]
    summary = fit_report(rows)

    assert summary["metrics"]["users"] == 4 and summary["metrics"]["scored_users"] == 2
    assert summary["metrics"]["no_eq_settings"] == {"released": 1, "fitted": 2}
    assert {key: summary[key] for key in ("gates", "passed")} == {
        key: fit_report(_rows())[key] for key in ("gates", "passed")
    }
    assert fit_report(rows[2:])["passed"] is False  # nothing left to score


def test_a_safe_proposal_the_fit_changes_fails_leaves_safe_alone():
    rows = _rows()
    rows[1]["fitted"]["loudness"] = -19.0

    assert _failed(rows) == {"leaves_safe_alone"}
    assert fit_report(rows)["passed"] is False


def test_a_proposal_with_more_limiter_pressure_fails_never_less_safe():
    rows = _rows()
    rows[1]["fitted"]["limiter_gain_reduction_db"] = 0.3  # past the 0.1 dB tolerance

    assert _failed(rows) == {"never_less_safe"}
    rows[1]["fitted"]["limiter_gain_reduction_db"] = 0.1
    assert _failed(rows) == set()


def test_a_user_verification_stops_accepting_fails_verification_no_harm():
    rows = _rows()
    rows[1]["fitted"]["verification"] = {"decision": "gave_up", "adjustments": 0, "limiter_pressure": False}

    assert _failed(rows) == {"verification_no_harm"}

    rows = _rows()
    rows[1]["fitted"]["verification"]["adjustments"] = 1  # one more reduction than released needed
    assert _failed(rows) == {"verification_no_harm"}


def test_fitted_proposals_that_stay_unsafe_are_counted_by_whether_no_makeup_keeps_headroom():
    rows = _rows()
    rows[0]["fitted"] = _arm(
        safe=False,
        reasons=("the microphone level leaves too little headroom; lower the input gain by about 3 dB",),
    )
    rows[1]["released"] = rows[1]["fitted"] = _arm(
        safe=False, reasons=("final full-chain headroom validation did not pass",)
    )

    metrics = fit_report(rows)["metrics"]

    assert metrics["fitted_still_unsafe"] == 2
    assert metrics["fitted_still_unsafe_input_too_hot"] == 1


# The narrow limiter allowance: `allowed_narrow` against `fitted`.


def _narrow_arm(
    *,
    complete: bool = True,
    safe: bool = True,
    loudness: float = -18.0,
    too_hot: bool = False,
    allowance: float = 3.0,
    limiter: float = 0.0,
    true_peak: float = 0.0,
    estoi: float = 0.9,
    decision: str = "accept",
    pressure: bool = False,
    settings: str = "settings",
) -> dict[str, Any]:
    if not complete:
        return {"complete": False}
    return {
        "complete": True,
        "auto_makeup": True,
        "input_too_hot": too_hot,
        "limiter_enabled": True,
        "limiter_allowance_db": 0.0 if too_hot else allowance,
        "loudness": loudness,
        "headroom_safe": safe,
        "offline_validation_passed": safe,
        "apply_recommended": safe,
        "reasons": [],
        "settings_sha256": {"eq_settings": settings},
        "limiter_gain_reduction_db": limiter,
        "final_check_render_gap": 0.0,
        "verification": {
            "decision": decision,
            "adjustments": 0,
            "limiter_pressure": pressure,
            "reason_codes": [decision],
        },
        "listen": {
            "estoi": estoi,
            "limiter_gain_reduction_db": limiter,
            "true_peak_limiter_gain_reduction_db": true_peak,
        },
    }


def _narrow_row(
    index: int, fitted: dict[str, Any], allowed: dict[str, Any], split: str = "test"
) -> dict[str, Any]:
    return {
        "split": split,
        "index": index,
        "speaker": f"p{index}",
        "released": fitted,
        "fitted": fitted,
        "allowed_narrow": allowed,
    }


def _narrow_rows() -> list[dict[str, Any]]:
    return [
        # The fit gave up on these; the allowance accepts the very same proposals.
        _narrow_row(0, _narrow_arm(safe=False), _narrow_arm(limiter=2.0)),
        _narrow_row(1, _narrow_arm(safe=False), _narrow_arm(limiter=3.0)),
        # The fit lowered this one to get the limiter off the peaks; the allowance does not need to.
        _narrow_row(
            2, _narrow_arm(loudness=-21.0), _narrow_arm(limiter=2.5, estoi=0.905, settings="louder")
        ),
        # A voice too hot for makeup gets exactly the fitted proposal.
        _narrow_row(3, _narrow_arm(too_hot=True), _narrow_arm(too_hot=True)),
    ]


def _narrow_failed(rows: list[dict[str, Any]]) -> set[str]:
    return {name for name, held in narrow_report(rows)["gates"].items() if not held}


def test_an_allowance_that_helps_and_changes_nothing_else_passes_every_gate():
    summary = narrow_report(_narrow_rows())
    metrics = summary["metrics"]

    assert all(summary["gates"].values())
    assert summary["passed"] is True
    assert metrics["headroom_safe"]["fitted"] == 2 and metrics["headroom_safe"]["allowed_narrow"] == 4
    assert metrics["allowance_applied_users"] == 3 and metrics["strict_users"] == 1
    assert metrics["newly_passing_users"] == 2
    assert metrics["evaluation_limiter_gain_reduction_db_max"] == 3.0
    assert metrics["estoi_change_min"] == 0.0 and metrics["estoi_change_max"] == pytest.approx(0.005)
    assert metrics["below_app_warning_readings"]["strict_passed"] is True
    # The per-split columns are descriptive; the overall one decides.
    assert set(summary["partitions"]) == {"test"}
    assert summary["partitions"]["test"]["gates"] == summary["gates"]


def test_users_without_eq_settings_in_any_arm_are_counted_and_left_out_of_the_narrow_rule():
    rows = _narrow_rows() + [
        _narrow_row(4, _narrow_arm(complete=False), _narrow_arm(complete=False), split="fit"),
        # One arm is enough to leave a user out.
        _narrow_row(5, _narrow_arm(), _narrow_arm(complete=False), split="fit"),
    ]
    summary = narrow_report(rows)
    metrics = summary["metrics"]

    assert metrics["users"] == 6 and metrics["scored_users"] == 4
    assert metrics["speakers"] == 6 and metrics["scored_speakers"] == 4
    assert metrics["no_eq_settings"] == {"released": 1, "fitted": 1, "allowed_narrow": 2}
    assert [(user["split"], user["index"]) for user in metrics["left_out"]] == [("fit", 4), ("fit", 5)]
    assert metrics["left_out"][1]["arms_without_settings"] == ["allowed_narrow"]
    assert summary["gates"] == narrow_report(_narrow_rows())["gates"]
    assert set(summary["partitions"]) == {"fit", "test"}
    assert summary["partitions"]["fit"]["passed"] is False  # nothing left to score there
    assert narrow_report(rows[4:])["passed"] is False


def test_a_user_whose_estoi_falls_too_far_fails_intelligibility_kept():
    rows = _narrow_rows()
    rows[2]["allowed_narrow"]["listen"]["estoi"] = 0.9 - 0.005
    assert _narrow_failed(rows) == set()
    rows[2]["allowed_narrow"]["listen"]["estoi"] = 0.9 - 0.02

    assert _narrow_failed(rows) == {"intelligibility_kept"}
    assert narrow_report(rows)["passed"] is False


@pytest.mark.parametrize(
    ("limiter", "true_peak", "reaches"),
    [(5.99, 2.99, False), (6.0, 0.0, True), (0.0, 3.0, True)],
    ids=["just-under", "limiter-at-6-dB", "true-peak-at-3-dB"],
)
def test_a_newly_passing_proposal_that_reaches_the_app_warning_fails_below_app_warning(
    limiter, true_peak, reaches
):
    rows = _narrow_rows()
    rows[0]["allowed_narrow"]["listen"].update(
        limiter_gain_reduction_db=limiter, true_peak_limiter_gain_reduction_db=true_peak
    )
    readings = narrow_report(rows)["metrics"]["below_app_warning_readings"]

    assert (_narrow_failed(rows) == {"below_app_warning"}) is reaches
    assert readings["strict_passed"] is not reaches and readings["attributed_passed"] is not reaches
    assert len(readings["counted_against_the_candidate"]) == int(reaches)
    assert readings["baseline_exceedances"] == []


def test_a_baseline_exceedance_with_an_identical_fitted_proposal_does_not_count():
    rows = _narrow_rows()
    # Fitted already passes its check with these very settings and reaches the warning too.
    rows[2]["fitted"] = _narrow_arm(limiter=6.5, settings="same")
    rows[2]["allowed_narrow"] = _narrow_arm(limiter=6.5, settings="same")
    summary = narrow_report(rows)
    readings = summary["metrics"]["below_app_warning_readings"]

    assert summary["gates"]["below_app_warning"] is True
    assert readings["attributed_passed"] is True and readings["strict_passed"] is False
    assert readings["proposals_reaching_warning"] == 1
    assert readings["counted_against_the_candidate"] == []
    assert [(user["split"], user["index"], user["speaker"]) for user in readings["baseline_exceedances"]] == [
        ("test", 2, "p2")
    ]
    assert readings["baseline_exceedances"][0]["fitted"]["limiter_gain_reduction_db"] == 6.5


@pytest.mark.parametrize(
    "fitted",
    [
        _narrow_arm(limiter=6.5, settings="other"),  # changed proposal
        _narrow_arm(limiter=2.0, settings="same"),  # fitted stays under the warning
        _narrow_arm(safe=False, limiter=6.5, settings="same"),  # fitted fails its own check
    ],
    ids=["changed-settings", "fitted-under-the-warning", "fitted-fails-its-check"],
)
def test_a_changed_or_newly_passing_proposal_that_reaches_the_warning_counts(fitted):
    rows = _narrow_rows()
    rows[2]["fitted"] = fitted
    rows[2]["allowed_narrow"] = _narrow_arm(limiter=6.5, settings="same")
    readings = narrow_report(rows)["metrics"]["below_app_warning_readings"]

    assert _narrow_failed(rows) == {"below_app_warning"}
    assert readings["strict_passed"] is False and readings["attributed_passed"] is False
    assert [user["index"] for user in readings["counted_against_the_candidate"]] == [2]
    assert readings["baseline_exceedances"] == []


def test_a_passing_proposal_quieter_than_the_fitted_one_fails_never_quieter():
    rows = _narrow_rows()
    rows[2]["fitted"]["loudness"], rows[2]["allowed_narrow"]["loudness"] = -18.0, -21.0

    assert _narrow_failed(rows) == {"never_quieter"}


def test_a_quieter_proposal_that_does_not_pass_fails_lowers_only_to_pass():
    rows = _narrow_rows()
    rows[0]["allowed_narrow"].update(loudness=-21.0, headroom_safe=False, apply_recommended=False)

    assert _narrow_failed(rows) == {"lowers_only_to_pass"}


def test_a_user_whose_verification_stops_accepting_fails_verification_no_harm():
    rows = _narrow_rows()
    rows[2]["allowed_narrow"]["verification"].update(decision="gave_up", reason_codes=["gave_up"])

    assert _narrow_failed(rows) == {"verification_no_harm"}


@pytest.mark.parametrize(
    "change",
    [
        {"settings_sha256": {"eq_settings": "different"}},
        {"limiter_allowance_db": 3.0},
        {"reasons": ["a different reason"]},
        {"apply_recommended": False},
    ],
    ids=["settings", "allowance-applied", "reasons", "recommendation"],
)
def test_a_too_hot_user_who_does_not_get_exactly_the_fitted_proposal_fails_strict_where_not_raised(change):
    rows = _narrow_rows()
    rows[3]["allowed_narrow"].update(change)

    assert "strict_where_not_raised" in _narrow_failed(rows)
    assert _narrow_failed(_narrow_rows()) == set()


def test_the_report_counts_why_the_allowance_did_not_apply():
    rows = _narrow_rows()
    assert narrow_report(rows)["metrics"]["allowance_not_applied"] == {
        "input_too_hot_with_auto_makeup": 1,
        "manual_makeup": 0,
        "limiter_disabled": 0,
    }

    # The analysis counts manual makeup as too hot for the allowance.
    rows[3]["fitted"]["auto_makeup"] = False
    rows[1]["fitted"]["limiter_enabled"] = False
    assert narrow_report(rows)["metrics"]["allowance_not_applied"] == {
        "input_too_hot_with_auto_makeup": 0,
        "manual_makeup": 1,
        "limiter_disabled": 1,
    }


def test_no_extra_passes_fewer_recommendations_or_more_limiter_give_ups_fail_benefit():
    # Nothing newly passes.
    rows = _narrow_rows()
    for index in (0, 1):
        rows[index]["allowed_narrow"] = deepcopy(rows[index]["fitted"])
    assert _narrow_failed(rows) == {"benefit"}

    # More pass, but fewer are recommended.
    rows = _narrow_rows()
    rows[2]["allowed_narrow"]["apply_recommended"] = False
    rows[0]["fitted"]["apply_recommended"] = rows[1]["fitted"]["apply_recommended"] = True
    assert _narrow_failed(rows) == {"benefit"}

    # More pass, but verification gives up on limiter pressure more often.
    rows = _narrow_rows()
    rows[2]["fitted"]["verification"]["decision"] = "gave_up"
    rows[2]["allowed_narrow"]["verification"].update(decision="gave_up", limiter_pressure=True)
    assert _narrow_failed(rows) == {"benefit"}


def test_the_narrow_report_states_what_it_scored():
    unscored = _narrow_row(4, _narrow_arm(complete=False), _narrow_arm(complete=False))
    limitations = _limitations("narrow", narrow_report(_narrow_rows() + [unscored])["metrics"])

    assert any("4 of 5 users are scored, from 4 of 5 speakers" in text for text in limitations)
    assert any("3.00 dB, against the app's 6 dB warning" in text for text in limitations)
    assert any("leaves a proposal as released" in text for text in _limitations("fit", {}))
