"""Explainability tests that keep Auto-EQ UI mapping outside DSP policy."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mic_eq.analysis.auto_eq_parts.target import get_target_curve
from mic_eq.ui.auto_eq_explanation import (
    TARGET_MODE_LABELS,
    explain_auto_eq,
    headroom_advice,
    target_profile_text,
)
from mic_eq.ui.eq_panel import _format_auto_eq_diagnostics


@pytest.mark.parametrize(
    ("reason", "expected_code", "expected_text"),
    [
        (
            "capture quality score is too low",
            "unusable_capture",
            "inconsistent or unclear",
        ),
        (
            "room-noise reference is invalid",
            "invalid_noise_reference",
            "room-noise reference",
        ),
        (
            "insufficient repeatable voiced windows",
            "insufficient_coverage",
            "repeatable speech",
        ),
    ],
)
def test_capture_abstention_reasons_map_without_mutation(
    reason: str,
    expected_code: str,
    expected_text: str,
) -> None:
    diagnostics = {
        "recommendation_status": "abstain",
        "abstention_reasons": [reason],
        "analysis_confidence": 0.2,
    }
    original = deepcopy(diagnostics)

    explanation = explain_auto_eq(diagnostics)

    assert diagnostics == original
    assert explanation.outcome_code == expected_code
    assert explanation.summary == "No correction applied"
    assert expected_text in " ".join(explanation.details)
    assert explanation.state == "bad"


def test_per_band_abstention_explains_skipped_bands_but_keeps_success() -> None:
    explanation = explain_auto_eq(
        {
            "recommendation_status": "apply",
            "low_confidence_active_bands": 3,
            "local_abstained_band_indices": [4, 6, 7],
        }
    )

    assert explanation.outcome_code == "low_band_reliability"
    assert explanation.summary == "Correction ready with unsupported bands skipped"
    assert "3 unsupported frequency bands were left unchanged." in explanation.details
    assert explanation.state == "ok"


def test_reduced_result_explains_conservative_success() -> None:
    explanation = explain_auto_eq(
        {
            "recommendation_status": "reduced",
            "recommendation_reasons": [
                "overall confidence is below full-strength threshold",
                "validation reduced the fitted correction",
            ],
        }
    )

    assert explanation.outcome_code == "conservative_success"
    assert explanation.summary == "Conservative correction ready"
    assert len(explanation.details) == 2
    assert explanation.state == "warn"


def test_ui_uses_raw_recommendation_status_not_a_new_confidence_threshold() -> None:
    diagnostics = {
        "recommendation_status": "abstain",
        "abstention_reasons": ["capture quality score is too low"],
        "analysis_confidence": 0.99,
        "eq_confidence": 0.99,
        "capture_confidence": 0.99,
        "validation_confidence": 0.99,
        "validation_before_error_db": 4.0,
        "validation_after_error_db": 4.0,
        "validation_gain_scale": 0.0,
    }

    text, state, tooltip = _format_auto_eq_diagnostics(diagnostics)

    assert "No correction applied" in text
    assert state == "bad"
    assert "Raw recommendation status: abstain" in tooltip
    assert "Overall confidence: 99%" in tooltip


def test_missing_status_is_reported_as_unknown_not_reclassified() -> None:
    explanation = explain_auto_eq({"analysis_confidence": 0.99})

    assert explanation.outcome_code == "unknown"
    assert explanation.state == "idle"


def _render(headroom: float, limiter: float = 0.0, true_peak: float = 0.0) -> dict[str, float]:
    return {
        "pre_limiter_true_peak_headroom_db": headroom,
        "limiter_gain_reduction_db": limiter,
        "true_peak_limiter_gain_reduction_db": true_peak,
    }


def _headroom(status: str, scale: float, before: dict, **extra: object) -> dict:
    return {
        "authoritative": True,
        "safe": status == "safe",
        "status": status,
        "gain_scale": scale,
        "before": before,
        "after": extra.pop("after", before),
        **extra,
    }


_NO_DYNAMICS = {"compressor": {"enabled": False}, "limiter": {"enabled": False}}
_OWNER_TAKE = _headroom(
    "risk_not_worsened",
    1.0,
    _render(-0.283),
    reference_excess_db=1.348,
    flat_excess_db=0.695,
    incumbent_excess_db=1.348,
    incumbent_output_true_peak_db=0.05,
)
_LIMITER_OFF_LINE = (
    "Turning the limiter on would catch the peaks, but it would not make this check pass."
)


def test_headroom_advice_for_the_owner_take_says_the_output_clips_and_lowers_two_decibels() -> None:
    assert headroom_advice(_OWNER_TAKE, _NO_DYNAMICS, proposed=True) == (
        "Your output already clips on loud peaks. "
        "This EQ is no worse than your current one. "
        "Lower the microphone gain by about 2 dB (in Windows or on the microphone) "
        "to leave enough room. " + _LIMITER_OFF_LINE
    )


def test_headroom_advice_for_a_flat_current_eq_and_a_hot_microphone() -> None:
    headroom = _headroom(
        "risk_not_worsened",
        0.0,
        _render(-1.9),
        after=_render(0.4),
        reference_excess_db=0.6,
        flat_excess_db=0.6,
        incumbent_excess_db=0.6,
        incumbent_output_true_peak_db=-0.4,
    )

    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=False) == (
        "Your current settings already leave too little room for loud peaks. "
        "Lower the microphone gain by about 3 dB (in Windows or on the microphone), "
        "then record again. " + _LIMITER_OFF_LINE
    )


_MAKEUP_ON = {
    "compressor": {"enabled": True, "auto_makeup_enabled": True},
    "limiter": {"enabled": True},
}
_ON_THE_LIMITER = _headroom(
    "risk_not_worsened",
    0.0,
    _render(-2.3, limiter=2.28, true_peak=0.03),
    reference_excess_db=3.3,
    flat_excess_db=3.3,
    incumbent_excess_db=3.3,
    incumbent_output_true_peak_db=-1.5,
)


def test_headroom_advice_names_the_target_loudness_when_automatic_makeup_is_on() -> None:
    assert headroom_advice(_ON_THE_LIMITER, _MAKEUP_ON, proposed=False) == (
        "Your current settings already leave too little room for loud peaks. "
        "Lower the compressor's target loudness, then record again. "
        "Automatic makeup would undo a lower microphone gain."
    )


@pytest.mark.parametrize(
    ("compressor", "wording"),
    [
        # Working: level in is not level out, so no figure; the lever is the gain.
        (
            {"enabled": True},
            "Lower the microphone gain (in Windows or on the microphone), then record again.",
        ),
        (
            {"enabled": True, "auto_makeup_enabled": False, "makeup_gain_db": 12.0},
            "Lower the microphone gain (in Windows or on the microphone), then record again.",
        ),
        # Not stated either way: no figure.
        (
            {},
            "Lower the microphone gain (in Windows or on the microphone), then record again.",
        ),
        # Off, with the automatic-makeup flag left on: it does nothing, so the gain it is.
        (
            {"enabled": False, "auto_makeup_enabled": True},
            "Lower the microphone gain by about 4 dB (in Windows or on the microphone), "
            "then record again.",
        ),
    ],
)
def test_headroom_advice_gives_a_figure_only_when_the_compressor_is_off(compressor, wording) -> None:
    chain = {"compressor": compressor, "limiter": {"enabled": True}}

    assert headroom_advice(_ON_THE_LIMITER, chain, proposed=False) == (
        "Your current settings already leave too little room for loud peaks. " + wording
    )


def test_headroom_advice_without_a_chain_gives_no_figure() -> None:
    assert headroom_advice(_ON_THE_LIMITER, None, proposed=False) == (
        "Your current settings already leave too little room for loud peaks. "
        "Lower the microphone gain (in Windows or on the microphone), then record again."
    )


@pytest.mark.parametrize("limiter", [{}, {"enabled": None}, {"enabled": True}, None])
def test_headroom_advice_names_the_limiter_only_when_the_chain_says_it_is_off(limiter) -> None:
    chain = {"compressor": {"enabled": False}, "limiter": limiter}

    assert _LIMITER_OFF_LINE not in headroom_advice(_OWNER_TAKE, chain, proposed=True)
    # Nor with no limiter key in the chain at all.
    assert _LIMITER_OFF_LINE not in headroom_advice(
        _OWNER_TAKE, {"compressor": {"enabled": False}}, proposed=True
    )


def test_headroom_advice_does_not_call_a_current_eq_that_meets_the_rule_too_hot() -> None:
    # The correction would leave less room than the current EQ, which is itself fine.
    headroom = _headroom(
        "risk_worse",
        0.0,
        _render(0.4),
        reference_excess_db=-0.5,
        flat_excess_db=0.6,
        incumbent_excess_db=-0.5,
        incumbent_output_true_peak_db=-3.0,
    )

    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=False) == (
        "Lower the microphone gain by about 1 dB (in Windows or on the microphone), "
        "then record again."
    )


def test_headroom_advice_for_a_correction_left_out_while_the_current_eq_is_over_the_rule() -> None:
    headroom = _headroom(
        "risk_worse",
        0.0,
        _render(-1.2),
        reference_excess_db=0.4,
        flat_excess_db=4.0,
        incumbent_excess_db=0.4,
        incumbent_output_true_peak_db=-0.6,
    )

    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=False) == (
        "Your current settings already leave too little room for loud peaks. "
        "Lower the microphone gain by about 3 dB (in Windows or on the microphone), "
        "then record again. " + _LIMITER_OFF_LINE
    )


def test_headroom_advice_keeps_the_relation_sentence_for_a_nearly_full_correction() -> None:
    nearly_full = _headroom("risk_not_worsened", 0.85, _render(-0.283), incumbent_excess_db=1.348)
    reduced = _headroom("risk_not_worsened", 0.55, _render(-0.283), incumbent_excess_db=1.348)

    assert "This EQ is no worse than your current one." in headroom_advice(
        nearly_full, _NO_DYNAMICS, proposed=True
    )
    # Below 0.70 the explanation already says the correction was made gentler to that end.
    assert headroom_advice(reduced, _NO_DYNAMICS, proposed=True) == (
        "Your current settings already leave too little room for loud peaks. "
        "Lower the microphone gain by about 2 dB (in Windows or on the microphone) "
        "to allow the full correction. " + _LIMITER_OFF_LINE
    )


def test_headroom_advice_for_a_safe_result_that_headroom_made_gentler() -> None:
    headroom = _headroom("safe", 0.55, _render(0.4))

    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=True) == (
        "Lower the microphone gain by about 1 dB (in Windows or on the microphone) "
        "to allow the full correction."
    )
    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=False) == (
        "Lower the microphone gain by about 1 dB (in Windows or on the microphone), "
        "then record again."
    )


@pytest.mark.parametrize(
    "headroom",
    [
        _headroom("safe", 1.0, _render(3.0)),
        _headroom("safe", 0.85, _render(0.5)),
        _headroom("safe", 0.70, _render(0.5)),  # the line is below 0.70, not at it
        {**_OWNER_TAKE, "authoritative": False},
        _headroom("risk", 0.0, _render(-1.0)),
        None,
    ],
)
def test_headroom_advice_is_empty_when_headroom_did_not_limit_the_result(headroom) -> None:
    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=True) == ""


def test_headroom_advice_starts_just_below_seventy_percent() -> None:
    assert headroom_advice(_headroom("safe", 0.69, _render(0.5)), _NO_DYNAMICS, proposed=True) != ""


_REDUCED = "headroom validation reduced the fitted correction"
_DECLINED = "no safe candidate remained after full-chain headroom validation"
_REMOVED = "headroom validation removed the usable correction"


def test_a_reduced_correction_over_the_rule_does_not_claim_there_was_room_to_spare() -> None:
    reduced = {"recommendation_status": "reduced", "recommendation_reasons": [_REDUCED]}

    assert explain_auto_eq(reduced).details == (
        "The correction was made gentler to leave enough headroom.",
    )
    kept = {**reduced, "headroom_validation": {"status": "risk_not_worsened"}}
    assert explain_auto_eq(kept).details == (
        "The correction was made gentler so it leaves no less room than your current EQ.",
    )


def test_a_correction_left_out_says_so_once() -> None:
    explanation = explain_auto_eq(
        {
            "recommendation_status": "abstain",
            "abstention_reasons": [_REMOVED, _DECLINED],
            "headroom_validation": {"status": "risk_worse"},
        }
    )

    assert explanation.details == ("The correction would leave less room than your current EQ does.",)
    assert explanation.outcome_code == "headroom_limited"


@pytest.mark.parametrize(
    ("reason", "expected_code", "expected_text"),
    [
        (
            "no validated correction improved the target safely",
            "no_safe_improvement",
            "No correction that passes the safety checks brings your voice closer "
            "to the tone preset.",
        ),
        (
            "constrained gain solve produced no safe correction",
            "fit_failed",
            "The tone fit could not settle on a correction that stays within the safety limits.",
        ),
        (
            "headroom validation removed the usable correction",
            "headroom_limited",
            "Not enough headroom was left for a useful correction.",
        ),
    ],
)
def test_optimizer_abstentions_say_what_happened_not_to_record_again(
    reason: str, expected_code: str, expected_text: str
) -> None:
    explanation = explain_auto_eq(
        {"recommendation_status": "abstain", "abstention_reasons": [reason]}
    )

    assert explanation.outcome_code == expected_code
    assert explanation.details == (expected_text,)


def test_explanations_use_plain_words() -> None:
    results = [
        explain_auto_eq({"recommendation_status": "apply"}),
        explain_auto_eq({"recommendation_status": "reduced"}),
        explain_auto_eq(
            {
                "recommendation_status": "reduced",
                "recommendation_reasons": ["validation reduced the fitted correction"],
            }
        ),
        explain_auto_eq({"analysis_confidence": 0.9}),
    ]

    text = " ".join(" ".join(result.details) for result in results)
    assert "gates" not in text and "existing" not in text and "diagnostic" not in text
    assert results[0].details == ("Every band that changed is well supported by your recording.",)
    assert results[1].details == ("AudioForge made the correction gentler to keep it safe.",)


def test_tone_mode_names_map_the_stored_values_the_analysis_accepts() -> None:
    assert TARGET_MODE_LABELS == {"adaptive": "Subtle", "static": "Full strength"}
    for stored in TARGET_MODE_LABELS:
        get_target_curve([1000.0], "broadcast", target_mode=stored)  # does not raise


@pytest.mark.parametrize(
    ("profile", "text"),
    [
        ("broadcast:adaptive", "broadcast, Subtle"),
        ("streaming:static", "streaming, Full strength"),
        ("warm:adaptive:fallback", "warm, Subtle"),
        ("broadcast:something-new", "broadcast:something-new"),
        (None, "--"),
    ],
)
def test_target_profiles_are_shown_with_the_plain_name_of_the_strength(profile, text) -> None:
    assert target_profile_text(profile) == text


def test_the_readme_uses_the_names_the_dialog_shows() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")

    assert all(f"**{label}**" in readme for label in TARGET_MODE_LABELS.values())
    assert "Adapt to my voice" not in readme and "Exact tone preset" not in readme


def test_the_eq_panel_tooltip_shows_the_strength_by_its_plain_name() -> None:
    _text, _state, tooltip = _format_auto_eq_diagnostics(
        {"recommendation_status": "apply", "target_profile": "broadcast:adaptive"}
    )

    assert "Target profile: broadcast, Subtle\n" in tooltip
    assert "adaptive" not in tooltip


def test_the_auto_eq_button_does_not_promise_a_fit_to_the_voice() -> None:
    from mic_eq.ui.shell_specs import ACTION_SPECS

    assert ACTION_SPECS["auto_eq_button"].tip == (
        "Record a short passage and set the EQ toward a tone preset\n"
        "Choose a tone preset, read a short passage, then review the result"
    )


# The full correction leaves 3.4 dB too little room. How much of that is the recording and how
# much is what the correction adds decides what the microphone level can do about it.
def _split(status: str, scale: float, flat: float | None, **extra: object) -> dict:
    headroom = _headroom(status, scale, _render(-2.4), **extra)  # 1.0 - (-2.4) = 3.4 dB over
    if flat is not None:
        headroom["flat_excess_db"] = flat
    return headroom


_BOOST_LEADS = _split("safe", 0.40, 0.6)
_BOOST_ONLY = _split("safe", 0.40, -0.5)


def test_headroom_advice_cuts_the_microphone_only_by_the_recordings_part_when_the_boost_leads() -> None:
    assert headroom_advice(_BOOST_LEADS, _NO_DYNAMICS, proposed=True) == (
        "Most of the shortfall comes from the correction's own boost (2.8 of 3.4 dB), "
        "not from your recording. "
        "Lower the microphone gain by about 1 dB (in Windows or on the microphone) "
        "to give your own peaks enough room."
    )
    assert headroom_advice(_BOOST_LEADS, _NO_DYNAMICS, proposed=False) == (
        "Most of the shortfall comes from the correction's own boost (2.8 of 3.4 dB), "
        "not from your recording. "
        "Lower the microphone gain by about 1 dB (in Windows or on the microphone), "
        "then record again."
    )


def test_headroom_advice_leaves_the_microphone_alone_when_only_the_boost_is_too_much() -> None:
    assert headroom_advice(_BOOST_ONLY, _NO_DYNAMICS, proposed=True) == (
        "Most of the shortfall comes from the correction's own boost (3.4 of 3.4 dB), "
        "not from your recording. Your microphone level is fine."
    )


def test_headroom_advice_gives_no_microphone_figure_for_the_boost_when_a_compressor_works() -> None:
    chain = {"compressor": {"enabled": True}, "limiter": {"enabled": True}}

    assert headroom_advice(_BOOST_LEADS, chain, proposed=True) == (
        "Most of the shortfall comes from the correction's own boost (2.8 of 3.4 dB), "
        "not from your recording. "
        "Lower the microphone gain (in Windows or on the microphone) "
        "to give your own peaks enough room."
    )


def test_headroom_advice_keeps_the_full_figure_when_the_recording_leads() -> None:
    headroom = _split("safe", 0.40, 3.0)

    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=True) == (
        "Lower the microphone gain by about 4 dB (in Windows or on the microphone) "
        "to allow the full correction."
    )


def test_headroom_advice_does_not_split_when_the_current_eq_is_over_the_rule() -> None:
    headroom = _split(
        "risk_not_worsened", 1.0, 0.2, incumbent_excess_db=3.4, incumbent_output_true_peak_db=-0.6
    )

    assert headroom_advice(headroom, _NO_DYNAMICS, proposed=True) == (
        "Your current settings already leave too little room for loud peaks. "
        "This EQ is no worse than your current one. "
        "Lower the microphone gain by about 4 dB (in Windows or on the microphone) "
        "to leave enough room. " + _LIMITER_OFF_LINE
    )


def test_headroom_advice_names_the_target_loudness_for_the_recordings_part_with_automatic_makeup() -> None:
    assert headroom_advice(_BOOST_LEADS, _MAKEUP_ON, proposed=True) == (
        "Most of the shortfall comes from the correction's own boost (2.8 of 3.4 dB), "
        "not from your recording. "
        "Lower the compressor's target loudness to give your own peaks enough room. "
        "Automatic makeup would undo a lower microphone gain."
    )
