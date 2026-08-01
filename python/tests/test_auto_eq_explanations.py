"""Explainability tests that keep Auto-EQ UI mapping outside DSP policy."""

from __future__ import annotations

from copy import deepcopy

import pytest

from mic_eq.ui.auto_eq_explanation import explain_auto_eq
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
