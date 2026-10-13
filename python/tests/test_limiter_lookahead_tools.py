"""Tests for the retained limiter-lookahead evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parents[2] / "evaluation/limiter-lookahead-2026-10-report.json"
# Real-speech main-peak gain reduction the evaluator targeted, and its tolerance.
REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB = 3.0
REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB = 0.10


def test_report_uses_real_speech_and_applies_objective_materiality_gate():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["schema_version"] == 6
    assert report["corpus"]["real_speech"]["case_count"] == 12
    assert report["aggregates"]["2.0"]["all"]["all_finite"] is True
    assert report["selected_lookahead_ms"] == 0.5
    assert report["objective_candidate_lookahead_ms"] == 0.5
    assert (
        report["candidate_checks"]["0.5"]["paired_event_deltas"]["maximum"]
        <= 1
    )
    assert (
        report["candidate_checks"]["1.0"]["checks"][
            "material_latency_reduction_at_least_1_5_ms"
        ]
        is False
    )
    assert "pre_true_peak_overshoot_regression_db_max" in report["predefined_gates"]
    assert "pre_true_peak_overshoot_db_max" not in report["predefined_gates"]
    assert report["audible_change"] is True
    assert (
        report["aggregates"]["2.0"]["real_speech"][
            "minimum_main_peak_gain_reduction_db"
        ]
        >= REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB
        - REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB
    )
