"""Tests for limiter-lookahead evaluation fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_limiter_lookahead.py"
SPEC = importlib.util.spec_from_file_location("evaluate_limiter_lookahead", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
limiter_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = limiter_eval
SPEC.loader.exec_module(limiter_eval)
REPORT_PATH = Path(__file__).resolve().parents[2] / "evaluation/limiter-lookahead-report.json"


def test_limiter_cases_are_finite_and_four_seconds_long():
    for audio in limiter_eval._cases().values():
        assert audio.shape == (limiter_eval.SAMPLE_RATE * 4,)
        assert audio.dtype.name == "float32"


def test_real_case_calibration_hits_pre_limiter_target(monkeypatch):
    def fake_render(audio, lookahead_ms):
        del lookahead_ms
        peak_db = 20.0 * np.log10(max(float(np.max(np.abs(audio))), 1e-12))
        return {
            "pre_limiter_true_peak_db": peak_db,
            "limiter_effective_ceiling_db": 0.0,
            "limiter_gain_reduction_db": max(0.0, peak_db),
        }

    monkeypatch.setattr(limiter_eval, "_render", fake_render)
    source = np.ones(4_800, dtype=np.float32)

    calibrated, provenance = limiter_eval._calibrate_real_case(source, {"id": "x"})

    measured = 20.0 * np.log10(float(np.max(np.abs(calibrated))))
    assert measured == pytest.approx(
        limiter_eval.REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB,
        abs=limiter_eval.REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB,
    )
    assert provenance["measured_main_limiter_gain_reduction_db"] == pytest.approx(
        limiter_eval.REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB,
        abs=limiter_eval.REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB,
    )


def test_gain_envelope_metric_ignores_static_gain_but_detects_modulation():
    reference = np.full(limiter_eval.SAMPLE_RATE, 0.25, dtype=np.float32)
    static = reference * 0.5
    time = np.arange(reference.size) / limiter_eval.SAMPLE_RATE
    modulated = reference * (0.6 + 0.3 * np.sin(2.0 * np.pi * 6.0 * time))

    assert limiter_eval._gain_envelope_variation_db(reference, static) < 1e-6
    assert limiter_eval._gain_envelope_variation_db(reference, modulated) > 1.0


def test_report_uses_real_speech_and_applies_objective_materiality_gate():
    import json

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["schema_version"] == 4
    assert report["corpus"]["real_speech"]["case_count"] == 12
    assert report["aggregates"]["2.0"]["all"]["all_finite"] is True
    assert report["selected_lookahead_ms"] == 2.0
    assert report["objective_candidate_lookahead_ms"] is None
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
        >= limiter_eval.REAL_MAIN_LIMITER_GAIN_REDUCTION_TARGET_DB
        - limiter_eval.REAL_MAIN_LIMITER_GAIN_REDUCTION_TOLERANCE_DB
    )
