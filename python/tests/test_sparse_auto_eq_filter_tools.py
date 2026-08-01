"""Contracts for the offline sparse/type-selecting Auto-EQ evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = (
    Path(__file__).parent.parent / "tools" / "evaluate_sparse_auto_eq_filters.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_sparse_auto_eq_filters",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _flat_bands() -> list[tuple[str, float, float, float, int, bool]]:
    frequencies = [
        80.0,
        160.0,
        320.0,
        640.0,
        1280.0,
        2500.0,
        5000.0,
        8000.0,
        12_000.0,
        16_000.0,
    ]
    return [
        (
            "low_shelf" if index == 0 else "high_shelf" if index == 9 else "bell",
            frequencies[index],
            0.0,
            1.0,
            12,
            True,
        )
        for index in range(10)
    ]


def test_sparse_selector_removes_costly_flat_sections() -> None:
    grid = np.geomspace(80.0, 16_000.0, 128)
    view = TOOL.AnalysisView(
        grid,
        np.zeros_like(grid),
        np.ones_like(grid),
    )

    selected, trace = TOOL._select_sparse_candidate(
        view,
        _flat_bands(),
        np.ones(10),
    )

    assert TOOL._active_count(selected) == 0
    assert len(trace) == 10
    assert all(step["operation"] == "disable" for step in trace)


def test_notch_eligibility_requires_depth_width_and_confidence() -> None:
    eligible = ("bell", 2500.0, -6.0, 3.0, 12, True)

    assert "notch" in TOOL._eligible_replacements(4, eligible, 0.65)
    assert "notch" not in TOOL._eligible_replacements(
        4,
        ("bell", 2500.0, -5.99, 3.0, 12, True),
        0.65,
    )
    assert "notch" not in TOOL._eligible_replacements(
        4,
        ("bell", 2500.0, -6.0, 2.99, 12, True),
        0.65,
    )
    assert "notch" not in TOOL._eligible_replacements(4, eligible, 0.649)


def test_gate_accepts_a_fully_passing_objective_candidate() -> None:
    aggregate = {
        "comparable_cases": 24,
        "median_heldout_improvement_db": 0.1,
        "p10_heldout_improvement_db": 0.0,
        "median_stability_regression_db": 0.0,
        "p90_stability_regression_db": 0.0,
        "median_active_section_reduction": 2.0,
        "all_outputs_finite": True,
        "max_true_peak_regression_db": 0.0,
        "p95_limiter_gr_regression_db": 0.0,
        "max_full_chain_true_peak_overshoot_db": 0.0,
        "p95_runtime_ratio": 0.8,
        "candidate_p95_realtime_factor": 0.001,
        "latency_samples": [0],
        "native_constraints_valid": True,
    }

    checks = TOOL._gate(aggregate)

    assert all(checks.values())


def test_gate_rejects_heldout_lower_tail_regression() -> None:
    aggregate = {
        "comparable_cases": 24,
        "median_heldout_improvement_db": 0.1,
        "p10_heldout_improvement_db": -0.5,
        "median_stability_regression_db": 0.0,
        "p90_stability_regression_db": 0.0,
        "median_active_section_reduction": 2.0,
        "all_outputs_finite": True,
        "max_true_peak_regression_db": 0.0,
        "p95_limiter_gr_regression_db": 0.0,
        "max_full_chain_true_peak_overshoot_db": 0.0,
        "p95_runtime_ratio": 0.8,
        "candidate_p95_realtime_factor": 0.001,
        "latency_samples": [0],
        "native_constraints_valid": True,
    }

    checks = TOOL._gate(aggregate)

    assert checks["median_heldout_noninferior"] is True
    assert checks["lower_decile_heldout_noninferior"] is False


def test_gate_rejects_native_validation_failures() -> None:
    aggregate = {
        "comparable_cases": 24,
        "median_heldout_improvement_db": 0.1,
        "p10_heldout_improvement_db": 0.0,
        "median_stability_regression_db": 0.0,
        "p90_stability_regression_db": 0.0,
        "median_active_section_reduction": 2.0,
        "all_outputs_finite": True,
        "max_true_peak_regression_db": 0.0,
        "p95_limiter_gr_regression_db": 0.0,
        "max_full_chain_true_peak_overshoot_db": 0.0,
        "p95_runtime_ratio": 0.8,
        "candidate_p95_realtime_factor": 0.001,
        "latency_samples": [0],
        "native_constraints_valid": False,
    }

    checks = TOOL._gate(aggregate)

    assert checks["native_constraints"] is False
