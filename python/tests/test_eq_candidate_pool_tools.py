"""Tests for fixed-variant Auto-EQ candidate-pool evaluation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).parents[1] / "tools" / "evaluate_eq_candidate_pool.py"
SPEC = importlib.util.spec_from_file_location("evaluate_eq_candidate_pool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _variant(*, passing: bool) -> dict:
    return {
        "checks": {
            "median_improvement": passing,
            "improved_fraction": passing,
            "lower_tail": passing,
            "runtime": passing,
            "risk": passing,
            "constraints": passing,
        }
    }


def test_fixed_variant_selection_uses_smallest_complete_pass() -> None:
    variants = {
        "12": _variant(passing=False),
        "14": _variant(passing=True),
        "16": _variant(passing=True),
    }

    assert TOOL._select_fixed_variant(variants) == 14


def test_no_fixed_variant_pass_keeps_incumbent() -> None:
    variants = {
        "12": _variant(passing=False),
        "14": _variant(passing=False),
        "16": _variant(passing=False),
    }

    assert TOOL._select_fixed_variant(variants) is None
