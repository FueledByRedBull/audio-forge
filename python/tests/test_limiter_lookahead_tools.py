"""Tests for limiter-lookahead evaluation fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_limiter_lookahead.py"
SPEC = importlib.util.spec_from_file_location("evaluate_limiter_lookahead", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
limiter_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = limiter_eval
SPEC.loader.exec_module(limiter_eval)


def test_limiter_cases_are_finite_and_four_seconds_long():
    for audio in limiter_eval._cases().values():
        assert audio.shape == (limiter_eval.SAMPLE_RATE * 4,)
        assert audio.dtype.name == "float32"
