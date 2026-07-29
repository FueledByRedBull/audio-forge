"""Tests for dynamics aliasing diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_dynamics_aliasing.py"
SPEC = importlib.util.spec_from_file_location("evaluate_dynamics_aliasing", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
aliasing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aliasing
SPEC.loader.exec_module(aliasing)


def test_alignment_recovers_small_positive_lag():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 0.1, 4_000)
    candidate = np.concatenate((np.zeros(17), reference))

    aligned_reference, aligned_candidate, lag = aliasing._align(reference, candidate)

    assert lag == 17
    assert np.allclose(aligned_reference, aligned_candidate)


def test_relative_error_is_very_low_for_identical_signals():
    signal = np.sin(np.linspace(0.0, 20.0, 4_000))

    assert aliasing._relative_error_db(signal, signal) < -200.0
