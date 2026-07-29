"""Tests for the real-speech auto-makeup evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mic_eq import CORE_AVAILABLE, simulate_auto_makeup_control


TOOL_PATH = (
    Path(__file__).parent.parent / "tools" / "evaluate_auto_makeup_real_speech.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_auto_makeup_real_speech", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def test_control_probabilities_interpolate_to_exact_block_count():
    result = evaluation._control_probabilities(
        np.asarray([0.0, 0.5, 1.0]),
        sample_count=4_800,
        block_count=10,
    )

    assert result.shape == (10,)
    assert np.all(np.diff(result) >= 0.0)
    assert 0.0 <= result[0] <= result[-1] <= 1.0


def test_pumping_score_prefers_two_to_eight_hz_modulation():
    time = np.arange(1_000) / evaluation.CONTROL_CADENCE_HZ
    fast = np.sin(2.0 * np.pi * 4.0 * time)
    slow = np.sin(2.0 * np.pi * 0.2 * time)

    assert evaluation._pumping_score(fast) > 5.0 * evaluation._pumping_score(slow)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")
def test_native_control_hook_returns_one_row_per_10ms_block():
    audio = np.zeros(1_440, dtype=np.float32)

    result = simulate_auto_makeup_control(
        audio,
        48_000.0,
        [0.0, 0.5, 1.0],
        -50.0,
        1.0,
    )

    assert result["control_block_size"] == 480
    assert len(result["makeup_gain_db"]) == 3
    assert len(result["activity"]) == 3
    assert result["p99_block_runtime_ms"] >= 0.0
