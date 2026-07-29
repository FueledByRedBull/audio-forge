"""Tests for processing-order evaluation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_processing_order.py"
SPEC = importlib.util.spec_from_file_location("evaluate_processing_order", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
ordering = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ordering
SPEC.loader.exec_module(ordering)


def test_control_probability_mapping_has_one_value_per_rnnoise_frame():
    result = ordering._control_probabilities(np.asarray([0.0, 1.0]), 1_440)

    assert result.shape == (3,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_pumping_focuses_on_two_to_eight_hz():
    time = np.arange(1_000) * ordering.FRAME_SIZE / ordering.SAMPLE_RATE

    assert ordering._pumping(np.sin(2 * np.pi * 4 * time)) > ordering._pumping(
        np.sin(2 * np.pi * 0.2 * time)
    )
