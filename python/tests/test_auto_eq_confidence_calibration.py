"""Tests for Auto-EQ confidence threshold calibration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = (
    Path(__file__).parent.parent / "tools" / "calibrate_auto_eq_confidence.py"
)
SPEC = importlib.util.spec_from_file_location("calibrate_auto_eq_confidence", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def test_classification_counts_and_scores():
    result = calibration._classification(
        np.asarray([0.9, 0.7, 0.4, 0.1]),
        np.asarray([True, False, True, False]),
        0.5,
    )

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_calibration_retains_current_without_both_validation_classes():
    rows = [
        {
            "split": "train" if index < 8 else "validation",
            "score": index / 10.0,
            "label": True,
        }
        for index in range(12)
    ]

    result = calibration._calibrate(
        rows,
        score_key="score",
        label_key="label",
        current_threshold=0.45,
    )

    assert result["selection"] == "current"
    assert result["selected_threshold"] == 0.45
