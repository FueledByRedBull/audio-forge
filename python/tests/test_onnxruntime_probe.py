"""Pure checks for the isolated ONNX Runtime probe evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_onnxruntime_probe.py"
SPEC = importlib.util.spec_from_file_location("evaluate_onnxruntime_probe", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_equal_posteriors_pass_all_decision_checks() -> None:
    values = np.asarray([0.1, 0.48, 0.9], dtype=np.float64)

    result = probe._compare_posteriors({"arr_0": values}, {"arr_0": values.copy()})

    assert result["frame_count"] == 3
    assert result["exact_equal_frames"] == 3
    assert result["max_abs_delta"] == 0.0
    assert all(change == 0 for change in result["threshold_changes"].values())
    assert result["hysteresis_surrogate"]["decision_changes"] == 0


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="no captures"):
        probe._compare_posteriors({}, {})


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape differs"):
        probe._compare_posteriors(
            {"arr_0": np.zeros(2)},
            {"arr_0": np.zeros(3)},
        )
