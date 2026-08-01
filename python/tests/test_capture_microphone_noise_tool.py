"""Tests for quiet-microphone capture qualification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = Path(__file__).parent.parent / "tools" / "capture_microphone_noise.py"
SPEC = importlib.util.spec_from_file_location("capture_microphone_noise", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_quiet_capture_passes_and_speech_contamination_fails() -> None:
    quiet = np.full(48_000 * 10, 1e-4, dtype=np.float32)
    passing = TOOL.assess_capture(quiet, [0.01] * 100)

    assert passing["passed"] is True
    contaminated = TOOL.assess_capture(quiet, [0.8] * 100)
    assert contaminated["passed"] is False
    assert contaminated["checks"]["speech_activity"] is False
    assert contaminated["checks"]["vad_p95"] is False


def test_loud_transient_capture_is_rejected() -> None:
    audio = np.full(48_000 * 10, 1e-4, dtype=np.float32)
    audio[100] = 0.5

    result = TOOL.assess_capture(audio, [0.01] * 100)

    assert result["passed"] is False
    assert result["checks"]["peak"] is False
