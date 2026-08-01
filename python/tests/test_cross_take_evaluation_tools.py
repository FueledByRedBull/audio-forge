"""Tests for cross-take Auto-EQ evaluation gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


TOOL_PATH = (
    Path(__file__).parents[1] / "tools" / "evaluate_cross_take_auto_eq.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_cross_take_auto_eq",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _row(improvement: float = 0.25, *, speaker: str = "actor-01") -> dict:
    return {
        "speaker": speaker,
        "single": {
            "settings": {"candidate": "single"},
            "heldout_target_error_db": 3.0,
            "runtime_seconds": 0.4,
            "recommendation_status": "apply",
        },
        "cross_take": {
            "settings": {"candidate": "cross"},
            "heldout_target_error_db": 3.0 - improvement,
            "runtime_seconds": 0.8,
            "recommendation_status": "apply",
            "headroom_safe": True,
        },
    }


def test_aggregate_and_gate_retain_noninferior_cross_take_candidate() -> None:
    aggregate = TOOL._aggregate(
        [
            _row(speaker=f"actor-{speaker:02d}")
            for speaker in range(1, 7)
            for _fold in range(2)
        ]
    )
    checks = TOOL._gate(aggregate)

    assert aggregate["median_heldout_improvement_db"] == 0.25
    assert aggregate["runtime_p95_ratio"] == 2.0
    assert aggregate["comparable_speaker_count"] == 6
    assert all(checks.values())


def test_lower_tail_regression_rejects_candidate() -> None:
    rows = [
        _row(speaker=f"actor-{speaker:02d}")
        for speaker in range(1, 6)
        for _fold in range(2)
    ]
    rows.extend(
        [_row(-2.0, speaker="actor-06"), _row(-2.0, speaker="actor-06")]
    )

    aggregate = TOOL._aggregate(rows)
    checks = TOOL._gate(aggregate)

    assert checks["median_heldout_noninferior"] is True
    assert checks["lower_decile_heldout_noninferior"] is False


def test_statement_folds_never_reuse_the_tuning_statement() -> None:
    folds = TOOL._statement_folds({"01", "02"})

    assert folds == [("01", "02"), ("02", "01")]
    assert all(tuning != heldout for tuning, heldout in folds)


def test_manifest_take_rejects_substituted_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "take.wav"
    TOOL.wavfile.write(
        audio_path,
        48_000,
        np.zeros(48_000, dtype=np.int16),
    )
    pair = {
        "id": "pair-01",
        "sample_rate": 48_000,
        "takes": {
            "01": {
                "path": audio_path.name,
                "frames": 48_000,
                "sha256": "0" * 64,
            }
        },
    }

    with pytest.raises(ValueError, match="Corpus hash mismatch"):
        TOOL._read_manifest_take(tmp_path, pair, "01", {})
