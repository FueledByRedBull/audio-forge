"""Contracts for the typed manual-EQ retention evaluator."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_eq_filter_types.py"
SPEC = importlib.util.spec_from_file_location("evaluate_eq_filter_types", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_eq_filter_types = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_eq_filter_types"] = evaluate_eq_filter_types
SPEC.loader.exec_module(evaluate_eq_filter_types)


def _write_tone(path: Path, frequency_hz: float) -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    audio = (0.5 * np.sin(2.0 * np.pi * frequency_hz * time)).astype(
        np.float32
    )
    wavfile.write(path, sample_rate, audio)


def _write_manifest(root: Path) -> None:
    captures = []
    for path in sorted(root.glob("*.wav")):
        captures.append(
            {
                "clean": {
                    "path": f"{root.name}/{path.name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
        )
    (root.parent / "manifest.json").write_text(
        json.dumps({"captures": captures}),
        encoding="utf-8",
    )


def test_corpus_selection_round_robins_prefix_groups(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    root.mkdir()
    for name, frequency in (
        ("p1_001.wav", 400.0),
        ("p1_002.wav", 500.0),
        ("p2_001.wav", 600.0),
        ("p2_002.wav", 700.0),
    ):
        _write_tone(root / name, frequency)
    _write_manifest(root)

    selected = evaluate_eq_filter_types._select_corpus_files(root, 4)

    assert [path.name for path in selected] == [
        "p1_001.wav",
        "p2_001.wav",
        "p1_002.wav",
        "p2_002.wav",
    ]


def test_corpus_selection_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    root.mkdir()
    path = root / "p1_001.wav"
    _write_tone(path, 400.0)
    _write_manifest(root)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_eq_filter_types._select_corpus_files(root, 1)


def test_evaluator_components_cover_math_audio_headroom_and_realtime(
    tmp_path: Path,
) -> None:
    _write_tone(tmp_path / "p1_001.wav", 997.0)
    _write_tone(tmp_path / "p2_001.wav", 1400.0)

    analytic = evaluate_eq_filter_types._analytic_measurements(8)
    corpus = evaluate_eq_filter_types._corpus_measurements(
        sorted(tmp_path.glob("*.wav")),
        0.5,
    )
    headroom = evaluate_eq_filter_types._headroom_prediction_measurement()

    assert analytic["default_response_max_absolute_delta_db"] <= 1.0e-9
    assert analytic["max_cutoff_absolute_error_db"] <= 1.0e-7
    assert analytic["notch"]["center_response_db"] <= -150.0
    assert analytic["random_boundary_stress"]["nonfinite_cases"] == 0
    assert corpus["default_audio_max_absolute_delta"] == 0.0
    assert corpus["nonfinite_outputs"] == 0
    assert corpus["full_chain_nonfinite_outputs"] == 0
    assert corpus["full_chain_max_true_peak_overshoot_db"] <= 0.05
    assert corpus["stress_max_limiter_gain_reduction_db"] > 0.1
    assert corpus["algorithmic_latency_samples"] == [0]
    assert headroom["absolute_error_db"] <= 0.1
