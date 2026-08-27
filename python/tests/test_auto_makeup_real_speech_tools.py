"""Tests for the real-speech auto-makeup evaluator."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from mic_eq import CORE_AVAILABLE
from mic_eq.mic_eq_core import simulate_auto_makeup_control


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


def _write_pair_manifest(root: Path) -> tuple[Path, Path]:
    clean = root / "Clean/english_room_snr0_0_mixture_clean.wav"
    noisy = root / "Noisy/english_room_snr0_0_mixture_noisy.wav"
    clean.parent.mkdir(parents=True)
    noisy.parent.mkdir(parents=True)
    wavfile.write(clean, 16_000, np.zeros(160, dtype=np.int16))
    wavfile.write(noisy, 16_000, np.zeros(160, dtype=np.int16))
    records = []
    for path, model in ((clean, "Clean"), (noisy, "Noisy")):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "model_name": model,
                "language": "english",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"files": records}), encoding="utf-8"
    )
    return clean, noisy


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


def test_pair_selection_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    _clean, noisy = _write_pair_manifest(tmp_path)
    noisy.write_bytes(noisy.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        evaluation._pairs(tmp_path, 1)


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
