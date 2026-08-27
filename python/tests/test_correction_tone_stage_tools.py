"""Contracts for the offline correction/tone-stage candidate."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


TOOL_PATH = (
    Path(__file__).parent.parent / "tools" / "evaluate_correction_tone_stages.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_correction_tone_stages",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _write_corpus(root: Path, *, tamper_first: bool = False) -> None:
    root.mkdir()
    captures = []
    for index in range(16):
        path = root / f"p{index % 2}_{index:03d}.wav"
        wavfile.write(path, 48_000, np.zeros(128, dtype=np.float32))
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
    if tamper_first:
        first = sorted(root.glob("*.wav"))[0]
        first.write_bytes(first.read_bytes() + b"tampered")


def test_combined_migration_preserves_native_response_exactly() -> None:
    tone = TOOL._tone_profiles()["presence"]
    payload = TOOL._migrate_combined(tone)
    correction, migrated_tone = TOOL._decode_candidate(payload)
    grid = np.geomspace(20.0, 20_000.0, 256)
    migrated = TOOL._combined_response(grid, correction, migrated_tone)
    incumbent = np.asarray(
        TOOL.eq_magnitude_response_v2(
            grid.tolist(),
            tone,
            float(TOOL.SAMPLE_RATE),
        )
    )

    np.testing.assert_allclose(migrated, incumbent, rtol=0.0, atol=1.0e-12)


def test_replacing_correction_preserves_canonical_tone_payload() -> None:
    tone = TOOL._tone_profiles()["warm"]
    payload = TOOL._migrate_combined(tone)
    replacement = TOOL._default_bands(enabled=False)
    replacement[4] = ("bell", 1500.0, -2.0, 1.4, 12, True)

    updated = TOOL._replace_correction(payload, replacement)
    correction, restored_tone = TOOL._decode_candidate(updated)

    assert correction == replacement
    assert restored_tone == tone
    assert json.dumps(updated["tone"], sort_keys=True) == json.dumps(
        payload["tone"],
        sort_keys=True,
    )


def test_candidate_schema_rejects_malformed_or_short_stages() -> None:
    payload = TOOL._migrate_combined(TOOL._tone_profiles()["flat"])
    payload["tone"] = payload["tone"][:-1]

    with pytest.raises(ValueError, match="exactly ten bands"):
        TOOL._decode_candidate(payload)


def test_gate_rejects_failed_cases_without_requiring_impossible_speedup() -> None:
    aggregate = {
        "case_count": 12,
        "tone_profiles": ["flat", "presence", "warm", "bass_cut"],
        "max_response_parity_delta_db": 0.0,
        "tone_payload_preserved": True,
        "schema_roundtrip": True,
        "all_outputs_finite": True,
        "max_true_peak_overshoot_db": 0.0,
        "p95_limiter_gr_db": 1.0,
        "candidate_p95_realtime_factor": 0.001,
        "p95_runtime_ratio": 2.0,
        "latency_samples": [0],
        "failed_cases": 1,
    }

    aggregate["incumbent_p95_realtime_factor"] = 0.0008
    checks = TOOL._gate(aggregate)

    assert checks["zero_failed_cases"] is False
    assert all(
        passed
        for name, passed in checks.items()
        if name != "zero_failed_cases"
    )


def test_corpus_cases_reject_manifest_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _write_corpus(root, tamper_first=True)

    with pytest.raises(ValueError, match="hash mismatch"):
        TOOL._corpus_cases(root)
