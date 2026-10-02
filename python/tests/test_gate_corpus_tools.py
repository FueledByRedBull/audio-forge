"""Contracts for the EARS/DEMAND gate evaluation tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import evaluate_gate_corpus as gate_eval


def _speech(seconds: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    time = np.arange(int(seconds * gate_eval.SAMPLE_RATE)) / gate_eval.SAMPLE_RATE
    # Syllable-rate envelope with short inter-word dips, like dense speech.
    envelope = 0.05 + np.maximum(np.sin(2 * np.pi * 3.0 * time), 0.0)
    return 0.05 * rng.standard_normal(time.size) * envelope


def test_inserted_pauses_are_silent_and_labelled_exactly():
    clean, labels = gate_eval._with_pauses(_speech(), np.random.default_rng(0))

    assert clean.shape == labels.shape
    assert np.all(clean[~labels] == 0.0)
    assert labels.mean() < 0.9
    onsets = np.flatnonzero(labels[1:] & ~labels[:-1])
    assert onsets.size >= 2


def test_scores_are_neutral_for_unprocessed_input_and_reward_quiet_pauses():
    clean, labels = gate_eval._with_pauses(_speech(), np.random.default_rng(0))
    noise = 0.002 * np.random.default_rng(2).standard_normal(clean.size)
    case = {"clean": clean, "noisy": clean + noise, "labels": labels}

    unprocessed = gate_eval.score(case, case["noisy"])
    assert unprocessed["pause_db"] == pytest.approx(0.0)
    assert unprocessed["tail_db"] == pytest.approx(0.0)

    gated = np.where(labels, case["noisy"], 0.1 * case["noisy"])
    scored = gate_eval.score(case, gated)
    assert scored["pause_db"] == pytest.approx(-20.0, abs=0.5)
    assert scored["estoi"] == pytest.approx(unprocessed["estoi"], abs=0.02)


def test_vad_control_only_uses_windows_finished_before_each_block():
    windows = np.arange(1, 11, dtype=np.float32) / 10  # window k reports (k + 1) / 10
    control = gate_eval.causal_control(windows, 10 * gate_eval.VAD_WINDOW_SAMPLES)
    for block, value in enumerate(control):
        block_start = block * gate_eval.FRAME
        if value == 0.0:
            assert block_start < gate_eval.VAD_WINDOW_SAMPLES
            continue
        window = round(float(value) * 10) - 1
        # The window ended by the block's first sample, and it is the latest such.
        assert (window + 1) * gate_eval.VAD_WINDOW_SAMPLES <= block_start
        assert (window + 2) * gate_eval.VAD_WINDOW_SAMPLES > block_start


def _mock_rows(adjust) -> dict:
    rows: dict = {"identity": {}, "rows": {}}
    for model in gate_eval.QUALIFY_MODELS:
        table = rows["rows"][model] = {}
        for speaker in range(12):
            for snr in gate_eval.SNRS_DB:
                for mode in gate_eval.MODES:
                    for available in (True, False):
                        key = f"s{speaker:02d}|ENV|{snr}|{mode}|{available}"
                        metrics = {metric: 0.0 for metric in gate_eval.METRICS}
                        metrics["estoi"] = 0.7 + 0.01 * speaker
                        table[key] = adjust(key, metrics) | {"control_sha256": key}
    return rows


def test_qualification_adopts_a_complete_improvement():
    # No deep pauses in either build is undefined, not missing.
    no_deep = {"deep_pause_db": float("nan"), "floor_modulation_db": float("nan")}
    old = _mock_rows(lambda key, m: m | no_deep)
    new = _mock_rows(lambda key, m: m | no_deep | {"estoi": m["estoi"] + 0.01, "onset_db": 2.0})

    verdict = gate_eval.compare_rows(old, new)["verdict"]

    assert verdict["vad_only"]["adopt"] and verdict["vad_assisted"]["adopt"]


def test_qualification_rejects_results_that_went_missing():
    old = _mock_rows(lambda key, m: m)
    new = _mock_rows(
        lambda key, m: m | {
            "estoi": float("nan") if int(key[1:3]) % 2 else m["estoi"] + 0.01,
            "onset_db": 2.0,
        }
    )

    comparison = gate_eval.compare_rows(old, new)

    assert not comparison["verdict"]["vad_only"]["adopt"]
    assert not comparison["verdict"]["vad_only"]["rnnoise_available_results_complete"]
    assert comparison["invalid_cases"]["rnnoise"]["vad_only_available"] == 18


def test_qualification_rejects_rule_metrics_missing_from_both_builds():
    def drop_odd_speakers(key, m):
        return m | {"estoi": float("nan")} if int(key[1:3]) % 2 else m

    old = _mock_rows(drop_odd_speakers)
    new = _mock_rows(lambda key, m: drop_odd_speakers(key, m | {"estoi": m["estoi"] + 0.01, "onset_db": 2.0}))

    comparison = gate_eval.compare_rows(old, new)

    assert not comparison["verdict"]["vad_only"]["adopt"]
    assert not comparison["verdict"]["vad_assisted"]["adopt"]
    assert comparison["invalid_cases"]["deepfilter-ll"]["vad_assisted_unavailable"] == 18


def test_build_subprocess_receives_the_requested_corpus(monkeypatch, tmp_path):
    seen = {}

    def fake_run(command, check, env):
        seen["command"] = command
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"identity": {}, "rows": {}}', encoding="utf-8")

    monkeypatch.setattr(gate_eval.subprocess, "run", fake_run)
    gate_eval.rows_with_build(tmp_path / "build", tmp_path / "fresh-corpus", ["--split", "test"])

    command = seen["command"]
    assert command[command.index("--root") + 1] == str(tmp_path / "fresh-corpus")
