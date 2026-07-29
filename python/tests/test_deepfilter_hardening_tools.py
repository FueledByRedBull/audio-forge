"""Tests for the retained DeepFilter hardening evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_deepfilter_hardening.py"
SPEC = importlib.util.spec_from_file_location("evaluate_deepfilter_hardening", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
deepfilter_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = deepfilter_eval
SPEC.loader.exec_module(deepfilter_eval)


def test_speech_lsd_uses_matching_unpadded_frame_geometry():
    time = np.arange(deepfilter_eval.SAMPLE_RATE * 2) / deepfilter_eval.SAMPLE_RATE
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * time)

    assert deepfilter_eval._speech_lsd(audio, audio) == 0.0


def test_best_lag_recovers_known_sample_delay():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 0.1, deepfilter_eval.SAMPLE_RATE)
    output = np.concatenate((np.zeros(1_440), reference))

    assert deepfilter_eval._best_lag(reference, output) == 1_440


def _case(
    model: str,
    attenuation_db: float,
    beta: float,
    *,
    noisy_improvement: float,
    noisy_lsd: float,
    clean_lsd: float = 0.5,
    dropout: float = 0.0,
) -> dict:
    return {
        "model": model,
        "attenuation_limit_db": attenuation_db,
        "post_filter_beta": beta,
        "runtime": {
            "rtf": 0.1,
            "p99_frame_seconds": 0.001,
            "max_frame_seconds": 0.002,
        },
        "segments": [
            {
                "noisy_si_sdr_improvement_db": noisy_improvement,
                "noisy_speech_lsd_db": noisy_lsd,
                "clean_si_sdr_db": 30.0,
                "clean_speech_lsd_db": clean_lsd,
                "clean_dropout_rate": dropout,
            }
        ],
    }


def test_attenuation_selection_chooses_lowest_candidate_passing_all_gates():
    cases = []
    values = {
        12.0: (8.0, 7.8),
        20.0: (9.9, 7.2),
        30.0: (10.8, 7.0),
        80.0: (11.0, 7.2),
    }
    for attenuation, (improvement, lsd) in values.items():
        for model in deepfilter_eval.MODELS:
            cases.append(
                _case(
                    model,
                    attenuation,
                    0.0,
                    noisy_improvement=improvement,
                    noisy_lsd=lsd,
                )
            )

    selected, decision = deepfilter_eval._choose_attenuation(cases)

    assert selected == 30.0
    assert decision["candidates"]["20.0"]["passes"] is False
    assert decision["candidates"]["30.0"]["passes"] is True


def test_post_filter_is_rejected_when_one_quality_metric_regresses():
    cases = []
    for beta, improvement, lsd in [
        (0.0, 10.8, 7.0),
        (0.02, 11.0, 7.2),
        (0.05, 11.1, 7.3),
    ]:
        for model in deepfilter_eval.MODELS:
            cases.append(
                _case(
                    model,
                    30.0,
                    beta,
                    noisy_improvement=improvement,
                    noisy_lsd=lsd,
                )
            )

    selected, decision = deepfilter_eval._choose_post_filter(cases, 30.0)

    assert selected == 0.0
    assert decision["candidates"]["0.02"]["passes_upgrade_gate"] is False
    assert decision["candidates"]["0.05"]["passes_upgrade_gate"] is False
