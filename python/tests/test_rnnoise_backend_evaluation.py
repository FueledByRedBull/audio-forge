from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_tool() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "tools"
        / "evaluate_rnnoise_backends.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_rnnoise_backends", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def test_delay_estimator_finds_positive_model_delay() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 0.1, 48_000)
    delayed = np.concatenate((np.zeros(480), reference[:-480]))
    assert TOOL._delay_samples(reference, delayed) == 480


def test_decision_rejects_quality_win_with_runtime_and_size_regressions() -> None:
    shipped_rows = [
        {
            "si_sdr_improvement_db": 2.0,
            "noisy_speech_lsd_db": 3.0,
            "noisy_speech_dropout_rate": 0.0,
            "clean_si_sdr_db": 30.0,
            "clean_speech_lsd_db": 1.0,
            "clean_speech_dropout_rate": 0.0,
        }
    ]
    upstream_rows = [
        {
            "si_sdr_improvement_db": 3.0,
            "noisy_speech_lsd_db": 2.5,
            "noisy_speech_dropout_rate": 0.0,
            "clean_si_sdr_db": 30.0,
            "clean_speech_lsd_db": 1.0,
            "clean_speech_dropout_rate": 0.0,
        }
    ]
    shipped_runtime = {"frame_p99_seconds": 0.0001}
    upstream_runtime = {"frame_p99_seconds": 0.001}
    result = TOOL._decision(
        shipped_rows,
        upstream_rows,
        shipped_runtime,
        upstream_runtime,
        10_000_000,
    )
    assert result["gates"]["material_noisy_quality_win"] is True
    assert result["gates"]["p99_runtime_ratio_at_most_1_5"] is False
    assert result["gates"]["estimated_archive_growth_at_most_5_percent"] is False
    assert result["decision"] == "retain_nnnoiseless"
