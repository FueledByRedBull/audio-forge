"""Tests for native-full-band DeepFilter evaluation math."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_deepfilter_fullband.py"
SPEC = importlib.util.spec_from_file_location("evaluate_deepfilter_fullband", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/deepfilter-fullband-report.json"
)


def test_band_noise_is_deterministic_and_spectrally_bounded() -> None:
    first = TOOL._band_noise(48_000, 12_000.0, 16_000.0, 123)
    second = TOOL._band_noise(48_000, 12_000.0, 16_000.0, 123)
    frequencies = np.fft.rfftfreq(first.size, d=1.0 / 48_000)
    magnitude = np.abs(np.fft.rfft(first))

    assert np.array_equal(first, second)
    assert math.isclose(TOOL._rms(first), 1.0, rel_tol=1e-9)
    assert float(np.max(magnitude[(frequencies < 12_000) | (frequencies > 16_000)])) < 1e-9


def test_mix_at_snr_preserves_reference_scaling_and_headroom() -> None:
    clean = 0.2 * np.sin(
        2.0 * np.pi * 1_000.0 * np.arange(48_000) / 48_000.0
    )
    noise = TOOL._band_noise(48_000, 8_000.0, 12_000.0, 456)

    reference, mixture = TOOL._mix_at_snr(clean, noise, 10.0)
    added = mixture - reference

    assert abs(TOOL._db_ratio(TOOL._rms(reference), TOOL._rms(added)) - 10.0) < 1e-6
    assert float(np.max(np.abs(mixture))) <= 0.98 + 1e-12


def test_relative_gate_rejects_material_regressions() -> None:
    baseline = {
        "clean_p10_si_sdr_db": 30.0,
        "clean_p90_hf_lsd_db": 2.0,
        "environmental_median_improvement_db": 8.0,
        "hiss_median_improvement_db": 4.0,
        "noise_only_median_attenuation_db": 12.0,
    }
    candidate = dict(baseline)
    assert all(TOOL._relative_checks(candidate, baseline).values())

    candidate["clean_p90_hf_lsd_db"] = 2.6
    checks = TOOL._relative_checks(candidate, baseline)
    assert checks["clean_hf_lsd"] is False


def test_speech_activity_interpolates_vad_at_analysis_frame_centres() -> None:
    reference = np.ones(4_800, dtype=np.float64)
    probabilities = np.asarray([0.4, 0.6, 0.4], dtype=np.float64)

    active = TOOL._speech_active(reference, probabilities)

    # A stepwise owning-window mapping would activate three consecutive frames.
    # Centre interpolation also retains the preceding transitional frame.
    assert np.flatnonzero(active).tolist() == [2, 3, 4, 5]


def test_dropout_ignores_frames_outside_vad_active_speech() -> None:
    reference = np.ones(4_800, dtype=np.float64)
    estimate = reference.copy()
    estimate[:960] = 0.0
    probabilities = np.ones(4, dtype=np.float64)
    probabilities[0] = 0.0

    assert TOOL._dropout_rate(reference, estimate, probabilities) == 0.0
    assert TOOL._dropout_rate(reference, estimate) > 0.0


def test_objective_selection_requires_every_stratum_to_pass() -> None:
    decisions: list[dict[str, Any]] = [
        {
            "arms": {
                "20.0": {"passed": True},
                "30.0": {"passed": True},
                "80.0": {"passed": False},
            }
        },
        {
            "arms": {
                "20.0": {"passed": False},
                "30.0": {"passed": True},
                "80.0": {"passed": True},
            }
        },
    ]
    assert TOOL._select_objective_attenuation(decisions) == 30.0

    decisions.append({"arms": {"80.0": {"passed": True}}})
    assert TOOL._select_objective_attenuation(decisions) is None


def test_noise_only_attenuation_is_diagnostic_not_a_default_selector() -> None:
    baseline = {
        "clean_p10_si_sdr_db": 30.0,
        "clean_p90_hf_lsd_db": 2.0,
        "environmental_median_improvement_db": 8.0,
        "hiss_median_improvement_db": 4.0,
        "noise_only_median_attenuation_db": 20.0,
    }
    candidate = dict(baseline)
    candidate["noise_only_median_attenuation_db"] = 10.0

    checks = TOOL._relative_checks(candidate, baseline)
    release_checks = {
        name: passed
        for name, passed in checks.items()
        if name in TOOL.RELATIVE_DECISION_CHECKS
    }

    assert checks["noise_only_attenuation"] is False
    assert all(release_checks.values())


def test_runtime_gate_uses_p99_not_scheduler_sensitive_hard_maximum() -> None:
    aggregate = {
        "clean_p10_si_sdr_db": 30.0,
        "clean_p90_lsd_db": 1.0,
        "clean_p90_hf_lsd_db": 2.0,
        "clean_max_dropout_rate": 0.0,
        "environmental_median_improvement_db": 4.0,
        "environmental_p10_improvement_db": 1.0,
        "hiss_median_improvement_db": 2.0,
        "hiss_p10_improvement_db": 0.0,
        "noise_only_median_attenuation_db": 6.0,
        "p99_frame_seconds": 0.007,
        "max_frame_seconds": 0.5,
        "clipped_samples": 0.0,
        "non_finite_samples": 0.0,
    }

    checks = TOOL._absolute_checks(aggregate, 1.0)

    assert checks["p99_realtime"] is True
    assert "max_realtime" not in checks
    aggregate["p99_frame_seconds"] = 0.009
    assert TOOL._absolute_checks(aggregate, 1.0)["p99_realtime"] is False


def test_report_retains_30_db_under_objective_policy() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "objective-retained-incumbent"
    assert report["decision"]["release_selected_attenuation_limit_db"] == 30.0
    assert report["decision"]["retained"] is True
