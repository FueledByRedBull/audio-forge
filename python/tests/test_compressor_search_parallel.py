"""Parallel calibration must retain deterministic native results and cancellation."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mic_eq.analysis import compressor_calibration, voice_setup
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.analysis.parallel import ANALYSIS_WORKERS
from mic_eq.config import Preset


def _run(
    progress_callback=None,
    cancel_check=None,
    auto_makeup_activity=None,
):
    preset = Preset()
    t = np.arange(4800, dtype=np.float32) / 48000
    return voice_setup._calibrate_compressor_threshold(
        speech_audio=(0.12 * np.sin(2 * np.pi * 180 * t)).astype(np.float32),
        sample_rate=48000,
        eq_settings=preset.eq.to_dict(),
        deesser_settings=asdict(preset.deesser),
        compressor_settings=asdict(preset.compressor),
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8,
        auto_makeup_activity=auto_makeup_activity,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def test_parallel_compressor_search_matches_serial_native_results(monkeypatch):
    progress = []
    activity = [(0.85, 1.0, -46.0, 0.9)] * 10
    parallel, parallel_diag = _run(
        lambda text, value: progress.append(value),
        auto_makeup_activity=activity,
    )
    pool = compressor_calibration.ThreadPoolExecutor
    monkeypatch.setattr(compressor_calibration, "ThreadPoolExecutor", lambda **_: pool(max_workers=1))
    serial, serial_diag = _run(auto_makeup_activity=activity)
    assert parallel == serial
    for key in parallel_diag:
        if key != "search_runtime_ms":
            assert parallel_diag[key] == serial_diag[key], key
    assert progress == sorted(progress)
    assert len(progress) <= compressor_calibration._COMPRESSOR_SEARCH_BUDGET


def test_compressor_search_reuses_frontend_audio_and_activity(monkeypatch):
    observed = []
    preset = Preset()
    audio = np.linspace(-0.12, 0.12, 4800, dtype=np.float32)
    activity = [(0.85, 1.0, -46.0, 0.9)] * 10
    progress = []

    def fake_simulation(candidate_audio, _sample_rate, eq_settings, chain):
        observed.append(
            (
                np.asarray(candidate_audio).copy(),
                eq_settings,
                chain,
            )
        )
        return {
            "simulation_backend": "rust",
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": 0.0,
            "silence_level_delta_db": 0.0,
            "non_finite_output": False,
        }

    monkeypatch.setattr(compressor_calibration, "simulate_candidate_chain", fake_simulation)
    voice_setup._calibrate_compressor_threshold(
        speech_audio=audio,
        sample_rate=48_000,
        eq_settings=preset.eq.to_dict(),
        deesser_settings=asdict(preset.deesser),
        compressor_settings=asdict(preset.compressor),
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8.0,
        auto_makeup_activity=activity,
        progress_callback=lambda _text, value: progress.append(value),
    )

    assert len(observed) > 1
    assert progress == sorted(progress)
    for candidate_audio, eq_settings, chain in observed:
        np.testing.assert_array_equal(candidate_audio, audio)
        assert eq_settings == preset.eq.to_dict()
        assert chain["auto_makeup_activity"] is activity
        assert "vad_probabilities" not in chain


def test_compressor_search_cancels_between_small_batches():
    completed = []
    with pytest.raises(AnalysisCancelled):
        _run(lambda text, value: completed.append(value), lambda: bool(completed))
    assert 1 <= len(completed) <= ANALYSIS_WORKERS


# The results are pinned from the search as it stood before it was reduced to
# a threshold-only search, so the reduction must leave every output unchanged.
CALIBRATION_PIN = Path(__file__).parent / "fixtures" / "golden" / "compressor_calibration.json"
CALIBRATION_PIN_CASES = {
    "incumbent-threshold-wins": {},
    "far-incumbent-threshold": {"compressor": {"threshold_db": -45.0}},
    "controls-outside-the-search-bounds": {
        "compressor": {"threshold_db": -2.0, "ratio": 12.0, "attack_ms": 1.0, "release_ms": 500.0}
    },
    "frontend-activity": {"auto_makeup_activity": [(0.85, 1.0, -46.0, 0.9)] * 100},
}


def calibrate_pin_case(case):
    preset = Preset()
    rng = np.random.default_rng(7)
    t = np.arange(48000, dtype=np.float32) / 48000
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio = 0.2 * envelope * np.sin(2 * np.pi * 180 * t) + 0.01 * rng.standard_normal(t.size)
    options = CALIBRATION_PIN_CASES[case]
    progress: list[int] = []
    settings, diagnostics = voice_setup._calibrate_compressor_threshold(
        speech_audio=audio.astype(np.float32),
        sample_rate=48000,
        eq_settings=preset.eq.to_dict(),
        deesser_settings=asdict(preset.deesser),
        compressor_settings=asdict(preset.compressor) | options.get("compressor", {}),
        target_p95_db=3.5,
        target_median_db=1.4,
        peak_cap_db=8.0,
        auto_makeup_activity=options.get("auto_makeup_activity"),
        progress_callback=lambda _text, value: progress.append(value),
    )
    diagnostics.pop("search_runtime_ms")
    return {"settings": settings, "diagnostics": diagnostics, "progress": progress}


def assert_close(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            assert_close(actual[key], value, f"{path}/{key}")
    else:
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), path


@pytest.mark.parametrize("case", CALIBRATION_PIN_CASES)
def test_threshold_calibration_matches_the_pinned_result(case):
    pinned = json.loads(CALIBRATION_PIN.read_text(encoding="utf-8"))[case]
    result = calibrate_pin_case(case)

    assert result["progress"] == pinned["progress"]
    assert_close(result["settings"], pinned["settings"])
    assert_close(result["diagnostics"], pinned["diagnostics"])
