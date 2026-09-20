"""Parallel calibration must retain deterministic native results and cancellation."""
from dataclasses import asdict

import numpy as np
import pytest

from mic_eq.analysis import voice_setup
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.config import Preset


def _run(progress_callback=None, cancel_check=None, vad_probabilities=None):
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
        vad_probabilities=vad_probabilities,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def test_parallel_compressor_search_matches_serial_native_results(monkeypatch):
    progress = []
    model_probabilities = np.linspace(0.1, 0.9, 6, dtype=np.float32)
    parallel, parallel_diag = _run(
        lambda text, value: progress.append(value),
        vad_probabilities=model_probabilities,
    )
    pool = voice_setup.ThreadPoolExecutor
    monkeypatch.setattr(voice_setup, "ThreadPoolExecutor", lambda **_: pool(max_workers=1))
    serial, serial_diag = _run(vad_probabilities=model_probabilities)
    assert parallel == serial
    for key in parallel_diag:
        if key != "search_runtime_ms":
            assert parallel_diag[key] == serial_diag[key], key
    assert progress == sorted(progress)
    assert len(progress) <= voice_setup._COMPRESSOR_SEARCH_BUDGET


def test_compressor_search_reuses_one_causal_vad_array(monkeypatch):
    observed: list[np.ndarray | None] = []
    simulate = voice_setup.simulate_candidate_chain

    def capture_vad(*args, **kwargs):
        chain = args[3]
        values = chain.get("vad_probabilities")
        observed.append(None if values is None else np.asarray(values).copy())
        return simulate(*args, **kwargs)

    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", capture_vad)
    model_probabilities = np.linspace(0.1, 0.9, 6, dtype=np.float32)
    _run(vad_probabilities=model_probabilities)

    expected = voice_setup.map_causal_vad_probabilities(
        model_probabilities,
        np.minimum(
            np.arange(1, 11, dtype=np.int64) * 480,
            4_800,
        ),
        48_000,
    )
    assert expected is not None
    assert len(observed) > 1
    assert all(values is not None for values in observed)
    for values in observed:
        assert values is not None
        assert np.array_equal(values, expected)


def test_compressor_search_cancels_between_small_batches():
    completed = []
    with pytest.raises(AnalysisCancelled):
        _run(lambda text, value: completed.append(value), lambda: bool(completed))
    assert 1 <= len(completed) <= 4
