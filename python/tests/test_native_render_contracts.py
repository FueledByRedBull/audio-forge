"""Native render ownership and evaluator measurements at the Python boundary."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from mic_eq.config import EQSettings
from mic_eq.mic_eq_core import (
    simulate_auto_eq_chain,
    simulate_auto_makeup_control,
    simulate_eq_v2,
    simulate_gate_suppressor_order,
)


def _bands():
    bands = EQSettings().bands
    return (
        [(b.frequency_hz, b.gain_db, b.q) for b in bands],
        [(b.filter_type, b.frequency_hz, b.gain_db, b.q, b.slope_db_per_octave, b.enabled)
         for b in bands],
    )


@pytest.mark.parametrize("renderer", ["eq", "chain", "makeup", "frontend"])
def test_native_audio_arrays_survive_result_release(renderer):
    audio = (0.02 * np.sin(np.arange(4800) * 2 * np.pi / 48)).astype(np.float32)
    legacy, typed = _bands()
    if renderer == "eq":
        result = simulate_eq_v2(audio, 48_000.0, typed, True)
    elif renderer == "chain":
        result = simulate_auto_eq_chain(audio, 48_000.0, legacy, {
            "compressor_enabled": False, "limiter_enabled": False,
            "deesser_enabled": False, "return_output_audio": True,
        })
    elif renderer == "makeup":
        result = simulate_auto_makeup_control(audio, 48_000.0, [0.8] * 10, -60.0, 1.0,
                                             {"return_output_audio": True})
    else:
        result = simulate_gate_suppressor_order(audio, [0.8] * 10, False, 0.0,
                                                {"return_dry_audio": True})
    arrays = [result[key] for key in ("output_audio", "dry_audio") if key in result]
    assert arrays
    for samples in arrays:
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert samples.shape == audio.shape
        assert samples.flags.c_contiguous
        assert np.isfinite(samples).all()
    expected = [samples.copy() for samples in arrays]
    del result
    gc.collect()
    for samples, reference in zip(arrays, expected):
        np.testing.assert_array_equal(samples, reference)
    if renderer in {"eq", "chain"}:
        np.testing.assert_allclose(arrays[0], audio, atol=1e-6)


@pytest.mark.parametrize("quiet_amplitude", [0.0, 1e-7, None, 1e-3])
def test_native_silence_delta_reports_actual_boost_or_unavailable(quiet_amplitude):
    t = np.arange(24_000) / 48_000
    voice = 0.1 * np.sin(2 * np.pi * 1000 * t)
    quiet = (voice if quiet_amplitude is None else quiet_amplitude * np.sin(2 * np.pi * 1000 * t))
    audio = np.concatenate((quiet, voice)).astype(np.float32)
    legacy, _ = _bands()
    results = [simulate_auto_eq_chain(audio, 48_000.0, legacy, {
        "compressor_enabled": True, "compressor_ratio": 1.0,
        "compressor_makeup_gain_db": gain, "compressor_auto_makeup_enabled": False,
        "limiter_enabled": False, "deesser_enabled": False,
    }) for gain in (0.0, 6.0)]
    if quiet_amplitude in (0.0, 1e-7, None):
        assert all(result["silence_level_delta_db"] is None for result in results)
        assert all(result["silence_analysis_block_count"] == 0 for result in results)
    else:
        assert all(result["silence_analysis_block_count"] > 0 for result in results)
        assert results[0]["silence_level_delta_db"] == pytest.approx(0.0, abs=0.1)
        assert results[1]["silence_level_delta_db"] == pytest.approx(6.0, abs=0.3)


@pytest.mark.parametrize("adaptive", [False, True])
def test_offline_compressor_release_uses_the_selected_mode(adaptive):
    t = np.arange(24_000) / 48_000
    amplitude = np.where(t < 0.15, 0.5, 0.01)
    audio = (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    legacy, _ = _bands()
    outputs = [simulate_auto_eq_chain(audio, 48_000.0, legacy, {
        "compressor_enabled": True, "compressor_threshold_db": -30.0,
        "compressor_ratio": 4.0, "compressor_attack_ms": 1.0,
        "compressor_release_ms": 200.0, "compressor_base_release_ms": base,
        "compressor_adaptive_release": adaptive,
        "compressor_auto_makeup_enabled": False,
        "limiter_enabled": False, "deesser_enabled": False,
        "return_output_audio": True,
    })["output_audio"] for base in (40.0, 300.0)]
    if adaptive:
        assert np.max(np.abs(outputs[0] - outputs[1])) > 0.001
    else:
        np.testing.assert_array_equal(outputs[0], outputs[1])


def test_compressor_search_penalizes_measured_room_noise_boost(monkeypatch):
    from mic_eq.analysis.voice_setup import _calibrate_compressor_threshold

    has_silence = False

    def simulation(_audio, _rate, _eq, chain):
        threshold = chain["compressor"]["threshold_db"]
        return {
            "simulation_backend": "rust", "non_finite_output": False,
            "compressor_gain_reduction_db": 3.7,
            "compressor_gain_reduction_median_db": 1.4,
            "compressor_gain_reduction_p95_db": 3.5,
            "compressor_gain_reduction_active_ratio": 1.0,
            "active_output_gain_db": 0.0, "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "pre_limiter_true_peak_headroom_db": 2.0,
            "compressor_pumping_score_db": 0.0,
            "silence_level_delta_db": abs(threshold + 36.0) if has_silence else None,
        }

    monkeypatch.setattr("mic_eq.analysis.compressor_calibration.simulate_candidate_chain", simulation)

    def search():
        return _calibrate_compressor_threshold(
            speech_audio=np.zeros(4800, dtype=np.float32), sample_rate=48_000,
            eq_settings=EQSettings().to_dict(), deesser_settings={"enabled": False},
            compressor_settings={"threshold_db": -24.0, "ratio": 2.0,
                                 "attack_ms": 20.0, "release_ms": 300.0,
                                 "target_lufs": -18.0, "measured_short_term_lufs": -18.0},
            target_p95_db=3.5, target_median_db=1.4, peak_cap_db=8.0,
        )

    without_silence, absent = search()
    has_silence = True
    quiet_winner, measured = search()
    assert without_silence["threshold_db"] == -24.0
    assert absent["silence_level_delta_db"] is None
    assert abs(quiet_winner["threshold_db"] + 36.0) < 1.0
    assert measured["total_objective"] < measured["incumbent_objective"]


def test_pumping_metric_excludes_partial_trace_block():
    audio = np.concatenate((np.zeros(480), np.full(480, 0.1), [0.9])).astype(np.float32)
    legacy, _ = _bands()
    result = simulate_auto_eq_chain(audio, 48_000.0, legacy, {
        "compressor_enabled": True, "compressor_auto_makeup_enabled": False,
        "auto_makeup_activity": [(0.8, 1.0, -60.0, 1.0)] * 3,
        "limiter_enabled": False, "deesser_enabled": False,
    })
    assert result["analysis_block_ms"] == 10.0
    # The estimator needs three uniformly spaced trace samples, but only two
    # complete blocks exist. The one-sample tail must not manufacture a third.
    assert result["compressor_pumping_score_db"] == 0.0
