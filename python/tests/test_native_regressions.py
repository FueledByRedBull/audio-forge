"""Regression and responsiveness checks for the native offline APIs."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mic_eq.mic_eq_core import (
    analyze_vad_probabilities,
    simulate_auto_eq_chain,
    simulate_auto_makeup_control,
    simulate_eq_v2,
    simulate_gate_suppressor_order,
)


SAMPLE_RATE = 48_000.0
EQ_BANDS = [
    (frequency, 0.0, 0.707)
    for frequency in (
        32.0,
        64.0,
        125.0,
        250.0,
        500.0,
        1_000.0,
        2_000.0,
        4_000.0,
        8_000.0,
        16_000.0,
    )
]


@pytest.mark.parametrize("mode", ["normal", "bypass", "raw"])
@pytest.mark.parametrize("limiter_enabled", [False, True])
def test_full_chain_final_output_is_bounded(mode, limiter_enabled):
    audio = np.full(960, 1.5, dtype=np.float32)
    result = simulate_auto_eq_chain(audio, SAMPLE_RATE, EQ_BANDS, {
        "full_chain": True, "processing_mode": mode, "input_pre_filtered": True,
        "vad_available": False, "gate_enabled": False, "suppressor_enabled": False,
        "eq_enabled": False, "compressor_enabled": False, "deesser_enabled": False,
        "limiter_enabled": limiter_enabled, "limiter_ceiling_db": -6.0,
        "limiter_careful_output_enabled": False, "return_output_audio": True,
    })
    ceiling = 10 ** (-6 / 20) if limiter_enabled else 1.0
    assert max(abs(sample) for sample in result["output_audio"]) <= ceiling + 1e-6
    assert result["limiter_effective_ceiling_db"] == pytest.approx(
        -6.0 if limiter_enabled else 0.0
    )
    if mode == "raw":
        assert result["chain_latency_samples"] == (320 if limiter_enabled else 0)
        assert result["limiter_gain_reduction_db"] == 0


@pytest.mark.parametrize("gain_db,safe", [(12.0, False), (0.0, True)])
def test_pre_limiter_headroom_is_measured_before_any_protection(gain_db, safe):
    from mic_eq.analysis.auto_eq_parts.headroom import _is_headroom_safe

    time_axis = np.arange(4800, dtype=np.float32) / SAMPLE_RATE
    audio = (0.4 * np.sin(2 * np.pi * 1000 * time_axis)).astype(np.float32)
    results = [simulate_auto_eq_chain(audio, SAMPLE_RATE, EQ_BANDS, {
        "eq_enabled": False, "compressor_enabled": True,
        "compressor_ratio": 1.0, "compressor_makeup_gain_db": gain_db,
        "compressor_auto_makeup_enabled": False, "deesser_enabled": False,
        "limiter_enabled": enabled, "limiter_ceiling_db": -1.5,
        "limiter_careful_output_enabled": False, "return_output_audio": True,
    }) for enabled in (False, True)]
    for enabled, result in zip((False, True), results):
        assert result["pre_limiter_true_peak_db"] == pytest.approx(
            20 * np.log10(0.4) + gain_db, abs=0.02
        )
        assert _is_headroom_safe(result) is safe
        assert max(abs(sample) for sample in result["output_audio"]) <= 1.0
        assert (result["true_peak_limiter_input_db"] is None) is (not enabled)
    assert results[0]["pre_limiter_true_peak_db"] == pytest.approx(
        results[1]["pre_limiter_true_peak_db"], abs=1e-5
    )
    if not safe:
        assert results[1]["output_true_peak_db"] < results[0]["output_true_peak_db"]


def test_hot_eq_candidate_cannot_pass_headroom_with_limiter_disabled():
    from mic_eq.analysis.auto_eq_parts.headroom import _is_headroom_safe

    audio = (0.4 * np.sin(2 * np.pi * 1000 * np.arange(4800) / SAMPLE_RATE)).astype(np.float32)
    bands = EQ_BANDS.copy()
    bands[5] = (1000.0, 12.0, 0.707)
    result = simulate_auto_eq_chain(audio, SAMPLE_RATE, bands, {
        "eq_enabled": True, "compressor_enabled": False, "deesser_enabled": False,
        "limiter_enabled": False, "return_output_audio": True,
    })
    assert result["pre_limiter_true_peak_db"] > 3.9
    assert max(result["output_audio"]) == 1.0
    assert not _is_headroom_safe(result)


@pytest.mark.parametrize("enabled", [False, True])
def test_raw_bounded_hot_signal_preserves_only_output_safety(enabled):
    audio = np.full(480, 0.95, dtype=np.float32)
    settings = {
        "full_chain": True, "processing_mode": "raw", "eq_enabled": True,
        "deesser_enabled": True, "compressor_enabled": True,
        "limiter_enabled": enabled, "limiter_ceiling_db": -1.5,
        "limiter_careful_output_enabled": False, "return_output_audio": True,
    }
    result = simulate_auto_eq_chain(audio, SAMPLE_RATE, EQ_BANDS, settings)
    assert result["compressor_gain_reduction_db"] == 0
    assert result["deesser_gain_reduction_db"] == 0
    if enabled:
        assert max(result["output_audio"]) <= 10 ** (-1.5 / 20) + 1e-6
        assert result["true_peak_limiter_gain_reduction_db"] > 0
    else:
        np.testing.assert_array_equal(result["output_audio"], audio)


@pytest.mark.parametrize("sample_rate", [16_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0])
@pytest.mark.parametrize("release_ms", [5.0, 80.0, 500.0])
def test_native_limiter_pair_bounds_short_burst_on_independent_references(sample_rate, release_ms):
    from scipy.signal import firwin, upfirdn

    audio = np.zeros(4096, dtype=np.float32)
    # A bounded near-Nyquist burst straddles a 480-sample block boundary.
    # This exposed a multi-dB overrun hidden by the original native detector.
    audio[479:527] = np.sin(2 * np.pi * .49 * np.arange(48) + .7)
    bands = [(100.0 * (index + 1), 0.0, 1.0) for index in range(10)]
    result = simulate_auto_eq_chain(audio, sample_rate, bands, {
        "eq_enabled": False, "compressor_enabled": False, "deesser_enabled": False,
        "limiter_enabled": True, "limiter_ceiling_db": -1.5,
        "limiter_release_ms": release_ms, "limiter_careful_output_enabled": False,
        "return_output_audio": True,
    })
    output = np.asarray(result["output_audio"], dtype=np.float64)
    assert output.shape == audio.shape
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) > .1  # Reject a silent or unflushed result.
    for taps in (2047, 4095):
        coefficients = 16 * firwin(taps, 1 / 16, window="blackman")
        # upfirdn includes the complete independent reconstruction tail.
        peak = np.max(np.abs(upfirdn(coefficients, output, up=16)))
        peak_db = 20 * np.log10(peak)
        assert peak_db <= -1.4, (sample_rate, release_ms, taps, peak_db)


def test_offline_native_apis_reject_nonfinite_audio_and_unsafe_rates() -> None:
    audio = np.zeros(48_000, dtype=np.float32)
    invalid_audio = np.array([np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="finite samples"):
        simulate_auto_eq_chain(invalid_audio, SAMPLE_RATE, EQ_BANDS)
    with pytest.raises(ValueError, match="finite and between"):
        simulate_auto_eq_chain(audio, 40.0, EQ_BANDS)
    with pytest.raises(ValueError, match="finite and between"):
        simulate_eq_v2(audio, 40.0, [])
    with pytest.raises(ValueError, match="finite samples"):
        simulate_auto_makeup_control(invalid_audio, SAMPLE_RATE, [], -60.0, 1.0)
    with pytest.raises(ValueError, match="must be finite"):
        simulate_auto_makeup_control(
            audio,
            SAMPLE_RATE,
            [],
            -60.0,
            1.0,
            {"makeup_gain_db": float("nan")},
        )
    with pytest.raises(ValueError, match="finite samples"):
        simulate_gate_suppressor_order(invalid_audio, [], False)
    with pytest.raises(ValueError, match="threshold must be finite"):
        analyze_vad_probabilities(audio, 48_000, float("nan"))


def test_auto_eq_native_work_does_not_starve_python_heartbeat() -> None:
    audio = np.sin(
        2.0 * np.pi * 220.0 * np.arange(480_000, dtype=np.float32) / SAMPLE_RATE
    ).astype(np.float32)
    heartbeat_times: list[float] = []
    ready = threading.Event()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            heartbeat_times.append(time.perf_counter())
            ready.set()
            time.sleep(0.005)

    worker = threading.Thread(target=heartbeat)
    worker.start()
    assert ready.wait(timeout=1.0)
    started = time.perf_counter()
    try:
        result = simulate_auto_eq_chain(audio, SAMPLE_RATE, EQ_BANDS)
    finally:
        finished = time.perf_counter()
        stop.set()
        worker.join()

    elapsed = finished - started
    in_call = [
        timestamp for timestamp in heartbeat_times if started <= timestamp <= finished
    ]
    timeline = [started, *in_call, finished]
    gaps = [right - left for left, right in zip(timeline, timeline[1:])]
    assert result["non_finite_output"] is False
    assert max(gaps, default=elapsed) < 0.1
