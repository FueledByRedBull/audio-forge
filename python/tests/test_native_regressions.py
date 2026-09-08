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
