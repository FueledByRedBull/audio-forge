from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

import self_test as TOOL


class _ProcessorStub:
    def __init__(self) -> None:
        self.queued_probe: np.ndarray | None = None

    def start_raw_recording(self, _duration: float) -> None:
        return None

    def queue_output_probe(self, probe: np.ndarray) -> None:
        self.queued_probe = np.asarray(probe)

    def is_output_probe_complete(self) -> bool:
        return self.queued_probe is not None

    def cancel_output_probe(self) -> None:
        return None

    def stop_raw_recording(self) -> np.ndarray:
        return np.zeros(4_800, dtype=np.float32)

    def get_runtime_diagnostics(self) -> dict[str, object]:
        return {}


def test_self_test_queues_resampled_probe_on_processor_route() -> None:
    processor = _ProcessorStub()
    result = TOOL._run_attempt(
        processor,
        duration=0.05,
        delay=0.0,
        capture_sample_rate=48_000,
        output_sample_rate=44_100,
        probe_duration_ms=10.0,
        expected_latency_min_ms=5.0,
        expected_latency_max_ms=50.0,
        expected_playback_jitter_ms=20.0,
    )

    assert processor.queued_probe is not None
    assert processor.queued_probe.dtype == np.float32
    assert processor.queued_probe.size == 441
    assert np.isfinite(result.round_trip_ms)


def test_retries_keep_the_full_latency_window_after_a_failed_attempt(
    monkeypatch,
) -> None:
    windows: list[tuple[float, float]] = []

    class Processor:
        def start(self, _input, _output):
            return "Started"

        def sample_rate(self):
            return 48_000

        def output_sample_rate(self):
            return 48_000

        def get_active_input_device(self):
            return "in"

        def get_active_output_device(self):
            return "out"

        def __getattr__(self, _name):
            return lambda *_args: None

    def run_attempt(_processor, **kwargs):
        windows.append(
            (kwargs["expected_latency_min_ms"], kwargs["expected_latency_max_ms"])
        )
        return TOOL.SelfTestAttempt(
            confidence=0.05,
            round_trip_ms=300.0,
            message="low confidence",
            diagnostics={},
            playback_delay_ms=0.0,
        )

    module = ModuleType("mic_eq")
    monkeypatch.setattr(module, "AudioProcessor", Processor, raising=False)
    monkeypatch.setitem(sys.modules, "mic_eq", module)
    monkeypatch.setattr(TOOL, "_run_attempt", run_attempt)
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        TOOL.sys, "argv", ["self_test.py", "--warmup", "0", "--retries", "2"]
    )

    assert TOOL.main() == 1
    assert windows == [(20.0, 500.0)] * 3
