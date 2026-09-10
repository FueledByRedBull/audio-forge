from __future__ import annotations

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
