"""Tests for Voice Setup analysis progress forwarding."""

from __future__ import annotations

import threading
import time

import numpy as np

from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.ui.voice_setup_dialog import VoiceSetupWorker


def _worker() -> VoiceSetupWorker:
    return VoiceSetupWorker(
        np.zeros(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        48_000,
        "broadcast",
        vad_available=False,
        dynamics_intensity="balanced",
        custom_target_p95_db=3.5,
        custom_peak_cap_db=8.0,
        limiter_settings={},
        noise_metadata=None,
        voice_metadata=None,
    )


def test_analysis_worker_forwards_backend_progress(qapp, monkeypatch):
    progress: list[tuple[str, int]] = []
    results: list[dict] = []

    def fake_analysis(*_args, progress_callback=None, **_kwargs):
        assert callable(progress_callback)
        progress_callback("Measuring captures...", 12)
        progress_callback("Tuning the voice chain...", 84)
        return {"status": "ok"}

    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.analyze_voice_setup", fake_analysis
    )
    worker = _worker()
    worker.step_progress.connect(lambda label, value: progress.append((label, value)))
    worker.result_ready.connect(results.append)

    worker.start()
    assert worker.wait(2_000)
    qapp.processEvents()

    assert progress == [
        ("Measuring captures...", 12),
        ("Tuning the voice chain...", 84),
    ]
    assert results == [{"status": "ok"}]


def test_analysis_worker_cancellation_still_stops_backend(qapp, monkeypatch):
    entered = threading.Event()
    results: list[dict] = []

    def blocking_analysis(*_args, cancel_check=None, **_kwargs):
        assert callable(cancel_check)
        entered.set()
        while not cancel_check():
            time.sleep(0.005)
        raise AnalysisCancelled("sentinel")

    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.analyze_voice_setup", blocking_analysis
    )
    worker = _worker()
    worker.result_ready.connect(results.append)

    worker.start()
    assert entered.wait(1.0)
    worker.stop()
    assert worker.wait(2_000)
    qapp.processEvents()

    assert results == []
