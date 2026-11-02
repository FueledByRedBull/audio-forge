"""Finish a take the way the dialogs now do: its post-processing runs on a worker."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QApplication

from mic_eq.ui.analysis_worker import CaptureAnalysisWorker


def settle(dialog: Any) -> None:
    """Join the take's post-processing, then deliver its queued results."""
    for worker in tuple(dialog._analysis_workers):
        if isinstance(worker, CaptureAnalysisWorker):
            worker.wait()
    app = QApplication.instance()
    assert app is not None
    app.processEvents()


def finish_take(dialog: Any, audio: np.ndarray) -> None:
    dialog._on_recording_complete(audio)
    settle(dialog)


RATE = 48_000


def _voice(level: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (level * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def _render(headroom: float, limiter: float = 0.0, true_peak: float = 0.0) -> dict[str, float]:
    return {
        "pre_limiter_true_peak_headroom_db": headroom,
        "limiter_gain_reduction_db": limiter,
        "true_peak_limiter_gain_reduction_db": true_peak,
    }


def _downstream_metrics(audio: np.ndarray, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
    del audio
    return {
        "simulation_backend": "rust",
        "compressor_gain_reduction_p95_db": 3.5,
        "compressor_gain_reduction_median_db": 1.47,
        "compressor_gain_reduction_db": 4.0,
        "compressor_pumping_score_db": 0.0,
        "output_true_peak_db": -1.0,
        "limiter_effective_ceiling_db": -0.5,
        "pre_limiter_true_peak_headroom_db": 1.0,
        "non_finite_output": False,
        "candidate_runtime_ms": 0.0,
    }
