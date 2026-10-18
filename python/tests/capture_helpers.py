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
