"""
Non-blocking analysis worker for Auto-EQ calibration.

Runs audio analysis in background thread with step-by-step progress signals.
"""
import logging
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any, NamedTuple
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..analysis.auto_eq import analyze_auto_eq
from ..analysis.cancellation import AnalysisCancelled
from ..analysis.noise_reference import NoiseReferenceAnalysis


logger = logging.getLogger(__name__)

UNEXPECTED_FAILURE = "The analysis could not finish. The cause is in the log file."


class CancellableWorker(QThread):
    """A worker thread that can be asked to stop and reports an unexpected error once.

    Subclasses implement ``_run``. A cancelled run says nothing; any other exception is logged with
    its traceback (``_failure_log``) and reaches the user as ``failed``'s sentence.
    """

    failed = Signal(str)  # Emits error message (generic)
    _failure_log = "Analysis failed"

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request cooperative cancellation."""
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def _run(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        try:
            self._run()
        except AnalysisCancelled:
            return
        except Exception as exc:
            logger.exception(self._failure_log)
            self.failed.emit(str(exc) or UNEXPECTED_FAILURE)


class AnalysisWorker(CancellableWorker):
    """
    Worker thread for non-blocking audio analysis.

    Runs analysis pipeline in background, emits progress signals for
    each step. Provides smooth UX with step-by-step feedback.
    """

    # Signals
    step_progress = Signal(str, int)  # (step_name, percentage)
    result_ready = Signal(dict)        # Emits eq_settings dict
    _failure_log = "Auto-EQ analysis failed"

    def __init__(
        self,
        audio_data,
        sample_rate,
        target_preset='broadcast',
        target_mode="adaptive",
        smoothing_strength="conservative",
        chain_settings=None,
        headroom_candidate_base_eq_settings=None,
        headroom_audio_data=None,
        headroom_chain_settings=None,
    ):
        """
        Initialize analysis worker.

        Args:
            audio_data: Recorded audio samples (float32 NumPy array)
            sample_rate: Sample rate in Hz from the active processor
            target_preset: Target key from the built-in curve catalog
            target_mode: Target behavior ('adaptive' or 'static')
            smoothing_strength: Auto-EQ smoothing strength
            chain_settings: Current deterministic DSP settings for offline headroom simulation
        """
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.target_preset = target_preset
        self.target_mode = target_mode
        self.smoothing_strength = smoothing_strength
        self.chain_settings = chain_settings
        self.headroom_candidate_base_eq_settings = deepcopy(
            headroom_candidate_base_eq_settings
        )
        self.headroom_audio_data = (
            None
            if headroom_audio_data is None
            else np.ascontiguousarray(headroom_audio_data, dtype=np.float32).copy()
        )
        self.headroom_chain_settings = deepcopy(headroom_chain_settings)
        self._start_time = None

    def _run(self) -> None:
        """Emit progress for each step, then the EQ settings or a failure message."""
        self._start_time = time.time()
        if self._should_stop():
            return

        self.step_progress.emit("fitting the correction and checking headroom...", 0)
        eq_settings, validation = analyze_auto_eq(
            self.audio_data,
            self.sample_rate,
            self.target_preset,
            target_mode=self.target_mode,
            smoothing_strength=self.smoothing_strength,
            chain_settings=self.chain_settings,
            headroom_candidate_base_eq_settings=(
                self.headroom_candidate_base_eq_settings
            ),
            headroom_audio_data=self.headroom_audio_data,
            headroom_chain_settings=self.headroom_chain_settings,
            cancel_check=self._should_stop,
        )
        if self._should_stop():
            return

        if not validation.passed:
            self.failed.emit(validation.reason)
            return

        elapsed = time.time() - self._start_time

        # Log duration for debugging (optional)
        if elapsed > 0.5:
            logger.info("Analysis took %.2fs (target <500ms)", elapsed)

        self.result_ready.emit(eq_settings)


class CaptureAnalysis(NamedTuple):
    """What a finished take needs before a dialog can decide about it."""

    audio: np.ndarray
    silent: bool
    noise_quality: NoiseReferenceAnalysis | None = None


class CaptureAnalysisWorker(CancellableWorker):
    """Runs the post-processing of a finished take off the GUI thread."""

    result_ready = Signal(object)

    def __init__(self, analyze: Callable[[], Any]) -> None:
        super().__init__()
        self._analyze = analyze

    def run(self) -> None:
        """After ``stop`` the result is dropped; the call in flight cannot be interrupted."""
        if self._should_stop():
            return
        try:
            result = self._analyze()
        except Exception as exc:
            logger.exception("Capture post-processing failed")
            if not self._should_stop():
                self.failed.emit(str(exc) or UNEXPECTED_FAILURE)
            return
        if not self._should_stop():
            self.result_ready.emit(result)


def start_capture_analysis(
    dialog: Any,
    analyze: Callable[[], Any],
    on_result: Callable[[CaptureAnalysis], object],
    on_failed: Callable[[str], object],
) -> None:
    """Post-process a finished take on a worker that ``dialog`` tracks, cancels and waits for."""
    worker = CaptureAnalysisWorker(analyze)
    dialog.analysis_worker = worker
    dialog._analysis_workers.append(worker)
    worker.result_ready.connect(on_result)
    worker.failed.connect(on_failed)
    worker.finished.connect(
        lambda finished_worker=worker: dialog._on_analysis_thread_finished(finished_worker)
    )
    worker.start()
