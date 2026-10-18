"""
Non-blocking analysis worker for Auto-EQ calibration.

Runs audio analysis in background thread with step-by-step progress signals.
"""
import logging
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..analysis.auto_eq import analyze_auto_eq
from ..analysis.cancellation import AnalysisCancelled


logger = logging.getLogger(__name__)

UNEXPECTED_FAILURE = "The analysis could not finish. The cause is in the log file."


class AnalysisWorker(QThread):
    """
    Worker thread for non-blocking audio analysis.

    Runs analysis pipeline in background, emits progress signals for
    each step. Provides smooth UX with step-by-step feedback.
    """

    # Signals
    step_progress = Signal(str, int)  # (step_name, percentage)
    result_ready = Signal(dict)        # Emits eq_settings dict
    failed = Signal(str)               # Emits error message (generic)

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
        self._stop_event = threading.Event()

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        """
        Run analysis in background thread.

        Called by QThread.start(). Emits progress signals for each step,
        then finished with EQ settings or failed with error message.
        """
        self._start_time = time.time()

        try:
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

        except AnalysisCancelled:
            return
        except Exception as e:
            logger.exception("Auto-EQ analysis failed")
            self.failed.emit(str(e) or UNEXPECTED_FAILURE)


class CaptureAnalysisWorker(QThread):
    """Runs the post-processing of a finished take off the GUI thread."""

    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, analyze: Callable[[], Any]) -> None:
        super().__init__()
        self._analyze = analyze
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Drop the result; the call in flight cannot be interrupted."""
        self._stop_event.set()

    def run(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            result = self._analyze()
        except Exception as exc:
            logger.exception("Capture post-processing failed")
            if not self._stop_event.is_set():
                self.failed.emit(str(exc) or UNEXPECTED_FAILURE)
            return
        if not self._stop_event.is_set():
            self.result_ready.emit(result)
