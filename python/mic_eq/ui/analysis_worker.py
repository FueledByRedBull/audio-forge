"""
Non-blocking analysis worker for Auto-EQ calibration.

Runs audio analysis in background thread with step-by-step progress signals.
"""
import logging
import threading
import time
from PyQt6.QtCore import QThread, pyqtSignal

from mic_eq.analysis import analyze_auto_eq


logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """
    Worker thread for non-blocking audio analysis.

    Runs analysis pipeline in background, emits progress signals for
    each step. Provides smooth UX with step-by-step feedback.
    """

    # Signals
    step_progress = pyqtSignal(str, int)  # (step_name, percentage)
    finished = pyqtSignal(dict)            # Emits eq_settings dict
    failed = pyqtSignal(str)               # Emits error message (generic)

    def __init__(self, audio_data, sample_rate, target_preset='broadcast'):
        """
        Initialize analysis worker.

        Args:
            audio_data: Recorded audio samples (float32 NumPy array)
            sample_rate: Sample rate in Hz from the active processor
            target_preset: Target curve name ('broadcast', 'podcast', 'streaming', 'flat')
        """
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.target_preset = target_preset
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

            self.step_progress.emit("Analyzing voice spectrum...", 10)
            eq_settings, validation = analyze_auto_eq(
                self.audio_data,
                self.sample_rate,
                self.target_preset,
            )
            if self._should_stop():
                return

            if not validation.passed:
                self.failed.emit(validation.reason)
                return

            self.step_progress.emit("Validating correction...", 95)

            # Complete
            elapsed = time.time() - self._start_time
            self.step_progress.emit("Complete!", 100)

            # Log duration for debugging (optional)
            if elapsed > 0.5:
                logger.info("Analysis took %.2fs (target <500ms)", elapsed)

            self.finished.emit(eq_settings)

        except Exception as e:
            # Catch any unexpected errors
            self.failed.emit(str(e))

    def estimated_duration(self) -> float:
        """
        Estimate analysis duration based on audio length.

        Returns:
            Estimated seconds (typically 0.1-0.5s for 10s recording)
        """
        audio_duration = len(self.audio_data) / self.sample_rate
        # Analysis is typically 5-10% of recording duration
        return audio_duration * 0.1
