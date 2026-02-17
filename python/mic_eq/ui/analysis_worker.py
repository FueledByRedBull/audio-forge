"""
Non-blocking analysis worker for Auto-EQ calibration.

Runs audio analysis in background thread with step-by-step progress signals.
Extends RecordingWorker pattern from Phase 18 for analysis workflow.
"""
import threading
import time
from PyQt6.QtCore import QThread, pyqtSignal

from mic_eq.analysis import (
    compute_voice_spectrum,
    smooth_spectrum_octave,
    get_target_curve,
    calculate_eq_bands,
    validate_analysis
)


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
            sample_rate: Sample rate in Hz (should be 48000)
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

            # Step 1: FFT analysis
            self.step_progress.emit("Computing FFT...", 10)
            freqs, spectrum_db = compute_voice_spectrum(
                self.audio_data,
                self.sample_rate
            )
            if self._should_stop():
                return

            # Step 2: Smoothing
            self.step_progress.emit("Smoothing spectrum...", 40)
            spectrum_smoothed = smooth_spectrum_octave(freqs, spectrum_db, fraction=6)
            if self._should_stop():
                return

            # Step 3: Get target curve
            self.step_progress.emit("Loading target curve...", 50)
            target_db = get_target_curve(freqs, self.target_preset)
            if self._should_stop():
                return

            # Step 4: Calculate EQ bands
            self.step_progress.emit("Finding frequency problems...", 70)
            eq_settings = calculate_eq_bands(freqs, spectrum_smoothed, target_db)
            if self._should_stop():
                return

            # Step 5: Validate results
            self.step_progress.emit("Validating results...", 95)
            validation = validate_analysis(eq_settings, spectrum_smoothed, freqs)
            if self._should_stop():
                return

            if not validation.passed:
                self.failed.emit(validation.reason)
                return

            # Complete
            elapsed = time.time() - self._start_time
            self.step_progress.emit("Complete!", 100)

            # Log duration for debugging (optional)
            if elapsed > 0.5:
                print(f"Analysis took {elapsed:.2f}s (should be <500ms)")

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
