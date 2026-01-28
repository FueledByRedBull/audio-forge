"""
Non-blocking audio recording worker for Auto-EQ calibration

Controls Rust AudioProcessor's raw recording tap via PyO3 bindings.
All recording happens in the Rust DSP thread - zero Python GC interference.
"""
import time
from PyQt6.QtCore import QThread, pyqtSignal


class RecordingWorker(QThread):
    """
    Worker thread for non-blocking recording using Rust tap.

    Controls the Rust raw recording tap (from Phase 18-01) via PyO3 bindings.
    Polls for progress, emits UI updates, retrieves final audio.

    CRITICAL: Recording happens entirely in Rust DSP thread.
    This is the ACTUAL audio being processed (before RNNoise).
    """

    # Signals
    progress = pyqtSignal(int)            # 0-100 progress percentage
    time_remaining = pyqtSignal(float)    # Seconds remaining
    level_update = pyqtSignal(float)      # Current RMS level in dB
    finished = pyqtSignal(object)         # Emits audio data (numpy array)
    failed = pyqtSignal(str)              # Emits error message

    def __init__(self, processor, duration: float = 10.0):
        """
        Initialize recording worker.

        Args:
            processor: AudioProcessor instance (from Rust core)
            duration: Recording duration in seconds (default: 10)
        """
        super().__init__()
        self.processor = processor
        self.duration = duration
        self._is_running = True
        self._start_time = None

    def run(self):
        """
        Run recording in background thread.

        Called by QThread.start(). Starts Rust tap, polls progress,
        emits updates until recording is complete or stopped.
        """
        self._start_time = time.time()

        try:
            # Start Rust tap (non-blocking, records in DSP thread)
            self.processor.start_raw_recording(self.duration)

            # Poll progress until done
            while self._is_running:
                # Check if recording is complete
                if self.processor.is_recording_complete():
                    break

                # Get progress from Rust (0.0 to 1.0)
                progress_float = self.processor.recording_progress()
                progress_pct = int(progress_float * 100)

                # Emit progress signals
                self.progress.emit(progress_pct)

                # Calculate remaining time
                remaining = max(0.0, self.duration * (1.0 - progress_float))
                self.time_remaining.emit(remaining)

                # Calculate level from recording buffer
                # Note: We can't directly access the recording buffer from Python,
                # so we estimate based on progress. For actual level, we'd need
                # a separate PyO3 binding to get intermediate samples.
                # For now, we'll use a placeholder or add get_recording_level() to Rust.
                # Using estimated level based on typical voice patterns:
                self.level_update.emit(-20.0)  # Placeholder - could be enhanced

                # Check if we should stop (user cancelled)
                if progress_pct >= 100:
                    break

                # Wait a bit before next poll (100ms = 10fps)
                time.sleep(0.1)

            # Recording complete - retrieve audio from Rust
            if self._is_running:
                audio = self.processor.stop_raw_recording()

                if audio is not None:
                    # Final progress signals
                    self.progress.emit(100)
                    self.time_remaining.emit(0.0)
                    self.finished.emit(audio)
                else:
                    self.failed.emit("Recording failed - no audio data")

        except Exception as e:
            self.failed.emit(f"Recording error: {str(e)}")

    def stop(self):
        """Stop recording early."""
        self._is_running = False
        # Note: The Rust tap will continue until we call stop_raw_recording(),
        # which happens when we break out of the loop above
