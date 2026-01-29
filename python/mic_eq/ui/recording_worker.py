"""
Non-blocking audio recording worker for Auto-EQ calibration

Controls Rust AudioProcessor's raw recording tap via PyO3 bindings.
All recording happens in the Rust DSP thread - zero Python GC interference.

DEBUG: Added terminal logging for calibration workflow verification
"""
import time
from PyQt6.QtCore import QThread, pyqtSignal

# Enable debug logging
DEBUG = True


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

    def __init__(self, processor, duration: float = 10.0, processor_was_started_by_us: bool = False):
        """
        Initialize recording worker.

        Args:
            processor: AudioProcessor instance (from Rust core)
            duration: Recording duration in seconds (default: 10)
            processor_was_started_by_us: True if dialog started processor for us
        """
        super().__init__()
        self.processor = processor
        self.duration = duration
        self._is_running = True
        self._start_time = None
        self._processor_was_started_by_us = processor_was_started_by_us  # Use passed flag

    def run(self):
        """
        Run recording in background thread.

        Called by QThread.start(). Starts Rust tap, polls progress,
        emits updates until recording is complete or stopped.
        """
        self._start_time = time.time()

        if DEBUG:
            print(f"[CALIBRATION] RecordingWorker started, duration={self.duration}s")

        try:
            # CRITICAL: Processor should already be running (started by main thread)
            # The worker only handles the recording tap, not processor lifecycle
            if not self.processor.is_running():
                if DEBUG:
                    print("[CALIBRATION] ERROR: Processor not running (main thread should have started it)")
                self.failed.emit("Audio processor not running. Please start processing first.")
                return

            if DEBUG:
                print(f"[CALIBRATION] Processor verified running (started_by_us={self._processor_was_started_by_us})")

            # Start Rust tap (non-blocking, records in DSP thread)
            if DEBUG:
                print(f"[CALIBRATION] Starting raw recording tap for {self.duration}s...")
            self.processor.start_raw_recording(self.duration)

            # Poll progress until done
            poll_count = 0
            while self._is_running:
                # Check if recording is complete
                if self.processor.is_recording_complete():
                    if DEBUG:
                        print("[CALIBRATION] Recording complete (is_recording_complete=True)")
                    break

                # Get progress from Rust (0.0 to 1.0)
                progress_float = self.processor.recording_progress()
                progress_pct = int(progress_float * 100)

                # Emit progress signals
                self.progress.emit(progress_pct)

                # Calculate remaining time
                remaining = max(0.0, self.duration * (1.0 - progress_float))
                self.time_remaining.emit(remaining)

                # Get actual recording level from Rust
                level_db = self.processor.recording_level_db()
                self.level_update.emit(level_db)

                # Debug logging every 10 polls (1 second)
                poll_count += 1
                if DEBUG and poll_count % 10 == 0:
                    print(f"[CALIBRATION] Progress: {progress_pct}%, Level: {level_db:.1f} dB, Remaining: {remaining:.1f}s")

                # Check if we should stop (user cancelled)
                if progress_pct >= 100:
                    if DEBUG:
                        print("[CALIBRATION] Recording reached 100%")
                    break

                # Wait a bit before next poll (100ms = 10fps)
                time.sleep(0.1)

            # Recording complete - retrieve audio from Rust
            if self._is_running:
                if DEBUG:
                    print("[CALIBRATION] Retrieving recorded audio...")
                audio = self.processor.stop_raw_recording()

                if audio is not None:
                    import numpy as np
                    audio_array = np.array(audio)
                    if DEBUG:
                        print(f"[CALIBRATION] Retrieved {len(audio_array)} samples ({len(audio_array)/48000:.2f}s)")
                        print(f"[CALIBRATION] Audio range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
                        print(f"[CALIBRATION] Audio RMS: {(np.mean(audio_array**2)**0.5):.6f}")

                    # Keep output muted after calibration to prevent user from hearing themselves
                    if DEBUG:
                        print("[CALIBRATION] Setting output mute=True (keeping muted after calibration)")
                    self.processor.set_output_mute(True)

                    # Final progress signals
                    self.progress.emit(100)
                    self.time_remaining.emit(0.0)
                    self.finished.emit(audio)
                else:
                    if DEBUG:
                        print("[CALIBRATION] ERROR: stop_raw_recording returned None")
                    self.failed.emit("Recording failed - no audio data")

        except Exception as e:
            if DEBUG:
                print(f"[CALIBRATION] EXCEPTION: {type(e).__name__}: {e}")
            self.failed.emit(f"Recording error: {str(e)}")
        finally:
            # CRITICAL: No cleanup needed here - dialog handles processor lifecycle
            # We only managed the recording tap, not the audio engine
            if DEBUG:
                print("[CALIBRATION] Worker finished (processor lifecycle managed by dialog)")

    def stop(self):
        """Stop recording early."""
        if DEBUG:
            print("[CALIBRATION] RecordingWorker.stop() called (user cancelled)")

        self._is_running = False

        # Note: We don't stop processor on cancel - dialog handles cleanup
        # This prevents race conditions between worker and main thread
        if DEBUG:
            print("[CALIBRATION] Recording cancelled, processor cleanup deferred to dialog")
