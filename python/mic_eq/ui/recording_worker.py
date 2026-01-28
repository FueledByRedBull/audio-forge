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
        self._was_running_before = False  # Track original state before we started recording
        self._we_started_processor = False     # Track if we started processing ourselves

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
            # CRITICAL: Ensure audio processing is running
            # The recording tap only executes when DSP loop is active
            self._was_running_before = self.processor.is_running()

            if DEBUG:
                print(f"[CALIBRATION] Processor was_running_before={self._was_running_before}")

            if not self._was_running_before:
                # Start audio processing automatically for recording
                # This activates the DSP loop where the recording tap lives
                try:
                    if DEBUG:
                        print("[CALIBRATION] Starting audio processor for recording...")
                    self.processor.start(None, None)
                    self._we_started_processor = True
                    if DEBUG:
                        print("[CALIBRATION] Audio processor started successfully")
                except Exception as e:
                    if DEBUG:
                        print(f"[CALIBRATION] ERROR: Failed to start processor: {e}")
                    self.failed.emit(f"Failed to start audio processing: {str(e)}")
                    return

            # Small delay to let DSP loop initialize
            time.sleep(0.1)

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

                    # Stop audio processing if we started it ourselves
                    # (NOT if it was already running when we began recording)
                    if self._we_started_processor:
                        if DEBUG:
                            print("[CALIBRATION] Stopping audio processor (we started it)")
                        self.processor.stop()
                    else:
                        if DEBUG:
                            print("[CALIBRATION] Leaving audio processor running (was already running)")
                else:
                    if DEBUG:
                        print("[CALIBRATION] ERROR: stop_raw_recording returned None")
                    self.failed.emit("Recording failed - no audio data")

        except Exception as e:
            if DEBUG:
                print(f"[CALIBRATION] EXCEPTION: {type(e).__name__}: {e}")
            self.failed.emit(f"Recording error: {str(e)}")

    def stop(self):
        """Stop recording early."""
        if DEBUG:
            print("[CALIBRATION] RecordingWorker.stop() called (user cancelled)")

        was_running = self._is_running
        self._is_running = False

        # If we started the processor ourselves, stop it when user cancels
        if was_running and self._we_started_processor:
            try:
                if DEBUG:
                    print("[CALIBRATION] Cancelling recording and stopping processor...")
                self.processor.stop_raw_recording()  # Get whatever audio we captured
                self.processor.stop()  # Stop audio processing
            except Exception as e:
                if DEBUG:
                    print(f"[CALIBRATION] Error during cancellation: {e}")
                pass  # Ignore cleanup errors during cancellation

        # Note: If user cancels, we don't emit finished() - just stop quietly
        if DEBUG:
            print("[CALIBRATION] RecordingWorker stopped")
