"""Build AudioForge's PipeWire microphone again after PipeWire restarts (Linux)."""

from __future__ import annotations

from collections.abc import Callable
import logging
import threading
import time


logger = logging.getLogger(__name__)

# The look is a cheap call; the rebuild waits on PipeWire for seconds, so a failed one is not repeated soon.
CHECK_SECONDS = 5.0
RETRY_SECONDS = 30.0


class VirtualMicrophoneWatch:
    """Rebuild the microphone, off the GUI thread, once PipeWire has dropped it."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._next_check = 0.0
        self._rebuilding = False
        self._stopped = False
        self._reported_gone = False

    def stop(self) -> None:
        """End the watch (the app is quitting); a rebuild already out is left to finish."""

        self._stopped = True

    def resume(self) -> None:
        """Watch again: the window that stopped it is shown after all (a relaunch that failed)."""

        self._stopped = False

    def poll(self) -> None:
        """Look at most every CHECK_SECONDS; never waits for PipeWire. Call from a timer."""

        now = self._clock()
        if self._stopped or self._rebuilding or now < self._next_check:
            return
        self._next_check = now + CHECK_SECONDS
        from .. import mic_eq_core

        if mic_eq_core.virtual_microphone_running():
            return
        # Once per outage: a machine without PipeWire would otherwise warn every retry.
        if not self._reported_gone:
            self._reported_gone = True
            logger.warning("AudioForge Mic is gone (PipeWire restarted?); building it again")
        self._rebuilding = True
        threading.Thread(target=self._rebuild, name="virtual-microphone", daemon=True).start()

    def _rebuild(self) -> None:
        from .. import mic_eq_core

        try:
            mic_eq_core.create_virtual_microphone()
        except Exception as error:  # noqa: BLE001 - PipeWire may be down for a while
            logger.info("AudioForge Mic could not be built yet: %s", error)
            self._next_check = self._clock() + RETRY_SECONDS
        else:
            self._reported_gone = False
            logger.info("AudioForge Mic is back")
        finally:
            self._rebuilding = False
