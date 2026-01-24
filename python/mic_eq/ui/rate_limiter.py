"""
Rate limiter for UI parameter updates.

Prevents flooding the audio thread with rapid parameter changes.
Uses throttling (not debouncing) to ensure updates still feel responsive.
"""

from PyQt6.QtCore import QTimer
from typing import Callable, Any
import time


class RateLimiter:
    """
    Throttle function calls to a maximum rate.

    Unlike debouncing (which waits for quiet period), throttling ensures
    the function is called at most once per interval, making UI feel
    responsive while limiting update frequency.

    Usage:
        limiter = RateLimiter(interval_ms=33)  # ~30Hz max

        def on_slider_changed(value):
            limiter.call(lambda: processor.set_gain(value))
    """

    def __init__(self, interval_ms: int = 33):
        """
        Initialize rate limiter.

        Args:
            interval_ms: Minimum time between calls (default 33ms = ~30Hz)
        """
        self.interval_ms = interval_ms
        self._last_call_time = 0.0
        self._pending_fn: Callable | None = None
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._execute_pending)

    def call(self, fn: Callable[[], Any]) -> None:
        """
        Schedule a function call, throttled to the configured rate.

        If called multiple times rapidly, only the most recent function
        is executed after the throttle interval.

        Args:
            fn: Zero-argument callable to execute
        """
        current_time = time.monotonic() * 1000  # Convert to ms
        elapsed = current_time - self._last_call_time

        if elapsed >= self.interval_ms:
            # Enough time has passed, execute immediately
            self._last_call_time = current_time
            fn()
        else:
            # Too soon, schedule for later (replace any pending call)
            self._pending_fn = fn
            remaining = self.interval_ms - elapsed
            if not self._timer.isActive():
                self._timer.start(int(remaining))

    def _execute_pending(self) -> None:
        """Execute the most recent pending call."""
        if self._pending_fn is not None:
            self._last_call_time = time.monotonic() * 1000
            fn = self._pending_fn
            self._pending_fn = None
            fn()

    def flush(self) -> None:
        """
        Execute any pending call immediately.

        Call this when user releases slider or on panel destruction
        to ensure final value is applied.
        """
        self._timer.stop()
        if self._pending_fn is not None:
            self._last_call_time = time.monotonic() * 1000
            fn = self._pending_fn
            self._pending_fn = None
            fn()
