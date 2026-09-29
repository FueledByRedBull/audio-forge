"""Compact health decision helpers for runtime metering."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


STREAM_EVENT_COUNTERS = frozenset({
    "input_callback_error_count", "output_callback_error_count",
    "input_stream_error_count", "output_stream_error_count", "stream_restart_count",
    "input_dropped_samples", "input_backlog_dropped_samples", "jitter_dropped_samples",
    "output_short_write_dropped_samples", "output_underrun_total",
    "rt_buffer_overflow_count", "input_backlog_recovery_count",
    "output_recovery_count", "output_recovery_event_count", "gate_chatter_event_count",
    "lock_contention_count", "suppressor_non_finite_count",
})


@dataclass(slots=True)
class RecentStreamHealth:
    """Keep a five-second warning window without changing cumulative evidence."""

    _counts: dict[str, int] = field(default_factory=dict)
    _last_events: dict[str, float] = field(default_factory=dict)
    recent_events: frozenset[str] = frozenset()

    def observe(self, diagnostics: Mapping[str, Any], *, now: float | None = None) -> frozenset[str]:
        current = time.monotonic() if now is None else now
        counts: dict[str, int] = {}
        for name in STREAM_EVENT_COUNTERS:
            value = diagnostics.get(name, self._counts.get(name, 0))
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or value < 0 or (isinstance(value, float)
                                    and (not math.isfinite(value) or not value.is_integer()))):
                raise ValueError(f"Invalid stream event counter: {name}")
            counts[name] = int(value)
        for name, count in counts.items():
            if count > self._counts.get(name, 0):
                self._last_events[name] = current
        self._counts = counts
        self._last_events = {
            name: timestamp for name, timestamp in self._last_events.items()
            if current - timestamp < 5.0
        }
        self.recent_events = frozenset(self._last_events)
        return self.recent_events


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def input_health_state(
    *,
    rms_db: float | None,
    clip_delta: bool = False,
    phase_rescue_active: bool = False,
    cleanup_rumble_detected: bool = False,
    cleanup_hum_detected: bool = False,
    cleanup_mode: str = "off",
    crest_factor_db: float | None = None,
) -> tuple[str, str]:
    """Return compact input health text/state."""

    rms_db = _float_or_none(rms_db)
    crest_factor_db = _float_or_none(crest_factor_db)
    if clip_delta:
        return "Input: CLIPPING", "bad"
    if phase_rescue_active:
        return "Input: PHASE", "warn"
    if cleanup_rumble_detected:
        return "Input: CLEANUP RUMBLE", "warn" if cleanup_mode == "strong" else "info"
    if cleanup_hum_detected:
        return "Input: CLEANUP HUM", "info"
    if rms_db is None:
        return "Input: --", "idle"
    if rms_db < -65.0:
        return f"Input: LOW ({rms_db:.0f}dB)", "warn"
    if rms_db > -3.0:
        return f"Input: HOT ({rms_db:.0f}dB)", "warn"
    if crest_factor_db is not None and rms_db > -45.0 and crest_factor_db < 3.0:
        return f"Input: DENSE (CF:{crest_factor_db:.1f}dB)", "warn"
    suffix = f" CF:{crest_factor_db:.0f}" if crest_factor_db is not None else ""
    return f"Input: OK ({rms_db:.0f}dB{suffix})", "ok"


def output_health_state(
    *,
    rms_db: float | None,
    clip_delta: bool = False,
    true_peak_delta: bool = False,
    output_clip_count: int = 0,
    true_peak_count: int = 0,
    true_peak_db: Any = None,
    true_peak_headroom_db: Any = None,
    short_term_lufs: Any = None,
    limiter_history_db: float = 0.0,
    true_peak_limiter_history_db: float = 0.0,
) -> tuple[str, str]:
    """Return compact output health text/state."""

    rms_db = _float_or_none(rms_db)
    true_peak_headroom = _float_or_none(true_peak_headroom_db)
    limiter_history = _float_or_none(limiter_history_db)
    true_peak_limiter_history = _float_or_none(true_peak_limiter_history_db)
    if clip_delta:
        return f"Output: CLIP (OCL:{output_clip_count})", "bad"
    if limiter_history is None or true_peak_limiter_history is None:
        return "Output: --", "idle"
    if limiter_history >= 6.0 or true_peak_limiter_history >= 3.0:
        return (
            f"Output: LIMITING HARD (L:{limiter_history_db:.1f} TP:{true_peak_limiter_history_db:.1f})",
            "warn",
        )
    if true_peak_delta:
        return f"Output: TRUE PEAK (OTP:{true_peak_count})", "warn"
    if true_peak_headroom is not None and true_peak_headroom < 0.75:
        return f"Output: LOW TP HEADROOM ({true_peak_headroom:.1f}dB)", "warn"
    if rms_db is None:
        return "Output: --", "idle"
    if rms_db > -1.0:
        return f"Output: HOT ({rms_db:.0f}dB)", "warn"

    true_peak = _float_or_none(true_peak_db)
    loudness = _float_or_none(short_term_lufs)
    tp_suffix = f" TP:{true_peak:.1f}" if true_peak is not None else ""
    lufs_suffix = f" LU:{loudness:.0f}" if loudness is not None and loudness > -119.0 else ""
    return f"Output: OK ({rms_db:.0f}dB{tp_suffix}{lufs_suffix})", "ok"
