"""Compact health decision helpers for runtime metering."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


# How far back the meter's peaks and clips count towards the Health lines.
HEALTH_PEAK_WINDOW_SECONDS = 8.0
# A device whose last callback is older than this is slow (warn) or stalled (bad).
DEVICE_ACTIVITY_WARN_MS = 250
DEVICE_ACTIVITY_BAD_MS = 1000
# Warn above this input peak; the advice aims for peaks around the target.
INPUT_PEAK_WARN_DB = -3.0
INPUT_PEAK_TARGET_DB = -6.0
# With the limiter off, the advice leaves this much room below full scale.
OUTPUT_PEAK_TARGET_DB = -1.0
# Where the settings the advice names are in the window; they are all behind an Advanced section.
_WHERE_MAKEUP = " Makeup gain is under Advanced on the Compressor card."

STREAM_EVENT_COUNTERS = frozenset({
    "input_callback_error_count", "output_callback_error_count",
    "input_stream_error_count", "output_stream_error_count", "stream_restart_count",
    "input_dropped_samples", "input_backlog_dropped_samples",
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


def plural(count: float, singular: str, plural_form: str | None = None) -> str:
    """Return "1 restart" or "2 restarts"."""

    return f"{count:.0f} {singular if round(count) == 1 else plural_form or singular + 's'}"


def _num(value: float, digits: int = 1) -> str:
    """A dB value to ``digits`` decimals; one that rounds to zero has no minus sign."""

    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def input_health_state(
    *,
    rms_db: float | None,
    peak_db: float | None = None,
    full_scale_samples: int = 0,
    phase_rescue_active: bool = False,
    cleanup_rumble_detected: bool = False,
    cleanup_hum_detected: bool = False,
    cleanup_mode: str = "off",
    crest_factor_db: float | None = None,
) -> tuple[str, str]:
    """Return compact input health text/state.

    ``peak_db`` is the loudest input peak of the last few seconds and
    ``full_scale_samples`` is how many times the input reached full scale in
    that time (0 when it did not); both come from the meter, not from the
    250 ms diagnostics tick.
    """

    rms_db = _float_or_none(rms_db)
    peak_db = _float_or_none(peak_db)
    crest_factor_db = _float_or_none(crest_factor_db)
    if full_scale_samples:
        peak = f"peak {max(peak_db or 0.0, 0.0):.1f} dB"
        return f"Input: CLIPPED ({peak}, full scale {plural(full_scale_samples, 'time')})", "bad"
    # Judge the number that is printed, so the text and the state cannot disagree.
    if peak_db is not None and round(peak_db, 1) > INPUT_PEAK_WARN_DB:
        return f"Input: PEAKS TOO HIGH ({_num(peak_db)} dB)", "warn"
    if phase_rescue_active:
        return "Input: PHASE", "warn"
    if cleanup_rumble_detected:
        return "Input: LOW RUMBLE", "warn" if cleanup_mode == "strong" else "info"
    if cleanup_hum_detected:
        return "Input: HUM DETECTED", "info"
    if rms_db is None:
        return "Input: --", "idle"
    if round(rms_db) < -65:
        return f"Input: LOW ({_num(rms_db, 0)} dB)", "warn"
    if crest_factor_db is not None and rms_db > -45.0 and round(crest_factor_db, 1) < 3.0:
        return f"Input: ALREADY COMPRESSED ({_num(crest_factor_db)} dB)", "warn"
    peaks = f", peaks {_num(peak_db)} dB" if peak_db is not None else ""
    return f"Input: OK ({_num(rms_db, 0)} dB average{peaks})", "ok"


def output_health_state(
    *,
    rms_db: float | None,
    at_full_scale: bool = False,
    limiter_enabled: bool = True,
    peak_db: float | None = None,
    true_peak_delta: bool = False,
    true_peak_db: Any = None,
    true_peak_headroom_db: Any = None,
    short_term_lufs: Any = None,
    limiter_history_db: float = 0.0,
    true_peak_limiter_history_db: float = 0.0,
    muted: bool = False,
) -> tuple[str, str]:
    """Return compact output health text/state.

    ``at_full_scale`` says the output reached full scale in the last few
    seconds and ``peak_db`` is the level the output stage was asked for. With
    the limiter on, an output sitting at its ceiling is normal. The levels are
    measured before the mute, so a muted output only reports them and is not judged.
    """

    rms_db = _float_or_none(rms_db)
    peak_db = _float_or_none(peak_db)
    true_peak_headroom = _float_or_none(true_peak_headroom_db)
    limiter_history = _float_or_none(limiter_history_db)
    true_peak_limiter_history = _float_or_none(true_peak_limiter_history_db)
    if muted:
        if rms_db is None:
            return "Output: Muted", "info"
        true_peak = _float_or_none(true_peak_db)
        peaks = f", peaks {_num(true_peak)} dB" if true_peak is not None else ""
        return f"Output: Muted (before mute: {_num(rms_db, 0)} dB average{peaks})", "info"
    if at_full_scale and not limiter_enabled:
        number = f" (peak {_num(max(peak_db, 0.0))} dB)" if peak_db is not None else ""
        return f"Output: AT FULL SCALE{number}", "bad"
    if limiter_history is None or true_peak_limiter_history is None:
        return "Output: --", "idle"
    if limiter_history >= 6.0 or true_peak_limiter_history >= 3.0:
        worst = max(limiter_history, true_peak_limiter_history)
        return f"Output: LIMITER WORKING HARD ({_num(worst)} dB)", "warn"
    if true_peak_delta:
        true_peak = _float_or_none(true_peak_db)
        number = f" (peak {_num(true_peak)} dB)" if true_peak is not None else ""
        return f"Output: PEAK ABOVE CEILING{number}", "warn"
    if true_peak_headroom is not None and true_peak_headroom < 0.75:
        return f"Output: LITTLE HEADROOM ({_num(true_peak_headroom)} dB)", "warn"
    if rms_db is None:
        return "Output: --", "idle"
    if round(rms_db) > -1:
        return f"Output: VERY LOUD ({_num(rms_db, 0)} dB)", "warn"

    true_peak = _float_or_none(true_peak_db)
    loudness = _float_or_none(short_term_lufs)
    peak_part = f", peaks {_num(true_peak)} dB" if true_peak is not None else ""
    lufs_part = f", {_num(loudness, 0)} LUFS" if loudness is not None and loudness > -119.0 else ""
    return f"Output: OK ({_num(rms_db, 0)} dB average{peak_part}{lufs_part})", "ok"


def _numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _input_peak_advice(numbers: list[float], opening: str) -> str:
    peak = numbers[0]
    drop = math.ceil(round(peak - INPUT_PEAK_TARGET_DB, 6))
    return (
        f"{opening.format(peak=f'{peak:.1f}')} so loud sounds can be cut off and sound "
        f"distorted. Lower the microphone level by about {drop} dB in Windows or on the interface."
    )


def _output_full_scale_advice(numbers: list[float]) -> str:
    action = "Turn the limiter on or lower the makeup gain"
    if not numbers:
        return (
            "The output reaches full scale and the limiter is off, so loud sounds are cut off "
            f"and sound distorted. {action}.{_WHERE_MAKEUP}"
        )
    drop = math.ceil(round(numbers[0] - OUTPUT_PEAK_TARGET_DB, 6))
    return (
        f"The output reaches full scale (peak {numbers[0]:.1f} dB) and the limiter is off, so loud sounds "
        f"are cut off and sound distorted. {action}; the makeup gain would need to come down by about {drop} dB.{_WHERE_MAKEUP}"
    )


def _headroom_advice(numbers: list[float]) -> str:
    headroom = numbers[0]
    gap = max(1, math.ceil(round(0.75 - headroom, 6)))
    side = f"only {headroom:.1f} dB below" if headroom >= 0 else f"{-headroom:.1f} dB above"
    return (
        f"The latest output peak is {side} the ceiling, so peaks can distort on some players. "
        f"Lower the makeup gain by about {gap} dB.{_WHERE_MAKEUP}"
    )


def _drops_advice(numbers: list[float]) -> str:
    if not numbers or numbers[0] < 0.5:  # Nothing was dropped, so there is no drop to explain.
        return ""
    return (
        f"Some audio was dropped ({plural(numbers[0], 'sample')} so far), which sounds like clicks or gaps. "
        "If it keeps happening, restart processing."
    )


def _recovery_advice(numbers: list[float]) -> str:
    restarts, output_recoveries = numbers[0], numbers[1]
    if restarts:
        what = f"The audio stream restarted ({plural(restarts, 'time')} so far)"
    elif output_recoveries:
        what = f"The audio output was restarted ({plural(output_recoveries, 'time')} so far)"
    else:
        what = "The audio stream is recovering"
    return f"{what}, which causes a short gap in sound. If this repeats, check the device connection."


def _device_activity_advice(numbers: list[float]) -> str:
    slowest = max(numbers, default=0)
    if slowest > DEVICE_ACTIVITY_WARN_MS:
        return (
            f"An audio device has not sent or asked for sound for {slowest:.0f} ms, so audio may be frozen. "
            "Check that the device is connected, then stop and start processing."
        )
    return (
        "An audio device has not started sending or asking for sound, so no audio is flowing. "
        "Check that the device is connected and not in use by another application, then stop and start processing."
    )


def _plain(text: str) -> Callable[[list[float]], str]:
    return lambda _numbers: text


# What to do about each warning a health chip can show, keyed by the token in
# the chip text. Each entry turns the numbers in the chip into one sentence
# that says what happened, why it matters and what to do. Longer tokens come
# first so "Input: PEAKS TOO HIGH" wins over a shorter token.
_ADVICE: tuple[tuple[str, Callable[[list[float]], str]], ...] = (
    ("Input: CLIPPED", lambda n: _input_peak_advice(
        n,
        "The microphone clipped: it reached full scale "
        f"{plural(n[1], 'time')} in the last few seconds (peak {{peak}} dB),",
    )),
    ("Input: PEAKS TOO HIGH", lambda n: _input_peak_advice(
        n, "Input peaks at {peak} dB, close to the top of the microphone's range,")),
    ("Input: PHASE", _plain(
        "The left and right microphone channels partly cancel each other, which makes your voice thin "
        "or quiet, and AudioForge is compensating. If it still sounds thin, pick Left, Right or "
        "Phase-safe mono under Settings.")),
    ("Input: LOW RUMBLE", _plain(
        "Low rumble is reaching the microphone, which muddies your voice and wastes headroom. "
        "Check for desk bumps or a fan, or keep input cleanup on.")),
    ("Input: HUM DETECTED", _plain(
        "A steady hum, usually electrical interference, is reaching the microphone. "
        "Move the microphone cable away from power cables, or keep input cleanup on.")),
    ("Input: LOW", lambda n: (
        f"The microphone is very quiet (about {n[0]:.0f} dB on average), so noise will be loud next to "
        "your voice. Raise the microphone level in Windows or move closer.")),
    ("Input: ALREADY COMPRESSED", lambda n: (
        "The microphone signal is already heavily compressed (its loud and quiet parts differ by only "
        f"{n[0]:.1f} dB), so more compression adds distortion. Turn off processing in the microphone "
        "driver or headset software.")),
    ("Output: AT FULL SCALE", _output_full_scale_advice),
    ("Output: LIMITER WORKING HARD", lambda n: (
        f"The limiter is turning the output down by up to {n[0]:.1f} dB, which can make loud words sound "
        f"squashed. Lower the compressor makeup gain or the microphone level by about {math.ceil(n[0])} dB.{_WHERE_MAKEUP}")),
    ("Output: PEAK ABOVE CEILING", _plain(
        "The output went above the limiter ceiling for a moment, which can distort on some players. "
        "Turn on careful output mode or lower the makeup gain. Careful output mode is under Advanced "
        f"on the Limiter card.{_WHERE_MAKEUP}")),
    ("Output: LITTLE HEADROOM", _headroom_advice),
    ("Output: VERY LOUD", lambda n: (
        f"The average output level is {n[0]:.0f} dB, which is extremely loud for speech and can overload "
        f"whatever receives it. Lower the makeup gain.{_WHERE_MAKEUP}")),
    ("Gate: CHATTERING", lambda n: (
        f"The gate opened and closed rapidly {plural(n[0], 'time')}, which makes your voice stutter. "
        "Raise the hold time or lower the threshold. Hold time is under Advanced on the Noise Gate card.")),
    ("Gate: EASED", lambda n: (
        f"The gate eased its threshold by itself to stop chattering ({plural(n[0], 'time')} so far). "
        "Consider lowering the threshold yourself so it does not need to.")),
    ("Backend:", _plain(
        "Noise suppression is not running as selected, so your voice may be passing through unprocessed. "
        "Check the Backend setting on the Noise Suppression card.")),
    ("Device activity:", _device_activity_advice),
    ("Dropouts:", lambda n: (
        f"Audio dropped out {plural(n[0], 'time')} because the output could not keep up, which sounds like gaps "
        "or crackles. Close heavy applications or choose a different output device.")),
    ("Buffer:", lambda n: (
        f"Audio is backing up ({plural(n[0], 'sample')} waiting), which adds delay and can end in dropouts. "
        "Close heavy applications or pick a lighter noise-suppression model.")),
    ("Drops:", _drops_advice),
    ("Recovery:", _recovery_advice),
)


def advice_for(chip_text: str) -> str:
    """Return plain-language advice for a warning chip, or an empty string."""

    for token, advice in _ADVICE:
        if chip_text.startswith(token):
            try:
                return advice(_numbers(chip_text[len(token):]))
            except IndexError:  # A chip that lost its number has no sentence to give.
                return ""
    return ""
