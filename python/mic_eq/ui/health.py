"""Health decisions: which chip says what, from a runtime snapshot and the live readings.

Nothing here touches Qt or the engine. :func:`runtime_health` turns one
diagnostics tick into chips, status messages and what the next tick needs to
remember; the window only renders the result.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..runtime_snapshot import RuntimeSnapshot
from .platform_text import SOUND_SETTINGS
from .shell_specs import (
    DROPPED_DIAGNOSTICS_TOOLTIP,
    LATENCY_LOWER_BOUND_NOTE,
    LATENCY_LOWER_BOUND_NOTE_NO_BUFFER,
    TEXT_SPECS,
)

# How far back the meter's peaks and clips count towards the Health lines.
HEALTH_PEAK_WINDOW_SECONDS = 8.0
# A device whose last callback is older than this is slow (warn) or stalled (bad).
DEVICE_ACTIVITY_WARN_MS = 250
DEVICE_ACTIVITY_BAD_MS = 1000
# The engine reports this age until a device has made its first callback.
NO_CALLBACK_YET = 2**64 - 1
# A device that has made no callback this long after it was first seen waiting is a
# warning. Any engine restart resets the callback ages, so the wait is timed from the
# moment the not-started state appears, not from when processing began.
DEVICE_STARTUP_GRACE_SECONDS = 5.0
INPUT_PHASE_WARNING_CORRELATION = -0.75
PHASE_FIX_LABELS = {
    "delay_left": "delaying the left channel",
    "delay_right": "delaying the right channel",
    "polarity_flip": "flipping one channel's polarity",
}
NOISE_SELECTION_DROPPED_WARNING = (
    "The noise suppression model you chose could not be applied, and the previous one is still "
    "running. Choose it again. If it keeps failing, check the model files or use Help > Export Diagnostics."
)
_BACKEND_STOPPED_WARNING = (
    "Noise suppression stopped working, so your voice is passing through unprocessed. "
    "Choose another model on the Noise Suppression card."
)
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

    def observe(self, snapshot: RuntimeSnapshot, *, now: float | None = None) -> frozenset[str]:
        current = time.monotonic() if now is None else now
        counts: dict[str, int] = {}
        for name in STREAM_EVENT_COUNTERS:
            # A counter the engine did not send keeps its last count.
            value = snapshot.reported(name) if name in snapshot.present else self._counts.get(name, 0)
            if (not isinstance(value, (int, float))
                    or value < 0 or (isinstance(value, float)
                                    and (not math.isfinite(value) or not value.is_integer()))):
                raise ValueError(f"Invalid stream event counter: {name}")
            counts[name] = int(value)
        for name, count in counts.items():
            previous = self._counts.get(name, 0)
            if count < previous:
                # The counter restarted with a new stream: its recent event
                # belongs to the old one and would read "(0 times)". Whatever
                # the new stream has counted already is new.
                self._last_events.pop(name, None)
                previous = 0
            if count > previous:
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
        f"distorted. Lower the microphone level by about {drop} dB in {SOUND_SETTINGS} or on the interface."
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
        f"your voice. Raise the microphone level in {SOUND_SETTINGS} or move closer.")),
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


@dataclass(frozen=True, slots=True)
class Chip:
    """One Health row: its text, its state (ok, info, warn, bad or idle) and its tooltip, if it sets one."""

    text: str
    state: str
    tip: str | None = None


@dataclass(frozen=True, slots=True)
class HealthReadings:
    """The live readings a diagnostics tick brings besides the runtime snapshot.

    The peak and clip readings come from the meters' held peaks, so a clip
    stays visible for several seconds.
    """

    latency_ms: float
    dsp_time_ms: float
    input_buf: int
    rnnoise_buf: int
    input_rms_db: float | None = None
    output_rms_db: float | None = None
    input_callback_age_ms: int | None = None
    output_callback_age_ms: int | None = None
    input_peak_db: float | None = None
    input_full_scale_samples: int = 0
    output_peak_db: float | None = None
    output_at_full_scale: bool = False
    limiter_enabled: bool = True
    output_muted: bool = False
    recovery_pending: bool = False
    processing: bool = False


@dataclass(frozen=True, slots=True)
class HealthMemory:
    """What one tick hands the next: the counts it saw, when a device began waiting, what it announced."""

    output_underrun_total: int = 0
    input_phase_warning_count: int = 0
    output_true_peak_event_count: int = 0
    device_wait_since: float | None = None
    noise_drops_seen: int = 0
    backend_warning: str | None = None


@dataclass(frozen=True, slots=True)
class HealthReport:
    """The chips one tick sets, keyed by their text-state name, and the status messages to show, in order."""

    chips: dict[str, Chip]
    messages: tuple[tuple[str, int], ...]
    memory: HealthMemory


# The technical counters the Drops tooltip lists, as (key, label).
_INPUT_TOKENS = (
    ("input_cleanup_mode", "CLN"),
    ("input_cleanup_hum_detected", "HUM"),
    ("input_cleanup_rumble_detected", "RMB"),
    ("input_cleanup_high_pass_hz", "HPF"),
    ("input_phase_rescue_strategy", "PRS"),
    ("input_phase_estimated_delay_samples", "PDL"),
    ("input_phase_polarity_flipped", "PFL"),
)
_ENGINE_TOKENS = (
    ("input_backlog_recovery_count", "IBR"),
    ("input_backlog_dropped_samples", "IBD"),
    ("output_short_write_dropped_samples", "OSW"),
    ("rt_buffer_overflow_count", "RTO"),
    ("input_callback_error_count", "ICE"),
    ("output_callback_error_count", "OCE"),
    ("clip_event_count", "CL"),
    ("clip_peak_db", "PK"),
    ("input_crest_factor_db", "ICF"),
    ("output_clip_event_count", "OCL"),
    ("output_clip_peak_db", "OPK"),
    ("output_crest_factor_db", "OCF"),
    ("output_short_term_lufs", "LU"),
    ("output_true_peak_event_count", "OTP"),
    ("output_true_peak_db", "TPK"),
    ("output_true_peak_headroom_db", "TPH"),
    ("limiter_gain_reduction_history_db", "LGR"),
    ("output_true_peak_gain_reduction_history_db", "TPGR"),
    ("limiter_effective_ceiling_db", "LIM"),
    ("gate_chatter_event_count", "GCH"),
    ("gate_auto_relax_active", "GAR"),
    ("deesser_detector_confidence", "DSC"),
    ("noise_selection_dropped_count", "NSD"),
)
_RECOVERY_EVENTS = frozenset({
    "stream_restart_count", "output_recovery_count", "output_recovery_event_count",
    "input_backlog_recovery_count",
})


def diagnostic_tokens(snapshot: RuntimeSnapshot, keys: Sequence[tuple[str, str]]) -> list[str]:
    """``LABEL:value`` for each key the engine sent a value for: Y/N, one decimal, or as sent."""
    tokens = []
    for key, label in keys:
        value = snapshot.reported(key)
        if value is None:
            continue
        if isinstance(value, bool):
            tokens.append(f"{label}:{'Y' if value else 'N'}")
        elif isinstance(value, float):
            tokens.append(f"{label}:{value:.1f}")
        else:
            tokens.append(f"{label}:{value}")
    return tokens


def _ago_text(age_ms: int | None, *, output: bool = False) -> str:
    if age_ms is None:
        return "--"
    if age_ms == NO_CALLBACK_YET:
        return "has not started playing sound yet" if output else "has not started sending sound yet"
    return f"{age_ms} ms ago"


def _buffer_chip(pipeline_buf: int) -> Chip:
    if pipeline_buf < 960:
        status, state = "OK", "ok"
    elif pipeline_buf < 1920:
        status, state = "FILLING UP", "warn"
    else:
        status, state = "BACKED UP", "bad"
    return Chip(f"Buffer: {status} ({plural(pipeline_buf, 'sample')})", state)


def _device_activity(
    readings: HealthReadings, wait_since: float | None, now: float
) -> tuple[Chip, float | None]:
    """The Device activity chip, and since when a device has been waiting for its first callback."""
    ages = (readings.input_callback_age_ms, readings.output_callback_age_ms)
    callback_ages = [age for age in ages if age is not None and 0 <= age < NO_CALLBACK_YET]
    not_started = NO_CALLBACK_YET in ages
    if not_started and readings.processing:
        wait_since = now if wait_since is None else wait_since
        start_overdue = now - wait_since >= DEVICE_STARTUP_GRACE_SECONDS
    else:
        wait_since = None
        start_overdue = False
    if not (callback_ages or not_started):
        return Chip("Device activity: --", "idle"), wait_since
    slowest = max(callback_ages, default=0)
    state = (
        "bad" if slowest > DEVICE_ACTIVITY_BAD_MS
        else "warn" if slowest > DEVICE_ACTIVITY_WARN_MS or start_overdue
        else "ok" if callback_ages
        else "idle"
    )
    text = (
        f"Device activity: input {_ago_text(readings.input_callback_age_ms)}, "
        f"output {_ago_text(readings.output_callback_age_ms, output=True)}"
    )
    return Chip(text, state), wait_since


def runtime_health(
    snapshot: RuntimeSnapshot,
    readings: HealthReadings,
    *,
    recent_events: frozenset[str],
    memory: HealthMemory,
    now: float,
) -> HealthReport:
    """The Health chips for one diagnostics tick.

    ``recent_events`` is what :class:`RecentStreamHealth` saw rise in the last
    five seconds, ``memory`` what the previous tick returned and ``now`` the
    monotonic time of this tick. The summary chip is not included: the window
    computes it from all the chips with :func:`health_summary`.
    """
    s, r = snapshot, readings
    chips: dict[str, Chip] = {}

    # A measured route holds everything after the engine; without one the
    # figure covers AudioForge, plus the output buffer when it was read.
    latency_tip = TEXT_SPECS["latency_label"].tip
    if float(s.latency_compensation_ms) > 0.0:
        bound = "about"
    else:
        bound = "at least"
        latency_tip += (
            LATENCY_LOWER_BOUND_NOTE if s.output_device_buffer_known else LATENCY_LOWER_BOUND_NOTE_NO_BUFFER
        )
    chips["latency_label"] = Chip(
        f"Latency: {bound} {r.latency_ms:.0f} ms (processing {r.dsp_time_ms:.1f} ms)", "info", latency_tip
    )
    chips["buffer_label"] = _buffer_chip(r.input_buf + r.rnnoise_buf)

    underruns = int(s.output_underrun_total)
    underrun_streak = int(s.output_underrun_streak)
    new_underruns_observed = underruns > memory.output_underrun_total
    phase_warning_count = int(s.input_phase_warning_count)
    new_phase_warning_observed = phase_warning_count > memory.input_phase_warning_count
    true_peak_count = int(s.output_true_peak_event_count)
    new_output_true_peak_observed = true_peak_count > memory.output_true_peak_event_count
    correlation = None if s.input_stereo_correlation is None else float(s.input_stereo_correlation)
    current_phase_warning = correlation is not None and correlation < INPUT_PHASE_WARNING_CORRELATION
    phase_rescue_strategy = s.input_phase_rescue_strategy or "none"
    phase_rescue_active = phase_rescue_strategy != "none"
    output_recovery_events = int(
        s.output_recovery_event_count if "output_recovery_event_count" in s.present else s.output_recovery_count
    )
    gate_chatter_count = int(s.gate_chatter_event_count)
    new_gate_chatter_observed = "gate_chatter_event_count" in recent_events
    rt_error_active = bool(s.rt_error_name and s.rt_error_name != "none")
    limiter_history_db = float(s.limiter_gain_reduction_history_db)
    true_peak_limiter_history_db = float(s.output_true_peak_gain_reduction_history_db)
    headroom = None if s.output_true_peak_headroom_db is None else float(s.output_true_peak_headroom_db)

    input_text, input_state = input_health_state(
        rms_db=r.input_rms_db,
        peak_db=r.input_peak_db,
        full_scale_samples=r.input_full_scale_samples,
        phase_rescue_active=phase_rescue_active,
        cleanup_rumble_detected=s.input_cleanup_rumble_detected,
        cleanup_hum_detected=s.input_cleanup_hum_detected,
        cleanup_mode=s.input_cleanup_mode or "off",
        crest_factor_db=s.input_crest_factor_db,
    )
    if phase_rescue_active:
        fix = PHASE_FIX_LABELS.get(phase_rescue_strategy, phase_rescue_strategy.replace("_", " "))
        input_text = f"Input: PHASE FIX ACTIVE ({fix})"
    chips["input_health_label"] = Chip(input_text, input_state)
    chips["output_health_label"] = Chip(*output_health_state(
        muted=r.output_muted,
        rms_db=r.output_rms_db,
        at_full_scale=r.output_at_full_scale,
        limiter_enabled=r.limiter_enabled,
        peak_db=r.output_peak_db,
        true_peak_delta=new_output_true_peak_observed,
        true_peak_db=s.output_true_peak_db,
        true_peak_headroom_db=headroom,
        short_term_lufs=s.output_short_term_lufs,
        limiter_history_db=limiter_history_db,
        true_peak_limiter_history_db=true_peak_limiter_history_db,
    ))

    if s.gate_auto_relax_active:
        chips["gate_health_label"] = Chip(f"Gate: EASED ({plural(gate_chatter_count, 'time')})", "warn")
    elif new_gate_chatter_observed:
        chips["gate_health_label"] = Chip(f"Gate: CHATTERING ({plural(gate_chatter_count, 'time')})", "warn")
    else:
        chips["gate_health_label"] = Chip("Gate: OK", "ok")

    chips["callback_health_label"], device_wait_since = _device_activity(r, memory.device_wait_since, now)

    underrun_text = f"Dropouts: {underruns}"
    if underrun_streak:
        underrun_text += f" ({underrun_streak} in a row)"
    chips["underrun_health_label"] = Chip(
        underrun_text, "warn" if underrun_streak or new_underruns_observed else "ok"
    )

    dropped = s.input_dropped_samples
    restart_count = s.stream_restart_count
    dropped_bits = [
        f"Drops: {dropped}",
        f"U:{underruns}",
        f"L:{s.lock_contention_count}",
        f"NF:{s.suppressor_non_finite_count}",
        f"RS:{restart_count}",
    ]
    if underrun_streak:
        dropped_bits.append(f"US:{underrun_streak}")
    if phase_warning_count:
        dropped_bits.append(f"PH:{phase_warning_count}")
    if correlation is not None:
        dropped_bits.append(f"COR:{correlation:.2f}")
    dropped_bits += diagnostic_tokens(s, _INPUT_TOKENS)
    dropped_bits += diagnostic_tokens(s, _ENGINE_TOKENS)
    if rt_error_active:
        dropped_bits.append(f"RT:{s.rt_error_name}")
    # Most of these also have a row of their own; this row adds them all up.
    attention = [
        # Gate chatter has its own entry below; it is a setting, not an engine fault.
        (bool(recent_events - {"gate_chatter_event_count"}), "the audio engine reported a fault"),
        (r.recovery_pending, "a restart is pending"),
        (underrun_streak > 0 or new_underruns_observed, "the output ran out of audio"),
        (rt_error_active, "the audio engine reported an error"),
        (new_output_true_peak_observed, "the output went above its ceiling"),
        (new_gate_chatter_observed or s.gate_auto_relax_active, "the gate is chattering"),
        (
            phase_rescue_active or current_phase_warning or new_phase_warning_observed,
            "the microphone channels cancel each other",
        ),
        (s.input_cleanup_rumble_detected, "low rumble was detected"),
        (
            limiter_history_db >= 6.0 or true_peak_limiter_history_db >= 3.0,
            "the limiter is working hard",
        ),
        (headroom is not None and headroom < 0.75, "the output has little headroom"),
    ]
    causes = [cause for active, cause in attention if active]
    if causes and not (dropped or underruns or restart_count):
        dropped_summary = f"Drops: nothing dropped, but {causes[0]}"
    else:
        dropped_summary = (
            f"Drops: {plural(dropped, 'sample')} dropped, {plural(underruns, 'dropout')}, "
            f"{plural(restart_count, 'restart')}"
        )
        if causes:
            dropped_summary += f" (needs attention: {causes[0]})"
    chips["dropped_label"] = Chip(
        dropped_summary,
        "warn" if causes else "ok",
        f"{DROPPED_DIAGNOSTICS_TOOLTIP}\n\nTechnical counters (for support reports):\n{' | '.join(dropped_bits)}",
    )

    # A start that cannot load a model chosen while it ran has no caller to tell. Warn once
    # per new drop, before the backend warning so that one stays on the line.
    messages: list[tuple[str, int]] = []
    noise_drops = int(s.noise_selection_dropped_count)
    if noise_drops > memory.noise_drops_seen:
        messages.append((NOISE_SELECTION_DROPPED_WARNING, 8000))
    backend_warning = memory.backend_warning
    if s.noise_model != "rnnoise" and (s.noise_backend_failed or not s.noise_backend_available):
        warning = _BACKEND_STOPPED_WARNING
        if s.noise_backend_error:
            warning += f" Details: {s.noise_backend_error}"
        if warning != backend_warning:
            messages.append((warning, 6000))
            backend_warning = warning
    elif s.noise_backend_available:
        backend_warning = None

    backend_bits = [
        s.noise_model,
        "working" if s.noise_backend_available else "failed" if s.noise_backend_failed else "not available",
    ]
    if s.suppressor_non_finite_count:
        backend_bits.append(f"{plural(s.suppressor_non_finite_count, 'invalid sample')} replaced")
    if s.noise_backend_error:
        backend_bits.append("reported an error")
    if s.input_resampler_active:
        backend_bits.append("resampling input")
    if s.output_resampler_active:
        backend_bits.append("resampling output")
    chips["backend_diag_label"] = Chip(
        f"Backend: {', '.join(backend_bits)}", "ok" if s.noise_backend_available else "warn"
    )

    recovery_bits = [
        plural(restart_count, "restart"),
        plural(output_recovery_events, "output recovery", "output recoveries"),
    ]
    if s.recovery_suppressed:
        recovery_bits.append("paused during setup")
    recent_recovery = bool(recent_events & _RECOVERY_EVENTS)
    if r.recovery_pending:
        recovery_bits.append("restart pending")
    elif recent_recovery:
        recovery_bits.append("restarted just now")
    chips["recovery_diag_label"] = Chip(
        f"Recovery: {', '.join(recovery_bits)}",
        "warn" if r.recovery_pending or recent_recovery else ("info" if s.recovery_suppressed else "ok"),
        f"Last recovery: {s.last_restart_reason}" if s.last_restart_reason else "No recovery recorded.",
    )

    return HealthReport(
        chips=chips,
        messages=tuple(messages),
        memory=HealthMemory(
            output_underrun_total=underruns,
            input_phase_warning_count=phase_warning_count,
            output_true_peak_event_count=true_peak_count,
            device_wait_since=device_wait_since,
            noise_drops_seen=noise_drops,
            backend_warning=backend_warning,
        ),
    )


def health_summary(rows: Sequence[Chip], *, mute_error: bool = False) -> Chip:
    """The summary chip for ``rows``, given in reporting order; its tip is the advice line.

    It takes the worst row's state and names that row's problem; info rows do not count.
    """
    levels = {row.state for row in rows}
    state = next((level for level in ("bad", "warn", "ok") if level in levels), "idle")
    if mute_error and state in {"ok", "idle"}:
        state = "warn"
    issues = [row.text for row in rows if row.state == "bad"] or [
        row.text for row in rows if row.state == "warn"
    ]
    advice = advice_for(issues[0]) if issues else ""
    if not advice:
        advice = {
            "ok": "Everything looks fine.",
            "idle": TEXT_SPECS["health_advice_label"].text,
        }.get(state, "Check the items marked below.")
    return Chip(issues[0] if issues else f"Health: {state.upper()}", state, advice)
