"""Microphone level verdicts shared by Voice Setup and Auto-EQ.

The 3-second check and the advice on a refused take are judged from the samples, where
clipping means what raw_capture_is_clipped says, the rule that refuses a take. The live meter
only sees block levels and the meter's peak, so it can warn about a single loud sample that
the check later treats as a click. Both dialogs share all of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..analysis.noise_reference import capture_is_silent
from ..analysis.voice_setup import CLIP_LEVEL, raw_capture_is_clipped
from .calibration_support import TOO_LOUD_DB, TOO_QUIET_DB

LEVEL_CHECK_SECONDS = 3.0
# A quiet verdict needs this many 100 ms level updates (1.5 s) without any block above
# the threshold, so the pauses between words do not trigger it.
LEVEL_HOLD_UPDATES = 15
# Where a corrected level should land: a peak this low keeps 3 dB below TOO_LOUD_DB, and the
# speech level is where the analysis has room above any ordinary room's noise.
TARGET_PEAK_DB = -6.0
TARGET_SPEECH_DB = -25.0
# The third loudest 100 ms frame stands for "how loud you talk": the pauses between words are
# quiet by nature, and a bump of two frames does not count, but 0.3 s of speech does.
_SPEECH_FRAME_RANK = 3
_FRAME_SECONDS = 0.1
# Unless the take is clipped, the two loudest samples are ignored so one click cannot
# decide the verdict.
_IGNORED_PEAK_SAMPLES = 2
# No microphone setting adds more than this; a voice further below the target needs more
# than a level change.
_MAX_RAISE_DB = 30
# Speech seldom peaks more than this far above its own loud frames. A larger gap means the
# frames were mostly silence, so the speech level says nothing about the microphone setting.
_MAX_PEAK_ABOVE_SPEECH_DB = 40
ROOM_LOUD_DB = -35.0


@dataclass(frozen=True)
class LevelVerdict:
    code: str  # good, too_hot, too_quiet, silent or unreadable
    message: str
    state: str  # the message_text_style state: ok, warn or bad
    peak_db: float | None = None
    speech_db: float | None = None
    # Positive: raise the microphone level by this much. Negative: lower it.
    change_db: int = 0

    @property
    def can_override(self) -> bool:
        """Whether the user may knowingly go on despite the verdict."""
        return self.code in ("too_hot", "too_quiet")


def _db(value: float, digits: int = 1) -> str:
    """A dB number for a message: a value that rounds to zero prints as 0, never -0."""
    text = f"{value:.{digits}f}"
    return text.removeprefix("-") if float(text) == 0 else text


def _silent_message(again: str) -> str:
    return (
        "No sound came from the microphone. Check that it is not muted and "
        f"that the right input is selected, then {again}."
    )


def verdict_from_levels(
    peak_db: float, speech_db: float, *, again: str = "check again", clipped: bool = False
) -> LevelVerdict:
    """Judge a peak and the level of the loud words, both in dB below full scale.

    The advice comes from the peak: a take can be hot without being clipped.
    """
    if peak_db > TOO_LOUD_DB:
        lower = max(1, math.ceil(peak_db - TARGET_PEAK_DB - 0.05))
        what = (
            f"Your voice is clipping (peaks reach {_db(peak_db)} dB)"
            if clipped
            else f"Your voice peaks at {_db(peak_db)} dB, close to the limit"
        )
        return LevelVerdict(
            "too_hot",
            f"{what}: lower the microphone level by about {lower} dB "
            f"(in Windows or on the microphone), then {again}.",
            "bad",
            peak_db,
            speech_db,
            -lower,
        )
    if speech_db < TOO_QUIET_DB:
        # Rounded down so a level just under the threshold never prints as the threshold.
        heard = math.floor(speech_db)
        if peak_db - speech_db > _MAX_PEAK_ABOVE_SPEECH_DB:
            return LevelVerdict(
                "too_quiet",
                f"Little or no speech was heard: the loudest words were {heard} dB, and setup "
                f"needs {TOO_QUIET_DB:.0f} dB or louder to hear them above the room. "
                f"Speak for the whole check, then {again}.",
                "warn",
                peak_db,
                speech_db,
            )
        # The least whole-dB raise that reaches the speech threshold; if it would push the
        # peak over the too-hot limit, a level change cannot fix this voice.
        least = math.ceil(TOO_QUIET_DB - speech_db)
        if peak_db + least > TOO_LOUD_DB:
            return LevelVerdict(
                "too_quiet",
                f"Your voice is quiet: {heard} dB when you speak, and setup needs "
                f"{TOO_QUIET_DB:.0f} dB or louder to hear it above the room, but your loudest "
                "moments are already close to the limit, so a higher level would clip them. "
                f"Move closer to the microphone or speak more evenly, then {again}.",
                "warn",
                peak_db,
                speech_db,
            )
        needed = max(least, math.floor(min(TARGET_SPEECH_DB - speech_db, TARGET_PEAK_DB - peak_db)))
        raise_by = min(needed, _MAX_RAISE_DB)
        fix = (
            f"Raise the microphone level by about {raise_by} dB (in Windows or on the "
            "microphone) or move closer"
            if needed <= _MAX_RAISE_DB
            else "Raise the microphone level as far as it goes (in Windows or on the "
            "microphone) and move closer"
        )
        return LevelVerdict(
            "too_quiet",
            f"Your voice is quiet: {heard} dB when you speak, and setup needs "
            f"{TOO_QUIET_DB:.0f} dB or louder to hear it above the room. {fix}, then {again}.",
            "warn",
            peak_db,
            speech_db,
            raise_by,
        )
    return LevelVerdict(
        "good",
        f"The level is good: your voice peaks at {_db(peak_db)} dB.",
        "ok",
        peak_db,
        speech_db,
    )


def verdict_from_audio(
    audio: np.ndarray, sample_rate: int = 48_000, *, again: str = "check again"
) -> LevelVerdict:
    """Judge a finished take from its samples, not from a meter that can miss a peak."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        return LevelVerdict(
            "unreadable", "The recording could not be read. Check the level again.", "bad"
        )
    if capture_is_silent(samples):
        return LevelVerdict("silent", _silent_message(again), "bad")
    clipped = raw_capture_is_clipped(samples, sample_rate)
    magnitudes = np.abs(samples)
    ignored = 0
    if not clipped:
        # Samples at full scale that the clipping rule lets through are clicks, not the voice.
        voice = magnitudes[magnitudes < CLIP_LEVEL]
        magnitudes = voice if voice.size else magnitudes
        ignored = min(_IGNORED_PEAK_SAMPLES, magnitudes.size - 1)
    peak = float(np.partition(magnitudes, magnitudes.size - 1 - ignored)[magnitudes.size - 1 - ignored])
    frame = max(1, int(round(_FRAME_SECONDS * sample_rate)))
    usable = (samples.size // frame) * frame
    if usable:
        rms = np.sqrt(np.mean(np.square(samples[:usable].reshape(-1, frame), dtype=np.float64), axis=1))
        frame_db = np.sort(20.0 * np.log10(rms + 1e-9))
        speech = float(frame_db[max(0, frame_db.size - _SPEECH_FRAME_RANK)])
    else:
        speech = float(20.0 * np.log10(np.sqrt(np.mean(np.square(samples, dtype=np.float64))) + 1e-9))
    return verdict_from_levels(
        20.0 * math.log10(peak + 1e-9),
        speech,
        again=again,
        clipped=clipped,
    )


def live_verdict(
    recent_rms_db: Sequence[float],
    recent_peak_db: Sequence[float],
    *,
    again: str = "record again",
) -> LevelVerdict | None:
    """The verdict for the meter during a take; None until there is something to say.

    The two windows hold the same recent blocks, so the peak named is the loudest one of
    them, not the latest block's. A loud peak is said at once. A quiet voice needs a full
    window of recent blocks, because the latest one dips on every pause between words.
    """
    full = len(recent_rms_db) >= LEVEL_HOLD_UPDATES
    loudest = max(recent_rms_db, default=-120.0)
    peak_db = max(recent_peak_db, default=-120.0)
    if full and loudest < TOO_QUIET_DB:
        return verdict_from_levels(min(peak_db, TOO_LOUD_DB), loudest, again=again)
    if peak_db > TOO_LOUD_DB or full:
        return verdict_from_levels(peak_db, loudest, again=again)
    return None


def room_noise_message(recent_rms_db: Sequence[float]) -> tuple[str, str]:
    """What the meter says about the room during the room-noise take, and its state."""
    loudest = max(recent_rms_db, default=-120.0)
    if loudest > ROOM_LOUD_DB:
        return (
            f"The room is loud ({_db(loudest, 0)} dB). Setup will play safe; a quieter "
            "room gives better results.",
            "warn",
        )
    return "The room is quiet enough.", "ok"


def clipped_message(audio: np.ndarray, sample_rate: int) -> str:
    """Refuse a finished take whose loud parts were held at full scale, with the fix in decibels."""
    change = verdict_from_audio(audio, sample_rate).change_db
    lower = -change if change < 0 else 6
    return (
        "The recording clipped: its peaks were held at full scale. Lower the microphone "
        f"level by about {lower} dB (in Windows or on the microphone) or move back a "
        "little, then record again."
    )


@dataclass(frozen=True)
class LevelChoices:
    primary: str  # the main button: go on when the level is good, otherwise check again
    override: str | None  # the clearly worded secondary action; None when going on is not allowed
    note: str  # what going on anyway would mean


def level_choices(verdict: LevelVerdict, proceed: str) -> LevelChoices:
    """The buttons and the warning that go with a verdict; ``proceed`` names the next take."""
    if verdict.code == "good":
        return LevelChoices(proceed, None, "")
    if verdict.code == "too_quiet":
        return LevelChoices(
            "Check Level Again",
            "Continue Anyway",
            "You can continue anyway, but a quiet voice makes the settings less reliable.",
        )
    if verdict.code == "too_hot":
        return LevelChoices(
            "Check Level Again",
            "Continue Despite Clipping Risk",
            "If you continue anyway, loud passages may clip and the recording may be refused.",
        )
    return LevelChoices("Check Level Again", None, "")
