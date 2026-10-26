"""Plain wording for what Voice Setup and Auto-EQ tell the user.

The analysis modules keep their reason strings, because evaluators and tests match on
them. The dialogs show these sentences instead: what happened, why, and what to do,
with the number where the reason carries one. A reason with no entry is shown as it was
given, never dropped; the test in test_setup_messages.py fails when one has no entry.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, Decimal
from typing import Any

from .auto_eq_explanation import reason_message as _auto_eq_reason_message
from ..analysis.noise_reference import SILENT_REFERENCE_REASON
from .calibration_support import diagnostic_state, format_percent

logger = logging.getLogger(__name__)

# (pattern the whole reason must match, sentence). Named groups fill the sentence.
_REASON_RULES: tuple[tuple[str, str], ...] = (
    # Voice Setup's own findings.
    (r"too little VAD-active speech",
     "Setup heard less than 2 seconds of speech. Read the whole passage without long pauses."),
    (r"speech-to-noise ratio is weak",
     "Your voice is not much louder than the room. Move closer to the microphone or record in a quieter room."),
    (r"too few voiced frames for a stable spectrum",
     "There were too few clear voiced sounds to measure the shape of your voice. Read the whole passage at a steady volume."),
    (r"overall capture quality is low",
     "The recording was unclear overall. Record again in a quieter room, closer to the microphone."),
    (r"full-chain validation failed: typed candidate EQ settings are unavailable.*",
     "The tone step gave no usable result, so the final safety check could not run. Record the voice again to retry."),
    (r"full-chain validation failed: .*",
     "The final safety check could not run on this recording. Record the voice again; if it keeps happening, restart AudioForge."),
    (r"full-chain gate/suppressor silenced active speech",
     "With these settings the noise gate or noise suppression would cut off your voice, so they are not recommended."),
    (r"raw full-chain capture is unavailable for input validation",
     "The unprocessed recording was not available for the final safety check."),
    (r"the microphone level leaves too little headroom; lower the input gain by about (?P<db>[\d.]+) dB",
     "Your microphone level is too high for safe loudness. Lower it by about {db} dB, in Windows or on the microphone, then record again."),
    (r"final full-chain headroom validation did not pass",
     "At this microphone level loud peaks would overload the limiter. Lower the microphone level and record again."),
    (r"offline DSP validation did not pass",
     "The final safety check on your recording did not pass, so these settings are suggestions only."),
    (r"offline DSP validation could not run, so these settings are advisory",
     "The final safety check could not run on this computer, so treat these settings as suggestions."),
    (r"Auto-EQ abstained from this capture",
     "The tone step found nothing it could safely change with this recording."),
    (r"compressor p95 gain reduction target is unavailable",
     "The compressor could not be tuned on your recording, so its settings are a rough guess."),
    (r"compressor p95 gain reduction target missed: measured (?P<m>[\d.]+) dB, target (?P<t>[\d.]+) dB.*",
     "The compressor evens out your voice by {m} dB; the chosen style aims for {t} dB. Pick another compression style, or record again."),
    # The room-noise recording.
    (r"room-noise capture is silent",
     "No background noise came through. That is normal when a headset or its software mutes the microphone between words; otherwise check the microphone."),
    (r"room-noise capture is too short",
     "The room-noise recording was too short. Record at least 1.5 seconds of room tone."),
    (r"room-noise capture has too few analysis windows",
     "The room-noise recording was too short to measure. Record at least 1.5 seconds of room tone."),
    (r"room-noise capture contains non-finite samples",
     "The room-noise recording contained invalid data. Restart processing and record the room noise again."),
    (r"room-noise capture is clipped",
     "The room-noise recording hit full scale. Lower the microphone level or avoid loud sounds, then record it again."),
    (r"room-noise capture contains isolated clipped samples",
     "A few moments of the room-noise recording hit full scale, as a tap would. Record it again without touching the microphone."),
    (r"room-noise capture is dominated by changing events",
     "The room noise changed a lot during the recording. Wait for the room to settle and record it again."),
    (r"room-noise capture is not stationary",
     "The room noise was not steady (movement, voices or sounds that come and go). Record it again while the room is still."),
    (r"room-noise capture contains dominant transient events",
     "Knocks or clicks dominated the room-noise recording. Record it again without keyboard, handling or impact sounds."),
    (r"room-noise capture contains strong transients",
     "The room-noise recording contained sharp sounds such as keys or knocks. Record it again without them."),
    (r"speech is present in the room-noise capture",
     "Speech was heard in the room-noise recording. Stay silent and record the room noise again."),
    (r"possible speech contamination in room-noise capture",
     "There may have been voices in the room-noise recording. Record it again without voices."),
    (r"sample rate changed between noise and voice captures",
     "The sample rate changed between the room-noise and voice recordings. Restart processing and record both again."),
    (r"(?P<what>.+) changed between noise and voice captures",
     "The {what} changed between the room-noise and voice recordings. Record both again without changing the microphone."),
    (r"(?P<which>.+) capture sample rate does not match analysis",
     "The {which} recording used a different sample rate than the audio engine. Restart processing and record again."),
    (r"room-noise capture timestamp is after voice capture",
     "The recordings are out of order. Record the room noise first, then your voice."),
    (r"room-noise reference is stale",
     "The room-noise recording is more than 10 minutes old. Record it again just before your voice."),
    (r"room-noise reference may be stale",
     "The room-noise recording is more than 2 minutes old. Record it again just before your voice."),
    (r"room noise does not match conditions during the voice capture",
     "The room sounded different while you spoke than during the room-noise recording, for example a fan or air conditioner changed. Record both again in the same conditions."),
    (r"room-noise reference only partly matches the voice capture",
     "The room sounded a little different while you spoke. Record both again for a more reliable result."),
    (r"room-noise level changed substantially before the voice capture",
     "The room was much quieter while you spoke than when the room noise was recorded. Record both again in the same conditions."),
    (r"room-noise reference is much louder than in-capture quiet frames",
     "The room was quieter while you spoke than when the room noise was recorded. Check whether a noise source was switched off, then record both again."),
    # Noise suppression and gate search.
    (r"incumbent retained unless a bounded candidate clearly improves all gates",
     "Nothing tested clearly beat your current noise suppression and gate, so they were left alone."),
    (r"bounded candidate passed speech, noise, safety, and runtime gates with compression bypassed",
     "New noise suppression and gate settings kept your voice intact and removed more noise."),
    (r"no candidate passed every predefined gate",
     "None of the noise suppression and gate settings tested passed every check, so yours were left alone."),
    (r"noise capture has no complete frame after the 600 ms settling interval",
     "The room-noise recording was too short once its first 0.6 seconds were skipped. Record about 2 seconds of room tone."),
    (r"(?P<n>[\d.]+) candidate evaluations were unavailable",
     "{n} noise suppression options could not be tested on this computer."),
    (r".+ unavailable: .*",
     "One noise suppressor could not be tested on this computer, so it was not considered."),
    (r"holdout skipped because detector evidence was unavailable",
     "The check on the second half of your recording was skipped because speech detection was not available."),
    (r"noise suppression is disabled in the incumbent settings",
     "Noise suppression is switched off, so it was not tuned."),
    (r"joint tuning captures were empty or non-finite",
     "The recordings were empty or unreadable, so noise suppression and the gate were not tuned."),
    (r"joint tuning captures are too short for a held-out split",
     "The recordings were too short to test noise suppression on a second half. Record a longer passage."),
    (r"speech/noise VAD availability differs; continuous evidence is unavailable",
     "Speech detection worked on only one of the two recordings, so noise suppression and the gate were not tuned. Record both again."),
    (r"native gate/suppressor simulator is unavailable",
     "This computer could not test noise suppression and the gate, so they were not tuned."),
    (r"joint candidate evaluation failed",
     "Noise suppression and the gate could not be tested, so they were left alone."),
    # The second take (verification).
    (r"verification passage was too short; repeat for at least 3 seconds",
     "The second recording was too short. Read the passage for at least 3 seconds."),
    (r"(?:verification passage|raw .+ capture) was non-finite or clipped.*",
     "The second recording was clipped or unreadable. Lower the microphone level and record it again."),
    (r"candidate (?:EQ settings are missing|EQ bands are incomplete|limiter settings are missing)",
     "The settings to check are incomplete. They were taken off again; record the voice again to start over."),
    (r"native verification renderer is unavailable",
     "This computer could not check the settings on your second recording, so they were taken off again."),
    (r"full-chain gate/suppressor reduced active speech by (?P<db>[\d.]+) dB",
     "With these settings the noise gate or noise suppression cut your voice by {db} dB. Lower the noise suppression strength."),
    (r"verification delivery differs too much from the setup passage \((?P<detail>.*)\)\. Repeat the same passage.*",
     "The second recording did not sound like the first ({detail}). Read the same passage again at the same distance and volume."),
    (r"delivery level delta was (?P<db>[\d.]+) dB",
     "The volume of the second recording differed by {db} dB. Read the same passage again at the same distance and volume."),
    (r"delivery shape delta was (?P<db>[\d.]+) dB",
     "The tone of the second recording differed by {db} dB. Read the same passage again at the same distance and volume."),
    (r"target error worsened from (?P<a>[\d.]+) to (?P<b>[\d.]+) dB",
     "The tone got further from the preset: {a} dB away before, {b} dB after."),
    (r"relative noise floor increased by (?P<db>[\d.]+) dB",
     "Background noise grew by {db} dB compared with your voice."),
    (r"candidate increased relative noise floor by (?P<db>[\d.]+) dB",
     "Background noise grew by {db} dB compared with your voice."),
    (r"SNR fell by (?P<db>[\d.]+) dB",
     "Your voice stood out from the background noise by {db} dB less than before."),
    (r"compressor peak reduction reached (?P<a>[\d.]+) dB \(cap (?P<b>[\d.]+) dB\)",
     "The compressor squeezed the loudest sounds by {a} dB, more than the {b} dB allowed."),
    (r"output true peak reached (?P<a>-?[\d.]+) dBTP \(ceiling (?P<b>-?[\d.]+) dBTP\)",
     "The loudest peak reached {a} dB, above the {b} dB limit."),
    (r"compressor p95 reduction exceeded target by (?P<db>[\d.]+) dB",
     "The compressor evened out your voice {db} dB more than planned."),
    (r"(?P<stage>.*) pressure reached (?P<db>[\d.]+) dB.*",
     "The limiter had to turn peaks down by {db} dB."),
    (r"de-esser p95 reduction reached (?P<db>[\d.]+) dB",
     "The de-esser turned your S sounds down by {db} dB."),
    (r"repeatability and .+ DSP constraints passed",
     "The second recording sounded like the first and the new settings passed every check."),
    # What the Auto-EQ engine already says plainly.
    (r"Recording too unclear\. Please try again\.",
     "Setup could not find a steady voice in this recording, because too few clear speech sounds stood out from the noise. Record again closer to the microphone and read the whole passage."),
    (r"The microphone level is too high for a safe result\. ?(?P<rest>.*)",
     "The microphone level is too high for a safe result. {rest}"),
)
_COMPILED_REASONS = tuple((re.compile(pattern, re.S), text) for pattern, text in _REASON_RULES)

_ERROR_RULES: tuple[tuple[str, str], ...] = (
    (r"(Audio capture exceeded its .*|No audio capture progress was received\.|Recording failed - no audio data)",
     "The microphone stopped sending sound. Check that it is plugged in and not used by another app, then record again."),
    (r"Could not find (?:audio )?processor",
     "AudioForge lost its audio engine. Close this window and open it again."),
    (r"Room-noise capture was too short for setup\.",
     "The room-noise recording was too short. Record at least 2 seconds of room tone."),
    (r"Voice capture was too short for setup\.",
     "The voice recording was too short. Read the whole passage."),
    (r"Audio too short for FFT.*",
     "The recording was too short to analyse. Record again and read the whole passage."),
    (r"Missing room-noise or speech capture\.",
     "A recording is missing. Start the setup again."),
    (r"raw setup captures are non-finite or clipped",
     "The recording hit full scale or held invalid data. Lower the microphone level and record again."),
    (r"Recording error: (?P<e>.*)",
     "The recording stopped unexpectedly ({e}). Check that the microphone is connected and not used by another app, then try again."),
)
_COMPILED_ERRORS = tuple((re.compile(pattern, re.S), text) for pattern, text in _ERROR_RULES)


def _apply(rules: tuple[tuple[re.Pattern[str], str], ...], text: str) -> str | None:
    for pattern, sentence in rules:
        match = pattern.fullmatch(text.strip())
        if match is not None:
            groups = match.groupdict()
            if "detail" in groups:  # the two delivery findings of a mismatched second take
                groups["detail"] = re.sub(
                    r"delivery level delta was ([\d.]+) dB", r"the volume differed by \1 dB", groups["detail"]
                )
                groups["detail"] = re.sub(
                    r"delivery shape delta was ([\d.]+) dB", r"the tone differed by \1 dB", groups["detail"]
                )
            return sentence.format(**groups).strip()
    return None


def is_mapped(reason: str) -> bool:
    """Whether a reason from the analysis has a plain sentence."""
    return _apply(_COMPILED_REASONS, reason) is not None or _auto_eq_reason_message(reason) is not None


def plain_reason(reason: str) -> str:
    """The sentence for an analysis reason; the reason itself when it has none."""
    sentence = _apply(_COMPILED_REASONS, reason)
    if sentence is not None:
        return sentence
    auto_eq = _auto_eq_reason_message(reason)
    return auto_eq[1] if auto_eq is not None else reason


def plain_reasons(reasons: Iterable[object]) -> list[str]:
    """Plain sentences for a list of reasons, each said once, in order."""
    sentences: list[str] = []
    for reason in reasons:
        sentence = plain_reason(str(reason))
        if sentence not in sentences:
            sentences.append(sentence)
    return sentences


def plain_error(text: str) -> str:
    """A sentence for an error raised or reported while recording or analysing."""
    sentence = _apply(_COMPILED_ERRORS, text)
    if sentence is not None:
        return sentence
    if is_mapped(text):
        return plain_reason(text)
    return f"The analysis could not finish: {text.rstrip('.')}. Record again; if it keeps happening, restart AudioForge."


def plain_stale(error: str) -> str:
    """One of two sentences for the many ways a finished result can stop matching its recordings."""
    if re.search(r"route|input settings|sample rate", error, re.I):
        return (
            "The microphone or its settings changed after you recorded. "
            "Record again so the settings match your microphone."
        )
    return "These settings no longer match your recordings. Record again."


def quality_word(score: float, *, weak: bool = False) -> tuple[str, str]:
    """The words for a capture-quality score and the status colour that goes with them."""
    state = diagnostic_state(score)
    if weak:
        return "Weak capture", "bad" if state == "bad" else "warn"
    return {"ok": "Good capture", "warn": "Usable capture"}.get(state, "Weak capture"), state


def quality_tooltip(score: float, detail: str = "") -> str:
    """The number behind a quality word, and what it is not."""
    return (
        f"Capture quality score {format_percent(score)}. It rates how clean and complete the "
        "recording was; it is not the chance that the settings are right."
        + (f" {detail}" if detail else "")
    )


def _floor(value: float | Decimal, places: int) -> str:
    """The value cut off after ``places`` decimals, never rounded up to a threshold it is under."""
    exact = value if isinstance(value, Decimal) else Decimal(repr(float(value)))
    return str(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_FLOOR))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


_WITHHELD = (
    "Automatic loudness was not switched on{makeup} and the settings are suggestions only."
)


def _capture_issues(
    diagnostics: Mapping[str, Any], compressor: Mapping[str, Any] | None = None
) -> list[tuple[str, str]]:
    """A stable code and a plain line for each reason the capture was judged weak or unreliable.

    ``capture_reliable`` has four clauses in the analysis (voice against the room noise
    under 6 dB, under 2 s of speech, a single-spectrum fallback, a room-noise recording
    that is not "usable"). Any one of them withholds automatic loudness. All but a
    questionable room-noise recording also make the capture weak, which turns the
    recommendation into suggestions only; a questionable one keeps the current noise
    suppression and gate instead. A weak capture can also come from a voice under 8 dB
    over the room or a low overall score. Each line says what was found, what was held
    back, and what to do.
    """
    makeup = _number((compressor or {}).get("makeup_gain_db"))
    fixed = f" (makeup is fixed at {makeup:.1f} dB)" if makeup is not None else ""
    withheld = _WITHHELD.format(makeup=fixed)
    suggestions = "The settings are suggestions only."
    lines: list[tuple[str, str]] = []
    snr = _number(diagnostics.get("noise_referenced_snr_db"))
    if snr is not None and snr < 6.0:
        lines.append((
            "snr_under_6_db",
            f"Your voice was only {_floor(snr, 1)} dB louder than the room noise; "
            f"setup needs at least 6 dB. {withheld} Record again closer to the microphone, or in a quieter room.",
        ))
    elif snr is not None and snr < 8.0:
        lines.append((
            "snr_under_8_db",
            f"Your voice was {_floor(snr, 1)} dB louder than the room noise; setup is happiest with 8 dB or more. "
            f"{suggestions} Record again closer to the microphone, or in a quieter room.",
        ))
    active = _number(diagnostics.get("vad_active_duration_s"))
    if active is not None and active < 2.0:
        lines.append((
            "speech_under_2_s",
            f"Setup heard {_floor(active, 1)} seconds of speech; it needs at least 2. "
            f"{withheld} Record again and speak for the whole take, without long pauses.",
        ))
    reasons = [str(reason) for reason in diagnostics.get("uncertainty_reasons") or ()]
    fallback = "too few voiced frames for a stable spectrum" in reasons
    if fallback:
        lines.append((
            "spectrum_fallback",
            "There were too few clear voiced sounds to measure the shape of your voice. "
            f"{withheld} Record again and speak for the whole take at a steady volume.",
        ))
    quality = diagnostics.get("noise_reference_quality")
    if isinstance(quality, Mapping) and quality.get("status", "usable") != "usable":
        found = plain_reasons(quality.get("reasons") or ()) or [
            "The room-noise recording was not clean enough to rely on."
        ]
        again = (
            "Check that the microphone is not muted, then record the room noise again."
            if SILENT_REFERENCE_REASON in (quality.get("reasons") or ())
            else "Record the room noise again without talking."
        )
        held = withheld if quality.get("status") != "questionable" else (
            "Your noise suppression and gate were left as they are, and automatic loudness was not "
            f"switched on{fixed}." + (f" {suggestions}" if diagnostics.get("weak_capture") else "")
        )
        lines.append(("noise_reference_not_usable", f"{found[0]} {held} {again}"))
    capture = _number(diagnostics.get("capture_confidence"))
    if capture is not None and capture < 0.50 and not fallback:
        lines.append((
            "capture_quality_low",
            f"The recording was unclear overall (capture quality "
            f"{_floor(Decimal(repr(capture)).scaleb(2), 0)}%). "
            f"{suggestions} Record again in a quieter room, closer to the microphone.",
        ))
    if diagnostics.get("weak_capture") and not lines:
        lines.append((
            "weak_capture",
            "The recording did not pass setup's quality checks"
            ", so the settings are suggestions only. "
            "Record again in a quiet room, closer to the microphone.",
        ))
    return lines


def capture_issue_lines(
    diagnostics: Mapping[str, Any], compressor: Mapping[str, Any] | None = None
) -> list[str]:
    """One plain line for each reason the capture was judged weak or unreliable."""
    return [line for _code, line in _capture_issues(diagnostics, compressor)]


def capture_issue_codes(diagnostics: Mapping[str, Any]) -> list[str]:
    """One stable code for each reason the capture was weak, for the diagnostics export."""
    return [code for code, _line in _capture_issues(diagnostics)]


# The reasons capture_issue_lines says in its own words.
_CAPTURE_REASONS = (
    "too little VAD-active speech",
    "speech-to-noise ratio is weak",
    "too few voiced frames for a stable spectrum",
    "overall capture quality is low",
)


def other_reasons(diagnostics: Mapping[str, Any]) -> list[str]:
    """The uncertainty reasons that capture_issue_lines does not already say."""
    told = set(_CAPTURE_REASONS)
    quality = diagnostics.get("noise_reference_quality")
    if isinstance(quality, Mapping) and quality.get("status", "usable") != "usable":
        told.update(list(quality.get("reasons") or ())[:1])
    return [str(reason) for reason in diagnostics.get("uncertainty_reasons") or () if reason not in told]


def verification_summary(result: Mapping[str, Any]) -> str:
    """The measured outcome of the second recording, in words."""
    def value(key: str, default: float) -> float:
        number = _number(result.get(key))
        return default if number is None else number

    return (
        f"Tone: {value('spectral_target_error_before_db', 0.0):.1f} dB from the preset before, "
        f"{value('spectral_target_error_after_db', 0.0):.1f} dB after. "
        f"The compressor evened out your voice by {value('compressor_gain_reduction_p95_db', 0.0):.1f} dB. "
        f"The loudest peak is {value('output_true_peak_db', -120.0):.1f} dB. "
        f"Your voice against the background noise changed by {value('snr_change_db', 0.0):+.1f} dB."
    )


def plain_advice(advice: object) -> str:
    """The note about noise suppression dulling consonants, as a sentence."""
    text = str(advice).strip().rstrip(".")
    match = re.fullmatch(r"noise suppression is dulling consonants by about (\d+) dB; lower its strength( or use DeepFilterNet LL)?", text)
    if match is None:
        return text[:1].upper() + text[1:] + "."
    return (
        f"Noise suppression is dulling your S and T sounds by about {match.group(1)} dB. "
        "Lower its strength" + (" or try DeepFilterNet LL." if match.group(2) else ".")
    )


def plain_incomplete(error: str) -> str:
    """Why a result cannot be used, as one sentence; parts the person never sees are summed up."""
    if error.startswith(("candidate", "voice setup result", "Validated")):
        return "Part of the result is missing."
    return plain_error(error)


def reason_code(reason: str) -> str:
    """A short stable code for an analysis reason, for the diagnostics export."""
    if not is_mapped(reason):
        return "other"
    words = [word for word in re.split(r"[^a-z]+", reason.lower()) if word]
    return "_".join(words)[:64].rstrip("_")


def numbers_from(source: Mapping[str, Any], keys: Iterable[str]) -> dict[str, float | bool]:
    """The finite numbers and flags among ``keys``, for the diagnostics export."""
    found: dict[str, float | bool] = {}
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            found[key] = value
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            found[key] = float(value)
    return found


def report_setup_result(
    owner: object,
    kind: str,
    outcome: str,
    *,
    reason_codes: Iterable[str] = (),
    messages: Iterable[str] = (),
    numbers: Mapping[str, Any] | None = None,
) -> None:
    """Tell the main window how a run ended, so the diagnostics export can say so.

    Owners that do not record results (a test's stand-in, an older window) are left alone.
    """
    record = getattr(owner, "record_setup_result", None)
    if not callable(record):
        return
    try:
        record(
            {
                "kind": kind,
                "outcome": outcome,
                "reason_codes": list(dict.fromkeys(reason_codes)),
                "messages": [str(message) for message in messages if message],
                "numbers": dict(numbers or {}),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception:
        logger.warning("The main window could not record the setup result", exc_info=True)
