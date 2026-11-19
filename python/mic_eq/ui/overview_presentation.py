"""What the Overview's status sentence says, worked out from the window's state.

Pure: the window gathers :class:`OverviewInputs` and the scene shows the
:class:`OverviewView`, so each state's wording is testable without Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from .platform_text import SOUND_SETTINGS

DOT = "·"
RECOMMEND_SETUP_HINT = (
    "Recommended first: sets EQ, noise gate, de-esser and compressor from two short recordings."
)
# Beside the disabled Start button, and in the status bar when Start is asked for anyway.
NO_MIC_REASON = "No microphone to start from"
NO_MIC_DETAIL = (
    "No input devices detected. Connect your microphone, then press Refresh audio devices: "
    "AudioForge looks for new devices only when you ask."
)
CLIPPING_NOTE = "Clipping happens before AudioForge, so the limiter can't fix it."
MUTED_DETAIL = (
    "Your microphone is still on, but nothing reaches the call. Press Ctrl+M to unmute."
)
# The engine refused the mute, so it is pending, not in force.
MUTE_PENDING_ISSUE = "Mute pending: the mute did not apply, so callers still hear your microphone."
UNMUTE_PENDING_ISSUE = "Unmute pending: the unmute did not apply, so callers may still hear silence."
CALLERS_LINE = "Callers hear nothing until the audio is back."
RETRYING_DETAIL = (
    "The selected devices stopped, and AudioForge keeps retrying them. "
    "Check that the device is connected."
)
LIVE_HEADLINES = {
    "normal": f"Live {DOT} your voice is clean",
    "bypass": f"Live {DOT} Bypass, voice chain off",
    "raw": f"Live {DOT} Raw Monitor, no processing",
}
LIVE_NAV_LINES = {
    "normal": f"Live {DOT} all stages",
    "bypass": f"Live {DOT} Bypass",
    "raw": f"Live {DOT} Raw Monitor",
}


@dataclass(frozen=True, slots=True)
class OverviewInputs:
    """The window state the Overview reads; texts are the Health rows' own words."""

    now: datetime
    running: bool = False
    no_mic: bool = False
    restart_pending: bool = False
    # The Input list's label of the microphone a switch from the device-lost page could not
    # start on; empty otherwise.
    failed_switch_device: str = ""
    device_bad: bool = False
    device_advice: str = ""
    input_bad: bool = False
    input_advice: str = ""
    # A mute the engine accepted; a refused one is ``mute_error`` instead.
    muted: bool = False
    mute_error: bool = False
    # What the refused change asked for: a mute, or an unmute.
    mute_requested: bool = False
    # The advice of each Health row that warns or fails, worst first.
    issues: tuple[str, ...] = ()
    mode: str = "normal"
    backend: str = ""
    output_peak_db: float | None = None
    input_name: str = ""
    output_name: str = ""
    output_found: bool = True
    # Whether the output looks like a virtual cable another app records from.
    output_virtual: bool = False
    preset_name: str = "Default"
    # When the newest Auto Voice Setup was saved, whether or not it still matches.
    last_setup: datetime | None = None


@dataclass(frozen=True, slots=True)
class OverviewView:
    state: str
    headline: str
    detail: str
    # idle, ok, warn, bad, mute or callers: the dot's colour; the words carry the meaning.
    tone: str
    nav_line: str
    ribbon_caption: str
    recommend_setup: bool
    setup_line: str
    # The card under the sentence: each line and whether it is "ok", a "warn" or a "note".
    checks_title: str
    checks: tuple[tuple[str, str], ...]
    # A manual restart while the engine waits to retry would cancel that retry, and if the
    # devices are still gone nothing would reconnect them later; it is offered only for
    # running audio whose callbacks stalled, and after a switch that could not start.
    offer_restart: bool = False
    # Another microphone is the way out when the device is gone, so it is offered while the
    # engine retries too: the stop cancels a retry for the device being replaced.
    offer_choose_microphone: bool = False
    # The notice under the sentence while callers get nothing; empty otherwise.
    callers_line: str = ""

    @property
    def callers_affected(self) -> bool:
        return self.tone == "callers"


def setup_age(last: datetime, now: datetime) -> str:
    days = (now.date() - last.astimezone(now.tzinfo).date()).days
    if days <= 0:
        return "today"
    return "yesterday" if days == 1 else f"{days} days ago"


def _peaks(peak_db: float | None) -> str:
    if peak_db is None or not math.isfinite(peak_db):
        return ""
    return f"peaks under {math.floor(peak_db) + 1} dB"


def _route(inputs: OverviewInputs) -> str:
    if not inputs.output_name:
        return "Choose an output, then press Start"
    return f"{inputs.input_name or 'Default input'} > {inputs.output_name} {DOT} {inputs.preset_name} preset"


def switch_failed_detail(device: str) -> str:
    return (
        f"AudioForge could not start on {device}, so the audio is stopped. "
        "Choose another microphone, or check that this one is connected and press Restart audio."
    )


def _sentence(inputs: OverviewInputs) -> tuple[str, str, str, str, str]:
    """The state key, headline, detail, tone and sidebar line, first match wins."""
    # Running audio follows its own path: a device that went away is a lost device.
    if inputs.no_mic and not inputs.running:
        return "no_mic", "No microphone found", NO_MIC_DETAIL, "warn", f"Stopped {DOT} no microphone"
    if inputs.failed_switch_device and not inputs.running and not inputs.restart_pending:
        detail = switch_failed_detail(inputs.failed_switch_device)
        return "device_lost", "Your callers can't hear you", detail, "callers", f"Stopped {DOT} microphone failed"
    if inputs.restart_pending or (inputs.running and inputs.device_bad):
        detail = inputs.device_advice if inputs.device_bad and inputs.device_advice else RETRYING_DETAIL
        return "device_lost", "Your callers can't hear you", detail, "callers", f"Live {DOT} device lost"
    if not inputs.running:
        return "ready", "Ready when you are", _route(inputs), "idle", f"Ready {DOT} not running"
    # Callers hear silence while muted, so a clipping microphone is not what they get.
    if inputs.muted:
        return "muted", f"Muted {DOT} others hear silence", MUTED_DETAIL, "mute", f"Live {DOT} muted"
    if inputs.input_bad:
        detail = " ".join(part for part in (inputs.input_advice, CLIPPING_NOTE) if part)
        return "clipping", f"Live {DOT} your mic is clipping", detail, "bad", f"Live {DOT} clipping"
    pending = MUTE_PENDING_ISSUE if inputs.mute_requested else UNMUTE_PENDING_ISSUE
    issues = ((pending,) if inputs.mute_error else ()) + inputs.issues
    if issues:
        count = len(issues)
        line = f"Live {DOT} {count} issue{'s' if count != 1 else ''}"
        return "live_issue", line, issues[0], "warn", line
    mode = inputs.mode if inputs.mode in LIVE_HEADLINES else "normal"
    peaks = _peaks(inputs.output_peak_db)
    if mode == "normal":
        parts = (inputs.backend or "Noise suppression off", peaks)
    else:
        parts = (peaks[:1].upper() + peaks[1:],)
    detail = f" {DOT} ".join(part for part in parts if part) or _route(inputs)
    return "live", LIVE_HEADLINES[mode], detail, "ok", LIVE_NAV_LINES[mode]


def _checks(state: str, inputs: OverviewInputs) -> tuple[str, tuple[tuple[str, str], ...]]:
    if state == "no_mic":
        output = (
            (f"Output found {DOT} {inputs.output_name}", "ok") if inputs.output_name
            else ("No output devices detected", "warn") if not inputs.output_found
            else ("Output found, none chosen yet", "warn")
        )
        return "What AudioForge checked", (
            (f"No input devices detected {DOT} the Input list is empty", "warn"),
            output,
            (f"Make sure your microphone is connected, and check it in {SOUND_SETTINGS}", "note"),
        )
    if state != "ready":
        return "", ()
    lines = [(f"Microphone found {DOT} {inputs.input_name or 'Default input'}", "ok")]
    if inputs.output_name:
        lines.append((f"Output set to {inputs.output_name}", "ok"))
        if inputs.output_virtual:
            lines.append((
                f"In Discord, OBS, Zoom or Teams, choose the other end of {inputs.output_name} "
                "as your microphone", "note",
            ))
    else:
        lines.append(("No output chosen yet", "warn"))
    title = "Route check" if any(kind == "warn" for _text, kind in lines) else f"Route check {DOT} All set"
    return title, tuple(lines)


def describe_overview(inputs: OverviewInputs) -> OverviewView:
    state, headline, detail, tone, nav_line = _sentence(inputs)
    if inputs.last_setup is None:
        setup_line = RECOMMEND_SETUP_HINT
    else:
        setup_line = f"Last Auto Voice Setup: {setup_age(inputs.last_setup, inputs.now)}"
    checks_title, checks = _checks(state, inputs)
    return OverviewView(
        state=state,
        headline=headline,
        detail=detail,
        tone=tone,
        nav_line=nav_line,
        ribbon_caption=f"Level {DOT} last 10 seconds" if inputs.running else f"Level {DOT} waiting for audio",
        recommend_setup=inputs.last_setup is None,
        setup_line=setup_line,
        checks_title=checks_title,
        checks=checks,
        offer_restart=state == "device_lost" and not inputs.restart_pending,
        offer_choose_microphone=state == "device_lost",
        callers_line=CALLERS_LINE if tone == "callers" else "",
    )
