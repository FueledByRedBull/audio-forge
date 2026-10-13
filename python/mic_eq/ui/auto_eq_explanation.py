"""User-facing explanations for existing Auto-EQ diagnostic decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..analysis.auto_eq_parts.headroom import headroom_shortfall_db


_REMOVED_REASON = "headroom validation removed the usable correction"
_REDUCED_REASON = "headroom validation reduced the fitted correction"
_DECLINED_REASON = "no safe candidate remained after full-chain headroom validation"
# A correction kept because it is no worse than the current EQ, while that EQ is over the rule.
_KEPT_REDUCED_MESSAGE = "The correction was made gentler so it leaves no less room than your current EQ."

_REASON_MESSAGES = {
    "insufficient repeatable voiced windows": (
        "insufficient_coverage",
        "Not enough repeatable speech was captured to support correction.",
    ),
    "capture quality score is too low": (
        "unusable_capture",
        "The speech capture was too inconsistent or unclear to trust.",
    ),
    "noise-referenced SNR is too low": (
        "unusable_capture",
        "Speech was too close to the measured room-noise level.",
    ),
    "room-noise reference is invalid": (
        "invalid_noise_reference",
        "The room-noise reference did not match a usable quiet capture.",
    ),
    "overall confidence is below full-strength threshold": (
        "conservative_success",
        "Measurement support was adequate, so a gentler correction was kept.",
    ),
    "validation reduced the fitted correction": (
        "conservative_success",
        "The full correction was not safe or did not help enough, so a gentler one is offered.",
    ),
    "room-noise reference is questionable": (
        "invalid_noise_reference",
        "The room-noise reference was usable only with conservative limits.",
    ),
    _REDUCED_REASON: (
        "conservative_success",
        "The correction was made gentler to leave enough headroom.",
    ),
    _REMOVED_REASON: (
        "headroom_limited",
        "Not enough headroom was left for a useful correction.",
    ),
    # The correction was tried at every strength and each left less room than the current EQ.
    _DECLINED_REASON: (
        "headroom_limited",
        "The correction would leave less room than your current EQ does.",
    ),
    # Each strength of the fitted correction, from full down to a quarter, was tried against
    # the tone preset; none both came closer to it and passed the EQ safety checks.
    "no validated correction improved the target safely": (
        "no_safe_improvement",
        "No correction that passes the safety checks brings your voice closer to the tone preset.",
    ),
    # The bounded fit failed to settle and left every band under 0.25 dB.
    "constrained gain solve produced no safe correction": (
        "fit_failed",
        "The tone fit could not settle on a correction that stays within the safety limits.",
    ),
}

# The calibration dialog appends the specific cause after this prefix.
_HEADROOM_FAILURE_PREFIX = "Full-chain headroom validation failed"
_HEADROOM_FAILURE_MESSAGE = (
    "headroom_unverified",
    "AudioForge could not confirm the correction is safe for the output level.",
)


# What the user reads for each stored ``target_mode`` value; the stored values stay as they are.
TARGET_MODE_LABELS = {"adaptive": "Subtle", "static": "Full strength"}


def target_profile_text(profile: object) -> str:
    """Show a stored target profile such as ``broadcast:adaptive`` with the strength's plain name."""
    if not isinstance(profile, str):
        return "--"
    preset, _, rest = profile.partition(":")
    label = TARGET_MODE_LABELS.get(rest.partition(":")[0])
    return f"{preset}, {label}" if label else profile


@dataclass(frozen=True, slots=True)
class AutoEqExplanation:
    outcome_code: str
    summary: str
    details: tuple[str, ...]
    state: str


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def explain_auto_eq(diagnostics: dict[str, Any] | None) -> AutoEqExplanation:
    """Map diagnostics to text without recomputing or changing a DSP decision."""
    if not diagnostics:
        return AutoEqExplanation(
            outcome_code="not_run",
            summary="No calibration result",
            details=(),
            state="idle",
        )

    status = diagnostics.get("recommendation_status")
    raw_reasons = (
        _string_list(diagnostics.get("abstention_reasons"))
        + _string_list(diagnostics.get("recommendation_reasons"))
    )
    headroom = diagnostics.get("headroom_validation")
    kept_over_rule = isinstance(headroom, Mapping) and headroom.get("status") == "risk_not_worsened"
    if _DECLINED_REASON in raw_reasons:  # it says the same, with the cause
        raw_reasons = tuple(reason for reason in raw_reasons if reason != _REMOVED_REASON)
    mapped_codes: list[str] = []
    details: list[str] = []
    for reason in raw_reasons:
        mapped = _REASON_MESSAGES.get(reason)
        if reason == _REDUCED_REASON and kept_over_rule:
            mapped = ("conservative_success", _KEPT_REDUCED_MESSAGE)
        if mapped is None and reason.startswith(_HEADROOM_FAILURE_PREFIX):
            mapped = _HEADROOM_FAILURE_MESSAGE
        if mapped is None:
            continue
        code, message = mapped
        if code not in mapped_codes:
            mapped_codes.append(code)
        if message not in details:
            details.append(message)

    low_band_count = diagnostics.get("low_confidence_active_bands", 0)
    if not isinstance(low_band_count, int) or isinstance(low_band_count, bool):
        low_band_count = 0
    if low_band_count > 0:
        mapped_codes.append("low_band_reliability")
        details.append(
            f"{low_band_count} unsupported frequency "
            f"{'band was' if low_band_count == 1 else 'bands were'} "
            "left unchanged."
        )

    if status == "abstain":
        primary = mapped_codes[0] if mapped_codes else "unusable_capture"
        return AutoEqExplanation(
            outcome_code=primary,
            summary="No correction applied",
            details=tuple(details)
            or ("The capture did not support a safe correction.",),
            state="bad",
        )
    if status == "reduced":
        return AutoEqExplanation(
            outcome_code=(
                mapped_codes[0]
                if mapped_codes
                else "conservative_success"
            ),
            summary="Conservative correction ready",
            details=tuple(details)
            or ("AudioForge made the correction gentler to keep it safe.",),
            state="warn",
        )
    if status == "apply":
        if low_band_count > 0:
            return AutoEqExplanation(
                outcome_code="low_band_reliability",
                summary="Correction ready with unsupported bands skipped",
                details=tuple(details),
                state="ok",
            )
        return AutoEqExplanation(
            outcome_code="correction_ready",
            summary="Correction ready",
            details=("Every band that changed is well supported by your recording.",),
            state="ok",
        )
    return AutoEqExplanation(
        outcome_code="unknown",
        summary="Calibration result unavailable",
        details=("The result did not say whether a correction is safe.",),
        state="idle",
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def headroom_advice(
    headroom: Mapping[str, Any] | None, chain: Mapping[str, Any] | None, proposed: bool
) -> str:
    """Say what to change when headroom cut the result short; empty when it did not.

    The whole-decibel figure is what lets the full correction meet the rule. It assumes
    level in is level out, so it is given only when the compressor is off.
    """
    if not isinstance(headroom, Mapping) or headroom.get("authoritative") is not True:
        return ""
    scale = _number(headroom.get("gain_scale")) or 0.0
    status = headroom.get("status")
    if status not in ("risk_not_worsened", "risk_worse") and not (
        status == "safe" and scale < 0.70
    ):
        return ""

    parts: list[str] = []
    incumbent = _number(headroom.get("incumbent_excess_db"))
    over_rule = incumbent is not None and incumbent > 0.0  # the current EQ itself is over the rule
    if over_rule:
        peak = _number(headroom.get("incumbent_output_true_peak_db"))
        parts.append(
            "Your output already clips on loud peaks."
            if peak is not None and peak > 0.0
            else "Your current settings already leave too little room for loud peaks."
        )
    if proposed and status == "risk_not_worsened" and scale >= 0.70:
        parts.append("This EQ is no worse than your current one.")
    compressor = (chain or {}).get("compressor") or {}
    makeup = bool(compressor.get("enabled") and compressor.get("auto_makeup_enabled"))
    shortfall = headroom_shortfall_db(headroom) if compressor.get("enabled") is False else None
    if proposed:
        purpose = " to allow the full correction" if scale < 1.0 else " to leave enough room"
    else:
        purpose = ", then record again"
    amount = f" by about {shortfall} dB" if shortfall is not None else ""
    parts.append(
        f"Lower the compressor's target loudness{amount}{purpose}. "
        "Automatic makeup would undo a lower microphone gain."
        if makeup
        else f"Lower the microphone gain{amount} (in Windows or on the microphone){purpose}."
    )
    if over_rule and ((chain or {}).get("limiter") or {}).get("enabled") is False:
        parts.append(
            "Turning the limiter on would catch the peaks, "
            "but it would not make this check pass."
        )
    return " ".join(parts)


__all__ = [
    "AutoEqExplanation",
    "TARGET_MODE_LABELS",
    "explain_auto_eq",
    "headroom_advice",
    "target_profile_text",
]
