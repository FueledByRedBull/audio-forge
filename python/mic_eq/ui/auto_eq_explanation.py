"""User-facing explanations for existing Auto-EQ diagnostic decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        "Held-out validation reduced correction strength to avoid overfitting.",
    ),
    "room-noise reference is questionable": (
        "invalid_noise_reference",
        "The room-noise reference was usable only with conservative limits.",
    ),
}


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
    mapped_codes: list[str] = []
    details: list[str] = []
    for reason in raw_reasons:
        mapped = _REASON_MESSAGES.get(reason)
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
            or ("AudioForge reduced the correction using existing safety gates.",),
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
            details=("All active bands passed the existing reliability gates.",),
            state="ok",
        )
    return AutoEqExplanation(
        outcome_code="unknown",
        summary="Calibration result unavailable",
        details=("The diagnostic decision status was missing or unsupported.",),
        state="idle",
    )


__all__ = ["AutoEqExplanation", "explain_auto_eq"]
