"""Compact health decision helpers for runtime metering."""

from __future__ import annotations

from typing import Any


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


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

    true_peak_headroom = _float_or_none(true_peak_headroom_db)
    if clip_delta:
        return f"Output: CLIP (OCL:{output_clip_count})", "bad"
    if limiter_history_db >= 6.0 or true_peak_limiter_history_db >= 3.0:
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
