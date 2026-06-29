"""Stream recovery decision state for the AudioForge UI."""

from __future__ import annotations

import time
from dataclasses import dataclass


def update_callback_stall_state(
    stall_started_at: float | None,
    now: float,
    input_cb_age_ms: int,
    output_cb_age_ms: int,
    processing_started_at: float | None,
    last_recovery_at: float,
    calibration_dialog_open: bool,
    warmup_s: float = 5.0,
    cooldown_s: float = 20.0,
    grace_s: float = 1.5,
    output_age_threshold_ms: int = 2000,
    input_age_threshold_ms: int = 1500,
) -> tuple[float | None, bool]:
    """Return the next callback-stall state and whether recovery should run."""
    if calibration_dialog_open:
        return None, False

    if processing_started_at is None:
        return None, False

    if now - processing_started_at < warmup_s:
        return None, False

    if now - last_recovery_at < cooldown_s:
        return stall_started_at, False

    suspicious = output_cb_age_ms > output_age_threshold_ms and input_cb_age_ms < input_age_threshold_ms
    if not suspicious:
        return None, False

    if stall_started_at is None:
        return now, False

    if now - stall_started_at < grace_s:
        return stall_started_at, False

    return None, True


@dataclass(slots=True)
class StreamRecoveryManager:
    """Track UI-side recovery heuristics without touching Qt widgets."""

    output_stall_started_at: float | None = None
    output_callback_stall_started_at: float | None = None
    last_output_recovery_at: float = 0.0
    processing_started_at: float | None = None

    def mark_processing_started(self, now: float | None = None) -> None:
        self.processing_started_at = time.monotonic() if now is None else now
        self.output_stall_started_at = None
        self.output_callback_stall_started_at = None

    def mark_processing_stopped(self) -> None:
        self.processing_started_at = None
        self.output_stall_started_at = None
        self.output_callback_stall_started_at = None

    def maybe_recover_output_stall(
        self,
        *,
        input_rms: float,
        output_rms: float,
        output_buf: int,
        calibration_dialog_open: bool,
        now: float | None = None,
        cooldown_s: float = 20.0,
        grace_s: float = 1.5,
    ) -> bool:
        """Return True when output-stall recovery should run."""
        if calibration_dialog_open:
            self.output_stall_started_at = None
            return False

        current_time = time.monotonic() if now is None else now
        if current_time - self.last_output_recovery_at < cooldown_s:
            return False

        suspicious = input_rms > -50.0 and output_rms < -85.0 and output_buf > 20000
        if not suspicious:
            self.output_stall_started_at = None
            return False

        if self.output_stall_started_at is None:
            self.output_stall_started_at = current_time
            return False

        if current_time - self.output_stall_started_at < grace_s:
            return False

        self.output_stall_started_at = None
        self.last_output_recovery_at = current_time
        return True

    def maybe_recover_callback_stall(
        self,
        *,
        input_cb_age_ms: int,
        output_cb_age_ms: int,
        calibration_dialog_open: bool,
        now: float | None = None,
    ) -> bool:
        """Return True when callback-stall recovery should run."""
        current_time = time.monotonic() if now is None else now
        new_state, should_recover = update_callback_stall_state(
            stall_started_at=self.output_callback_stall_started_at,
            now=current_time,
            input_cb_age_ms=input_cb_age_ms,
            output_cb_age_ms=output_cb_age_ms,
            processing_started_at=self.processing_started_at,
            last_recovery_at=self.last_output_recovery_at,
            calibration_dialog_open=calibration_dialog_open,
        )
        self.output_callback_stall_started_at = new_state
        if should_recover:
            self.last_output_recovery_at = current_time
        return should_recover


__all__ = ["StreamRecoveryManager", "update_callback_stall_state"]
