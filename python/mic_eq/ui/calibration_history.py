"""Save calibration evidence, never recordings or reusable stale confidence."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from ..config import save_config
from ..config_parts.calibration import (
    CalibrationResult,
    calibration_settings_parts,
)


def calibration_settings(owner: Any) -> dict:
    return json.loads(owner._preset_payload(owner._get_current_preset()))


def calibration_inputs(
    owner: Any,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """Return capture, correction, and downstream verification identities.

    MainWindow can use this read-only projection when a configuration or route
    command changes. The status view should describe these identities, never
    mutate DSP state while rendering them.
    """
    settings = calibration_settings(owner)
    correction, verification = calibration_settings_parts(settings)
    return owner._calibration_context_key(), correction, verification


def current_calibration(owner: Any, scope: str) -> CalibrationResult | None:
    settings = calibration_settings(owner)
    correction, _verification = calibration_settings_parts(settings)
    context = owner._calibration_context_key()
    for result in reversed(owner.config.calibration_results):
        if result.scope != scope:
            continue
        # Version 1 stored the complete preset identity. Keep matching it
        # against that same canonical payload while newer records can split
        # correction from downstream verification.
        if (
            result.matches(context, settings)
            if result.version == 1
            else result.matches_capture_and_correction(context, correction)
        ):
            return result
    return None


def calibration_verification_matches(owner: Any, result: CalibrationResult) -> bool:
    """Whether a matching capture/correction record still verifies output."""
    settings = calibration_settings(owner)
    _correction, verification = calibration_settings_parts(settings)
    return result.matches_verification(
        settings if result.version == 1 else verification
    )


def save_calibration_result(
    owner: Any, scope: str, curve: str, stages: tuple[str, ...]
) -> bool:
    context = owner._calibration_context_key()
    if not context:
        raise ValueError("Audio device and capture format are unavailable")
    reliability = (
        (
            owner.compressor_panel.get_compressor_settings(
                include_calibration=True
            ).get("noise_reference_reliability", 0.0)
        )
        if scope == "full_voice_setup"
        else 0.0
    )
    settings = calibration_settings(owner)
    correction, verification = calibration_settings_parts(settings)
    result = CalibrationResult.capture(
        scope,
        context,
        settings,
        curve,
        stages,
        reliability,
        correction_settings=correction,
        verification_settings=verification,
    )
    previous = list(owner.config.calibration_results)
    owner.config.calibration_results = [
        entry
        for entry in previous
        if (entry.scope, entry.context_hash) != (result.scope, result.context_hash)
    ][-15:] + [result]
    try:
        saved = save_config(owner.config)
    except Exception:
        owner.config.calibration_results = previous
        raise
    if not saved:
        owner.config.calibration_results = previous
    return saved


def calibration_summary(owner: Any) -> str:
    results = owner.config.calibration_results
    if not results:
        return "No saved calibration"
    for scope, label in (("full_voice_setup", "Voice setup"), ("eq_only", "Auto-EQ")):
        if (result := current_calibration(owner, scope)) is not None:
            if (
                scope == "full_voice_setup"
                and not calibration_verification_matches(owner, result)
            ):
                return (
                    f"{label}: microphone/correction match; output verification "
                    f"outdated ({result.created_at[:10]})"
                )
            return f"{label}: matching devices/settings ({result.created_at[:10]})"
    return "Saved calibration is stale: devices or settings differ; recalibrate"


def persist_calibration(
    dialog: Any, owner: Any, scope: str, curve: str, stages: tuple[str, ...]
) -> None:
    # Standalone dialogs without application persistence still support calibration.
    if not hasattr(owner, "config") or not hasattr(owner, "_get_current_preset"):
        return
    while True:
        try:
            if save_calibration_result(owner, scope, curve, stages):
                sync = getattr(owner, "_sync_calibration_evidence", None)
                if callable(sync):
                    sync()
                owner._update_session_summary()
                return
            reason = "Settings storage refused the write."
        except (OSError, ValueError, TypeError) as error:
            reason = str(error)
        if (
            QMessageBox.warning(
                dialog,
                "Calibration applied, but not saved",
                "Your processing settings are applied for this session, but the calibration "
                f"record was not saved. {reason}",
                QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Close,
                QMessageBox.StandardButton.Retry,
            )
            != QMessageBox.StandardButton.Retry
        ):
            return
