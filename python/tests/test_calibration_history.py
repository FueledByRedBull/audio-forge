"""Persist evidence only for the captured route and accepted processing settings."""

from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mic_eq.config import AppConfig, Preset
from mic_eq.config_parts.calibration import (
    CalibrationResult,
    calibration_settings_parts,
    parse_calibration_results,
)
from mic_eq.ui import calibration_history
from mic_eq.ui.main_window import MainWindow


class _CalibrationPanel:
    def __init__(self, reliability: float) -> None:
        self.reliability = reliability
        self.set_calls: list[dict[str, float]] = []

    def get_compressor_settings(self, *, include_calibration: bool) -> dict[str, float]:
        assert include_calibration
        return {"noise_reference_reliability": self.reliability}

    def set_compressor_settings(self, settings: dict[str, float]) -> None:
        self.set_calls.append(settings)
        self.reliability = float(settings["noise_reference_reliability"])


def _calibration_owner(result: CalibrationResult, *, context: str = "route"):
    owner = SimpleNamespace(
        config=AppConfig(calibration_results=[result]),
        _calibration_context_key=lambda: context,
        _get_current_preset=Preset,
        _preset_payload=MainWindow._preset_payload,
        calibration_status_label=SimpleNamespace(setText=lambda _text: None),
        compressor_panel=_CalibrationPanel(result.noise_reference_reliability),
    )
    return owner


def test_saved_evidence_roundtrip_invalidation_and_privacy():
    settings = {"eq": {"enabled": True}, "compressor": {"threshold_db": -18.0}}
    result = CalibrationResult.capture(
        "full_voice_setup",
        "private endpoint + format",
        settings,
        "broadcast",
        ("eq", "compressor", "limiter"),
        0.8,
    )
    config = AppConfig(calibration_results=[result])
    loaded = AppConfig.from_dict(config.to_dict())
    assert loaded.calibration_results == [result]
    assert result.matches("private endpoint + format", settings)
    assert not result.matches("replacement microphone", settings)
    changed = deepcopy(settings)
    changed["compressor"]["threshold_db"] = -20.0
    assert not result.matches("private endpoint + format", changed)
    assert not result.matches(None, settings)
    assert "private endpoint" not in str(config.to_dict()["calibration_results"])
    assert "audio" not in result.to_dict()
    assert len(parse_calibration_results([result.to_dict()] * 30)) == 16


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", []),
        ("version", 99),
        ("context_hash", "bad"),
        ("verified_stages", [[]]),
        ("noise_reference_reliability", float("nan")),
        ("noise_reference_reliability", True),
        ("created_at", "2026-09-12"),
    ],
)
def test_malformed_evidence_is_never_reused(field, value):
    result = CalibrationResult.capture(
        "eq_only", "context", {}, "flat", ("eq",)
    ).to_dict()
    result[field] = value
    assert parse_calibration_results([result]) == []


def test_failed_save_restores_previous_record_and_retry_replaces_same_context(
    monkeypatch,
):
    owner = SimpleNamespace(
        config=AppConfig(),
        _calibration_context_key=lambda: "route/48000",
        _get_current_preset=Preset,
        _preset_payload=MainWindow._preset_payload,
    )
    monkeypatch.setattr(calibration_history, "save_config", lambda config: False)
    assert not calibration_history.save_calibration_result(
        owner, "eq_only", "flat", ("eq",)
    )
    assert owner.config.calibration_results == []
    monkeypatch.setattr(calibration_history, "save_config", lambda config: True)
    assert calibration_history.save_calibration_result(
        owner, "eq_only", "flat", ("eq",)
    )
    assert calibration_history.save_calibration_result(
        owner, "eq_only", "broadcast", ("eq",)
    )
    assert len(owner.config.calibration_results) == 1
    current = calibration_history.current_calibration(owner, "eq_only")
    assert current is not None and current.target_curve == "broadcast"
    owner._calibration_context_key = lambda: "route/44100"
    assert calibration_history.current_calibration(owner, "eq_only") is None
    assert "stale" in calibration_history.calibration_summary(owner)


def test_layered_calibration_preserves_correction_when_tone_changes():
    settings = {
        "eq": {
            "schema_version": 3,
            "enabled": True,
            "layers": {
                "correction": [{"frequency_hz": 100.0, "gain_db": -1.0}],
                "tone": [{"frequency_hz": 100.0, "gain_db": 0.0}],
            },
        },
        "gate": {"gate_mode": 2},
        "rnnoise": {"model": "rnnoise"},
    }
    correction, verification = calibration_settings_parts(settings)
    result = CalibrationResult.capture(
        "full_voice_setup",
        "route/48000",
        settings,
        "broadcast",
        ("eq", "compressor", "limiter"),
        0.8,
        correction_settings=correction,
        verification_settings=verification,
    )

    changed = deepcopy(settings)
    changed["eq"]["layers"]["tone"][0]["gain_db"] = 2.0
    changed_correction, changed_verification = calibration_settings_parts(changed)

    assert result.matches_capture_and_correction("route/48000", changed_correction)
    assert not result.matches_verification(changed_verification)


def test_new_calibration_does_not_invalidate_on_marketing_version_change():
    settings = {"eq": {"enabled": True}}
    result = CalibrationResult.capture(
        "eq_only", "route", settings, "flat", ("eq",)
    )
    payload = result.to_dict()
    payload["app_version"] = "future-marketing-version"
    parsed = CalibrationResult.from_value(payload)

    assert parsed is not None
    assert parsed.matches("route", settings)


def test_legacy_calibration_record_remains_conservative():
    settings = {"eq": {"enabled": True}}
    payload = CalibrationResult.capture(
        "eq_only", "route", settings, "flat", ("eq",)
    ).to_dict()
    payload["version"] = 1
    for field in ("correction_hash", "verification_hash", "compatibility"):
        payload.pop(field, None)

    parsed = CalibrationResult.from_value(payload)

    assert parsed is not None and parsed.version == 1
    assert parsed.matches("route", settings)
    payload["app_version"] = "future-marketing-version"
    future = CalibrationResult.from_value(payload)
    assert future is not None
    assert not future.matches("route", settings)

    full_settings = json.loads(MainWindow._preset_payload(Preset()))
    full_payload = CalibrationResult.capture(
        "full_voice_setup", "route", full_settings, "broadcast", ("eq",)
    ).to_dict()
    full_payload["version"] = 1
    for field in ("correction_hash", "verification_hash", "compatibility"):
        full_payload.pop(field, None)
    legacy_full = CalibrationResult.from_value(full_payload)
    assert legacy_full is not None
    owner = SimpleNamespace(
        config=AppConfig(calibration_results=[legacy_full]),
        _calibration_context_key=lambda: "route",
        _get_current_preset=Preset,
        _preset_payload=MainWindow._preset_payload,
    )
    assert (
        calibration_history.current_calibration(cast(Any, owner), "full_voice_setup")
        is legacy_full
    )


def test_status_refresh_is_read_only_for_calibration_dsp():
    settings = json.loads(MainWindow._preset_payload(Preset()))
    correction, verification = calibration_settings_parts(settings)
    result = CalibrationResult.capture(
        "full_voice_setup",
        "route",
        settings,
        "broadcast",
        ("eq", "compressor", "limiter"),
        0.8,
        correction_settings=correction,
        verification_settings=verification,
    )
    owner = _calibration_owner(result)

    MainWindow._refresh_calibration_status(cast(Any, owner))

    assert owner.compressor_panel.set_calls == []
    assert owner.compressor_panel.reliability == pytest.approx(0.8)


def test_explicit_calibration_sync_clears_noise_reference_after_route_change():
    settings = json.loads(MainWindow._preset_payload(Preset()))
    correction, verification = calibration_settings_parts(settings)
    result = CalibrationResult.capture(
        "full_voice_setup",
        "route-a",
        settings,
        "broadcast",
        ("eq", "compressor", "limiter"),
        0.8,
        correction_settings=correction,
        verification_settings=verification,
    )
    context = {"value": "route-a"}
    owner = _calibration_owner(result)
    owner._calibration_context_key = lambda: context["value"]

    MainWindow._sync_calibration_evidence(cast(Any, owner))
    assert owner.compressor_panel.set_calls == []

    context["value"] = "route-b"
    MainWindow._sync_calibration_evidence(cast(Any, owner))

    assert owner.compressor_panel.set_calls == [
        {"noise_reference_reliability": 0.0}
    ]
