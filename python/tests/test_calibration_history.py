"""Persist evidence only for the captured route and accepted processing settings."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mic_eq.config import AppConfig, Preset
from mic_eq.config_parts.calibration import CalibrationResult, parse_calibration_results
from mic_eq.ui import calibration_history
from mic_eq.ui.main_window import MainWindow


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
        ("version", 2),
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
