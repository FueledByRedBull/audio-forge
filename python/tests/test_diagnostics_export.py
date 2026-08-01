"""Privacy, schema, and UI tests for support-snapshot export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mic_eq.config import DeviceIdentity, Preset
from mic_eq.diagnostics_export import (
    MAX_SERIALIZED_BYTES,
    build_diagnostics_snapshot,
    diagnostics_filename,
    serialize_diagnostics_snapshot,
    write_diagnostics_snapshot,
)
from mic_eq.ui import main_window


FIXED_TIME = datetime(2026, 7, 31, 12, 34, 56, tzinfo=timezone.utc)
FIXED_SYSTEM = {
    "operating_system": "Windows",
    "os_release": "11",
    "os_version": "10.0.26100",
    "architecture": "AMD64",
    "python_version": "3.12.10",
    "python_implementation": "CPython",
}


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        input_channel_mode="average",
        input_cleanup_mode="gentle",
        use_measured_latency=True,
        main_control_tab_index=2,
        voice_setup_dynamics_intensity="balanced",
        voice_setup_custom_p95_db=3.5,
        voice_setup_custom_peak_cap_db=8.0,
        latency_calibration_profiles={
            "C:\\Users\\private\\device-route": object(),
        },
        last_input_device="Secret Microphone",
        last_output_device="Secret Output",
        last_preset="C:\\Users\\private\\preset.json",
    )


def _runtime() -> dict[str, object]:
    return {
        "input_dropped_samples": 2,
        "output_true_peak_db": -2.5,
        "output_short_term_lufs": float("nan"),
        "noise_backend_available": True,
        "noise_model": "deepfilter-ll",
        "noise_backend_error": "C:\\Users\\private\\model.onnx failed",
        "last_stream_error": "token=secret-value",
        "last_restart_reason": "/home/private/device changed",
        "unknown_path": "C:\\Users\\private\\raw.wav",
        "raw_audio": [0.1, 0.2],
        "API_TOKEN": "secret-value",
    }


def _build(*, key: bytes = b"K" * 32) -> dict[str, Any]:
    return build_diagnostics_snapshot(
        app_version="1.10.1",
        runtime_diagnostics=_runtime(),
        config=_config(),
        processing_settings=Preset().to_dict(),
        input_device=DeviceIdentity(
            name="Private USB Microphone",
            is_default=True,
        ),
        output_device=DeviceIdentity(name="Private Virtual Cable"),
        processing_sample_rate_hz=48_000,
        output_sample_rate_hz=48_000,
        running=True,
        generated_at=FIXED_TIME,
        pseudonym_key=key,
        system_info=FIXED_SYSTEM,
    )


def test_snapshot_is_deterministic_finite_bounded_and_private() -> None:
    first = _build()
    second = _build()
    payload = serialize_diagnostics_snapshot(first)
    text = payload.decode("utf-8")

    assert first == second
    assert len(payload) < MAX_SERIALIZED_BYTES
    assert first["runtime"]["output_short_term_lufs"] is None
    assert first["runtime"]["backend_error_present"] is True
    assert first["runtime"]["stream_error_present"] is True
    assert first["runtime"]["restart_reason_present"] is True
    assert first["configuration"]["saved_latency_profile_count"] == 1
    assert first["configuration"]["device_preset_binding_count"] == 0
    assert first["processing"]["eq"]["schema_version"] == 2
    assert len(first["processing"]["eq"]["bands"]) == 10
    assert "Private USB Microphone" not in text
    assert "Private Virtual Cable" not in text
    assert "C:\\Users" not in text
    assert "/home/private" not in text
    assert "secret-value" not in text
    assert '"raw_audio":' not in text
    assert '"API_TOKEN":' not in text
    assert '"unknown_path":' not in text
    assert b"NaN" not in payload
    assert b"Infinity" not in payload


def test_report_local_key_changes_device_pseudonyms() -> None:
    first = _build(key=b"A" * 32)
    second = _build(key=b"B" * 32)

    assert (
        first["audio_engine"]["input_device"]["pseudonym"]
        != second["audio_engine"]["input_device"]["pseudonym"]
    )


def test_same_named_endpoint_devices_have_distinct_private_pseudonyms() -> None:
    first = build_diagnostics_snapshot(
        app_version="1.11.0",
        runtime_diagnostics=_runtime(),
        config=_config(),
        processing_settings=Preset().to_dict(),
        input_device=DeviceIdentity(
            name="USB Microphone",
            endpoint_id="endpoint-a",
            direction="input",
        ),
        output_device=None,
        processing_sample_rate_hz=48_000,
        output_sample_rate_hz=48_000,
        running=False,
        generated_at=FIXED_TIME,
        pseudonym_key=b"K" * 32,
        system_info=FIXED_SYSTEM,
    )
    second = build_diagnostics_snapshot(
        app_version="1.11.0",
        runtime_diagnostics=_runtime(),
        config=_config(),
        processing_settings=Preset().to_dict(),
        input_device=DeviceIdentity(
            name="USB Microphone",
            endpoint_id="endpoint-b",
            direction="input",
        ),
        output_device=None,
        processing_sample_rate_hz=48_000,
        output_sample_rate_hz=48_000,
        running=False,
        generated_at=FIXED_TIME,
        pseudonym_key=b"K" * 32,
        system_info=FIXED_SYSTEM,
    )

    first_device = first["audio_engine"]["input_device"]
    second_device = second["audio_engine"]["input_device"]
    assert first_device["pseudonym"] != second_device["pseudonym"]
    serialized = serialize_diagnostics_snapshot(first).decode("utf-8")
    assert "endpoint-a" not in serialized
    assert "USB Microphone" not in serialized


def test_short_key_naive_timestamp_and_unexpected_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        _build(key=b"short")
    with pytest.raises(ValueError, match="timezone-aware"):
        diagnostics_filename("1.10.1", datetime(2026, 7, 31))

    snapshot = _build()
    snapshot["raw_path"] = "C:\\Users\\private"
    with pytest.raises(ValueError, match="unexpected root fields"):
        serialize_diagnostics_snapshot(snapshot)


def test_snapshot_writer_is_round_trip_json_and_leaves_no_temp_file(
    tmp_path,
) -> None:
    destination = tmp_path / "support.json"
    snapshot = _build()

    write_diagnostics_snapshot(destination, snapshot)

    assert json.loads(destination.read_text(encoding="utf-8")) == snapshot
    assert list(tmp_path.glob("*.tmp")) == []


class _FakeProcessor:
    def get_runtime_diagnostics(self) -> dict[str, object]:
        return _runtime()

    def sample_rate(self) -> int:
        return 48_000

    def output_sample_rate(self) -> int:
        return 48_000

    def is_running(self) -> bool:
        return True


class _FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int) -> None:
        self.messages.append((text, timeout))


class _FakeWindow:
    def __init__(self) -> None:
        self.processor = _FakeProcessor()
        self.config = _config()
        self.input_combo = object()
        self.output_combo = object()
        self.status_bar = _FakeStatusBar()

    def _combo_device_identity(self, combo: object) -> DeviceIdentity:
        if combo is self.input_combo:
            return DeviceIdentity(name="Private USB Microphone", is_default=True)
        return DeviceIdentity(name="Private Virtual Cable")

    def _get_current_preset(self) -> Preset:
        return Preset()


def test_main_window_export_uses_privacy_safe_builder(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "ui-support.json"
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "JSON Files (*.json)"),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "information",
        lambda _parent, title, text: messages.append((title, text)),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "critical",
        lambda *_args, **_kwargs: pytest.fail("export unexpectedly failed"),
    )
    window = _FakeWindow()

    main_window.MainWindow._export_diagnostics(cast(Any, window))

    payload = destination.read_text(encoding="utf-8")
    assert "Private USB Microphone" not in payload
    assert "Private Virtual Cable" not in payload
    assert json.loads(payload)["privacy"]["raw_audio_included"] is False
    assert messages[0][0] == "Diagnostics Exported"
    assert window.status_bar.messages[-1][0] == "Privacy-safe diagnostics exported"
