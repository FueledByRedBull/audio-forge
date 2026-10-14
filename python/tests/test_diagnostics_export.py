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
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_diagnostics_snapshot,
    diagnostics_filename,
    sanitize_setup_result,
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
        "output_priming_underrun_total": 3,
        "output_underrun_total": 0,
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


def _build(
    *, key: bytes = b"K" * 32, processing: dict[str, Any] | None = None,
    setup_result: dict[str, Any] | None = None,
    interface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    processing = Preset().to_dict() if processing is None else processing
    processing["eq"]["bands"][0]["gain_db"] = float("inf")
    return build_diagnostics_snapshot(
        app_version="1.10.1",
        runtime_diagnostics=_runtime(),
        config=_config(),
        processing_settings=processing,
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
        setup_result=setup_result,
        interface=interface,
    )


def test_snapshot_is_deterministic_finite_bounded_and_private() -> None:
    first = _build()
    second = _build()
    payload = serialize_diagnostics_snapshot(first)
    text = payload.decode("utf-8")

    assert first == second
    assert first["schema"] == {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}
    assert first["privacy"] == {
        "raw_audio_included": False,
        "environment_variables_included": False,
        "arbitrary_paths_included": False,
        "raw_device_names_included": False,
        "secrets_included": False,
        "pseudonyms_are_report_local": True,
        "setup_message_text_included": False,
    }
    assert first["schema"]["version"] == 3
    assert first["setup_result"] is None
    assert len(payload) < MAX_SERIALIZED_BYTES
    assert first["runtime"]["output_short_term_lufs"] is None
    # Start-up priming is kept apart from dropouts so a start stuck in priming shows.
    assert first["runtime"]["output_priming_underrun_total"] == 3
    assert first["runtime"]["output_underrun_total"] == 0
    assert first["runtime"]["backend_error_present"] is True
    assert first["runtime"]["stream_error_present"] is True
    assert first["runtime"]["restart_reason_present"] is True
    assert first["configuration"]["saved_latency_profile_count"] == 1
    assert first["configuration"]["device_preset_binding_count"] == 0
    assert first["processing"]["eq"]["schema_version"] == 3
    assert len(first["processing"]["eq"]["bands"]) == 10
    assert "gain_db" not in first["processing"]["eq"]["bands"][0]
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


def test_interface_is_exported_as_fixed_words_only() -> None:
    quick = {"view": "quick", "classic_reason": "none", "graphics_api": "direct3d11"}
    assert _build(interface=quick)["interface"] == quick
    classic = {"view": "classic", "classic_reason": "qml_failed", "graphics_api": "not_applicable"}
    assert _build(interface=classic)["interface"] == classic

    # Anything outside the vocabulary, such as an error text or a path, never leaves the app.
    leaky = {
        "view": "C:/Users/private/Main.qml", "classic_reason": "failed: token=secret-value",
        "graphics_api": "Radeon RX private", "extra": "dropped",
    }
    exported = _build(interface=leaky)["interface"]
    assert exported == {"view": "other", "classic_reason": "other", "graphics_api": "other"}
    payload = serialize_diagnostics_snapshot(_build(interface=leaky)).decode("utf-8")
    assert "private" not in payload and "secret-value" not in payload

    snapshot = _build(interface=quick)
    snapshot["interface"]["renderer_name"] = "leak"
    with pytest.raises(ValueError, match="interface"):
        serialize_diagnostics_snapshot(snapshot)


def test_report_local_key_changes_device_pseudonyms() -> None:
    first = _build(key=b"A" * 32)
    second = _build(key=b"B" * 32)

    assert (
        first["audio_engine"]["input_device"]["pseudonym"]
        != second["audio_engine"]["input_device"]["pseudonym"]
    )


def test_snapshot_preserves_private_bounded_eq_layers() -> None:
    processing = Preset().to_dict()
    correction = [dict(band) for band in processing["eq"]["bands"]]
    tone = [dict(band) for band in processing["eq"]["bands"]]
    correction[0].update(gain_db=-3.25, private_path="C:/Users/private/measurement.wav")
    tone[0].update(gain_db=2.75, label="secret-value", q=float("nan"))
    processing["eq"]["layers"] = {
        "correction": correction, "tone": tone, "secret-value": "private-path",
    }
    snapshot = _build(processing=processing)
    layers = snapshot["processing"]["eq"]["layers"]
    assert set(layers) == {"correction", "tone"}
    assert layers["correction"][0]["gain_db"] == -3.25
    assert layers["tone"][0]["gain_db"] == 2.75
    assert "q" not in layers["tone"][0]
    payload = serialize_diagnostics_snapshot(snapshot)
    assert b"private" not in payload and b"secret-value" not in payload
    assert len(payload) < MAX_SERIALIZED_BYTES

    layers["tone"][0]["private_path"] = "must not escape"
    with pytest.raises(ValueError, match="EQ band"):
        serialize_diagnostics_snapshot(snapshot)


def test_snapshot_omits_oversized_or_malformed_eq_layers() -> None:
    processing = Preset().to_dict()
    processing["eq"]["layers"] = {
        "correction": processing["eq"]["bands"] * 4,
        "tone": {"path": "private"},
    }
    snapshot = _build(processing=processing)
    assert snapshot["processing"]["eq"]["layers"] == {}


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

    def get_input_rms_db(self) -> float:
        return -120.0

    def get_input_peak_db(self) -> float:
        return -96.5


class _FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int) -> None:
        self.messages.append((text, timeout))


class _FakeWindow:
    def __init__(self) -> None:
        self.processor = _FakeProcessor()
        self.last_setup_result: Any = None
        self.config = _config()
        self.input_choice = object()
        self.output_choice = object()
        self.status_message = _FakeStatusBar()

    def _combo_device_identity(self, combo: object) -> DeviceIdentity:
        if combo is self.input_choice:
            return DeviceIdentity(name="Private USB Microphone", is_default=True)
        return DeviceIdentity(name="Private Virtual Cable")

    def _get_current_preset(self) -> Preset:
        return Preset()

    def _interface_snapshot(self) -> dict[str, str]:
        return {"view": "quick", "classic_reason": "none", "graphics_api": "software"}


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
    # A microphone that is silent at rest has to be visible in a support report.
    runtime = json.loads(payload)["runtime"]
    assert (runtime["input_rms_db"], runtime["input_peak_db"]) == (-120.0, -96.5)
    assert "Private USB Microphone" not in payload
    assert "Private Virtual Cable" not in payload
    assert json.loads(payload)["privacy"]["raw_audio_included"] is False
    assert messages[0][0] == "Diagnostics Exported"
    # No Voice Setup or Auto-EQ run has happened, so the dialog does not claim a file entry for one.
    assert "last Voice Setup" not in messages[0][1] and "Auto-EQ" not in messages[0][1]
    assert json.loads(payload)["interface"]["view"] == "quick"
    assert window.status_message.messages[-1][0] == "Privacy-safe diagnostics exported"


class _FailingLevelProcessor(_FakeProcessor):
    def get_input_rms_db(self) -> float:
        raise RuntimeError("level meter unavailable")


def test_main_window_export_survives_a_failed_level_read(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "ui-support.json"
    messages: list[str] = []
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "JSON Files (*.json)"),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "information",
        lambda _parent, title, _text: messages.append(title),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "critical",
        lambda *_args, **_kwargs: pytest.fail("one failed read lost the whole report"),
    )
    window = _FakeWindow()
    window.processor = _FailingLevelProcessor()

    main_window.MainWindow._export_diagnostics(cast(Any, window))

    runtime = json.loads(destination.read_text(encoding="utf-8"))["runtime"]
    assert (runtime["input_rms_db"], runtime["input_peak_db"]) == (None, -96.5)
    assert messages == ["Diagnostics Exported"]


def _setup_result(**changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "voice_setup",
        "outcome": "rolled_back",
        "reason_codes": ["tonal_check_failed", "loudness_target_missed"],
        "messages": [
            "Your Private USB Microphone is too hot: lower it by about 6 dB.",
            "Full-chain check failed: C:/Users/private/model.onnx not found",
        ],
        "numbers": {
            "speech_peak_db": -0.4,
            "setup_confidence": 0.62,
            "noise_floor_db": -61,
            "limiter_on": True,
            "level_override": "too_hot",
            "mic_label": "alice_laptop_mic",
            "free_text": "Private USB Microphone",
            "Bad Key!": 1,
            "not_finite": float("nan"),
        },
        "recorded_at": "2026-10-14T09:30:00+02:00",
    }
    result.update(changes)
    return result


def test_setup_result_is_validated_bounded_and_never_raises() -> None:
    clean = sanitize_setup_result(_setup_result(extra="dropped", reasons="dropped"))
    assert clean is not None
    assert set(clean) == {"kind", "outcome", "reason_codes", "messages", "numbers", "recorded_at"}
    assert clean["recorded_at"] == "2026-10-14T07:30:00Z"
    assert clean["numbers"]["speech_peak_db"] == -0.4
    assert clean["numbers"]["noise_floor_db"] == -61
    assert clean["numbers"]["limiter_on"] is True
    assert "Bad Key!" not in clean["numbers"] and "not_finite" not in clean["numbers"]

    for garbage in (None, 3, "text", [], object()):
        assert sanitize_setup_result(garbage) is None

    huge = sanitize_setup_result({
        "kind": "voice_setup", "outcome": "applied",
        "reason_codes": [f"code_{n}" for n in range(500)] + ["Not A Code", 7, "x" * 200],
        "messages": ["m" * 5000] * 100,
        "numbers": {f"n_{n}": n for n in range(500)},
        "recorded_at": "yesterday-ish",
    })
    assert huge is not None
    assert len(huge["reason_codes"]) <= 16 and all(code.startswith("code_") for code in huge["reason_codes"])
    assert len(huge["messages"]) <= 8 and all(len(message) <= 240 for message in huge["messages"])
    assert len(huge["numbers"]) <= 32
    assert huge["recorded_at"].endswith("Z")

    unknown = sanitize_setup_result({"kind": "other_tool", "outcome": "exploded"})
    assert unknown is not None and (unknown["kind"], unknown["outcome"]) == ("other", "other")


@pytest.mark.parametrize("code", ["good", "too_hot", "too_quiet", "silent", "unreadable"])
def test_every_level_check_verdict_is_exported_and_nothing_else_is(code) -> None:
    exported = _build(setup_result=_setup_result(numbers={"level_override": code}))["setup_result"]
    assert exported["numbers"] == {"level_override": code}
    exported = _build(setup_result=_setup_result(numbers={"level_override": f"{code}_x"}))["setup_result"]
    assert exported["numbers"] == {}


def test_setup_numbers_are_bounded_and_sequences_are_accepted() -> None:
    limit = 2**53
    clean = sanitize_setup_result({
        "numbers": {"huge": 10**5000, "over": limit + 1, "under": -limit - 1,
                    "edge": limit, "low_edge": -limit, "float_big": 1e300},
        "reason_codes": ("tonal_check_failed", "Not A Code"),
        "messages": ("first", "second"),
    })
    assert clean is not None
    assert clean["numbers"] == {"edge": limit, "low_edge": -limit, "float_big": 1e300}
    assert clean["reason_codes"] == ["tonal_check_failed"]
    assert clean["messages"] == ["first", "second"]
    # A huge integer used to be accepted and then break every later export.
    snapshot = _build(setup_result=_setup_result(numbers={"x": 10**5000}))
    assert snapshot["setup_result"]["numbers"] == {}
    assert serialize_diagnostics_snapshot(snapshot)


def test_snapshot_exports_the_setup_result_as_codes_and_numbers_only() -> None:
    snapshot = _build(setup_result=_setup_result())
    assert snapshot["setup_result"] == {
        "kind": "voice_setup",
        "outcome": "rolled_back",
        "reason_codes": ["tonal_check_failed", "loudness_target_missed"],
        "numbers": {
            "level_override": "too_hot",
            "limiter_on": True,
            "noise_floor_db": -61,
            "setup_confidence": 0.62,
            "speech_peak_db": -0.4,
        },
        "message_count": 2,
        "recorded_at": "2026-10-14T07:30:00Z",
    }
    payload = serialize_diagnostics_snapshot(snapshot).decode("utf-8")
    # Messages can carry device names and exception text; only their count leaves the app.
    assert "Private USB Microphone" not in payload
    assert "model.onnx" not in payload and "Users" not in payload
    # A lowercase string that is not a known code can still be private.
    assert "alice_laptop_mic" not in payload and "mic_label" not in payload
    assert len(payload) < MAX_SERIALIZED_BYTES
    snapshot["setup_result"]["messages"] = ["leak"]
    with pytest.raises(ValueError, match="setup_result"):
        serialize_diagnostics_snapshot(snapshot)


def test_main_window_keeps_the_most_recent_setup_result_and_exports_it(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "setup.json"
    monkeypatch.setattr(
        main_window.QFileDialog, "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "JSON Files (*.json)"),
    )
    dialogs: list[str] = []
    monkeypatch.setattr(
        main_window.QMessageBox, "information", lambda _parent, _title, text: dialogs.append(text)
    )
    monkeypatch.setattr(
        main_window.QMessageBox, "critical",
        lambda *_args, **_kwargs: pytest.fail("export unexpectedly failed"),
    )
    window = _FakeWindow()
    main_window.MainWindow._export_diagnostics(cast(Any, window))
    assert json.loads(destination.read_text(encoding="utf-8"))["setup_result"] is None
    assert "last Voice Setup" not in dialogs[-1]

    main_window.MainWindow.record_setup_result(cast(Any, window), _setup_result(outcome="refused"))
    main_window.MainWindow.record_setup_result(cast(Any, window), _setup_result(kind="auto_eq", outcome="applied"))
    assert (window.last_setup_result["kind"], window.last_setup_result["outcome"]) == ("auto_eq", "applied")
    main_window.MainWindow._export_diagnostics(cast(Any, window))
    exported = json.loads(destination.read_text(encoding="utf-8"))["setup_result"]
    assert (exported["kind"], exported["outcome"]) == ("auto_eq", "applied")
    assert "last Voice Setup or Auto-EQ run" in dialogs[-1]

    # A malformed report from a dialog is ignored and cannot raise into it.
    main_window.MainWindow.record_setup_result(cast(Any, window), cast(Any, "garbage"))
    assert window.last_setup_result["kind"] == "auto_eq"


def test_a_new_main_window_has_no_setup_result(isolated_main_window) -> None:
    window: Any = isolated_main_window
    assert window.last_setup_result is None
    window.record_setup_result(_setup_result())
    assert cast(Any, window.last_setup_result)["outcome"] == "rolled_back"
