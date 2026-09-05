"""Tests for preset/config v1.7 migration and latency profile persistence."""

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from mic_eq import config
from mic_eq.config_parts.presets import MAX_PRESET_FILE_BYTES
from mic_eq.config_parts import app_config as app_config_module
from mic_eq.config_parts import presets as presets_module
from mic_eq.config_parts import shared as shared_module


Preset = config.Preset
AppConfig = config.AppConfig
LatencyCalibrationProfile = config.LatencyCalibrationProfile
DeviceIdentity = config.DeviceIdentity
DevicePresetBinding = config.DevicePresetBinding
build_latency_profile_key = config.build_latency_profile_key
legacy_latency_profile_key = config.legacy_latency_profile_key


def test_preset_save_is_atomic_and_preserves_existing_file_on_replace_failure(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "voice.json"
    destination.write_text("existing", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(presets_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        config.save_preset(Preset(name="Voice"), destination, overwrite=True)

    assert destination.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.glob("*.tmp")) == []


def test_auto_preset_paths_reject_sanitization_collisions_without_overwrite(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "presets"
    monkeypatch.setattr(presets_module, "get_presets_dir", lambda: presets_dir)
    first_path = config.save_preset(Preset(name="A/B"))

    with pytest.raises(FileExistsError):
        config.save_preset(Preset(name="A:B"), overwrite=False)

    assert config.load_preset(first_path).name == "A/B"


def test_list_presets_skips_recursively_malformed_json(tmp_path, monkeypatch):
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    monkeypatch.setattr(presets_module, "get_presets_dir", lambda: presets_dir)
    depth = 1200
    malformed = "{" + '"nested":{' * depth + "0" + "}" * depth + "\n"
    (presets_dir / "deep.json").write_text(malformed, encoding="utf-8")

    assert config.list_presets() == []


def test_load_preset_normalizes_deep_malformed_json(tmp_path, monkeypatch):
    presets_dir = tmp_path / "presets"
    imports_dir = tmp_path / "imports"
    presets_dir.mkdir()
    imports_dir.mkdir()
    monkeypatch.setattr(presets_module, "get_presets_dir", lambda: presets_dir)
    monkeypatch.setattr(presets_module, "get_preset_imports_dir", lambda: imports_dir)
    depth = 1200
    filepath = presets_dir / "deep.json"
    filepath.write_text(
        "{" + '"nested":{' * depth + "0" + "}" * depth + "\n",
        encoding="utf-8",
    )

    with pytest.raises(config.PresetValidationError, match="invalid or corrupted"):
        config.load_preset(filepath)


def test_exclusive_preset_creation_wins_against_a_competing_writer(
    tmp_path, monkeypatch
):
    destination = tmp_path / "race.json"

    def competing_writer(_source, target):
        Path(target).write_text("occupied", encoding="utf-8")
        raise FileExistsError("destination appeared during publication")

    monkeypatch.setattr(presets_module.os, "link", competing_writer)

    with pytest.raises(FileExistsError):
        config.save_preset(Preset(name="Race"), destination)

    assert destination.read_text(encoding="utf-8") == "occupied"
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_config_is_preserved_before_falling_back_to_defaults(
    tmp_path, monkeypatch
):
    filepath = tmp_path / "config.json"
    source = '{"window_geometry":'
    filepath.write_text(source, encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    assert app_config_module.load_config() == AppConfig()
    assert filepath.read_text(encoding="utf-8") == source
    assert filepath.with_name("config.json.corrupt").read_text(encoding="utf-8") == source


def test_newer_config_schema_is_preserved_before_falling_back_to_defaults(
    tmp_path, monkeypatch
):
    filepath = tmp_path / "config.json"
    source = json.dumps({"schema_version": config.CONFIG_SCHEMA_VERSION + 1})
    filepath.write_text(source, encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    loaded = app_config_module.load_config()
    assert loaded == AppConfig()
    assert loaded.save_blocked_reason
    assert app_config_module.save_config(loaded) is False
    assert filepath.read_text(encoding="utf-8") == source
    assert filepath.with_name("config.json.corrupt").read_text(encoding="utf-8") == source


def test_invalid_config_backup_collision_uses_exclusive_next_slot(tmp_path, monkeypatch):
    filepath = tmp_path / "config.json"
    filepath.write_text('{"broken":', encoding="utf-8")
    existing_backup = filepath.with_name("config.json.corrupt")
    existing_backup.write_text("keep this recovery copy", encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    app_config_module.load_config()

    assert existing_backup.read_text(encoding="utf-8") == "keep this recovery copy"
    assert filepath.with_name("config.json.corrupt.1").read_text(encoding="utf-8") == (
        '{"broken":'
    )


def test_negative_config_schema_is_rejected():
    with pytest.raises(config.PresetValidationError, match="cannot be negative"):
        AppConfig.from_dict({"schema_version": -1})


def test_config_write_is_blocked_when_recovery_copy_cannot_be_created(
    tmp_path, monkeypatch
):
    filepath = tmp_path / "config.json"
    filepath.write_text('{"broken":', encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)
    monkeypatch.setattr(
        app_config_module.shutil,
        "copyfileobj",
        Mock(side_effect=OSError("read-only")),
    )

    loaded = app_config_module.load_config()

    assert loaded.save_blocked_reason
    assert app_config_module.save_config(loaded) is False
    assert filepath.read_text(encoding="utf-8") == '{"broken":'
    assert not filepath.with_name("config.json.corrupt").exists()


def test_legacy_config_migration_retries_after_a_staged_copy_failure(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / "appdata"
    legacy_dir = base_dir / shared_module.LEGACY_APPDATA_DIR_NAME
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(shared_module, "_config_base_dir", lambda: base_dir)

    real_copytree = shared_module.shutil.copytree
    attempts = 0

    def fail_once(source, destination, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            Path(destination).mkdir()
            (Path(destination) / "partial").write_text("x", encoding="utf-8")
            raise OSError("simulated migration failure")
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(shared_module.shutil, "copytree", fail_once)

    config_dir = base_dir / shared_module.APPDATA_DIR_NAME
    assert shared_module._config_dir() == config_dir
    assert not config_dir.exists()
    assert app_config_module.save_config(AppConfig()) is False
    assert not config_dir.exists()
    assert shared_module._config_dir() == config_dir
    assert attempts == 2
    assert (config_dir / "config.json").read_text(encoding="utf-8") == "{}"


def test_preset_migration_to_v17_adds_deesser_defaults():
    old_data = {
        "name": "Legacy",
        "version": "1.6.0",
        "gate": {
            "enabled": True,
            "threshold_db": -40.0,
            "attack_ms": 10.0,
            "release_ms": 100.0,
        },
        "eq": {
            "enabled": True,
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        },
        "rnnoise": {
            "enabled": True,
            "strength": 1.0,
            "model": "rnnoise",
        },
        "compressor": {
            "enabled": True,
            "threshold_db": -20.0,
            "ratio": 4.0,
            "attack_ms": 10.0,
            "release_ms": 200.0,
            "makeup_gain_db": 0.0,
            "adaptive_release": False,
            "base_release_ms": 50.0,
            "auto_makeup_enabled": False,
            "target_lufs": -18.0,
        },
        "limiter": {
            "enabled": True,
            "ceiling_db": -0.5,
            "release_ms": 50.0,
        },
        "bypass": False,
    }

    preset = Preset.from_dict(old_data)

    assert preset.version == config.CURRENT_VERSION
    assert preset.deesser.enabled is False
    assert preset.deesser.auto_enabled is True
    assert preset.deesser.auto_amount == 0.5
    assert preset.deesser.low_cut_hz == 4000.0
    assert preset.deesser.high_cut_hz == 11000.0
    assert preset.deesser.threshold_db == -28.0
    assert preset.deesser.max_reduction_db == 6.0
    assert preset.compressor.sidechain_highpass_enabled is True
    assert preset.limiter.careful_output_enabled is True
    assert preset.value_provenance["gate.enabled"] == "explicit"
    assert preset.value_provenance["deesser.enabled"] == "migration_default"
    assert preset.value_provenance["compressor.sidechain_highpass_enabled"] == (
        "migration_default"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", {"not": "text"}, "preset name must be a string"),
        ("description", ["not", "text"], "preset description must be a string"),
    ],
)
def test_preset_rejects_non_string_metadata(field, value, message):
    with pytest.raises(config.PresetValidationError, match=message):
        Preset.from_dict({field: value})


def test_app_config_latency_profiles_round_trip():
    profile = LatencyCalibrationProfile(
        measured_round_trip_ms=36.5,
        estimated_one_way_ms=18.25,
        applied_compensation_ms=18.25,
        confidence=0.92,
        sample_rate=48000,
        timestamp_utc="2026-02-16T00:00:00Z",
        route_latency_ms=36.5,
        engine_latency_ms=24.0,
        total_latency_ms=60.5,
        engine_config_signature='{"noise_model":"rnnoise"}',
    )
    input_identity = DeviceIdentity(name="Mic A", is_default=False)
    output_identity = DeviceIdentity(name="Out B", is_default=True)
    key = build_latency_profile_key(input_identity, output_identity)

    cfg = AppConfig(
        last_input_device="Mic A",
        last_output_device="Out B",
        last_input_device_identity=input_identity,
        last_output_device_identity=output_identity,
        input_channel_mode="phase_safe_mono",
        main_splitter_sizes=[420, 680],
        main_control_tab_index=1,
        use_measured_latency=True,
        latency_calibration_profiles={key: profile},
    )

    raw = cfg.to_dict()
    assert raw["schema_version"] == config.CONFIG_SCHEMA_VERSION
    restored = AppConfig.from_dict(raw)

    assert restored.use_measured_latency is True
    assert restored.main_splitter_sizes == [420, 680]
    assert restored.main_control_tab_index == 1
    assert restored.input_channel_mode == "phase_safe_mono"
    assert key in restored.latency_calibration_profiles
    assert restored.last_input_device_identity == input_identity
    assert restored.last_output_device_identity == output_identity
    restored_profile = restored.latency_calibration_profiles[key]
    assert restored_profile.measured_round_trip_ms == 36.5
    assert restored_profile.estimated_one_way_ms == 18.25
    assert restored_profile.applied_compensation_ms == 18.25
    assert restored_profile.route_latency_ms == 36.5
    assert restored_profile.engine_latency_ms == 24.0
    assert restored_profile.total_latency_ms == 60.5
    assert restored_profile.engine_config_signature == '{"noise_model":"rnnoise"}'
    assert restored_profile.confidence == 0.92


def test_invalid_latency_profile_is_dropped_without_discarding_other_config():
    route_key = build_latency_profile_key(
        DeviceIdentity(name="Mic"),
        DeviceIdentity(name="Output"),
    )

    restored = AppConfig.from_dict(
        {
            "input_channel_mode": "left",
            "latency_calibration_profiles": {
                route_key: {
                    "measured_round_trip_ms": 20.0,
                    "estimated_one_way_ms": 0.0,
                    "applied_compensation_ms": 20.0,
                    "confidence": 0.9,
                    "route_latency_ms": "nan",
                }
            },
        }
    )

    assert restored.input_channel_mode == "left"
    assert restored.latency_calibration_profiles == {}


def test_latency_profile_total_is_derived_from_validated_components():
    profile = LatencyCalibrationProfile.from_dict(
        {
            "measured_round_trip_ms": 20.0,
            "estimated_one_way_ms": 0.0,
            "applied_compensation_ms": 20.0,
            "confidence": 0.9,
            "route_latency_ms": 20.0,
            "engine_latency_ms": 7.5,
            "total_latency_ms": 9999.0,
        }
    )

    assert profile.total_latency_ms == 27.5


def test_device_route_keys_ignore_default_status_and_survive_endpoint_rename():
    original_input = DeviceIdentity(
        name="Old Mic Name",
        is_default=False,
        endpoint_id="endpoint-input",
        host_api="WASAPI",
        direction="input",
    )
    renamed_input = DeviceIdentity(
        name="New Mic Name",
        is_default=True,
        endpoint_id="endpoint-input",
        host_api="wasapi",
        direction="input",
    )
    output = DeviceIdentity(
        name="Cable",
        endpoint_id="endpoint-output",
        host_api="WASAPI",
        direction="output",
    )

    assert build_latency_profile_key(
        original_input, output
    ) == build_latency_profile_key(renamed_input, output)


def test_fallback_route_keys_survive_windows_format_changes():
    original_input = DeviceIdentity(
        name="Microphone",
        host_api="WASAPI",
        direction="input",
        sample_rate=44_100,
        channels=1,
        name_ordinal=0,
    )
    changed_input = DeviceIdentity(
        name="Microphone",
        host_api="wasapi",
        direction="input",
        sample_rate=48_000,
        channels=2,
        name_ordinal=0,
    )
    output = DeviceIdentity(name="Cable", direction="output", name_ordinal=0)

    assert build_latency_profile_key(
        original_input, output
    ) == build_latency_profile_key(changed_input, output)


def test_device_identity_rejects_nonfinite_or_mistyped_persisted_fields():
    identity = DeviceIdentity.from_dict(
        {
            "name": "Microphone",
            "is_default": "false",
            "endpoint_id": {"not": "an id"},
            "host_api": ["WASAPI"],
            "direction": 123,
            "sample_rate": math.inf,
            "channels": 1.5,
            "name_ordinal": -1,
        }
    )

    assert identity == DeviceIdentity(name="Microphone")
    assert DeviceIdentity.from_dict({"name": {"not": "a name"}}) is None


def test_endpoint_route_keys_round_trip_through_app_config_without_collapsing():
    input_device = DeviceIdentity(
        name="Microphone",
        endpoint_id="endpoint-input",
        host_api="WASAPI",
        direction="input",
    )
    output_device = DeviceIdentity(
        name="Cable",
        endpoint_id="endpoint-output",
        host_api="WASAPI",
        direction="output",
    )
    route_key = build_latency_profile_key(input_device, output_device)
    profile = LatencyCalibrationProfile(
        measured_round_trip_ms=12.0,
        estimated_one_way_ms=12.0,
        applied_compensation_ms=12.0,
        confidence=0.9,
        route_latency_ms=12.0,
    )
    configured = AppConfig(
        latency_calibration_profiles={route_key: profile},
        device_preset_bindings={route_key: DevicePresetBinding("builtin:broadcast")},
    )

    restored = AppConfig.from_dict(configured.to_dict())

    assert set(restored.latency_calibration_profiles) == {route_key}
    assert set(restored.device_preset_bindings) == {route_key}
    assert '{"input":null,"output":null}' not in restored.device_preset_bindings


def test_structured_route_keys_preserve_delimiters_in_device_names():
    input_device = DeviceIdentity(
        name="USB Mic || Desk",
        host_api="WASAPI",
        direction="input",
    )
    output_device = DeviceIdentity(name="Speakers", direction="output")
    route_key = build_latency_profile_key(input_device, output_device)
    profile = LatencyCalibrationProfile(
        measured_round_trip_ms=12.0,
        estimated_one_way_ms=12.0,
        applied_compensation_ms=12.0,
        confidence=0.9,
    )

    restored = AppConfig.from_dict(
        {"latency_calibration_profiles": {route_key: profile.to_dict()}}
    )

    assert set(restored.latency_calibration_profiles) == {route_key}
    parsed = config.parse_latency_profile_key(route_key)
    assert parsed is not None
    parsed_input, parsed_output = parsed
    assert parsed_input is not None
    assert parsed_output is not None
    assert parsed_input.name == input_device.name.lower()
    assert parsed_input.host_api == input_device.host_api.lower()
    assert parsed_output.name == output_device.name.lower()


def test_malformed_route_keys_are_dropped_instead_of_collapsing_to_null_route():
    profile = LatencyCalibrationProfile(
        measured_round_trip_ms=12.0,
        estimated_one_way_ms=12.0,
        applied_compensation_ms=12.0,
        confidence=0.9,
    )
    malformed_keys = ("{}", '{"input":null,"output":null}', "||")
    restored = AppConfig.from_dict(
        {
            "latency_calibration_profiles": {
                key: profile.to_dict() for key in malformed_keys
            },
            "device_preset_bindings": {
                key: "builtin:broadcast" for key in malformed_keys
            },
        }
    )

    assert restored.latency_calibration_profiles == {}
    assert restored.device_preset_bindings == {}


def test_malformed_legacy_device_names_are_not_stringified():
    restored = AppConfig.from_dict(
        {
            "last_input_device": {"unexpected": "object"},
            "last_output_device": 42,
        }
    )

    assert restored.last_input_device == ""
    assert restored.last_output_device == ""
    assert restored.last_input_device_identity is None
    assert restored.last_output_device_identity is None


def test_device_preset_bindings_round_trip_with_provenance_and_legacy_migration():
    route_key = build_latency_profile_key(
        DeviceIdentity(name="Mic", is_default=True),
        DeviceIdentity(name="Cable", is_default=False),
    )
    config_value = AppConfig(
        auto_apply_device_presets=False,
        device_preset_bindings={
            route_key: DevicePresetBinding("builtin:broadcast", "explicit_user")
        },
    )

    restored = AppConfig.from_dict(config_value.to_dict())
    assert restored.auto_apply_device_presets is False
    assert restored.device_preset_bindings[route_key] == DevicePresetBinding(
        "builtin:broadcast", "explicit_user"
    )

    migrated = AppConfig.from_dict(
        {"device_preset_bindings": {"Mic||Cable": "custom:Voice.json"}}
    )
    migrated_key = build_latency_profile_key(
        DeviceIdentity(name="Mic"), DeviceIdentity(name="Cable")
    )
    assert migrated.device_preset_bindings[migrated_key] == DevicePresetBinding(
        "custom:Voice.json", "legacy_migration"
    )


def test_first_run_setup_progress_round_trips_and_invalid_values_fail_closed():
    configured = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step="latency",
        first_run_setup_steps={
            "devices": "completed",
            "route": "completed",
            "latency": "pending",
            "voice": "skipped",
        },
    )
    restored = AppConfig.from_dict(configured.to_dict())
    assert restored.first_run_setup_state == "in_progress"
    assert restored.first_run_setup_step == "latency"
    assert restored.first_run_setup_steps == configured.first_run_setup_steps

    invalid = AppConfig.from_dict(
        {
            "first_run_setup_state": "magical",
            "first_run_setup_step": "telemetry",
            "first_run_setup_steps": {"devices": "unknown", "route": "completed"},
        }
    )
    assert invalid.first_run_setup_state == "not_started"
    assert invalid.first_run_setup_step == "devices"
    assert invalid.first_run_setup_steps == {
        "devices": "pending",
        "route": "completed",
        "latency": "pending",
        "voice": "pending",
    }


def test_existing_pre_wizard_config_does_not_trigger_first_run_setup():
    migrated = AppConfig.from_dict(
        {
            "last_input_device": "Existing Microphone",
            "use_measured_latency": False,
        }
    )

    assert migrated.first_run_setup_state == "completed_with_skips"
    assert set(migrated.first_run_setup_steps.values()) == {"skipped"}

    assert AppConfig.from_dict({}).first_run_setup_state == "not_started"


def test_voice_setup_dynamics_preferences_round_trip_and_migrate():
    configured = AppConfig(
        voice_setup_dynamics_intensity="custom",
        voice_setup_custom_p95_db=4.25,
        voice_setup_custom_peak_cap_db=9.5,
    )
    restored = AppConfig.from_dict(configured.to_dict())

    assert restored.voice_setup_dynamics_intensity == "custom"
    assert restored.voice_setup_custom_p95_db == 4.25
    assert restored.voice_setup_custom_peak_cap_db == 9.5

    legacy = AppConfig.from_dict({})
    assert legacy.voice_setup_dynamics_intensity == "balanced"
    assert legacy.voice_setup_custom_p95_db == 3.5
    assert legacy.voice_setup_custom_peak_cap_db == 8.0

    invalid = AppConfig.from_dict(
        {
            "voice_setup_dynamics_intensity": "maximum",
            "voice_setup_custom_p95_db": 50,
            "voice_setup_custom_peak_cap_db": -2,
        }
    )
    assert invalid.voice_setup_dynamics_intensity == "balanced"
    assert invalid.voice_setup_custom_p95_db == 3.5
    assert invalid.voice_setup_custom_peak_cap_db == 8.0


@pytest.mark.parametrize("payload", [[], "bad", 123, True, None])
def test_app_config_non_object_payload_returns_defaults(payload):
    restored = AppConfig.from_dict(payload)

    assert restored == AppConfig()


def test_app_config_restores_false_boolean_without_truthy_coercion():
    restored = AppConfig.from_dict({"use_measured_latency": False})

    assert restored.use_measured_latency is False


def test_app_config_rejects_corrupt_boolean_to_default():
    restored = AppConfig.from_dict({"use_measured_latency": "false"})

    assert restored.use_measured_latency is True


@pytest.mark.parametrize("value", ["bad", "", 1, True, None, ["left"]])
def test_app_config_normalizes_invalid_input_channel_mode(value):
    restored = AppConfig.from_dict({"input_channel_mode": value})

    assert restored.input_channel_mode == "phase_safe_mono"


def test_app_config_uses_phase_safe_input_for_new_config():
    assert AppConfig().input_channel_mode == "phase_safe_mono"
    assert AppConfig.from_dict({}).input_channel_mode == "phase_safe_mono"


@pytest.mark.parametrize(
    "value", ["average", "left", "right", "max_rms", "phase_safe_mono"]
)
def test_app_config_preserves_valid_input_channel_mode(value):
    restored = AppConfig.from_dict({"input_channel_mode": value})

    assert restored.input_channel_mode == value


@pytest.mark.parametrize("payload", [[], "bad", 123, True, None])
def test_load_config_falls_back_to_defaults_for_non_object_json(
    payload, monkeypatch, tmp_path
):
    config_path = tmp_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    monkeypatch.setattr(app_config_module, "get_config_file", lambda: config_path)

    restored = app_config_module.load_config()

    assert restored == AppConfig()


def test_app_config_ignores_invalid_window_geometry_values():
    for value in (
        ["not", "a", "dict"],
        "invalid",
        {"x": 1, "y": 2},
        {"x": 1, "y": 2, "width": math.inf, "height": 700},
    ):
        restored = AppConfig.from_dict({"window_geometry": value})

        assert restored.window_geometry is None


@pytest.mark.parametrize(
    "value", [123, ["builtin:voice"], {"preset": "voice"}, True, None]
)
def test_app_config_normalizes_invalid_last_preset_values_to_safe_string(value):
    restored = AppConfig.from_dict({"last_preset": value})

    assert restored.last_preset == ""
    assert isinstance(restored.last_preset, str)


def test_app_config_preserves_valid_last_preset_string():
    restored = AppConfig.from_dict({"last_preset": "builtin:voice"})

    assert restored.last_preset == "builtin:voice"


@pytest.mark.parametrize(
    "field", ["last_preset", "startup_preset"]
)
@pytest.mark.parametrize(
    "value", [123, ["builtin:voice"], {"preset": "voice"}, True, None]
)
def test_app_config_normalizes_invalid_preset_ids_to_safe_strings(field, value):
    restored = AppConfig.from_dict({field: value})

    assert getattr(restored, field) == ""
    assert isinstance(getattr(restored, field), str)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([420, 680], [420, 680]),
        ([420.0, "680"], [420, 680]),
        (None, None),
        (123, None),
        ([420], None),
        ([420, math.inf], None),
        ([420, -1], None),
        ([420, 1_000_001], None),
    ],
)
def test_app_config_validates_splitter_sizes_without_resetting_other_fields(
    value, expected
):
    restored = AppConfig.from_dict(
        {"main_splitter_sizes": value, "last_preset": "builtin:voice"}
    )

    assert restored.main_splitter_sizes == expected
    assert restored.last_preset == "builtin:voice"


@pytest.mark.parametrize("value", [math.inf, -1, 65, 1.5, "invalid", [], True])
def test_app_config_invalid_tab_index_falls_back_without_resetting_other_fields(value):
    restored = AppConfig.from_dict(
        {"main_control_tab_index": value, "last_preset": "builtin:voice"}
    )

    assert restored.main_control_tab_index == 0
    assert restored.last_preset == "builtin:voice"


def test_app_config_accepts_and_clamps_valid_window_geometry():
    restored = AppConfig.from_dict(
        {"window_geometry": {"x": 10.2, "y": 20.7, "width": 300.0, "height": 200.0}}
    )

    assert restored.window_geometry == {"x": 10, "y": 21, "width": 640, "height": 480}


def test_app_config_migrates_legacy_device_names_and_profile_keys():
    profile = LatencyCalibrationProfile(
        measured_round_trip_ms=28.0,
        estimated_one_way_ms=14.0,
        applied_compensation_ms=14.0,
        confidence=0.8,
        sample_rate=48000,
        timestamp_utc="2026-03-27T00:00:00Z",
    )
    legacy_key = legacy_latency_profile_key("Mic A", "Out B")

    restored = AppConfig.from_dict(
        {
            "last_input_device": "Mic A",
            "last_output_device": "Out B",
            "latency_calibration_profiles": {
                legacy_key: profile.to_dict(),
            },
        }
    )

    expected_key = build_latency_profile_key(
        DeviceIdentity(name="Mic A", is_default=False),
        DeviceIdentity(name="Out B", is_default=False),
    )

    assert restored.last_input_device_identity == DeviceIdentity(
        name="Mic A", is_default=False
    )
    assert restored.last_output_device_identity == DeviceIdentity(
        name="Out B", is_default=False
    )
    assert expected_key in restored.latency_calibration_profiles
    assert legacy_key not in restored.latency_calibration_profiles
    restored_profile = restored.latency_calibration_profiles[expected_key]
    assert restored_profile.measured_round_trip_ms == 28.0
    assert restored_profile.route_latency_ms == 28.0
    assert restored_profile.engine_latency_ms == 0.0
    assert restored_profile.total_latency_ms == 28.0


def test_load_preset_rejects_path_outside_allowlisted_roots():
    with (
        tempfile.TemporaryDirectory() as appdata_dir,
        tempfile.TemporaryDirectory() as outside_dir,
    ):
        old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = appdata_dir
        try:
            outside_path = Path(outside_dir) / "outside.json"
            with open(outside_path, "w", encoding="utf-8") as f:
                json.dump(Preset(name="Outside").to_dict(), f, indent=2)

            try:
                config.load_preset(outside_path)
                assert False, (
                    "Expected PresetValidationError for outside allowlisted roots"
                )
            except config.PresetValidationError as e:
                assert "allowed preset roots" in str(e)
        finally:
            if old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old_appdata


def test_load_preset_allows_imports_root():
    with tempfile.TemporaryDirectory() as appdata_dir:
        old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = appdata_dir
        try:
            imports_dir = config.get_preset_imports_dir()
            preset_path = imports_dir / "imported.json"
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(Preset(name="Imported").to_dict(), f, indent=2)

            loaded = config.load_preset(preset_path)
            assert loaded.name == "Imported"
        finally:
            if old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old_appdata


def test_load_preset_rejects_oversized_file_before_json_parse(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    preset_path = config.get_presets_dir() / "oversized.json"
    preset_path.write_bytes(b"{" + b" " * MAX_PRESET_FILE_BYTES)

    with pytest.raises(config.PresetValidationError, match="file too large"):
        config.load_preset(preset_path)


def test_preset_rejects_non_finite_numeric_values():
    data = Preset(name="Bad").to_dict()
    for band in data["eq"]["bands"]:
        band["gain_db"] = math.nan

    try:
        Preset.from_dict(data)
        assert False, "Expected non-finite EQ gain to be rejected"
    except config.PresetValidationError as e:
        assert "finite number" in str(e)


def test_preset_rejects_string_booleans():
    data = Preset(name="Bad").to_dict()
    data["eq"]["enabled"] = "false"

    try:
        Preset.from_dict(data)
        assert False, "Expected string boolean to be rejected"
    except config.PresetValidationError as e:
        assert "must be true or false" in str(e)


def test_eq_band_frequencies_round_trip():
    data = Preset(name="Auto EQ").to_dict()
    frequencies = [
        72.0,
        144.0,
        300.0,
        650.0,
        1300.0,
        2600.0,
        5100.0,
        8200.0,
        11800.0,
        15500.0,
    ]
    for band, frequency in zip(data["eq"]["bands"], frequencies):
        band["frequency_hz"] = frequency

    preset = Preset.from_dict(data)

    assert preset.eq.band_freqs == frequencies


def test_saved_auto_eq_preset_round_trips_dynamic_band_frequencies():
    with tempfile.TemporaryDirectory() as appdata_dir:
        old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = appdata_dir
        try:
            dynamic_freqs = [
                83.0,
                205.0,
                310.0,
                490.0,
                760.0,
                2300.0,
                3650.0,
                5800.0,
                8400.0,
                15600.0,
            ]
            preset = Preset(name="Auto EQ Dynamic")
            preset.eq.band_freqs = dynamic_freqs
            preset.eq.band_gains = [
                -1.0,
                0.0,
                0.5,
                -0.3,
                1.0,
                2.4,
                -1.2,
                0.7,
                -0.4,
                1.1,
            ]
            preset.eq.band_qs = [1.0, 1.4, 1.6, 1.8, 2.1, 4.8, 2.4, 1.9, 1.3, 0.8]

            path = config.save_preset(
                preset, config.get_presets_dir() / "auto_eq_dynamic.json"
            )
            loaded = config.load_preset(path)

            assert loaded.eq.band_freqs == dynamic_freqs
            assert loaded.eq.band_gains == preset.eq.band_gains
            assert loaded.eq.band_qs == preset.eq.band_qs
        finally:
            if old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old_appdata
