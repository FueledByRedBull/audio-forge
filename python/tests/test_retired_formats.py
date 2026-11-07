"""Formats that 1.24.0 no longer converts are refused, and the 1.23.0 format still loads."""

import json
from pathlib import Path

import pytest

from mic_eq import config
from mic_eq.config_parts import app_config as app_config_module
from mic_eq.config_parts import presets as presets_module
from mic_eq.config_parts import shared as shared_module
from mic_eq.config_parts.calibration import CalibrationResult, parse_calibration_results

Preset = config.Preset
AppConfig = config.AppConfig
EQSettings = config.EQSettings
PresetValidationError = config.PresetValidationError

OLDEST = shared_module.OLDEST_OPENED_VERSION


def _preset_payload(**changes) -> dict:
    payload = Preset().to_dict()
    payload.update(changes)
    return payload


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_oldest_opened_preset_is_the_last_release_that_converted():
    assert OLDEST == "1.23.0"


def test_an_unversioned_preset_is_refused_with_the_resave_instruction():
    payload = _preset_payload()
    del payload["version"]

    with pytest.raises(PresetValidationError) as refusal:
        Preset.from_dict(payload)

    assert "1.23.0" in str(refusal.value)
    assert "save" in str(refusal.value).lower()


def test_a_preset_stamped_1_22_0_is_refused_with_the_resave_instruction():
    with pytest.raises(PresetValidationError) as refusal:
        Preset.from_dict(_preset_payload(version="1.22.0"))

    assert "1.22.0" in str(refusal.value)
    assert "1.23.0" in str(refusal.value)


@pytest.mark.parametrize("version", ["1.0.0", "1.6.0", "1.10.0", "", "not a version", None])
def test_every_older_or_unreadable_stamp_is_refused(version):
    with pytest.raises(PresetValidationError, match="1.23.0"):
        Preset.from_dict(_preset_payload(version=version))


def test_a_refused_preset_file_is_reported_by_import_and_load_not_raised_raw(tmp_path):
    source = _write_json(tmp_path / "old.json", _preset_payload(version="1.22.0"))

    with pytest.raises(PresetValidationError, match="1.23.0"):
        presets_module.import_preset(source)
    presets_dir = presets_module.get_presets_dir()
    copied = _write_json(presets_dir / "old.json", _preset_payload(version="1.22.0"))
    with pytest.raises(PresetValidationError, match="1.23.0"):
        presets_module.load_preset(copied)
    assert presets_module.list_presets() == []


def test_a_preset_newer_than_this_build_is_still_refused():
    with pytest.raises(PresetValidationError, match="newer"):
        Preset.from_dict(_preset_payload(version="99.0.0"))


def test_a_preset_from_the_oldest_opened_release_loads_in_a_later_build(monkeypatch):
    monkeypatch.setattr(presets_module, "CURRENT_VERSION", "1.99.0")
    payload = _preset_payload(version=OLDEST)

    loaded = Preset.from_dict(payload)

    assert loaded.version == "1.99.0"
    assert loaded.eq.to_dict() == Preset().eq.to_dict()
    assert loaded.gate == Preset().gate


def test_a_preset_that_stores_the_eq_as_band_arrays_is_refused(monkeypatch):
    monkeypatch.setattr(presets_module, "CURRENT_VERSION", "1.99.0")
    legacy_eq = {
        "enabled": True,
        "band_freqs": list(config.EQ_FREQUENCIES),
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
    }

    with pytest.raises(PresetValidationError, match="eq"):
        Preset.from_dict(_preset_payload(version=OLDEST, eq=legacy_eq))
    with pytest.raises(ValueError, match="typed"):
        EQSettings.from_dict(legacy_eq)


def test_a_combined_stage_eq_schema_2_is_refused():
    payload = Preset().eq.to_dict()
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema"):
        EQSettings.from_dict(payload)


def test_a_calibration_record_version_1_is_dropped():
    record = CalibrationResult.capture(
        "eq_only", "route", {"eq": {"enabled": True}}, "flat", ("eq",)
    ).to_dict()
    record["version"] = 1
    for name in ("correction_hash", "verification_hash", "compatibility"):
        del record[name]

    assert CalibrationResult.from_value(record) is None
    assert parse_calibration_results([record]) == []
    loaded = AppConfig.from_dict(
        {"schema_version": config.CONFIG_SCHEMA_VERSION, "calibration_results": [record]}
    )
    assert loaded.calibration_results == []


def test_a_latency_profile_from_before_route_aware_calibration_is_dropped():
    profile = {
        "measured_round_trip_ms": 20.0,
        "estimated_one_way_ms": 10.0,
        "applied_compensation_ms": 10.0,
        "confidence": 0.9,
    }
    with pytest.raises(ValueError, match="route_latency_ms"):
        config.LatencyCalibrationProfile.from_dict(profile)

    key = config.build_device_route_key(
        config.DeviceIdentity(name="Mic"), config.DeviceIdentity(name="Speakers")
    )
    loaded = AppConfig.from_dict(
        {
            "schema_version": config.CONFIG_SCHEMA_VERSION,
            "close_to_tray": True,
            "latency_calibration_profiles": {key: profile},
        }
    )
    assert loaded.latency_calibration_profiles == {}
    assert loaded.close_to_tray is True


@pytest.mark.parametrize("stamp", [0, 1])
def test_a_config_stamped_older_than_schema_2_is_refused(stamp):
    with pytest.raises(PresetValidationError, match="1.23.0"):
        AppConfig.from_dict({"schema_version": stamp, "close_to_tray": True})


@pytest.mark.parametrize(
    "old_config",
    [{"close_to_tray": True}, {"schema_version": 0}, {"schema_version": 1, "close_to_tray": True}],
    ids=["unstamped", "schema-0", "schema-1"],
)
def test_loading_an_old_config_keeps_a_recovery_copy_and_says_why(
    tmp_path, monkeypatch, old_config
):
    filepath = tmp_path / "config.json"
    source = json.dumps(old_config)
    filepath.write_text(source, encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    loaded = app_config_module.load_config()

    assert loaded == AppConfig()
    assert loaded.load_warning is not None and "1.23.0" in loaded.load_warning
    assert loaded.save_blocked_reason is None
    assert filepath.with_name("config.json.corrupt").read_text(encoding="utf-8") == source


def test_a_bare_string_device_preset_binding_is_dropped():
    assert config.DevicePresetBinding.from_value("custom:Warm") is None
    key = config.build_device_route_key(
        config.DeviceIdentity(name="Mic"), config.DeviceIdentity(name="Speakers")
    )
    loaded = AppConfig.from_dict(
        {
            "schema_version": config.CONFIG_SCHEMA_VERSION,
            "device_preset_bindings": {key: "custom:Warm"},
        }
    )
    assert loaded.device_preset_bindings == {}


def test_a_plain_friendly_name_is_not_an_input_preference_key():
    preference = {"channel_mode": "left", "cleanup_mode": "off"}
    loaded = AppConfig.from_dict(
        {
            "schema_version": config.CONFIG_SCHEMA_VERSION,
            "input_device_preferences": {"Studio Mic": preference},
        }
    )
    assert loaded.input_device_preferences == {}


def test_a_miceq_folder_without_an_audioforge_folder_is_not_migrated(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / "appdata"
    old_dir = base_dir / "MicEq"
    (old_dir / "presets").mkdir(parents=True)
    old_config = json.dumps({"schema_version": 2, "close_to_tray": True})
    (old_dir / "config.json").write_text(old_config, encoding="utf-8")
    _write_json(old_dir / "presets" / "old.json", _preset_payload())
    monkeypatch.setattr(shared_module, "_config_base_dir", lambda: base_dir)

    config_dir = base_dir / shared_module.APPDATA_DIR_NAME
    assert shared_module.get_config_file() == config_dir / "config.json"

    assert list(config_dir.iterdir()) == []
    assert app_config_module.load_config() == AppConfig()
    assert app_config_module.load_config().load_warning is None
    assert presets_module.list_presets() == []
    assert app_config_module.save_config(AppConfig()) is True
    assert (old_dir / "config.json").read_text(encoding="utf-8") == old_config
    assert sorted(path.name for path in base_dir.iterdir()) == ["AudioForge", "MicEq"]


def test_a_fresh_install_still_starts_the_first_run_setup(tmp_path, monkeypatch):
    base_dir = tmp_path / "appdata"
    monkeypatch.setattr(shared_module, "_config_base_dir", lambda: base_dir)

    fresh = app_config_module.load_config()

    assert fresh.first_run_setup_state == "not_started"
    assert set(fresh.first_run_setup_steps.values()) == {"pending"}


def test_a_1_23_0_preset_written_by_the_current_writer_loads_unchanged(tmp_path):
    preset = Preset(name="Warm voice", description="kept")
    preset.gate.threshold_db = -35.0
    path = presets_module.save_preset(preset, tmp_path / "warm.json")
    written = json.loads(path.read_text(encoding="utf-8"))

    assert written["version"] == config.CURRENT_VERSION
    reloaded = presets_module.import_preset(path)[0]
    assert reloaded.to_dict() == preset.to_dict()
    resaved = presets_module.save_preset(reloaded, tmp_path / "again.json")
    assert resaved.read_bytes() == path.read_bytes()


def test_a_1_23_0_config_written_by_the_current_writer_loads_unchanged(
    tmp_path, monkeypatch
):
    mic = config.DeviceIdentity(
        name="Studio Mic", endpoint_id="{0.0.1.00000000}.{abc}", host_api="WASAPI",
        direction="input",
    )
    out = config.DeviceIdentity(name="Speakers", endpoint_id="{0.0.0.00000000}.{def}",
                                host_api="WASAPI", direction="output")
    route = config.build_device_route_key(mic, out)
    result = CalibrationResult.capture(
        "eq_only", "route", {"eq": Preset().eq.to_dict()}, "flat", ("eq",)
    )
    original = AppConfig(
        last_input_device=mic.name,
        last_output_device=out.name,
        last_input_device_identity=mic,
        last_output_device_identity=out,
        input_channel_mode="left",
        last_preset="builtin:voice",
        startup_preset="builtin:voice",
        window_geometry={"x": 10, "y": 20, "width": 900, "height": 700},
        latency_calibration_profiles={
            route: config.LatencyCalibrationProfile(
                measured_round_trip_ms=20.0,
                estimated_one_way_ms=10.0,
                applied_compensation_ms=20.0,
                confidence=0.9,
                route_latency_ms=20.0,
                engine_latency_ms=5.0,
                total_latency_ms=25.0,
            )
        },
        device_preset_bindings={
            route: config.DevicePresetBinding("builtin:voice", "legacy_migration")
        },
        input_device_preferences={
            config.build_input_device_preference_key(mic): config.InputDevicePreference(
                "left", "gentle"
            )
        },
        route_input_preferences={route: config.InputDevicePreference("right", "off")},
        calibration_results=[result],
        first_run_setup_state="completed",
        first_run_setup_step="voice",
        first_run_setup_steps={step: "completed" for step in app_config_module.FIRST_RUN_SETUP_STEPS},
    )
    filepath = tmp_path / "config.json"
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    assert app_config_module.save_config(original) is True
    written = filepath.read_bytes()
    loaded = app_config_module.load_config()

    assert loaded.load_warning is None and loaded.save_blocked_reason is None
    assert loaded == original
    assert app_config_module.save_config(loaded) is True
    assert filepath.read_bytes() == written
    assert json.loads(written)["schema_version"] == 2


def test_only_an_age_refusal_is_a_too_old_error():
    old = _preset_payload(version="1.22.0")
    unstamped = _preset_payload()
    del unstamped["version"]
    for payload in (old, unstamped):
        with pytest.raises(presets_module.PresetTooOldError) as refusal:
            Preset.from_dict(payload)
        assert isinstance(refusal.value, PresetValidationError)

    broken = _preset_payload()
    broken["gate"]["threshold_db"] = "bad"
    for payload in (broken, _preset_payload(version="99.0.0")):
        with pytest.raises(PresetValidationError) as other:
            Preset.from_dict(payload)
        assert not isinstance(other.value, presets_module.PresetTooOldError)


def test_presets_refused_for_age_are_found_in_the_presets_and_imports_folders():
    presets_dir = presets_module.get_presets_dir()
    imports_dir = presets_module.get_preset_imports_dir()
    _write_json(presets_dir / "old.json", _preset_payload(name="Old Warm", version="1.22.0"))
    unstamped = _preset_payload(name="Never Stamped")
    del unstamped["version"]
    _write_json(imports_dir / "unstamped.json", unstamped)
    broken = _preset_payload(name="Broken")
    broken["gate"]["threshold_db"] = "bad"
    _write_json(presets_dir / "broken.json", broken)
    (presets_dir / "garbage.json").write_text("{not json", encoding="utf-8")
    _write_json(presets_dir / "fine.json", _preset_payload(name="Fine"))

    refused = presets_module.find_presets_refused_for_age()

    assert sorted((item.path.name, item.name) for item in refused) == [
        ("old.json", "Old Warm"),
        ("unstamped.json", "Never Stamped"),
    ]
    assert all("1.23.0" in item.reason for item in refused)
    assert [name for name, _path in presets_module.list_presets()] == ["Fine"]


def test_a_refused_preset_is_matched_to_a_startup_or_route_id_by_file_or_name():
    from mic_eq.ui.startup_presets import refusal_reason_for_preset_id

    refused = [
        presets_module.RefusedPreset(Path("old.json"), "Old Warm", "reason one"),
        presets_module.RefusedPreset(Path("other.json"), "Other", "reason two"),
    ]

    assert refusal_reason_for_preset_id("custom-file:old.json", refused) == "reason one"
    assert refusal_reason_for_preset_id("custom:Other", refused) == "reason two"
    assert refusal_reason_for_preset_id("custom:other.json", refused) == "reason two"
    assert refusal_reason_for_preset_id("custom-file:missing.json", refused) is None
    assert refusal_reason_for_preset_id("builtin:voice", refused) is None
    twins = refused + [presets_module.RefusedPreset(Path("old2.json"), "Old Warm", "x")]
    assert refusal_reason_for_preset_id("custom:Old Warm", twins) is None


def test_an_imported_refused_file_is_not_the_preset_a_startup_or_route_id_names():
    from mic_eq.ui.startup_presets import refusal_reason_for_preset_id

    presets_dir = presets_module.get_presets_dir()
    imports_dir = presets_module.get_preset_imports_dir()
    _write_json(imports_dir / "old.json", _preset_payload(name="Old Warm", version="1.22.0"))

    refused = presets_module.find_presets_refused_for_age()

    assert [item.path for item in refused] == [imports_dir.resolve() / "old.json"]
    assert refusal_reason_for_preset_id("custom-file:old.json", refused) is None
    assert refusal_reason_for_preset_id("custom:Old Warm", refused) is None
    _write_json(presets_dir / "old.json", _preset_payload(name="Old Warm", version="1.22.0"))
    refused = presets_module.find_presets_refused_for_age()
    assert len(refused) == 2
    assert "1.23.0" in (refusal_reason_for_preset_id("custom-file:old.json", refused) or "")
