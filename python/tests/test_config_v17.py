"""Tests for preset/config v1.7 migration and latency profile persistence."""

import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "mic_eq" / "config.py"
config_spec = importlib.util.spec_from_file_location("mic_eq.config", CONFIG_PATH)
assert config_spec is not None and config_spec.loader is not None
config = importlib.util.module_from_spec(config_spec)
sys.modules["mic_eq.config"] = config
config_spec.loader.exec_module(config)


Preset = config.Preset
AppConfig = config.AppConfig
LatencyCalibrationProfile = config.LatencyCalibrationProfile
DeviceIdentity = config.DeviceIdentity
build_latency_profile_key = config.build_latency_profile_key
legacy_latency_profile_key = config.legacy_latency_profile_key


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

    assert preset.version == "1.7.17"
    assert preset.deesser.enabled is False
    assert preset.deesser.auto_enabled is True
    assert preset.deesser.auto_amount == 0.5
    assert preset.deesser.low_cut_hz == 4000.0
    assert preset.deesser.high_cut_hz == 9000.0
    assert preset.deesser.threshold_db == -28.0
    assert preset.deesser.max_reduction_db == 6.0


def test_app_config_latency_profiles_round_trip():
    profile = LatencyCalibrationProfile(
        measured_round_trip_ms=36.5,
        estimated_one_way_ms=18.25,
        applied_compensation_ms=18.25,
        confidence=0.92,
        sample_rate=48000,
        timestamp_utc="2026-02-16T00:00:00Z",
    )
    input_identity = DeviceIdentity(name="Mic A", is_default=False)
    output_identity = DeviceIdentity(name="Out B", is_default=True)
    key = build_latency_profile_key(input_identity, output_identity)

    cfg = AppConfig(
        last_input_device="Mic A",
        last_output_device="Out B",
        last_input_device_identity=input_identity,
        last_output_device_identity=output_identity,
        main_splitter_sizes=[420, 680],
        main_control_tab_index=1,
        use_measured_latency=True,
        latency_calibration_profiles={key: profile},
    )

    raw = cfg.to_dict()
    restored = AppConfig.from_dict(raw)

    assert restored.use_measured_latency is True
    assert restored.main_splitter_sizes == [420, 680]
    assert restored.main_control_tab_index == 1
    assert key in restored.latency_calibration_profiles
    assert restored.last_input_device_identity == input_identity
    assert restored.last_output_device_identity == output_identity
    restored_profile = restored.latency_calibration_profiles[key]
    assert restored_profile.measured_round_trip_ms == 36.5
    assert restored_profile.estimated_one_way_ms == 18.25
    assert restored_profile.applied_compensation_ms == 18.25
    assert restored_profile.confidence == 0.92


def test_app_config_restores_false_boolean_without_truthy_coercion():
    restored = AppConfig.from_dict({"use_measured_latency": False})

    assert restored.use_measured_latency is False


def test_app_config_rejects_corrupt_boolean_to_default():
    restored = AppConfig.from_dict({"use_measured_latency": "false"})

    assert restored.use_measured_latency is True


def test_app_config_ignores_invalid_window_geometry_values():
    for value in (["not", "a", "dict"], "invalid", {"x": 1, "y": 2}, {"x": 1, "y": 2, "width": math.inf, "height": 700}):
        restored = AppConfig.from_dict({"window_geometry": value})

        assert restored.window_geometry is None


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

    assert restored.last_input_device_identity == DeviceIdentity(name="Mic A", is_default=False)
    assert restored.last_output_device_identity == DeviceIdentity(name="Out B", is_default=False)
    assert expected_key in restored.latency_calibration_profiles
    assert legacy_key not in restored.latency_calibration_profiles
    restored_profile = restored.latency_calibration_profiles[expected_key]
    assert restored_profile.measured_round_trip_ms == 28.0


def test_load_preset_rejects_path_outside_allowlisted_roots():
    with tempfile.TemporaryDirectory() as appdata_dir, tempfile.TemporaryDirectory() as outside_dir:
        old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = appdata_dir
        try:
            outside_path = Path(outside_dir) / "outside.json"
            with open(outside_path, "w", encoding="utf-8") as f:
                json.dump(Preset(name="Outside").to_dict(), f, indent=2)

            try:
                config.load_preset(outside_path)
                assert False, "Expected PresetValidationError for outside allowlisted roots"
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


def test_preset_rejects_non_finite_numeric_values():
    data = Preset(name="Bad").to_dict()
    data["eq"]["band_gains"] = [math.nan] * 10

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
    data["eq"]["band_freqs"] = [72.0, 144.0, 300.0, 650.0, 1300.0, 2600.0, 5100.0, 8200.0, 11800.0, 15500.0]

    preset = Preset.from_dict(data)

    assert preset.eq.band_freqs == data["eq"]["band_freqs"]


def test_saved_auto_eq_preset_round_trips_dynamic_band_frequencies():
    with tempfile.TemporaryDirectory() as appdata_dir:
        old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = appdata_dir
        try:
            dynamic_freqs = [83.0, 205.0, 310.0, 490.0, 760.0, 2300.0, 3650.0, 5800.0, 8400.0, 15600.0]
            preset = Preset(name="Auto EQ Dynamic")
            preset.eq.band_freqs = dynamic_freqs
            preset.eq.band_gains = [-1.0, 0.0, 0.5, -0.3, 1.0, 2.4, -1.2, 0.7, -0.4, 1.1]
            preset.eq.band_qs = [1.0, 1.4, 1.6, 1.8, 2.1, 4.8, 2.4, 1.9, 1.3, 0.8]

            path = config.save_preset(preset, config.get_presets_dir() / "auto_eq_dynamic.json")
            loaded = config.load_preset(path)

            assert loaded.eq.band_freqs == dynamic_freqs
            assert loaded.eq.band_gains == preset.eq.band_gains
            assert loaded.eq.band_qs == preset.eq.band_qs
        finally:
            if old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old_appdata
