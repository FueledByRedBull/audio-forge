"""Tests for preset/config v1.7 migration and latency profile persistence."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "mic_eq" / "config.py"
config_spec = importlib.util.spec_from_file_location("mic_eq.config", CONFIG_PATH)
config = importlib.util.module_from_spec(config_spec)
sys.modules["mic_eq.config"] = config
config_spec.loader.exec_module(config)


Preset = config.Preset
AppConfig = config.AppConfig
LatencyCalibrationProfile = config.LatencyCalibrationProfile


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

    assert preset.version == "1.7.0"
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

    cfg = AppConfig(
        last_input_device="Mic A",
        last_output_device="Out B",
        use_measured_latency=True,
        auto_eq_webrtc_apm_enabled=True,
        auto_eq_distill_mos_enabled=False,
        latency_calibration_profiles={"Mic A||Out B": profile},
    )

    raw = cfg.to_dict()
    restored = AppConfig.from_dict(raw)

    assert restored.use_measured_latency is True
    assert restored.auto_eq_webrtc_apm_enabled is True
    assert restored.auto_eq_distill_mos_enabled is False
    assert "Mic A||Out B" in restored.latency_calibration_profiles
    restored_profile = restored.latency_calibration_profiles["Mic A||Out B"]
    assert restored_profile.measured_round_trip_ms == 36.5
    assert restored_profile.estimated_one_way_ms == 18.25
    assert restored_profile.applied_compensation_ms == 18.25
    assert restored_profile.confidence == 0.92


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
