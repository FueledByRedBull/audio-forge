import os

import pytest

from mic_eq import config
from mic_eq.config import GateSettings, Preset, load_preset, save_preset


def test_vad_preset_persistence(tmp_path):
    appdata_dir = tmp_path / "appdata"
    appdata_dir.mkdir(parents=True, exist_ok=True)
    old_appdata = os.environ.get("APPDATA")
    os.environ["APPDATA"] = str(appdata_dir)
    try:
        test_file = config.get_presets_dir() / "test_vad_preset.json"

        original = Preset(
            name="VAD Test",
            version="1.10.0",
            gate=GateSettings(
                enabled=True,
                threshold_db=-35.0,
                attack_ms=5.0,
                release_ms=50.0,
                gate_mode=1,  # VAD Assisted
                vad_threshold=0.4,
                vad_hold_time_ms=150.0,
                vad_pre_gain=2.5,
                auto_threshold_enabled=True,
                gate_margin_db=8.0,
            ),
        )

        save_preset(original, test_file)
        loaded = load_preset(test_file)

        assert loaded.gate.gate_mode == 1
        assert loaded.gate.vad_threshold == 0.4
        assert loaded.gate.vad_hold_time_ms == 150.0
        assert loaded.gate.vad_pre_gain == 2.5
        assert loaded.gate.auto_threshold_enabled is True
        assert loaded.gate.gate_margin_db == 8.0
    finally:
        if old_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = old_appdata


def test_backward_compatibility_defaults():
    old_preset_data = {
        "name": "Old Preset",
        "version": "1.1.0",
        "gate": {
            "enabled": True,
            "threshold_db": -40.0,
            "attack_ms": 10.0,
            "release_ms": 100.0,
        },
    }

    loaded = Preset.from_dict(old_preset_data)

    assert loaded.gate.gate_mode == 0
    assert loaded.gate.vad_threshold == 0.48
    assert loaded.gate.vad_hold_time_ms == 200.0
    assert loaded.gate.vad_pre_gain == 1.0
    assert loaded.gate.auto_threshold_enabled is True
    assert loaded.gate.gate_margin_db == 10.0


@pytest.mark.parametrize(
    ("version", "gate", "provenance", "threshold", "recorded"),
    [
        (
            "1.5.0",
            {
                "enabled": True,
                "threshold_db": -40.0,
                "attack_ms": 10.0,
                "release_ms": 100.0,
                "vad_threshold": 0.5,
            },
            None,
            0.5,
            "explicit",
        ),
        ("1.9.0", {"vad_threshold": 0.4}, None, 0.4, "explicit"),
        ("1.9.0", {"vad_threshold": 0.4}, {"gate.vad_threshold": "migration_default"}, 0.48, "migration_default"),
        ("1.9.0", {"vad_threshold": 0.43}, None, 0.43, "explicit"),
    ],
    ids=["legacy-explicit", "v19-explicit", "known-migration-default", "nondefault-v19"],
)
def test_migration_keeps_explicit_vad_values_and_moves_the_known_default(
    version, gate, provenance, threshold, recorded
):
    payload = {"name": "VAD threshold migration", "version": version, "gate": gate}
    if provenance is not None:
        payload["value_provenance"] = provenance

    loaded = Preset.from_dict(payload)

    assert loaded.gate.vad_threshold == threshold
    assert loaded.value_provenance["gate.vad_threshold"] == recorded


def test_preset_migration_does_not_mutate_caller_payload():
    payload = {
        "name": "Legacy",
        "version": "1.5.0",
        "gate": {"vad_threshold": 0.5},
    }
    original = {
        "name": "Legacy",
        "version": "1.5.0",
        "gate": {"vad_threshold": 0.5},
    }

    Preset.from_dict(payload)

    assert payload == original


def test_newly_saved_preset_records_explicit_value_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    preset = Preset(name="Provenance", gate=GateSettings(vad_threshold=0.41))
    path = save_preset(preset)

    loaded = load_preset(path)

    assert loaded.gate.vad_threshold == 0.41
    assert loaded.value_provenance["gate.vad_threshold"] == "explicit"
    assert loaded.value_provenance["compressor.target_lufs"] == "explicit"


def test_audioforge_appdata_migrates_legacy_miceq_dir(tmp_path, monkeypatch):
    appdata_dir = tmp_path / "appdata"
    legacy_dir = appdata_dir / config.LEGACY_APPDATA_DIR_NAME
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.json").write_text('{"version":"1.8.2"}', encoding="utf-8")

    monkeypatch.setenv("APPDATA", str(appdata_dir))

    monkeypatch.setenv("XDG_CONFIG_HOME", str(appdata_dir))

    config_file = config.get_config_file()

    assert config_file == appdata_dir / config.APPDATA_DIR_NAME / "config.json"
    assert config.APPDATA_DIR_NAME == "AudioForge"
    assert config.LEGACY_APPDATA_DIR_NAME == "MicEq"
    assert config_file.read_text(encoding="utf-8") == '{"version":"1.8.2"}'
