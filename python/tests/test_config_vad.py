import os


from mic_eq import config
from mic_eq.config import GateSettings, Preset, load_preset, save_preset
from mic_eq.config_parts import presets as presets_module
from mic_eq.config_parts.shared import OLDEST_OPENED_VERSION


def test_vad_preset_persistence(tmp_path):
    appdata_dir = tmp_path / "appdata"
    appdata_dir.mkdir(parents=True, exist_ok=True)
    old_appdata = os.environ.get("APPDATA")
    os.environ["APPDATA"] = str(appdata_dir)
    try:
        test_file = config.get_presets_dir() / "test_vad_preset.json"

        original = Preset(
            name="VAD Test",
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


def test_a_preset_from_the_oldest_opened_release_fills_missing_fields_with_defaults(
    monkeypatch,
):
    monkeypatch.setattr(presets_module, "CURRENT_VERSION", "1.99.0")
    older_preset_data = {
        "name": "Older Preset",
        "version": OLDEST_OPENED_VERSION,
        "gate": {
            "enabled": True,
            "threshold_db": -40.0,
            "attack_ms": 10.0,
            "release_ms": 100.0,
            "vad_threshold": 0.43,
        },
    }

    loaded = Preset.from_dict(older_preset_data)

    assert loaded.gate.gate_mode == 0
    assert loaded.gate.vad_threshold == 0.43
    assert loaded.gate.vad_hold_time_ms == 200.0
    assert loaded.gate.vad_pre_gain == 1.0
    assert loaded.gate.auto_threshold_enabled is True
    assert loaded.gate.gate_margin_db == 10.0
    assert loaded.value_provenance["gate.vad_threshold"] == "explicit"
    assert loaded.value_provenance["gate.gate_margin_db"] == "migration_default"


def test_loading_a_preset_does_not_mutate_caller_payload(monkeypatch):
    monkeypatch.setattr(presets_module, "CURRENT_VERSION", "1.99.0")
    payload = {
        "name": "Older",
        "version": OLDEST_OPENED_VERSION,
        "gate": {"vad_threshold": 0.5},
    }
    original = {
        "name": "Older",
        "version": OLDEST_OPENED_VERSION,
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
