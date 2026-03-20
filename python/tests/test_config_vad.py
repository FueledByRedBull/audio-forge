import os

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
            version="1.7.12",
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
    assert loaded.gate.vad_threshold == 0.4
    assert loaded.gate.vad_hold_time_ms == 200.0
    assert loaded.gate.vad_pre_gain == 1.0
    assert loaded.gate.auto_threshold_enabled is False
    assert loaded.gate.gate_margin_db == 10.0
