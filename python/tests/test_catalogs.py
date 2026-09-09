"""Regression checks for built-in preset catalog construction."""

from mic_eq.config import EQSettings, GateSettings, Preset, RNNoiseSettings
from mic_eq.config_parts.catalogs import build_builtin_presets


def test_builtin_presets_share_values_without_sharing_mutable_defaults() -> None:
    presets = build_builtin_presets(
        Preset,
        GateSettings,
        EQSettings,
        RNNoiseSettings,
    )

    assert {
        key: (
            preset.gate.threshold_db,
            preset.gate.attack_ms,
            preset.gate.release_ms,
        )
        for key, preset in presets.items()
    } == {
        "voice": (-40.0, 10.0, 100.0),
        "bass_cut": (-40.0, 10.0, 100.0),
        "presence": (-40.0, 10.0, 100.0),
        "flat": (-40.0, 10.0, 100.0),
        "minimal": (-45.0, 5.0, 150.0),
        "aggressive_denoise": (-35.0, 5.0, 50.0),
    }
    assert all(preset.rnnoise == RNNoiseSettings() for preset in presets.values())

    presets["voice"].gate.threshold_db = -1.0
    presets["voice"].rnnoise.model = "deepfilter"
    assert presets["bass_cut"].gate.threshold_db == -40.0
    assert presets["bass_cut"].rnnoise.model == "rnnoise"
