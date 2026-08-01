"""Versioned EQ schema, migration, and immutability tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mic_eq.config import (
    EQBandSettings,
    EQSettings,
    EQ_BAND_COUNT,
    EQ_SCHEMA_VERSION,
    Preset,
    PresetValidationError,
    q_from_bandwidth_octaves,
)


def _legacy_preset() -> dict:
    payload = Preset(name="Legacy").to_dict()
    payload["eq"] = {
        "enabled": True,
        "band_freqs": [
            80.0,
            160.0,
            320.0,
            640.0,
            1280.0,
            2500.0,
            5000.0,
            8000.0,
            12000.0,
            16000.0,
        ],
        "band_gains": [float(index - 5) / 2.0 for index in range(10)],
        "band_qs": [1.0 + index * 0.1 for index in range(10)],
    }
    payload["value_provenance"] = {
        path: source
        for path, source in payload["value_provenance"].items()
        if not path.startswith("eq.")
    }
    payload["value_provenance"].update(
        {
            "eq.enabled": "explicit",
            "eq.band_freqs": "explicit",
            "eq.band_gains": "explicit",
            "eq.band_qs": "explicit",
        }
    )
    return payload


def test_legacy_eq_constructor_accepts_numpy_sequences() -> None:
    defaults = EQSettings()
    settings = EQSettings(
        band_freqs=np.asarray(defaults.band_freqs),
        band_gains=np.zeros(EQ_BAND_COUNT),
        band_qs=np.full(EQ_BAND_COUNT, 1.41),
    )

    assert settings.band_freqs == defaults.band_freqs


def test_legacy_eq_constructor_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="must contain 10 bands"):
        EQSettings(band_freqs=[], band_gains=[], band_qs=[])


def test_future_preset_version_is_rejected_before_lossy_loading() -> None:
    payload = Preset().to_dict()
    payload["version"] = "99.0.0"

    with pytest.raises(PresetValidationError, match="newer than this AudioForge"):
        Preset.from_dict(payload)


def test_legacy_arrays_migrate_to_one_typed_band_source_with_provenance() -> None:
    legacy = _legacy_preset()

    preset = Preset.from_dict(legacy)
    serialized = preset.to_dict()

    assert preset.eq.schema_version == EQ_SCHEMA_VERSION
    assert len(preset.eq.bands) == EQ_BAND_COUNT
    assert [band.filter_type for band in preset.eq.bands] == [
        "low_shelf",
        *(["bell"] * 8),
        "high_shelf",
    ]
    assert set(serialized["eq"]) == {"schema_version", "enabled", "bands"}
    assert "band_freqs" not in serialized["eq"]
    for index in range(EQ_BAND_COUNT):
        assert (
            preset.value_provenance[f"eq.bands.{index}.frequency_hz"]
            == "explicit"
        )
        assert (
            preset.value_provenance[f"eq.bands.{index}.gain_db"]
            == "explicit"
        )
        assert (
            preset.value_provenance[f"eq.bands.{index}.q"]
            == "explicit"
        )
        assert (
            preset.value_provenance[f"eq.bands.{index}.filter_type"]
            == "migration_default"
        )
        assert (
            preset.value_provenance[f"eq.bands.{index}.stage"]
            == "migration_default"
        )
    assert not any(
        path.startswith("eq.band_") for path in preset.value_provenance
    )


def test_all_filter_contract_fields_round_trip_exactly() -> None:
    base = EQSettings()
    types = (
        "low_shelf",
        "bell",
        "notch",
        "high_pass",
        "low_pass",
        "bell",
        "notch",
        "bell",
        "bell",
        "high_shelf",
    )
    bands = tuple(
        replace(
            band,
            filter_type=types[index],
            bandwidth_mode="octaves" if index == 2 else "q",
            bandwidth_octaves=1.25 if index == 2 else None,
            q=(
                q_from_bandwidth_octaves(band.frequency_hz, 1.25)
                if index == 2
                else band.q
            ),
            slope_db_per_octave=24 if index in {0, 3, 4, 9} else 12,
            stage="combined",
            enabled=index != 7,
        )
        for index, band in enumerate(base.bands)
    )
    preset = Preset(name="Typed", eq=EQSettings(bands=bands))

    restored = Preset.from_dict(preset.to_dict())

    assert restored.eq == preset.eq
    assert restored.to_dict()["eq"] == preset.to_dict()["eq"]


def test_bands_are_immutable_and_legacy_views_do_not_alias() -> None:
    settings = EQSettings()
    gains = [float(index) / 10.0 for index in range(EQ_BAND_COUNT)]

    settings.band_gains = gains
    gains[0] = 9.0
    returned = settings.band_gains
    returned[1] = 9.0

    assert settings.band_gains[0] == 0.0
    assert settings.band_gains[1] == 0.1
    with pytest.raises(FrozenInstanceError):
        settings.bands[0].gain_db = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda eq: eq.update({"unknown": True}),
            "unknown fields",
        ),
        (
            lambda eq: eq["bands"][0].update({"unknown": True}),
            "unknown fields",
        ),
        (
            lambda eq: eq.update({"schema_version": 999}),
            "unsupported EQ schema version",
        ),
        (
            lambda eq: eq["bands"][0].update({"filter_type": "magic"}),
            "unsupported EQ filter type",
        ),
        (
            lambda eq: eq["bands"][0].update({"enabled": "yes"}),
            "must be true or false",
        ),
        (
            lambda eq: eq.update({"bands": eq["bands"][:-1]}),
            "10 typed bands",
        ),
    ],
)
def test_unknown_or_malformed_eq_schema_is_rejected(
    mutator,
    message: str,
) -> None:
    payload = Preset(name="Malformed").to_dict()
    mutator(payload["eq"])

    with pytest.raises(PresetValidationError, match=message):
        Preset.from_dict(payload)


def test_eq_band_constructor_rejects_incoherent_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth_octaves is required"):
        EQBandSettings(
            filter_type="bell",
            frequency_hz=1000.0,
            gain_db=0.0,
            q=1.41,
            bandwidth_mode="octaves",
            bandwidth_octaves=None,
        )


@pytest.mark.parametrize("stage", ["correction", "tone"])
def test_unimplemented_eq_stage_ownership_is_rejected(stage: str) -> None:
    with pytest.raises(ValueError, match="unsupported EQ stage"):
        EQBandSettings(
            filter_type="bell",
            frequency_hz=1000.0,
            gain_db=0.0,
            q=1.41,
            stage=stage,
        )


def test_eq_band_constructor_rejects_conflicting_q_and_octave_sources() -> None:
    with pytest.raises(ValueError, match="q must match"):
        EQBandSettings(
            filter_type="notch",
            frequency_hz=1000.0,
            gain_db=0.0,
            q=1.41,
            bandwidth_mode="octaves",
            bandwidth_octaves=1.0,
        )


def test_eq_band_constructor_rejects_octave_mode_for_non_band_filters() -> None:
    q = q_from_bandwidth_octaves(1000.0, 1.0)
    with pytest.raises(ValueError, match="only for bell and notch"):
        EQBandSettings(
            filter_type="high_pass",
            frequency_hz=1000.0,
            gain_db=0.0,
            q=q,
            bandwidth_mode="octaves",
            bandwidth_octaves=1.0,
        )


def test_q_mode_rejects_redundant_octave_bandwidth() -> None:
    with pytest.raises(ValueError, match="must be null"):
        EQBandSettings(
            filter_type="bell",
            frequency_hz=1000.0,
            gain_db=0.0,
            q=1.41,
            bandwidth_mode="q",
            bandwidth_octaves=1.0,
        )
