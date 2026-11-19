"""Versioned EQ schema, migration, and immutability tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mic_eq.config import (
    EQBandSettings,
    EQSettings,
    EQ_BAND_COUNT,
    Preset,
    PresetValidationError,
    q_from_bandwidth_octaves,
)


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(bandwidth_mode="octaves", bandwidth_octaves=None), "bandwidth_octaves is required"),
        (dict(stage="correction"), "unsupported EQ stage"),
        (dict(stage="tone"), "unsupported EQ stage"),
        (dict(filter_type="notch", bandwidth_mode="octaves", bandwidth_octaves=1.0), "q must match"),
        (
            dict(
                filter_type="high_pass",
                q=q_from_bandwidth_octaves(1000.0, 1.0),
                bandwidth_mode="octaves",
                bandwidth_octaves=1.0,
            ),
            "only for bell and notch",
        ),
        (dict(bandwidth_mode="q", bandwidth_octaves=1.0), "must be null"),
    ],
    ids=["octave-mode-without-octaves", "correction-stage", "tone-stage", "conflicting-q-and-octaves",
         "octave-mode-for-non-band-filter", "q-mode-with-octaves"],
)
def test_eq_band_constructor_rejects_incoherent_bandwidth_or_stage(overrides: dict, message: str) -> None:
    band = dict(filter_type="bell", frequency_hz=1000.0, gain_db=0.0, q=1.41)
    with pytest.raises(ValueError, match=message):
        EQBandSettings(**{**band, **overrides})


def test_independent_stages_roundtrip():
    correction = tuple(replace(band, gain_db=-2.0) for band in EQSettings().bands)
    tone = list(EQSettings().bands)
    tone[0] = replace(tone[0], filter_type="high_pass", frequency_hz=120.0, slope_db_per_octave=48)
    settings = EQSettings(correction_bands=correction, tone_bands=tone)
    restored = EQSettings.from_dict(settings.to_dict())
    assert restored.correction_bands == correction
    assert restored.bands == tuple(tone)
    restored.band_freqs = [125.0, *restored.band_freqs[1:]]
    assert restored.correction_bands == correction
    assert EQSettings.from_dict(restored.to_dict()).band_freqs[0] == 125.0
