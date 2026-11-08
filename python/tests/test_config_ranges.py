"""The processing-settings ranges have one source: the native core's table."""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from dataclasses import asdict

import pytest

from mic_eq import mic_eq_core
from mic_eq.config import (
    CompressorSettings,
    DeEsserSettings,
    GateSettings,
    LimiterSettings,
    Preset,
    PresetValidationError,
    RNNoiseSettings,
)
from mic_eq.config_parts.validation import noise_model_ids, validation_ranges

# The ranges as they were saved by every release before the table moved to the
# native core. Presets and the controls depend on these exact values, so a
# change here is a change to what a saved preset accepts.
PINNED_NUMERIC_RANGES = {
    "gate": {
        "threshold_db": (-80.0, -10.0),
        "attack_ms": (0.1, 100.0),
        "release_ms": (10.0, 1000.0),
        "gate_mode": (0, 2),
        "vad_threshold": (0.3, 0.7),
        "vad_hold_time_ms": (0.0, 500.0),
        "vad_pre_gain": (1.0, 10.0),
        "gate_margin_db": (0.0, 20.0),
    },
    "rnnoise": {
        "strength": (0.0, 1.0),
    },
    "deesser": {
        "auto_amount": (0.0, 1.0),
        "low_cut_hz": (2000.0, 12000.0),
        "high_cut_hz": (2200.0, 16000.0),
        "threshold_db": (-60.0, -6.0),
        "ratio": (1.0, 20.0),
        "attack_ms": (0.1, 50.0),
        "release_ms": (5.0, 500.0),
        "max_reduction_db": (0.0, 24.0),
    },
    "compressor": {
        "threshold_db": (-60.0, 0.0),
        "ratio": (1.0, 20.0),
        "attack_ms": (0.1, 100.0),
        "release_ms": (10.0, 1000.0),
        "makeup_gain_db": (0.0, 24.0),
        "base_release_ms": (20.0, 200.0),
        "target_lufs": (-24.0, -12.0),
    },
    "limiter": {
        "ceiling_db": (-12.0, 0.0),
        "release_ms": (10.0, 500.0),
    },
}
PINNED_NOISE_MODELS = ["rnnoise", "deepfilter-ll", "deepfilter"]

# The persisted settings classes, which the native table does not describe.
SETTINGS_CLASSES = {
    "gate": GateSettings,
    "rnnoise": RNNoiseSettings,
    "deesser": DeEsserSettings,
    "compressor": CompressorSettings,
    "limiter": LimiterSettings,
}


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def test_the_native_table_holds_every_pinned_range_bit_for_bit():
    native = mic_eq_core.processing_ranges()

    assert {section: sorted(ranges) for section, ranges in native.items()} == {
        section: sorted(ranges) for section, ranges in PINNED_NUMERIC_RANGES.items()
    }
    for section, ranges in PINNED_NUMERIC_RANGES.items():
        for name, (low, high) in ranges.items():
            native_low, native_high = native[section][name]
            assert _bits(native_low) == _bits(float(low)), (section, name)
            assert _bits(native_high) == _bits(float(high)), (section, name)


def test_the_native_noise_model_ids_are_the_pinned_ones():
    assert mic_eq_core.noise_model_ids() == PINNED_NOISE_MODELS


def test_the_validation_table_is_the_native_table():
    native = mic_eq_core.processing_ranges()

    assert validation_ranges() == native
    assert list(noise_model_ids()) == mic_eq_core.noise_model_ids()


def test_the_shared_noise_model_ids_cannot_be_changed_by_a_caller():
    ids = noise_model_ids()
    assert isinstance(ids, tuple)
    assert noise_model_ids() is ids


def test_the_shared_table_cannot_be_mutated():
    ranges = validation_ranges()

    with pytest.raises(TypeError):
        ranges["gate"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        ranges["gate"]["threshold_db"] = (-200.0, 0.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        del ranges["limiter"]  # type: ignore[attr-defined]
    assert validation_ranges()["gate"]["threshold_db"] == (-80.0, -10.0)


@pytest.mark.parametrize("section", SETTINGS_CLASSES)
def test_every_default_lies_inside_its_native_range(section):
    native = mic_eq_core.processing_ranges()[section]

    for name, value in asdict(SETTINGS_CLASSES[section]()).items():
        if name in native:
            low, high = native[name]
            assert low <= value <= high, (section, name)


# A range's end only holds when the other bound it is checked against allows it.
COMPANIONS = {
    ("deesser", "low_cut_hz", "max"): {"high_cut_hz": 16000.0},
    ("deesser", "high_cut_hz", "min"): {"low_cut_hz": 2000.0},
}
PRESET_RANGES = [
    (section, name, bounds)
    for section, ranges in PINNED_NUMERIC_RANGES.items()
    for name, bounds in ranges.items()
]


def _preset_with(section: str, name: str, value: float, end: str) -> Preset:
    data = Preset(name="Ranges").to_dict()
    data[section].update(COMPANIONS.get((section, name, end), {}))
    data[section][name] = value
    return Preset.from_dict(data)


@pytest.mark.parametrize(("section", "name", "bounds"), PRESET_RANGES)
def test_a_saved_preset_accepts_each_end_of_a_range_and_nothing_beyond(
    section, name, bounds
):
    low, high = (float(bound) for bound in bounds)

    for end, value in (("min", low), ("max", high)):
        accepted = asdict(getattr(_preset_with(section, name, value, end), section))
        assert accepted[name] == value
    for end, value in (
        ("min", math.nextafter(low, -math.inf)),
        ("max", math.nextafter(high, math.inf)),
    ):
        with pytest.raises(PresetValidationError, match=f"Invalid {name} in {section}"):
            _preset_with(section, name, value, end)


def test_a_saved_preset_rejects_a_noise_model_outside_the_native_ids():
    data = Preset(name="Ranges").to_dict()
    data["rnnoise"]["model"] = "dpdfnet"

    with pytest.raises(PresetValidationError, match="Invalid model in rnnoise") as refused:
        Preset.from_dict(data)
    assert str(refused.value).endswith(f"(must be one of {list(PINNED_NOISE_MODELS)})")
    for model in PINNED_NOISE_MODELS:
        data["rnnoise"]["model"] = model
        assert Preset.from_dict(data).rnnoise.model == model


def test_config_imports_without_the_native_core_and_validation_names_the_missing_core():
    script = textwrap.dedent(
        """
        import sys

        sys.modules["mic_eq.mic_eq_core"] = None  # makes the import raise ImportError
        import mic_eq
        import mic_eq.config as config

        assert not mic_eq.CORE_AVAILABLE
        data = config.Preset(name="No core").to_dict()
        try:
            config.Preset.from_dict(data)
        except ImportError as error:
            print(error)
        else:
            raise SystemExit("validation ran without the native core")
        """
    )
    package_root = Path(mic_eq_core.__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(package_root)},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "mic_eq_core is unavailable" in result.stdout
    assert "maturin develop" in result.stdout
