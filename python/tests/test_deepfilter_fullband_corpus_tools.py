"""Tests for the native-full-band DeepFilter corpus fetcher."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


TOOL_PATH = (
    Path(__file__).parent.parent / "tools" / "fetch_deepfilter_fullband_corpus.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fetch_deepfilter_fullband_corpus", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _wav_payload(sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, np.zeros(sample_rate, dtype=np.int16))
    return buffer.getvalue()


def test_selection_is_balanced_and_member_names_are_stable() -> None:
    speakers = [name.split("_", 1)[0] for name in TOOL.SELECTED_BASENAMES]

    assert speakers.count("p232") == 12
    assert speakers.count("p257") == 12
    assert (
        TOOL._member_name("clean", "p232_001.wav")
        == "clean_testset_wav/p232_001.wav"
    )


def test_wav_validation_requires_native_mono_48_khz() -> None:
    sample_rate, frames, peak = TOOL._validate_wav(
        _wav_payload(48_000), "valid.wav"
    )

    assert sample_rate == 48_000
    assert frames == 48_000
    assert peak == 0.0
    with pytest.raises(ValueError, match="expected native 48000 Hz"):
        TOOL._validate_wav(_wav_payload(16_000), "narrowband.wav")
