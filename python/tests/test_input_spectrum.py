"""The input analyser's spectrum API on the real extension."""

from __future__ import annotations

import numpy as np
import pytest

import mic_eq

pytestmark = pytest.mark.skipif(not mic_eq.CORE_AVAILABLE, reason="native extension is not built")


@pytest.fixture
def processor():
    from mic_eq import AudioProcessor

    processor = AudioProcessor()
    try:
        yield processor
    finally:
        processor.stop()


def test_there_are_96_bands(processor):
    from mic_eq.mic_eq_core import AudioProcessor

    assert processor.input_spectrum_bin_count() == 96
    assert AudioProcessor.input_spectrum_bin_count() == 96


def test_bin_frequencies_rise_from_about_20_hz_to_about_20_khz(processor):
    frequencies = np.zeros(processor.input_spectrum_bin_count(), dtype=np.float32)

    assert processor.input_spectrum_bin_frequencies(frequencies) is None

    assert np.all(np.diff(frequencies) > 0)
    assert 20.0 <= frequencies[0] < 22.0
    assert 18_000.0 < frequencies[-1] <= 20_000.0
    # Log-spaced: every band is the same ratio above the one before.
    ratios = frequencies[1:] / frequencies[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-4)


def test_zero_comes_back_before_a_frame_exists(processor):
    out = np.full(96, 5.0, dtype=np.float32)

    assert processor.read_input_spectrum(out) == 0
    assert np.all(out == -120.0)


@pytest.mark.parametrize(
    "out",
    [
        np.zeros(96, dtype=np.float64),
        np.zeros(96, dtype=np.int32),
        np.zeros((96, 1), dtype=np.float32),
        [0.0] * 96,
    ],
    ids=["float64", "int32", "2-D", "list"],
)
def test_another_type_dtype_or_shape_is_a_type_error(processor, out):
    with pytest.raises(TypeError, match="1-D numpy float32 array of length 96"):
        processor.read_input_spectrum(out)
    with pytest.raises(TypeError, match="1-D numpy float32 array of length 96"):
        processor.input_spectrum_bin_frequencies(out)


def _read_only():
    out = np.zeros(96, dtype=np.float32)
    out.flags.writeable = False
    return out


@pytest.mark.parametrize(
    "out",
    [
        np.zeros(95, dtype=np.float32),
        np.zeros(97, dtype=np.float32),
        np.zeros(192, dtype=np.float32)[::2],
        _read_only(),
    ],
    ids=["short", "long", "strided", "read-only"],
)
def test_another_length_a_strided_or_a_read_only_array_is_a_value_error(processor, out):
    with pytest.raises(ValueError, match="writable, contiguous 1-D numpy float32 array of length 96"):
        processor.read_input_spectrum(out)
    with pytest.raises(ValueError, match="writable, contiguous 1-D numpy float32 array of length 96"):
        processor.input_spectrum_bin_frequencies(out)


def test_the_dropped_block_count_is_in_the_runtime_diagnostics(processor):
    assert processor.get_runtime_diagnostics()["input_spectrum_dropped_blocks"] == 0
