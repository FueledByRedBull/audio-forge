"""Regression tests for shared evaluation WAV conversion."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eq.analysis.wav_io import pcm_to_float_mono


def test_signed_stereo_is_normalized_before_downmix() -> None:
    raw = np.asarray([[32767, 32767], [-32768, -32768]], dtype=np.int16)

    converted = pcm_to_float_mono(raw, dtype=np.float64)

    assert converted[0] == pytest.approx(32767.0 / 32768.0)
    assert converted[1] == pytest.approx(-1.0)


def test_unsigned_pcm_is_centered_around_zero() -> None:
    raw = np.asarray([0, 128, 255], dtype=np.uint8)

    converted = pcm_to_float_mono(raw, dtype=np.float64)

    assert converted.tolist() == pytest.approx([-1.0, 0.0, 127.0 / 128.0])


def test_nonfinite_float_audio_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        pcm_to_float_mono(np.asarray([0.0, np.nan], dtype=np.float32))


def test_mono_only_contract_rejects_stereo() -> None:
    with pytest.raises(ValueError, match="mono"):
        pcm_to_float_mono(
            np.zeros((2, 2), dtype=np.int16),
            allow_stereo=False,
        )
