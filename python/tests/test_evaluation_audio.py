"""Regression tests for shared evaluation WAV conversion."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mic_eq.analysis.wav_io import _read_wav, pcm_to_float_mono, read_mono_wav

# Bytes 4..16 of the KSDATAFORMAT_SUBTYPE GUID that follows the format code in an extensible chunk.
GUID_TAIL = bytes.fromhex("00001000800000aa00389b71")


def _wav(
    path: Path,
    payload: bytes,
    *,
    tag: int = 1,
    channels: int = 1,
    bits: int = 16,
    extensible: bool = False,
    before_data: bytes = b"",
    after_data: bytes = b"",
    block: int | None = None,
    guid_tail: bytes = GUID_TAIL,
    data_size: int | None = None,
    magic: bytes = b"RIFF",
) -> Path:
    block = channels * bits // 8 if block is None else block
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, 8000, 8000 * block, block, bits)
    if extensible:
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<I", tag) + guid_tail
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + before_data
    size = len(payload) if data_size is None else data_size
    chunks += b"data" + struct.pack("<I", size) + payload + b"\x00" * (len(payload) & 1) + after_data
    path.write_bytes(magic + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
    return path


@pytest.mark.parametrize(
    ("kwargs", "samples", "expected"),
    [
        ({"bits": 8}, np.array([0, 128, 255], np.uint8), [-1.0, 0.0, 127 / 128]),
        ({"bits": 16}, np.array([-32768, 0, 16384], "<i2"), [-1.0, 0.0, 0.5]),
        ({"bits": 32}, np.array([-(2**31), 2**30], "<i4"), [-1.0, 0.5]),
        ({"tag": 3, "bits": 32}, np.array([-1.5, 0.25], "<f4"), [-1.5, 0.25]),
        ({"tag": 3, "bits": 64}, np.array([0.125, -0.75], "<f8"), [0.125, -0.75]),
        ({"tag": 3, "bits": 32, "extensible": True}, np.array([0.5, -0.25], "<f4"), [0.5, -0.25]),
        ({"bits": 16, "extensible": True}, np.array([8192, -8192], "<i2"), [0.25, -0.25]),
    ],
)
def test_read_mono_wav_normalizes_every_sample_format(tmp_path, kwargs, samples, expected) -> None:
    path = _wav(tmp_path / "a.wav", samples.tobytes(), **kwargs)

    rate, audio = read_mono_wav(path, dtype=np.float64)

    assert rate == 8000
    assert audio.tolist() == pytest.approx(expected)


def test_read_mono_wav_decodes_24_bit_pcm(tmp_path) -> None:
    values = [-(2**23), 0, 2**22, 2**23 - 1]
    payload = b"".join(v.to_bytes(3, "little", signed=True) for v in values)

    _, audio = read_mono_wav(_wav(tmp_path / "a.wav", payload, bits=24), dtype=np.float64)

    assert audio.tolist() == pytest.approx([-1.0, 0.0, 0.5, (2**23 - 1) / 2**23], abs=1e-6)


def test_read_mono_wav_averages_channels_and_skips_other_chunks(tmp_path) -> None:
    stereo = np.array([[1000, 3000], [-2000, -4000]], "<i2")
    # An odd-sized chunk before the data is padded to an even length.
    path = _wav(tmp_path / "a.wav", stereo.tobytes(), channels=2, before_data=b"LIST\x03\x00\x00\x00abc\x00")

    _, audio = read_mono_wav(path, dtype=np.float64)
    assert audio.tolist() == pytest.approx([2000 / 32768, -3000 / 32768])
    with pytest.raises(ValueError, match="mono"):
        read_mono_wav(path, allow_stereo=False)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"not a wav file at all", "RIFF"),
        (b"RIFF\x04\x00\x00\x00WAVE", "no data chunk"),
    ],
)
def test_read_mono_wav_rejects_malformed_files(tmp_path, content, match) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(content)

    with pytest.raises(ValueError, match=match):
        read_mono_wav(path)


def test_read_mono_wav_rejects_truncated_and_unsupported_data(tmp_path) -> None:
    good = _wav(tmp_path / "good.wav", bytes(8)).read_bytes()
    truncated = tmp_path / "truncated.wav"
    truncated.write_bytes(good[:-2])
    with pytest.raises(ValueError, match="truncated"):
        read_mono_wav(truncated)
    # One byte past the end of the file, with no pad byte to hide it.
    with pytest.raises(ValueError, match="truncated"):
        read_mono_wav(_wav(tmp_path / "long.wav", bytes(10), bits=8, data_size=11))
    with pytest.raises(ValueError, match="unsupported WAV format"):
        read_mono_wav(_wav(tmp_path / "adpcm.wav", bytes(8), tag=2, bits=16))


class _RecordedReads:
    """File proxy that logs read sizes and never serves more than 1 MiB per read."""

    def __init__(self, stream, sizes: list[int]) -> None:
        self._stream, self._sizes = stream, sizes

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._stream.close()

    def read(self, size: int = -1) -> bytes:
        self._sizes.append(size)
        return self._stream.read(min(size, 1 << 20) if size >= 0 else size)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


@pytest.mark.parametrize("chunk", [b"fmt ", b"data"])
def test_read_wav_checks_a_declared_chunk_size_before_reading_it(tmp_path, monkeypatch, chunk) -> None:
    good = _wav(tmp_path / "good.wav", bytes(8)).read_bytes()
    at = good.index(chunk)
    path = tmp_path / "huge.wav"
    path.write_bytes(good[: at + 4] + struct.pack("<I", 0xFFFFFFF0) + good[at + 8 :])
    sizes: list[int] = []
    monkeypatch.setattr(
        "mic_eq.analysis.wav_io.open",
        lambda *args: _RecordedReads(open(*args), sizes),
        raising=False,
    )

    with pytest.raises(ValueError, match="truncated"):
        _read_wav(path)

    assert max(sizes) <= path.stat().st_size


def test_read_wav_rejects_a_format_chunk_cut_short_by_the_file_end(tmp_path) -> None:
    good = _wav(tmp_path / "good.wav", bytes(8)).read_bytes()
    path = tmp_path / "cut.wav"
    path.write_bytes(good[: good.index(b"fmt ") + 8 + 10])
    with pytest.raises(ValueError, match="truncated format"):
        _read_wav(path)


ODD_CHUNK = b"LIST\x03\x00\x00\x00abc\x00"


def _random_payload(bits: int, tag: int, samples: int) -> bytes:
    rng = np.random.default_rng(bits + tag)
    if tag == 3:
        return rng.normal(size=samples).astype(f"<f{bits // 8}").tobytes()
    return rng.integers(0, 256, samples * bits // 8, dtype=np.uint8).tobytes()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 8},
        {"bits": 8, "before_data": ODD_CHUNK, "after_data": ODD_CHUNK},
        {"bits": 16, "channels": 2},
        {"bits": 24},
        {"bits": 24, "channels": 2, "before_data": ODD_CHUNK},
        {"bits": 32, "channels": 2},
        {"tag": 3, "bits": 32},
        {"tag": 3, "bits": 64, "channels": 2, "after_data": ODD_CHUNK},
        {"bits": 16, "extensible": True},
        {"bits": 24, "channels": 2, "extensible": True},
        {"tag": 3, "bits": 32, "channels": 2, "extensible": True},
    ],
)
def test_read_wav_matches_scipy_on_the_variants_both_accept(tmp_path, kwargs) -> None:
    wavfile = pytest.importorskip("scipy.io.wavfile")
    channels, bits, tag = kwargs.get("channels", 1), kwargs["bits"], kwargs.get("tag", 1)
    path = _wav(tmp_path / "a.wav", _random_payload(bits, tag, 7 * channels), **kwargs)

    rate, audio = _read_wav(path)
    expected_rate, expected = wavfile.read(path)

    assert rate == expected_rate
    assert audio.dtype == expected.dtype
    np.testing.assert_array_equal(audio, expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"magic": b"RIFX"}, "not a RIFF WAV"),
        ({"magic": b"RF64"}, "not a RIFF WAV"),
        ({"bits": 64}, "unsupported WAV format 1 with 64 bits"),
        ({"tag": 7, "bits": 8}, "unsupported WAV format 7"),
        ({"data_size": 1000}, "truncated"),
        ({"data_size": 0xFFFFFFFF}, "truncated"),
        ({"bits": 20, "block": 3}, "inconsistent"),
        ({"bits": 24, "block": 4}, "inconsistent"),
        ({"channels": 0, "block": 0}, "inconsistent"),
        ({"extensible": True, "guid_tail": bytes(12)}, "non-standard extensible sub-format"),
        ({"extensible": True, "tag": 2}, "unsupported WAV format 2"),
    ],
)
def test_read_wav_rejects_what_scipy_repairs_or_misreads(tmp_path, kwargs, match) -> None:
    path = _wav(tmp_path / "a.wav", bytes(8), **kwargs)

    with pytest.raises(ValueError, match=match):
        _read_wav(path)


def test_read_wav_rejects_an_odd_trailing_byte_and_a_short_extensible_chunk(tmp_path) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        _read_wav(_wav(tmp_path / "odd.wav", bytes(9)))
    good = _wav(tmp_path / "ext.wav", bytes(8), extensible=True).read_bytes()
    at = good.index(b"fmt ")
    short = tmp_path / "short.wav"
    # The 40-byte format chunk cut to 36 bytes, so its sub-format GUID is incomplete.
    short.write_bytes(good[: at + 4] + struct.pack("<I", 36) + good[at + 8 : at + 44] + good[at + 48 :])
    with pytest.raises(ValueError, match="truncated extensible"):
        _read_wav(short)


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
