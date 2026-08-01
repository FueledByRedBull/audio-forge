"""Tests for deterministic RAVDESS repeated-take corpus acquisition."""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


TOOL_PATH = Path(__file__).parents[1] / "tools" / "fetch_cross_take_corpus.py"
SPEC = importlib.util.spec_from_file_location("fetch_cross_take_corpus", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _wav_payload(frequency_hz: float, *, stereo: bool = False) -> bytes:
    time = np.arange(4_800, dtype=np.float64) / 48_000.0
    audio = np.asarray(
        np.rint(8_000.0 * np.sin(2.0 * np.pi * frequency_hz * time)),
        dtype=np.int16,
    )
    if stereo:
        audio = np.column_stack((audio, audio // 2))
    payload = io.BytesIO()
    wavfile.write(payload, 48_000, audio)
    return payload.getvalue()


def test_member_parser_selects_only_neutral_audio_speech() -> None:
    selected = TOOL._parse_member("Actor_01/03-01-01-01-02-02-01.wav")

    assert selected == {
        "name": "03-01-01-01-02-02-01.wav",
        "statement": "02",
        "repetition": "02",
        "actor": "01",
        "emotion": "01",
        "intensity": "01",
    }
    assert TOOL._parse_member("Actor_01/03-01-05-01-02-02-01.wav") is None
    assert TOOL._parse_member("Actor_01/01-01-01-01-02-02-01.wav") is None
    assert TOOL._parse_member("Actor_01/03-01-02-02-02-02-01.wav") is not None


def test_archive_download_url_is_pinned_to_zenodo(monkeypatch) -> None:
    assert TOOL._validated_archive_url() == TOOL.ARCHIVE_URL
    monkeypatch.setattr(TOOL, "ARCHIVE_URL", "https://example.invalid/archive.zip")
    with pytest.raises(ValueError, match="trusted Zenodo HTTPS"):
        TOOL._validated_archive_url()


def test_extract_freezes_complete_repeated_take_pairs_and_hashes(tmp_path) -> None:
    archive_path = tmp_path / "ravdess.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for actor in (1, 2):
            for emotion, intensity in TOOL.EMOTIONS_AND_INTENSITIES:
                for statement in ("01", "02"):
                    for repetition in ("01", "02"):
                        name = (
                            f"Actor_{actor:02d}/03-01-{emotion}-{intensity}-"
                            f"{statement}-{repetition}-{actor:02d}.wav"
                        )
                        archive.writestr(
                            name,
                            _wav_payload(
                                200.0 + actor * 20.0,
                                stereo=(
                                    actor == 1
                                    and emotion == "02"
                                    and intensity == "01"
                                ),
                            ),
                        )
        archive.writestr(
            "Actor_01/03-01-05-01-01-01-01.wav",
            _wav_payload(900.0),
        )

    manifest = TOOL.extract(
        archive_path,
        tmp_path / "corpus",
        actors=(1, 2),
    )

    assert manifest["selection"]["pair_count"] == 12
    assert manifest["speaker_disjoint_splits"] == {
        "fixture": ["actor-01", "actor-02"]
    }
    assert all(set(pair["takes"]) == {"01", "02"} for pair in manifest["pairs"])
    assert all(
        len(take["sha256"]) == 64
        for pair in manifest["pairs"]
        for take in pair["takes"].values()
    )
    assert any(
        take["source_channels"] == 2
        for pair in manifest["pairs"]
        for take in pair["takes"].values()
    )
