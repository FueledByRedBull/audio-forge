"""Reject remote metadata paths before reading or writing outside a corpus."""

from pathlib import Path

import pytest

import fetch_dpdfnet_evaluation_assets as fetcher


@pytest.mark.parametrize("name", ["../outside.wav", "sub/../../outside.wav",
                                 "C:/outside.wav", "C:outside.wav", "\\\\server\\share\\a.wav",
                                 "\\outside.wav"])
def test_remote_dataset_path_cannot_escape_corpus(monkeypatch, tmp_path: Path, name: str):
    row = {"model_name": "Clean", "language": "en", "noise_type": "room",
           "snr_db": "0", "file_name": name}
    monkeypatch.setattr(fetcher, "_load_metadata", lambda: ("", [row]))
    monkeypatch.setattr(fetcher, "MODEL_OUTPUTS", ("Clean",))
    monkeypatch.setattr(fetcher, "_download", lambda *_: pytest.fail("download must not run"))
    with pytest.raises(ValueError, match="within the corpus"):
        fetcher.fetch_dataset_subset(tmp_path)
