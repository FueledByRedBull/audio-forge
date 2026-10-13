from __future__ import annotations

from pathlib import Path

import numpy as np


import build_vad_evaluation_corpus as vad_corpus


def test_deterministic_noise_segment_repeats_and_removes_dc():
    noise = np.linspace(-1.0, 2.0, 10_000)

    first = vad_corpus._deterministic_noise_segment(noise, 2_000, "fixture")
    second = vad_corpus._deterministic_noise_segment(noise, 2_000, "fixture")

    np.testing.assert_array_equal(first, second)
    assert abs(float(np.mean(first))) < 1e-12


def test_pitch_up_shortens_waveform_without_non_finite_samples():
    source = np.sin(np.linspace(0.0, 100.0, 1_300))

    pitched = vad_corpus._pitch_up(source)

    assert pitched.size == 1_000
    assert np.isfinite(pitched).all()


def test_corpus_manifests_never_embed_machine_absolute_paths(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    external = tmp_path / "speaker.wav"
    external.write_bytes(b"fixture")

    monkeypatch.setattr(vad_corpus, "REPO_ROOT", repo_root)
    assert vad_corpus._portable_path(external) == "speaker.wav"
    assert vad_corpus._portable_path(repo_root / "speech" / "speaker.wav") == "speech/speaker.wav"
