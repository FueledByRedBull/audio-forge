"""Tests for spectrum analysis helpers used by Auto-EQ."""

import importlib.util
from pathlib import Path

import numpy as np


SPECTRUM_PATH = Path(__file__).parent.parent / "mic_eq" / "analysis" / "spectrum.py"
spectrum_spec = importlib.util.spec_from_file_location(
    "mic_eq.analysis.spectrum", SPECTRUM_PATH
)
spectrum = importlib.util.module_from_spec(spectrum_spec)
spectrum_spec.loader.exec_module(spectrum)

compute_voice_spectrum = spectrum.compute_voice_spectrum


def test_voiced_frame_selection_reduces_background_hum_bias():
    fs = 48_000
    duration_s = 2.0
    n = int(fs * duration_s)
    t = np.arange(n, dtype=float) / fs

    hum = 0.08 * np.sin(2.0 * np.pi * 60.0 * t)
    speech = np.zeros_like(hum)
    speech_start = int(1.5 * fs)
    speech_t = np.arange(n - speech_start, dtype=float) / fs
    speech[speech_start:] = 0.18 * np.sin(2.0 * np.pi * 1000.0 * speech_t)

    freqs, spectrum_db = compute_voice_spectrum(hum + speech, fs=fs, nperseg=2048)
    idx_60 = int(np.argmin(np.abs(freqs - 60.0)))
    idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))

    assert spectrum_db[idx_1k] > spectrum_db[idx_60] + 3.0


def test_compute_voice_spectrum_keeps_valid_output_when_voiced_mask_is_sparse():
    fs = 48_000
    nperseg = 2048
    audio = np.zeros(nperseg * 4, dtype=float)
    audio[-nperseg // 4:] = 0.005

    freqs, spectrum_db = compute_voice_spectrum(audio, fs=fs, nperseg=nperseg)
    assert freqs.shape == spectrum_db.shape
    assert len(freqs) > 0
