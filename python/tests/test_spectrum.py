"""Tests for spectrum analysis helpers used by Auto-EQ."""

import importlib.util
from pathlib import Path

import numpy as np


SPECTRUM_PATH = Path(__file__).parent.parent / "mic_eq" / "analysis" / "spectrum.py"
spectrum_spec = importlib.util.spec_from_file_location(
    "mic_eq.analysis.spectrum", SPECTRUM_PATH
)
assert spectrum_spec is not None and spectrum_spec.loader is not None
spectrum = importlib.util.module_from_spec(spectrum_spec)
spectrum_spec.loader.exec_module(spectrum)

compute_voice_spectrum = spectrum.compute_voice_spectrum
evaluate_spectrum_estimators = spectrum.evaluate_spectrum_estimators
find_octave_spaced_peaks = spectrum.find_octave_spaced_peaks


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


def test_find_octave_spaced_peaks_handles_degenerate_frequency_grids():
    for freqs in (
        np.array([0.0]),
        np.array([100.0]),
        np.array([100.0, 100.0]),
    ):
        peak_freqs, peak_values = find_octave_spaced_peaks(np.zeros(len(freqs)), freqs)
        assert peak_freqs.size == 0
        assert peak_values.size == 0


def test_multiresolution_experiment_keeps_welch_without_material_all_band_gain():
    fs = 48_000
    seconds = 3.0
    t = np.arange(int(fs * seconds), dtype=float) / fs
    fixtures = []
    rng = np.random.default_rng(7319)
    # Two labelled synthetic speakers, each captured at three microphone
    # positions with different reflection delay and high-frequency loss.
    for fundamental_hz, spectral_tilt in ((125.0, 0.82), (185.0, 0.72)):
        base = np.zeros_like(t)
        for harmonic in range(1, 48):
            frequency = fundamental_hz * harmonic
            if frequency >= 11_000.0:
                break
            base += (spectral_tilt**harmonic) * np.sin(2.0 * np.pi * frequency * t)
        envelope = 0.35 + 0.25 * np.sin(2.0 * np.pi * 2.3 * t) ** 2
        base *= envelope / max(float(np.max(np.abs(base))), 1e-9)
        for position, delay_samples in enumerate((7, 19, 41)):
            reflected = base + (0.18 - 0.035 * position) * np.roll(base, delay_samples)
            if position:
                reflected = np.convolve(
                    reflected,
                    np.ones(1 + position * 2) / (1 + position * 2),
                    mode="same",
                )
            fixtures.append(
                (0.16 * reflected + 0.0025 * rng.normal(size=t.size)).astype(np.float32)
            )

    evaluation = evaluate_spectrum_estimators(fixtures, fs=fs)

    assert set(evaluation.improvement_db) == {
        "low_frequency",
        "formant",
        "sibilance",
    }
    assert all(np.isfinite(value) for value in evaluation.improvement_db.values())
    assert evaluation.material_improvement is False
    assert evaluation.selected_estimator == "welch_hamming"
