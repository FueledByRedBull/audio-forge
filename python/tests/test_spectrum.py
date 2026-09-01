"""Tests for spectrum analysis helpers used by Auto-EQ."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPECTRUM_PATH = Path(__file__).parent.parent / "mic_eq" / "analysis" / "spectrum.py"
spectrum_spec = importlib.util.spec_from_file_location(
    "mic_eq.analysis.spectrum", SPECTRUM_PATH
)
assert spectrum_spec is not None and spectrum_spec.loader is not None
spectrum = importlib.util.module_from_spec(spectrum_spec)
spectrum_spec.loader.exec_module(spectrum)

compute_voice_spectrum = spectrum.compute_voice_spectrum
analyze_voice_spectrum = spectrum.analyze_voice_spectrum
evaluate_spectrum_estimators = spectrum.evaluate_spectrum_estimators
find_octave_spaced_peaks = spectrum.find_octave_spaced_peaks
smooth_spectrum_octave = spectrum.smooth_spectrum_octave
smooth_spectrum_perceptual = spectrum.smooth_spectrum_perceptual
_window_spectrum_db = spectrum._window_spectrum_db
_measurement_reliability = spectrum._measurement_reliability


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


def test_window_spectrum_is_invariant_to_dc_offset():
    fs = 48_000
    n = 4096
    t = np.arange(n, dtype=float) / fs
    tone = 0.1 * np.sin(2.0 * np.pi * 750.0 * t)

    freqs, centered = _window_spectrum_db(tone, fs)
    offset_freqs, offset = _window_spectrum_db(tone + 0.35, fs)

    assert np.array_equal(freqs, offset_freqs)
    assert np.allclose(centered, offset, atol=1e-8)


def test_explicit_noise_capture_produces_frequency_dependent_snr():
    fs = 48_000
    seconds = 4.0
    rng = np.random.default_rng(4402)
    t = np.arange(int(fs * seconds), dtype=float) / fs
    noise = 0.0015 * rng.normal(size=t.size)
    speech_gate = ((np.floor(t * 2.0) % 2.0) == 0.0).astype(float)
    speech = noise + speech_gate * (
        0.05 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.025 * np.sin(2.0 * np.pi * 900.0 * t)
    )

    result = analyze_voice_spectrum(
        speech.astype(np.float32),
        fs,
        noise_audio=noise.astype(np.float32),
    )

    assert result.noise_reference_source == "explicit_capture"
    assert result.spectral_snr_db is not None
    assert result.noise_spectrum_db is not None
    assert result.spectral_snr_db.shape == result.freqs.shape
    assert result.snr_db > 10.0


def test_sparse_fallback_snr_uses_the_same_power_scale_as_noise_reference():
    fs = 48_000
    seconds = 4
    rng = np.random.default_rng(90210)
    t = np.arange(fs * seconds, dtype=float) / fs
    noise = 0.002 * rng.normal(size=t.size)
    voiced = 0.05 * np.sin(2.0 * np.pi * 220.0 * t)
    sparse = noise.copy()
    burst_samples = fs // 4
    sparse[fs : fs + burst_samples] += voiced[fs : fs + burst_samples]
    vad = np.zeros(int(np.ceil(sparse.size / 1536)), dtype=float)
    vad[30:40] = 1.0

    sparse_result = analyze_voice_spectrum(
        sparse.astype(np.float32),
        fs,
        vad_probabilities=vad,
        noise_audio=noise.astype(np.float32),
    )

    assert sparse_result.used_single_spectrum_fallback is True
    assert sparse_result.noise_reference_source == "explicit_capture"
    assert sparse_result.spectral_snr_db is not None
    assert sparse_result.noise_spectrum_db is not None
    mask = (sparse_result.freqs >= 80.0) & (sparse_result.freqs <= 8000.0)
    noise_power = np.power(10.0, sparse_result.noise_spectrum_db[mask] / 10.0)
    signal_power = noise_power * np.power(
        10.0, sparse_result.spectral_snr_db[mask] / 10.0
    )
    matched_snr = 10.0 * np.log10(np.sum(signal_power) / np.sum(noise_power))
    assert sparse_result.snr_db == pytest.approx(matched_snr, abs=1e-3)


def test_missing_noise_reference_is_reported_as_unavailable():
    fs = 48_000
    t = np.arange(fs * 3, dtype=float) / fs
    continuous = 0.05 * np.sin(2.0 * np.pi * 180.0 * t)

    result = analyze_voice_spectrum(continuous.astype(np.float32), fs)

    assert result.noise_reference_source == "unavailable"
    assert result.spectral_snr_db is None
    assert result.noise_spectrum_db is None
    assert result.residual_confidence <= 0.70


def test_measurement_uncertainty_rewards_longer_independent_evidence():
    rng = np.random.default_rng(9917)
    freqs = np.geomspace(80.0, 10_000.0, 192)
    base = -18.0 - 3.0 * np.log2(freqs / 1000.0)

    def make_rows(count: int) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        for _ in range(count):
            broad_noise = np.interp(
                np.linspace(0.0, 1.0, freqs.size),
                np.linspace(0.0, 1.0, 18),
                rng.normal(0.0, 2.0, 18),
            )
            rows.append(base + broad_noise)
        starts = np.arange(count, dtype=int) * 4096
        return np.asarray(rows), starts

    short_rows, short_starts = make_rows(6)
    long_rows, long_starts = make_rows(36)
    _, _, short_uncertainty, _, short_blocks = _measurement_reliability(
        freqs,
        short_rows,
        short_starts,
        4096,
    )
    _, _, long_uncertainty, _, long_blocks = _measurement_reliability(
        freqs,
        long_rows,
        long_starts,
        4096,
    )

    assert long_blocks > short_blocks
    assert np.median(long_uncertainty) < np.median(short_uncertainty)


def test_phonetic_coverage_is_separate_from_precise_homogeneous_capture():
    freqs = np.geomspace(80.0, 10_000.0, 192)
    base = -18.0 - 3.0 * np.log2(freqs / 1000.0)
    starts = np.arange(36, dtype=int) * 4096
    homogeneous = np.tile(base, (36, 1))
    diverse = homogeneous.copy()
    log_freqs = np.log2(freqs)
    for index, row in enumerate(diverse):
        centre = np.log2((180.0, 600.0, 1800.0, 5200.0)[index % 4])
        row += 8.0 * np.exp(-0.5 * np.square((log_freqs - centre) / 0.38))

    homogeneous_reliability, _, homogeneous_uncertainty, homogeneous_coverage, _ = (
        _measurement_reliability(freqs, homogeneous, starts, 4096)
    )
    _, _, _, diverse_coverage, _ = _measurement_reliability(
        freqs,
        diverse,
        starts,
        4096,
    )

    assert np.median(homogeneous_uncertainty) < 0.5
    assert np.median(homogeneous_reliability) > 0.95
    assert homogeneous_coverage < 0.05
    assert diverse_coverage > homogeneous_coverage + 0.35


def test_find_octave_spaced_peaks_handles_degenerate_frequency_grids():
    for freqs in (
        np.array([0.0]),
        np.array([100.0]),
        np.array([100.0, 100.0]),
    ):
        peak_freqs, peak_values = find_octave_spaced_peaks(np.zeros(len(freqs)), freqs)
        assert peak_freqs.size == 0
        assert peak_values.size == 0


def test_perceptual_smoothing_off_is_exact_and_unknown_mode_fails():
    freqs = np.array([100.0, 200.0, 400.0])
    values = np.array([0.0, 12.0, -3.0])

    result = smooth_spectrum_perceptual(freqs, values, strength="off")

    assert np.array_equal(result, values)
    assert result is not values
    with pytest.raises(ValueError, match="Unknown spectrum smoothing strength"):
        smooth_spectrum_perceptual(freqs, values, strength="typo")


def test_octave_reconstruction_interpolates_on_log_frequency(monkeypatch):
    monkeypatch.setattr(
        spectrum,
        "get_octave_frequencies",
        lambda _fraction: (
            np.array([100.0, 400.0]),
            np.array([90.0, 350.0]),
            np.array([150.0, 450.0]),
        ),
    )
    freqs = np.array([100.0, 200.0, 400.0])
    values = np.array([0.0, 99.0, 10.0])

    result = smooth_spectrum_octave(freqs, values)

    assert result[1] == pytest.approx(5.0)


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
