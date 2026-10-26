"""Tests for spectrum analysis helpers used by Auto-EQ."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mic_eq.analysis.noise_reference import analyze_noise_reference


SPECTRUM_PATH = Path(__file__).parent.parent / "mic_eq" / "analysis" / "spectrum.py"
spectrum_spec = importlib.util.spec_from_file_location(
    "mic_eq.analysis.spectrum", SPECTRUM_PATH
)
assert spectrum_spec is not None and spectrum_spec.loader is not None
spectrum = importlib.util.module_from_spec(spectrum_spec)
spectrum_spec.loader.exec_module(spectrum)

compute_voice_spectrum = spectrum.compute_voice_spectrum
analyze_voice_spectrum = spectrum.analyze_voice_spectrum
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


@pytest.mark.parametrize(("signal_scale", "expected_snr_db"), [(1.0, 0.0), (10.0, 20.0)])
@pytest.mark.parametrize("noise_kind", ["tonal", "broadband"])
def test_noise_reference_snr_matches_rms_for_tonal_and_broadband_noise(
    noise_kind,
    signal_scale,
    expected_snr_db,
):
    fs = 48_000
    t = np.arange(fs * 4, dtype=float) / fs
    noise_rms = 0.001 / np.sqrt(2.0)
    if noise_kind == "tonal":
        noise = 0.001 * np.sin(2.0 * np.pi * 100.0 * t)
    else:
        rng = np.random.default_rng(6813)
        broadband = rng.normal(size=t.size)
        broadband_freqs = np.fft.rfftfreq(t.size, 1.0 / fs)
        broadband_spectrum = np.fft.rfft(broadband)
        broadband_spectrum[(broadband_freqs < 80.0) | (broadband_freqs > 8000.0)] = 0.0
        noise = np.fft.irfft(broadband_spectrum, n=t.size)
        noise *= noise_rms / np.sqrt(np.mean(np.square(noise)))
    speech_signal = 0.001 * signal_scale * np.sin(2.0 * np.pi * 1000.0 * t)
    speech = noise + speech_signal

    noise_reference = analyze_noise_reference(
        noise,
        speech,
        fs,
        speech_vad_probabilities=np.ones(20),
    )
    result = analyze_voice_spectrum(
        speech,
        fs,
        noise_spectrum_override=(
            noise_reference.frequencies,
            noise_reference.conservative_spectrum_db,
        ),
    )

    measured_snr_db = 10.0 * np.log10(
        np.mean(np.square(speech_signal)) / np.mean(np.square(noise))
    )
    assert measured_snr_db == pytest.approx(expected_snr_db, abs=0.1)
    assert result.snr_db == pytest.approx(expected_snr_db, abs=3.0)


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


@pytest.fixture
def fresh_octave_plans():
    spectrum._octave_plan.cache_clear()
    yield
    spectrum._octave_plan.cache_clear()


def test_octave_reconstruction_interpolates_on_log_frequency(monkeypatch, fresh_octave_plans):
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

    result = smooth_spectrum_perceptual(freqs, values)

    assert result[1] == pytest.approx(5.0)


def _per_spectrum_perceptual(freqs, values, strength):
    wide, medium, fine, very_wide = (
        smooth_spectrum_octave(freqs, values, fraction=n) for n in (3, 6, 12, 2)
    )
    smoothed = medium.copy()
    low = freqs < 180.0
    mid = (freqs >= 180.0) & (freqs < 3500.0)
    sibilance = (freqs >= 3500.0) & (freqs <= 9000.0)
    high = freqs > 9000.0
    smoothed[low] = wide[low]
    smoothed[mid] = medium[mid]
    smoothed[sibilance] = fine[sibilance]
    smoothed[high] = wide[high]
    if strength == "conservative":
        smoothed[mid] = 0.65 * medium[mid] + 0.35 * wide[mid]
        smoothed[sibilance] = 0.60 * fine[sibilance] + 0.40 * medium[sibilance]
        smoothed = 0.85 * smoothed + 0.15 * very_wide
    elif strength == "broad":
        smoothed = 0.50 * smoothed + 0.50 * very_wide
    return smoothed


def _smoothing_grids():
    fft_grid = np.fft.rfftfreq(4096, 1.0 / 48_000)
    shuffled = np.random.default_rng(3).permutation(np.geomspace(30.0, 18_000.0, 300))
    return [
        fft_grid,
        np.geomspace(20.0, 20_000.0, 257),
        np.linspace(0.0, 24_000.0, 40),
        shuffled,
        np.array([500.0, 510.0, 520.0]),
    ]


@pytest.mark.parametrize("strength", ["balanced", "conservative", "broad"])
def test_smoothing_equals_the_per_spectrum_recipe_exactly(strength, fresh_octave_plans):
    rng = np.random.default_rng(1234)
    for freqs in _smoothing_grids():
        rows = rng.normal(-40.0, 15.0, (129, freqs.size))
        expected = np.asarray(
            [_per_spectrum_perceptual(freqs, row, strength) for row in rows]
        )

        assert np.array_equal(spectrum._smooth_perceptual_rows(freqs, rows, strength), expected)
        assert np.array_equal(
            spectrum._smooth_perceptual_rows(freqs, rows[:5], strength), expected[:5]
        )
        for row, want in zip(rows[:3], expected[:3]):
            assert np.array_equal(smooth_spectrum_perceptual(freqs, row, strength), want)


def test_batched_smoothing_does_not_depend_on_memory_layout():
    rng = np.random.default_rng(55)
    freqs = np.fft.rfftfreq(4096, 1.0 / 48_000)
    rows = rng.normal(-40.0, 15.0, (7, freqs.size))

    assert np.array_equal(
        spectrum._smooth_perceptual_rows(freqs, np.asfortranarray(rows), "balanced"),
        spectrum._smooth_perceptual_rows(freqs, rows, "balanced"),
    )


def _edge_rows(kind, size):
    rng = np.random.default_rng(77)
    rows = rng.normal(-40.0, 15.0, (6, size))
    if kind == "nan":
        rows[0, size // 3] = np.nan
        rows[1, [0, size // 2, -1]] = np.nan
        rows[2, :] = np.nan
    elif kind == "inf":
        rows[0, size // 3] = np.inf
        rows[1, size // 2] = -np.inf
        rows[2, [1, -2]] = [np.inf, -np.inf]
        rows[3, size // 4] = np.nan
        rows[3, size // 5] = np.inf
    elif kind == "float32":
        rows = rows.astype(np.float32)
    elif kind == "all_equal":
        rows[:] = np.array([-40.0, 0.0, 1.0, -300.0, 12.3456789, 1e-9]).reshape(-1, 1)
    elif kind == "huge":
        rows *= 1e12
        rows[1] = 3000.0  # 10 ** (3000 / 10) overflows
        rows[2] = np.where(np.arange(size) % 2, 2900.0, -2900.0)
        rows[3] = 1e300
    elif kind == "tiny":
        rows *= 1e-12
        rows[1] = 5e-324
        rows[2] = -5e-324
        rows[3] = 1e-300
    return rows


def _outcome(call):
    """The result array, or the type of the exception that was raised."""
    with np.errstate(all="ignore"):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - the type is the thing compared
            return type(exc)


def _same_outcome(a, b):
    if isinstance(a, type) or isinstance(b, type):
        return a is b
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("strength", ["balanced", "conservative", "broad"])
@pytest.mark.parametrize("kind", ["nan", "inf", "float32", "all_equal", "huge", "tiny"])
def test_smoothing_matches_the_per_spectrum_recipe_on_edge_values(kind, strength, fresh_octave_plans):
    for freqs in _smoothing_grids():
        rows = _edge_rows(kind, freqs.size)
        # The public function converts to float64 first, as the recipe it replaced did.
        wide_rows = np.asarray(rows, dtype=float)
        expected = [
            _outcome(lambda row=row: _per_spectrum_perceptual(freqs, row, strength))
            for row in wide_rows
        ]
        batch = _outcome(lambda: spectrum._smooth_perceptual_rows(freqs, wide_rows, strength))
        for index, (row, want) in enumerate(zip(rows, expected)):
            got = _outcome(lambda row=row: smooth_spectrum_perceptual(freqs, row, strength))
            assert _same_outcome(got, want), (kind, strength, freqs.size, index)
            if not isinstance(want, type) and not isinstance(batch, type):
                assert np.array_equal(batch[index], want, equal_nan=True), (kind, index)


@pytest.mark.parametrize(
    "spectrum_db",
    [np.zeros((5, 1)), np.zeros((1, 5)), np.zeros((5, 5)), np.float64(1.0), np.zeros(4), np.zeros(6)],
    ids=["column", "row", "square", "scalar", "short", "long"],
)
def test_smoothing_rejects_wrongly_shaped_spectra_with_index_error(spectrum_db):
    freqs = np.linspace(100.0, 5000.0, 5)
    with pytest.raises(IndexError, match="one-dimensional and the same length"):
        smooth_spectrum_perceptual(freqs, spectrum_db, "balanced")


def test_smoothing_rejects_scalar_and_two_dimensional_freqs_with_index_error():
    with pytest.raises(IndexError, match="one-dimensional and the same length"):
        smooth_spectrum_perceptual(np.float64(100.0), np.float64(1.0))
    with pytest.raises(IndexError, match="one-dimensional and the same length"):
        smooth_spectrum_perceptual(np.array([[100.0], [200.0]]), np.zeros((2, 1)))
