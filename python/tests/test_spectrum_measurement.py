"""Targeted tests for VAD fusion and robust Auto-EQ measurement aggregation."""

import numpy as np

from mic_eq.analysis.spectrum import _robust_median_spectrum, _voiced_frame_mask


def test_vad_posterior_can_retain_quiet_speech_below_energy_gate():
    frame_rms_db = np.asarray([-20.0, -50.0, -52.0, -21.0, -22.0, -60.0])
    frame_starts = np.arange(frame_rms_db.size, dtype=int) * 1536
    vad_probabilities = np.asarray([0.90, 0.90, 0.80, 0.90, 0.10, 0.10])

    mask = _voiced_frame_mask(
        frame_rms_db,
        vad_probabilities=vad_probabilities,
        frame_starts=frame_starts,
        frame_size=1536,
        sample_rate=48_000,
    )

    # Frames 1 and 2 are deliberately below the ordinary energy gate, but
    # strong Silero posterior evidence keeps them in the speech measurement.
    assert mask[:4].tolist() == [True, True, True, True]
    assert mask[4:].tolist() == [False, False]


def test_robust_median_spectrum_rejects_shape_outlier():
    rng = np.random.default_rng(20260727)
    freqs = np.linspace(0.0, 10_000.0, 513)
    base = -42.0 + 4.0 * np.sin(np.log2(np.maximum(freqs, 20.0) / 180.0))
    spectra = np.asarray(
        [base + rng.normal(0.0, 0.08, size=base.size) for _ in range(6)],
        dtype=float,
    )
    outlier = base + 18.0 * np.exp(-0.5 * ((freqs - 2200.0) / 250.0) ** 2)
    spectra = np.vstack([spectra, outlier])

    robust, inlier_ratio = _robust_median_spectrum(freqs, spectra)

    assert 0.80 < inlier_ratio < 1.0
    outlier_bin = int(np.argmin(abs(freqs - 2200.0)))
    assert abs(float(robust[outlier_bin]) - float(base[outlier_bin])) < 1.0
