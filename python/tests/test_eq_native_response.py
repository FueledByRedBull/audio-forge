"""Cross-language contracts for the native EQ response renderer."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eq import eq_magnitude_response
from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response
from mic_eq.analysis.eq_quality import evaluate_eq_quality
from mic_eq.mic_eq_core import simulate_auto_eq_chain
from mic_eq.ui.eq_curve import EQCurveWidget


DEFAULT_BANDS = [
    (80.0, 0.0, 1.41),
    (160.0, 0.0, 1.41),
    (320.0, 0.0, 1.41),
    (640.0, 0.0, 1.41),
    (1280.0, 0.0, 1.41),
    (2500.0, 0.0, 1.41),
    (5000.0, 0.0, 1.41),
    (8000.0, 0.0, 1.41),
    (12000.0, 0.0, 1.41),
    (16000.0, 0.0, 1.41),
]


def test_native_eq_response_validates_contract():
    with pytest.raises(ValueError, match="expected 10 EQ bands"):
        eq_magnitude_response([1000.0], DEFAULT_BANDS[:-1], 48_000.0)
    with pytest.raises(ValueError, match="sample_rate"):
        eq_magnitude_response([1000.0], DEFAULT_BANDS, 0.0)
    with pytest.raises(ValueError, match="Nyquist"):
        eq_magnitude_response([24_001.0], DEFAULT_BANDS, 48_000.0)


@pytest.mark.parametrize(
    ("band_index", "frequency_hz", "gain_db", "q"),
    [
        (1, 160.0, -12.0, 0.1),
        (4, 1000.0, 6.0, 2.0),
        (7, 8000.0, 12.0, 10.0),
    ],
)
def test_native_peaking_response_reaches_configured_center_gain(
    band_index: int,
    frequency_hz: float,
    gain_db: float,
    q: float,
):
    bands = list(DEFAULT_BANDS)
    bands[band_index] = (frequency_hz, gain_db, q)

    response = eq_magnitude_response([frequency_hz], bands, 48_000.0)

    assert response[0] == pytest.approx(gain_db, abs=1.0e-8)


@pytest.mark.parametrize("sample_rate", [44_100.0, 48_000.0, 96_000.0])
def test_python_eq_prediction_matches_native_at_capture_rate(sample_rate: float):
    bands = [
        (80.0, 2.5, 0.8),
        (160.0, -3.0, 1.2),
        (320.0, 1.0, 1.0),
        (640.0, 0.0, 1.41),
        (1280.0, 4.0, 0.9),
        (2500.0, -2.0, 2.4),
        (5000.0, 0.0, 1.41),
        (8000.0, 3.0, 1.1),
        (12000.0, -1.5, 0.7),
        (16000.0, 2.0, 0.8),
    ]
    frequencies = np.geomspace(20.0, min(20_000.0, sample_rate * 0.45), 257)
    predicted = _predict_eq_response(
        frequencies,
        [gain for _, gain, _ in bands],
        [q for _, _, q in bands],
        [frequency for frequency, _, _ in bands],
        sample_rate=sample_rate,
    )
    native = np.asarray(eq_magnitude_response(frequencies.tolist(), bands, sample_rate))

    np.testing.assert_allclose(predicted, native, rtol=0.0, atol=1.0e-8)


@pytest.mark.parametrize("sample_rate", [44_100.0, 96_000.0])
def test_eq_quality_uses_capture_rate_for_response_metrics(sample_rate: float):
    bands = [
        (80.0, 2.5, 0.8),
        (160.0, -3.0, 1.2),
        (320.0, 1.0, 1.0),
        (640.0, 0.0, 1.41),
        (1280.0, 4.0, 0.9),
        (2500.0, -2.0, 2.4),
        (5000.0, 0.0, 1.41),
        (8000.0, 3.0, 1.1),
        (12000.0, -1.5, 0.7),
        (16000.0, 2.0, 0.8),
    ]
    frequencies = np.logspace(
        np.log10(20.0),
        np.log10(min(20_000.0, sample_rate / 2.0 - 1.0)),
        256,
    )
    native = np.asarray(eq_magnitude_response(frequencies.tolist(), bands, sample_rate))
    metrics = evaluate_eq_quality(
        [frequency for frequency, _, _ in bands],
        [gain for _, gain, _ in bands],
        [q for _, _, q in bands],
        sample_rate,
    )

    assert metrics.max_boost_db == pytest.approx(max(0.0, float(np.max(native))))
    assert metrics.max_cut_db == pytest.approx(max(0.0, float(-np.min(native))))


def test_native_full_chain_keeps_48khz_frontend_boundary():
    with pytest.raises(ValueError, match="48000"):
        simulate_auto_eq_chain(
            np.zeros(480, dtype=np.float32),
            44_100.0,
            DEFAULT_BANDS,
            {"full_chain": True, "processing_mode": "normal"},
        )


def test_eq_curve_widget_uses_native_target_response(qapp):
    widget = EQCurveWidget()
    bands = list(DEFAULT_BANDS)
    bands[0] = (90.0, 4.0, 0.7)
    bands[4] = (1100.0, -6.0, 3.0)
    bands[9] = (15_000.0, 3.0, 0.9)

    widget.set_all_params(bands)
    direct = eq_magnitude_response(widget.freq_points, bands, widget.sample_rate)

    np.testing.assert_allclose(widget.response_db, direct, rtol=0.0, atol=1.0e-12)
    assert not hasattr(widget, "_calc_biquad_coefficients")
    widget.close()
