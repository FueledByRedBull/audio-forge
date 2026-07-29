"""Cross-language contracts for the native EQ response renderer."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eq import eq_magnitude_response
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
