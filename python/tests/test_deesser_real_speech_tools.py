"""De-esser qualification metrics preserve the clean reference and voice body."""

import numpy as np
import pytest

import evaluate_deesser_real_speech as evaluator


@pytest.mark.parametrize("high_gain,residual_db", [(2**0.5, 3.0102999566), (0.5, 6.0205999133)])
def test_audio_metrics_measure_absolute_residual_and_actual_body_band(high_gain, residual_db):
    time = np.arange(96_000) / 48_000
    body = 0.02 * np.sin(2 * np.pi * 1_000 * time)
    sibilance = 0.03 * np.sin(2 * np.pi * 7_000 * time)
    clean = body + sibilance
    capture = body + 2.0 * sibilance
    output = body * 10 ** (-0.2 / 20) + high_gain * sibilance
    sibilant = np.arange(200) < 100

    metrics = evaluator.audio_metrics(capture, output, clean, sibilant, ~sibilant)

    assert metrics["sibilant_residual_db"] == pytest.approx(residual_db, abs=0.001)
    assert metrics["body_db"] == pytest.approx(-0.2, abs=0.001)
    assert metrics["voiced_db"] == pytest.approx(20 * np.log10(high_gain / 2), abs=0.001)
    assert metrics["finite_output"] is True
    assert np.isfinite(metrics["estoi"])


@pytest.mark.parametrize("invalid_output", [np.ones(47_999), np.full(48_000, np.nan)])
def test_audio_metrics_reject_incomplete_or_nonfinite_renders(invalid_output):
    clean = np.ones(48_000)
    frames = np.ones(100, dtype=bool)

    metrics = evaluator.audio_metrics(clean, invalid_output, clean, frames, frames)

    assert metrics["finite_output"] is False
    assert metrics["sibilant_residual_db"] is None
    assert metrics["body_db"] is None
    assert metrics["estoi"] is None


def test_recommended_disabled_render_preserves_capture():
    time = np.arange(48_000) / 48_000
    capture = (0.1 * np.sin(2 * np.pi * 7_000 * time)).astype(np.float32)
    settings = {"enabled": False, "auto_enabled": False, "threshold_db": -40.0,
                "ratio": 10.0, "max_reduction_db": 12.0}

    disabled = evaluator._render(capture, settings, force_enabled=False)
    forced = evaluator._render(capture, settings)

    np.testing.assert_allclose(disabled, capture, atol=1e-7)
    assert np.mean(forced**2) < np.mean(disabled**2) / 2
