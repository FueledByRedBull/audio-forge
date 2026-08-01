"""Tests for Auto-EQ confidence threshold calibration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


TOOL_PATH = Path(__file__).parent.parent / "tools" / "calibrate_auto_eq_confidence.py"
SPEC = importlib.util.spec_from_file_location("calibrate_auto_eq_confidence", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def test_classification_counts_and_scores():
    result = calibration._classification(
        np.asarray([0.9, 0.7, 0.4, 0.1]),
        np.asarray([True, False, True, False]),
        0.5,
    )

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_calibration_retains_current_without_both_validation_classes():
    rows = [
        {
            "split": "train" if index < 8 else "validation",
            "score": index / 10.0,
            "label": True,
        }
        for index in range(12)
    ]

    result = calibration._calibrate(
        rows,
        score_key="score",
        label_key="label",
        current_threshold=0.45,
    )

    assert result["selection"] == "current"
    assert result["selected_threshold"] == 0.45


def test_snr_render_hits_requested_level() -> None:
    rng = np.random.default_rng(1801)
    clean = rng.normal(0.0, 0.05, 48_000).astype(np.float32)
    noise = rng.normal(0.0, 0.02, 48_000).astype(np.float32)

    rendered, scaled_noise, measured_snr, scale = calibration._render_at_snr(
        clean,
        noise,
        10.0,
    )

    assert np.isclose(measured_snr, 10.0, atol=1e-6)
    assert np.allclose(rendered, clean + scaled_noise)
    assert scale > 0.0


def test_pcm_reader_centers_unsigned_and_scales_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo-u8.wav"
    raw = np.column_stack(
        (
            np.asarray([0, 128, 255], dtype=np.uint8),
            np.asarray([255, 128, 0], dtype=np.uint8),
        )
    )
    wavfile.write(path, 16_000, raw)

    sample_rate, audio = calibration._read_mono(path)

    assert sample_rate == 16_000
    assert np.allclose(audio, np.zeros(3), atol=1.0 / 128.0)
