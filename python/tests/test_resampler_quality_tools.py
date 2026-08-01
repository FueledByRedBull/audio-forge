"""Tests for the product resampler quality evaluator."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_resampler_quality.py"
SPEC = importlib.util.spec_from_file_location("evaluate_resampler_quality", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_resampler_quality = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_resampler_quality"] = evaluate_resampler_quality
SPEC.loader.exec_module(evaluate_resampler_quality)


def test_db_ratio_and_tone_projection_are_scale_consistent():
    sample_rate = 48_000
    tone = evaluate_resampler_quality._sine(sample_rate, 1_000.0, 1.0)

    amplitude = evaluate_resampler_quality._tone_amplitude(
        tone, sample_rate, 1_000.0
    )

    assert math.isclose(amplitude, 0.5, abs_tol=1e-6)
    assert math.isclose(
        evaluate_resampler_quality._db_ratio(amplitude, 0.5),
        0.0,
        abs_tol=1e-9,
    )


def test_run_trims_padding_to_native_expected_count():
    def simulator(samples, input_rate, output_rate, chunk_size, sinc_len, window):
        del input_rate, output_rate, chunk_size, sinc_len, window
        return list(samples) + [0.0, 0.0], 7, len(samples), [10]

    source = np.arange(8, dtype=np.float64)
    output, delay, timings = evaluate_resampler_quality._run(
        source, 44_100, 48_000, simulator
    )

    assert output.tolist() == source.tolist()
    assert delay == 7
    assert timings == [10]


def test_offline_reference_is_flat_at_twenty_kilohertz():
    source = evaluate_resampler_quality._sine(44_100, 20_000.0, 0.75)

    output = evaluate_resampler_quality._offline_reference(source, 44_100, 48_000)
    gain_db = evaluate_resampler_quality._db_ratio(
        evaluate_resampler_quality._rms(
            evaluate_resampler_quality._steady_slice(output, 48_000)
        ),
        evaluate_resampler_quality._rms(
            evaluate_resampler_quality._steady_slice(source, 44_100)
        ),
    )

    assert abs(gain_db) < 0.01


def test_native_product_configuration_is_explicit():
    assert evaluate_resampler_quality.product_resampler_configuration() == (
        128,
        "blackman",
        "cubic",
        256,
        1024,
    )
