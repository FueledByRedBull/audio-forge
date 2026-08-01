"""Tests for processing-order evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


TOOL_PATH = Path(__file__).parent.parent / "tools" / "evaluate_processing_order.py"
SPEC = importlib.util.spec_from_file_location("evaluate_processing_order", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
ordering = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ordering
SPEC.loader.exec_module(ordering)
REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/processing-order-report.json"
)


def test_control_probability_mapping_has_one_value_per_rnnoise_frame():
    result = ordering._control_probabilities(np.asarray([0.0, 1.0]), 1_440)

    assert result.shape == (3,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_pumping_focuses_on_two_to_eight_hz():
    time = np.arange(1_000) * ordering.FRAME_SIZE / ordering.SAMPLE_RATE

    assert ordering._pumping(np.sin(2 * np.pi * 4 * time)) > ordering._pumping(
        np.sin(2 * np.pi * 0.2 * time)
    )


def test_report_keeps_both_incumbent_product_orders() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["schema_version"] == 4
    assert report["audible_change"] is True
    assert report["status"] == "incumbents-retained-objective-gates-failed"
    assert report["decision"]["product_chain_changed"] is False
    assert report["decision"]["gate_suppressor"] == "retain_gate_before_suppressor"
    assert report["decision"]["eq_deesser"] == "retain_deesser_before_eq"

    source_paths = set(report["provenance"]["source_hashes"])
    asset_paths = set(report["provenance"]["asset_hashes"])
    assert all(not path.startswith("models/") for path in source_paths)
    assert asset_paths == {
        "models/dpdfnet_eval_subset/manifest.json",
        "models/silero_vad.onnx",
    }


def test_pcm_reader_centers_unsigned_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    raw = np.column_stack(
        (
            np.asarray([0, 128, 255], dtype=np.uint8),
            np.asarray([255, 128, 0], dtype=np.uint8),
        )
    )
    wavfile.write(path, 48_000, raw)

    sample_rate, audio = ordering._read_mono(path)

    assert sample_rate == 48_000
    assert np.allclose(audio, np.zeros(3), atol=1.0 / 128.0)
