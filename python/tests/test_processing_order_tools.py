"""Tests for the processing-order evaluation report."""

from __future__ import annotations

import json
from pathlib import Path


REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/processing-order-2026-09-report.json"
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
