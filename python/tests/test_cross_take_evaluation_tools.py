"""Checks on the tracked cross-take Auto-EQ report."""

from __future__ import annotations

import json
from pathlib import Path

REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation/cross-take-auto-eq-report.json"
)


def test_report_separates_external_assets_from_source_freshness() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    source_paths = set(report["provenance"]["source_hashes"])
    asset_paths = set(report["provenance"]["asset_hashes"])

    assert all(not path.startswith("models/") for path in source_paths)
    assert asset_paths == {
        "models/cross_take_eval/manifest.json",
        "models/silero_vad.onnx",
    }
