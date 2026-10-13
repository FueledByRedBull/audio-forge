"""Tests for the retained DeepFilter full-band evaluation report."""

from __future__ import annotations

import json
from pathlib import Path


REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/deepfilter-fullband-report.json"
)


def test_report_retains_30_db_under_objective_policy() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "objective-retained-incumbent"
    assert report["decision"]["release_selected_attenuation_limit_db"] == 30.0
    assert report["decision"]["retained"] is True
