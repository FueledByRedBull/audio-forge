"""The decision block of evaluation reports, and reading the decision of every tracked report."""

from __future__ import annotations

import json
from pathlib import Path

import check_evaluation_hygiene as hygiene
import evaluation_decision as ed
import pytest

EVALUATION_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
REPORTS = sorted(EVALUATION_ROOT.glob("*.json"))
GOOD = {"outcome": "adopt", "reason": "All gates passed.", "decided": "2026-10-26", "by": "the maintainer"}


def _has_block(path: Path) -> bool:
    decision = json.loads(path.read_text(encoding="utf-8")).get("decision")
    return isinstance(decision, dict) and "outcome" in decision


@pytest.mark.parametrize("path", REPORTS, ids=lambda path: path.name)
def test_every_tracked_report_reads_as_a_decision(path):
    decision = ed.read_decision(path)

    assert decision.keys() == ed.FIELDS
    assert decision["outcome"] in ed.OUTCOMES
    assert decision["reason"] is None or decision["reason"].strip()


def test_the_legacy_table_covers_exactly_the_reports_without_a_block():
    names = {path.name for path in REPORTS}
    without_block = {path.name for path in REPORTS if not _has_block(path)}

    assert without_block <= set(ed.LEGACY_OUTCOMES) <= names


# One report for each way the reports before the block state their decision.
LEGACY_SHAPES = [
    ("auto-eq-fixed-grid-2026-10-report.json", "a token, with the reason in `summary`", "retain", "2026-10-01", "A simpler Auto-EQ fitter"),
    ("processing-order-2026-10-report.json", "a token, with the reason in `reason`", "retain", "2026-10-01", "With the default Threshold Only gate"),
    ("dynamics-aliasing-2026-09-report.json", "a sentence", "retain", "2026-09-29", "retain current dynamics path; no oversampling justified"),
    ("compressor-control-report.json", "an object with a reason", "adopt", None, "Every predefined control and realtime-safety gate passed"),
    ("processing-order-2026-09-report.json", "an object of tokens for each area", "retain", "2026-09-29", "Both candidates fail"),
    ("sparse-auto-eq-filter-report.json", "an object with a product action", "retain", "2026-09-09", "retain incumbent; candidate remains evaluation-only"),
    ("rnnoise-backend-comparison.json", "an object of measurements", "retain", "2026-07-29", None),
    ("product-joint-tuning-2026-09-report.json", "a top-level boolean", "adopt", None, None),
    ("cross-take-auto-eq-report.json", "a top-level status", "retain", "2026-09-08", "The evaluation-only cross-take candidate failed"),
    ("deepfilter-hardening-report.json", "separate decisions", "adopt", "2026-09-06", None),
    ("voice-setup-arms-2026-10-report.json", "none: measurements only", "measured", None, None),
]


@pytest.mark.parametrize(
    ("name", "shape", "outcome", "decided", "reason"), LEGACY_SHAPES, ids=[shape[1] for shape in LEGACY_SHAPES]
)
def test_the_old_ways_of_stating_a_decision_read_alike(name, shape, outcome, decided, reason):
    decision = ed.read_decision(EVALUATION_ROOT / name)

    assert decision["outcome"] == outcome and decision["decided"] == decided and decision["by"] is None
    if reason is None:
        assert decision["reason"] is None
    else:
        assert decision["reason"].startswith(reason)


def test_a_report_with_the_block_reads_as_written(tmp_path):
    path = tmp_path / "new-report.json"
    path.write_text(json.dumps({"decision": GOOD}), encoding="utf-8")

    assert ed.read_decision(path) == GOOD


def test_a_new_report_without_the_block_is_refused(tmp_path):
    path = tmp_path / "new-report.json"
    path.write_text(json.dumps({"decision": "keep_everything"}), encoding="utf-8")

    with pytest.raises(ValueError, match="no decision block"):
        ed.read_decision(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"outcome": "adopted"}, "outcome must be one of"),
        ({"reason": " "}, "reason must be a sentence"),
        ({"decided": "26 October"}, "ISO date"),
        ({"decided": None}, "ISO date"),
        ({"by": None}, "must name who decided"),
        ({"outcome": "measured"}, "records no decided date or decider"),
    ],
)
def test_a_decision_block_that_is_not_well_formed_says_why(change, message):
    errors = ed.block_errors({**GOOD, **change})

    assert any(message in error for error in errors)


def test_a_decision_block_has_exactly_four_fields():
    assert ed.block_errors(GOOD) == []
    assert ed.block_errors({"outcome": "measured", "reason": "Numbers only.", "decided": None, "by": None}) == []
    assert ed.block_errors({key: value for key, value in GOOD.items() if key != "by"}) == ["decision has no by"]
    assert ed.block_errors({**GOOD, "extra": 1}) == ["decision has unknown field extra"]
    assert ed.block_errors("keep_everything") == ["decision must be an object"]
    with pytest.raises(ValueError, match="outcome must be one of"):
        ed.decision_block("maybe", "Unsure.")


def test_the_hygiene_check_rejects_a_malformed_decision_block(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"decision": {**GOOD, "decided": "yesterday"}}), encoding="utf-8")

    assert any("ISO date" in error for error in hygiene.validate_report(path))
    path.write_text(json.dumps({"decision": GOOD}), encoding="utf-8")
    assert hygiene.validate_report(path) == []
