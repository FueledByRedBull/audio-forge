"""The held-out gate comparison must reject regressions and mismatched cases."""

import pytest

from evaluate_voice_regressions import compare


def test_voice_regression_gates():
    baseline = {"corpus_manifest_sha256": "same", "cases": [
        {"path": "speech", "speech": True, "retention_db": -2.0},
        {"path": "noise", "speech": False, "retention_db": -20.0},
    ]}
    assert compare(baseline, baseline)["passed"]
    candidate = {**baseline, "cases": [
        {"path": "speech", "speech": True, "retention_db": -3.0},
        baseline["cases"][1],
    ]}
    assert not compare(candidate, baseline)["passed"]
    with pytest.raises(ValueError, match="case set differs"):
        compare({**baseline, "cases": baseline["cases"][:1]}, baseline)
