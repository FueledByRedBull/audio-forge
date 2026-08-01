"""Tests for diagnostics-export retention gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TOOL_PATH = (
    Path(__file__).parents[1] / "tools" / "evaluate_diagnostics_export.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_diagnostics_export",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_adversarial_fixture_passes_every_privacy_gate() -> None:
    report = TOOL.evaluate()

    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["metrics"]["serialized_bytes"] <= report["metrics"][
        "maximum_bytes"
    ]
