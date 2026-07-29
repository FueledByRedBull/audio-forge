"""Tests for evaluation evidence portability and contract enforcement."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).parent.parent / "tools" / "check_evaluation_hygiene.py"
SPEC = importlib.util.spec_from_file_location("check_evaluation_hygiene", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
hygiene = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hygiene
SPEC.loader.exec_module(hygiene)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _contract() -> dict:
    return {
        "configuration": {},
        "asset_hashes": {},
        "runtime": {"max_p99_frame_seconds": 0.001},
        "latency": {},
        "clean_preservation": {},
        "listening_status": {"status": "not_run", "reason": "automated gate"},
    }


def test_portable_audible_report_passes(tmp_path: Path):
    path = tmp_path / "report.json"
    _write(
        path,
        {
            "schema_version": 2,
            "audible_change": True,
            "evaluation_contract": _contract(),
            "corpus": {"root": "models/evaluation"},
        },
    )

    assert hygiene.validate_report(path) == []


def test_machine_local_paths_are_rejected_recursively(tmp_path: Path):
    path = tmp_path / "report.json"
    _write(path, {"schema_version": 1, "capture": {"path": "C:/Users/test/a.wav"}})

    errors = hygiene.validate_report(path)

    assert any("machine-local absolute path" in error for error in errors)


def test_audible_report_requires_complete_contract(tmp_path: Path):
    path = tmp_path / "report.json"
    _write(path, {"schema_version": 2, "audible_change": True})

    errors = hygiene.validate_report(path)

    assert any("lacks evaluation_contract" in error for error in errors)


def test_release_trends_reject_duplicate_versions(tmp_path: Path):
    path = tmp_path / "release-trends.json"
    _write(
        path,
        {
            "schema_version": 1,
            "releases": [
                {"version": "1.10.0"},
                {"version": "1.10.0"},
            ],
        },
    )

    errors = hygiene.validate_release_trends(path)

    assert any("duplicate version 1.10.0" in error for error in errors)
