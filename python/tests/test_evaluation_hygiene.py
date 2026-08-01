"""Tests for evaluation evidence portability and contract enforcement."""

from __future__ import annotations

import importlib.util
import hashlib
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
    }


def test_portable_audible_report_passes(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    path = tmp_path / "report.json"
    _write(
        path,
        {
            "schema_version": 2,
            "audible_change": True,
            "source_sha256": {
                "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
            },
            "evaluation_contract": _contract(),
            "corpus": {"root": "models/evaluation"},
        },
    )
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)

    assert hygiene.validate_report(path) == []


def test_audible_report_requires_source_hashes(tmp_path: Path):
    path = tmp_path / "report.json"
    _write(
        path,
        {
            "schema_version": 2,
            "audible_change": True,
            "evaluation_contract": _contract(),
        },
    )

    errors = hygiene.validate_report(path)

    assert any("lacks verifiable source SHA-256 hashes" in error for error in errors)


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


def test_release_trends_validate_embedded_hardware_privacy(tmp_path: Path):
    path = tmp_path / "release-trends.json"
    _write(
        path,
        {
            "schema_version": 1,
            "releases": [
                {
                    "version": "1.10.1",
                    "status": "published",
                    "commit": "a" * 40,
                    "package": {
                        "bundle": {"status": "not_measured", "reason": "test"},
                        "archive": {"status": "not_measured", "reason": "test"},
                    },
                    "runtime": {"status": "not_measured", "reason": "test"},
                    "quality": {"status": "not_measured", "reason": "test"},
                    "hardware": {
                        "status": "measured",
                        "value": {
                            "schema_version": 3,
                            "routes": {
                                "correlation": {
                                    "input": "Private microphone",
                                    "output": "device-" + "b" * 16,
                                }
                            },
                        },
                    },
                }
            ],
        },
    )

    errors = hygiene.validate_release_trends(path)

    assert any("report-local device pseudonym" in error for error in errors)


def test_stale_declared_source_hash_is_rejected(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("before\n", encoding="utf-8")
    path = tmp_path / "report.json"
    _write(
        path,
        {
            "source_sha256": {
                "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
            }
        },
    )
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)
    source.write_text("after\n", encoding="utf-8")

    errors = hygiene.validate_report(path)

    assert any("stale source SHA-256" in error for error in errors)


def test_hardware_reports_require_pseudonymized_routes(tmp_path: Path):
    raw = tmp_path / "hardware-validation-raw.json"
    _write(
        raw,
        {
            "schema_version": 3,
            "routes": {
                "correlation": {"input": "Private Mic", "output": "device-" + "a" * 16}
            },
        },
    )
    sanitized = tmp_path / "hardware-validation-sanitized.json"
    _write(
        sanitized,
        {
            "schema_version": 3,
            "routes": {
                "correlation": {
                    "input": "device-" + "b" * 16,
                    "output": "device-" + "a" * 16,
                }
            },
        },
    )

    assert any(
        "must use a report-local device pseudonym" in error
        for error in hygiene.validate_report(raw)
    )
    assert hygiene.validate_report(sanitized) == []


def test_historical_hardware_report_requires_redaction_provenance(tmp_path: Path):
    path = tmp_path / "hardware-validation-v1.10.1-published.json"
    _write(
        path,
        {
            "schema_version": 2,
            "routes": {
                "correlation": {
                    "input": "device-" + "a" * 16,
                    "output": "device-" + "b" * 16,
                }
            },
        },
    )

    errors = hygiene.validate_report(path)

    assert any("lacks privacy-redaction provenance" in error for error in errors)
