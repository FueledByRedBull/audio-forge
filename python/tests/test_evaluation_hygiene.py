"""Tests for evaluation evidence portability and contract enforcement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import check_evaluation_hygiene as hygiene
import pytest


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


def _audible_report(**overrides: object) -> dict:
    report: dict[str, object] = {
        "schema_version": 2,
        "audible_change": True,
        "evaluation_contract": _contract(),
    }
    report.update(overrides)
    return report


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


def test_oversized_tracked_report_is_rejected(tmp_path: Path):
    path = tmp_path / "report.json"
    _write(path, {"rows": ["x" * hygiene.MAX_TRACKED_REPORT_BYTES]})

    errors = hygiene.validate_report(path)

    assert any("tracked report exceeds" in error for error in errors)


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


def test_implementation_source_hashes_are_verified(tmp_path: Path, monkeypatch):
    source = tmp_path / "implementation.py"
    source.write_text("implementation\n", encoding="utf-8")
    manifest = tmp_path / "release-assets.json"
    manifest.write_text("{}\n", encoding="utf-8")
    path = tmp_path / "report.json"
    _write(
        path,
        _audible_report(
            implementation_sha256={
                "implementation.py": hashlib.sha256(source.read_bytes()).hexdigest(),
                "release-assets.json": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        ),
    )
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)

    assert hygiene.validate_report(path) == []


def test_stale_implementation_source_hash_is_rejected(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "implementation.py"
    source.write_text("implementation\n", encoding="utf-8")
    path = tmp_path / "report.json"
    _write(
        path,
        _audible_report(
            implementation_sha256={"implementation.py": "0" * 64},
        ),
    )
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)

    errors = hygiene.validate_report(path)

    assert any("stale source SHA-256" in error for error in errors)


def test_implementation_artifact_hash_does_not_count_as_source_provenance(
    tmp_path: Path,
):
    path = tmp_path / "report.json"
    _write(
        path,
        _audible_report(
            implementation_sha256={
                "native_extension.pyd": "0" * 64,
                "future.identity": "1" * 64,
            },
        ),
    )

    errors = hygiene.validate_report(path)

    assert any("lacks verifiable source SHA-256 hashes" in error for error in errors)
    unverified: list[str] = []
    hygiene.validate_report(path, unverified=unverified)
    assert len(unverified) == 2
    assert all("retained implementation hash is unverified" in note for note in unverified)


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


def test_declared_text_source_hash_is_portable_across_line_endings(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.py"
    report = tmp_path / "report.json"
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)
    for actual, recorded in (
        (b"first\nsecond\n", b"first\r\nsecond\r\n"),
        (b"first\r\nsecond\r\n", b"first\nsecond\n"),
    ):
        source.write_bytes(actual)
        _write(
            report,
            {
                "source_sha256": {
                    "source.py": hashlib.sha256(recorded).hexdigest()
                }
            },
        )
        assert hygiene.validate_report(report) == []


def test_mixed_text_source_hash_must_use_canonical_line_endings(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.py"
    source.write_bytes(b"first\r\nsecond\n")
    report = tmp_path / "report.json"
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)

    _write(
        report,
        {
            "source_sha256": {
                "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
            }
        },
    )
    assert any(
        "stale source SHA-256" in error for error in hygiene.validate_report(report)
    )

    canonical = b"first\nsecond\n"
    _write(
        report,
        {
            "source_sha256": {
                "source.py": hashlib.sha256(canonical).hexdigest()
            }
        },
    )
    assert hygiene.validate_report(report) == []


def test_declared_binary_source_hash_remains_byte_exact(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first\r\nsecond\r\n")
    report = tmp_path / "report.json"
    _write(
        report,
        {
            "source_sha256": {
                "source.bin": hashlib.sha256(b"first\nsecond\n").hexdigest()
            }
        },
    )
    monkeypatch.setattr(hygiene, "REPO_ROOT", tmp_path)

    errors = hygiene.validate_report(report)

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


def test_nonfinite_numeric_values_are_rejected_recursively(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text('{"nested": {"value": 1e1000}}', encoding="utf-8")

    errors = hygiene.validate_report(path)

    assert any("non-finite numeric value" in error for error in errors)


def test_nonstandard_json_constants_are_rejected(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text('{"nested": [NaN]}', encoding="utf-8")

    errors = hygiene.validate_report(path)

    assert any("invalid JSON" in error for error in errors)


def test_runtime_metrics_must_be_nonnegative_numbers(tmp_path: Path):
    path = tmp_path / "report.json"
    contract = _contract()
    contract["runtime"] = {"max_p99_frame_seconds": -0.001}
    _write(path, _audible_report(evaluation_contract=contract))

    errors = hygiene.validate_report(path)

    assert any("runtime.max_p99_frame_seconds" in error for error in errors)
    assert any("non-negative" in error for error in errors)


def test_runtime_boolean_measurements_are_rejected(tmp_path: Path):
    path = tmp_path / "report.json"
    contract = _contract()
    contract["runtime"] = {"max_p99_frame_seconds": True}
    _write(path, _audible_report(evaluation_contract=contract))

    errors = hygiene.validate_report(path)

    assert any("runtime.max_p99_frame_seconds" in error for error in errors)
    assert any("finite nonnegative number" in error for error in errors)


@pytest.mark.parametrize("value", ["1", {}, [], None, True])
def test_named_runtime_duration_must_be_numeric(tmp_path: Path, value):
    path = tmp_path / "report.json"
    contract = _contract()
    contract["runtime"] = {
        "max_p99_frame_seconds": 0.001,
        "max_case_runtime_ms": value,
    }
    _write(path, _audible_report(evaluation_contract=contract))

    errors = hygiene.validate_report(path)

    assert any("runtime.max_case_runtime_ms" in error for error in errors)
    assert any("finite nonnegative number" in error for error in errors)


def test_malformed_schema_version_is_a_controlled_validation_error(tmp_path: Path):
    path = tmp_path / "report.json"
    _write(path, _audible_report(schema_version="2"))

    errors = hygiene.validate_report(path)

    assert errors
    assert any("schema_version must be a positive integer" in error for error in errors)
