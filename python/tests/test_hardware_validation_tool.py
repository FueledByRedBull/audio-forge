from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool("evaluate_hardware_validation")
HEALTH_TOOL = _load_tool("health_check")
MATRIX_TOOL = _load_tool("evaluate_hardware_matrix")


def test_hardware_result_parsers_require_success_and_evidence() -> None:
    self_test = {
        "return_code": 0,
        "stdout": ["Self-test passed: rt=106.18ms confidence=0.865"],
    }
    health = {
        "return_code": 0,
        "stdout": [
            'Health summary: max_input_age_ms=5 max_output_age_ms=4 restarts=0 underrun_baseline=3 diagnostics={"input_dropped_samples":0}'
        ],
    }

    parsed_self_test = TOOL._parse_self_test(self_test)
    parsed_health = TOOL._parse_health(health)

    assert parsed_self_test == {
        "passed": True,
        "route_latency_ms": 106.18,
        "confidence": 0.865,
    }
    assert parsed_health["passed"] is True
    assert parsed_health["max_input_callback_age_ms"] == 5
    assert parsed_health["stream_restarts"] == 0
    assert parsed_health["output_underrun_baseline"] == 3
    assert parsed_health["runtime_diagnostics"]["input_dropped_samples"] == 0


def test_hardware_report_provenance_uses_project_version_and_dirty_revision(
    monkeypatch,
) -> None:
    assert TOOL._project_version() == "1.11.0"

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    outputs = iter((Result("abc123\n"), Result(" M changed.py\n")))
    monkeypatch.setattr(TOOL.subprocess, "run", lambda *args, **kwargs: next(outputs))

    assert TOOL._source_revision() == "abc123+uncommitted"


def test_hardware_runtime_provenance_is_portable_and_hashes_exact_binary() -> None:
    provenance = TOOL._runtime_provenance()

    assert set(provenance) == {
        "native_extension",
        "self_test",
        "health_check",
        "latency_analysis",
    }
    for record in provenance.values():
        assert not Path(record["path"]).is_absolute()
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64


def test_health_gate_rejects_missing_or_nonzero_critical_diagnostics() -> None:
    clean: dict[str, object] = {
        key: 0 for key in HEALTH_TOOL._ZERO_REQUIRED_DIAGNOSTICS
    }
    clean.update(
        {
            # Normal clock-drift retiming may deliberately compress samples.
            "jitter_dropped_samples": 953,
            "output_retime_adjustment_count": 2021,
            "noise_backend_available": True,
            "noise_backend_failed": False,
            "last_stream_error": None,
            "output_underrun_total": 3,
        }
    )
    assert (
        HEALTH_TOOL._critical_diagnostic_failures(
            clean,
            output_underrun_baseline=3,
        )
        == []
    )

    broken = dict(clean)
    broken.pop("suppressor_non_finite_count")
    broken["input_dropped_samples"] = 4
    broken["noise_backend_failed"] = True
    broken["last_stream_error"] = "device lost"
    broken["output_underrun_total"] = 4

    failures = HEALTH_TOOL._critical_diagnostic_failures(
        broken,
        output_underrun_baseline=3,
    )

    assert "suppressor_non_finite_count=missing" in failures
    assert "input_dropped_samples=4" in failures
    assert "noise_backend_failed=true" in failures
    assert "last_stream_error=set" in failures
    assert "output_underrun_total=4 (baseline 3)" in failures


def test_hardware_report_privacy_filter_removes_all_selected_device_names() -> None:
    raw_names = ["Private USB Microphone", "Private Virtual Cable"]
    runs, pseudonyms = TOOL._privacy_filter_runs(
        [
            {
                "stdout": [
                    "Selected Private USB Microphone -> Private Virtual Cable"
                ],
                "stderr": ["Private USB Microphone recovered"],
                "return_code": 0,
            }
        ],
        raw_names,
        key=b"p" * 32,
    )
    serialized = json.dumps(runs)
    assert all(name not in serialized for name in raw_names)
    assert all(value.startswith("device-") for value in pseudonyms.values())
    assert len(set(pseudonyms.values())) == 2

    nested = TOOL._replace_private_strings(
        {"routes": {"input": raw_names[0]}, "message": f"using {raw_names[1]}"},
        pseudonyms,
    )
    assert nested["routes"]["input"] == pseudonyms[raw_names[0]]
    assert nested["message"] == f"using {pseudonyms[raw_names[1]]}"


def test_hardware_privacy_filter_handles_empty_overlapping_and_case_variant_names() -> None:
    runs, pseudonyms = TOOL._privacy_filter_runs(
        [{"stdout": ["MIC ARRAY selected after Mic"]}],
        ["", "Mic", "Mic Array"],
        key=b"q" * 32,
    )

    assert "" not in pseudonyms
    assert set(pseudonyms) == {"Mic", "Mic Array"}
    output = runs[0]["stdout"][0]
    assert output == f"{pseudonyms['Mic Array']} selected after {pseudonyms['Mic']}"


def test_hardware_evaluation_rejects_empty_device_names_before_running(tmp_path) -> None:
    with pytest.raises(ValueError, match="health input"):
        TOOL.evaluate(
            health_input="",
            health_output="Output",
            correlation_input="Loopback",
            correlation_output="Output",
            health_duration=1.0,
            report_path=tmp_path / "report.json",
        )


def _matrix_case(
    *,
    case_id: str,
    os_release: str,
    device_class: str,
    sample_rate: int,
    scenario: str,
    archive_sha256: str,
) -> dict:
    return {
        "schema_version": 3,
        "qualification_kind": "exact-artifact-hardware",
        "status": "passed",
        "passed": True,
        "source_revision": "a" * 40,
        "artifact": {"archive_sha256": archive_sha256},
        "machine": {"release": os_release},
        "case": {
            "id": case_id,
            "device_class": device_class,
            "nominal_sample_rate_hz": sample_rate,
            "scenario": scenario,
            "evidence_kind": (
                "automated" if scenario == "baseline" else "operator_observed"
            ),
            "operator_attestation": scenario != "baseline",
            "scenario_evidence_valid": True,
        },
        "requested_health_duration_seconds": 1800.0,
        "package_smoke": {"passed": True},
        "executable_startup": {"passed": True},
        "model_discovery": {"passed": True},
        "selected_route_correlation": {"passed": True},
        "sustained_health": {"passed": True},
        "routes": {
            "correlation": {
                "input": "device-0123456789abcdef",
                "output": "device-fedcba9876543210",
            },
            "sustained_health": {
                "input": "device-0123456789abcdef",
                "output": "device-fedcba9876543210",
            },
        },
    }


def test_hardware_matrix_accepts_one_digest_bound_automated_baseline(tmp_path) -> None:
    archive_hash = "a" * 64
    path = tmp_path / "win11-virtual-baseline.json"
    path.write_text(
        json.dumps(
            _matrix_case(
                case_id="win11-virtual-baseline",
                os_release="11",
                device_class="virtual",
                sample_rate=48_000,
                scenario="baseline",
                archive_sha256=archive_hash,
            )
        ),
        encoding="utf-8",
    )

    result = MATRIX_TOOL.aggregate(
        [path],
        expected_archive_sha256=archive_hash,
        output=tmp_path / "matrix.json",
    )

    assert result["passed"] is True
    assert result["coverage"]["missing"]["automated_baseline_cases"] == 0


def test_hardware_matrix_requires_an_automated_baseline_without_fabrication(
    tmp_path,
) -> None:
    archive_hash = "b" * 64
    case_path = tmp_path / "baseline.json"
    case_path.write_text(
        json.dumps(
            _matrix_case(
                case_id="win11-usb-reconnect",
                os_release="11",
                device_class="usb",
                sample_rate=48_000,
                scenario="device_reconnect",
                archive_sha256=archive_hash,
            )
        ),
        encoding="utf-8",
    )

    result = MATRIX_TOOL.aggregate(
        [case_path],
        expected_archive_sha256=archive_hash,
        output=tmp_path / "matrix.json",
        allow_incomplete=True,
    )

    assert result["passed"] is False
    assert result["coverage"]["missing"]["automated_baseline_cases"] == 1


def test_hardware_matrix_rejects_forged_top_level_pass(tmp_path) -> None:
    archive_hash = "c" * 64
    case = _matrix_case(
        case_id="forged",
        os_release="11",
        device_class="usb",
        sample_rate=48_000,
        scenario="device_reconnect",
        archive_sha256=archive_hash,
    )
    case["requested_health_duration_seconds"] = 30.0
    case["case"]["evidence_kind"] = "automated"
    case["case"]["operator_attestation"] = False
    case["case"]["scenario_evidence_valid"] = False
    case["sustained_health"] = {"passed": False}
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(case), encoding="utf-8")

    result = MATRIX_TOOL.aggregate(
        [path],
        expected_archive_sha256=archive_hash,
        expected_source_revision="d" * 40,
        output=tmp_path / "matrix.json",
        allow_incomplete=True,
    )

    assert result["passed"] is False
    assert any("below 1800" in error for error in result["errors"])
    assert any("operator evidence" in error for error in result["errors"])
    assert any("operator attestation" in error for error in result["errors"])
    assert any("sustained_health did not pass" in error for error in result["errors"])
    assert any("source revision differs" in error for error in result["errors"])
