from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
    assert TOOL._project_version() == "1.10.1"

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
