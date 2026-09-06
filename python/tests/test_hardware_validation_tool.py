from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

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
BUNDLE_RUNTIME = _load_tool("bundle_runtime")


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


def test_power_event_reader_rejects_unbounded_or_entity_xml(monkeypatch) -> None:
    valid_xml = """
    <Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
      <System>
        <Provider Name="Microsoft-Windows-Kernel-Power" />
        <EventID>42</EventID>
        <TimeCreated SystemTime="2026-09-06T14:00:00Z" />
      </System>
    </Event>
    """
    entity_xml = '<!DOCTYPE Event [<!ENTITY unused "expanded">]>' + valid_xml
    oversized_xml = valid_xml.replace(
        "</Event>", "<EventData>" + ("x" * 32) + "</EventData></Event>"
    )
    outputs = iter((valid_xml, entity_xml, oversized_xml))

    monkeypatch.setattr(TOOL.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        TOOL, "MAX_POWER_EVENT_FRAGMENT_CHARS", len(valid_xml) + 1
    )
    monkeypatch.setattr(
        TOOL.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=next(outputs), stderr=""
        ),
    )

    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert len(TOOL._read_power_events(since)) == 1
    assert TOOL._read_power_events(since) == []
    assert TOOL._read_power_events(since) == []


def test_hardware_report_provenance_uses_project_version_and_dirty_revision(
    monkeypatch,
) -> None:
    with (TOOL.REPO_ROOT / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]
    assert TOOL._project_version() == expected_version

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    outputs = iter((Result("abc123\n"), Result(" M changed.py\n")))
    monkeypatch.setattr(TOOL.subprocess, "run", lambda *args, **kwargs: next(outputs))

    assert TOOL._source_revision() == "abc123+uncommitted"


def test_hardware_subprocess_uses_source_python_path(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(TOOL.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "inherited")

    result = TOOL._run(["python", "probe.py"])

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(TOOL.REPO_ROOT / "python"),
        "inherited",
    ]
    assert result["return_code"] == 0


def test_bundle_runtime_pins_the_bundled_vad_model(tmp_path, monkeypatch) -> None:
    internal = tmp_path / "_internal"
    model_root = internal / "models"
    native_root = internal / "mic_eq"
    model_root.mkdir(parents=True)
    native_root.mkdir()
    (tmp_path / "AudioForge.exe").write_bytes(b"exe")
    (internal / "df.dll").write_bytes(b"dll")
    vad_model = model_root / "silero_vad.onnx"
    vad_model.write_bytes(b"vad")
    (native_root / "mic_eq_core.test.pyd").write_bytes(b"native")

    module = ModuleType("mic_eq_core")
    configured: list[tuple[str, str]] = []
    setattr(
        module,
        "configure_deepfilter_runtime_paths",
        lambda library, models: configured.append((library, models)),
    )

    class Loader:
        @staticmethod
        def exec_module(_module) -> None:
            return None

    spec = SimpleNamespace(loader=Loader())
    monkeypatch.setattr(BUNDLE_RUNTIME.os, "add_dll_directory", lambda _path: object())
    monkeypatch.setattr(
        BUNDLE_RUNTIME.importlib.util,
        "spec_from_file_location",
        lambda *_args: spec,
    )
    monkeypatch.setattr(
        BUNDLE_RUNTIME.importlib.util,
        "module_from_spec",
        lambda _spec: module,
    )
    monkeypatch.delenv("VAD_MODEL_PATH", raising=False)

    loaded = BUNDLE_RUNTIME.load_bundled_core(tmp_path)

    assert loaded is module
    assert BUNDLE_RUNTIME.os.environ["VAD_MODEL_PATH"] == str(vad_model.resolve())
    assert configured == [(str((internal / "df.dll").resolve()), str(model_root.resolve()))]


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


def test_health_gate_requires_standard_deepfilter_inference_and_output() -> None:
    clean = {
        "noise_model": "deepfilter",
        "suppressor_latency_samples": 1_440,
        "suppressor_successful_inference_frames": 12,
        "output_true_peak_db": -18.0,
    }

    assert (
        HEALTH_TOOL._selected_noise_model_failures(
            clean, expected_model="deepfilter"
        )
        == []
    )
    clean["suppressor_latency_samples"] = 1_919
    assert (
        HEALTH_TOOL._selected_noise_model_failures(
            clean, expected_model="deepfilter"
        )
        == []
    )
    clean["suppressor_latency_samples"] = 1_920
    assert "suppressor_latency_samples=1920" in HEALTH_TOOL._selected_noise_model_failures(
        clean, expected_model="deepfilter"
    )

    broken = dict(clean)
    broken.update(
        noise_model="rnnoise",
        suppressor_latency_samples=480,
        suppressor_successful_inference_frames=0,
        output_true_peak_db=-120.0,
    )
    failures = HEALTH_TOOL._selected_noise_model_failures(
        broken, expected_model="deepfilter"
    )
    assert "noise_model='rnnoise'" in failures
    assert "suppressor_latency_samples=480" in failures
    assert "suppressor_successful_inference_frames=0" in failures
    assert "output_true_peak_db=-120.0" in failures


def test_lifecycle_model_probe_requires_new_inference_frames() -> None:
    diagnostics = {
        "noise_model": "deepfilter",
        "suppressor_latency_samples": 1_440,
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "suppressor_successful_inference_frames": 12,
        "output_true_peak_db": -18.0,
    }

    assert TOOL._model_diagnostics_healthy(
        diagnostics, "deepfilter", minimum_inference_frames=12
    ) is False
    diagnostics["suppressor_successful_inference_frames"] = 13
    assert TOOL._model_diagnostics_healthy(
        diagnostics, "deepfilter", minimum_inference_frames=12
    ) is True


def test_lifecycle_settle_rejects_diagnostic_counter_growth() -> None:
    baseline: dict[str, Any] = {
        key: 0 for key in HEALTH_TOOL._ZERO_REQUIRED_DIAGNOSTICS
    }
    baseline.update(
        {
            "output_underrun_total": 0,
            "noise_backend_available": True,
            "noise_backend_failed": False,
            "last_stream_error": None,
        }
    )
    current = dict(baseline)
    current["output_recovery_count"] = 1

    assert "output_recovery_count changed" in TOOL._stable_diagnostic_failures(
        baseline, current
    )


@pytest.mark.parametrize("new_underruns", [0, 1])
def test_lifecycle_model_probe_uses_bundled_deepfilter_and_restores_cleanly(
    monkeypatch, tmp_path, new_underruns
) -> None:
    input_device = SimpleNamespace(
        name="Input", endpoint_id="input-endpoint", is_default=True, sample_rate=48_000
    )
    output_device = SimpleNamespace(
        name="Output", endpoint_id="output-endpoint", is_default=True, sample_rate=48_000
    )
    environment_seen: list[str | None] = []

    def diagnostics(model: str, frames: int) -> dict[str, object]:
        result: dict[str, Any] = {
            key: 0 for key in HEALTH_TOOL._ZERO_REQUIRED_DIAGNOSTICS
        }
        result.update(
            {
                "output_underrun_total": new_underruns if model == "deepfilter" else 0,
                "noise_backend_available": True,
                "noise_backend_failed": False,
                "last_stream_error": None,
                "noise_model": model,
                "suppressor_latency_samples": (
                    1_440 if model == "deepfilter" else 480
                ),
                "suppressor_successful_inference_frames": frames,
                "output_true_peak_db": -18.0,
            }
        )
        return result

    class Processor:
        def __init__(self) -> None:
            self.model = "rnnoise"
            self.deepfilter_reads = 0

        def start(self, _input: str, _output: str) -> None:
            return None

        def stop(self) -> None:
            return None

        def get_active_input_device(self) -> str:
            return "Input"

        def get_active_output_device(self) -> str:
            return "Output"

        def get_input_callback_age_ms(self) -> int:
            return 1

        def get_output_callback_age_ms(self) -> int:
            return 1

        def get_runtime_diagnostics(self) -> dict[str, object]:
            if self.model == "deepfilter":
                self.deepfilter_reads += 1
                return diagnostics("deepfilter", max(0, self.deepfilter_reads - 1))
            return diagnostics("rnnoise", 0)

        def service_recovery(self) -> None:
            return None

        def list_noise_models(self) -> list[tuple[str, str]]:
            return [("rnnoise", "RNNoise"), ("deepfilter", "DeepFilter")]

        def get_noise_model(self) -> str:
            return self.model

        def set_noise_model(self, model: str) -> bool:
            self.model = model
            self.deepfilter_reads = 0
            return True

    def runtime_api(_bundle_root: Path | None):
        environment_seen.append(os.environ.get("AUDIOFORGE_ENABLE_DEEPFILTER"))
        return (
            Processor,
            lambda: [input_device],
            lambda: [output_device],
        )

    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.setattr(TOOL, "_runtime_api", runtime_api)
    monkeypatch.setattr(TOOL, "LIFECYCLE_POLL_SECONDS", 0.001)
    result = TOOL._run_lifecycle_probe(
        scenario="model_configuration_change",
        health_input="Input",
        health_output="Output",
        bundle_root=tmp_path,
        timeout_seconds=0.5,
        settle_seconds=0.0,
    )

    if new_underruns:
        assert result["passed"] is False
        assert result["event"]["reason"] == "model_switch_not_observed"
        return

    assert result["passed"] is True, json.dumps(result, indent=2)
    assert result["event"]["alternate_inference_frames"] == 1
    assert result["event"]["latency_samples_before"] == 480
    assert result["event"]["latency_samples_alternate"] == 1_440
    assert result["event"]["latency_samples_restored"] == 480
    assert environment_seen == ["1"]
    assert "AUDIOFORGE_ENABLE_DEEPFILTER" not in os.environ


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


def test_hardware_privacy_filter_redacts_windows_endpoint_ids() -> None:
    endpoint_id = "{0.0.1.00000000}.{12345678-1234-1234-1234-123456789abc}"
    runs, _pseudonyms = TOOL._privacy_filter_runs(
        [{"stderr": [f"WASAPI endpoint {endpoint_id} failed"]}], []
    )

    assert endpoint_id not in json.dumps(runs)
    assert "endpoint-redacted" in runs[0]["stderr"][0]


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


def _lifecycle_evidence(scenario: str) -> dict:
    event = {
        "observed": True,
        "backend_observed": True,
    }
    if scenario == "device_reconnect":
        event.update(
            selected_endpoint_absent=True,
            selected_endpoint_reappeared=True,
        )
    elif scenario == "default_device_change":
        event.update(
            default_endpoint_changed=True,
            selected_route_correct=True,
        )
    elif scenario == "sleep_resume":
        event.update(os_suspend_event=True, os_resume_event=True)
    else:
        event.update(
            model_switched=True,
            model_restored=True,
            diagnostics_healthy=True,
        )
    return {
        "scenario": scenario,
        "passed": True,
        "bounded": True,
        "event": event,
        "recovery": {
            "bounded": True,
            "recovered": True,
            "settled_clean": True,
        },
        "diagnostics": {"before": {}, "after": {}},
    }


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
            "observed_input_sample_rate_hz": sample_rate,
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
        "lifecycle_evidence": (
            None if scenario == "baseline" else _lifecycle_evidence(scenario)
        ),
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


def _complete_matrix_cases(archive_hash: str) -> list[dict]:
    return [
        _matrix_case(
            case_id="win10-built-in-baseline",
            os_release="10",
            device_class="built_in",
            sample_rate=44_100,
            scenario="baseline",
            archive_sha256=archive_hash,
        ),
        _matrix_case(
            case_id="win11-usb-reconnect",
            os_release="11",
            device_class="usb",
            sample_rate=48_000,
            scenario="device_reconnect",
            archive_sha256=archive_hash,
        ),
        _matrix_case(
            case_id="win11-virtual-default-device",
            os_release="11",
            device_class="virtual",
            sample_rate=44_100,
            scenario="default_device_change",
            archive_sha256=archive_hash,
        ),
        _matrix_case(
            case_id="win10-built-in-sleep-resume",
            os_release="10",
            device_class="built_in",
            sample_rate=48_000,
            scenario="sleep_resume",
            archive_sha256=archive_hash,
        ),
        _matrix_case(
            case_id="win11-usb-model-configuration",
            os_release="11",
            device_class="usb",
            sample_rate=44_100,
            scenario="model_configuration_change",
            archive_sha256=archive_hash,
        ),
    ]


def test_hardware_matrix_accepts_required_risk_based_coverage(tmp_path) -> None:
    archive_hash = "a" * 64
    paths = []
    for index, case in enumerate(_complete_matrix_cases(archive_hash)):
        path = tmp_path / f"case-{index}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)

    result = MATRIX_TOOL.aggregate(
        paths,
        expected_archive_sha256=archive_hash,
        output=tmp_path / "matrix.json",
    )

    assert result["passed"] is True
    assert result["coverage"]["missing"]["automated_baseline_cases"] == 0
    assert result["coverage"]["missing"]["scenarios"] == []


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
    assert result["coverage"]["missing"]["device_classes"] == ["built_in", "virtual"]
    assert "model_configuration_change" in result["coverage"]["missing"]["scenarios"]


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
    case["requested_health_duration_seconds"] = float("nan")
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
