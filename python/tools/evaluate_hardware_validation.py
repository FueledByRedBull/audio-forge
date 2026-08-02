"""Run and record selected-route correlation plus sustained audio health."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "models" / "evaluation-details" / "hardware-validation.json"
SELF_TEST_RESULT = re.compile(
    r"Self-test passed: rt=(?P<latency>[0-9.]+)ms confidence=(?P<confidence>[0-9.]+)"
)
HEALTH_SUMMARY = re.compile(
    r"Health summary: max_input_age_ms=(?P<input_age>\d+) "
    r"max_output_age_ms=(?P<output_age>\d+) restarts=(?P<restarts>\d+) "
    r"underrun_baseline=(?P<underrun_baseline>\d+) "
    r"diagnostics=(?P<diagnostics>\{.*\})"
)
HARDWARE_SCENARIOS = (
    "baseline",
    "device_reconnect",
    "default_device_change",
    "sleep_resume",
    "buffer_negotiation",
    "route_change",
)
DEVICE_CLASSES = ("built_in", "usb", "virtual", "other")
EVIDENCE_KINDS = ("automated", "operator_observed")


def _project_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _source_revision() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{head}+uncommitted" if dirty else head


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_build_info(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "_internal" / "audioforge-build.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid bundle build metadata: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        raise RuntimeError("bundle build metadata lacks a string version")
    return value


def _tree_fingerprint(bundle_root: Path) -> dict[str, Any]:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(
        (candidate for candidate in bundle_root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(bundle_root)
        .as_posix()
        .casefold(),
    ):
        relative = path.relative_to(bundle_root).as_posix()
        rows.append((relative, path.stat().st_size, _sha256(path)))
    digest = hashlib.sha256()
    for relative, size, file_hash in rows:
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "file_count": len(rows),
        "total_bytes": sum(size for _relative, size, _hash in rows),
        "normalized_tree_sha256": digest.hexdigest(),
    }


def _artifact_provenance(
    archive: Path,
    checksum: Path,
    bundle_root: Path,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    archive = archive.resolve(strict=True)
    checksum = checksum.resolve(strict=True)
    bundle_root = bundle_root.resolve(strict=True)
    actual_hash = _sha256(archive)
    expected_hash = expected_archive_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("expected archive SHA-256 must contain 64 lowercase hex digits")
    sidecar_fields = checksum.read_text(encoding="utf-8-sig").strip().split()
    if len(sidecar_fields) < 2:
        raise RuntimeError("checksum sidecar is malformed")
    sidecar_hash = sidecar_fields[0].lower()
    sidecar_name = sidecar_fields[-1].lstrip("*")
    if sidecar_name != archive.name:
        raise RuntimeError(
            f"checksum sidecar names {sidecar_name!r}, expected {archive.name!r}"
        )
    if actual_hash != expected_hash or sidecar_hash != expected_hash:
        raise RuntimeError(
            "archive hash, expected hash, and checksum sidecar do not match"
        )
    return {
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": actual_hash,
        "sha256": actual_hash,
        "checksum_name": checksum.name,
        "checksum_sha256": _sha256(checksum),
        "bundle": _tree_fingerprint(bundle_root),
        "build": _bundle_build_info(bundle_root),
    }


def _portable_runtime_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display_path = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_path = resolved.name
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _runtime_provenance(bundle_root: Path | None = None) -> dict[str, Any]:
    if bundle_root is None:
        native_spec = importlib.util.find_spec("mic_eq.mic_eq_core")
        if native_spec is None or native_spec.origin is None:
            raise RuntimeError("mic_eq native extension is unavailable")
        native = _portable_runtime_file(Path(native_spec.origin))
    else:
        candidates = list(
            (bundle_root / "_internal" / "mic_eq").glob("mic_eq_core*.pyd")
        )
        if len(candidates) != 1:
            raise RuntimeError("bundle must contain exactly one native extension")
        native = {
            "path": "_internal/mic_eq/" + candidates[0].name,
            "bytes": candidates[0].stat().st_size,
            "sha256": _sha256(candidates[0]),
        }
    return {
        "native_extension": native,
        "self_test": _portable_runtime_file(REPO_ROOT / "python/tools/self_test.py"),
        "health_check": _portable_runtime_file(
            REPO_ROOT / "python/tools/health_check.py"
        ),
        "latency_analysis": _portable_runtime_file(
            REPO_ROOT / "python/mic_eq/analysis/latency_calibration.py"
        ),
    }


def _run(command: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    source_python = str(REPO_ROOT / "python")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_python, inherited_pythonpath))
        if inherited_pythonpath
        else source_python
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "return_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout.strip().splitlines(),
        "stderr": completed.stderr.strip().splitlines(),
    }


def _device_pseudonyms(
    device_names: list[str],
    *,
    key: bytes | None = None,
) -> dict[str, str]:
    """Create report-local, unlinkable endpoint pseudonyms."""
    pseudonym_key = key or secrets.token_bytes(32)
    mapping: dict[str, str] = {}
    for raw_name in dict.fromkeys(device_names):
        normalized_name = raw_name.strip()
        if not normalized_name:
            continue
        digest = hmac.new(
            pseudonym_key,
            normalized_name.casefold().encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]
        mapping[raw_name] = f"device-{digest}"
    return mapping


def _replace_private_strings(value: Any, mapping: dict[str, str]) -> Any:
    """Recursively replace selected endpoint names in report values."""
    if isinstance(value, str):
        result = value
        for raw_name in sorted(mapping, key=len, reverse=True):
            result = re.sub(
                re.escape(raw_name),
                mapping[raw_name],
                result,
                flags=re.IGNORECASE,
            )
        return result
    if isinstance(value, dict):
        return {
            key: _replace_private_strings(child, mapping)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_private_strings(child, mapping) for child in value]
    return value


def _privacy_filter_runs(
    runs: list[dict[str, Any]],
    device_names: list[str],
    *,
    key: bytes | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Remove raw endpoint names from persisted command output."""
    mapping = _device_pseudonyms(device_names, key=key)

    filtered = [_replace_private_strings(run, mapping) for run in runs]
    return filtered, mapping


def _package_smoke(bundle_root: Path, bundle_version: str) -> dict[str, Any]:
    command = [
        str(Path(sys.executable).resolve()),
        "python/tools/package_smoke.py",
        "--dist",
        str(bundle_root),
    ]
    if bundle_version == "1.10.1":
        command.extend(
            ["--allow-historical-ucrt-for-version", bundle_version]
        )
    result = _run(command)
    return {
        **result,
        "passed": int(result["return_code"]) == 0,
        "historical_ucrt_exception": (
            "exact v1.10.1 46-file payload" if bundle_version == "1.10.1" else None
        ),
    }


def _hidden_executable_startup(bundle_root: Path, duration_seconds: float = 12.0) -> dict[str, Any]:
    executable = bundle_root / "AudioForge.exe"
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(executable)],
        cwd=bundle_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    exited_early = False
    exit_code: int | None = None
    try:
        exit_code = process.wait(timeout=duration_seconds)
        exited_early = True
    except subprocess.TimeoutExpired:
        pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
    return {
        "passed": not exited_early,
        "probe_seconds": duration_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "exited_early": exited_early,
        "early_exit_code": exit_code,
        "mode": "QT_QPA_PLATFORM=offscreen",
    }


def _load_bundle_runtime_module() -> Any:
    path = REPO_ROOT / "python/tools/bundle_runtime.py"
    spec = importlib.util.spec_from_file_location("audioforge_bundle_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bundle runtime helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundled_model_discovery(bundle_root: Path) -> dict[str, Any]:
    previous = os.environ.get("AUDIOFORGE_ENABLE_DEEPFILTER")
    os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = "1"
    try:
        core = _load_bundle_runtime_module().load_bundled_core(bundle_root)
        processor = core.AudioProcessor()
        models = [
            {"id": str(identifier), "name": str(name)}
            for identifier, name in processor.list_noise_models()
        ]
    finally:
        if previous is None:
            os.environ.pop("AUDIOFORGE_ENABLE_DEEPFILTER", None)
        else:
            os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = previous
    model_ids = {row["id"] for row in models}
    expected = {"rnnoise", "deepfilter-ll", "deepfilter"}
    return {
        "passed": expected <= model_ids,
        "models": models,
        "expected_model_ids": sorted(expected),
        "missing_model_ids": sorted(expected - model_ids),
    }


def _parse_self_test(result: dict[str, Any]) -> dict[str, Any]:
    output = "\n".join(result["stdout"])
    match = SELF_TEST_RESULT.search(output)
    return {
        "passed": int(result["return_code"]) == 0 and match is not None,
        "route_latency_ms": float(match.group("latency")) if match else None,
        "confidence": float(match.group("confidence")) if match else None,
    }


def _parse_health(result: dict[str, Any]) -> dict[str, Any]:
    output = "\n".join(result["stdout"])
    match = HEALTH_SUMMARY.search(output)
    return {
        "passed": int(result["return_code"]) == 0 and match is not None,
        "max_input_callback_age_ms": int(match.group("input_age")) if match else None,
        "max_output_callback_age_ms": int(match.group("output_age")) if match else None,
        "stream_restarts": int(match.group("restarts")) if match else None,
        "output_underrun_baseline": (
            int(match.group("underrun_baseline")) if match else None
        ),
        "runtime_diagnostics": json.loads(match.group("diagnostics")) if match else {},
    }


def evaluate(
    *,
    health_input: str,
    health_output: str,
    correlation_input: str,
    correlation_output: str,
    health_duration: float,
    report_path: Path,
    bundle_root: Path | None = None,
    archive: Path | None = None,
    checksum: Path | None = None,
    expected_archive_sha256: str | None = None,
    case_id: str = "local-baseline",
    device_class: str = "other",
    nominal_sample_rate_hz: int = 48_000,
    scenario: str = "baseline",
    evidence_kind: str = "automated",
    confirm_scenario_observed: bool = False,
) -> dict[str, Any]:
    device_names = {
        "health input": health_input,
        "health output": health_output,
        "correlation input": correlation_input,
        "correlation output": correlation_output,
    }
    for label, device_name in device_names.items():
        if (
            not isinstance(device_name, str)
            or not device_name.strip()
            or len(device_name) > 1024
            or "\n" in device_name
            or "\r" in device_name
        ):
            raise ValueError(f"{label} must be a bounded non-empty device name")
    if (
        isinstance(health_duration, bool)
        or not isinstance(health_duration, (int, float))
        or not math.isfinite(float(health_duration))
        or not 0.0 < float(health_duration) <= 86_400.0
    ):
        raise ValueError("health duration must be between 0 and 86400 seconds")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", case_id):
        raise ValueError("case ID must be a portable lowercase identifier")
    if device_class not in DEVICE_CLASSES:
        raise ValueError(f"unsupported device class: {device_class}")
    if nominal_sample_rate_hz not in {44_100, 48_000}:
        raise ValueError("nominal sample rate must be 44100 or 48000 Hz")
    if scenario not in HARDWARE_SCENARIOS:
        raise ValueError(f"unsupported hardware scenario: {scenario}")
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError(f"unsupported evidence kind: {evidence_kind}")
    python = str(Path(sys.executable).resolve())
    artifact: dict[str, Any] | None = None
    package_smoke: dict[str, Any] | None = None
    executable_startup: dict[str, Any] | None = None
    model_discovery: dict[str, Any] | None = None
    bundle_arguments: list[str] = []
    if bundle_root is not None:
        if archive is None or checksum is None or expected_archive_sha256 is None:
            raise ValueError(
                "bundle qualification requires archive, checksum, and expected SHA-256"
            )
        bundle_root = bundle_root.resolve(strict=True)
        artifact = _artifact_provenance(
            archive,
            checksum,
            bundle_root,
            expected_archive_sha256,
        )
        package_smoke = _package_smoke(
            bundle_root, str(artifact["build"]["version"])
        )
        executable_startup = _hidden_executable_startup(bundle_root)
        model_discovery = _bundled_model_discovery(bundle_root)
        bundle_arguments = ["--bundle-root", str(bundle_root)]

    self_test = _run(
        [
            python,
            "python/tools/self_test.py",
            "--input-device",
            correlation_input,
            "--output-device",
            correlation_output,
            "--duration",
            "3",
            "--retries",
            "2",
            *bundle_arguments,
        ]
    )
    health = _run(
        [
            python,
            "python/tools/health_check.py",
            "--duration",
            str(health_duration),
            "--input-device",
            health_input,
            "--output-device",
            health_output,
            *bundle_arguments,
        ]
    )
    parsed_self_test = _parse_self_test(self_test)
    parsed_health = _parse_health(health)
    filtered_runs, device_pseudonyms = _privacy_filter_runs(
        [self_test, health],
        [health_input, health_output, correlation_input, correlation_output],
    )
    self_test_for_report, health_for_report = filtered_runs
    artifact_checks = (
        package_smoke is None
        or (
            package_smoke["passed"]
            and executable_startup is not None
            and executable_startup["passed"]
            and model_discovery is not None
            and model_discovery["passed"]
        )
    )
    scenario_evidence_valid = scenario == "baseline" or (
        evidence_kind == "operator_observed" and confirm_scenario_observed
    )
    passed = bool(
        artifact_checks
        and parsed_self_test["passed"]
        and parsed_health["passed"]
        and scenario_evidence_valid
    )
    diagnostics = parsed_health["runtime_diagnostics"]
    machine = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    report = {
        "schema_version": 3,
        "status": "passed" if passed else "failed",
        "qualification_kind": "exact-artifact-hardware",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_version": (
            str(artifact["build"]["version"])
            if artifact is not None
            else _project_version()
        ),
        "source_revision": _source_revision(),
        "runtime_provenance": _runtime_provenance(bundle_root),
        "machine": machine,
        "audible_change": False,
        "case": {
            "id": case_id,
            "device_class": device_class,
            "nominal_sample_rate_hz": nominal_sample_rate_hz,
            "scenario": scenario,
            "evidence_kind": evidence_kind,
            "automated_measurements": True,
            "operator_observation_required": scenario != "baseline",
            "operator_attestation": bool(confirm_scenario_observed),
            "scenario_evidence_valid": scenario_evidence_valid,
        },
        "purpose": (
            "Exact extracted release-artifact route and sustained callback health gate."
            if artifact is not None
            else "Release-machine selected-route and sustained callback health gate."
        ),
        "artifact": artifact,
        "package_smoke": package_smoke,
        "executable_startup": executable_startup,
        "model_discovery": model_discovery,
        "routes": {
            "correlation": {
                "input": device_pseudonyms[correlation_input],
                "output": device_pseudonyms[correlation_output],
            },
            "sustained_health": {
                "input": device_pseudonyms[health_input],
                "output": device_pseudonyms[health_output],
            },
        },
        "requested_health_duration_seconds": health_duration,
        "selected_route_correlation": {
            **parsed_self_test,
            "run": self_test_for_report,
        },
        "sustained_health": {**parsed_health, "run": health_for_report},
        "latency": {
            "route_round_trip_ms": parsed_self_test["route_latency_ms"],
            "engine_ms": diagnostics.get("engine_latency_ms"),
            "total_ms": diagnostics.get("total_latency_ms"),
            "configured_compensation_ms": (
                float(diagnostics.get("total_latency_ms", 0.0))
                - float(diagnostics.get("engine_latency_ms", 0.0))
                if isinstance(diagnostics.get("total_latency_ms"), (int, float))
                and isinstance(diagnostics.get("engine_latency_ms"), (int, float))
                else None
            ),
        },
        "passed": passed,
        "limitations": [
            (
                "One Windows machine and one "
                f"{device_class.replace('_', '-')} device-route case."
            ),
            (
                "Both checks use the explicitly selected routes; physical topology "
                "is not inferred from device names."
            ),
            "This is objective device/runtime evidence, not a perceptual listening session.",
            "Device names are replaced with report-local HMAC pseudonyms in routes and logs.",
            (
                "For extracted bundles, external source-controlled harnesses load the "
                "exact bundled native extension/DLL/models; a separate offscreen launch "
                "checks the bundled Python/UI executable."
            ),
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("hardware validation failed; inspect generated report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-input", required=True)
    parser.add_argument("--health-output", required=True)
    parser.add_argument("--correlation-input", required=True)
    parser.add_argument("--correlation-output", required=True)
    parser.add_argument("--health-duration", type=float, default=1800.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        help="exact extracted release bundle to qualify",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="release archive corresponding to --bundle-root",
    )
    parser.add_argument(
        "--checksum",
        type=Path,
        help="published SHA-256 sidecar corresponding to --archive",
    )
    parser.add_argument(
        "--expected-archive-sha256",
        help="independently known SHA-256 required for bundle qualification",
    )
    parser.add_argument("--case-id", default="local-baseline")
    parser.add_argument("--device-class", choices=DEVICE_CLASSES, default="other")
    parser.add_argument(
        "--nominal-sample-rate-hz",
        type=int,
        choices=(44_100, 48_000),
        default=48_000,
    )
    parser.add_argument("--scenario", choices=HARDWARE_SCENARIOS, default="baseline")
    parser.add_argument(
        "--evidence-kind", choices=EVIDENCE_KINDS, default="automated"
    )
    parser.add_argument(
        "--confirm-scenario-observed",
        action="store_true",
        help=(
            "attest that the selected non-baseline lifecycle event was "
            "actually performed and observed during this run"
        ),
    )
    args = parser.parse_args()
    report = evaluate(
        health_input=args.health_input,
        health_output=args.health_output,
        correlation_input=args.correlation_input,
        correlation_output=args.correlation_output,
        health_duration=args.health_duration,
        report_path=args.report.resolve(),
        bundle_root=args.bundle_root,
        archive=args.archive,
        checksum=args.checksum,
        expected_archive_sha256=args.expected_archive_sha256,
        case_id=args.case_id,
        device_class=args.device_class,
        nominal_sample_rate_hz=args.nominal_sample_rate_hz,
        scenario=args.scenario,
        evidence_kind=args.evidence_kind,
        confirm_scenario_observed=args.confirm_scenario_observed,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "route_latency_ms": report["selected_route_correlation"][
                    "route_latency_ms"
                ],
                "correlation_confidence": report["selected_route_correlation"][
                    "confidence"
                ],
                "max_input_callback_age_ms": report["sustained_health"][
                    "max_input_callback_age_ms"
                ],
                "max_output_callback_age_ms": report["sustained_health"][
                    "max_output_callback_age_ms"
                ],
                "stream_restarts": report["sustained_health"]["stream_restarts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
