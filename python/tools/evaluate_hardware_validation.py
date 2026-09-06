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
import xml.etree.ElementTree as ET  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from health_check import (
    _ZERO_REQUIRED_DIAGNOSTICS,
    _critical_diagnostic_failures,
    _suppressor_latency_is_valid,
)
from hardware_qualification import (
    PHYSICAL_SCENARIOS,
    SUPPORTED_DEVICE_CLASSES,
    SUPPORTED_SCENARIOS,
)
from release_provenance import sha256_file as _sha256


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
HARDWARE_SCENARIOS = tuple(sorted(SUPPORTED_SCENARIOS))
DEVICE_CLASSES = tuple(sorted(SUPPORTED_DEVICE_CLASSES))
EVIDENCE_KINDS = ("automated", "operator_observed")
LIFECYCLE_PROBE_TIMEOUT_SECONDS = 120.0
LIFECYCLE_SETTLE_SECONDS = 5.0
LIFECYCLE_POLL_SECONDS = 0.5
CALLBACK_UNKNOWN_AGE_MS = 1 << 63
POWER_EVENT_IDS = {
    ("microsoft-windows-kernel-power", 42): ("sleep", "s3"),
    ("microsoft-windows-kernel-power", 107): ("resume", "s3"),
    ("microsoft-windows-power-troubleshooter", 1): ("resume", "s3"),
    ("microsoft-windows-kernel-power", 506): ("sleep", "modern_standby"),
    ("microsoft-windows-kernel-power", 507): ("resume", "modern_standby"),
}
POWER_EVENT_QUERY = (
    "*[System[(EventID=1 or EventID=42 or EventID=107 or "
    "EventID=506 or EventID=507) "
    "and (Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] or "
    "Provider[@Name='Microsoft-Windows-Kernel-Power'])]]"
)
MAX_POWER_EVENT_OUTPUT_CHARS = 16 * 1024 * 1024
MAX_POWER_EVENT_FRAGMENT_CHARS = 1 * 1024 * 1024
POWER_EVENT_DTD_MARKER = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


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
        return _WINDOWS_ENDPOINT_ID.sub("endpoint-redacted", result)
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



def _runtime_api(bundle_root: Path | None) -> tuple[Any, Any, Any]:
    if bundle_root is None:
        from mic_eq import AudioProcessor, list_input_devices, list_output_devices

        return AudioProcessor, list_input_devices, list_output_devices
    core = _load_bundle_runtime_module().load_bundled_core(bundle_root)
    return core.AudioProcessor, core.list_input_devices, core.list_output_devices


def _device_key(device: Any) -> str | None:
    endpoint_id = str(getattr(device, "endpoint_id", "") or "").strip()
    return f"endpoint:{endpoint_id}" if endpoint_id else None


def _device_snapshot(
    list_input_devices: Any,
    list_output_devices: Any,
    input_name: str,
    output_name: str,
) -> dict[str, Any]:
    def direction(devices: list[Any], selected_name: str) -> dict[str, Any]:
        selected = [device for device in devices if device.name == selected_name]
        return {
            "keys": tuple(
                key for device in selected if (key := _device_key(device)) is not None
            ),
            "defaults": tuple(
                key
                for device in devices
                if device.is_default and (key := _device_key(device)) is not None
            ),
            "count": len(selected),
            "endpoint_count": sum(_device_key(device) is not None for device in selected),
            "sample_rates": tuple(
                int(device.sample_rate)
                for device in selected
                if device.sample_rate is not None
            ),
        }

    return {
        "input": direction(list(list_input_devices()), input_name),
        "output": direction(list(list_output_devices()), output_name),
    }


def _callbacks_are_healthy(processor: Any) -> bool:
    try:
        input_age = int(processor.get_input_callback_age_ms())
        output_age = int(processor.get_output_callback_age_ms())
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        0 <= input_age < CALLBACK_UNKNOWN_AGE_MS
        and 0 <= output_age < CALLBACK_UNKNOWN_AGE_MS
        and input_age <= 2_000
        and output_age <= 2_000
    )


def _diagnostic_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        *_ZERO_REQUIRED_DIAGNOSTICS,
        "noise_backend_available",
        "noise_backend_failed",
        "output_underrun_total",
        "noise_model",
        "suppressor_successful_inference_frames",
    )
    summary = {key: diagnostics.get(key) for key in keys if key in diagnostics}
    summary["last_stream_error_present"] = bool(diagnostics.get("last_stream_error"))
    return summary


_EVENT_DISRUPTION_COUNTERS = frozenset(
    {
        "input_callback_error_count",
        "output_callback_error_count",
        "output_recovery_count",
        "output_recovery_event_count",
        "output_underrun_total",
        "output_underrun_streak",
        "stream_restart_count",
    }
)
_WINDOWS_ENDPOINT_ID = re.compile(
    r"(?i)(?<![a-z0-9])(?:\{\d+(?:\.\d+){3}\}\.\{[0-9a-f-]{36}\}|"
    r"\\\\\?\\(?:SWD|MMDEVAPI|HDAUDIO|USB)#[^\s\"']+)"
)


def _diagnostic_failures(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    event_window: bool = False,
) -> list[str]:
    allowed = _EVENT_DISRUPTION_COUNTERS if event_window else frozenset()
    failures = _critical_diagnostic_failures(
        after,
        output_underrun_baseline=(
            None
            if event_window
            else int(before.get("output_underrun_total", 0) or 0)
        ),
        ignored_zero_diagnostics=allowed,
    )
    if event_window:
        failures = [failure for failure in failures if failure != "last_stream_error=set"]
        for key in allowed:
            value = after.get(key)
            try:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise ValueError
            except (TypeError, ValueError):
                failures.append(f"{key}=invalid")
    for key in (*_ZERO_REQUIRED_DIAGNOSTICS, "output_underrun_total"):
        if key in allowed:
            continue
        try:
            if int(after[key]) != int(before[key]):
                failures.append(f"{key} changed")
        except (KeyError, TypeError, ValueError):
            failures.append(f"{key}=missing_or_invalid")
    return failures


def _stable_diagnostic_failures(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    for key in (*_ZERO_REQUIRED_DIAGNOSTICS, "output_underrun_total"):
        try:
            if int(current[key]) != int(baseline[key]):
                failures.append(f"{key} changed")
        except (KeyError, TypeError, ValueError):
            failures.append(f"{key}=missing_or_invalid")
    try:
        if int(current["output_underrun_streak"]) != 0:
            failures.append("output_underrun_streak=nonzero")
    except (KeyError, TypeError, ValueError):
        failures.append("output_underrun_streak=missing_or_invalid")
    if not bool(current.get("noise_backend_available", False)):
        failures.append("noise_backend_available=false")
    if bool(current.get("noise_backend_failed", False)):
        failures.append("noise_backend_failed=true")
    if current.get("last_stream_error"):
        failures.append("last_stream_error=set")
    return failures


def _model_diagnostics_healthy(
    diagnostics: dict[str, Any],
    model: str,
    *,
    minimum_inference_frames: int = 0,
) -> bool:
    if (
        diagnostics.get("noise_model") != model
        or not bool(diagnostics.get("noise_backend_available", False))
        or bool(diagnostics.get("noise_backend_failed", False))
    ):
        return False
    expected_latency = {
        "rnnoise": 480,
        "deepfilter-ll": 480,
        "deepfilter": 1_440,
    }.get(model)
    if expected_latency is not None and not _suppressor_latency_is_valid(
        diagnostics.get("suppressor_latency_samples"), expected_latency
    ):
        return False
    peak = diagnostics.get("output_true_peak_db")
    if (
        not isinstance(peak, (int, float))
        or not math.isfinite(float(peak))
        or float(peak) <= -119.0
    ):
        return False
    if model.startswith("deepfilter"):
        frames = diagnostics.get("suppressor_successful_inference_frames")
        if (
            not isinstance(frames, (int, float))
            or not math.isfinite(float(frames))
            or int(frames) <= minimum_inference_frames
        ):
            return False
    return True


def _read_power_events(since: datetime) -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []
    try:
        result = subprocess.run(
            [
                "wevtutil",
                "qe",
                "System",
                f"/q:{POWER_EVENT_QUERY}",
                "/f:xml",
                "/rd:true",
                "/c:128",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    if (
        len(result.stdout) > MAX_POWER_EVENT_OUTPUT_CHARS
        or POWER_EVENT_DTD_MARKER.search(result.stdout)
    ):
        return []
    events: list[dict[str, Any]] = []
    for match in re.finditer(r"<Event\b.*?</Event>", result.stdout, re.DOTALL):
        fragment = match.group(0)
        if len(fragment) > MAX_POWER_EVENT_FRAGMENT_CHARS:
            continue
        try:
            # Fixed local wevtutil XML is bounded; DTD/entity declarations are
            # rejected.
            root = ET.fromstring(fragment)
            system = next(
                child for child in root if child.tag.rsplit("}", 1)[-1] == "System"
            )
            event_id = next(
                int(child.text or "")
                for child in system
                if child.tag.rsplit("}", 1)[-1] == "EventID"
            )
            provider = next(
                child.attrib.get("Name", "")
                for child in system
                if child.tag.rsplit("}", 1)[-1] == "Provider"
            )
            created = next(
                child.attrib.get("SystemTime", "")
                for child in system
                if child.tag.rsplit("}", 1)[-1] == "TimeCreated"
            )
            timestamp = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (ET.ParseError, StopIteration, TypeError, ValueError):
            continue
        event_key = (provider.casefold(), event_id)
        if event_key not in POWER_EVENT_IDS or timestamp <= since:
            continue
        kind, family = POWER_EVENT_IDS[event_key]
        events.append(
            {
                "event_id": event_id,
                "kind": kind,
                "family": family,
                "provider": provider,
                "timestamp": timestamp.isoformat(),
            }
        )
    return sorted(events, key=lambda event: str(event["timestamp"]))


def _lifecycle_failure(
    scenario: str,
    reason: str,
    *,
    timeout_seconds: float,
    settle_seconds: float,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "passed": False,
        "bounded": True,
        "timeout_seconds": timeout_seconds,
        "settle_seconds": settle_seconds,
        "event": {"observed": False, "reason": reason},
        "recovery": {
            "bounded": True,
            "recovered": False,
            "settled_clean": False,
            "service_attempts": 0,
        },
        "diagnostics": {"before": {}, "after": {}},
    }


def _run_lifecycle_probe_impl(
    *,
    scenario: str,
    health_input: str,
    health_output: str,
    bundle_root: Path | None,
    timeout_seconds: float = LIFECYCLE_PROBE_TIMEOUT_SECONDS,
    settle_seconds: float = LIFECYCLE_SETTLE_SECONDS,
) -> dict[str, Any]:
    supported = PHYSICAL_SCENARIOS | {"model_configuration_change"}
    if scenario not in supported:
        return _lifecycle_failure(
            scenario,
            "scenario_probe_unimplemented",
            timeout_seconds=timeout_seconds,
            settle_seconds=settle_seconds,
        )
    try:
        processor_class, list_input_devices, list_output_devices = _runtime_api(
            bundle_root
        )
        processor = processor_class()
        before_devices = _device_snapshot(
            list_input_devices, list_output_devices, health_input, health_output
        )
    except Exception:
        return _lifecycle_failure(
            scenario,
            "runtime_or_device_observation_failed",
            timeout_seconds=timeout_seconds,
            settle_seconds=settle_seconds,
        )
    evidence = _lifecycle_failure(
        scenario,
        "probe_failed",
        timeout_seconds=timeout_seconds,
        settle_seconds=settle_seconds,
    )
    if any(
        before_devices[direction]["count"] != 1
        or before_devices[direction]["endpoint_count"] != 1
        for direction in ("input", "output")
    ):
        evidence["event"]["reason"] = "selected_route_is_ambiguous"
        return evidence
    rates = before_devices["input"]["sample_rates"]
    if len(rates) != 1:
        evidence["event"]["reason"] = "selected_input_rate_unavailable"
        return evidence
    evidence["observed_input_sample_rate_hz"] = rates[0]

    started_utc = datetime.now(timezone.utc)
    deadline = time.monotonic() + timeout_seconds

    def route_ok() -> bool:
        try:
            current = _device_snapshot(
                list_input_devices, list_output_devices, health_input, health_output
            )
            return (
                processor.get_active_input_device() == health_input
                and processor.get_active_output_device() == health_output
                and all(
                    current[direction]["count"] == 1
                    and current[direction]["endpoint_count"] == 1
                    and set(current[direction]["keys"])
                    == set(before_devices[direction]["keys"])
                    for direction in ("input", "output")
                )
            )
        except Exception:
            return False

    def settle(
        base: dict[str, Any],
        event_diagnostics: dict[str, Any] | None = None,
        post_recovery_ok: Any | None = None,
    ) -> tuple[bool, dict[str, Any], dict[str, Any]]:
        event_window = event_diagnostics is not None
        recovery: dict[str, Any] = {
            "bounded": True,
            "recovered": False,
            "settled_clean": False,
            "service_attempts": 0,
            "event_counter_deltas": {},
        }
        if event_diagnostics is not None:
            for key in (*_ZERO_REQUIRED_DIAGNOSTICS, "output_underrun_total"):
                try:
                    delta = int(event_diagnostics[key]) - int(base[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if delta:
                    recovery["event_counter_deltas"][key] = delta
        stable: dict[str, Any] | None = None
        recovered_since: float | None = None
        last = dict(base)
        while time.monotonic() < deadline:
            try:
                result = processor.service_recovery()
                if result is False:
                    recovery["reason"] = "recovery_failed"
                    break
                if result is not None:
                    recovery["service_attempts"] += 1
                current = dict(processor.get_runtime_diagnostics())
                if _diagnostic_failures(base, current, event_window=event_window):
                    recovery["reason"] = "diagnostics_changed_before_settle"
                    break
                healthy = (
                    route_ok()
                    and _callbacks_are_healthy(processor)
                    and (post_recovery_ok is None or bool(post_recovery_ok()))
                )
            except Exception:
                recovery["reason"] = "backend_observation_failed"
                break
            last = current
            if healthy:
                if stable is None:
                    stable = current
                    recovered_since = time.monotonic()
                    recovery["recovered"] = True
                    recovery["post_recovery_diagnostics"] = _diagnostic_summary(current)
                failures = _stable_diagnostic_failures(stable, current)
                if failures:
                    recovery["reason"] = "diagnostics_grew_after_recovery"
                    recovery["diagnostic_failures"] = failures
                    break
                if (
                    recovered_since is not None
                    and time.monotonic() - recovered_since >= settle_seconds
                ):
                    recovery["settled_clean"] = True
                    break
            else:
                stable = None
                recovered_since = None
                recovery["recovered"] = False
            time.sleep(LIFECYCLE_POLL_SECONDS)
        if not recovery["settled_clean"] and "reason" not in recovery:
            recovery["reason"] = "recovery_timeout"
        return bool(recovery["settled_clean"]), last, recovery

    try:
        processor.start(health_input, health_output)
        before_diagnostics: dict[str, Any] | None = None
        warmup_deadline = min(deadline, time.monotonic() + 10.0)
        while time.monotonic() < warmup_deadline:
            try:
                current = dict(processor.get_runtime_diagnostics())
                if (
                    route_ok()
                    and _callbacks_are_healthy(processor)
                    and not _diagnostic_failures(current, current)
                ):
                    before_diagnostics = current
                    break
            except Exception:
                pass
            time.sleep(LIFECYCLE_POLL_SECONDS)
        if before_diagnostics is None:
            evidence["event"]["reason"] = "initial_runtime_not_healthy"
            return evidence
        evidence["diagnostics"]["before"] = _diagnostic_summary(before_diagnostics)

        if scenario == "model_configuration_change":
            models = list(processor.list_noise_models())
            original = str(processor.get_noise_model())
            alternatives = [
                str(identifier)
                for identifier, _name in models
                if str(identifier) != original
            ]
            if not alternatives:
                evidence["event"]["reason"] = "no_alternate_model"
                return evidence
            preferred = "deepfilter" if "deepfilter" in alternatives else None
            alternate = preferred or alternatives[0]
            switched = False
            try:
                if not processor.set_noise_model(alternate):
                    evidence["event"]["reason"] = "model_switch_failed"
                    return evidence
                switched = True
                alternate_frames: int | None = None
                switched_diagnostics: dict[str, Any] | None = None
                switch_deadline = min(deadline, time.monotonic() + 15.0)
                while time.monotonic() < switch_deadline:
                    current = dict(processor.get_runtime_diagnostics())
                    frames = current.get("suppressor_successful_inference_frames")
                    if (
                        alternate_frames is None
                        and isinstance(frames, (int, float))
                        and math.isfinite(float(frames))
                    ):
                        alternate_frames = int(frames)
                    if (
                        processor.get_noise_model() == alternate
                        and _model_diagnostics_healthy(
                            current,
                            alternate,
                            minimum_inference_frames=alternate_frames or 0,
                        )
                        and _callbacks_are_healthy(processor)
                        and not _diagnostic_failures(before_diagnostics, current)
                    ):
                        switched_diagnostics = current
                        break
                    time.sleep(LIFECYCLE_POLL_SECONDS)
                if switched_diagnostics is None:
                    evidence["event"]["reason"] = "model_switch_not_observed"
                    return evidence
                if not processor.set_noise_model(original):
                    evidence["event"]["reason"] = "model_restore_failed"
                    return evidence
                original_inference_floor = 0
                original_frames = before_diagnostics.get(
                    "suppressor_successful_inference_frames"
                )
                if (
                    original.startswith("deepfilter")
                    and isinstance(original_frames, (int, float))
                    and math.isfinite(float(original_frames))
                ):
                    original_inference_floor = int(original_frames)
                restore_deadline = min(deadline, time.monotonic() + 15.0)
                restored_diagnostics: dict[str, Any] | None = None
                while time.monotonic() < restore_deadline:
                    current = dict(processor.get_runtime_diagnostics())
                    if (
                        processor.get_noise_model() == original
                        and _model_diagnostics_healthy(
                            current,
                            original,
                            minimum_inference_frames=original_inference_floor,
                        )
                        and _callbacks_are_healthy(processor)
                        and not _diagnostic_failures(before_diagnostics, current)
                    ):
                        restored_diagnostics = current
                        break
                    time.sleep(LIFECYCLE_POLL_SECONDS)
                if restored_diagnostics is None:
                    evidence["event"]["reason"] = "model_restore_not_observed"
                    return evidence
                recovered, after, recovery = settle(before_diagnostics)
                evidence["event"] = {
                    "observed": True,
                    "backend_observed": True,
                    "model_switched": True,
                    "model_restored": True,
                    "diagnostics_healthy": True,
                    "latency_samples_before": before_diagnostics.get(
                        "suppressor_latency_samples"
                    ),
                    "latency_samples_alternate": switched_diagnostics.get(
                        "suppressor_latency_samples"
                    ),
                    "latency_samples_restored": restored_diagnostics.get(
                        "suppressor_latency_samples"
                    ),
                    "alternate_inference_frames": switched_diagnostics.get(
                        "suppressor_successful_inference_frames"
                    ),
                }
                evidence["recovery"] = recovery
                evidence["diagnostics"]["after"] = _diagnostic_summary(after)
                evidence["passed"] = recovered
                return evidence
            finally:
                if switched:
                    try:
                        if processor.get_noise_model() != original:
                            processor.set_noise_model(original)
                    except Exception:
                        pass

        event_diagnostics: dict[str, Any] | None = None
        post_recovery_ok: Any | None = None
        if scenario == "device_reconnect":
            print(
                "Perform the selected endpoint unplug/reconnect now; "
                f"the qualification probe waits up to {timeout_seconds:.0f}s."
            )
            selected = set(before_devices["input"]["keys"]) | set(
                before_devices["output"]["keys"]
            )
            absent = False
            while time.monotonic() < deadline:
                current = _device_snapshot(
                    list_input_devices, list_output_devices, health_input, health_output
                )
                present = set(current["input"]["keys"]) | set(current["output"]["keys"])
                if not absent and not selected <= present:
                    absent = True
                if absent and selected <= present and route_ok():
                    event_diagnostics = dict(processor.get_runtime_diagnostics())
                    break
                time.sleep(LIFECYCLE_POLL_SECONDS)
            if event_diagnostics is None:
                evidence["event"]["reason"] = (
                    "selected_endpoint_absence_and_reappearance_not_observed"
                )
                return evidence
            event = {
                "observed": True,
                "backend_observed": True,
                "selected_endpoint_absent": absent,
                "selected_endpoint_reappeared": True,
            }
        elif scenario == "default_device_change":
            before_defaults = {
                direction: set(before_devices[direction]["defaults"])
                for direction in ("input", "output")
            }
            if not any(before_defaults.values()):
                evidence["event"]["reason"] = "default_endpoint_not_observed"
                return evidence
            print(
                "Change a Windows default input or output endpoint now; "
                f"the qualification probe waits up to {timeout_seconds:.0f}s."
            )
            changed_direction: str | None = None
            while time.monotonic() < deadline:
                current = _device_snapshot(
                    list_input_devices, list_output_devices, health_input, health_output
                )
                for direction in ("input", "output"):
                    if set(current[direction]["defaults"]) != before_defaults[direction]:
                        changed_direction = direction
                        break
                if changed_direction and route_ok():
                    event_diagnostics = dict(processor.get_runtime_diagnostics())
                    break
                time.sleep(LIFECYCLE_POLL_SECONDS)
            if event_diagnostics is None:
                evidence["event"]["reason"] = "default_endpoint_transition_not_observed"
                return evidence
            if changed_direction is None:
                evidence["event"]["reason"] = "default_endpoint_direction_not_observed"
                return evidence
            changed_default_ids = frozenset(
                current[changed_direction]["defaults"]
            )

            def default_persisted() -> bool:
                latest = _device_snapshot(
                    list_input_devices, list_output_devices, health_input, health_output
                )
                return (
                    changed_direction is not None
                    and frozenset(latest[changed_direction]["defaults"])
                    == changed_default_ids
                )

            post_recovery_ok = default_persisted
            event = {
                "observed": True,
                "backend_observed": True,
                "default_endpoint_changed": True,
                "changed_direction": changed_direction,
                "selected_route_correct": True,
            }
        else:
            print(
                "Put Windows to sleep and resume now; the qualification probe "
                f"waits up to {timeout_seconds:.0f}s."
            )
            sleep_event: dict[str, Any] | None = None
            resume_event: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                for observed in _read_power_events(started_utc):
                    if observed["kind"] == "sleep":
                        sleep_event = observed
                    elif (
                        sleep_event is not None
                        and observed["kind"] == "resume"
                        and observed["family"] == sleep_event["family"]
                    ):
                        resume_event = observed
                        break
                if resume_event is not None:
                    event_diagnostics = dict(processor.get_runtime_diagnostics())
                    break
                time.sleep(LIFECYCLE_POLL_SECONDS)
            if event_diagnostics is None or sleep_event is None or resume_event is None:
                evidence["event"]["reason"] = "os_sleep_resume_events_not_observed"
                return evidence
            event = {
                "observed": True,
                "backend_observed": True,
                "os_suspend_event": True,
                "os_resume_event": True,
                "sleep_event_id": sleep_event["event_id"],
                "resume_event_id": resume_event["event_id"],
            }

        recovered, after, recovery = settle(
            before_diagnostics, event_diagnostics, post_recovery_ok
        )
        evidence["event"] = event
        evidence["recovery"] = recovery
        evidence["diagnostics"]["after"] = _diagnostic_summary(after)
        evidence["passed"] = recovered
        return evidence
    except Exception:
        event = evidence.get("event")
        if isinstance(event, dict):
            event["reason"] = "probe_observation_failed"
        return evidence
    finally:
        try:
            processor.stop()
        except Exception:
            pass


def _run_lifecycle_probe(
    *,
    scenario: str,
    health_input: str,
    health_output: str,
    bundle_root: Path | None,
    timeout_seconds: float = LIFECYCLE_PROBE_TIMEOUT_SECONDS,
    settle_seconds: float = LIFECYCLE_SETTLE_SECONDS,
) -> dict[str, Any]:
    previous_deepfilter_env = os.environ.get("AUDIOFORGE_ENABLE_DEEPFILTER")
    if bundle_root is not None:
        os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = "1"
    try:
        return _run_lifecycle_probe_impl(
            scenario=scenario,
            health_input=health_input,
            health_output=health_output,
            bundle_root=bundle_root,
            timeout_seconds=timeout_seconds,
            settle_seconds=settle_seconds,
        )
    finally:
        if bundle_root is not None:
            if previous_deepfilter_env is None:
                os.environ.pop("AUDIOFORGE_ENABLE_DEEPFILTER", None)
            else:
                os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] = previous_deepfilter_env


def _observed_input_sample_rate(
    *,
    bundle_root: Path | None,
    input_name: str,
    output_name: str,
) -> int | None:
    try:
        _processor, list_input_devices, list_output_devices = _runtime_api(bundle_root)
        snapshot = _device_snapshot(
            list_input_devices, list_output_devices, input_name, output_name
        )
        if (
            snapshot["input"]["count"] != 1
            or snapshot["input"]["endpoint_count"] != 1
            or snapshot["output"]["count"] != 1
            or snapshot["output"]["endpoint_count"] != 1
            or len(snapshot["input"]["sample_rates"]) != 1
        ):
            return None
        return snapshot["input"]["sample_rates"][0]
    except Exception:
        return None


def _selected_endpoint_ids(
    *,
    bundle_root: Path | None,
    input_name: str,
    output_name: str,
) -> list[str]:
    try:
        _processor, list_input_devices, list_output_devices = _runtime_api(bundle_root)
        snapshot = _device_snapshot(
            list_input_devices, list_output_devices, input_name, output_name
        )
        return [
            key.removeprefix("endpoint:")
            for direction in ("input", "output")
            for key in snapshot[direction]["keys"]
        ]
    except Exception:
        return []
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
            *(["--noise-model", "deepfilter"] if bundle_root is not None else []),
            *bundle_arguments,
        ]
    )
    parsed_self_test = _parse_self_test(self_test)
    parsed_health = _parse_health(health)
    lifecycle_evidence: dict[str, Any] | None = None
    scenario_metadata_valid = (
        (scenario == "baseline" and evidence_kind == "automated")
        or (
            scenario in PHYSICAL_SCENARIOS
            and evidence_kind == "operator_observed"
        )
        or (
            scenario == "model_configuration_change"
            and evidence_kind in EVIDENCE_KINDS
        )
    )
    if scenario == "baseline":
        scenario_evidence_valid = scenario_metadata_valid
    elif not parsed_health["passed"]:
        lifecycle_evidence = _lifecycle_failure(
            scenario,
            "sustained_health_did_not_pass",
            timeout_seconds=LIFECYCLE_PROBE_TIMEOUT_SECONDS,
            settle_seconds=LIFECYCLE_SETTLE_SECONDS,
        )
        scenario_evidence_valid = False
    else:
        lifecycle_evidence = _run_lifecycle_probe(
            scenario=scenario,
            health_input=health_input,
            health_output=health_output,
            bundle_root=bundle_root,
        )
        scenario_evidence_valid = bool(lifecycle_evidence.get("passed"))
        if scenario in PHYSICAL_SCENARIOS:
            scenario_evidence_valid = (
                scenario_evidence_valid and confirm_scenario_observed
            )
    observed_input_sample_rate_hz = (
        lifecycle_evidence.get("observed_input_sample_rate_hz")
        if isinstance(lifecycle_evidence, dict)
        else None
    )
    if not isinstance(observed_input_sample_rate_hz, int):
        observed_input_sample_rate_hz = _observed_input_sample_rate(
            bundle_root=bundle_root,
            input_name=health_input,
            output_name=health_output,
        )
    scenario_evidence_valid = (
        scenario_evidence_valid
        and observed_input_sample_rate_hz == nominal_sample_rate_hz
    )
    scenario_evidence_valid = scenario_evidence_valid and scenario_metadata_valid
    private_endpoint_ids = _selected_endpoint_ids(
        bundle_root=bundle_root,
        input_name=health_input,
        output_name=health_output,
    )
    filtered_runs, device_pseudonyms = _privacy_filter_runs(
        [self_test, health],
        [
            health_input,
            health_output,
            correlation_input,
            correlation_output,
            *private_endpoint_ids,
        ],
    )
    parsed_health = _replace_private_strings(parsed_health, device_pseudonyms)
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
        "audible_change": None,
        "case": {
            "id": case_id,
            "device_class": device_class,
            "nominal_sample_rate_hz": nominal_sample_rate_hz,
            "observed_input_sample_rate_hz": observed_input_sample_rate_hz,
            "scenario": scenario,
            "evidence_kind": evidence_kind,
            "automated_measurements": True,
            "operator_observation_required": scenario in PHYSICAL_SCENARIOS,
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
        "lifecycle_evidence": lifecycle_evidence,
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
                "Selected routes are tied to unique backend endpoint IDs; ambiguous "
                "duplicate names fail qualification."
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
