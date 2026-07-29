"""Run and record selected-route correlation plus sustained audio health."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "hardware-validation.json"
SELF_TEST_RESULT = re.compile(
    r"Self-test passed: rt=(?P<latency>[0-9.]+)ms confidence=(?P<confidence>[0-9.]+)"
)
HEALTH_SUMMARY = re.compile(
    r"Health summary: max_input_age_ms=(?P<input_age>\d+) "
    r"max_output_age_ms=(?P<output_age>\d+) restarts=(?P<restarts>\d+) "
    r"underrun_baseline=(?P<underrun_baseline>\d+) "
    r"diagnostics=(?P<diagnostics>\{.*\})"
)


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


def _runtime_provenance() -> dict[str, Any]:
    native_spec = importlib.util.find_spec("mic_eq.mic_eq_core")
    if native_spec is None or native_spec.origin is None:
        raise RuntimeError("mic_eq native extension is unavailable")
    return {
        "native_extension": _portable_runtime_file(Path(native_spec.origin)),
        "self_test": _portable_runtime_file(REPO_ROOT / "python/tools/self_test.py"),
        "health_check": _portable_runtime_file(
            REPO_ROOT / "python/tools/health_check.py"
        ),
        "latency_analysis": _portable_runtime_file(
            REPO_ROOT / "python/mic_eq/analysis/latency_calibration.py"
        ),
    }


def _run(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
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
) -> dict[str, Any]:
    python = str(Path(sys.executable).resolve())
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
        ]
    )
    parsed_self_test = _parse_self_test(self_test)
    parsed_health = _parse_health(health)
    passed = bool(parsed_self_test["passed"] and parsed_health["passed"])
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_version": _project_version(),
        "source_revision": _source_revision(),
        "runtime_provenance": _runtime_provenance(),
        "platform": platform.platform(),
        "audible_change": False,
        "purpose": "Release-machine selected-route and sustained callback health gate.",
        "routes": {
            "correlation": {
                "input": correlation_input,
                "output": correlation_output,
            },
            "sustained_health": {
                "input": health_input,
                "output": health_output,
            },
        },
        "requested_health_duration_seconds": health_duration,
        "selected_route_correlation": {**parsed_self_test, "run": self_test},
        "sustained_health": {**parsed_health, "run": health},
        "passed": passed,
        "limitations": [
            "One Windows machine and one physical microphone/virtual-cable route.",
            "Correlation uses VB-Cable loopback; sustained health uses the intended physical microphone route.",
            "This is objective device/runtime evidence, not a perceptual listening session.",
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
    args = parser.parse_args()
    report = evaluate(
        health_input=args.health_input,
        health_output=args.health_output,
        correlation_input=args.correlation_input,
        correlation_output=args.correlation_output,
        health_duration=args.health_duration,
        report_path=args.report.resolve(),
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
