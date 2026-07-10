"""Run the reviewed AudioForge Semgrep rulesets and emit SARIF."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RULESET_FILE = REPO_ROOT / "semgrep-rulesets.txt"


def _rulesets() -> list[str]:
    return [
        line
        for raw_line in RULESET_FILE.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def _error_findings(sarif_path: Path) -> list[str]:
    payload = json.loads(sarif_path.read_text(encoding="utf-8"))
    findings: list[str] = []
    for run in payload.get("runs", []):
        for result in run.get("results", []):
            if result.get("level") == "error":
                findings.append(str(result.get("ruleId", "unknown-rule")))
    return findings


def _semgrep_executable() -> str:
    scripts_dir = Path(sys.executable).resolve().parent
    for name in ("semgrep.exe", "semgrep"):
        candidate = scripts_dir / name
        if candidate.is_file():
            return str(candidate)
    executable = shutil.which("semgrep")
    if executable is None:
        raise RuntimeError("Semgrep executable is not installed")
    return executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarif", type=Path, default=Path("semgrep-results.sarif"))
    args = parser.parse_args()
    sarif_path = args.sarif.resolve()

    command = [
        _semgrep_executable(),
        "scan",
        "--metrics=off",
        "--sarif",
        "--output",
        str(sarif_path),
        "--exclude",
        ".venv",
        "--exclude",
        "build",
        "--exclude",
        "dist",
        "--exclude",
        "target",
        "--exclude",
        "static_analysis_semgrep_*",
    ]
    for ruleset in _rulesets():
        command.extend(("--config", ruleset))
    command.append(str(REPO_ROOT))

    child_env = os.environ.copy()
    child_env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    # The reviewed executable/configs and user-selected SARIF path are separate
    # argv entries; shell execution is disabled, so no command text is evaluated.
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
    completed = subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not sarif_path.is_file():
        print(f"Semgrep did not create {sarif_path}", file=sys.stderr)
        return 2

    error_findings = _error_findings(sarif_path)
    if error_findings:
        print(
            "Semgrep found reviewed ERROR-severity findings: "
            + ", ".join(sorted(set(error_findings))),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
