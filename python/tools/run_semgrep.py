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
        rules = {
            str(rule.get("id", "")): str(
                rule.get("defaultConfiguration", {}).get("level", "")
            )
            for rule in run.get("tool", {}).get("driver", {}).get("rules", [])
        }
        for result in run.get("results", []):
            rule_id = str(result.get("ruleId", "unknown-rule"))
            severity = str(result.get("level") or rules.get(rule_id, ""))
            if severity == "error":
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


def _prepare_sarif_path(path: Path) -> Path:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.is_file():
        resolved.unlink()
    return resolved


def _scan_command(scan_output: Path) -> list[str]:
    command = [
        _semgrep_executable(),
        "scan",
        "--metrics=off",
        # Release CI runs from a clean clone, but local pre-commit audits must
        # also cover newly added source files before they become Git-tracked.
        "--no-git-ignore",
        "--sarif",
        "--output",
        str(scan_output),
        "--exclude",
        ".venv",
        "--exclude",
        ".git",
        "--exclude",
        "build",
        "--exclude",
        "dist",
        "--exclude",
        "target",
        "--exclude",
        "models",
        "--exclude",
        "downloads",
        "--exclude",
        "__pycache__",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        ".ruff_cache",
        "--exclude",
        ".pyright",
        "--exclude",
        "static_analysis_semgrep_*",
        # Never feed a previous scanner report back into the next scan. SARIF
        # embeds matched examples and can therefore look like source secrets.
        "--exclude",
        "*.sarif",
    ]
    for ruleset in _rulesets():
        command.extend(("--config", ruleset))
    command.append(str(REPO_ROOT))
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarif", type=Path, default=Path("semgrep-results.sarif"))
    args = parser.parse_args()
    sarif_path = _prepare_sarif_path(args.sarif)
    scan_output = REPO_ROOT / ".semgrep-results.sarif"
    if scan_output.is_file():
        scan_output.unlink()

    command = _scan_command(scan_output)

    child_env = os.environ.copy()
    child_env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=False)
    if scan_output.is_file():
        shutil.copyfile(scan_output, sarif_path)
        scan_output.unlink()
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
