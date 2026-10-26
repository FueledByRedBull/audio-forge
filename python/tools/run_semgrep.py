"""Run the reviewed AudioForge Semgrep rulesets and emit SARIF."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath


REPO_ROOT = Path(__file__).resolve().parents[2]
RULESET_FILE = REPO_ROOT / "semgrep-rulesets.txt"
# Third-party collections are cloned here at their pinned commit; the directory
# is fixed so rule ids, and the nosemgrep comments that name them, are stable.
RULES_DIR = REPO_ROOT / "build" / "semgrep-rules"
GIT_TIMEOUT_SECONDS = 300
COLLECTION_LINE = re.compile(
    r"git (https://github\.com/([\w.-]+)/([\w.-]+)) ([0-9a-f]{40})((?: [\w.-][\w./-]*)+)"
)


def _checkout_rules(url: str, commit: str, checkout: Path) -> None:
    """Make `checkout` hold `url` at exactly `commit`, unmodified; semgrep cannot load a URL."""

    def git(*args: str, check: bool = True) -> str:
        # core.longpaths: some collections carry paths beyond the Windows limit.
        try:
            result = subprocess.run(
                ["git", "-c", "core.longpaths=true", "-C", str(checkout), *args],
                capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            raise RuntimeError("git is required to fetch the pinned Semgrep rule collections") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git {args[0]} for {url} took over {GIT_TIMEOUT_SECONDS} s") from None
        if check and result.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed for {url}: {result.stderr.strip()[:300]}")
        return result.stdout.strip()

    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").is_dir():
        git("init", "--quiet")
    if git("rev-parse", "--verify", "--quiet", "HEAD", check=False) != commit:
        git("fetch", "--quiet", "--depth=1", url, commit)
        git("checkout", "--quiet", "--force", "--detach", "FETCH_HEAD")
    # The right commit is not enough: an edited, added or ignored rule file is
    # loaded all the same, so any difference from the commit is discarded.
    if git("status", "--porcelain", "--untracked-files=all", "--ignored"):
        git("reset", "--quiet", "--hard", "HEAD")
        git("clean", "--quiet", "-ffdx")


def _rulesets() -> list[str]:
    configs: list[str] = []
    for raw_line in RULESET_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("git "):
            configs.append(line)
            continue
        match = COLLECTION_LINE.fullmatch(line)
        if match is None or any(".." in part.split("/") for part in match[5].split()):
            raise RuntimeError(f"Unusable semgrep collection line: {line}")
        url, owner, repo, commit, directories = match.groups()
        checkout = RULES_DIR / f"{owner}-{repo}"
        _checkout_rules(url, commit, checkout)
        configs.extend(
            (checkout / directory).relative_to(REPO_ROOT).as_posix()
            for directory in directories.split()
        )
    return configs


def _suppressed_in_source(result: dict[str, object]) -> bool:
    suppressions = result.get("suppressions")
    return isinstance(suppressions, list) and any(
        isinstance(suppression, dict)
        and suppression.get("kind") == "inSource"
        and suppression.get("status") in (None, "accepted")
        for suppression in suppressions
    )


def _blocking_findings(sarif_path: Path) -> list[str]:
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
            if severity in ("error", "warning") and not _suppressed_in_source(result):
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
        # Joined values prevent the Windows CLI from expanding bare wildcards.
        "--exclude=.venv*",
        "--exclude=.env",
        "--exclude=.env.*",
        "--exclude=credentials.*",
        "--exclude=secrets.*",
        "--exclude=.git",
        "--exclude=build",
        "--exclude=dist",
        "--exclude=target",
        "--exclude=models",
        "--exclude=downloads",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=.pyright",
        "--exclude=static_analysis_semgrep_*",
        "--exclude=.tmp*",
        "--exclude=.pytest-tmp*",
        "--exclude=pytest-cache-files-*",
        "--exclude=target-local*",
        "--exclude=IGNORE*",
        # Never feed a previous scanner report back into the next scan. SARIF
        # embeds matched examples and can therefore look like source secrets.
        "--exclude=*.sarif",
    ]
    for ruleset in _rulesets():
        command.extend(("--config", ruleset))
    command.append(str(REPO_ROOT))
    return command


def _sarif_file_uris(value: dict[str, object]) -> dict[str, object]:
    uri = value.get("uri")
    if isinstance(uri, str):
        path = PureWindowsPath(uri)
        if path.is_absolute():
            # Semgrep emits native Windows paths; SARIF requires URI syntax.
            value["uri"] = path.as_uri()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarif", type=Path, default=Path("semgrep-results.sarif"))
    args = parser.parse_args()
    sarif_path = _prepare_sarif_path(args.sarif)
    scan_output = REPO_ROOT / ".semgrep-results.sarif"
    if scan_output.is_file():
        scan_output.unlink()

    try:
        command = _scan_command(scan_output)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    child_env = os.environ.copy()
    child_env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=False)
    if scan_output.is_file():
        payload = json.loads(
            scan_output.read_text(encoding="utf-8"), object_hook=_sarif_file_uris,
        )
        for run in payload.get("runs", []):
            # GitHub Code Scanning ignores SARIF suppressions, so reviewed
            # `nosemgrep` findings would otherwise stay open as alerts.
            if isinstance(run.get("results"), list):
                run["results"] = [
                    result
                    for result in run["results"]
                    if not _suppressed_in_source(result)
                ]
        sarif_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        scan_output.unlink()
    if completed.returncode != 0:
        return completed.returncode
    if not sarif_path.is_file():
        print(f"Semgrep did not create {sarif_path}", file=sys.stderr)
        return 2

    blocking_findings = _blocking_findings(sarif_path)
    if blocking_findings:
        print(
            "Semgrep found unsuppressed ERROR or WARNING findings: "
            + ", ".join(sorted(set(blocking_findings))),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
