"""Generate a deterministic Markdown roadmap index from GitHub issues."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ORDER_PATTERN = re.compile(r"<!--\s*audioforge-roadmap-order:\s*(\d+)\s*-->")
GITHUB_REMOTE_PATTERNS = (
    re.compile(r"^https://github\.com/([^/]+/[^/]+?)(?:\.git)?$"),
    re.compile(r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$"),
    re.compile(r"^ssh://git@github\.com/([^/]+/[^/]+?)(?:\.git)?$"),
)


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def _repository_from_remote(remote: str) -> str:
    value = remote.strip().replace("\\", "/")
    for pattern in GITHUB_REMOTE_PATTERNS:
        match = pattern.fullmatch(value)
        if match:
            return match.group(1)
    raise ValueError(f"origin is not a supported GitHub remote: {remote.strip()}")


def _default_repository() -> str:
    return _repository_from_remote(_run(("git", "remote", "get-url", "origin")))


def _load_issues(repository: str) -> list[dict[str, Any]]:
    raw = _run(
        (
            "gh",
            "issue",
            "list",
            "--repo",
            repository,
            "--label",
            "roadmap",
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,title,state,stateReason,body,labels,milestone,url",
        )
    )
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("GitHub issue response must be a JSON list")
    return value


def _project_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    version = project.get("version")
    if not isinstance(version, str):
        raise ValueError("pyproject.toml does not contain project.version")
    return version


def _label_names(issue: dict[str, Any]) -> set[str]:
    labels = issue.get("labels", [])
    if not isinstance(labels, list):
        return set()
    return {
        name
        for label in labels
        if isinstance(label, dict)
        and isinstance((name := label.get("name")), str)
    }


def _issue_order(issue: dict[str, Any]) -> tuple[int, int]:
    body = issue.get("body")
    match = ORDER_PATTERN.search(body if isinstance(body, str) else "")
    order = int(match.group(1)) if match else 1_000_000
    number = issue.get("number")
    return order, int(number) if isinstance(number, int) else 1_000_000


def _milestone_key(issue: dict[str, Any]) -> tuple[int, str]:
    milestone = issue.get("milestone")
    if not isinstance(milestone, dict):
        return 1_000_000, "Unscheduled"
    number = milestone.get("number")
    title = milestone.get("title")
    return (
        int(number) if isinstance(number, int) else 1_000_000,
        title if isinstance(title, str) else "Unscheduled",
    )


def _issue_line(issue: dict[str, Any]) -> str:
    checked = "x" if issue.get("state") == "CLOSED" else " "
    number = issue.get("number")
    title = issue.get("title")
    url = issue.get("url")
    if not isinstance(number, int) or not isinstance(title, str) or not isinstance(url, str):
        raise ValueError("roadmap issue is missing number, title, or URL")
    labels = _label_names(issue)
    priority = next(
        (
            name.removeprefix("priority:").upper()
            for name in sorted(labels)
            if name.startswith("priority:")
        ),
        None,
    )
    outcome = " — not planned" if issue.get("stateReason") == "NOT_PLANNED" else ""
    priority_suffix = f" — {priority}" if priority else ""
    return (
        f"- [{checked}] [#{number} — {title}]({url})"
        f"{outcome}{priority_suffix}"
    )


def render_index(issues: Sequence[dict[str, Any]], repository: str, version: str) -> str:
    """Render the complete generated index."""

    ordered = sorted(issues, key=lambda issue: (_milestone_key(issue), _issue_order(issue)))
    decisions = [
        issue
        for issue in ordered
        if "decision" in _label_names(issue) and issue.get("state") == "CLOSED"
    ]
    actionable = [issue for issue in ordered if issue not in decisions]
    open_count = sum(issue.get("state") == "OPEN" for issue in actionable)
    not_planned_count = sum(
        issue.get("state") == "CLOSED"
        and issue.get("stateReason") == "NOT_PLANNED"
        for issue in actionable
    )
    completed_count = sum(
        issue.get("state") == "CLOSED"
        and issue.get("stateReason") != "NOT_PLANNED"
        for issue in actionable
    )

    lines = [
        "# AudioForge roadmap index",
        "",
        "> Generated by `python/tools/generate_todo_index.py` from versioned GitHub",
        "> issues and milestones. Do not edit this file by hand.",
        "",
        f"Current source version: `v{version}`",
        "",
        f"Roadmap source: [milestones](https://github.com/{repository}/milestones) · "
        f"[issues](https://github.com/{repository}/issues?q=is%3Aissue+label%3Aroadmap)",
        "",
        f"Actionable status: **{open_count} open**, "
        f"**{completed_count} completed**, "
        f"**{not_planned_count} not planned**.",
        "",
        "Shipped behavior and release history live in "
        f"[CHANGELOG.md](https://github.com/{repository}/blob/master/CHANGELOG.md) "
        f"and [GitHub Releases](https://github.com/{repository}/releases).",
        "",
        "## Actionable milestones",
        "",
    ]

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for issue in actionable:
        grouped.setdefault(_milestone_key(issue), []).append(issue)

    for (_, title), milestone_issues in grouped.items():
        lines.extend((f"### {title}", ""))
        milestone = milestone_issues[0].get("milestone")
        if isinstance(milestone, dict):
            description = milestone.get("description")
            if isinstance(description, str) and description.strip():
                lines.extend((description.strip(), ""))
        lines.extend(_issue_line(issue) for issue in milestone_issues)
        lines.append("")

    lines.extend(
        (
            "## Decisions and holds",
            "",
            "Closed decision issues are intentional records, not completed implementations. "
            "Reopen them only when their documented evidence gates are met.",
            "",
        )
    )
    if decisions:
        lines.extend(_issue_line(issue) for issue in sorted(decisions, key=_issue_order))
    else:
        lines.append("_No versioned decision records found._")
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repository in OWNER/NAME form")
    parser.add_argument("--output", type=Path, help="write the index to this path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if --output does not exactly match the generated index",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check and args.output is None:
        raise ValueError("--check requires --output")

    repository = args.repo or _default_repository()
    generated = render_index(_load_issues(repository), repository, _project_version())
    if args.output is None:
        sys.stdout.write(generated)
        return 0

    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != generated:
            print(f"Roadmap index is stale: {output}", file=sys.stderr)
            return 1
        print(f"Roadmap index is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"Wrote roadmap index: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Roadmap generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
