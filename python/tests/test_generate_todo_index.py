"""Tests for the GitHub roadmap index generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).parent.parent / "tools" / "generate_todo_index.py"
SPEC = importlib.util.spec_from_file_location("generate_todo_index", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_todo_index = importlib.util.module_from_spec(SPEC)
sys.modules["generate_todo_index"] = generate_todo_index
SPEC.loader.exec_module(generate_todo_index)


def _issue(
    number: int,
    title: str,
    *,
    order: int,
    state: str = "OPEN",
    milestone_number: int | None = 1,
    milestone_title: str = "P0",
    labels: tuple[str, ...] = ("roadmap", "priority:p1"),
) -> dict[str, object]:
    milestone = (
        {
            "number": milestone_number,
            "title": milestone_title,
            "description": f"{milestone_title} description.",
        }
        if milestone_number is not None
        else None
    )
    return {
        "number": number,
        "title": title,
        "state": state,
        "body": f"<!-- audioforge-roadmap-order: {order} -->",
        "labels": [{"name": label} for label in labels],
        "milestone": milestone,
        "url": f"https://github.com/acme/audio-forge/issues/{number}",
    }


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://github.com/acme/audio-forge.git", "acme/audio-forge"),
        ("git@github.com:acme/audio-forge.git", "acme/audio-forge"),
        ("ssh://git@github.com/acme/audio-forge.git", "acme/audio-forge"),
    ],
)
def test_repository_from_remote_accepts_supported_github_forms(remote, expected):
    assert generate_todo_index._repository_from_remote(remote) == expected


def test_repository_from_remote_rejects_non_github_remote():
    with pytest.raises(ValueError, match="not a supported GitHub remote"):
        generate_todo_index._repository_from_remote("https://example.invalid/repo.git")


def test_render_index_groups_orders_and_separates_decisions():
    issues = [
        _issue(3, "Later", order=20),
        _issue(2, "Earlier", order=10),
        _issue(
            4,
            "Held",
            order=900,
            state="CLOSED",
            milestone_number=None,
            labels=("roadmap", "decision", "on-hold", "priority:p3"),
        ),
    ]

    rendered = generate_todo_index.render_index(issues, "acme/audio-forge", "1.2.3")

    assert "Current source version: `v1.2.3`" in rendered
    assert "Actionable status: **2 open**, **0 completed**." in rendered
    assert rendered.index("#2 — Earlier") < rendered.index("#3 — Later")
    assert "### P0" in rendered
    assert "- [x] [#4 — Held]" in rendered
    assert rendered.index("## Decisions and holds") < rendered.index("#4 — Held")


def test_main_check_detects_stale_and_current_output(tmp_path, monkeypatch):
    issue = _issue(1, "Work", order=1)
    output = tmp_path / "TODO.md"
    monkeypatch.setattr(generate_todo_index, "_load_issues", lambda repository: [issue])
    monkeypatch.setattr(generate_todo_index, "_project_version", lambda: "1.2.3")

    assert (
        generate_todo_index.main(
            ["--repo", "acme/audio-forge", "--output", str(output)]
        )
        == 0
    )
    assert (
        generate_todo_index.main(
            ["--repo", "acme/audio-forge", "--output", str(output), "--check"]
        )
        == 0
    )

    output.write_text("stale\n", encoding="utf-8")
    assert (
        generate_todo_index.main(
            ["--repo", "acme/audio-forge", "--output", str(output), "--check"]
        )
        == 1
    )
