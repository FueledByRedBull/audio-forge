"""Tests for release-to-release hardening trend records."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TOOL_PATH = Path(__file__).parent.parent / "tools" / "update_release_trends.py"
SPEC = importlib.util.spec_from_file_location("update_release_trends", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
trends_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trends_tool
SPEC.loader.exec_module(trends_tool)


def test_update_replaces_same_version_and_orders_semantically(tmp_path: Path):
    path = tmp_path / "trends.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": "test",
                "releases": [
                    {"version": "1.10.0", "marker": "old"},
                    {"version": "1.9.0", "marker": "older"},
                ],
            }
        ),
        encoding="utf-8",
    )

    trends_tool.update_trends(path, {"version": "1.10.0", "marker": "new"})
    result = json.loads(path.read_text(encoding="utf-8"))

    assert [entry["version"] for entry in result["releases"]] == ["1.9.0", "1.10.0"]
    assert result["releases"][-1]["marker"] == "new"


def test_directory_metrics_count_only_files(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "one.bin").write_bytes(b"123")
    (tmp_path / "nested" / "two.bin").write_bytes(b"4567")

    assert trends_tool._directory_metrics(tmp_path) == {"bytes": 7, "file_count": 2}


def test_git_identity_marks_dirty_worktree(monkeypatch):
    outputs = iter(["abc123\n", " M changed.py\n"])

    class _Result:
        def __init__(self, stdout: str):
            self.stdout = stdout

    monkeypatch.setattr(
        trends_tool.subprocess,
        "run",
        lambda *args, **kwargs: _Result(next(outputs)),
    )

    assert trends_tool._git_commit() == "abc123+uncommitted"
