"""Tests for release-to-release hardening trend records."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


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


def test_published_entry_uses_exact_hardware_artifact_and_does_not_relabel_source_evidence(
    tmp_path: Path,
):
    archive = tmp_path / "AudioForge-v1.10.1-win64-ultra.7z"
    archive.write_bytes(b"published archive")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "artifact": {
                    "archive_sha256": archive_hash,
                    "bundle": {"total_bytes": 321, "file_count": 7},
                },
            }
        ),
        encoding="utf-8",
    )
    source_level = tmp_path / "deepfilter.json"
    source_level.write_text(
        json.dumps(
            {
                "evaluation_contract": {
                    "runtime": {"max_p99_frame_seconds": 0.001},
                    "clean_preservation": {},
                },
                "selected_runtime_config": {},
                "alignment_gate": {"passes": True},
            }
        ),
        encoding="utf-8",
    )

    entry = trends_tool.build_entry(
        version="1.10.1",
        status="published",
        commit="a" * 40,
        bundle=None,
        archive=archive,
        deepfilter_report=source_level,
        hardware_report=hardware,
    )

    assert entry["package"]["archive"]["value"]["sha256"] == archive_hash
    assert entry["package"]["bundle"]["value"] == {"bytes": 321, "file_count": 7}
    hardware_value = entry["hardware"]["value"]
    assert hardware_value["archive_sha256"] == archive_hash
    assert hardware_value["report_sha256"] == hashlib.sha256(
        hardware.read_bytes()
    ).hexdigest()
    assert "artifact" not in hardware_value
    assert entry["runtime"]["status"] == "not_measured"
    assert "not bound" in entry["runtime"]["reason"]


def test_candidate_entry_accepts_fullband_deepfilter_report_schema(
    tmp_path: Path,
):
    report = tmp_path / "deepfilter-fullband.json"
    report.write_text(
        json.dumps(
            {
                "evaluation_contract": {
                    "runtime": {"max_p99_frame_seconds": 0.0012},
                    "clean_preservation": {"passed": True},
                    "configuration": {
                        "attenuation_limit_db": 30.0,
                        "post_filter_beta": 0.0,
                    },
                },
                "decision": {"retained": True},
            }
        ),
        encoding="utf-8",
    )

    entry = trends_tool.build_entry(
        version="1.11.0",
        status="candidate",
        commit="a" * 40 + "+uncommitted",
        bundle=None,
        archive=None,
        deepfilter_report=report,
        hardware_report=None,
    )

    assert entry["runtime"]["value"] == {"max_p99_frame_seconds": 0.0012}
    quality = entry["quality"]["value"]
    assert quality["selected_runtime_config"] == {
        "attenuation_limit_db": 30.0,
        "post_filter_beta": 0.0,
    }
    assert quality["clean_preservation"] == {"passed": True}
    assert quality["alignment_gate_status"] == "not_measured_by_report"


def test_published_entry_rejects_dirty_commit_and_mismatched_hardware(tmp_path: Path):
    archive = tmp_path / "release.7z"
    archive.write_bytes(b"archive")
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "artifact": {"archive_sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact 40-character commit"):
        trends_tool.build_entry(
            version="1.10.1",
            status="published",
            commit="a" * 40 + "+uncommitted",
            bundle=None,
            archive=archive,
            deepfilter_report=None,
            hardware_report=hardware,
        )
    with pytest.raises(ValueError, match="does not match"):
        trends_tool.build_entry(
            version="1.10.1",
            status="published",
            commit="a" * 40,
            bundle=None,
            archive=archive,
            deepfilter_report=None,
            hardware_report=hardware,
        )


def test_entry_rejects_non_object_hardware_report(tmp_path: Path):
    hardware = tmp_path / "hardware.json"
    hardware.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be an object"):
        trends_tool.build_entry(
            version="1.11.0",
            status="candidate",
            commit="a" * 40,
            bundle=None,
            archive=None,
            deepfilter_report=None,
            hardware_report=hardware,
        )
