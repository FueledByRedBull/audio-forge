from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import check_evaluation_hygiene as hygiene


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _pinned_report(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "repo"
    evaluator = root / "python" / "tools" / "evaluator.py"
    report = root / "evaluation" / "report.json"
    evaluator.parent.mkdir(parents=True)
    report.parent.mkdir()
    evaluator.write_bytes(b"historical evaluator\n")
    payload = {
        "measurement": {"value": 1},
        "source_sha256": {
            "python/tools/evaluator.py": hashlib.sha256(evaluator.read_bytes()).hexdigest()
        },
    }
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "hygiene-test")
    _git(root, "config", "user.email", "hygiene-test@example.invalid")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=hygiene-test", "-c", "user.email=hygiene-test@example.invalid", "commit", "--quiet", "-m", "baseline")
    revision = _git(root, "rev-parse", "HEAD")
    payload["source_revision"] = revision
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    evaluator.write_bytes(b"current worktree evaluator\n")
    return root, report, revision


def test_historical_report_uses_pinned_git_blob(tmp_path: Path, monkeypatch) -> None:
    root, report, _revision = _pinned_report(tmp_path)
    monkeypatch.setattr(hygiene, "REPO_ROOT", root)

    assert hygiene.validate_report(report) == []


def test_historical_report_rejects_source_hash_edit(tmp_path: Path, monkeypatch) -> None:
    root, report, _revision = _pinned_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["source_sha256"]["python/tools/evaluator.py"] = "0" * 64
    report.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(hygiene, "REPO_ROOT", root)

    errors = hygiene.validate_report(report)

    assert any("stale source SHA-256 in source_revision" in error for error in errors)


def test_historical_report_rejects_measurement_edit(tmp_path: Path, monkeypatch) -> None:
    root, report, _revision = _pinned_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["measurement"]["value"] = 2
    report.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(hygiene, "REPO_ROOT", root)

    errors = hygiene.validate_report(report)

    assert any("report contents differ from source_revision" in error for error in errors)


def test_historical_report_rejects_unavailable_revision(
    tmp_path: Path, monkeypatch
) -> None:
    root, report, _revision = _pinned_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["source_revision"] = "a" * 40
    report.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(hygiene, "REPO_ROOT", root)

    errors = hygiene.validate_report(report)

    assert any("source_revision is unavailable" in error for error in errors)
