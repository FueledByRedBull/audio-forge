"""Executable checks for the release-promotion origin gate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "release-promote.yml"


def _origin_gate_script() -> str:
    source = WORKFLOW.read_text(encoding="utf-8")
    start = source.index("          $canonicalTag =")
    end = source.index("\n      - name: Install pinned validation runtime", start)
    return textwrap.dedent(source[start:end])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (None, None),
        (("databaseId", 999), "identity mismatch"),
        (("workflowName", "Other workflow"), "expected 'Release package'"),
        (("conclusion", "failure"), "not a successful completed"),
        (("headSha", "b" * 40), "did not build the release tag commit"),
        (("event", "schedule"), "unsupported event"),
        (("attempt", 0), "invalid attempt number"),
    ],
)
def test_promotion_origin_gate_executes_commit_and_run_binding(
    tmp_path: Path, mutation: tuple[str, object] | None, expected_error: str | None
) -> None:
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required for the Windows promotion workflow")

    commit = "a" * 40
    run = {
        "databaseId": 123,
        "workflowName": "Release package",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "headSha": commit,
        "attempt": 1,
    }
    if mutation:
        run[mutation[0]] = mutation[1]
    output = tmp_path / "github-output"
    setup = textwrap.dedent(
        """
        $ErrorActionPreference = 'Stop'
        function python {
          param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
          $global:LASTEXITCODE = 0
          if ($CommandArgs[1] -eq 'tag') { $env:TEST_TAG; return }
          if ($CommandArgs[1] -eq 'prerelease') { 'false'; return }
          throw "unexpected python command"
        }
        function git {
          param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
          $global:LASTEXITCODE = 0
          $env:TEST_COMMIT
        }
        function gh {
          param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
          $global:LASTEXITCODE = 0
          $env:TEST_RUN_JSON
        }
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "RELEASE_TAG": "v1.12.0",
            "CANDIDATE_RUN_ID": "123",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_OUTPUT": str(output),
            "TEST_TAG": "v1.12.0",
            "TEST_COMMIT": commit,
            "TEST_RUN_JSON": json.dumps(run),
        }
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", setup + _origin_gate_script()],
        env=env,
        capture_output=True,
        text=True,
    )
    diagnostics = result.stdout + result.stderr
    if expected_error is None:
        assert result.returncode == 0, diagnostics
        assert "candidate_run_id=123" in output.read_text(encoding="utf-8")
    else:
        assert result.returncode != 0, diagnostics
        assert expected_error in diagnostics
