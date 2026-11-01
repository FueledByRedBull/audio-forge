"""Run a file of Qt Quick checks in a child pytest started with a given environment.

Qt reads some settings (scale factor, graphics backend) once, as the application
starts, so a test that needs one of them runs the checks in a fresh process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BusName:
    """A window's D-Bus name, held from start() until close(); each call is noted in ``events``."""

    def __init__(self, events: list[str], *, free: bool = True) -> None:
        self.events, self.free = events, free

    def close(self) -> None:
        self.events.append("name released")

    def start(self) -> bool:
        self.events.append("name taken back")
        return self.free


def run_checks(checks: Path, environment: dict[str, str], *pytest_args: str) -> str:
    """Run ``checks`` offscreen, fail with the child's output unless every check passed."""

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(checks), "-q", "-p", "no:cacheprovider",
            "--tb=short", *pytest_args,
        ],
        cwd=ROOT,
        env={**os.environ, **environment, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = f"{result.stdout[-4000:]}\n{result.stderr[-1000:]}"
    assert result.returncode == 0, output
    assert " passed" in result.stdout and "skipped" not in result.stdout, output
    return result.stdout
