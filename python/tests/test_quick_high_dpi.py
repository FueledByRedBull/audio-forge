"""The Qt Quick scene at a device pixel ratio of 1.5 and 2.

Qt reads ``QT_SCALE_FACTOR`` once, as the application starts, so each ratio runs
the checks in ``quick_high_dpi_checks.py`` in a child pytest started with it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKS = Path(__file__).with_name("quick_high_dpi_checks.py")
# One window of each kind and the two painted items that scale (the EQ graph, a meter) keep
# the child to a few seconds.
SELECTION = "not unsaved and not output and not compressor and not deesser"


@pytest.mark.parametrize("scale", ["1.5", "2"])
def test_the_scene_loads_without_warnings_and_nothing_overflows_at_scale(scale) -> None:
    environment = {
        **os.environ,
        "QT_SCALE_FACTOR": scale,
        "AUDIOFORGE_EXPECT_SCALE": scale,
        "QT_QPA_PLATFORM": "offscreen",
    }
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(CHECKS), "-q", "-p", "no:cacheprovider",
            "--tb=short", "-k", SELECTION,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"scale {scale}:\n{result.stdout[-4000:]}\n{result.stderr[-1000:]}"
    assert " passed" in result.stdout and "skipped" not in result.stdout
