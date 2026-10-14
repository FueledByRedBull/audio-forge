"""The Qt Quick scene at a device pixel ratio of 1.5 and 2.

Qt reads ``QT_SCALE_FACTOR`` once, as the application starts, so each ratio runs
the checks in ``quick_high_dpi_checks.py`` in a child pytest started with it.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from pathlib import Path

import pytest

from quick_child import run_checks

CHECKS = Path(__file__).with_name("quick_high_dpi_checks.py")
# One window of each kind and the two painted items that scale (the EQ graph, a meter) keep
# the child to a few seconds.
SELECTION = "not unsaved and not output and not compressor and not deesser"


@pytest.mark.parametrize("scale", ["1.5", "2"])
def test_the_scene_loads_without_warnings_and_nothing_overflows_at_scale(scale) -> None:
    run_checks(
        CHECKS, {"QT_SCALE_FACTOR": scale, "AUDIOFORGE_EXPECT_SCALE": scale}, "-k", SELECTION
    )
