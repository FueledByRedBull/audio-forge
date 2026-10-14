"""The Qt Quick scene under software rendering, the path of a machine with no usable GPU.

Qt reads ``QT_QUICK_BACKEND`` once, as the application starts, so the checks in
``quick_software_checks.py`` run in a child pytest started with it.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from pathlib import Path

from quick_child import run_checks

CHECKS = Path(__file__).with_name("quick_software_checks.py")


def test_the_scene_works_with_software_rendering_forced() -> None:
    run_checks(CHECKS, {"QT_QUICK_BACKEND": "software"})
