"""The Qt Quick scene works with software rendering requested.

Qt reads ``QT_QUICK_BACKEND`` once, as the application starts, so the checks in
``quick_software_checks.py`` run in a child pytest started with it. The child
runs on the offscreen platform, which renders in software whatever is asked, so
this proves the scene works in software and that the request reached the child;
it cannot prove the variable is what selected software. A child on the real
``windows`` platform could, but it opens windows on the desktop and has not been
run on a GitHub runner, so it is not here.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from pathlib import Path

from quick_child import run_checks

CHECKS = Path(__file__).with_name("quick_software_checks.py")


def test_the_scene_works_under_software_rendering() -> None:
    run_checks(CHECKS, {"QT_QUICK_BACKEND": "software"})
