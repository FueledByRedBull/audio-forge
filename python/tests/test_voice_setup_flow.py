"""Contracts for the Voice Setup flow, which runs without a window."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_the_flow_module_does_not_import_the_widget_toolkit() -> None:
    script = (
        "import sys\n"
        "import mic_eq.ui.voice_setup_flow\n"
        "print('PySide6.QtWidgets' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
