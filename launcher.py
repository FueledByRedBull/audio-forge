#!/usr/bin/env python3
"""
AudioForge launcher script for PyInstaller
"""
import os
import sys
from pathlib import Path


def _configure_frozen_runtime():
    """Configure frozen-runtime paths and DLL search directories before imports."""
    if not getattr(sys, "frozen", False):
        return

    exe_dir = Path(sys.executable).resolve().parent
    meipass = Path(getattr(sys, "_MEIPASS", "")) if hasattr(sys, "_MEIPASS") else None

    # Improve DLL resolution for df.dll and ORT/Qt dependencies in bundled runtime.
    for dll_dir in [exe_dir, meipass, (meipass / "_internal") if meipass else None]:
        if not dll_dir or not dll_dir.exists():
            continue
        try:
            os.add_dll_directory(str(dll_dir))
        except Exception:
            # Best-effort only; keep startup resilient on older Python/Windows modes.
            pass

_configure_frozen_runtime()

from mic_eq.ui.main_window import (  # noqa: E402 - frozen paths must be configured first
    run_app,
)

if __name__ == "__main__":
    sys.exit(run_app())
