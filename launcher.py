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

_SMOKE_TEST_FLAG = "--smoke-test"


def _run() -> int:
    if _SMOKE_TEST_FLAG in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != _SMOKE_TEST_FLAG]
        from mic_eq.ui.app_bootstrap import run_smoke_test  # noqa: E402
        from mic_eq.ui.main_window import MainWindow  # noqa: E402

        return run_smoke_test(MainWindow)

    from mic_eq.ui.main_window import run_app  # noqa: E402

    return run_app()

if __name__ == "__main__":
    sys.exit(_run())
