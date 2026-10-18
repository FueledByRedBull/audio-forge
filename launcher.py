#!/usr/bin/env python3
"""
AudioForge launcher script for PyInstaller
"""
import os
import logging
import sys
from pathlib import Path


def _configure_frozen_runtime():
    """Configure frozen-runtime paths and DLL search directories before imports."""
    if not getattr(sys, "frozen", False):
        return

    # The bundle's DLLs (df.dll, ORT, Qt) all live in the PyInstaller content
    # directory; the executable's own folder holds only the exe, so it stays off
    # the search path.
    meipass = getattr(sys, "_MEIPASS", None)
    dll_dir = Path(meipass) if meipass else None
    if not dll_dir or not dll_dir.exists():
        return
    try:
        os.add_dll_directory(str(dll_dir))
    except OSError:
        # Best-effort; keep startup resilient on older Python/Windows modes.
        logging.getLogger(__name__).warning("Could not add %s to the DLL search path", dll_dir)

_configure_frozen_runtime()

_SMOKE_TEST_FLAG = "--smoke-test"


def _run() -> int:
    if sys.argv[1:] == ["--remove-login-startup"]:
        if not getattr(sys, "frozen", False):
            return 1
        from mic_eq.ui.login_startup import set_login_startup

        try:
            set_login_startup(Path(sys.executable), False)
        except Exception:
            logging.getLogger(__name__).exception("Login shortcut cleanup failed; left unchanged")
            return 1
        return 0

    if _SMOKE_TEST_FLAG in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != _SMOKE_TEST_FLAG]
        from mic_eq.ui.app_bootstrap import run_smoke_test  # noqa: E402
        from mic_eq.ui.main_window import MainWindow  # noqa: E402

        return run_smoke_test(MainWindow)

    from mic_eq.ui.main_window import run_app  # noqa: E402

    return run_app()

if __name__ == "__main__":
    sys.exit(_run())
