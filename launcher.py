#!/usr/bin/env python3
"""
AudioForge launcher script for PyInstaller
"""
import os
import sys
from pathlib import Path


def _first_existing_path(candidates):
    for path in candidates:
        if path and path.exists():
            return path
    return None


def _configure_frozen_runtime():
    """Configure paths/env so bundled models and DLLs are discoverable."""
    if not getattr(sys, "frozen", False):
        return

    exe_dir = Path(sys.executable).resolve().parent
    meipass = Path(getattr(sys, "_MEIPASS", "")) if hasattr(sys, "_MEIPASS") else None

    # Ensure relative model lookups in Rust (./models) resolve from bundle root.
    try:
        os.chdir(exe_dir)
    except Exception:
        pass

    # Improve DLL resolution for df.dll and ORT/Qt dependencies in bundled runtime.
    for dll_dir in [exe_dir, meipass, (meipass / "_internal") if meipass else None]:
        if not dll_dir or not dll_dir.exists():
            continue
        try:
            os.add_dll_directory(str(dll_dir))
        except Exception:
            # Best-effort only; keep startup resilient on older Python/Windows modes.
            pass

    model_dirs = [
        exe_dir / "models",
        (meipass / "models") if meipass else None,
        (meipass / "_internal" / "models") if meipass else None,
    ]
    model_dir = _first_existing_path(model_dirs)

    if model_dir:
        vad_model = model_dir / "silero_vad.onnx"
        if vad_model.exists():
            os.environ.setdefault("VAD_MODEL_PATH", str(vad_model))

        ll_model = model_dir / "DeepFilterNet3_ll_onnx.tar.gz"
        std_model = model_dir / "DeepFilterNet3_onnx.tar.gz"
        if ll_model.exists() and std_model.exists():
            os.environ.setdefault("AUDIOFORGE_ENABLE_DEEPFILTER", "1")


_configure_frozen_runtime()

from mic_eq.ui import run_app

if __name__ == "__main__":
    sys.exit(run_app())
