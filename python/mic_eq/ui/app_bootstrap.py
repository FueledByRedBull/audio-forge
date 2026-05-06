"""Application bootstrap helpers for the AudioForge UI."""

import os
import sys
from pathlib import Path
from typing import Type

from PyQt6.QtCore import QFileInfo
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QFileIconProvider, QMainWindow


def configure_deepfilter_env() -> None:
    """Enable local DeepFilter runtime when local assets are present."""
    if "AUDIOFORGE_ENABLE_DEEPFILTER" in os.environ:
        return

    if os.name == "nt":
        lib_names = ["df.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libdf.dylib"]
    else:
        lib_names = ["libdf.so"]

    repo_root = Path(__file__).resolve().parents[3]
    search_dirs = [Path.cwd(), repo_root]
    lib_path = None
    for lib_name in lib_names:
        for base in search_dirs:
            candidate = base / lib_name
            if candidate.exists():
                lib_path = candidate
                break
        if lib_path:
            break

    model_dirs = [
        Path.cwd() / "models",
        repo_root / "models",
    ]
    model_found = False
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        ll_model = model_dir / "DeepFilterNet3_ll_onnx.tar.gz"
        std_model = model_dir / "DeepFilterNet3_onnx.tar.gz"
        if ll_model.exists() or std_model.exists():
            model_found = True
            break

    if lib_path and model_found:
        os.environ.setdefault("DEEPFILTER_LIB_PATH", str(lib_path))
        os.environ.setdefault("AUDIOFORGE_ENABLE_DEEPFILTER", "1")


def run_qt_app(window_cls: Type[QMainWindow]) -> int:
    """Run the Qt application for the provided main window class."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if getattr(sys, "frozen", False):
        provider = QFileIconProvider()
        exe_icon = provider.icon(QFileInfo(sys.executable))
        if not exe_icon.isNull():
            app.setWindowIcon(exe_icon)
    else:
        icon_path = Path("mic_eq.ico")
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))

    configure_deepfilter_env()

    window = window_cls()
    window.show()

    return app.exec()
