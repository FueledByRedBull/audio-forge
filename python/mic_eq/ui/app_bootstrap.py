"""Application bootstrap helpers for the AudioForge UI."""

import os
import sys
from pathlib import Path
from typing import Type

from PyQt6.QtCore import QFileInfo
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QFileIconProvider, QMainWindow


def _application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def _trusted_runtime_roots() -> list[Path]:
    app_root = _application_root()
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", app_root)).resolve()
        roots = [meipass, app_root]
    else:
        roots = [app_root]

    trusted_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in trusted_roots:
            trusted_roots.append(resolved)
    return trusted_roots


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def configure_deepfilter_env() -> None:
    """Enable local DeepFilter runtime when local assets are present."""
    allow_external = _truthy_env("AUDIOFORGE_ALLOW_EXTERNAL_DF")
    if allow_external and "AUDIOFORGE_ENABLE_DEEPFILTER" in os.environ:
        return

    if os.name == "nt":
        lib_names = ["df.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libdf.dylib"]
    else:
        lib_names = ["libdf.so"]

    search_dirs = _trusted_runtime_roots()
    lib_path = None
    for lib_name in lib_names:
        for base in search_dirs:
            candidate = base / lib_name
            if candidate.is_file():
                lib_path = candidate
                break
        if lib_path:
            break

    model_dirs = [root / "models" for root in search_dirs]
    model_found = False
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        ll_model = model_dir / "DeepFilterNet3_ll_onnx.tar.gz"
        std_model = model_dir / "DeepFilterNet3_onnx.tar.gz"
        if ll_model.is_file() or std_model.is_file():
            model_found = True
            break

    if lib_path and model_found:
        if allow_external:
            os.environ.setdefault("DEEPFILTER_LIB_PATH", str(lib_path))
        else:
            os.environ["DEEPFILTER_LIB_PATH"] = str(lib_path)
        os.environ.setdefault("AUDIOFORGE_ENABLE_DEEPFILTER", "1")


def configure_vad_env() -> None:
    """Point VAD at a bundled app-owned model when one is present."""
    if "VAD_MODEL_PATH" in os.environ:
        return

    for root in _trusted_runtime_roots():
        candidate = root / "models" / "silero_vad.onnx"
        if candidate.is_file():
            os.environ["VAD_MODEL_PATH"] = str(candidate)
            return


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
    configure_vad_env()

    window = window_cls()
    window.show()

    return app.exec()
