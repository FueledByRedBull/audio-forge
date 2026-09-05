"""Application bootstrap helpers for the AudioForge UI."""

import ctypes
import importlib
import logging
import os
import subprocess
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Type

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow

from .. import configure_deepfilter_runtime_paths
from ..app_logging import configure_app_logging, get_log_file
from .theme import application_palette


WINDOWS_APP_USER_MODEL_ID = "FueledByRedBull.AudioForge"


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


def configure_deepfilter_env() -> None:
    """Register canonical bundled assets without rewriting external overrides."""

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
    model_dir_with_model = None
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        ll_model = model_dir / "DeepFilterNet3_ll_onnx.tar.gz"
        std_model = model_dir / "DeepFilterNet3_onnx.tar.gz"
        if ll_model.is_file() or std_model.is_file():
            model_dir_with_model = model_dir
            break

    if lib_path and model_dir_with_model:
        configure_deepfilter_runtime_paths(str(lib_path), str(model_dir_with_model))
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


def configure_windows_app_id() -> None:
    """Give Windows a stable taskbar identity for the app."""
    if os.name != "nt":
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            WINDOWS_APP_USER_MODEL_ID
        )
    except Exception:
        pass


def _application_icon() -> QIcon:
    if getattr(sys, "frozen", False):
        icon = QIcon(str(Path(sys.executable).resolve()))
        if not icon.isNull():
            return icon

    for root in _trusted_runtime_roots():
        icon_path = root / "AudioForge.ico"
        if icon_path.is_file():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon

    return QIcon()


def apply_windows_window_icon(window: QMainWindow) -> None:
    """Set native small/large window icons from the frozen executable resource."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return

    hicon_large = ctypes.c_void_p()
    hicon_small = ctypes.c_void_p()
    extracted = ctypes.windll.shell32.ExtractIconExW(
        str(Path(sys.executable).resolve()),
        0,
        ctypes.byref(hicon_large),
        ctypes.byref(hicon_small),
        1,
    )
    if extracted <= 0:
        return

    hwnd = int(window.winId())
    user32 = ctypes.windll.user32
    wm_seticon = 0x0080
    icon_small = 0
    icon_big = 1

    if hicon_small.value:
        user32.SendMessageW(hwnd, wm_seticon, icon_small, hicon_small.value)
    if hicon_large.value:
        user32.SendMessageW(hwnd, wm_seticon, icon_big, hicon_large.value)

    setattr(window, "_audioforge_native_icons", (hicon_large, hicon_small))


def apply_windows_taskbar_properties(window: QMainWindow) -> None:
    """Set Explorer taskbar relaunch metadata for the top-level window."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return

    try:
        propsys = importlib.import_module("win32com.propsys.propsys")
        pscon = importlib.import_module("win32com.propsys.pscon")

        exe_path = str(Path(sys.executable).resolve())
        store = propsys.SHGetPropertyStoreForWindow(
            int(window.winId()),
            propsys.IID_IPropertyStore,
        )
        store.SetValue(
            pscon.PKEY_AppUserModel_ID,
            propsys.PROPVARIANTType(WINDOWS_APP_USER_MODEL_ID),
        )
        store.SetValue(
            pscon.PKEY_AppUserModel_RelaunchCommand,
            propsys.PROPVARIANTType(subprocess.list2cmdline([exe_path])),
        )
        store.SetValue(
            pscon.PKEY_AppUserModel_RelaunchDisplayNameResource,
            propsys.PROPVARIANTType("AudioForge"),
        )
        store.SetValue(
            pscon.PKEY_AppUserModel_RelaunchIconResource,
            propsys.PROPVARIANTType(f"{exe_path},0"),
        )
        store.Commit()
    except Exception:
        pass


def run_qt_app(window_cls: Type[QMainWindow], *, smoke_test: bool = False) -> int:
    """Run the Qt application for the provided main window class."""
    isolated_config = tempfile.TemporaryDirectory(prefix="audioforge-smoke-") if smoke_test else None
    previous_config_env = {
        name: os.environ.get(name)
        for name in ("APPDATA", "XDG_CONFIG_HOME", "AUDIOFORGE_SMOKE_TEST")
    }
    if isolated_config is not None:
        os.environ["APPDATA"] = isolated_config.name
        os.environ["XDG_CONFIG_HOME"] = isolated_config.name
        os.environ["AUDIOFORGE_SMOKE_TEST"] = "1"

    try:
        return _run_qt_app(window_cls, smoke_test=smoke_test)
    finally:
        if isolated_config is not None:
            log_file = get_log_file()
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                if isinstance(handler, RotatingFileHandler) and Path(
                    handler.baseFilename
                ) == log_file:
                    root_logger.removeHandler(handler)
                    handler.close()
        for name, value in previous_config_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if isolated_config is not None:
            isolated_config.cleanup()


def _run_qt_app(window_cls: Type[QMainWindow], *, smoke_test: bool) -> int:
    """Create the app and event loop, optionally running a startup smoke test."""
    configure_app_logging()
    configure_windows_app_id()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(application_palette())

    app_icon = _application_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    configure_deepfilter_env()
    configure_vad_env()

    window = window_cls()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    apply_windows_window_icon(window)
    apply_windows_taskbar_properties(window)
    window.show()

    if smoke_test:

        def finish_smoke_test() -> None:
            try:
                if not window.isVisible():
                    raise RuntimeError("AudioForge main window did not become visible")
                processor = getattr(window, "processor", None)
                is_running = getattr(processor, "is_running", None)
                if not callable(is_running):
                    raise RuntimeError("AudioForge main window has no processor state")
                if bool(is_running()):
                    raise RuntimeError("startup smoke test unexpectedly started audio")
            except Exception:
                logging.getLogger(__name__).exception(
                    "AudioForge packaged startup smoke test failed"
                )
                app.exit(1)
                return

            window.close()
            app.exit(0)

        # A zero-delay timer runs only after QApplication has dispatched the
        # first event, proving that construction and the event loop both work.
        QTimer.singleShot(0, finish_smoke_test)

    return app.exec()


def run_smoke_test(window_cls: Type[QMainWindow]) -> int:
    """Construct the real window and event loop using isolated user data."""
    return run_qt_app(window_cls, smoke_test=True)
