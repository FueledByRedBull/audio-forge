"""Application bootstrap helpers for the AudioForge UI."""

import ctypes
import os
import sys
from pathlib import Path
from typing import Type

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow


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


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def configure_deepfilter_env() -> None:
    """Enable local DeepFilter runtime when local assets are present."""
    allow_external = _truthy_env("AUDIOFORGE_ALLOW_EXTERNAL_DF")

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
        if allow_external:
            os.environ.setdefault("DEEPFILTER_LIB_PATH", str(lib_path))
            os.environ.setdefault("DEEPFILTER_MODEL_PATH", str(model_dir_with_model))
        else:
            os.environ["DEEPFILTER_LIB_PATH"] = str(lib_path)
            os.environ["DEEPFILTER_MODEL_PATH"] = str(model_dir_with_model)
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
        icon_path = root / "mic_eq.ico"
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
        from win32com.propsys import propsys, pscon

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
            propsys.PROPVARIANTType(exe_path),
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


def run_qt_app(window_cls: Type[QMainWindow]) -> int:
    """Run the Qt application for the provided main window class."""
    configure_windows_app_id()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

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

    return app.exec()
