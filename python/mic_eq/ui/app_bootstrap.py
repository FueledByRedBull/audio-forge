"""Application bootstrap helpers for the AudioForge UI."""

import contextlib
import ctypes
import faulthandler
import importlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Iterator, Type, cast

from PySide6.QtCore import (
    QCoreApplication, QEvent, QMessageLogContext, QtMsgType, QTimer, qInstallMessageHandler,
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from .. import configure_deepfilter_runtime_paths
from ..app_logging import configure_app_logging, get_log_file
from .desktop_integration import SingleInstanceCoordinator, activate_window
from .login_startup import LOGIN_STARTUP_ARGUMENT
from .theme import apply_application_theme


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


def _check_windows_shortcut_roundtrip() -> None:
    """Exercise native COM imports without registering anything at login."""
    if sys.platform != "win32":
        return
    import pythoncom
    from win32com.shell import shell  # pyright: ignore[reportMissingModuleSource]

    from .login_startup import _shell_link

    with tempfile.TemporaryDirectory(prefix="audioforge-shortcut-smoke-") as directory:
        path = Path(directory) / "probe.lnk"
        link = _shell_link()
        link.SetPath(sys.executable)
        link.SetArguments(LOGIN_STARTUP_ARGUMENT)
        link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(path), True)
        loaded = _shell_link()
        loaded.QueryInterface(pythoncom.IID_IPersistFile).Load(str(path))
        target, _ = loaded.GetPath(shell.SLGP_RAWPATH)
        if (
            os.path.normcase(target) != os.path.normcase(sys.executable)
            or loaded.GetArguments() != LOGIN_STARTUP_ARGUMENT
        ):
            raise RuntimeError("Windows shortcut roundtrip failed")


def _check_auto_eq_solvers() -> None:
    """Fit a synthetic bassy mic so the Auto-EQ solvers must load and converge."""
    import numpy as np

    from ..analysis.auto_eq import NUM_EQ_BANDS, calculate_eq_bands

    freqs = np.geomspace(20.0, 20_000.0, 64)
    measured_db = -70.0 + 10.0 / (1.0 + (freqs / 200.0) ** 2)
    fit = calculate_eq_bands(freqs, measured_db, np.zeros_like(freqs))
    gains = fit["band_gains"]
    # Both the nonlinear and the constraint solve converged, and the low bands
    # were cut as a bassy mic requires.
    if not (
        fit["nonlinear_solver_success"]
        and fit["constraint_solver_success"]
        and len(gains) == NUM_EQ_BANDS
        and np.all(np.isfinite(gains))
        and min(gains[:2]) < -1.0
    ):
        raise RuntimeError(
            "Auto-EQ solver check failed: "
            f"nonlinear={fit['nonlinear_solver_success']}, "
            f"constraint={fit['constraint_solver_success']}, gains={gains}"
        )


_REPEAT_KEYS_MAX = 256
_QT_MESSAGE_KEY_CHARS = 200
_OVERFLOW_KEY = "<keys beyond the table size>"


def _count_repeat(counts: dict, key: object) -> tuple[object, int, bool]:
    """Count one more occurrence of key; returns the key counted, its count, and whether it deserves a log
    line (1st, 10th, 100th, every 1000th).

    Keys that arrive once the table is full share one overflow counter (the key returned is then
    ``_OVERFLOW_KEY``), so the table never grows past the cap.
    """
    if key not in counts and len(counts) >= _REPEAT_KEYS_MAX:
        key = _OVERFLOW_KEY
    counts[key] = count = counts.get(key, 0) + 1
    return key, count, count in (1, 10, 100) or count % 1000 == 0


def _describe_exception_key(key: object) -> str:
    if not isinstance(key, tuple):
        return str(key)
    exc_type, place = key
    return f"{exc_type.__name__} at {place[0]}:{place[1]}" if place else exc_type.__name__


def _describe_qt_key(key: object) -> str:
    if not isinstance(key, tuple):
        return str(key)
    _kind, category, message = key
    return f"{category}: {message}"


_QT_LOG_LEVELS = {
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


@contextlib.contextmanager
def _crash_diagnostics() -> Iterator[Callable[[], None]]:
    """Send uncaught exceptions, Qt warnings and hard crashes to the app log, then restore.

    Qt messages reach stderr from the start but the app log only once the yielded callable is
    called: a second launch must not open app.log, whose rollover belongs to the first instance.
    """
    logger = logging.getLogger(__name__)
    qt_logger = logging.getLogger("qt")
    previous_excepthook = sys.excepthook
    previous_thread_hook = threading.excepthook
    exception_counts: dict = {}
    qt_counts: dict = {}
    in_qt_handler = threading.local()
    qt_to_log = threading.Event()

    def log_uncaught(exc_type, exc_value, exc_traceback) -> None:
        # A slot that fails on every timer tick would otherwise flush the log in seconds.
        origin = exc_traceback
        while origin is not None and origin.tb_next is not None:
            origin = origin.tb_next
        place = (origin.tb_frame.f_code.co_filename, origin.tb_lineno) if origin else None
        _, count, log_it = _count_repeat(exception_counts, (exc_type, place))
        if count == 1:
            logger.critical(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
            )
        elif log_it:
            logger.critical("Uncaught exception repeated %d times: %r", count, exc_value)
        previous_excepthook(exc_type, exc_value, exc_traceback)

    def log_thread_uncaught(args) -> None:
        if args.exc_type is not SystemExit:
            logger.critical(
                "Uncaught exception in thread %s",
                getattr(args.thread, "name", "?"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        previous_thread_hook(args)

    def log_qt_message(kind: QtMsgType, context: QMessageLogContext, message: str) -> None:
        # A windowed build has no stderr, so QML errors would otherwise leave no trace.
        if sys.stderr is not None:
            print(message, file=sys.stderr)
        level = _QT_LOG_LEVELS.get(kind)
        if level is None or not qt_to_log.is_set() or getattr(in_qt_handler, "busy", False):
            return
        in_qt_handler.busy = True
        try:
            key = (kind, context.category, message[:_QT_MESSAGE_KEY_CHARS])
            key, count, log_it = _count_repeat(qt_counts, key)
            if log_it:
                suffix = ""
                if key is _OVERFLOW_KEY:
                    suffix = f" (message {count} beyond the {_REPEAT_KEYS_MAX} distinct ones tracked)"
                elif count > 1:
                    suffix = f" (seen {count} times)"
                qt_logger.log(level, "%s: %s%s", context.category, message, suffix)
        finally:
            in_qt_handler.busy = False

    # A sibling file: a second handle on app.log would block its rotation on Windows.
    crash_file = None
    try:
        crash_file = open(
            configure_app_logging().with_name("crash.log"), "a", encoding="utf-8"
        )
        faulthandler.enable(crash_file)
    except OSError:
        pass

    sys.excepthook = log_uncaught
    threading.excepthook = log_thread_uncaught
    previous_qt_handler = qInstallMessageHandler(log_qt_message)
    try:
        yield qt_to_log.set
    except Exception:
        # The hooks are restored below, before any top-level handler could log this.
        logger.critical("Uncaught exception", exc_info=True)
        raise
    finally:
        qInstallMessageHandler(previous_qt_handler)
        sys.excepthook = previous_excepthook
        threading.excepthook = previous_thread_hook
        for key, count in exception_counts.items():
            if count > 1:
                logger.critical("Uncaught exception %s: %d occurrences in total, repeats logged sparsely", _describe_exception_key(key), count)
        if qt_to_log.is_set():
            for key, count in qt_counts.items():
                if count > 1:
                    qt_logger.warning("%s: %d occurrences in total, repeats logged sparsely", _describe_qt_key(key), count)
        if crash_file is not None:
            faulthandler.disable()
            # Another instance can hold crash.log open, and Windows then refuses
            # the delete; an empty leftover file is harmless.
            with contextlib.suppress(OSError):
                crash_file.close()
                if not Path(crash_file.name).stat().st_size:
                    Path(crash_file.name).unlink(missing_ok=True)


def run_qt_app(window_cls: Type[QMainWindow], *, smoke_test: bool = False) -> int:
    """Run the Qt application for the provided main window class."""
    isolated_config = tempfile.TemporaryDirectory(prefix="audioforge-smoke-") if smoke_test else None
    previous_config_env = {
        name: os.environ.get(name)
        for name in (
            "APPDATA", "XDG_CONFIG_HOME", "AUDIOFORGE_SMOKE_TEST",
            "QML_DISK_CACHE_PATH", "QT_DISABLE_SHADER_DISK_CACHE",
        )
    }
    if isolated_config is not None:
        os.environ["APPDATA"] = isolated_config.name
        os.environ["XDG_CONFIG_HOME"] = isolated_config.name
        os.environ["AUDIOFORGE_SMOKE_TEST"] = "1"
        # Qt's persistent caches can overlap the Windows MSI installation path.
        os.environ["QML_DISK_CACHE_PATH"] = str(Path(isolated_config.name) / "qmlcache")
        os.environ["QT_DISABLE_SHADER_DISK_CACHE"] = "1"

    try:
        with _crash_diagnostics() as enable_qt_log:
            return _run_qt_app(window_cls, smoke_test=smoke_test, enable_qt_log=enable_qt_log)
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


def _run_qt_app(
    window_cls: Type[QMainWindow],
    *,
    smoke_test: bool,
    enable_qt_log: Callable[[], None],
) -> int:
    """Create the app and event loop, optionally running a startup smoke test."""
    configure_app_logging()
    configure_windows_app_id()

    login_startup = not smoke_test and LOGIN_STARTUP_ARGUMENT in sys.argv
    app = QApplication([arg for arg in sys.argv if arg != LOGIN_STARTUP_ARGUMENT])
    apply_application_theme(app)

    app_icon = _application_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    instance = None
    if not smoke_test:
        instance = SingleInstanceCoordinator()
        instance_status = (
            instance.acquire(activate_existing=False) if login_startup else instance.acquire()
        )
        if instance_status in (
            SingleInstanceCoordinator.FORWARDED, SingleInstanceCoordinator.ALREADY_RUNNING
        ):
            return 0
        if instance_status != SingleInstanceCoordinator.ACQUIRED:
            logging.getLogger(__name__).error(
                "Could not start the single AudioForge instance: %s",
                instance.last_error or "unknown error",
            )
            if not login_startup:
                QMessageBox.critical(
                    None,
                    "AudioForge Could Not Open",
                    "Another AudioForge session may already be running, but its window "
                    "could not be reached. Check the tray icon or close that session "
                    "before opening AudioForge again.\n\n"
                    + (instance.last_error or "The session lock could not be acquired."),
                )
            return 1
        app.aboutToQuit.connect(instance.close)

    enable_qt_log()
    try:
        configure_deepfilter_env()
        configure_vad_env()
        window = (
            cast(Callable[..., QMainWindow], window_cls)(login_startup=True)
            if login_startup else window_cls()
        )
        setattr(window, "single_instance", instance)
        if not app_icon.isNull():
            window.setWindowIcon(app_icon)
        apply_windows_window_icon(window)
        apply_windows_taskbar_properties(window)
        if login_startup:
            getattr(window, "begin_login_startup")()
        else:
            window.show()
        if instance is not None:
            instance.set_activation_callback(lambda: activate_window(window))
    except Exception:
        if instance is not None:
            instance.close()
        if login_startup:
            logging.getLogger(__name__).exception("AudioForge login startup failed")
            return 1
        # The re-raise below is logged by _crash_diagnostics.
        if getattr(sys, "frozen", False) and not smoke_test:
            QMessageBox.critical(
                None,
                "AudioForge Could Not Start",
                "AudioForge hit an error while starting and could not open. "
                f"Details are in the log file:\n\n{get_log_file()}",
            )
        raise

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
                failure = getattr(window, "quick_view_failure", None)
                if failure is not None:
                    raise RuntimeError(f"the Qt Quick view did not load ({failure})")
                _check_windows_shortcut_roundtrip()
                _check_auto_eq_solvers()
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

    code = app.exec()
    # Destroy the window and everything parented to it while QApplication still exists: left to
    # interpreter shutdown, they are freed in an order that crashes on the native Windows platform.
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return code


def run_smoke_test(window_cls: Type[QMainWindow]) -> int:
    """Construct the real window and event loop using isolated user data."""
    return run_qt_app(window_cls, smoke_test=True)
