from __future__ import annotations

import json
import logging
import runpy
import subprocess
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from mic_eq.app_logging import configure_app_logging, get_log_file
from mic_eq.ui import app_bootstrap


def _deepfilter_lib_name() -> str:
    if app_bootstrap.os.name == "nt":
        return "df.dll"
    if app_bootstrap.sys.platform == "darwin":
        return "libdf.dylib"
    return "libdf.so"


def _write_deepfilter_assets(root: Path, *, low_latency: bool = True) -> None:
    (root / _deepfilter_lib_name()).write_bytes(b"x")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    name = "DeepFilterNet3_ll_onnx.tar.gz" if low_latency else "DeepFilterNet3_onnx.tar.gz"
    (models / name).write_bytes(b"x")


def _write_vad_asset(root: Path) -> None:
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "silero_vad.onnx").write_bytes(b"x")


def _capture_deepfilter_registration(monkeypatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_bootstrap,
        "configure_deepfilter_runtime_paths",
        lambda library, model: calls.append((library, model)),
    )
    return calls


def test_deepfilter_bootstrap_ignores_cwd_assets(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    trusted = tmp_path / "trusted"
    cwd.mkdir()
    trusted.mkdir()
    _write_deepfilter_assets(cwd)
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.delenv("DEEPFILTER_LIB_PATH", raising=False)
    monkeypatch.delenv("DEEPFILTER_MODEL_PATH", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])
    registrations = _capture_deepfilter_registration(monkeypatch)

    app_bootstrap.configure_deepfilter_env()

    assert "AUDIOFORGE_ENABLE_DEEPFILTER" not in app_bootstrap.os.environ
    assert "DEEPFILTER_LIB_PATH" not in app_bootstrap.os.environ
    assert "DEEPFILTER_MODEL_PATH" not in app_bootstrap.os.environ
    assert registrations == []


def test_deepfilter_bootstrap_uses_trusted_runtime_root(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    trusted = tmp_path / "trusted"
    cwd.mkdir()
    trusted.mkdir()
    _write_deepfilter_assets(cwd)
    _write_deepfilter_assets(trusted)
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.delenv("DEEPFILTER_LIB_PATH", raising=False)
    monkeypatch.delenv("DEEPFILTER_MODEL_PATH", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])
    registrations = _capture_deepfilter_registration(monkeypatch)

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] == "1"
    assert "DEEPFILTER_LIB_PATH" not in app_bootstrap.os.environ
    assert "DEEPFILTER_MODEL_PATH" not in app_bootstrap.os.environ
    assert registrations == [
        (str(trusted / _deepfilter_lib_name()), str(trusted / "models"))
    ]


def test_deepfilter_bootstrap_registers_bundled_assets_without_trusting_external_env(
    tmp_path, monkeypatch
):
    trusted = tmp_path / "trusted"
    external = tmp_path / "external" / _deepfilter_lib_name()
    external_model = tmp_path / "external" / "DeepFilterNet3_ll_onnx.tar.gz"
    trusted.mkdir()
    external.parent.mkdir()
    external.write_bytes(b"x")
    external_model.write_bytes(b"x")
    _write_deepfilter_assets(trusted)
    monkeypatch.setenv("AUDIOFORGE_ENABLE_DEEPFILTER", "1")
    monkeypatch.setenv("DEEPFILTER_LIB_PATH", str(external))
    monkeypatch.setenv("DEEPFILTER_MODEL_PATH", str(external_model))
    monkeypatch.delenv("AUDIOFORGE_ALLOW_EXTERNAL_DF", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])
    registrations = _capture_deepfilter_registration(monkeypatch)

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["DEEPFILTER_LIB_PATH"] == str(external)
    assert app_bootstrap.os.environ["DEEPFILTER_MODEL_PATH"] == str(external_model)
    assert registrations == [
        (str(trusted / _deepfilter_lib_name()), str(trusted / "models"))
    ]


def test_deepfilter_bootstrap_allows_explicit_external_override(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    external = tmp_path / "external" / _deepfilter_lib_name()
    external_model = tmp_path / "external" / "DeepFilterNet3_ll_onnx.tar.gz"
    trusted.mkdir()
    external.parent.mkdir()
    external.write_bytes(b"x")
    external_model.write_bytes(b"x")
    _write_deepfilter_assets(trusted)
    monkeypatch.setenv("AUDIOFORGE_ENABLE_DEEPFILTER", "1")
    monkeypatch.setenv("AUDIOFORGE_ALLOW_EXTERNAL_DF", "1")
    monkeypatch.setenv("DEEPFILTER_LIB_PATH", str(external))
    monkeypatch.setenv("DEEPFILTER_MODEL_PATH", str(external_model))
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])
    registrations = _capture_deepfilter_registration(monkeypatch)

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["DEEPFILTER_LIB_PATH"] == str(external)
    assert app_bootstrap.os.environ["DEEPFILTER_MODEL_PATH"] == str(external_model)
    assert registrations == [
        (str(trusted / _deepfilter_lib_name()), str(trusted / "models"))
    ]


def test_deepfilter_bootstrap_external_override_gets_missing_bundled_defaults(
    tmp_path, monkeypatch
):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    _write_deepfilter_assets(trusted)
    monkeypatch.setenv("AUDIOFORGE_ALLOW_EXTERNAL_DF", "1")
    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.delenv("DEEPFILTER_LIB_PATH", raising=False)
    monkeypatch.delenv("DEEPFILTER_MODEL_PATH", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])
    registrations = _capture_deepfilter_registration(monkeypatch)

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] == "1"
    assert "DEEPFILTER_LIB_PATH" not in app_bootstrap.os.environ
    assert "DEEPFILTER_MODEL_PATH" not in app_bootstrap.os.environ
    assert registrations == [
        (str(trusted / _deepfilter_lib_name()), str(trusted / "models"))
    ]


def test_deepfilter_bootstrap_uses_standard_model_when_ll_absent(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    _write_deepfilter_assets(trusted, low_latency=False)
    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.delenv("DEEPFILTER_LIB_PATH", raising=False)
    monkeypatch.delenv("DEEPFILTER_MODEL_PATH", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])
    registrations = _capture_deepfilter_registration(monkeypatch)

    app_bootstrap.configure_deepfilter_env()

    assert registrations == [
        (str(trusted / _deepfilter_lib_name()), str(trusted / "models"))
    ]


def test_vad_bootstrap_uses_trusted_runtime_model(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    trusted = tmp_path / "trusted"
    cwd.mkdir()
    trusted.mkdir()
    _write_vad_asset(cwd)
    _write_vad_asset(trusted)
    monkeypatch.chdir(cwd)
    # ``configure_vad_env`` may create this key; isolate the whole mapping so
    # a key that was absent before the test is restored after it finishes.
    monkeypatch.setattr(app_bootstrap.os, "environ", app_bootstrap.os.environ.copy())
    monkeypatch.delenv("VAD_MODEL_PATH", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])

    app_bootstrap.configure_vad_env()

    assert app_bootstrap.os.environ["VAD_MODEL_PATH"] == str(
        trusted / "models" / "silero_vad.onnx"
    )


def test_vad_bootstrap_preserves_explicit_override(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    external = tmp_path / "external" / "silero_vad.onnx"
    trusted.mkdir()
    external.parent.mkdir()
    external.write_bytes(b"x")
    _write_vad_asset(trusted)
    monkeypatch.setenv("VAD_MODEL_PATH", str(external))
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])

    app_bootstrap.configure_vad_env()

    assert app_bootstrap.os.environ["VAD_MODEL_PATH"] == str(external)


def test_app_logging_uses_audioforge_rotating_log(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    log_file = get_log_file()

    configured = configure_app_logging()

    try:
        assert configured == log_file
        assert log_file == tmp_path / "AudioForge" / "logs" / "app.log"
        assert log_file.parent.is_dir()
        assert any(
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == log_file
            for handler in logging.getLogger().handlers
        )
    finally:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if (
                isinstance(handler, RotatingFileHandler)
                and Path(handler.baseFilename) == log_file
            ):
                root_logger.removeHandler(handler)
                handler.close()


def test_launcher_leaves_model_discovery_to_app_bootstrap(tmp_path, monkeypatch):
    exe_dir = tmp_path / "dist" / "AudioForge"
    meipass = tmp_path / "meipass"
    exe_dir.mkdir(parents=True)
    _write_vad_asset(meipass / "_internal")
    monkeypatch.delenv("VAD_MODEL_PATH", raising=False)
    monkeypatch.delenv("DEEPFILTER_LIB_PATH", raising=False)
    monkeypatch.delenv("DEEPFILTER_MODEL_PATH", raising=False)
    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "AudioForge.exe"))

    launcher_path = Path(__file__).resolve().parents[2] / "launcher.py"
    runpy.run_path(str(launcher_path))

    assert "VAD_MODEL_PATH" not in app_bootstrap.os.environ


def test_windows_taskbar_relaunch_command_quotes_executable_path(qapp, monkeypatch):
    class Store:
        def __init__(self):
            self.values = {}

        def SetValue(self, key, value):
            self.values[key] = value

        def Commit(self):
            return None

    class PropVariant:
        def __init__(self, value):
            self.value = value

    store = Store()
    propsys = type(
        "Propsys",
        (),
        {
            "IID_IPropertyStore": object(),
            "PROPVARIANTType": PropVariant,
            "SHGetPropertyStoreForWindow": lambda *_args: store,
        },
    )
    pscon = type(
        "Pscon",
        (),
        {
            "PKEY_AppUserModel_ID": "id",
            "PKEY_AppUserModel_RelaunchCommand": "command",
            "PKEY_AppUserModel_RelaunchDisplayNameResource": "display",
            "PKEY_AppUserModel_RelaunchIconResource": "icon",
        },
    )
    modules = {
        "win32com.propsys.propsys": propsys,
        "win32com.propsys.pscon": pscon,
    }
    monkeypatch.setattr(
        app_bootstrap.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(app_bootstrap.sys, "frozen", True, raising=False)
    executable = r"C:\Program Files\AudioForge\AudioForge.exe"
    monkeypatch.setattr(app_bootstrap.sys, "executable", executable)

    from PySide6.QtWidgets import QMainWindow

    window = QMainWindow()
    try:
        app_bootstrap.apply_windows_taskbar_properties(window)
    finally:
        window.close()

    assert store.values["command"].value == app_bootstrap.subprocess.list2cmdline(
        [executable]
    )


@pytest.mark.parametrize("missing_timezone", [False, True] if sys.platform == "win32" else [False])
def test_packaged_startup_smoke_runs_real_event_loop_in_isolated_config(tmp_path, missing_timezone):
    qml_source = tmp_path / "Smoke.qml"
    qml_source.write_text("import QtQml\nQtObject { property int value: 42 }\n", encoding="utf-8")
    script = """
import json
import os
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlComponent, QQmlEngine
from PySide6.QtQuick import QQuickGraphicsConfiguration
from PySide6.QtWidgets import QMainWindow
from mic_eq.ui.app_bootstrap import run_smoke_test

previous_cache = os.environ["QML_DISK_CACHE_PATH"]
previous_shader_cache = os.environ["QT_DISABLE_SHADER_DISK_CACHE"]
observed = {}

if os.environ["SMOKE_MISSING_TIMEZONE"] == "1":
    import sys
    sys.modules.pop("win32timezone", None)
    class MissingTimezone:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "win32timezone":
                observed["timezone_import_blocked"] = True
                raise ModuleNotFoundError("simulated missing win32timezone")
    sys.meta_path.insert(0, MissingTimezone())

class Processor:
    def is_running(self):
        return False

class SmokeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = Processor()
        observed["config"] = os.environ["APPDATA"]
        observed["cache"] = os.environ["QML_DISK_CACHE_PATH"]
        observed["pipeline_cache"] = QQuickGraphicsConfiguration().isAutomaticPipelineCacheEnabled()
        engine = QQmlEngine()
        component = QQmlComponent(engine, QUrl.fromLocalFile(os.environ["SMOKE_QML_SOURCE"]))
        assert component.status() == QQmlComponent.Status.Ready, component.errors()
        observed["cache_files"] = len(list(Path(observed["cache"]).glob("*.qmlc")))

status = run_smoke_test(SmokeWindow)
assert os.environ["QML_DISK_CACHE_PATH"] == previous_cache
assert os.environ["QT_DISABLE_SHADER_DISK_CACHE"] == previous_shader_cache
print(json.dumps(observed))
raise SystemExit(status)
"""
    environment = dict(app_bootstrap.os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["QML_DISK_CACHE_PATH"] = str(tmp_path / "persistent-cache")
    environment["QT_DISABLE_SHADER_DISK_CACHE"] = "0"
    environment["SMOKE_QML_SOURCE"] = str(qml_source)
    environment["SMOKE_MISSING_TIMEZONE"] = str(int(missing_timezone))
    environment.pop("QML_DISABLE_DISK_CACHE", None)
    environment.pop("QML_DISK_CACHE", None)
    environment.pop("QSG_RHI_DISABLE_SHADER_DISK_CACHE", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == int(missing_timezone), result.stderr or result.stdout
    observed = json.loads(result.stdout)
    if missing_timezone:
        assert observed["timezone_import_blocked"] is True
    assert observed["cache_files"] > 0
    assert Path(observed["cache"]).is_relative_to(Path(observed["config"]))
    assert not Path(observed["cache"]).exists()
    assert not (tmp_path / "persistent-cache").exists()
    assert observed["pipeline_cache"] is False


def test_auto_eq_solver_check_runs_all_eq_fit_solvers(monkeypatch):
    from mic_eq.analysis.auto_eq_parts import optimizer

    calls = []

    def recording(name):
        real = getattr(optimizer, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)

        return wrapper

    for name in ("bounded_least_squares", "project_gains", "refine_gains"):
        monkeypatch.setattr(optimizer, name, recording(name))

    app_bootstrap._check_auto_eq_solvers()

    assert calls.count("bounded_least_squares") == 2  # stage 1 and stage 2
    assert "project_gains" in calls and "refine_gains" in calls


def test_auto_eq_solver_check_rejects_a_fit_that_did_not_converge_or_correct(monkeypatch):
    from mic_eq.analysis import auto_eq

    real_fit = auto_eq.calculate_eq_bands
    fits = []

    def recording_fit(*args, **kwargs):
        fits.append(real_fit(*args, **kwargs))
        return fits[-1]

    monkeypatch.setattr(auto_eq, "calculate_eq_bands", recording_fit)
    app_bootstrap._check_auto_eq_solvers()
    converged = fits[0]

    for broken in (
        {"nonlinear_solver_success": False},
        {"constraint_solver_success": False},
        {"band_gains": [0.0] * 10},
        {"band_gains": [float("nan")] + converged["band_gains"][1:]},
        {"band_gains": converged["band_gains"][:-1]},
    ):
        monkeypatch.setattr(
            auto_eq, "calculate_eq_bands", lambda *args, _broken=broken, **kwargs: {**converged, **_broken}
        )
        with pytest.raises(RuntimeError, match="Auto-EQ solver check failed"):
            app_bootstrap._check_auto_eq_solvers()


def test_packaged_startup_smoke_fails_when_the_eq_solver_module_cannot_load():
    script = """
import json
import logging
import sys
from PySide6.QtWidgets import QMainWindow
from mic_eq.ui.app_bootstrap import run_smoke_test

class Processor:
    def is_running(self):
        return False

class SmokeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = Processor()

messages = []

class Capture(logging.Handler):
    def emit(self, record):
        messages.append(self.format(record))

logging.getLogger().addHandler(Capture())
sys.modules["mic_eq.analysis.auto_eq_parts.eq_fit"] = None
status = run_smoke_test(SmokeWindow)
print(json.dumps({"status": status, "log": " ".join(messages)}))
raise SystemExit(status)
"""
    environment = dict(app_bootstrap.os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1, result.stderr or result.stdout
    log = json.loads(result.stdout)["log"]
    assert "startup smoke test failed" in log
    assert "auto_eq_parts.eq_fit" in log


@pytest.fixture
def crash_diagnostics(monkeypatch, caplog):
    """Run `_crash_diagnostics` with faulthandler faked and the log handler removed after."""
    import faulthandler

    calls: list = []
    monkeypatch.setattr(faulthandler, "enable", lambda file=None: calls.append(("enable", file)))
    monkeypatch.setattr(faulthandler, "disable", lambda: calls.append(("disable", None)))
    # monkeypatch restores both hooks even if a test fails inside the context.
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    caplog.set_level(logging.INFO)
    yield calls
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, RotatingFileHandler):
            root_logger.removeHandler(handler)
            handler.close()


def test_uncaught_exception_is_logged_with_traceback_and_chained(crash_diagnostics, caplog):
    seen = []
    sys.excepthook = lambda *info: seen.append(info)
    installed = sys.excepthook

    with app_bootstrap._crash_diagnostics():
        assert sys.excepthook is not installed
        try:
            raise ValueError("slot exploded")
        except ValueError:
            sys.excepthook(*sys.exc_info())

    record = next(r for r in caplog.records if r.getMessage() == "Uncaught exception")
    assert record.levelno == logging.CRITICAL
    assert record.exc_info is not None and "slot exploded" in str(record.exc_info[1])
    assert len(seen) == 1 and seen[0][0] is ValueError
    assert sys.excepthook is installed


def test_uncaught_thread_exception_is_logged_and_chained(crash_diagnostics, caplog):
    seen = []
    threading.excepthook = seen.append

    with app_bootstrap._crash_diagnostics():
        def fail():
            raise ValueError("worker exploded")

        worker = threading.Thread(target=fail, name="probe-worker")
        worker.start()
        worker.join()

    record = next(r for r in caplog.records if "probe-worker" in r.getMessage())
    assert record.exc_info is not None and "worker exploded" in str(record.exc_info[1])
    assert len(seen) == 1


def test_faulthandler_writes_next_to_the_app_log_and_leaves_no_empty_file(crash_diagnostics):
    crash_file = get_log_file().with_name("crash.log")

    with app_bootstrap._crash_diagnostics():
        assert crash_diagnostics[0][0] == "enable"
        assert Path(crash_diagnostics[0][1].name) == crash_file
    assert crash_diagnostics[-1][0] == "disable"
    assert not crash_file.exists()

    with app_bootstrap._crash_diagnostics():
        crash_diagnostics[-1][1].write("Fatal Python error\n")
    assert crash_file.read_text(encoding="utf-8") == "Fatal Python error\n"


def test_exception_leaving_the_context_is_logged_once_and_reraised(crash_diagnostics, caplog):
    with pytest.raises(RuntimeError, match="exec failed"):
        with app_bootstrap._crash_diagnostics():
            raise RuntimeError("exec failed")

    logged = [r for r in caplog.records if r.exc_info]
    assert len(logged) == 1 and logged[0].levelno == logging.CRITICAL


@pytest.mark.parametrize("exit_exception", [SystemExit(0), KeyboardInterrupt()])
def test_exit_requests_leaving_the_context_are_not_logged(
    crash_diagnostics, caplog, exit_exception
):
    with pytest.raises(type(exit_exception)):
        with app_bootstrap._crash_diagnostics():
            raise exit_exception

    assert not [r for r in caplog.records if r.exc_info]


def test_crash_file_cleanup_failure_does_not_change_the_run_result(
    crash_diagnostics, monkeypatch
):
    # Another instance can hold crash.log open; Windows then refuses the delete.
    def refuse(self, missing_ok=False):
        raise PermissionError(32, "in use by another process")

    monkeypatch.setattr(Path, "unlink", refuse)
    monkeypatch.setattr(app_bootstrap, "_run_qt_app", lambda *_args, **_kwargs: 0)

    assert app_bootstrap.run_qt_app(app_bootstrap.QMainWindow) == 0


@pytest.mark.parametrize(
    ("frozen", "smoke_test", "dialog_expected"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_startup_failure_is_logged_once_and_only_the_windowed_app_shows_a_dialog(
    qapp, monkeypatch, caplog, crash_diagnostics, frozen, smoke_test, dialog_expected
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    instance = SimpleNamespace(
        acquire=Mock(return_value="acquired"), close=Mock(), last_error=None,
    )
    factory = Mock(return_value=instance, ACQUIRED="acquired", FORWARDED="forwarded")
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    for name in ("configure_windows_app_id", "configure_deepfilter_env", "configure_vad_env"):
        monkeypatch.setattr(app_bootstrap, name, lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["AudioForge.exe"])
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    dialog = Mock()
    monkeypatch.setattr(app_bootstrap.QMessageBox, "critical", dialog)
    caplog.set_level(logging.ERROR)

    def broken_window():
        raise RuntimeError("saved configuration failed")

    with pytest.raises(RuntimeError, match="saved configuration failed"):
        with app_bootstrap._crash_diagnostics():
            app_bootstrap._run_qt_app(broken_window, smoke_test=smoke_test)  # type: ignore[arg-type]

    logged = [r for r in caplog.records if r.exc_info]
    assert len(logged) == 1
    assert "saved configuration failed" in str(logged[0].exc_info[1])  # type: ignore[index]
    assert dialog.called is dialog_expected
    if dialog_expected:
        assert str(get_log_file()) in dialog.call_args.args[2]
