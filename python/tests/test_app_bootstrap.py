from __future__ import annotations

import logging
import runpy
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

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

    from PyQt6.QtWidgets import QMainWindow

    window = QMainWindow()
    try:
        app_bootstrap.apply_windows_taskbar_properties(window)
    finally:
        window.close()

    assert store.values["command"].value == app_bootstrap.subprocess.list2cmdline(
        [executable]
    )


def test_packaged_startup_smoke_runs_real_event_loop_in_isolated_config():
    script = """
from PyQt6.QtWidgets import QMainWindow
from mic_eq.ui.app_bootstrap import run_smoke_test

class Processor:
    def is_running(self):
        return False

class SmokeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = Processor()

raise SystemExit(run_smoke_test(SmokeWindow))
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

    assert result.returncode == 0, result.stderr or result.stdout
