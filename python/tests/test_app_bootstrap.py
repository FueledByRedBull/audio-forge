from __future__ import annotations

import logging
import runpy
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


def test_launcher_preserves_explicit_vad_override_in_frozen_bundle(tmp_path, monkeypatch):
    exe_dir = tmp_path / "dist" / "AudioForge"
    meipass = tmp_path / "meipass"
    external = tmp_path / "external" / "silero_vad.onnx"
    exe_dir.mkdir(parents=True)
    external.parent.mkdir()
    external.write_bytes(b"x")
    _write_vad_asset(meipass / "_internal")
    monkeypatch.setenv("VAD_MODEL_PATH", str(external))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "AudioForge.exe"))

    launcher_path = Path(__file__).resolve().parents[2] / "launcher.py"
    runpy.run_path(str(launcher_path))

    assert app_bootstrap.os.environ["VAD_MODEL_PATH"] == str(external)
