from __future__ import annotations

from pathlib import Path

from mic_eq.ui import app_bootstrap


def _deepfilter_lib_name() -> str:
    if app_bootstrap.os.name == "nt":
        return "df.dll"
    if app_bootstrap.sys.platform == "darwin":
        return "libdf.dylib"
    return "libdf.so"


def _write_deepfilter_assets(root: Path) -> None:
    (root / _deepfilter_lib_name()).write_bytes(b"x")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "DeepFilterNet3_ll_onnx.tar.gz").write_bytes(b"x")


def _write_vad_asset(root: Path) -> None:
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "silero_vad.onnx").write_bytes(b"x")


def test_deepfilter_bootstrap_ignores_cwd_assets(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    trusted = tmp_path / "trusted"
    cwd.mkdir()
    trusted.mkdir()
    _write_deepfilter_assets(cwd)
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("AUDIOFORGE_ENABLE_DEEPFILTER", raising=False)
    monkeypatch.delenv("DEEPFILTER_LIB_PATH", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])

    app_bootstrap.configure_deepfilter_env()

    assert "AUDIOFORGE_ENABLE_DEEPFILTER" not in app_bootstrap.os.environ
    assert "DEEPFILTER_LIB_PATH" not in app_bootstrap.os.environ


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
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["AUDIOFORGE_ENABLE_DEEPFILTER"] == "1"
    assert app_bootstrap.os.environ["DEEPFILTER_LIB_PATH"] == str(
        trusted / _deepfilter_lib_name()
    )


def test_deepfilter_bootstrap_prefers_bundled_assets_over_external_env(
    tmp_path, monkeypatch
):
    trusted = tmp_path / "trusted"
    external = tmp_path / "external" / _deepfilter_lib_name()
    trusted.mkdir()
    external.parent.mkdir()
    external.write_bytes(b"x")
    _write_deepfilter_assets(trusted)
    monkeypatch.setenv("AUDIOFORGE_ENABLE_DEEPFILTER", "1")
    monkeypatch.setenv("DEEPFILTER_LIB_PATH", str(external))
    monkeypatch.delenv("AUDIOFORGE_ALLOW_EXTERNAL_DF", raising=False)
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["DEEPFILTER_LIB_PATH"] == str(
        trusted / _deepfilter_lib_name()
    )


def test_deepfilter_bootstrap_allows_explicit_external_override(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    external = tmp_path / "external" / _deepfilter_lib_name()
    trusted.mkdir()
    external.parent.mkdir()
    external.write_bytes(b"x")
    _write_deepfilter_assets(trusted)
    monkeypatch.setenv("AUDIOFORGE_ENABLE_DEEPFILTER", "1")
    monkeypatch.setenv("AUDIOFORGE_ALLOW_EXTERNAL_DF", "1")
    monkeypatch.setenv("DEEPFILTER_LIB_PATH", str(external))
    monkeypatch.setattr(app_bootstrap, "_trusted_runtime_roots", lambda: [trusted])

    app_bootstrap.configure_deepfilter_env()

    assert app_bootstrap.os.environ["DEEPFILTER_LIB_PATH"] == str(external)


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
