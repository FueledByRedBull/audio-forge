from __future__ import annotations

from pathlib import Path

from mic_eq.ui import app_bootstrap


def _write_deepfilter_assets(root: Path) -> None:
    (root / "df.dll").write_bytes(b"x")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "DeepFilterNet3_ll_onnx.tar.gz").write_bytes(b"x")


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
    assert app_bootstrap.os.environ["DEEPFILTER_LIB_PATH"] == str(trusted / "df.dll")
