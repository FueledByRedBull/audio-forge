"""Config save atomicity: the file is replaced whole or not at all."""

import json

import pytest

from mic_eq import config
from mic_eq.config_parts import app_config as app_config_module
from mic_eq.config_parts import shared as shared_module


AppConfig = config.AppConfig


# save_config: atomic write


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: path)
    return path


def test_save_config_writes_complete_json_and_leaves_no_temp_file(config_path):
    original = AppConfig(last_preset="Voice")

    assert app_config_module.save_config(original) is True

    assert AppConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8"))) == original
    assert [path.name for path in config_path.parent.iterdir()] == ["config.json"]


def test_save_config_replace_failure_keeps_the_original_and_removes_the_temp_file(
    config_path, monkeypatch
):
    config_path.write_text("existing", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(shared_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        app_config_module.save_config(AppConfig(last_preset="New"))

    assert config_path.read_text(encoding="utf-8") == "existing"
    assert [path.name for path in config_path.parent.iterdir()] == ["config.json"]


def test_save_config_fsync_failure_never_publishes_and_removes_the_temp_file(
    config_path, monkeypatch
):
    config_path.write_text("existing", encoding="utf-8")

    def fail_fsync(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(app_config_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        app_config_module.save_config(AppConfig(last_preset="New"))

    assert config_path.read_text(encoding="utf-8") == "existing"
    assert [path.name for path in config_path.parent.iterdir()] == ["config.json"]


@pytest.mark.parametrize("bad", [float("nan")])
def test_save_config_rejects_non_finite_values_without_touching_the_file(config_path, bad):
    config_path.write_text("existing", encoding="utf-8")
    broken = AppConfig()
    broken.to_dict = lambda: {"value": bad}  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        app_config_module.save_config(broken)

    assert config_path.read_text(encoding="utf-8") == "existing"
    assert [path.name for path in config_path.parent.iterdir()] == ["config.json"]
