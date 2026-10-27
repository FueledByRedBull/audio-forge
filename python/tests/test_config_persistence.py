"""Config save atomicity and the legacy-folder migration state machine (marker, staging, swap, restore)."""

import json
import logging
import os
from pathlib import Path

import pytest

from mic_eq import app_logging, config
from mic_eq.config_parts import app_config as app_config_module
from mic_eq.config_parts import shared as shared_module


AppConfig = config.AppConfig


# save_config: atomic write


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: path)
    monkeypatch.setattr(app_config_module, "migration_pending", lambda: False)
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


# Legacy folder migration


LEGACY = shared_module.LEGACY_APPDATA_DIR_NAME
CURRENT = shared_module.APPDATA_DIR_NAME


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    base_dir = tmp_path / "appdata"
    base_dir.mkdir()
    monkeypatch.setattr(shared_module, "_config_base_dir", lambda: base_dir)
    return base_dir


def _folder(base_dir: Path, name: str, text: str) -> Path:
    folder = base_dir / name
    folder.mkdir()
    (folder / "config.json").write_text(text, encoding="utf-8")
    return folder


def _marker(base_dir: Path, state: str | None = None) -> Path:
    marker = shared_module._migration_marker(base_dir)
    if state is not None:
        marker.write_text(state, encoding="ascii")
    return marker


def _partials(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob(f".{CURRENT}.partial-*"))


def test_staged_marker_with_a_published_destination_is_cleared_without_copying(
    appdata, monkeypatch
):
    _folder(appdata, LEGACY, "legacy")
    current = _folder(appdata, CURRENT, "migrated")
    _marker(appdata, "staged")
    monkeypatch.setattr(
        shared_module.shutil, "copytree", lambda *a, **k: pytest.fail("must not copy again")
    )

    assert shared_module._config_dir() == current

    assert not _marker(appdata).exists()
    assert not shared_module.migration_pending()
    assert (current / "config.json").read_text(encoding="utf-8") == "migrated"


def test_staged_marker_that_cannot_be_removed_does_not_block_saves(appdata, monkeypatch):
    _folder(appdata, CURRENT, "migrated")
    _marker(appdata, "staged")
    real_unlink = Path.unlink

    def refuse_marker(self, *args, **kwargs):
        if self == _marker(appdata):
            raise PermissionError("locked")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_marker)

    shared_module._config_dir()

    assert _marker(appdata).exists()
    assert not shared_module.migration_pending()


def test_successful_migration_copies_the_legacy_folder_and_clears_the_marker(appdata):
    _folder(appdata, LEGACY, "legacy")

    config_dir = shared_module._config_dir()

    assert (config_dir / "config.json").read_text(encoding="utf-8") == "legacy"
    assert (appdata / LEGACY / "config.json").read_text(encoding="utf-8") == "legacy"
    assert not _marker(appdata).exists()
    assert not shared_module.migration_pending()


def test_pending_marker_swaps_in_the_legacy_copy_and_keeps_the_previous_destination(
    appdata, caplog
):
    _folder(appdata, LEGACY, "legacy")
    current = _folder(appdata, CURRENT, "first attempt")
    _marker(appdata, "pending")

    with caplog.at_level(logging.WARNING, logger=shared_module.logger.name):
        assert shared_module._config_dir() == current

    (kept,) = _partials(appdata)
    assert (current / "config.json").read_text(encoding="utf-8") == "legacy"
    assert (kept / "config.json").read_text(encoding="utf-8") == "first attempt"
    assert str(kept) in caplog.text
    assert not _marker(appdata).exists()


def test_failed_swap_restores_the_previous_destination_and_stays_pending(appdata, monkeypatch):
    _folder(appdata, LEGACY, "legacy")
    current = _folder(appdata, CURRENT, "mine")
    _marker(appdata, "pending")
    real_replace = Path.replace

    def fail_publication(self, target):
        if Path(target) == current and self.name.startswith(f".{CURRENT}.migrate-"):
            raise OSError("simulated publish failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publication)

    assert shared_module._config_dir() == current

    assert (current / "config.json").read_text(encoding="utf-8") == "mine"
    assert _partials(appdata) == []
    assert list(appdata.glob(f".{CURRENT}.migrate-*")) == []
    assert shared_module.migration_pending()
    assert app_config_module.save_config(AppConfig()) is False
    assert (current / "config.json").read_text(encoding="utf-8") == "mine"


def test_pending_marker_without_a_destination_loads_defaults_and_blocks_saves(
    appdata, monkeypatch
):
    _folder(appdata, LEGACY, "{}")
    _marker(appdata, "pending")
    monkeypatch.setattr(
        shared_module.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("busy"))
    )

    loaded = app_config_module.load_config()

    assert loaded == AppConfig()
    assert "config migration" in (loaded.load_warning or "")
    assert loaded.save_blocked_reason
    assert app_config_module.save_config(loaded) is False
    assert not (appdata / CURRENT).exists()


def test_pending_marker_with_a_readable_destination_loads_it_but_blocks_saves(
    appdata, monkeypatch
):
    _folder(appdata, LEGACY, "{}")
    current = _folder(appdata, CURRENT, json.dumps({"last_preset": "Kept"}))
    _marker(appdata, "pending")
    monkeypatch.setattr(
        shared_module.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("disk is busy"))
    )

    loaded = app_config_module.load_config()

    assert loaded.last_preset == "Kept"
    assert "disk is busy" in (loaded.load_warning or "")
    assert app_config_module.save_config(loaded) is False
    assert json.loads((current / "config.json").read_text(encoding="utf-8")) == {
        "last_preset": "Kept"
    }


# A marker left behind must never block saves for good (PC-002)


@pytest.mark.parametrize("state", ["pending", "staged"])
def test_marker_without_a_legacy_folder_is_cleared_and_saves_work(appdata, state):
    current = _folder(appdata, CURRENT, "{}")
    _marker(appdata, state)

    assert shared_module._config_dir() == current

    assert not _marker(appdata).exists()
    assert not shared_module.migration_pending()
    assert app_config_module.save_config(AppConfig(last_preset="Saved")) is True
    assert json.loads((current / "config.json").read_text(encoding="utf-8"))["last_preset"] == "Saved"


@pytest.mark.parametrize("state", ["", "written by an older build"])
def test_unrecognised_marker_is_retried_like_a_pending_one(appdata, state):
    _folder(appdata, LEGACY, "legacy")
    current = _folder(appdata, CURRENT, "other")
    _marker(appdata, state)

    shared_module._config_dir()

    assert (current / "config.json").read_text(encoding="utf-8") == "legacy"
    assert (_partials(appdata)[0] / "config.json").read_text(encoding="utf-8") == "other"
    assert not _marker(appdata).exists()


def test_stranded_previous_destination_is_restored_when_the_legacy_folder_is_gone(
    appdata, caplog
):
    stranded = _folder(appdata, f".{CURRENT}.partial-abc123", "my settings")
    _marker(appdata, "pending")

    with caplog.at_level(logging.WARNING, logger=shared_module.logger.name):
        config_dir = shared_module._config_dir()

    assert (config_dir / "config.json").read_text(encoding="utf-8") == "my settings"
    assert not stranded.exists()
    assert not _marker(appdata).exists()
    assert "my settings" not in caplog.text and str(stranded) in caplog.text


def test_marker_with_nothing_to_restore_or_migrate_is_cleared(appdata):
    _marker(appdata, "pending")

    config_dir = shared_module._config_dir()

    assert config_dir.is_dir()
    assert not _marker(appdata).exists()


def test_failed_staged_write_after_publication_still_clears_the_marker(appdata, monkeypatch):
    _folder(appdata, LEGACY, "legacy")
    real_write = shared_module._write_migration_marker

    def fail_staged(marker, state):
        if state == "staged":
            raise OSError("disk full")
        real_write(marker, state)

    monkeypatch.setattr(shared_module, "_write_migration_marker", fail_staged)

    config_dir = shared_module._config_dir()

    assert (config_dir / "config.json").read_text(encoding="utf-8") == "legacy"
    assert not _marker(appdata).exists()
    assert _partials(appdata) == []
    assert os.path.isdir(config_dir)


# Starting file logging must not pre-empt the legacy migration


@pytest.fixture
def clean_root_handlers():
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


def test_configuring_logging_first_still_migrates_the_legacy_settings(
    tmp_path, monkeypatch, clean_root_handlers
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    legacy = tmp_path / LEGACY
    legacy.mkdir()
    (legacy / "config.json").write_text(json.dumps({"last_preset": "Old"}), encoding="utf-8")

    log_file = app_logging.configure_app_logging()
    logging.getLogger("probe").warning("after the swap")
    loaded = app_config_module.load_config()

    assert loaded.last_preset == "Old"
    assert log_file == tmp_path / CURRENT / "logs" / "app.log"
    assert log_file.is_file()
    assert not shared_module.migration_pending()


def test_a_failed_migration_logs_to_the_temp_folder_and_leaves_the_config_folder_alone(
    tmp_path, monkeypatch, clean_root_handlers
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    temp = tmp_path / "temp"
    monkeypatch.setattr(app_logging.tempfile, "gettempdir", lambda: str(temp))
    legacy = tmp_path / LEGACY
    legacy.mkdir()
    (legacy / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        shared_module.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("busy"))
    )

    log_file = app_logging.configure_app_logging()
    logging.getLogger("probe").warning("after the failure")

    assert log_file == temp / CURRENT / "logs" / "app.log"
    text = log_file.read_text(encoding="utf-8")
    assert "busy" in text and "after the failure" in text
    assert not (tmp_path / CURRENT).exists()
    assert shared_module.migration_pending()


# Review follow-ups: restore failure, unreadable marker, visible failure reason


def _fail_replace_of(monkeypatch, name_prefix: str):
    real_replace = Path.replace

    def fail(self, target):
        if self.name.startswith(name_prefix):
            raise OSError("simulated rename failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail)


def test_failed_restore_keeps_the_marker_and_does_not_create_an_empty_destination(
    appdata, monkeypatch
):
    stranded = _folder(appdata, f".{CURRENT}.partial-abc123", "my settings")
    _marker(appdata, "pending")
    with monkeypatch.context() as failing:
        _fail_replace_of(failing, f".{CURRENT}.partial-")
        loaded = app_config_module.load_config()

        assert _marker(appdata).exists()
        assert not (appdata / CURRENT).exists()
        assert shared_module.migration_pending()
        assert loaded.save_blocked_reason
        assert "could not restore" in (loaded.load_warning or "")
        assert app_config_module.save_config(loaded) is False
        assert (stranded / "config.json").read_text(encoding="utf-8") == "my settings"

    config_dir = shared_module._config_dir()
    assert (config_dir / "config.json").read_text(encoding="utf-8") == "my settings"
    assert not _marker(appdata).exists()


def test_unreadable_marker_never_swaps_a_live_destination_for_the_legacy_copy(
    appdata, monkeypatch
):
    _folder(appdata, LEGACY, "stale legacy")
    current = _folder(appdata, CURRENT, "live")
    _marker(appdata, "staged")
    real_read = Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == _marker(appdata):
            raise PermissionError("transient lock")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)

    assert shared_module._config_dir() == current

    assert (current / "config.json").read_text(encoding="utf-8") == "live"
    assert _partials(appdata) == []
    assert not shared_module.migration_pending()


def test_unreadable_marker_without_a_destination_stays_pending(appdata, monkeypatch):
    _marker(appdata, "pending")
    real_read = Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == _marker(appdata):
            raise PermissionError("transient lock")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    _folder(appdata, LEGACY, "{}")

    assert shared_module.migration_pending()


def test_failed_migration_reason_reaches_the_load_warning(appdata, monkeypatch):
    _folder(appdata, LEGACY, "{}")
    monkeypatch.setattr(
        shared_module.shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(OSError("disk is busy"))
    )

    loaded = app_config_module.load_config()

    assert "disk is busy" in (loaded.load_warning or "")
    assert loaded.save_blocked_reason
