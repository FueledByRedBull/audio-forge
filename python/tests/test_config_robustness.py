"""Edge cases for config/preset persistence, key parsing and the DLL launcher."""

from __future__ import annotations

import errno
import json
import logging
import os
import random
import runpy
import stat
import sys
import time
from pathlib import Path

import pytest

from mic_eq import config
from mic_eq.config_parts import app_config as app_config_module
from mic_eq.config_parts import presets as presets_module
from mic_eq.config_parts import shared as shared_module

Preset = config.Preset
AppConfig = config.AppConfig


@pytest.fixture
def presets_dirs(tmp_path, monkeypatch):
    presets_dir = tmp_path / "presets"
    imports_dir = tmp_path / "imports"
    monkeypatch.setattr(presets_module, "get_presets_dir", lambda: presets_dir)
    monkeypatch.setattr(presets_module, "get_preset_imports_dir", lambda: imports_dir)
    return presets_dir, imports_dir


def _fail_replace_times(monkeypatch, failures: int):
    real_replace = os.replace
    calls = []

    def flaky(source, destination):
        calls.append(destination)
        if len(calls) <= failures:
            raise PermissionError(errno.EACCES, "held open by a scanner")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return calls


# PC-003


def test_unchanged_invalid_config_keeps_one_recovery_copy(tmp_path, monkeypatch):
    filepath = tmp_path / "config.json"
    filepath.write_text('{"broken":', encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    for _ in range(3):
        assert app_config_module.load_config().load_warning

    assert sorted(path.name for path in tmp_path.glob("config.json.corrupt*")) == [
        "config.json.corrupt"
    ]


def test_unchanged_newer_schema_config_keeps_one_recovery_copy(tmp_path, monkeypatch):
    filepath = tmp_path / "config.json"
    filepath.write_text(
        json.dumps({"schema_version": config.CONFIG_SCHEMA_VERSION + 1}), encoding="utf-8"
    )
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)

    for _ in range(3):
        assert app_config_module.load_config().save_blocked_reason

    assert len(list(tmp_path.glob("config.json.corrupt*"))) == 1


def test_changed_invalid_config_still_gets_its_own_recovery_copy(tmp_path, monkeypatch):
    filepath = tmp_path / "config.json"
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)
    filepath.write_text('{"broken":', encoding="utf-8")
    app_config_module.load_config()
    filepath.write_text('{"other":', encoding="utf-8")
    app_config_module.load_config()

    assert filepath.with_name("config.json.corrupt.1").read_text(encoding="utf-8") == (
        '{"other":'
    )


# PC-005


def test_save_config_retries_a_transient_permission_error(tmp_path, monkeypatch):
    filepath = tmp_path / "config.json"
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)
    calls = _fail_replace_times(monkeypatch, 2)

    assert app_config_module.save_config(AppConfig(last_preset="Keep")) is True

    assert len(calls) == 3
    assert json.loads(filepath.read_text(encoding="utf-8"))["last_preset"] == "Keep"
    assert list(tmp_path.glob("*.tmp")) == []


def test_replace_file_does_not_wait_on_a_read_only_target(tmp_path, monkeypatch):
    destination = tmp_path / "config.json"
    destination.write_text("existing", encoding="utf-8")
    destination.chmod(stat.S_IREAD)
    source = tmp_path / "config.json.tmp"
    source.write_text("new", encoding="utf-8")
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    try:
        with pytest.raises(PermissionError):
            shared_module.replace_file(source, destination)
    finally:
        destination.chmod(stat.S_IWRITE | stat.S_IREAD)

    assert sleeps == []
    assert destination.read_text(encoding="utf-8") == "existing"


def test_replace_file_still_retries_a_writable_target_held_open(tmp_path, monkeypatch):
    destination = tmp_path / "config.json"
    destination.write_text("existing", encoding="utf-8")
    source = tmp_path / "config.json.tmp"
    source.write_text("new", encoding="utf-8")
    calls = _fail_replace_times(monkeypatch, 2)

    shared_module.replace_file(source, destination)

    assert len(calls) == 3
    assert destination.read_text(encoding="utf-8") == "new"


def test_save_config_gives_up_after_repeated_permission_errors(tmp_path, monkeypatch):
    filepath = tmp_path / "config.json"
    filepath.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(app_config_module, "get_config_file", lambda: filepath)
    calls = _fail_replace_times(monkeypatch, 100)

    with pytest.raises(PermissionError):
        app_config_module.save_config(AppConfig())

    assert 1 < len(calls) < 100
    assert filepath.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.glob("*.tmp")) == []


def test_overwriting_a_preset_retries_a_transient_permission_error(tmp_path, monkeypatch):
    destination = tmp_path / "voice.json"
    destination.write_text("old", encoding="utf-8")
    calls = _fail_replace_times(monkeypatch, 2)

    config.save_preset(Preset(name="Voice"), destination, overwrite=True)

    assert len(calls) == 3
    assert Preset.from_dict(json.loads(destination.read_text("utf-8"))).name == "Voice"


# PC-006


def _no_hard_links(monkeypatch):
    def refuse(_source, _destination):
        raise OSError(errno.EPERM, "hard links are not supported")

    monkeypatch.setattr(os, "link", refuse)


def test_publishing_never_replaces_an_existing_file(tmp_path):
    destination = tmp_path / "taken.json"
    destination.write_text("occupied", encoding="utf-8")
    staged = tmp_path / "staged.tmp"
    staged.write_text("new", encoding="utf-8")

    with pytest.raises(FileExistsError):
        presets_module._publish_exclusive(staged, destination)

    assert destination.read_text(encoding="utf-8") == "occupied"
    assert staged.read_text(encoding="utf-8") == "new"


def test_save_preset_works_without_hard_links(presets_dirs, monkeypatch):
    _no_hard_links(monkeypatch)

    path = config.save_preset(Preset(name="Voice"))

    assert config.load_preset(path).name == "Voice"
    assert [p.name for p in path.parent.iterdir()] == ["Voice.json"]


def test_save_preset_without_hard_links_still_refuses_to_overwrite(presets_dirs, monkeypatch):
    _no_hard_links(monkeypatch)
    path = config.save_preset(Preset(name="Voice", description="first"))

    with pytest.raises(FileExistsError):
        config.save_preset(Preset(name="Voice", description="second"))

    assert config.load_preset(path).description == "first"
    assert [p.name for p in path.parent.iterdir()] == ["Voice.json"]


def test_import_preset_works_without_hard_links(presets_dirs, tmp_path, monkeypatch):
    _no_hard_links(monkeypatch)
    source = tmp_path / "outside.json"
    source.write_text(json.dumps(Preset(name="Imported").to_dict()), encoding="utf-8")

    _, imported = config.import_preset(source)
    other = tmp_path / "other"
    other.mkdir()
    changed = other / "outside.json"
    changed.write_text(
        json.dumps(Preset(name="Imported", description="changed").to_dict()),
        encoding="utf-8",
    )
    _, second = config.import_preset(changed)

    assert imported.name == "outside.json"
    assert second.name == "outside (1).json"
    assert sorted(p.name for p in imported.parent.iterdir()) == [
        "outside (1).json",
        "outside.json",
    ]


# PC-007, TS-007


def test_a_colliding_file_name_gets_a_suffix_instead_of_replacing_another_preset(presets_dirs):
    first = config.save_preset(Preset(name="A/B", description="slash"))
    second = config.save_preset(Preset(name="A:B", description="colon"))
    third = config.save_preset(Preset(name="A_B", description="underscore"))

    assert len({first, second, third}) == 3
    assert config.load_preset(first).description == "slash"
    assert config.load_preset(second).description == "colon"
    assert config.load_preset(third).description == "underscore"


def test_saving_the_same_preset_name_again_still_asks_for_overwrite(presets_dirs):
    first = config.save_preset(Preset(name="A/B", description="one"))

    with pytest.raises(FileExistsError):
        config.save_preset(Preset(name="A/B", description="two"))

    assert config.save_preset(Preset(name="A/B", description="two"), overwrite=True) == first
    assert config.load_preset(first).description == "two"


def test_an_unreadable_file_with_the_target_name_is_not_replaced(presets_dirs):
    presets_dir, _ = presets_dirs
    presets_dir.mkdir()
    (presets_dir / "Voice.json").write_text("not json", encoding="utf-8")

    path = config.save_preset(Preset(name="Voice"))

    assert path.name == "Voice_2.json"
    assert (presets_dir / "Voice.json").read_text(encoding="utf-8") == "not json"


@pytest.mark.parametrize(
    "name",
    [
        "CON",
        "nul",
        "Aux",
        "COM1",
        "lpt9",
        "***",
        "é" * 300,
        "x" * 300,
        "Ünïcode ✓ 声",
    ],
    ids=["CON", "nul", "Aux", "COM1", "lpt9", "symbols", "long-unicode", "long-ascii", "unicode"],
)
def test_awkward_preset_names_save_and_load(presets_dirs, name):
    path = config.save_preset(Preset(name=name))

    assert path.stem.upper() not in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}
    assert len(path.name) <= 110
    assert config.load_preset(path).name == name


# PC-012


@pytest.mark.parametrize(
    "geometry",
    [
        {"x": 3_000_000_000, "y": 0, "width": 800, "height": 600},
        {"x": 0, "y": -3_000_000_000, "width": 800, "height": 600},
        {"x": 0, "y": 0, "width": 1e30, "height": 600},
        {"x": 0, "y": 0, "width": 800, "height": 2**31},
    ],
)
def test_app_config_ignores_window_geometry_beyond_qt_int_range(geometry):
    assert AppConfig.from_dict({"window_geometry": geometry}).window_geometry is None


def test_app_config_keeps_multi_monitor_window_geometry():
    geometry = {"x": -3840, "y": 2160, "width": 1280, "height": 720}

    assert AppConfig.from_dict({"window_geometry": geometry}).window_geometry == geometry


# PC-013


def test_deeply_nested_route_keys_are_skipped_not_fatal():
    deep = "[" * 200_000
    profile = {
        "output_latency_ms": 20.0,
        "input_buffer_frames": 480,
        "output_buffer_frames": 480,
    }

    assert shared_module.parse_latency_profile_key(deep) is None
    restored = AppConfig.from_dict(
        {
            "last_preset": "Keep",
            "latency_calibration_profiles": {deep: profile},
            "input_device_preferences": {deep: {"channel_mode": "left"}},
            "route_input_preferences": {deep: {"channel_mode": "left"}},
        }
    )
    assert restored.last_preset == "Keep"
    assert restored.latency_calibration_profiles == {}


# PC-014


def test_retried_migration_reports_where_the_replaced_destination_went(
    tmp_path, monkeypatch, caplog
):
    base_dir = tmp_path / "appdata"
    legacy_dir = base_dir / shared_module.LEGACY_APPDATA_DIR_NAME
    config_dir = base_dir / shared_module.APPDATA_DIR_NAME
    legacy_dir.mkdir(parents=True)
    config_dir.mkdir()
    (legacy_dir / "config.json").write_text("legacy", encoding="utf-8")
    (config_dir / "config.json").write_text("first attempt", encoding="utf-8")
    shared_module._migration_marker(base_dir).write_text("pending", encoding="ascii")
    monkeypatch.setattr(shared_module, "_config_base_dir", lambda: base_dir)

    with caplog.at_level(logging.WARNING, logger=shared_module.logger.name):
        assert shared_module._config_dir() == config_dir

    (kept,) = base_dir.glob(f".{shared_module.APPDATA_DIR_NAME}.partial-*")
    assert (config_dir / "config.json").read_text(encoding="utf-8") == "legacy"
    assert (kept / "config.json").read_text(encoding="utf-8") == "first attempt"
    assert str(kept) in caplog.text


# PC-011


def test_frozen_launcher_registers_only_the_bundle_content_directory(tmp_path, monkeypatch):
    exe_dir = tmp_path / "AudioForge"
    meipass = exe_dir / "_internal"
    meipass.mkdir(parents=True)
    registered: list[str] = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "AudioForge.exe"))
    monkeypatch.setattr(os, "add_dll_directory", registered.append, raising=False)

    runpy.run_path(str(Path(__file__).resolve().parents[2] / "launcher.py"))

    assert registered == [str(meipass)]


# TS-011


def _mutate_text(rng: random.Random, text: str) -> str:
    kind = rng.randrange(4)
    at = rng.randrange(len(text))
    if kind == 0:
        return text[:at]
    if kind == 1:
        return text[:at] + rng.choice('{}[]",:-0') + text[at + 1 :]
    if kind == 2:
        swap = rng.choice(["null", "true", "[]", "{}", "1e999", "-1", "99999999999999999999", '"x"'])
        end = min(len(text), at + rng.randint(1, 8))
        return text[:at] + swap + text[end:]
    return text[:at] + "[" * rng.randint(1, 3000) + text[at:]


def test_preset_loader_only_raises_validation_errors_on_seeded_corruption(tmp_path):
    rng = random.Random(1181)
    valid = json.dumps(Preset(name="Fuzz").to_dict(), indent=1)
    path = tmp_path / "fuzz.json"
    accepted = 0
    for _ in range(600):
        path.write_text(_mutate_text(rng, valid), encoding="utf-8")
        try:
            presets_module._load_resolved_preset(path)
        except config.PresetValidationError:
            continue
        accepted += 1
    assert 0 < accepted < 600


def test_config_loader_never_raises_on_seeded_corruption(tmp_path, monkeypatch):
    rng = random.Random(1181)
    valid = json.dumps(AppConfig().to_dict(), indent=1)
    for index in range(300):
        filepath = tmp_path / str(index) / "config.json"
        filepath.parent.mkdir()
        filepath.write_text(_mutate_text(rng, valid), encoding="utf-8")
        monkeypatch.setattr(app_config_module, "get_config_file", lambda path=filepath: path)

        assert isinstance(app_config_module.load_config(), AppConfig)
