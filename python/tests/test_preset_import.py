"""Safe external preset import behavior."""

import json
from pathlib import Path

import pytest

from mic_eq import config
from mic_eq.config_parts import presets as presets_module


def _preset_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    presets_dir = tmp_path / "presets"
    imports_dir = tmp_path / "imports"
    presets_dir.mkdir()
    imports_dir.mkdir()
    monkeypatch.setattr(presets_module, "get_presets_dir", lambda: presets_dir)
    monkeypatch.setattr(
        presets_module,
        "get_preset_imports_dir",
        lambda: imports_dir,
    )
    monkeypatch.setattr(presets_module, "migration_pending", lambda: False)
    return presets_dir, imports_dir


def _write_preset(path: Path, name: str) -> None:
    path.write_text(
        json.dumps(config.Preset(name=name).to_dict(), indent=2),
        encoding="utf-8",
    )


def test_invalid_external_import_keeps_existing_collision_untouched(
    tmp_path, monkeypatch
):
    _presets_dir, imports_dir = _preset_dirs(tmp_path, monkeypatch)
    source = tmp_path / "shared.json"
    destination = imports_dir / source.name
    _write_preset(destination, "Existing")
    original = destination.read_bytes()
    source.write_text("{ invalid", encoding="utf-8")

    with pytest.raises(config.PresetValidationError):
        config.import_preset(source)

    assert destination.read_bytes() == original
    assert not list(imports_dir.glob("*.tmp"))


def test_import_reuses_identical_copy_and_chooses_unique_name_for_collision(
    tmp_path, monkeypatch
):
    _presets_dir, imports_dir = _preset_dirs(tmp_path, monkeypatch)
    source = tmp_path / "shared.json"
    _write_preset(source, "Imported")

    identical = imports_dir / source.name
    identical.write_bytes(source.read_bytes())
    preset, imported_path = config.import_preset(source)
    assert preset.name == "Imported"
    assert imported_path == identical
    assert not (imports_dir / "shared (1).json").exists()

    different = tmp_path / "shared.json"
    _write_preset(different, "Replacement")
    preset, imported_path = config.import_preset(different)
    assert preset.name == "Replacement"
    assert imported_path == imports_dir / "shared (1).json"
    assert identical.read_bytes() != imported_path.read_bytes()


def test_import_cleans_staging_file_when_atomic_publication_fails(
    tmp_path, monkeypatch
):
    _presets_dir, imports_dir = _preset_dirs(tmp_path, monkeypatch)
    source = tmp_path / "failed.json"
    _write_preset(source, "Failed")

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(presets_module.os, "link", fail_link)

    with pytest.raises(OSError, match="simulated publication failure"):
        config.import_preset(source)

    assert not (imports_dir / source.name).exists()
    assert not any(path.suffix == ".tmp" for path in imports_dir.iterdir())
