from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from PySide6.QtWidgets import QMessageBox

from mic_eq import __version__, config
from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow


def test_runtime_preset_catalog_versions_follow_package_version() -> None:
    assert config.CURRENT_VERSION == __version__
    assert config.Preset().version == __version__
    assert {
        preset.version for preset in config.BUILTIN_PRESETS.values()
    } == {__version__}


def test_saved_custom_preset_uses_package_version(monkeypatch, tmp_path: Path) -> None:
    window: Any = MainWindow.__new__(MainWindow)
    preset = config.Preset(version="0.0.0")
    window.configuration = SimpleNamespace(current=preset, flush_writes=Mock())
    window._configuration_host = SimpleNamespace(mute_held=lambda _reason: False)
    window._save_preset_file = Mock(return_value=tmp_path / "saved.json")
    monkeypatch.setattr(
        main_window.QMessageBox,
        "question",
        Mock(return_value=QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(main_window.QMessageBox, "information", Mock())

    MainWindow._prompt_save_current_preset(
        window,
        title="Save",
        question="Save?",
        preset_name="Saved",
        description="",
    )

    assert preset.version == __version__
    window._save_preset_file.assert_called_once_with(preset)
