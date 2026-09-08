from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QMainWindow, QMenu, QMessageBox

from mic_eq.config import AppConfig, DevicePresetBinding, Preset
from mic_eq.ui import main_window
from mic_eq.ui.config_history import BoundedConfigurationHistory, ConfigurationSnapshot
from mic_eq.ui.main_window import MainWindow, _startup_custom_id


def test_saved_processing_settings_restore_on_next_launch(qapp, monkeypatch, tmp_path):
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        window.eq_panel.band_sliders[4].slider.setValue(23)
        window.gate_panel.threshold_spinbox.setValue(-33.0)
        preset = window._get_current_preset()
        preset.name = "Saved Calibration"
        filepath = window._save_preset_file(preset)
        assert filepath is not None and filepath.is_file()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    restored = MainWindow()
    try:
        assert restored.current_preset_path == filepath
        assert restored.config.last_preset == str(filepath)
        assert restored.eq_panel.band_sliders[4].slider.value() == 23
        assert restored.gate_panel.threshold_spinbox.value() == -33.0
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


def _save_window(tmp_path: Path) -> MainWindow:
    window = MainWindow.__new__(MainWindow)
    previous_path = tmp_path / "previous.json"
    window.config = AppConfig(last_preset=str(previous_path))
    window.current_preset_path = previous_path
    return window


def test_save_preset_updates_last_used_identity(monkeypatch, tmp_path):
    window = _save_window(tmp_path)
    saved_path = tmp_path / "fresh.json"
    save_config = Mock(return_value=True)
    monkeypatch.setattr(main_window, "save_preset", Mock(return_value=saved_path))
    monkeypatch.setattr(main_window, "save_config", save_config)
    monkeypatch.setattr(
        main_window.QInputDialog,
        "getText",
        Mock(side_effect=[("Fresh", True), ("Description", True)]),
    )
    monkeypatch.setattr(main_window.QMessageBox, "information", Mock())
    window.status_bar = Mock()
    window._get_current_preset = lambda: Preset()

    MainWindow._save_preset(window)

    assert window.current_preset_path == saved_path
    assert window.config.last_preset == str(saved_path)
    save_config.assert_called_once_with(window.config)


def test_prompt_save_cancel_keeps_last_used_identity(monkeypatch, tmp_path):
    window = _save_window(tmp_path)
    save_preset = Mock()
    monkeypatch.setattr(main_window, "save_preset", save_preset)
    monkeypatch.setattr(
        main_window.QMessageBox,
        "question",
        Mock(return_value=QMessageBox.StandardButton.No),
    )

    MainWindow._prompt_save_current_preset(
        window,
        title="Save",
        question="Save?",
        preset_name="Cancelled",
        description="",
    )

    assert window.current_preset_path == tmp_path / "previous.json"
    assert window.config.last_preset == str(tmp_path / "previous.json")
    save_preset.assert_not_called()


def test_failed_save_keeps_last_used_identity(monkeypatch, tmp_path):
    window = _save_window(tmp_path)
    monkeypatch.setattr(
        main_window.QMessageBox,
        "question",
        Mock(return_value=QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(main_window, "save_preset", Mock(side_effect=OSError("disk full")))
    critical = Mock()
    monkeypatch.setattr(main_window.QMessageBox, "critical", critical)
    window._get_current_preset = lambda: Preset()

    MainWindow._prompt_save_current_preset(
        window,
        title="Save",
        question="Save?",
        preset_name="Failed",
        description="",
    )

    assert window.current_preset_path == tmp_path / "previous.json"
    assert window.config.last_preset == str(tmp_path / "previous.json")
    critical.assert_called_once()


@pytest.mark.parametrize("save_config_result", [False, OSError("config locked")])
def test_saved_file_without_identity_persistence_is_reported(
    monkeypatch, tmp_path, save_config_result
):
    window = _save_window(tmp_path)
    saved_path = tmp_path / "fresh.json"
    monkeypatch.setattr(main_window, "save_preset", Mock(return_value=saved_path))
    if isinstance(save_config_result, Exception):
        save_config = Mock(side_effect=save_config_result)
    else:
        save_config = Mock(return_value=save_config_result)
    monkeypatch.setattr(main_window, "save_config", save_config)
    monkeypatch.setattr(main_window.QMessageBox, "information", Mock())
    monkeypatch.setattr(
        main_window.QInputDialog,
        "getText",
        Mock(side_effect=[("Fresh", True), ("", True)]),
    )
    window.status_bar = Mock()
    window._get_current_preset = lambda: Preset()

    MainWindow._save_preset(window)

    assert window.current_preset_path == tmp_path / "previous.json"
    assert window.config.last_preset == str(tmp_path / "previous.json")
    assert window._last_preset_identity_persisted is False
    assert "could not remember" in main_window.QMessageBox.information.call_args.args[2]


def _submenu(window: QMainWindow, title: str) -> QMenu:
    menubar = window.menuBar()
    assert menubar is not None
    for action in menubar.actions():
        menu = action.menu()
        if menu is None:
            continue
        for child in menu.actions():
            submenu = child.menu()
            if submenu is not None and submenu.title() == title:
                return submenu
    raise AssertionError(f"missing submenu {title!r}")


def test_preset_menus_refresh_new_files_and_keep_selection(qapp, monkeypatch, tmp_path):
    presets: list[tuple[str, Path]] = []
    monkeypatch.setattr(main_window, "list_presets", lambda: list(presets))
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    window.config = AppConfig(startup_preset="")
    window._current_device_route_key = lambda: "route"
    window.config.device_preset_bindings["route"] = DevicePresetBinding(
        preset_id=_startup_custom_id("fresh.json"),
        provenance="explicit_user",
    )
    MainWindow._setup_options_menu(window)

    fresh_path = tmp_path / "fresh.json"
    presets.append(("Fresh", fresh_path))
    startup_menu = _submenu(window, "Startup &Preset...")
    route_menu = _submenu(window, "Preset for Current &Route")
    startup_menu.aboutToShow.emit()
    route_menu.aboutToShow.emit()

    startup_actions = [action for action in startup_menu.actions() if action.isCheckable()]
    route_actions = [action for action in route_menu.actions() if action.text() == "Fresh"]
    assert any(action.text() == "Fresh" for action in startup_actions)
    assert next(action for action in startup_actions if action.text() == "Last Used").isChecked()
    assert len(route_actions) == 1
    assert route_actions[0].isChecked()

    window.hide()
    window.deleteLater()
    qapp.processEvents()


def test_manual_history_edit_clears_auto_eq_diagnostics_only_when_recorded():
    window = cast(Any, MainWindow.__new__(MainWindow))
    history = BoundedConfigurationHistory(limit=4)
    baseline = ConfigurationSnapshot.from_preset(
        Preset(), label="baseline", source="startup"
    )
    history.initialize(baseline)
    window._configuration_history = history
    window._history_ready = True
    window._history_replaying = False
    window._history_timer = Mock()
    window._current_value_provenance = {}

    edited = Preset()
    edited.gate.threshold_db = -35.0
    window._get_current_preset = lambda: edited
    window.compressor_panel = SimpleNamespace(
        get_compressor_settings=lambda include_calibration: {
            "noise_reference_reliability": 0.0
        }
    )
    window._calibration_context_key = lambda: None
    window.eq_panel = SimpleNamespace(set_auto_eq_diagnostics=Mock())
    window._update_history_actions = Mock()
    window.status_bar = Mock()

    assert MainWindow._commit_pending_configuration_snapshot(window, source="ui")
    window.eq_panel.set_auto_eq_diagnostics.assert_called_once_with(None)
