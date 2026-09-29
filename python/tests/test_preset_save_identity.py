from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QMainWindow, QMenu, QMessageBox
from PyQt6.QtGui import QCloseEvent

from mic_eq.config import AppConfig, DevicePresetBinding, Preset
from mic_eq.ui import main_window
from mic_eq.ui.config_history import BoundedConfigurationHistory, ConfigurationSnapshot
from mic_eq.ui.main_window import MainWindow


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
        assert restored._processing_mode() == "normal"
        assert not restored.preset_modified
        assert restored.eq_panel.band_sliders[4].slider.value() == 23
        assert restored.gate_panel.threshold_spinbox.value() == -33.0
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


def test_raw_mode_save_is_session_only_and_close_save_reloads_normal(
    qapp, monkeypatch, tmp_path
):
    from mic_eq.config import get_presets_dir, load_preset
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    restored = None
    try:
        filepath = get_presets_dir() / "Raw Session.json"
        preset = window._get_current_preset()
        preset.name = "Raw Session"
        assert window._save_preset_file(preset, filepath=filepath) == filepath
        assert not window.preset_modified

        raw_index = window.processing_mode_combo.findData("raw")
        window.processing_mode_combo.setCurrentIndex(raw_index)
        assert window._processing_mode() == "raw"
        assert window.preset_modified
        assert "not stored in presets" in window.processing_mode_combo.toolTip().lower()

        saved = load_preset(filepath)
        assert saved.bypass is False
        assert "processing_mode" not in saved.to_dict()

        window._save_preset()
        assert window._processing_mode() == "raw"
        assert window.preset_modified
        assert "session-only" in window.status_bar.currentMessage().lower()

        normal_index = window.processing_mode_combo.findData("normal")
        window.processing_mode_combo.setCurrentIndex(normal_index)
        window._commit_pending_configuration_snapshot()
        assert window._processing_mode() == "normal"
        assert not window.preset_modified

        window.processing_mode_combo.setCurrentIndex(raw_index)
        assert window.preset_modified
        close_prompt = Mock(return_value=QMessageBox.StandardButton.Save)
        monkeypatch.setattr(
            main_window.QMessageBox,
            "question",
            close_prompt,
        )
        assert window.close()
        assert window.preset_modified
        assert "session-only" in close_prompt.call_args.args[2]

        restored = MainWindow()
        assert restored.current_preset_path == filepath
        assert restored._processing_mode() == "normal"
        assert not restored.preset_modified
    finally:
        window.hide()
        window.deleteLater()
        if restored is not None:
            restored.close()
            restored.deleteLater()
        qapp.processEvents()


def test_invalid_last_used_preset_falls_back_with_persistent_warning(
    qapp, monkeypatch, tmp_path
):
    from mic_eq.config import get_presets_dir
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    broken = get_presets_dir() / "broken.json"
    broken.write_text("{invalid json", encoding="utf-8")
    main_window.save_config(AppConfig(last_preset=str(broken)))

    window = MainWindow()
    try:
        assert window.current_preset_path is None
        assert window.config.last_preset == ""
        assert not window.config_warning_banner.isHidden()
        assert "could not restore" in window.config_warning_banner.text().lower()
        assert "could not restore" in window.status_bar.currentMessage().lower()
    finally:
        window.close()
        window.deleteLater()
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


def test_cancel_save_as_does_not_save_or_change_identity(monkeypatch, tmp_path):
    window = _save_window(tmp_path)
    save = Mock()
    window._save_preset_file = save
    window._get_current_preset = lambda: Preset()
    monkeypatch.setattr(
        main_window.QInputDialog, "getText",
        Mock(return_value=("Cancelled", False)),
    )

    MainWindow._save_preset(window)

    save.assert_not_called()
    assert window.current_preset_path == tmp_path / "previous.json"
    assert window.config.last_preset == str(tmp_path / "previous.json")


@pytest.mark.parametrize("valid", [False, True])
def test_import_collision_preserves_previous_preset(qapp, monkeypatch, tmp_path, valid):
    from mic_eq.config import get_preset_imports_dir, load_preset, save_preset
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path / "config")
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])
    monkeypatch.setattr(main_window.QMessageBox, "warning", Mock())
    existing = get_preset_imports_dir() / "voice.json"
    save_preset(Preset(name="Existing voice"), existing)
    original = existing.read_bytes()
    external = tmp_path / "voice.json"
    if valid:
        save_preset(Preset(name="New voice"), external)
    else:
        external.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", lambda *_: (str(external), ""))
    window = MainWindow()
    try:
        window._apply_preset(load_preset(existing), preset_path=existing)
        window._load_preset()
        assert existing.read_bytes() == original
        if valid:
            assert window.current_preset_name == "New voice"
            assert window.current_preset_path != existing
            assert window.current_preset_path is not None
            assert load_preset(window.current_preset_path).name == "New voice"
        else:
            assert window.current_preset_name == "Existing voice"
            assert window.current_preset_path == existing
            main_window.QMessageBox.warning.assert_called_once()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("strength", [0.735, 0.333, 0.29, 0.58, 0.0, 1.0])
def test_preset_strength_round_trip_matches_live_processor(qapp, monkeypatch, tmp_path, strength):
    from mic_eq.config import load_preset
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])
    window = MainWindow()
    try:
        preset = window._get_current_preset()
        preset.rnnoise.strength = strength
        window._apply_preset(preset)
        assert window.processor.get_rnnoise_strength() == pytest.approx(strength)
        assert window.strength_label.text() == f"{window.strength_slider.value()}%"
        saved = window._get_current_preset()
        assert saved.rnnoise.strength == pytest.approx(strength)
        saved.name = "Strength round trip"
        path = window._save_preset_file(saved)
        assert path is not None
        restored = load_preset(path)
        assert restored.rnnoise.strength == pytest.approx(strength)
        window._apply_preset(restored)
        assert window.processor.get_rnnoise_strength() == pytest.approx(strength)
        assert window.current_preset_modified is False

        edited_value = (window.strength_slider.value() + 1) % 101
        window.strength_slider.setValue(edited_value)
        assert window.processor.get_rnnoise_strength() == pytest.approx(
            edited_value / 100.0
        )
        assert window._get_current_preset().rnnoise.strength == pytest.approx(
            edited_value / 100.0
        )
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_non_grid_dynamics_values_survive_restore_and_snapshot(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])
    window = MainWindow()
    try:
        preset = window._get_current_preset()
        preset.gate.threshold_db = -31.237
        preset.deesser.auto_amount = 0.33789
        preset.compressor.ratio = 3.14159
        preset.limiter.ceiling_db = -1.23456

        window.apply_processing_configuration(preset)
        restored = window._get_current_preset()

        assert restored.gate.threshold_db == pytest.approx(-31.237)
        assert restored.deesser.auto_amount == pytest.approx(0.33789)
        assert restored.compressor.ratio == pytest.approx(3.14159)
        assert restored.limiter.ceiling_db == pytest.approx(-1.23456)
        assert window.processor.get_deesser_auto_amount() == pytest.approx(0.33789)
        assert window._commit_pending_configuration_snapshot()
        assert window._configuration_history.current is not None
        assert window._configuration_history.current.to_preset().compressor.ratio == pytest.approx(
            3.14159
        )
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("write_result", [
    False, OSError("disk full"), TypeError("invalid value"), ValueError("non-finite value"),
])
def test_close_stops_processing_before_reporting_settings_failure(
    qapp, monkeypatch, tmp_path, write_result,
):
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])
    window = MainWindow()
    events = []
    original_processor = window.processor
    try:
        with monkeypatch.context() as patch:
            patch.setattr(main_window, "save_config", Mock(
                side_effect=write_result if isinstance(write_result, Exception) else None,
                return_value=write_result,
            ))
            patch.setattr(main_window.QMessageBox, "warning", lambda *_: events.append("warning"))
            cast(Any, window).processor = SimpleNamespace(
                is_running=lambda: True, stop=lambda: events.append("stop"),
            )
            assert window._save_ui_state() is False
            event = QCloseEvent()
            window.closeEvent(event)
            assert events == ["stop", "warning"]
            assert event.isAccepted()
    finally:
        window.processor = original_processor
        window.close()
        window.deleteLater()
        qapp.processEvents()


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


@pytest.mark.parametrize(
    "save_config_result",
    [False, OSError("config locked"), TypeError("invalid config")],
)
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

    assert window.current_preset_path == saved_path
    assert window.config.last_preset == str(tmp_path / "previous.json")
    assert window._last_preset_identity_persisted is False
    assert "could not remember" in window.status_bar.showMessage.call_args.args[0]


@pytest.mark.parametrize("save_config_result", [False, TypeError("invalid config")])
def test_loaded_preset_keeps_live_identity_when_remembering_fails(
    monkeypatch, tmp_path, save_config_result
):
    window = MainWindow.__new__(MainWindow)
    window.config = AppConfig(last_preset="previous.json")
    window.current_preset_name = "Previous"
    window.current_preset_description = ""
    window.current_preset_path = Path("previous.json")
    window._saved_preset_payload = None
    window._current_value_provenance = {}
    window.preset_modified = False
    window._history_ready = False
    window._history_replaying = False
    window.status_bar = Mock()
    window.apply_processing_configuration = Mock()
    window._set_preset_modified = Mock()
    window._update_session_summary = Mock()
    if isinstance(save_config_result, Exception):
        monkeypatch.setattr(main_window, "save_config", Mock(side_effect=save_config_result))
    else:
        monkeypatch.setattr(main_window, "save_config", Mock(return_value=save_config_result))

    loaded_path = tmp_path / "loaded.json"
    loaded = Preset(name="Loaded")
    assert MainWindow._apply_preset(window, loaded, preset_path=loaded_path)

    assert window.current_preset_name == "Loaded"
    assert window.current_preset_path == loaded_path
    assert window.config.last_preset == "previous.json"
    assert "could not be remembered" in window.status_bar.showMessage.call_args.args[0]


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
    selected_path = tmp_path / "selected.json"
    window.config = AppConfig(startup_preset=f"custom-file:{selected_path.name}")
    window._current_device_route_key = lambda: "route"
    window.config.device_preset_bindings["route"] = DevicePresetBinding(
        preset_id="custom-file:fresh.json",
        provenance="explicit_user",
    )
    MainWindow._setup_options_menu(window)

    fresh_path = tmp_path / "fresh.json"
    presets.extend([("Fresh", fresh_path), ("Fresh", selected_path)])
    startup_menu = _submenu(window, "Startup &Preset...")
    route_menu = _submenu(window, "Preset for Current &Route")
    startup_menu.aboutToShow.emit()
    route_menu.aboutToShow.emit()

    startup_actions = [action for action in startup_menu.actions() if action.isCheckable()]
    route_actions = [action for action in route_menu.actions() if action.text() == "Fresh"]
    fresh_actions = [action for action in startup_actions if action.text() == "Fresh"]
    assert [action.data() for action in fresh_actions] == [
        "custom-file:fresh.json",
        "custom-file:selected.json",
    ]
    assert not next(
        action for action in startup_actions if action.text() == "Last Used"
    ).isChecked()
    assert fresh_actions[1].isChecked()
    assert len(route_actions) == 2
    assert route_actions[0].isChecked()
    assert not route_actions[1].isChecked()

    window.hide()
    window.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize(
    ("presets", "binding_id", "checked_filename"),
    [
        (["single.json"], "custom:Fresh", "single.json"),
        (["single.json"], "custom:single.json", "single.json"),
        (["first.json", "second.json"], "custom:Fresh", None),
    ],
)
def test_route_preset_menu_normalizes_only_unambiguous_legacy_bindings(
    qapp, monkeypatch, tmp_path, presets, binding_id, checked_filename
):
    custom_presets = [("Fresh", tmp_path / filename) for filename in presets]
    monkeypatch.setattr(main_window, "list_presets", lambda: custom_presets)
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    window.config = AppConfig()
    window.config.device_preset_bindings["route"] = DevicePresetBinding(
        preset_id=binding_id,
        provenance="explicit_user",
    )
    window._current_device_route_key = lambda: "route"
    MainWindow._setup_options_menu(window)

    route_menu = _submenu(window, "Preset for Current &Route")
    route_menu.aboutToShow.emit()
    actions = [action for action in route_menu.actions() if action.text() == "Fresh"]

    assert len(actions) == len(custom_presets)
    assert [
        action.data()
        for action in actions
        if action.isChecked()
    ] == (
        [f"custom-file:{checked_filename}"]
        if checked_filename is not None
        else []
    )
    assert not window._clear_route_preset_action.isChecked()

    window.hide()
    window.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize(
    ("startup_id", "expected_name", "expected_id"),
    [
        ("custom-file:second.json", "second.json", "custom-file:second.json"),
        ("custom:Unique", "unique.json", "custom-file:unique.json"),
        ("custom:Duplicate", None, "custom:Duplicate"),
    ],
)
def test_restore_startup_preset_resolves_file_id_and_migrates_legacy_name(
    qapp, monkeypatch, tmp_path, startup_id, expected_name, expected_id
):
    presets = [
        ("Duplicate", tmp_path / "first.json"),
        ("Duplicate", tmp_path / "second.json"),
        ("Unique", tmp_path / "unique.json"),
    ]
    loaded_paths = []
    monkeypatch.setattr(main_window, "list_presets", lambda: presets)
    monkeypatch.setattr(
        main_window,
        "load_preset",
        lambda path: loaded_paths.append(path) or Preset(),
    )

    window = cast(Any, MainWindow.__new__(MainWindow))
    window.config = AppConfig(startup_preset=startup_id)
    window.input_combo = SimpleNamespace(blockSignals=Mock())
    window.output_combo = SimpleNamespace(blockSignals=Mock())
    window.status_bar = Mock()
    window._apply_preset = Mock(return_value=True)
    window._restore_ui_state = Mock()
    window._apply_latency_compensation_for_current_devices = Mock()
    window._apply_bound_preset_for_current_route = Mock(return_value=False)
    window._save_config_safely = Mock(return_value=True)

    MainWindow._restore_from_config(window)

    assert loaded_paths == ([] if expected_name is None else [tmp_path / expected_name])
    assert window.config.startup_preset == expected_id
    assert window._save_config_safely.call_count == int(startup_id != expected_id)


def test_startup_file_id_survives_display_name_rename(monkeypatch, tmp_path):
    from mic_eq.ui.main_window import _startup_preset_selection

    filepath = tmp_path / "stable.json"
    preset_id, selected = _startup_preset_selection(
        "custom-file:stable.json", [("Renamed preset", filepath)]
    )

    assert preset_id == "custom-file:stable.json"
    assert selected == ("Renamed preset", filepath)


def test_route_preset_file_id_takes_precedence_over_legacy_name_collision(
    monkeypatch, tmp_path
):
    filename_match = tmp_path / "target.json"
    name_collision = tmp_path / "other.json"
    presets = [
        ("Target", filename_match),
        ("target.json", name_collision),
    ]
    loaded_paths = []
    monkeypatch.setattr(main_window, "list_presets", lambda: presets)
    monkeypatch.setattr(
        main_window,
        "load_preset",
        lambda path: loaded_paths.append(path) or Preset(),
    )
    window = cast(Any, MainWindow.__new__(MainWindow))
    window._apply_preset = Mock(return_value=True)

    assert MainWindow._load_device_preset_id(window, "custom-file:target.json")
    assert loaded_paths == [filename_match]


def test_route_preset_ambiguous_legacy_name_is_rejected(monkeypatch, tmp_path):
    presets = [
        ("Duplicate", tmp_path / "first.json"),
        ("Duplicate", tmp_path / "second.json"),
    ]
    monkeypatch.setattr(main_window, "list_presets", lambda: presets)
    apply_preset = Mock(return_value=True)
    window = cast(Any, MainWindow.__new__(MainWindow))
    window._apply_preset = apply_preset

    assert not MainWindow._load_device_preset_id(window, "custom:Duplicate")
    apply_preset.assert_not_called()


def test_route_preset_legacy_filename_collision_with_display_name_is_rejected(
    monkeypatch, tmp_path
):
    filename_match = tmp_path / "Shared.json"
    display_name_match = tmp_path / "other.json"
    presets = [
        ("First", filename_match),
        ("Shared.json", display_name_match),
    ]
    monkeypatch.setattr(main_window, "list_presets", lambda: presets)
    monkeypatch.setattr(main_window, "load_preset", lambda _path: Preset())
    apply_preset = Mock(return_value=True)
    window = cast(Any, MainWindow.__new__(MainWindow))
    window._apply_preset = apply_preset

    assert not MainWindow._load_device_preset_id(window, "custom:Shared.json")
    apply_preset.assert_not_called()


def test_automatic_route_apply_resolves_legacy_filename_and_shows_display_name(
    monkeypatch, tmp_path
):
    filepath = tmp_path / "custom-voice.json"
    monkeypatch.setattr(main_window, "list_presets", lambda: [("Warm Voice", filepath)])
    monkeypatch.setattr(main_window, "load_preset", lambda _path: Preset())
    apply_preset = Mock(return_value=True)
    window = cast(Any, MainWindow.__new__(MainWindow))
    window.config = AppConfig(
        auto_apply_device_presets=True,
        device_preset_bindings={
            "route": DevicePresetBinding("custom:custom-voice.json")
        },
    )
    window._current_device_route_key = lambda: "route"
    window._apply_preset = apply_preset
    window.status_bar = Mock()
    window._save_config_safely = Mock(return_value=True)

    assert MainWindow._apply_bound_preset_for_current_route(window)
    apply_preset.assert_called_once()
    assert (
        window.config.device_preset_bindings["route"].preset_id
        == "custom-file:custom-voice.json"
    )
    window._save_config_safely.assert_called_once()
    window.status_bar.showMessage.assert_called_once_with("Route preset: Warm Voice", 5000)


def test_missing_custom_file_id_does_not_match_display_name(monkeypatch, tmp_path):
    colliding_display_name = tmp_path / "other.json"
    presets = [("missing.json", colliding_display_name)]
    loaded_paths = []
    monkeypatch.setattr(main_window, "list_presets", lambda: presets)
    monkeypatch.setattr(
        main_window,
        "load_preset",
        lambda path: loaded_paths.append(path) or Preset(),
    )

    window = cast(Any, MainWindow.__new__(MainWindow))
    window.config = AppConfig(startup_preset="custom-file:missing.json")
    window.input_combo = SimpleNamespace(blockSignals=Mock())
    window.output_combo = SimpleNamespace(blockSignals=Mock())
    window.status_bar = Mock()
    window._apply_preset = Mock(return_value=True)
    window._restore_ui_state = Mock()
    window._apply_latency_compensation_for_current_devices = Mock()
    window._apply_bound_preset_for_current_route = Mock(return_value=False)
    window._save_config_safely = Mock(return_value=True)

    MainWindow._restore_from_config(window)

    assert loaded_paths == []
    assert window.config.startup_preset == "custom-file:missing.json"


def test_missing_route_file_id_does_not_match_display_name(monkeypatch, tmp_path):
    other_path = tmp_path / "other.json"
    monkeypatch.setattr(main_window, "list_presets", lambda: [("missing.json", other_path)])
    load_preset = Mock(side_effect=AssertionError("missing file identity was loaded"))
    monkeypatch.setattr(main_window, "load_preset", load_preset)
    window = cast(Any, MainWindow.__new__(MainWindow))
    window._apply_preset = Mock(return_value=True)

    assert not MainWindow._load_device_preset_id(window, "custom-file:missing.json")
    load_preset.assert_not_called()


def test_manual_history_edit_clears_auto_eq_diagnostics_only_when_recorded():
    window = cast(Any, MainWindow.__new__(MainWindow))
    window.config = AppConfig()
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


def test_save_updates_owned_path_without_name_prompt(monkeypatch, tmp_path):
    window = _save_window(tmp_path)
    window.current_preset_name = "My Sound"
    window.current_preset_description = "Keep this description"
    window._get_current_preset = Preset
    window._save_preset_file = Mock(return_value=window.current_preset_path)
    monkeypatch.setattr(main_window, "get_presets_dir", lambda: tmp_path)
    prompt = Mock(side_effect=AssertionError("Save must not ask for a new name"))
    monkeypatch.setattr(main_window.QInputDialog, "getText", prompt)
    assert window._save_preset()
    args, kwargs = window._save_preset_file.call_args
    assert kwargs["filepath"] == tmp_path / "previous.json"
    assert args[0].name == "My Sound"
    assert args[0].description == "Keep this description"


@pytest.mark.parametrize("reply,save_ok,expected", [
    (QMessageBox.StandardButton.Cancel, True, False),
    (QMessageBox.StandardButton.Discard, False, True),
    (QMessageBox.StandardButton.Save, False, False),
    (QMessageBox.StandardButton.Save, True, True),
])
def test_unsaved_decision_preserves_failed_or_cancelled_saves(monkeypatch, reply, save_ok, expected):
    window = MainWindow.__new__(MainWindow)
    window._history_ready = True
    window.preset_modified = True
    window._set_preset_modified = Mock()
    window._save_preset = Mock(return_value=save_ok)
    monkeypatch.setattr(main_window.QMessageBox, "question", lambda *_: reply)
    assert window._confirm_discard_changes() is expected
    assert window._save_preset.call_count == int(reply == QMessageBox.StandardButton.Save)
