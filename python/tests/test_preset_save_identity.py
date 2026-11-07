# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox
from PySide6.QtGui import QCloseEvent

from mic_eq.config import AppConfig, DevicePresetBinding, Preset
from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.processing_configuration import PresetOrigin
from test_eq_editing_state import eq_control
from test_gate_controls_state import window_gate_control


def test_saved_processing_settings_restore_on_next_launch(qapp, config_in_tmp_without_devices):
    window = MainWindow()
    try:
        eq_control(window, "gain", 4).setValue(23.0)
        window_gate_control(window, "Threshold").setValue(-33.0)
        preset = window.configuration.current
        preset.name = "Saved Calibration"
        filepath = window._save_preset_file(preset)
        assert filepath is not None and filepath.is_file()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    restored = MainWindow()
    try:
        identity = restored.configuration.identity
        assert (identity.name, identity.path, identity.modified) == ("Saved Calibration", filepath, False)
        assert restored.config.last_preset == str(filepath)
        assert restored._processing_mode() == "normal"
        assert eq_control(restored, "gain", 4).value == 23
        assert window_gate_control(restored, "Threshold").value == -33.0
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


def test_raw_mode_save_is_session_only_and_close_save_reloads_normal(
    qapp, config_in_tmp_without_devices, monkeypatch
):
    from mic_eq.config import get_presets_dir, load_preset

    window = MainWindow()
    restored = None
    try:
        filepath = get_presets_dir() / "Raw Session.json"
        preset = window.configuration.current
        preset.name = "Raw Session"
        assert window._save_preset_file(preset, filepath=filepath) == filepath
        assert not window.configuration.identity.modified

        raw_index = window.processing_mode_choice.findData("raw")
        window.processing_mode_choice.setCurrentIndex(raw_index)
        assert window._processing_mode() == "raw"
        assert window.configuration.identity.modified
        assert "not stored in presets" in str(window.processing_mode_choice.toolTip).lower()

        saved = load_preset(filepath)
        assert saved.bypass is False
        assert "processing_mode" not in saved.to_dict()

        window._save_preset()
        assert window._processing_mode() == "raw"
        assert window.configuration.identity.modified
        assert "session-only" in str(window.status_message.text).lower()

        normal_index = window.processing_mode_choice.findData("normal")
        window.processing_mode_choice.setCurrentIndex(normal_index)
        window.configuration.commit_edit()
        assert window._processing_mode() == "normal"
        assert not window.configuration.identity.modified

        window.processing_mode_choice.setCurrentIndex(raw_index)
        assert window.configuration.identity.modified
        close_prompt = Mock(return_value=QMessageBox.StandardButton.Save)
        monkeypatch.setattr(
            main_window.QMessageBox,
            "question",
            close_prompt,
        )
        assert window.close()
        assert window.configuration.identity.modified
        assert "session-only" in close_prompt.call_args.args[2]

        restored = MainWindow()
        assert restored.configuration.identity.path == filepath
        assert restored._processing_mode() == "normal"
        assert not restored.configuration.identity.modified
    finally:
        window.hide()
        window.deleteLater()
        if restored is not None:
            restored.close()
            restored.deleteLater()
        qapp.processEvents()


def test_invalid_last_used_preset_falls_back_with_persistent_warning(
    qapp, config_in_tmp_without_devices
):
    from mic_eq.config import get_presets_dir, load_config

    broken = get_presets_dir() / "broken.json"
    broken.write_text("{invalid json", encoding="utf-8")
    main_window.save_config(AppConfig(last_preset=str(broken)))

    window = MainWindow()
    try:
        assert window.configuration.identity.path is None
        assert window.config.last_preset == ""
        assert window.shell_texts["config_warning_banner"].shown
        assert "could not restore" in window.shell_texts["config_warning_banner"].text.lower()
        assert "could not restore" in str(window.status_message.text).lower()
        assert load_config().last_preset == ""
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.fixture
def config_in_tmp_without_devices(monkeypatch, tmp_path):
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])


@pytest.fixture
def save_window(isolated_main_window, monkeypatch, tmp_path) -> MainWindow:
    """A window showing ``previous.json``, the last-used preset; its folder is not the presets folder."""
    window = isolated_main_window
    monkeypatch.setattr(main_window, "get_presets_dir", lambda: tmp_path / "presets")
    previous = window.configuration.current
    previous.name = "Previous"
    assert window.configuration.mark_saved(previous, tmp_path / "previous.json")
    return window


def _identity(window: MainWindow, tmp_path: Path) -> tuple[Path | None, bool]:
    """The preset the window shows, and whether it still remembers ``previous.json``."""
    return window.configuration.identity.path, window.config.last_preset == str(tmp_path / "previous.json")


def test_cancel_save_as_does_not_save_or_change_identity(save_window, monkeypatch, tmp_path):
    save = Mock()
    monkeypatch.setattr(save_window, "_save_preset_file", save)
    monkeypatch.setattr(main_window.QInputDialog, "getText", Mock(return_value=("Cancelled", False)))

    save_window._save_preset()

    save.assert_not_called()
    assert _identity(save_window, tmp_path) == (tmp_path / "previous.json", True)


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
        window._report_load(
            window.configuration.load(load_preset(existing), PresetOrigin(path=existing), remember=True)
        )
        window._load_preset()
        assert existing.read_bytes() == original
        identity = window.configuration.identity
        if valid:
            assert identity.name == "New voice"
            assert identity.path is not None and identity.path != existing
            assert load_preset(identity.path).name == "New voice"
        else:
            assert (identity.name, identity.path) == ("Existing voice", existing)
            main_window.QMessageBox.warning.assert_called_once()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("strength", [0.735, 0.29, 0.0, 1.0])
def test_preset_strength_round_trip_matches_live_processor(
    qapp, config_in_tmp_without_devices, strength
):
    from mic_eq.config import load_preset

    window = MainWindow()
    configuration = window.configuration
    strength_control = next(
        row["proxy"] for row in cast(Any, window).quick_bridge.model["stages"][0]["rows"]
        if row["label"] == "Strength"
    )
    try:
        preset = configuration.current
        preset.rnnoise.strength = strength
        window._report_load(configuration.load(preset, PresetOrigin(), remember=True))
        assert window.processor.get_rnnoise_strength() == pytest.approx(strength)
        assert strength_control.text == f"{int(strength_control.value)}%"
        saved = configuration.current
        assert saved.rnnoise.strength == pytest.approx(strength)
        saved.name = "Strength round trip"
        path = window._save_preset_file(saved)
        assert path is not None
        restored = load_preset(path)
        assert restored.rnnoise.strength == pytest.approx(strength)
        window._report_load(configuration.load(restored, PresetOrigin(), remember=True))
        assert window.processor.get_rnnoise_strength() == pytest.approx(strength)
        assert configuration.identity.modified is False

        edited_value = (int(strength_control.value) + 1) % 101
        strength_control.setValue(float(edited_value))
        assert window.processor.get_rnnoise_strength() == pytest.approx(
            edited_value / 100.0
        )
        assert configuration.current.rnnoise.strength == pytest.approx(edited_value / 100.0)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("write_result", [
    False, OSError("disk full"), pytest.param(ValueError("non-finite value"), id="write_result3"),
])
def test_close_stops_processing_before_reporting_settings_failure(
    qapp, config_in_tmp_without_devices, monkeypatch, write_result,
):
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
            cast(Any, window).processor = window.engine_session.processor = SimpleNamespace(
                is_running=lambda: True, stop=lambda: events.append("stop"),
            )
            assert window._save_ui_state() is False
            event = QCloseEvent()
            window.closeEvent(event)
            assert events == ["stop", "warning"]
            assert event.isAccepted()
    finally:
        window.processor = window.engine_session.processor = original_processor
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_prompt_save_cancel_keeps_last_used_identity(save_window, monkeypatch, tmp_path):
    save_preset = Mock()
    monkeypatch.setattr(main_window, "save_preset", save_preset)
    monkeypatch.setattr(main_window.QMessageBox, "question", Mock(return_value=QMessageBox.StandardButton.No))

    save_window._prompt_save_current_preset(title="Save", question="Save?", preset_name="Cancelled", description="")

    assert _identity(save_window, tmp_path) == (tmp_path / "previous.json", True)
    save_preset.assert_not_called()


def test_failed_save_keeps_last_used_identity(save_window, monkeypatch, tmp_path):
    monkeypatch.setattr(main_window.QMessageBox, "question", Mock(return_value=QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(main_window, "save_preset", Mock(side_effect=OSError("disk full")))
    critical = Mock()
    monkeypatch.setattr(main_window.QMessageBox, "critical", critical)

    save_window._prompt_save_current_preset(title="Save", question="Save?", preset_name="Failed", description="")

    assert _identity(save_window, tmp_path) == (tmp_path / "previous.json", True)
    critical.assert_called_once()


@pytest.mark.parametrize("save_config_result", [False, OSError("config locked")])
def test_saved_file_without_identity_persistence_is_reported(
    save_window, monkeypatch, tmp_path, save_config_result
):
    saved_path = tmp_path / "fresh.json"
    monkeypatch.setattr(main_window, "save_preset", Mock(return_value=saved_path))
    if isinstance(save_config_result, Exception):
        save_config = Mock(side_effect=save_config_result)
    else:
        save_config = Mock(return_value=save_config_result)
    monkeypatch.setattr(main_window.QInputDialog, "getText", Mock(return_value=("Fresh", True)))

    with monkeypatch.context() as patch:  # the window's close must save again
        patch.setattr(main_window, "save_config", save_config)
        save_window._save_preset()

    assert _identity(save_window, tmp_path) == (saved_path, True)
    assert save_window.configuration.identity.remembered is False
    assert "could not remember" in str(save_window.status_message.text)


@pytest.mark.parametrize("refusal", [False, TypeError("invalid config")], ids=["false", "raises"])
def test_a_load_that_cannot_be_remembered_says_so(isolated_main_window, monkeypatch, tmp_path, refusal):
    window = isolated_main_window
    loaded = Preset(name="Loaded")
    save = Mock(side_effect=refusal) if isinstance(refusal, Exception) else Mock(return_value=refusal)
    with monkeypatch.context() as patch:  # the window's close must save again
        patch.setattr(main_window, "save_config", save)
        assert window._report_load(
            window.configuration.load(loaded, PresetOrigin(path=tmp_path / "loaded.json"), remember=True)
        )

    assert "could not be remembered" in str(window.status_message.text)


def _submenu(window: QMainWindow, title: str) -> QMenu:
    menubar = window.menuBar()
    assert menubar is not None
    for action in menubar.actions():
        menu = QMenu.menuInAction(action)
        if menu is None:
            continue
        for child in menu.actions():
            submenu = QMenu.menuInAction(child)
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


def test_automatic_route_apply_resolves_legacy_filename_and_shows_display_name(
    isolated_main_window, monkeypatch, tmp_path
):
    window = isolated_main_window
    filepath = tmp_path / "custom-voice.json"
    monkeypatch.setattr(main_window, "list_presets", lambda: [("Warm Voice", filepath)])
    monkeypatch.setattr(main_window, "load_preset", lambda _path: Preset())
    window.config.auto_apply_device_presets = True
    window.config.device_preset_bindings["route"] = DevicePresetBinding("custom:custom-voice.json")
    monkeypatch.setattr(window, "_current_device_route_key", lambda: "route")
    save = Mock(return_value=True)
    monkeypatch.setattr(window, "_save_config_safely", save)

    assert window._apply_bound_preset_for_current_route()
    assert window.config.device_preset_bindings["route"].preset_id == "custom-file:custom-voice.json"
    save.assert_called_once()
    assert window.status_message.text == "Route preset: Warm Voice"


def test_save_updates_owned_path_without_name_prompt(save_window, monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, "get_presets_dir", lambda: tmp_path)
    owned = save_window.configuration.current
    owned.name, owned.description = "My Sound", "Keep this description"
    save_window.configuration.mark_saved(owned, tmp_path / "previous.json")
    save = Mock(return_value=tmp_path / "previous.json")
    monkeypatch.setattr(save_window, "_save_preset_file", save)
    monkeypatch.setattr(
        main_window.QInputDialog, "getText", Mock(side_effect=AssertionError("Save must not ask for a new name"))
    )
    assert save_window._save_preset()
    args, kwargs = save.call_args
    assert kwargs["filepath"] == tmp_path / "previous.json"
    assert (args[0].name, args[0].description) == ("My Sound", "Keep this description")


@pytest.mark.parametrize("reply,save_ok,expected", [
    (QMessageBox.StandardButton.Cancel, True, False),
    (QMessageBox.StandardButton.Discard, False, True),
    (QMessageBox.StandardButton.Save, False, False),
    (QMessageBox.StandardButton.Save, True, True),
])
def test_unsaved_decision_preserves_failed_or_cancelled_saves(monkeypatch, reply, save_ok, expected):
    window = MainWindow.__new__(MainWindow)
    window._save_preset = Mock(return_value=save_ok)
    monkeypatch.setattr(main_window.QMessageBox, "question", lambda *_: reply)
    assert window._ask_to_save_sound_changes() is expected
    assert window._save_preset.call_count == int(reply == QMessageBox.StandardButton.Save)


def test_a_startup_preset_that_loads_counts_as_restored(qapp, config_in_tmp_without_devices):
    from mic_eq.config import load_config

    # A legacy stored key: it loads, and its migrated ID is saved.
    main_window.save_config(AppConfig(startup_preset="voice"))
    window = MainWindow()
    try:
        assert window.status_message.text == "Restored partial settings (some devices/presets unavailable)"
        assert load_config().startup_preset == "builtin:voice"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_a_last_used_preset_the_engine_refused_once_is_kept_for_the_next_launch(
    qapp, config_in_tmp_without_devices, monkeypatch
):
    from mic_eq.config import load_config
    from mic_eq.ui.noise_suppression_state import NoiseSuppressionState

    main_window.save_config(AppConfig(last_preset="builtin:voice"))
    original = NoiseSuppressionState.set_settings
    refusals = [RuntimeError("engine busy")]

    def refuse_once(state, settings):
        if refusals:
            raise refusals.pop()
        original(state, settings)

    monkeypatch.setattr(NoiseSuppressionState, "set_settings", refuse_once)
    window = MainWindow()
    try:
        assert "engine busy" in (window.config.load_warning or "")
        assert window.config.last_preset == "builtin:voice"
        assert load_config().last_preset == "builtin:voice"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_cancelling_the_unsaved_changes_prompt_for_a_route_preset_is_not_reported_unavailable(
    isolated_main_window, monkeypatch
):
    window = isolated_main_window
    window.config.auto_apply_device_presets = True
    window.config.device_preset_bindings["route"] = DevicePresetBinding("builtin:voice")
    monkeypatch.setattr(window, "_current_device_route_key", lambda: "route")
    window.gate_state.set_settings({"threshold_db": window.gate_state.get_settings()["threshold_db"] + 3})
    window.status_message.showMessage("before", 0)
    monkeypatch.setattr(main_window.QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Cancel)

    assert not window._apply_bound_preset_for_current_route()
    assert window.status_message.text == "before"
