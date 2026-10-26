"""Settings that did not save, and startup problems, stay on the warning banner until resolved."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mic_eq.config import AppConfig, BUILTIN_PRESETS, DeviceIdentity
from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow


def _banner(window: Any):
    return window.shell_texts["config_warning_banner"]


def _device(name: str, endpoint_id: str, direction: str, *, default: bool = True):
    return SimpleNamespace(
        name=name, is_default=default, endpoint_id=endpoint_id, host_api="WASAPI",
        sample_rate=48_000, channels=2, name_ordinal=0, direction=direction,
    )


@pytest.fixture
def start_window(qapp, monkeypatch):
    """Builds a window from a config; the windows are closed at the end of the test."""
    windows: list[MainWindow] = []

    def build(config: AppConfig, *, inputs=(), outputs=()) -> MainWindow:
        monkeypatch.setattr(main_window, "load_config", lambda: config)
        monkeypatch.setattr(main_window, "save_config", lambda _config: True)
        monkeypatch.setattr(main_window, "list_presets", lambda: [])
        monkeypatch.setattr(main_window, "list_input_devices", lambda: list(inputs))
        monkeypatch.setattr(main_window, "list_output_devices", lambda: list(outputs))
        window = MainWindow()
        windows.append(window)
        return window

    yield build
    for window in windows:
        window.close()
        window.deleteLater()
    qapp.processEvents()


def test_a_failed_save_stays_on_the_banner_until_a_save_works(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window
    saves = {"works": False}
    monkeypatch.setattr(main_window, "save_config", lambda _config: saves["works"])
    assert not _banner(window).shown

    window._on_user_mute_toggled(True)
    window.status_message.showMessage("Output unmuted", 1)  # the strip moves on; the banner does not

    assert _banner(window).shown and "could not be saved" in _banner(window).text
    assert window._save_config_safely() is False and _banner(window).shown

    saves["works"] = True
    assert window._save_config_safely() is True
    assert not _banner(window).shown and _banner(window).text == ""


def test_a_save_that_raises_is_reported_the_same_way(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window

    def locked(_config):
        raise PermissionError("config.json is locked")

    monkeypatch.setattr(main_window, "save_config", locked)

    assert window._save_config_safely() is False
    assert _banner(window).shown and "could not be saved" in _banner(window).text
    monkeypatch.setattr(main_window, "save_config", lambda _config: True)  # closing saves too


def test_a_save_problem_is_added_to_the_load_warning_and_leaves_it_when_resolved(
    start_window, monkeypatch
):
    config = AppConfig()
    config.load_warning = "AudioForge found an invalid config.json."
    window = start_window(config)
    assert _banner(window).text == "AudioForge found an invalid config.json."

    monkeypatch.setattr(main_window, "save_config", lambda _config: False)
    window._save_config_safely()
    assert _banner(window).text.startswith("AudioForge found an invalid config.json.\n")
    assert "could not be saved" in _banner(window).text

    monkeypatch.setattr(main_window, "save_config", lambda _config: True)
    window._save_config_safely()
    assert _banner(window).text == "AudioForge found an invalid config.json." and _banner(window).shown


def test_a_missing_last_preset_stays_on_the_banner_until_a_preset_is_saved(start_window, qapp):
    window = start_window(AppConfig(last_preset="builtin:no_such_preset"))
    qapp.processEvents()
    window.status_message.showMessage("Ready")  # the strip has moved on

    assert _banner(window).shown and "no_such_preset" in _banner(window).text

    key = next(iter(BUILTIN_PRESETS))
    assert window._apply_preset(BUILTIN_PRESETS[key], preset_key=key)  # another preset is no fix
    assert _banner(window).shown and "no_such_preset" in _banner(window).text

    assert window._save_preset_file(window._get_current_preset()) is not None
    assert not _banner(window).shown


def test_a_missing_last_preset_is_resolved_when_that_preset_loads(start_window, monkeypatch):
    window = start_window(AppConfig(last_preset="builtin:ghost"))
    assert _banner(window).shown

    monkeypatch.setitem(BUILTIN_PRESETS, "ghost", next(iter(BUILTIN_PRESETS.values())))
    assert window._apply_preset(BUILTIN_PRESETS["ghost"], preset_key="ghost")

    assert not _banner(window).shown


def test_the_startup_preset_problem_outlasts_other_and_automatic_loads(start_window, monkeypatch):
    window = start_window(AppConfig(startup_preset="builtin:ghost"))
    assert _banner(window).shown and "ghost" in _banner(window).text
    key = next(iter(BUILTIN_PRESETS))

    assert window._apply_preset(BUILTIN_PRESETS[key], preset_key=key)
    assert window._load_device_preset_id(f"builtin:{key}")  # a route-bound preset, loaded by itself
    assert _banner(window).shown and "ghost" in _banner(window).text

    monkeypatch.setitem(BUILTIN_PRESETS, "ghost", BUILTIN_PRESETS[key])
    assert window._apply_preset(BUILTIN_PRESETS["ghost"], preset_key="ghost")
    assert not _banner(window).shown


def test_choosing_another_startup_preset_resolves_its_problem(start_window):
    window = start_window(AppConfig(startup_preset="builtin:ghost"))
    assert _banner(window).shown

    window._set_startup_preset("")

    assert not _banner(window).shown


def test_a_saved_device_that_is_unplugged_stays_on_the_banner_until_it_is_back(
    start_window, monkeypatch
):
    mic = DeviceIdentity(name="USB Mic", endpoint_id="usb-mic", direction="input", host_api="WASAPI")
    config = AppConfig(last_input_device="USB Mic", last_input_device_identity=mic)
    window = start_window(
        config,
        inputs=[_device("Built-in Mic", "builtin", "input")],
        outputs=[_device("Speakers", "spk", "output")],
    )
    window.status_message.showMessage("Ready")

    assert _banner(window).shown and "USB Mic" in _banner(window).text

    monkeypatch.setattr(
        main_window, "list_input_devices",
        lambda: [_device("Built-in Mic", "builtin", "input", default=False),
                 _device("USB Mic", "usb-mic", "input")],
    )
    window._refresh_devices()
    assert window.input_choice.currentData().endpoint_id == "usb-mic"
    assert not _banner(window).shown


def test_choosing_a_replacement_destination_clears_its_banner_line(start_window):
    cable = DeviceIdentity(name="Virtual Cable", endpoint_id="cable", direction="output", host_api="WASAPI")
    config = AppConfig(last_output_device="Virtual Cable", last_output_device_identity=cable)
    window = start_window(
        config,
        inputs=[_device("Mic", "mic", "input")],
        outputs=[_device("Speakers", "spk", "output")],
    )
    assert _banner(window).shown and "Virtual Cable" in _banner(window).text

    window.output_choice.setCurrentIndex(0)

    assert not _banner(window).shown
