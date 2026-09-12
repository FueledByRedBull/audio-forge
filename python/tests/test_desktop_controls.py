"""Focused checks for the optional desktop controls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QComboBox

from mic_eq.ui.desktop_integration import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    parse_global_hotkey,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui import desktop_integration


class _ModeCombo:
    def __init__(self) -> None:
        self._mode = "normal"
        self._blocked = False

    def currentData(self) -> str:
        return self._mode

    def currentIndex(self) -> int:
        return {"normal": 0, "bypass": 1, "raw": 2}[self._mode]

    def findData(self, mode: str) -> int:
        return {"normal": 0, "bypass": 1, "raw": 2}.get(mode, -1)

    def blockSignals(self, blocked: bool) -> None:
        self._blocked = blocked

    def setCurrentIndex(self, index: int) -> None:
        self._mode = ("normal", "bypass", "raw")[index]


class _Processor:
    def __init__(self) -> None:
        self.bypass: bool | None = None
        self.raw: bool | None = None

    def set_bypass(self, enabled: bool) -> None:
        self.bypass = enabled

    def set_raw_monitor_enabled(self, enabled: bool) -> None:
        self.raw = enabled


@pytest.mark.parametrize(
    ("shortcut", "modifiers", "key"),
    (
        ("Ctrl+Alt+M", MOD_CONTROL | MOD_ALT, ord("M")),
        ("Shift+F12", MOD_SHIFT, 0x7B),
        ("Control+Space", MOD_CONTROL, 0x20),
    ),
)
def test_parse_global_hotkey(shortcut: str, modifiers: int, key: int) -> None:
    assert parse_global_hotkey(shortcut) == (modifiers, key)


def test_parse_global_hotkey_rejects_ambiguous_shortcut() -> None:
    with pytest.raises(ValueError, match="modifier"):
        parse_global_hotkey("M")
    with pytest.raises(ValueError, match="function key"):
        parse_global_hotkey("Ctrl+F25")


def test_processing_mode_updates_both_native_flags() -> None:
    owner: Any = MainWindow.__new__(MainWindow)
    owner.processing_mode_combo = _ModeCombo()
    owner.processor = _Processor()
    owner.status_bar = SimpleNamespace(showMessage=lambda *_args: None)
    owner._update_session_summary = lambda: None

    MainWindow._set_processing_mode(owner, "bypass")
    assert owner._processing_mode() == "bypass"
    assert owner.processor.bypass is True
    assert owner.processor.raw is False

    MainWindow._set_processing_mode(owner, "raw")
    assert owner._processing_mode() == "raw"
    assert owner.processor.bypass is False
    assert owner.processor.raw is True

    MainWindow._set_processing_mode(owner, "invalid")
    assert owner._processing_mode() == "normal"
    assert owner.processor.bypass is False
    assert owner.processor.raw is False


def test_tray_close_keeps_audio_running_until_explicit_quit() -> None:
    owner: Any = SimpleNamespace(
        _quitting=False, _tray_icon=SimpleNamespace(isVisible=lambda: True, hide=Mock()),
        _close_to_tray_action=SimpleNamespace(isChecked=lambda: True), hide=Mock(),
        status_bar=Mock(), _unregister_mute_hotkey=Mock(), config=SimpleNamespace(),
        x=lambda: 0, y=lambda: 0, width=lambda: 1280, height=lambda: 850,
        _save_ui_state=lambda: True, processor=Mock(),
    )
    event = Mock()
    MainWindow.closeEvent(owner, event)
    owner.hide.assert_called_once()
    owner.processor.stop.assert_not_called()
    event.ignore.assert_called_once()
    owner._quitting = True
    MainWindow.closeEvent(owner, event)
    owner.processor.stop.assert_called_once()
    owner._unregister_mute_hotkey.assert_called_once()
    event.accept.assert_called_once()


def test_hotkey_registration_is_idempotent_and_released(qapp, monkeypatch):
    register = Mock(return_value=True)
    unregister = Mock(return_value=True)
    monkeypatch.setattr(desktop_integration.ctypes.windll.user32, "RegisterHotKey", register)
    monkeypatch.setattr(desktop_integration.ctypes.windll.user32, "UnregisterHotKey", unregister)
    hotkey = desktop_integration.GlobalMuteHotkey("Ctrl+Alt+M", lambda: None)
    try:
        assert hotkey.register() == (True, None)
        assert hotkey.register() == (True, None)
        register.assert_called_once()
        assert register.call_args.args[2] & 0x4000
    finally:
        hotkey.unregister()
    hotkey.unregister()
    unregister.assert_called_once()
    assert not hotkey.registered


def test_failed_hotkey_registration_is_not_shown_or_saved_as_enabled(monkeypatch):
    hotkey = SimpleNamespace(register=lambda: (False, "Shortcut already in use"))
    monkeypatch.setattr("mic_eq.ui.main_window.GlobalMuteHotkey", lambda *_: hotkey)
    owner: Any = SimpleNamespace(
        config=SimpleNamespace(mute_hotkey="Ctrl+Alt+M"),
        _unregister_mute_hotkey=lambda: None,
        _toggle_mute_from_hotkey=lambda: None,
        _mute_hotkey_action=Mock(), status_bar=Mock(),
    )
    assert not MainWindow._register_mute_hotkey(owner, "Ctrl+Alt+M")
    assert owner.config.mute_hotkey == ""
    owner._mute_hotkey_action.setChecked.assert_called_once_with(False)


def test_failed_model_switch_retains_previous_model_and_selection(qapp, monkeypatch):
    combo = QComboBox()
    for model in ("rnnoise", "deepfilter-ll", "deepfilter"):
        combo.addItem(model, model)
    combo.setCurrentIndex(2)
    warning = Mock()
    monkeypatch.setattr("mic_eq.ui.main_window.QMessageBox.warning", warning)
    owner: Any = SimpleNamespace(
        model_combo=combo,
        processor=SimpleNamespace(get_noise_model=lambda: "deepfilter-ll",
                                  set_noise_model=Mock(return_value=False)),
        _set_noise_suppression_latency_label=Mock(), status_bar=Mock(),
    )
    owner._apply_noise_model = lambda model: MainWindow._apply_noise_model(owner, model)
    MainWindow._on_model_changed(owner, 2)
    assert combo.currentData() == "deepfilter-ll"
    assert not combo.signalsBlocked()
    owner.processor.set_noise_model.assert_called_once_with("deepfilter")
    owner._set_noise_suppression_latency_label.assert_not_called()
    warning.assert_called_once()
