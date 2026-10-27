"""Failures and leftovers around the main window: logged, kept, or cleaned up."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QSystemTrayIcon

from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow
from test_eq_editing_state import eq_control


class _BrokenProcessor:
    """The real processor, except for the calls named in `broken`."""

    def __init__(self, real: Any, *broken: str) -> None:
        self._real = real
        self._broken = set(broken)

    def __getattr__(self, name: str):
        if name in self._broken:
            def fail(*_args, **_kwargs):
                raise RuntimeError(f"{name} failed")

            return fail
        return getattr(self._real, name)


@pytest.mark.parametrize(
    ("method", "argument", "failing_call"),
    [
        ("_apply_input_channel_mode", "phase_safe_mono", "set_input_channel_mode"),
        ("_apply_input_cleanup_mode", "off", "set_input_cleanup_mode"),
        ("_apply_output_mute", None, "set_output_mute"),
    ],
)
def test_a_processor_call_that_fails_is_logged_above_debug(
    isolated_main_window, caplog, method, argument, failing_call
):
    window: Any = isolated_main_window
    window.processor = _BrokenProcessor(window.processor, failing_call)
    caplog.set_level(logging.INFO)

    getattr(window, method)(*(() if argument is None else (argument,)))

    assert any(
        r.levelno >= logging.WARNING and r.exc_info and f"{failing_call} failed" in str(r.exc_info[1])
        for r in caplog.records
    )


def test_a_recovery_service_that_fails_every_tick_is_logged_once_per_streak(
    isolated_main_window, caplog
):
    window: Any = isolated_main_window
    window.processor = _BrokenProcessor(window.processor, "service_recovery")
    caplog.set_level(logging.INFO)

    tick = window._service_stream_recovery
    for _ in range(5):
        tick()
    failures = [r for r in caplog.records if "recovery service failed" in r.getMessage()]
    assert len(failures) == 1 and failures[0].levelno == logging.WARNING

    window.processor = _BrokenProcessor(window.processor._real)  # works again
    tick()
    window.processor = _BrokenProcessor(window.processor._real, "service_recovery")
    tick()
    failures = [r for r in caplog.records if "recovery service failed" in r.getMessage()]
    assert len(failures) == 2


def test_an_unreadable_last_preset_is_kept_for_the_next_launch(
    qapp, monkeypatch, tmp_path, caplog
):
    from mic_eq.config import load_config
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        preset = window._get_current_preset()
        preset.name = "Kept Sound"
        filepath = window._save_preset_file(preset)
        assert filepath is not None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    def locked(_path):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(main_window, "load_preset", locked)
    restored = MainWindow()
    try:
        assert restored.config.last_preset == str(filepath)
        assert "Could not restore the previous preset" in (restored.config.load_warning or "")
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()
    assert load_config().last_preset == str(filepath)


def test_a_last_preset_locked_while_loading_is_kept_for_the_next_launch(
    qapp, monkeypatch, tmp_path
):
    """The real ``open`` inside ``load_preset`` fails; its error arrives wrapped in PresetValidationError."""
    from mic_eq.config import load_config
    from mic_eq.config_parts import presets, shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        preset = window._get_current_preset()
        preset.name = "Locked Sound"
        filepath = window._save_preset_file(preset)
        assert filepath is not None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    def locked(*_args, **_kwargs):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(presets, "open", locked, raising=False)
    restored = MainWindow()
    try:
        assert restored.config.last_preset == str(filepath)
        assert "Could not restore the previous preset" in (restored.config.load_warning or "")
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()
    assert load_config().last_preset == str(filepath)


def test_a_corrupt_last_preset_still_clears_its_reference(qapp, monkeypatch, tmp_path):
    from mic_eq.config import load_config
    from mic_eq.config_parts import shared

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        preset = window._get_current_preset()
        preset.name = "Broken Sound"
        filepath = window._save_preset_file(preset)
        assert filepath is not None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    filepath.write_text("{not json", encoding="utf-8")
    restored = MainWindow()
    try:
        assert restored.config.last_preset == ""
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()
    assert load_config().last_preset == ""


class _CountingTray(QSystemTrayIcon):
    def __init__(self) -> None:
        super().__init__()
        self.tooltips: list[str] = []

    def setToolTip(self, tip: str) -> None:
        self.tooltips.append(tip)
        super().setToolTip(tip)


def test_the_tray_tooltip_is_only_rewritten_when_its_text_changes(isolated_main_window):
    window: Any = isolated_main_window
    tray = _CountingTray()
    window._tray_icon = tray

    for _ in range(4):
        window._update_session_summary()
    assert len(tray.tooltips) == 1

    window.user_muted = True
    window._update_session_summary()
    assert len(tray.tooltips) == 2 and "Muted" in tray.tooltips[-1]


class _ProvisionalSettingsDialog(QDialog):
    """Stands in for a setup dialog: rejecting it puts the earlier setting back."""

    def __init__(self, parent, restore) -> None:
        super().__init__(parent)
        self._restore = restore

    def reject(self) -> None:
        self._restore()
        super().reject()


def test_quitting_from_the_tray_during_setup_rolls_back_before_asking_to_save(
    isolated_main_window, monkeypatch
):
    window: Any = isolated_main_window
    window._set_preset_modified(False)
    gain = eq_control(window, "gain", 4)
    saved_gain = gain.value
    gain.setValue(saved_gain + 6.0)  # the dialog's provisional candidate
    assert window.preset_modified
    prompts: list[str] = []

    def question(_parent, title, *_args, **_kwargs):
        prompts.append(title)
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: None))
    dialog = _ProvisionalSettingsDialog(window, lambda: gain.setValue(saved_gain))
    dialog.setModal(True)
    dialog.show()
    try:
        window._quit_from_tray()
    finally:
        dialog.hide()

    assert prompts == []
    assert gain.value == saved_gain


def test_quitting_from_the_tray_rolls_back_a_setup_dialog_under_a_message_box(
    isolated_main_window, monkeypatch
):
    window: Any = isolated_main_window
    window._set_preset_modified(False)
    gain = eq_control(window, "gain", 4)
    saved_gain = gain.value
    gain.setValue(saved_gain + 6.0)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QGuiApplication, "quit", staticmethod(lambda: None))
    dialog = _ProvisionalSettingsDialog(window, lambda: gain.setValue(saved_gain))
    dialog.setModal(True)
    dialog.show()
    box = QMessageBox(QMessageBox.Icon.Warning, "Setup", "Something went wrong", parent=dialog)
    box.setModal(True)
    box.show()
    try:
        assert QApplication.activeModalWidget() is box
        window._quit_from_tray()
    finally:
        box.hide()
        dialog.hide()

    assert gain.value == saved_gain


class _FakeSetupDialog(QDialog):
    """A real dialog that returns from exec() at once; the heavy setup dialogs are not under test."""

    setup_applied = Signal(str)
    auto_eq_applied = Signal(object)

    def __init__(self, parent, *_args, **_kwargs) -> None:
        super().__init__(parent)

    def exec(self) -> int:
        return int(QDialog.DialogCode.Rejected)


@pytest.mark.parametrize(
    ("dialog_name", "opener"),
    [
        ("VoiceSetupDialog", "_on_auto_voice_setup_clicked"),
        ("CalibrationDialog", "_on_auto_eq_clicked"),
        ("FirstRunSetupDialog", "_show_first_run_setup"),
    ],
)
def test_a_setup_dialog_is_destroyed_once_it_closes(
    isolated_main_window, monkeypatch, dialog_name, opener
):
    window: Any = isolated_main_window
    opened: list[_FakeSetupDialog] = []

    def make(parent, *args, **kwargs) -> _FakeSetupDialog:
        opened.append(_FakeSetupDialog(parent, *args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(main_window, dialog_name, make)

    getattr(window, opener)()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert len(opened) == 1 and not shiboken6.isValid(opened[0])

