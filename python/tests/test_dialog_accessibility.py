"""What the setup dialogs tell assistive technology, and the order the keyboard visits them."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QAbstractButton, QLabel

from capture_helpers import finish_take
from mic_eq.config import AppConfig
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.first_run_setup_dialog import FirstRunSetupDialog
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog
from mic_eq.ui.listening_comparison_dialog import ListeningComparisonDialog, RawProcessedDialog
from mic_eq.ui.theme import message_text_style
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from test_first_run_setup import _Owner, _Processor
from test_listening_comparison import _settings
from test_setup_level_check_flow import _voice, voice  # noqa: F401  (a fixture)
from test_voice_setup_integration import real_main_window  # noqa: F401  (a fixture)


def _name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface is not None
    return interface.text(QAccessible.Text.Name)


def _first_run(monkeypatch, step: str = "devices") -> tuple[FirstRunSetupDialog, _Owner]:
    monkeypatch.setattr("mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: True)
    config = AppConfig(
        first_run_setup_state="in_progress",
        first_run_setup_step=step,
        first_run_setup_steps={name: "pending" for name in ("devices", "route", "latency", "voice")},
    )
    owner = _Owner(config, _Processor())
    return FirstRunSetupDialog(owner), owner


def _listening(monkeypatch, *, test_my_sound: bool = False):
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    capture = {"audio_data": np.zeros(480, dtype=np.float32), "sample_rate": 48_000}
    if test_my_sound:
        return RawProcessedDialog(settings=_settings(), **capture)
    return ListeningComparisonDialog(current_settings=_settings(), proposed_settings=_settings(), **capture)


@pytest.fixture
def dialogs(isolated_main_window, monkeypatch, qapp):
    """Every setup dialog, built with its real widgets and nothing recording."""
    window = isolated_main_window
    first_run, _owner = _first_run(monkeypatch)
    shown = {
        "voice": VoiceSetupDialog(window),
        "auto_eq": CalibrationDialog(window),
        "latency": LatencyCalibrationDialog(window),
        "first_run": first_run,
        "listening": _listening(monkeypatch),
        "test_my_sound": _listening(monkeypatch, test_my_sound=True),
    }
    yield shown
    for dialog in shown.values():
        dialog.close()
        dialog.deleteLater()
    qapp.processEvents()


def test_a_label_is_read_by_its_text_not_by_a_fixed_name(dialogs) -> None:
    fixed = [
        f"{key}: {label.text()[:40]!r} is named {label.accessibleName()!r}"
        for key, dialog in dialogs.items()
        for label in dialog.findChildren(QLabel)
        if label.accessibleName() not in ("", label.text())
    ]
    assert fixed == []


def test_a_status_label_that_changes_is_read_with_its_new_text(dialogs, qapp) -> None:
    voice_dialog, auto_eq = dialogs["voice"], dialogs["auto_eq"]
    for label in (
        voice_dialog.phase_label, voice_dialog.time_label, voice_dialog.warning_label,
        auto_eq.time_label, auto_eq.warning_label,
        dialogs["latency"].status_label,
        dialogs["listening"].status_label, dialogs["listening"].progress_label,
    ):
        label.setText("Moving closer helps")
        assert _name(label) == "Moving closer helps"


def _visible_text(button: QAbstractButton) -> str:
    return " ".join(button.text().replace("&", "").split()).rstrip(".").casefold()


def test_a_button_name_contains_what_the_button_says(dialogs, monkeypatch) -> None:
    voice_dialog, auto_eq = dialogs["voice"], dialogs["auto_eq"]
    # The buttons that appear later in a run are named too.
    for dialog in (voice_dialog, auto_eq):
        dialog.level_override_btn.setText("Continue Anyway")
        dialog.level_override_btn.setVisible(True)
        dialog.retake_btn.setVisible(True)
    candidates = dict(dialogs)
    for step in ("route", "voice"):
        candidates[f"first_run_{step}"], _owner = _first_run(monkeypatch, step)
    mismatched = [
        f"{key}: {button.text()!r} is named {button.accessibleName()!r}"
        for key, dialog in candidates.items()
        for button in dialog.findChildren(QAbstractButton)
        if button.accessibleName() and _visible_text(button) not in button.accessibleName().casefold()
    ]
    assert mismatched == []


def test_the_setup_action_button_is_named_by_the_text_it_shows_at_each_step(monkeypatch, qapp) -> None:
    dialog, owner = _first_run(monkeypatch, "route")
    assert dialog.action_button.text() == "Check levels"
    assert _name(dialog.action_button) == "Check levels"
    dialog._run_current_step()
    dialog._finish_route_check()
    assert dialog.action_button.text() == "Confirm signal"
    assert _name(dialog.action_button) == "Confirm signal"
    assert _name(dialog.back_button) == "Back" and _name(dialog.skip_button) == "Skip this step"
    dialog._run_current_step()
    assert dialog.current_step == "voice"
    assert _name(dialog.action_button) == "Open Voice Setup"
    dialog.deleteLater()


def test_the_setup_status_uses_the_failure_and_success_colours(monkeypatch, qapp) -> None:
    dialog, owner = _first_run(monkeypatch)
    dialog.input_device_selector.clear()
    dialog._run_current_step()
    assert "must be available" in dialog.status_label.text()
    assert dialog.status_label.styleSheet() == message_text_style("bad") != message_text_style("idle")

    dialog, owner = _first_run(monkeypatch, "route")
    dialog._run_current_step()
    owner.processor.running = False
    dialog._finish_route_check()
    assert dialog.status_label.styleSheet() == message_text_style("bad")
    owner.processor.running = True
    dialog._run_current_step()
    dialog._finish_route_check()
    assert "Confirm" in dialog.status_label.text() or "confirm" in dialog.status_label.text()
    assert dialog.status_label.styleSheet() == message_text_style("ok") != message_text_style("idle")


def _visited(dialog, named: list) -> list:
    """The named widgets in the order Tab visits them, leaving out any that cannot take focus."""
    visited, widget = [], dialog.nextInFocusChain()
    for _ in range(1000):
        if widget is dialog:
            break
        if (
            any(widget is item for item in named)
            and widget.isVisibleTo(dialog) and widget.isEnabled()
            and widget.focusPolicy() & Qt.FocusPolicy.TabFocus
        ):
            visited.append(widget)
        widget = widget.nextInFocusChain()
    return visited


def _left_to_right(dialog, buttons: list) -> bool:
    xs = [button.mapTo(dialog, QPoint(0, 0)).x() for button in buttons]
    return xs == sorted(xs)


def _show(dialog, qapp) -> None:
    dialog.show()
    qapp.processEvents()


@pytest.mark.parametrize("name", ["voice", "auto_eq"])
def test_a_recording_dialog_visits_its_fields_then_its_buttons_left_to_right(dialogs, qapp, name) -> None:
    dialog = dialogs[name]
    dialog.dynamics_combo.setCurrentIndex(dialog.dynamics_combo.findData("custom")) if name == "voice" else None
    fields = (
        [dialog.curve_combo, dialog.dynamics_combo, dialog.custom_p95_spin, dialog.custom_peak_spin,
         dialog.target_lufs_spin]
        if name == "voice"
        else [dialog.curve_combo, dialog.target_mode_combo, dialog.smoothing_combo]
    )
    buttons = [
        dialog.cancel_btn, dialog.retake_btn, dialog.compare_button, dialog.level_override_btn,
        dialog.start_button,
    ]
    for button in buttons:
        button.setVisible(True)
        button.setEnabled(True)
    _show(dialog, qapp)
    assert _visited(dialog, [*fields, *buttons]) == [*fields, *buttons]
    assert _left_to_right(dialog, buttons)


def test_the_latency_dialog_visits_its_buttons_left_to_right(dialogs, qapp) -> None:
    dialog = dialogs["latency"]
    buttons = [dialog.close_button, dialog.reset_button, dialog.run_button, dialog.accept_button]
    dialog.accept_button.setEnabled(True)
    _show(dialog, qapp)
    assert _visited(dialog, buttons) == buttons
    assert _left_to_right(dialog, buttons)


def test_the_listening_dialog_visits_reject_before_keep_as_they_sit(dialogs, qapp) -> None:
    dialog = dialogs["listening"]
    buttons = [dialog.reject_button, dialog.keep_button]
    dialog.keep_button.setEnabled(True)
    _show(dialog, qapp)
    assert _visited(dialog, buttons) == buttons
    assert _left_to_right(dialog, buttons)


@pytest.mark.parametrize("step", ["devices", "route", "voice"])
def test_the_first_run_dialog_visits_top_to_bottom_then_its_buttons_left_to_right(
    monkeypatch, qapp, step
) -> None:
    dialog, _owner = _first_run(monkeypatch, step)
    dialog.back_button.setEnabled(True)
    _show(dialog, qapp)
    page = [dialog.mute_checkbox, dialog.input_device_selector, dialog.output_device_selector,
            dialog.advanced_latency_button]
    buttons = [dialog.pause_button, dialog.back_button, dialog.skip_button, dialog.action_button]
    visited = _visited(dialog, [*page, *buttons])
    assert visited == [widget for widget in (*page, *buttons) if widget.isVisibleTo(dialog)]
    assert visited[-4:] == buttons and _left_to_right(dialog, buttons)
    ys = [widget.mapTo(dialog, QPoint(0, 0)).y() for widget in visited[:-4]]
    assert ys == sorted(ys)
    dialog.close()
    dialog.deleteLater()


def test_the_recording_note_does_not_stand_over_a_running_analysis(voice, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(voice, "_start_analysis", lambda: None)
    voice.setup_state = "voice_recording"
    voice.time_label.setText("Complete!")
    finish_take(voice, _voice(0.2, seconds=10.0))
    assert voice.setup_state == "analyzing"
    assert voice.phase_label.text() == "Analyzing recordings"
    assert voice.time_label.text() == "Analysis in progress"
