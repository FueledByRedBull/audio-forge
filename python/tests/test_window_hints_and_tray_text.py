"""What the main window tells people when a control is locked, a start cannot happen, or the tray is read."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QWidget

from mic_eq.ui import latency_calibration_dialog as latency_ui
from mic_eq.ui.components import StepHeader
from mic_eq.ui.control_specs import control_label
from mic_eq.ui.desktop_integration import TRAY_TOOLTIP_LIMIT, build_tray_tooltip
from mic_eq.ui.health import advice_for
from mic_eq.ui.main_window import NO_DESTINATION_WARNING
from mic_eq.ui.theme import PRIMARY_ACTION_BUTTON_STYLE

LONG_INPUT = "Microphone (Realtek High Definition Audio(SST)) with a very long suffix"
LONG_OUTPUT = "CABLE Input (VB-Audio Virtual Cable) with a very long suffix on it too"


def test_the_tray_tooltip_fits_what_windows_shows_and_keeps_both_ends_of_the_route():
    tooltip = build_tray_tooltip(
        processing=True, muted=True, health="Healthy", route=(LONG_INPUT, LONG_OUTPUT)
    )

    assert len(tooltip) <= TRAY_TOOLTIP_LIMIT
    assert tooltip.startswith("AudioForge: Running / Muted | Health: Healthy | Microphone")
    assert " -> CABLE" in tooltip and tooltip.endswith("...")


def test_a_short_input_leaves_the_rest_of_the_room_to_the_output_name():
    tooltip = build_tray_tooltip(processing=True, muted=False, health="OK", route=("Mic", LONG_OUTPUT))

    assert len(tooltip) <= TRAY_TOOLTIP_LIMIT
    assert "| Mic -> CABLE Input (VB-Audio Virtual Cable) with a very long" in tooltip


def test_the_route_is_left_out_when_the_state_leaves_no_room_for_it():
    tooltip = build_tray_tooltip(processing=True, muted=True, health="x" * 120, route=("Mic", "Cable"))

    assert len(tooltip) == TRAY_TOOLTIP_LIMIT and "->" not in tooltip


def test_locked_device_pickers_say_why(isolated_main_window):
    window: Any = isolated_main_window
    for choice in (window.input_choice, window.output_choice):
        assert choice.toolTip == ""
        choice.setEnabled(False)
        assert choice.toolTip == "Stop processing to change devices."
        choice.setEnabled(True)
        assert choice.toolTip == ""
    refresh = window.quick_bridge.model["top"]["refresh"]
    assert "while processing" not in refresh.toolTip
    window.refresh_btn_action.setEnabled(False)
    assert "while processing" in refresh.toolTip
    # Pickers without a reason to lock keep their own help.
    window.processing_mode_choice.setEnabled(False)
    assert window.processing_mode_choice.toolTip.startswith("Normal applies")


def test_start_without_a_destination_says_so_where_warnings_are_shown(isolated_main_window):
    window: Any = isolated_main_window
    banner = window.shell_texts["device_warning_banner"]
    banner.update(shown=False)  # devices exist, but none is chosen as the destination

    assert window._start_processing(interactive=False) is False
    assert banner.shown and banner.text == NO_DESTINATION_WARNING

    window._on_device_changed()
    assert not banner.shown


def test_a_device_warning_that_is_already_up_is_not_replaced(isolated_main_window):
    window: Any = isolated_main_window
    banner = window.shell_texts["device_warning_banner"]
    banner.update(text="Warning: No output devices detected.", shown=True)

    window._start_processing(interactive=False)
    window._on_device_changed()

    assert banner.text == "Warning: No output devices detected." and banner.shown


def test_settings_rows_that_open_a_menu_show_the_current_choice(isolated_main_window):
    window: Any = isolated_main_window
    rows = {row["proxy"].text: row for card in window.shell_settings_cards for row in card["rows"]}

    assert rows["Startup Preset: Last Used"]["kind"] == "menu"
    startup = rows["Startup Preset: Last Used"]["proxy"]
    seen: list[str] = []
    startup.changed.connect(lambda: seen.append(startup.text))
    window._startup_preset_menu.triggered.emit(window._startup_preset_menu.actions()[0])
    assert seen


def test_health_advice_names_settings_by_the_labels_the_window_uses():
    assert control_label("compressor", "makeup_gain_db") == "Makeup gain"
    assert control_label("limiter", "careful_output_enabled") == "Careful output mode"
    assert control_label("gate", "vad_hold_time_ms") == "Hold time"

    ceiling = advice_for("Output: PEAK ABOVE CEILING (peak 0.4 dB)")
    assert "Careful output mode is under Advanced on the Limiter card" in ceiling
    assert "Makeup gain is under Advanced on the Compressor card" in ceiling
    assert "Hold time is under Advanced on the Noise Gate card" in advice_for("Gate: CHATTERING (3 times)")
    assert "Backend setting" in advice_for("Backend: rnnoise failed")


def _latency_dialog(profile: dict | None = None) -> Any:
    owner: Any = QWidget()
    owner.processor = Mock()
    owner.processor.get_engine_latency_ms.return_value = 0.0
    owner.processor.stop_raw_recording.return_value = np.zeros(10, dtype=np.float32)
    dialog = latency_ui.LatencyCalibrationDialog(owner, existing_profile=profile)
    setattr(dialog, "_owner", owner)  # the dialog does not own its parent
    return dialog


def test_latency_dialog_makes_run_the_main_action_until_there_is_a_result(qapp):
    dialog = _latency_dialog()
    try:
        assert not dialog.accept_button.isEnabled()
        assert dialog.run_button.styleSheet() == PRIMARY_ACTION_BUTTON_STYLE
        dialog._set_accept_enabled(True)
        assert dialog.run_button.styleSheet() == ""
        assert dialog.accept_button.styleSheet() == PRIMARY_ACTION_BUTTON_STYLE
    finally:
        dialog.close()


def test_latency_dialog_shows_the_two_readings_that_matter_and_keeps_the_rest_behind_details(qapp):
    dialog = _latency_dialog()
    try:
        dialog.show()
        qapp.processEvents()
        assert dialog.total_label.isVisibleTo(dialog) and dialog.confidence_label.isVisibleTo(dialog)
        for label in (dialog.round_trip_label, dialog.agreement_label, dialog.ambiguity_label, dialog.engine_label):
            assert not label.isVisibleTo(dialog)

        dialog.details_button.click()
        assert dialog.agreement_label.isVisibleTo(dialog) and dialog.details_button.text() == "Hide details"
        dialog.details_button.click()
        assert not dialog.agreement_label.isVisibleTo(dialog)
    finally:
        dialog.close()


def test_a_failed_device_listing_leaves_the_list_empty_and_says_so_in_the_warning(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window

    def broken():
        raise OSError("driver went away")

    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", broken)
    window._refresh_devices()

    assert window.input_choice.count() == 0
    banner = window.shell_texts["device_warning_banner"]
    assert banner.shown and "Could not list input devices (driver went away)" in banner.text


def test_the_step_header_names_the_current_step(qapp):
    header = StepHeader(("Level", "Record", "Review"))
    assert header.accessibleName() == "Step 1 of 3: Level"
    header.set_current(2)
    assert header.accessibleName() == "Step 3 of 3: Review"


def test_a_new_status_message_is_announced_once(isolated_main_window, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(QAccessible, "isActive", staticmethod(lambda: True))
    monkeypatch.setattr(QAccessible, "updateAccessibility", staticmethod(lambda event: sent.append(event.__class__.__name__)))
    status = isolated_main_window.status_message
    status.showMessage("Output muted")
    status.showMessage("Output muted")
    status.showMessage("Output unmuted")
    assert sent == ["QAccessibleAnnouncementEvent"] * 2
