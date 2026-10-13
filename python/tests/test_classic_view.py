"""The widget view renders the same shared state as the Qt Quick view."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QButtonGroup, QLayout, QMenu, QWidget

from mic_eq.config import AppConfig
from mic_eq.ui.main_window import CLASSIC_ATTRIBUTES, PAGE_NAMES, MainWindow


@pytest.fixture
def classic_window(qapp, monkeypatch):
    monkeypatch.setenv("AUDIOFORGE_QML", "0")
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    window.show()
    qapp.processEvents()
    return window


def test_widget_attribute_table_lists_exactly_what_the_builder_assigns(classic_window) -> None:
    built = {
        name
        for name, value in vars(classic_window).items()
        if isinstance(value, (QWidget, QLayout, QButtonGroup))
        and not isinstance(value, QMenu)
        and (not name.startswith("_") or name in CLASSIC_ATTRIBUTES)
    }
    assert built == set(CLASSIC_ATTRIBUTES)


def test_transport_follows_the_stop_action(classic_window) -> None:
    window = classic_window
    assert window.stop_btn.isHidden() and not window.start_btn.isHidden()
    # The action states _start_processing and _stop_processing set.
    window.start_btn_action.setEnabled(False)
    window.stop_btn_action.setEnabled(True)
    assert window.start_btn.isHidden() and not window.stop_btn.isHidden()
    window.start_btn_action.setEnabled(True)
    window.stop_btn_action.setEnabled(False)
    assert window.stop_btn.isHidden() and not window.start_btn.isHidden()


def test_pages_follow_the_shared_page_state_in_both_directions(classic_window, monkeypatch) -> None:
    window = classic_window
    opened: list[int] = []
    monkeypatch.setattr(
        window, "_refresh_login_startup_action", lambda: opened.append(window.shell_page.currentIndex())
    )
    # The stack, the navigation buttons and the state share one page order.
    assert window.page_stack.count() == len(window.nav_group.buttons()) == len(PAGE_NAMES)
    assert window.page_stack.widget(window.HEALTH_PAGE_INDEX) is window.health_details
    assert window.page_stack.widget(window.SETTINGS_PAGE_INDEX) is window.settings_page
    window.shell_page.setIndex(len(PAGE_NAMES))
    assert window.shell_page.currentIndex() == 0

    window.nav_group.button(window.HEALTH_PAGE_INDEX).click()
    assert window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX
    assert window.health_details_button.isChecked()
    assert opened == []
    window.nav_group.button(window.SETTINGS_PAGE_INDEX).click()
    assert window.page_stack.currentIndex() == window.SETTINGS_PAGE_INDEX
    assert not window.health_details_button.isChecked()
    window.shell_page.setIndex(0)
    window.shell_page.setIndex(window.SETTINGS_PAGE_INDEX)
    assert opened == [window.SETTINGS_PAGE_INDEX] * 2
    assert window.nav_group.checkedId() == window.SETTINGS_PAGE_INDEX

    window.health_details_button.click()
    assert window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX
    window.health_details_button.click()
    assert window.shell_page.currentIndex() == 0


def test_a_combo_edit_reaches_the_shared_choice_and_the_processor(classic_window) -> None:
    window = classic_window
    window.processing_mode_combo.setCurrentIndex(window.processing_mode_combo.findData("bypass"))
    assert window.processing_mode_choice.currentData() == "bypass"
    assert window.processor.is_bypass()
    window.processing_mode_choice.setIndex(0)
    assert window.processing_mode_combo.currentData() == "normal"


def test_shared_mute_action_updates_classic_switch_painting(classic_window, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    window = classic_window
    window.user_mute_checkbox_action.setChecked(True)
    assert window.user_muted
    assert window.user_mute_checkbox.isChecked()
    assert window.user_mute_checkbox._position == 1.0
    window.user_mute_checkbox.click()
    assert not window.user_muted
    assert not window.user_mute_checkbox_action.isChecked()
    assert window.user_mute_checkbox._position == 0.0
