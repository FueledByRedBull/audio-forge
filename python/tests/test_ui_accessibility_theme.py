# pyright: reportAttributeAccessIssue=false
"""Deterministic accessibility and semantic-theme contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QScrollArea, QWidget

from mic_eq.config import AppConfig
from mic_eq.ui.accessibility import audit_widget_tree, set_accessible
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.first_run_setup_dialog import FirstRunSetupDialog
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog
from mic_eq.ui.main_window import MainWindow, _fit_window_geometry_to_screens
from mic_eq.ui.theme import (
    PALETTE,
    TEXT_CONTRAST_PAIRS,
    application_palette,
    contrast_ratio,
    prefers_reduced_motion,
    qcolor,
)
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog


UI_ROOT = Path(__file__).resolve().parents[1] / "mic_eq" / "ui"
LITERAL_COLOR = re.compile(r"#[0-9A-Fa-f]{6,8}|QColor\s*\(")
PIXEL_FONT = re.compile(r"font-size\s*:\s*[0-9.]+px")


@pytest.fixture
def isolated_main_window(qapp, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: None)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    yield window
    try:
        window.processor.stop()
    except Exception:
        pass
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_all_semantic_text_pairs_meet_wcag_aa_contrast() -> None:
    measured = {
        name: contrast_ratio(foreground, background)
        for name, foreground, background in TEXT_CONTRAST_PAIRS
    }
    assert measured
    assert min(measured.values()) >= 4.5, measured


def test_application_palette_uses_the_same_semantic_surface() -> None:
    palette = application_palette()
    assert palette.color(QPalette.ColorRole.Window).name() == PALETTE.app_surface
    assert palette.color(QPalette.ColorRole.WindowText).name() == PALETTE.text_primary
    assert palette.color(QPalette.ColorRole.Text).name() == PALETTE.text_primary
    assert palette.color(QPalette.ColorRole.Highlight).name() == PALETTE.action_primary


def test_saved_geometry_is_fitted_to_exactly_one_available_screen() -> None:
    screens = [QRect(0, 0, 1920, 1040), QRect(1920, 0, 1920, 1040)]
    fitted = _fit_window_geometry_to_screens(
        {"x": 200, "y": 20, "width": 3600, "height": 1200},
        screens,
    )
    assert fitted is not None
    assert any(screen.contains(fitted) for screen in screens)
    assert fitted.width() == 1920
    assert fitted.height() == 1040

    offscreen = _fit_window_geometry_to_screens(
        {"x": -5000, "y": 3000, "width": 1280, "height": 850},
        screens,
    )
    assert offscreen is not None
    assert screens[0].contains(offscreen)


def test_theme_color_alpha_is_clamped_and_invalid_tokens_fail() -> None:
    assert qcolor("#123456", alpha=-10).alpha() == 0
    assert qcolor("#123456", alpha=999).alpha() == 255
    with pytest.raises(ValueError, match="Invalid color token"):
        contrast_ratio("not-a-color", "#ffffff")


def test_ui_sources_do_not_bypass_semantic_color_or_scalable_type_tokens() -> None:
    violations: list[str] = []
    for path in sorted(UI_ROOT.glob("*.py")):
        if path.name == "theme.py":
            continue
        source = path.read_text(encoding="utf-8")
        if LITERAL_COLOR.search(source):
            violations.append(f"{path.name}: literal color")
        if PIXEL_FONT.search(source):
            violations.append(f"{path.name}: pixel font size")
    assert violations == []


def test_reduced_motion_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "yes")
    assert prefers_reduced_motion() is True
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "0")
    assert prefers_reduced_motion() is False


def test_accessible_name_rejects_empty_text(qapp) -> None:
    widget = QScrollArea()
    with pytest.raises(ValueError, match="must not be empty"):
        set_accessible(widget, "  &  ")
    widget.deleteLater()
    qapp.processEvents()


def test_main_window_and_workflow_dialogs_have_named_controls(
    isolated_main_window,
    qapp,
) -> None:
    window = isolated_main_window
    dialogs = (
        CalibrationDialog(window),
        VoiceSetupDialog(window),
        LatencyCalibrationDialog(window),
    )
    roots = (window, *dialogs)

    problems = {
        type(root).__name__: audit_widget_tree(root)
        for root in roots
        if audit_widget_tree(root)
    }
    assert problems == {}

    for dialog in dialogs:
        dialog.close()
        dialog.deleteLater()
    qapp.processEvents()


def test_main_keyboard_order_and_small_viewport_are_operable(
    isolated_main_window,
    qapp,
) -> None:
    window = isolated_main_window
    expected_top_row = (
        window.output_combo,
        window.input_channel_mode_combo,
        window.input_cleanup_mode_combo,
        window.refresh_btn,
    )
    current = window.input_combo
    for expected in expected_top_row:
        current = current.nextInFocusChain()
        while not current.focusPolicy() & Qt.FocusPolicy.TabFocus:
            current = current.nextInFocusChain()
        assert current is expected

    assert window.minimumWidth() <= 900
    assert window.minimumHeight() <= 640
    assert isinstance(window.centralWidget(), QScrollArea)
    assert (
        window.content_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert (
        window.content_scroll_area.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        window.eq_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )

    window.resize(900, 640)
    window.show()
    qapp.processEvents()
    qapp.processEvents()
    assert window.width() == 900
    assert window.height() == 640
    assert window.main_splitter.orientation() == Qt.Orientation.Vertical
    assert window.content_scroll_area.horizontalScrollBar().maximum() == 0
    assert window.eq_scroll_area.horizontalScrollBar().maximum() == 0
    assert (
        window.content_scroll_area.widget().width()
        <= window.content_scroll_area.viewport().width()
    )
    assert (
        window.eq_scroll_area.widget().width()
        <= window.eq_scroll_area.viewport().width()
    )

    window.content_scroll_area.ensureWidgetVisible(window.start_btn)
    qapp.processEvents()
    viewport_rect = window.content_scroll_area.viewport().rect()
    start_top_left = window.start_btn.mapTo(
        window.content_scroll_area.viewport(), window.start_btn.rect().topLeft()
    )
    assert viewport_rect.contains(start_top_left)


@pytest.mark.parametrize(
    ("width", "height", "orientation"),
    (
        (900, 640, Qt.Orientation.Vertical),
        (1024, 700, Qt.Orientation.Vertical),
        (1159, 760, Qt.Orientation.Vertical),
        (1280, 800, Qt.Orientation.Horizontal),
        (1600, 900, Qt.Orientation.Horizontal),
        (1920, 1040, Qt.Orientation.Horizontal),
    ),
)
def test_main_window_has_no_horizontal_overflow_across_breakpoints(
    isolated_main_window,
    qapp,
    width: int,
    height: int,
    orientation: Qt.Orientation,
) -> None:
    window = isolated_main_window
    window.resize(width, height)
    window.show()
    qapp.processEvents()
    qapp.processEvents()

    assert window.main_splitter.orientation() == orientation
    assert window.content_scroll_area.horizontalScrollBar().maximum() == 0
    assert window.eq_scroll_area.horizontalScrollBar().maximum() == 0
    assert window.eq_panel._band_layout_columns in window.eq_panel.BAND_COLUMN_OPTIONS
    for index in range(window.control_tabs.count()):
        tab_scroll = window.control_tabs.widget(index)
        assert isinstance(tab_scroll, QScrollArea)
        tab_body = tab_scroll.widget()
        tab_viewport = tab_scroll.viewport()
        horizontal_bar = tab_scroll.horizontalScrollBar()
        assert tab_body is not None
        assert tab_viewport is not None
        assert horizontal_bar is not None
        oversized = [
            (
                type(widget).__name__,
                getattr(widget, "title", lambda: widget.objectName())(),
                widget.minimumSizeHint().width(),
                widget.sizeHint().width(),
            )
            for widget in tab_body.findChildren(QWidget)
            if widget.minimumSizeHint().width() > tab_viewport.width()
        ]
        assert horizontal_bar.maximum() == 0, (
            window.control_tabs.tabText(index),
            oversized,
        )


@pytest.mark.parametrize(
    ("dialog_type", "width", "height"),
    (
        (CalibrationDialog, 480, 360),
        (VoiceSetupDialog, 500, 380),
        (LatencyCalibrationDialog, 460, 360),
    ),
)
def test_workflow_dialogs_scroll_vertically_without_horizontal_overflow(
    isolated_main_window,
    qapp,
    dialog_type,
    width: int,
    height: int,
) -> None:
    dialog = dialog_type(isolated_main_window)
    dialog.resize(width, height)
    dialog.show()
    qapp.processEvents()
    qapp.processEvents()

    scroll_area = dialog.content_scroll_area
    oversized = [
        (
            type(widget).__name__,
            getattr(widget, "title", lambda: widget.objectName())(),
            widget.minimumSizeHint().width(),
            widget.sizeHint().width(),
        )
        for widget in scroll_area.widget().findChildren(QWidget)
        if widget.minimumSizeHint().width() > scroll_area.viewport().width()
    ]
    assert scroll_area.horizontalScrollBar().maximum() == 0, oversized
    assert scroll_area.widget().width() <= scroll_area.viewport().width()

    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def test_first_run_setup_buttons_fit_the_minimum_dialog(
    isolated_main_window,
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda _config: None,
    )
    dialog = FirstRunSetupDialog(isolated_main_window)
    dialog.resize(440, 280)
    dialog.show()
    qapp.processEvents()
    qapp.processEvents()

    content_rect = dialog.contentsRect()
    for button in (
        dialog.back_button,
        dialog.skip_button,
        dialog.pause_button,
        dialog.action_button,
    ):
        assert content_rect.contains(button.geometry())

    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def test_reduced_motion_lowers_nonessential_meter_refresh(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: None)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    assert window.meter_timer.interval() == 100
    window.processor.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()
