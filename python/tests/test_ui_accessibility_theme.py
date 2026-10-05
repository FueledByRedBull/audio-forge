# pyright: reportAttributeAccessIssue=false
"""Deterministic accessibility and semantic-theme contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget

from mic_eq.config import AppConfig
from mic_eq.ui.accessibility import audit_widget_tree, set_accessible
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.first_run_setup_dialog import FirstRunSetupDialog
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog
from mic_eq.ui.main_window import MainWindow, _fit_window_geometry_to_screens
from mic_eq.ui.level_meter import SCALE_TEXT_GAP, _scale_mark_geometry
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
def widget_view(monkeypatch):
    """Layout tests that measure the widget view, which remains the fallback."""

    monkeypatch.setenv("AUDIOFORGE_QML", "0")


@pytest.fixture
def isolated_main_window(qapp, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
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


@pytest.mark.parametrize("states, mute_error, expected", [
    (("idle", "idle"), False, "idle"),
    (("idle", "ok"), False, "ok"),
    (("ok", "warn"), False, "warn"),
    (("warn", "bad"), True, "bad"),
    (("idle", "idle"), True, "warn"),
])
def test_compact_health_aggregation_and_stop(isolated_main_window, states, mute_error, expected):
    window = isolated_main_window
    labels = tuple(QLabel(window) for _ in states)
    for label, state in zip(labels, states):
        label.setProperty("health_state", state)
    window._health_decision_widgets = labels
    window._health_layout_widgets = ()
    window._output_mute_error = mute_error
    window._update_diagnostic_labels(
        diagnostics={}, latency_ms=0, dsp_time_ms=0,
        input_buf=0, output_buf=0, rnnoise_buf=0,
    )
    assert window.health_summary_label.property("health_state") == expected
    window._stop_processing()
    window._update_meters()
    assert window.health_summary_label.text() == "Health: --"
    assert window.health_summary_label.property("health_state") == "idle"


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


def test_level_meter_scale_text_is_separated_from_its_tick() -> None:
    tick_end, text_rect = _scale_mark_geometry(20, 12.0, 28)

    assert text_rect.left() - tick_end == SCALE_TEXT_GAP
    assert text_rect.right() <= 48


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
    widget_view,
    isolated_main_window,
    qapp,
) -> None:
    window = isolated_main_window
    expected_top_row = (
        window.output_combo,
        window.refresh_btn,
        window.processing_mode_combo,
        window.user_mute_checkbox,
    )
    current = window.input_combo
    for expected in expected_top_row:
        current = current.nextInFocusChain()
        while not current.focusPolicy() & Qt.FocusPolicy.TabFocus:
            current = current.nextInFocusChain()
        assert current is expected

    assert window.minimumWidth() <= 900
    assert window.minimumHeight() <= 640
    assert (
        window.content_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert (
        window.content_scroll_area.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    window.resize(900, 640)
    window.show()
    qapp.processEvents()
    qapp.processEvents()
    assert window.width() == 900
    assert window.height() == 640
    assert window._responsive_layout_compact is True
    assert window.content_scroll_area.horizontalScrollBar().maximum() == 0
    assert (
        window.content_scroll_area.widget().width()
        <= window.content_scroll_area.viewport().width()
    )
    # The transport control sits in the fixed top bar, never behind a scroll.
    assert window.start_btn.isVisible()
    assert window.stop_btn.isHidden()
    assert window.rect().contains(
        window.start_btn.mapTo(window, window.start_btn.rect().bottomRight())
    )


def test_default_window_opens_on_the_mic_page_and_reaches_health(
    widget_view,
    isolated_main_window,
    qapp,
) -> None:
    window = isolated_main_window
    window.resize(1280, 850)
    window.show()
    qapp.processEvents()
    qapp.processEvents()

    assert window._responsive_layout_compact is False
    assert window.menuBar().isHidden()
    assert window.content_scroll_area.horizontalScrollBar().maximum() == 0
    assert window.page_stack.currentIndex() == 0
    assert window.health_details.isHidden()
    assert window.rect().contains(window.health_details_button.mapTo(
        window, window.health_details_button.rect().bottomRight()
    ))
    window.health_details_button.click()
    qapp.processEvents()
    assert window.health_details.isVisible()
    assert window.dropped_label.isVisible()
    assert window.nav_group.checkedId() == window.HEALTH_PAGE_INDEX
    window.health_details_button.click()
    assert window.health_details.isHidden()
    assert window.nav_group.checkedId() == 0


@pytest.mark.parametrize(
    ("width", "height", "compact"),
    (
        (900, 640, True),
        (1024, 700, True),
        (1199, 760, True),
        (1280, 800, False),
        (1600, 900, False),
        (1920, 1040, False),
    ),
)
def test_main_window_has_no_horizontal_overflow_across_breakpoints(
    widget_view,
    isolated_main_window,
    qapp,
    width: int,
    height: int,
    compact: bool,
) -> None:
    window = isolated_main_window
    window.resize(width, height)
    window.show()
    qapp.processEvents()
    qapp.processEvents()

    assert window.width() == width
    assert window._responsive_layout_compact is compact
    for page_index in range(window.page_stack.count()):
        window.page_stack.setCurrentIndex(page_index)
        qapp.processEvents()
        page = window.page_stack.currentWidget()
        assert isinstance(page, QScrollArea)
        body = page.widget()
        assert body is not None
        oversized = [
            (
                type(widget).__name__,
                getattr(widget, "title", lambda: widget.objectName())(),
                widget.minimumSizeHint().width(),
            )
            for widget in body.findChildren(QWidget)
            if widget.minimumSizeHint().width() > page.viewport().width()
        ]
        assert page.horizontalScrollBar().maximum() == 0, (page_index, oversized)
        assert body.width() <= page.viewport().width(), (page_index, oversized)


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


@pytest.mark.parametrize("step", ("devices", "route", "voice"))
def test_first_run_setup_buttons_fit_the_minimum_dialog(
    isolated_main_window,
    qapp,
    monkeypatch,
    step,
) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config",
        lambda _config: True,
    )
    isolated_main_window.config.first_run_setup_step = step
    dialog = FirstRunSetupDialog(isolated_main_window)
    dialog.resize(440, 340)
    dialog.show()
    qapp.processEvents()
    qapp.processEvents()

    scrollbar = dialog.content_scroll_area.horizontalScrollBar()
    assert scrollbar is not None
    assert scrollbar.maximum() == 0
    # The action row sits below the scrolling page, so it cannot scroll into view.
    assert (dialog.width(), dialog.height()) == (440, 340)
    for button in (
        dialog.back_button,
        dialog.skip_button,
        dialog.pause_button,
        dialog.action_button,
    ):
        assert dialog.rect().contains(button.geometry())

    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def test_reduced_motion_lowers_nonessential_meter_refresh(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    assert window.meter_timer.interval() == 100
    window.processor.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()
