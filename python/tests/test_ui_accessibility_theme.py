# pyright: reportAttributeAccessIssue=false
"""Deterministic accessibility and semantic-theme contract tests."""

from __future__ import annotations

import ctypes
import re
import sys
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QRect, QSize
from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import QScrollArea, QWidget

from mic_eq.config import AppConfig
from mic_eq.ui.accessibility import audit_widget_tree, set_accessible
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.first_run_setup_dialog import FirstRunSetupDialog
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog
from mic_eq.ui.main_window import (
    HEALTH_DECISION_KEYS,
    HEALTH_LAYOUT_KEYS,
    MainWindow,
    _fit_window_geometry_to_screens,
)
from mic_eq.ui.level_meter import SCALE_TEXT_GAP, _scale_mark_geometry
from mic_eq.ui.shell_specs import TEXT_SPECS
from mic_eq.ui import theme
from mic_eq.ui.theme import (
    CONTROL_BOUNDARY_PAIRS,
    PALETTE,
    TEXT_CONTRAST_PAIRS,
    SemanticPalette,
    application_palette,
    application_stylesheet,
    build_palette,
    contrast_ratio,
    contrast_theme_changed,
    control_boundary_pairs,
    muted_icon,
    prefers_reduced_motion,
    qcolor,
    windows_contrast_colors,
)
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog


UI_ROOT = Path(__file__).resolve().parents[1] / "mic_eq" / "ui"
LITERAL_COLOR = re.compile(r"#[0-9A-Fa-f]{6,8}|QColor\s*\(")
PIXEL_FONT = re.compile(r"font-size\s*:\s*[0-9.]+px")


@pytest.mark.parametrize("states, mute_error, expected", [
    (("idle", "idle"), False, "idle"),
    (("idle", "ok"), False, "ok"),
    (("ok", "warn"), False, "warn"),
    (("warn", "bad"), True, "bad"),
    (("idle", "idle"), True, "warn"),
])
def test_compact_health_aggregation_and_stop(isolated_main_window, states, mute_error, expected):
    window = isolated_main_window
    keys = HEALTH_DECISION_KEYS + HEALTH_LAYOUT_KEYS
    for key in keys:
        window._set_health_chip(key, key, "idle")
    for key, state in zip(keys, states):
        window._set_health_chip(key, key, state)
    window._output_mute_error = mute_error
    window._update_health_summary()
    assert window.shell_texts["health_summary_label"].state == expected
    window._stop_processing()
    window._update_meters()
    assert window.shell_texts["health_summary_label"].text == "Health: --"
    assert window.shell_texts["health_summary_label"].state == "idle"
    # Nothing the last reading said about itself is left behind.
    assert window.shell_texts["health_advice_label"].text == TEXT_SPECS["health_advice_label"].text
    assert window.shell_texts["health_summary_label"].toolTip == TEXT_SPECS["health_summary_label"].tip


def test_all_semantic_text_pairs_meet_wcag_aa_contrast() -> None:
    measured = {
        name: contrast_ratio(foreground, background)
        for name, foreground, background in TEXT_CONTRAST_PAIRS
    }
    assert measured
    assert min(measured.values()) >= 4.5, measured


def test_control_edges_switch_tracks_and_slider_tracks_meet_three_to_one() -> None:
    measured = {name: contrast_ratio(fg, bg) for name, fg, bg in CONTROL_BOUNDARY_PAIRS}
    assert measured and min(measured.values()) >= 3.0, measured
    # The edge also stands out from a field while it is hovered, and the knob from its off track.
    assert contrast_ratio(PALETTE.control_edge, PALETTE.control_hover) >= 3.0
    assert contrast_ratio(PALETTE.text_primary, PALETTE.control_edge) >= 3.0
    # The widgets' stylesheet draws its edges and slider grooves from the same token.
    sheet = application_stylesheet()
    assert f"border: 1px solid {PALETTE.control_edge}" in sheet
    assert f"background-color: {PALETTE.control_edge}" in sheet


def test_control_boundaries_keep_three_to_one_under_a_contrast_theme() -> None:
    pairs = control_boundary_pairs(build_palette(SYSTEM_COLORS))
    assert min(contrast_ratio(fg, bg) for _name, fg, bg in pairs) >= 3.0


def test_the_muted_tray_icon_crosses_out_a_corner_and_keeps_the_rest(qapp) -> None:
    base = QPixmap(32, 32)
    base.fill(QColor(PALETTE.card_surface))
    icon = QIcon(base)
    muted = muted_icon(icon)
    assert muted.availableSizes() == icon.availableSizes()
    before = icon.pixmap(QSize(32, 32)).toImage()
    after = muted.pixmap(QSize(32, 32)).toImage()
    assert after.pixelColor(2, 2) == before.pixelColor(2, 2)
    # The badge is not colour alone: a light slash crosses its fill.
    assert after.pixelColor(22, 28).name() == PALETTE.action_destructive
    assert after.pixelColor(22, 22).red() > 200
    assert muted_icon(QIcon()).isNull()


def test_application_palette_uses_the_same_semantic_surface() -> None:
    palette = application_palette()
    assert palette.color(QPalette.ColorRole.Window).name() == PALETTE.app_surface
    assert palette.color(QPalette.ColorRole.WindowText).name() == PALETTE.text_primary
    assert palette.color(QPalette.ColorRole.Text).name() == PALETTE.text_primary
    assert palette.color(QPalette.ColorRole.Highlight).name() == PALETTE.action_primary


def test_larger_default_geometry_fits_screen_and_preserves_saved_size() -> None:
    screen = QRect(0, 0, 1920, 1040)
    assert _fit_window_geometry_to_screens(None, [screen]) == QRect(240, 40, 1440, 960)
    small_screen = QRect(0, 0, 1280, 720)
    assert _fit_window_geometry_to_screens(None, [small_screen]) == small_screen
    saved = {"x": 50, "y": 60, "width": 1100, "height": 740}
    assert _fit_window_geometry_to_screens(saved, [screen]) == QRect(50, 60, 1100, 740)


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


def test_maximized_window_saves_its_restored_geometry_and_reopens_maximized(
    isolated_main_window,
    qapp,
    monkeypatch,
) -> None:
    window = isolated_main_window
    window.show()
    window.setGeometry(40, 50, 900, 640)
    qapp.processEvents()
    normal = window.geometry()
    window.showMaximized()
    qapp.processEvents()
    assert window.isMaximized()
    window._quitting = True  # a real quit, not close-to-tray
    window.close()
    assert window.config.window_geometry == {
        "x": normal.x(),
        "y": normal.y(),
        "width": normal.width(),
        "height": normal.height(),
        "maximized": True,
    }

    saved = AppConfig.from_dict(window.config.to_dict())
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", lambda: saved)
    reopened = MainWindow()
    try:
        assert reopened.isMaximized()
    finally:
        reopened.close()
        reopened.deleteLater()


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


SYSTEM_COLORS = {
    "window": "#102030", "window_text": "#f0e0d0", "gray_text": "#80ff80",
    "highlight": "#ffee00", "highlight_text": "#000011",
}


def test_without_a_contrast_theme_the_dark_look_is_unchanged() -> None:
    dark = build_palette(None)
    assert dark == SemanticPalette()
    assert (dark.app_surface, dark.text_primary, dark.accent) == ("#14171c", "#eef1f5", "#f2743a")
    assert dark.hover_text == dark.text_primary


def test_a_contrast_theme_takes_every_surface_text_and_accent_from_the_system() -> None:
    palette = build_palette(SYSTEM_COLORS)
    surfaces = [
        palette.app_surface, palette.rail_surface, palette.card_surface, palette.control_surface,
        palette.data_surface, palette.status_ok_surface, palette.status_bad_surface,
        palette.action_secondary,
    ]
    assert set(surfaces) == {SYSTEM_COLORS["window"]}
    assert {palette.text_primary, palette.text_muted, palette.data_text, palette.border} == {
        SYSTEM_COLORS["window_text"]
    }
    assert palette.action_disabled_text == SYSTEM_COLORS["gray_text"]
    assert (palette.accent, palette.action_primary, palette.control_hover) == (SYSTEM_COLORS["highlight"],) * 3
    assert (palette.text_on_accent, palette.hover_text) == (SYSTEM_COLORS["highlight_text"],) * 2
    assert contrast_ratio(palette.text_primary, palette.app_surface) > 10
    assert "hover_text" in asdict(palette)  # the QML tokens come from this mapping


# Data colors a contrast theme keeps on purpose; every other token must follow the system.
KEPT_UNDER_CONTRAST_THEME = {
    "eq_band_colors", "text_on_data_color",
    "meter_safe", "meter_caution", "meter_danger", "meter_clip", "meter_reduction",
}


def test_every_color_token_follows_the_contrast_theme_or_is_kept_on_purpose() -> None:
    dark, contrast = SemanticPalette(), build_palette(SYSTEM_COLORS)
    followed = {f.name for f in fields(dark) if getattr(contrast, f.name) != getattr(dark, f.name)}
    assert {f.name for f in fields(dark)} - followed == KEPT_UNDER_CONTRAST_THEME


def test_a_white_contrast_theme_keeps_the_peak_line_and_band_numbers_readable() -> None:
    palette = build_palette({
        "window": "#ffffff", "window_text": "#000000", "gray_text": "#600000",
        "highlight": "#37006e", "highlight_text": "#ffffff",
    })
    assert contrast_ratio(palette.meter_peak, palette.data_surface) >= 3
    assert all(
        contrast_ratio(palette.text_on_data_color, color) >= 4.5
        for color in (*palette.eq_band_colors, palette.meter_clip)
    )


def test_the_widget_palette_and_stylesheet_follow_the_contrast_theme(monkeypatch) -> None:
    monkeypatch.setattr(theme, "PALETTE", build_palette(SYSTEM_COLORS))
    palette = application_palette()
    assert palette.color(QPalette.ColorRole.Window).name() == SYSTEM_COLORS["window"]
    assert palette.color(QPalette.ColorRole.WindowText).name() == SYSTEM_COLORS["window_text"]
    assert palette.color(QPalette.ColorRole.Highlight).name() == SYSTEM_COLORS["highlight"]
    stylesheet = application_stylesheet()
    assert SYSTEM_COLORS["window_text"] in stylesheet
    assert SemanticPalette().app_surface not in stylesheet and SemanticPalette().accent not in stylesheet


def _fake_windows(monkeypatch, *, contrast_on: bool) -> None:
    def parameters(_action, _size, reference, _flags) -> int:
        reference._obj.dwFlags = 1 if contrast_on else 0
        return 1

    def sys_color(index: int) -> int:  # 0x00BBGGRR
        return {5: 0x302010, 8: 0xD0E0F0, 17: 0x80FF80, 13: 0x00EEFF, 14: 0x110000}[index]

    user32 = SimpleNamespace(SystemParametersInfoW=parameters, GetSysColor=sys_color)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=user32), raising=False)


def test_windows_contrast_colors_are_read_only_while_a_contrast_theme_is_on(monkeypatch) -> None:
    _fake_windows(monkeypatch, contrast_on=True)
    assert windows_contrast_colors() == SYSTEM_COLORS
    _fake_windows(monkeypatch, contrast_on=False)
    assert windows_contrast_colors() is None
    monkeypatch.setattr(sys, "platform", "linux")
    assert windows_contrast_colors() is None


def test_a_changed_contrast_theme_asks_for_a_restart(isolated_main_window, monkeypatch) -> None:
    window = isolated_main_window
    monkeypatch.setattr(theme, "SYSTEM_CONTRAST_COLORS", None)
    monkeypatch.setattr(theme, "windows_contrast_colors", lambda: None)
    assert not contrast_theme_changed()
    window.status_message.clearMessage()
    window.changeEvent(QEvent(QEvent.Type.ThemeChange))
    assert window.status_message.text == ""
    monkeypatch.setattr(theme, "windows_contrast_colors", lambda: SYSTEM_COLORS)
    assert contrast_theme_changed()
    window.changeEvent(QEvent(QEvent.Type.ThemeChange))
    assert "Restart AudioForge" in window.status_message.text


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


def test_workflow_dialogs_have_named_controls(
    isolated_main_window,
    qapp,
) -> None:
    window = isolated_main_window
    dialogs = (
        CalibrationDialog(window),
        VoiceSetupDialog(window),
        LatencyCalibrationDialog(window),
    )

    problems = {
        type(dialog).__name__: audit_widget_tree(dialog)
        for dialog in dialogs
        if audit_widget_tree(dialog)
    }
    assert problems == {}

    for dialog in dialogs:
        dialog.close()
        dialog.deleteLater()
    qapp.processEvents()


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


def test_reduced_motion_keeps_the_meters_at_their_normal_refresh(
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
    assert window.meter_timer.interval() == 16
    window.processor.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()
