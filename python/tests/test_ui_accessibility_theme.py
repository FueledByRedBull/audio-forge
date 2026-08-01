# pyright: reportAttributeAccessIssue=false
"""Deterministic accessibility and semantic-theme contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea

from mic_eq.config import AppConfig
from mic_eq.ui.accessibility import audit_widget_tree, set_accessible
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.theme import (
    TEXT_CONTRAST_PAIRS,
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
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        window.content_scroll_area.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        window.eq_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    window.resize(900, 640)
    window.show()
    qapp.processEvents()
    assert window.width() == 900
    assert window.height() == 640


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
