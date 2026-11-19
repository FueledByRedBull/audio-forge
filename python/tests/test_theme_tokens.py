"""The shared design tokens: type floor, motion durations, glass and the focus ring."""

from __future__ import annotations

from dataclasses import asdict

import pytest
from PySide6.QtCore import QByteArray, QUrl
from PySide6.QtGui import QColor, QFont
from PySide6.QtQml import QQmlComponent, QQmlEngine

from mic_eq.ui.quick_scene import QML_DIR, quick_theme
from mic_eq.ui.theme import (
    MOTION,
    PALETTE,
    REDUCED_MOTION_FADE_MS,
    TYPE_FLOOR_PX,
    TYPE_SCALE_PX,
    TYPE_TICK_PX,
    application_stylesheet,
    build_palette,
    contrast_ratio,
    flatten,
    motion_durations,
)

SYSTEM_COLORS = {
    "window": "#102030", "window_text": "#f0e0d0", "gray_text": "#80ff80",
    "highlight": "#ffee00", "highlight_text": "#000011",
}
SIZES = ("fsTick", "fsCaption", "fsBody", "fsLead", "fsHeading", "fsTitle", "fsDisplay")


def _look_sizes(qapp, point_size: float) -> dict[str, int]:
    """Look's text sizes, read in a scene loaded beside it, while the app font has ``point_size``."""

    font = QFont(qapp.font())
    previous = QFont(qapp.font())
    font.setPointSizeF(point_size)
    qapp.setFont(font)
    try:
        engine = QQmlEngine()
        engine.rootContext().setContextProperty("theme", quick_theme())
        component = QQmlComponent(engine)
        source = "import QtQuick\nQtObject {\n" + "".join(
            f"    readonly property int {name}: Look.{name}\n" for name in SIZES
        ) + "}\n"
        component.setData(QByteArray(source.encode()), QUrl.fromLocalFile(str(QML_DIR / "SizeProbe.qml")))
        probe = component.create()
        assert probe is not None, [error.toString() for error in component.errors()]
        sizes = {name: probe.property(name) for name in SIZES}
        probe.deleteLater()
        engine.deleteLater()
        return sizes
    finally:
        qapp.setFont(previous)


def test_text_sizes_follow_the_scale_at_the_default_app_font(qapp) -> None:
    sizes = _look_sizes(qapp, 9)
    assert sizes == {
        "fsTick": TYPE_TICK_PX,
        "fsCaption": TYPE_SCALE_PX["caption"],
        "fsBody": TYPE_SCALE_PX["body"],
        "fsLead": TYPE_SCALE_PX["lead"],
        "fsHeading": TYPE_SCALE_PX["heading"],
        "fsTitle": TYPE_SCALE_PX["title"],
        "fsDisplay": TYPE_SCALE_PX["display"],
    }
    assert min(TYPE_SCALE_PX.values()) == TYPE_FLOOR_PX == 12 and TYPE_TICK_PX == 11


def test_a_small_app_font_never_takes_text_below_the_twelve_pixel_floor(qapp) -> None:
    sizes = _look_sizes(qapp, 5)
    assert sizes["fsTick"] == TYPE_TICK_PX
    assert all(sizes[name] >= TYPE_FLOOR_PX for name in SIZES if name != "fsTick")
    assert sizes["fsCaption"] == TYPE_FLOOR_PX
    # A larger app font still scales every size up, ticks included.
    large = _look_sizes(qapp, 18)
    assert large["fsBody"] == 26 and large["fsTick"] == 22


def test_scenes_get_the_motion_durations(qapp, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "0")
    theme = quick_theme()
    assert theme["reducedMotion"] is False
    assert (theme["motion"], theme["motionStatus"], theme["motionCrossfade"]) == (150, 150, 120)
    assert (theme["motionSwitch"], theme["motionRow"]) == (120, 120)
    assert (theme["motionPanelOpen"], theme["motionPanelClose"]) == (300, 200)
    assert theme["easePanelOpen"] == [0.16, 1.0, 0.3, 1.0, 1.0, 1.0]
    assert theme["easePanelClose"][-2:] == [1.0, 1.0]


def test_reduced_motion_swaps_every_duration_for_a_short_fade(qapp, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    theme = quick_theme()
    assert theme["reducedMotion"] is True
    assert theme["motion"] == 0
    fades = [theme[key] for key in theme if key.startswith("motion") and key != "motion"]
    assert len(fades) == len(asdict(MOTION)) - 1
    assert set(fades) == {REDUCED_MOTION_FADE_MS}
    assert set(asdict(motion_durations(True)).values()) == {REDUCED_MOTION_FADE_MS}
    assert motion_durations(False) == MOTION


def test_glass_is_translucent_in_the_dark_look_and_the_window_color_under_a_contrast_theme(qapp) -> None:
    glass_tokens = ("glass", "glass_solid", "glass_modal")
    assert all(0 < QColor(getattr(PALETTE, name)).alpha() < 255 for name in glass_tokens)
    contrast = build_palette(SYSTEM_COLORS)
    assert {getattr(contrast, name) for name in glass_tokens} == {SYSTEM_COLORS["window"]}
    assert QColor(contrast.glass_shadow).alpha() == 0
    theme = quick_theme()
    assert (theme["radiusGlass"], theme["glassBlur"], theme["glassModalBlur"]) == (22, 26, 40)


def test_flatten_lays_a_translucent_token_over_its_ground() -> None:
    assert flatten("#80ffffff", "#000000") == "#808080"
    assert flatten("#123456", "#ffffff") == "#123456"
    assert flatten(PALETTE.glass, PALETTE.app_surface) != PALETTE.app_surface
    with pytest.raises(ValueError, match="Invalid color token"):
        flatten("not-a-color", "#000000")


def test_the_focus_ring_has_its_own_token_in_scenes_and_widgets() -> None:
    ring = (QML_DIR / "FocusRing.qml").read_text(encoding="utf-8")
    assert "border.color: Look.t.focus_ring" in ring
    assert f"border-color: {PALETTE.focus_ring}" in application_stylesheet()
    assert contrast_ratio(PALETTE.focus_ring, PALETTE.app_surface) >= 3
    assert build_palette(SYSTEM_COLORS).focus_ring == SYSTEM_COLORS["highlight"]


def test_level_colors_are_not_told_apart_by_red_against_green_alone() -> None:
    # The safe end of a level is blue, not green: blue against red stays apart for a
    # red-green colour-blind user, and the clip mark adds its own shape.
    safe, danger = QColor(PALETTE.meter_safe), QColor(PALETTE.meter_danger)
    assert 190 <= safe.hslHue() <= 250
    assert safe.blue() > safe.green() and safe.blue() > safe.red()
    assert danger.hslHue() < 20 or danger.hslHue() > 340
