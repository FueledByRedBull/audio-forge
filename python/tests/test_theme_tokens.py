"""The shared design tokens: type floor, motion durations, glass and the focus ring."""

from __future__ import annotations

import ast
from dataclasses import asdict, fields
from pathlib import Path
import re
import sys
import types

import pytest
from PySide6.QtCore import QByteArray, QUrl
from PySide6.QtGui import QColor, QFont
from PySide6.QtQml import QQmlComponent, QQmlEngine
from PySide6.QtWidgets import QMessageBox

from mic_eq.ui import theme
from mic_eq.ui.quick_scene import QML_DIR, quick_theme
from mic_eq.ui.theme import (
    MENU_HOVER_WHITE,
    MOTION,
    PALETTE,
    REDUCED_MOTION_FADE_MS,
    RESERVED_FOR_LATER,
    SemanticPalette,
    TYPE_FLOOR_PX,
    TYPE_SCALE_PX,
    TYPE_TICK_PX,
    application_stylesheet,
    themed_menu,
    build_palette,
    contrast_ratio,
    flatten,
    motion_durations,
    prefers_reduced_transparency,
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


def test_prefers_reduced_transparency_reads_env_override_and_registry(qapp, monkeypatch) -> None:
    values: dict[str, int] = {"EnableTransparency": 0}
    opened: list[str] = []

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def open_key(_root, path):
        opened.append(path)
        return Key()

    def query(_key, name):
        if name not in values:
            raise FileNotFoundError(name)
        return values[name], 4

    winreg = types.SimpleNamespace(HKEY_CURRENT_USER=object(), OpenKey=open_key, QueryValueEx=query)
    monkeypatch.setitem(sys.modules, "winreg", winreg)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("AUDIOFORGE_REDUCED_TRANSPARENCY", raising=False)
    assert prefers_reduced_transparency() is True
    assert opened == [r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"]
    assert quick_theme()["reducedTransparency"] is True
    values["EnableTransparency"] = 1
    assert prefers_reduced_transparency() is False
    # Windows writes the value only once the switch is used; until then transparency is on.
    del values["EnableTransparency"]
    assert prefers_reduced_transparency() is False

    monkeypatch.setenv("AUDIOFORGE_REDUCED_TRANSPARENCY", "1")
    assert prefers_reduced_transparency() is True
    values["EnableTransparency"] = 0
    monkeypatch.setenv("AUDIOFORGE_REDUCED_TRANSPARENCY", "0")
    assert prefers_reduced_transparency() is False
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("AUDIOFORGE_REDUCED_TRANSPARENCY")
    assert prefers_reduced_transparency() is False


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


def _scene_readers(names: list[str]) -> set[str]:
    """Tokens a scene reads: ``t.name``, or ``t["head" + x + "tail"]`` for every name it can make."""

    source = "\n".join(path.read_text(encoding="utf-8") for path in QML_DIR.glob("*.qml"))
    read = {name for name in names if re.search(rf"\bt\.{name}\b", source)}
    for head, tail in re.findall(r'\bt\["(\w+)"\s*\+[^\]]*?\+\s*"(\w+)"\]', source):
        read |= {n for n in names if n.startswith(head) and n.endswith(tail) and len(n) > len(head + tail)}
    return read


def _is_described_not_read(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True inside theme.py's palette definition, contrast theme builder and contrast tables."""

    described_functions = {"build_palette", "control_boundary_pairs", "SemanticPalette"}
    tables = {"TEXT_CONTRAST_PAIRS", "CONTROL_BOUNDARY_PAIRS"}
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in described_functions:
            return True
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in tables for target in targets):
                return True
    return False


def _python_readers(names: list[str]) -> set[str]:
    """Tokens the app's Python code reads as an attribute, outside theme.py's own descriptions."""

    package = Path(theme.__file__).parent.parent
    read: set[str] = set()
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in names:
                if path.name != "theme.py" or not _is_described_not_read(node, parents):
                    read.add(node.attr)
    return read


def test_every_palette_token_has_a_reader() -> None:
    names = [field.name for field in fields(SemanticPalette)]
    read = _scene_readers(names) | _python_readers(names)
    assert set(RESERVED_FOR_LATER) <= set(names)
    # A token that gains a reader leaves the reserved list, so the list never goes stale.
    assert not set(RESERVED_FOR_LATER) & read, sorted(set(RESERVED_FOR_LATER) & read)
    assert sorted(set(names) - read - set(RESERVED_FOR_LATER)) == []
    assert "analyser" not in names


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


def test_the_meter_caution_step_reads_its_own_token_and_keeps_its_hue_under_a_contrast_theme() -> None:
    from mic_eq.ui import level_meter

    source = Path(level_meter.__file__).read_text(encoding="utf-8")
    assert "status_warn" not in source and "data_warning" not in source
    assert level_meter.LevelMeter.COLOR_YELLOW == QColor(PALETTE.meter_caution)
    contrast = build_palette(SYSTEM_COLORS)
    assert contrast.meter_caution == PALETTE.meter_caution
    caution, danger = QColor(contrast.meter_caution), QColor(contrast.meter_danger)
    assert min(abs(caution.hslHue() - danger.hslHue()), 360 - abs(caution.hslHue() - danger.hslHue())) >= 30


def test_callers_affected_has_its_own_hue_and_stays_readable() -> None:
    mark = QColor(PALETTE.callers_affected)

    def hue_gap(other: str) -> int:
        gap = abs(mark.hslHue() - QColor(other).hslHue())
        return min(gap, 360 - gap)

    # Not the amber of actions, the yellow of warnings, the red of errors, the violet of
    # Mute or the blues of the level traces.
    for taken in (
        PALETTE.accent, PALETTE.status_warn_text, PALETTE.status_bad_text, PALETTE.mute,
        PALETTE.signal_in, PALETTE.signal_in_fill, PALETTE.status_ok_text,
    ):
        assert hue_gap(taken) >= 30, taken
    for ground in (PALETTE.pane_surface, PALETTE.card_surface, PALETTE.sidebar_top):
        assert contrast_ratio(PALETTE.callers_affected_text, ground) >= 4.5
        assert contrast_ratio(PALETTE.callers_affected, ground) >= 3
    fill = flatten(PALETTE.callers_affected_fill, PALETTE.pane_surface)
    assert contrast_ratio(PALETTE.callers_affected_text, fill) >= 4.5
    contrast = build_palette(SYSTEM_COLORS)
    assert contrast.callers_affected == contrast.callers_affected_text == SYSTEM_COLORS["window_text"]


def test_a_switch_that_is_on_differs_from_one_that_is_off_by_more_than_hue() -> None:
    # On is a filled light track and off an outline in the edge colour: a colour-blind user
    # tells them apart by lightness and fill, and the on track holds 3:1 where switches sit.
    on, off = QColor(PALETTE.switch_on), QColor(PALETTE.control_edge)
    assert on.lightness() - off.lightness() >= 60
    for ground in (PALETTE.app_surface, PALETTE.sidebar_top, PALETTE.card_surface):
        assert contrast_ratio(PALETTE.switch_on, ground) >= 3
    assert build_palette(SYSTEM_COLORS).switch_on == SYSTEM_COLORS["window_text"]


@pytest.mark.parametrize("name", ["Main.qml", "SwitchButton.qml"])
def test_scene_animations_take_their_length_from_the_motion_tokens(name) -> None:
    durations = re.findall(r"duration:\s*([^;}\n]+)", (QML_DIR / name).read_text(encoding="utf-8"))
    assert durations
    assert all(re.search(r"\bt\.motion\w*\b", value) for value in durations), durations


def test_new_band_colors_keep_band_numbers_readable() -> None:
    assert all(
        contrast_ratio(PALETTE.text_on_data_color, color) >= 4.5 for color in PALETTE.eq_band_colors
    )


def test_adjacent_band_colors_differ_in_hue() -> None:
    hues = [QColor(color).hsvHueF() * 360 for color in PALETTE.eq_band_colors]
    for left, right in zip(hues, hues[1:]):
        gap = abs(left - right)
        assert min(gap, 360 - gap) >= 20, (left, right)


def test_message_boxes_and_menus_use_the_theme_surfaces(qapp) -> None:
    sheet = application_stylesheet()
    menu_ground = flatten(PALETTE.glass_solid, PALETTE.app_surface)
    default_edge = flatten(PALETTE.accent_edge, PALETTE.pane_surface)
    assert f"QDialog {{ background-color: {PALETTE.pane_surface}" in sheet
    assert "QDialog QLabel { font-size: 13px" in sheet
    assert f"QDialog QPushButton:default {{ border-color: {default_edge}; }}" in sheet
    assert f"background-color: {menu_ground}; color: {PALETTE.text_primary}" in sheet
    assert "border-radius: 16px; padding: 6px;" in sheet
    assert "QMenu::indicator { width: 16px; height: 16px;" in sheet
    assert f"QMenu::separator {{ height: 1px; background-color: {flatten(PALETTE.glass_edge, menu_ground)}" in sheet

    box = QMessageBox(QMessageBox.Icon.Question, "Quit", "Quit now?", QMessageBox.StandardButton.Yes)
    box.show()
    qapp.processEvents()
    assert box.grab().toImage().pixelColor(2, 2).name() == PALETTE.pane_surface
    assert [button.height() for button in box.buttons()] == [36]
    box.close()

    menu = themed_menu()
    menu.addAction("First")
    menu.addAction("Second")
    menu.show()
    menu.setActiveAction(menu.actions()[0])
    qapp.processEvents()
    image = menu.grab().toImage()
    row = menu.actionGeometry(menu.actions()[0])
    assert image.pixelColor(image.width() // 2, 3).name() == menu_ground
    # The rounded corner is see-through rather than a square of window ground.
    assert image.pixelColor(0, 0).alpha() == 0
    assert image.pixelColor(row.left() + 4, row.center().y()).name() == flatten(MENU_HOVER_WHITE, menu_ground)
    menu.close()


def test_the_widget_dialogs_and_menus_still_follow_the_contrast_theme(monkeypatch) -> None:
    dark = SemanticPalette()
    monkeypatch.setattr(theme, "PALETTE", build_palette(SYSTEM_COLORS))
    sheet = application_stylesheet()
    for name in ("pane_surface", "app_surface", "accent_edge", "control_hover"):
        assert getattr(dark, name) not in sheet, name
    assert flatten(dark.glass_solid, dark.app_surface) not in sheet
    assert f"QDialog {{ background-color: {SYSTEM_COLORS['window']}" in sheet
    assert f"QMenu::item:selected {{ background-color: {SYSTEM_COLORS['highlight']}" in sheet
