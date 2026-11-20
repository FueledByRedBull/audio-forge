"""The shared design tokens: type floor, motion durations, glass and the focus ring."""

from __future__ import annotations

import ast
from dataclasses import asdict, fields
from pathlib import Path
import re
import sys
import types
from typing import cast

import pytest
from PySide6.QtCore import QByteArray, QUrl
from PySide6.QtGui import QColor, QFont
from PySide6.QtQml import QQmlComponent, QQmlEngine
from PySide6.QtQuick import QQuickItem
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
    control_boundary_pairs,
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
    assert (theme["motionColor"], theme["motionStatus"], theme["motionCrossfade"]) == (150, 150, 120)
    assert theme["motionBusy"] == 900
    assert (theme["motionSwitch"], theme["motionRow"]) == (120, 120)
    assert (theme["motionPanelOpen"], theme["motionPanelClose"]) == (300, 200)
    assert theme["easePanelOpen"] == [0.16, 1.0, 0.3, 1.0, 1.0, 1.0]
    assert theme["easePanelClose"][-2:] == [1.0, 1.0]


def test_reduced_motion_swaps_every_duration_for_a_short_fade(qapp, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    theme = quick_theme()
    assert theme["reducedMotion"] is True
    # Colour and opacity fade briefly; what moves takes 0 ms through Look.movement, and the
    # busy sweep stops, so its period is not a fade.
    assert theme["motionColor"] == REDUCED_MOTION_FADE_MS
    fades = [theme[key] for key in theme if key.startswith("motion") and key != "motionBusy"]
    assert len(fades) == len(asdict(MOTION)) - 1
    assert set(fades) == {REDUCED_MOTION_FADE_MS}
    assert theme["motionBusy"] == MOTION.busy
    assert {n: v for n, v in asdict(motion_durations(True)).items() if n != "busy"} == {
        n: REDUCED_MOTION_FADE_MS for n in asdict(MOTION) if n != "busy"
    }
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


def test_the_meter_caution_step_reads_its_own_token_and_follows_a_contrast_theme() -> None:
    from mic_eq.ui import level_meter

    source = Path(level_meter.__file__).read_text(encoding="utf-8")
    assert "status_warn" not in source and "data_warning" not in source
    assert level_meter.LevelMeter.COLOR_YELLOW == QColor(PALETTE.meter_caution)
    caution, danger = QColor(PALETTE.meter_caution), QColor(PALETTE.meter_danger)
    assert min(abs(caution.hslHue() - danger.hslHue()), 360 - abs(caution.hslHue() - danger.hslHue())) >= 30
    # The dark look's pastels fail 3:1 on a light window, so a contrast theme gives the steps
    # its own colors: Highlight for the low end and the reduction bar, WindowText above.
    contrast = build_palette(SYSTEM_COLORS)
    assert contrast.meter_safe == contrast.meter_reduction == SYSTEM_COLORS["highlight"]
    assert contrast.meter_caution == contrast.meter_danger == contrast.meter_clip == SYSTEM_COLORS["window_text"]


# The colors Windows gives its black and white contrast themes.
HC_BLACK = {
    "window": "#000000", "window_text": "#ffffff", "gray_text": "#3ff23f",
    "highlight": "#1aebff", "highlight_text": "#000000",
}
HC_WHITE = {
    "window": "#ffffff", "window_text": "#000000", "gray_text": "#600000",
    "highlight": "#37006e", "highlight_text": "#ffffff",
}
CONTRAST_THEMES = {"HC Black": HC_BLACK, "HC White": HC_WHITE, "test colours": SYSTEM_COLORS}


@pytest.mark.parametrize("system", CONTRAST_THEMES.values(), ids=CONTRAST_THEMES.keys())
def test_each_level_meter_step_holds_three_to_one_on_its_track_under_a_contrast_theme(system) -> None:
    palette = build_palette(system)
    track = flatten(palette.meter_track, palette.pane_surface)
    for token in ("meter_safe", "meter_caution", "meter_danger", "meter_clip", "meter_reduction", "meter_peak"):
        assert contrast_ratio(getattr(palette, token), track) >= 3, token
    # The same pairs are in the boundary table, for the dark look and for every contrast theme.
    names = {name for name, _fg, _bg in control_boundary_pairs(palette)}
    assert {"meter safe step on its track", "meter clip strip on its track"} <= names


@pytest.mark.parametrize("system", CONTRAST_THEMES.values(), ids=CONTRAST_THEMES.keys())
def test_marks_hold_their_glyph_against_the_disc_in_every_theme(system) -> None:
    # The step's check glyph and the CLIP word sit on a mark that is WindowText in a contrast theme.
    palette = build_palette(system)
    assert palette.status_ok_dot == system["window_text"] and palette.meter_clip == system["window_text"]
    assert contrast_ratio(palette.text_on_mark, palette.status_ok_dot) >= 4.5
    assert contrast_ratio(palette.text_on_mark, palette.meter_clip) >= 4.5
    assert contrast_ratio(PALETTE.text_on_mark, PALETTE.status_ok_dot) >= 4.5
    assert PALETTE.text_on_mark == PALETTE.text_on_data_color


@pytest.mark.parametrize("system", [None, *CONTRAST_THEMES.values()], ids=["dark", *CONTRAST_THEMES.keys()])
def test_a_step_check_draws_its_glyph_in_the_colour_opposite_its_disc(qapp, monkeypatch, system) -> None:
    from mic_eq.ui import quick_scene

    palette = PALETTE if system is None else build_palette(system)
    monkeypatch.setattr(quick_scene, "PALETTE", palette)
    engine = QQmlEngine()
    engine.rootContext().setContextProperty("theme", quick_theme())
    component = QQmlComponent(engine)
    source = 'import QtQuick\nStepMark { shape: "check"; color: Look.t.status_ok_dot }\n'
    component.setData(QByteArray(source.encode()), QUrl.fromLocalFile(str(QML_DIR / "StepMarkProbe.qml")))
    mark = cast(QQuickItem, component.create())
    assert mark is not None, [error.toString() for error in component.errors()]
    pending, glyph = [mark], None
    while pending:
        item = pending.pop()
        pending.extend(item.childItems())
        if item.property("text") == "✓":
            glyph = item
    assert glyph is not None
    disc, ink = QColor(mark.property("color")).name(), QColor(glyph.property("color")).name()
    assert disc == palette.status_ok_dot and ink == palette.text_on_mark
    assert contrast_ratio(ink, disc) >= 4.5
    mark.deleteLater()
    engine.deleteLater()


@pytest.mark.parametrize("system", CONTRAST_THEMES.values(), ids=CONTRAST_THEMES.keys())
def test_the_selected_sidebar_row_and_its_ring_read_on_the_highlight_under_a_contrast_theme(system) -> None:
    palette = build_palette(system)
    fill = flatten(palette.accent_row_start, palette.sidebar_top)
    # The name, its detail line and the ring are the highlighted text colour; the ring is a mark (3:1).
    assert palette.hover_text == palette.accent_soft == system["highlight_text"]
    assert contrast_ratio(palette.hover_text, fill) >= 4.5
    assert contrast_ratio(palette.accent_soft, fill) >= 4.5
    # The focus colour is the fill itself there, so the ring keeps to the text colour on the fill.
    assert palette.focus_ring == fill == system["highlight"]


def test_the_selected_sidebar_row_keeps_its_dark_look() -> None:
    fill = flatten(PALETTE.accent_row_start, PALETTE.sidebar_top)
    assert PALETTE.hover_text == PALETTE.text_primary
    assert contrast_ratio(PALETTE.hover_text, fill) >= 4.5
    assert contrast_ratio(PALETTE.focus_ring, fill) >= 3


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


# Properties that move something; an animation on one takes 0 ms under reduced motion.
_MOVING = {"x", "y", "z", "scale", "rotation", "width", "height", "progress", "contentX", "contentY", "position"}
_QML_FILES = sorted(QML_DIR.glob("*.qml"))


def _timed_animations(source: str) -> list[tuple[str, str]]:
    """Every ``duration:`` in ``source`` as (the property it animates, its expression).

    The property is the ``property:`` of the animation block or the ``on <name>`` of its header
    or of an enclosing one; it is empty when neither says (a reusable animation type).
    """

    blocks: list[tuple[int, int, str]] = []
    opened: list[tuple[int, str]] = []
    for match in re.finditer(r"[{}]", source):
        if match.group() == "{":
            header = re.split(r"[{}\n]", source[: match.start()])[-1].strip()
            opened.append((match.start(), header))
        else:
            start, header = opened.pop()
            blocks.append((start, match.start(), header))
    found = []
    for duration in re.finditer(r"\bduration:\s*([^;}\r\n]+)", source):
        around = sorted((b for b in blocks if b[0] < duration.start() < b[1]), key=lambda b: b[0])
        start, end, _ = around[-1]
        body = source[start + 1 : end]
        while "{" in body:
            body = re.sub(r"\{[^{}]*\}", "", body)
        named = re.search(r'\bproperty:\s*"(\w+)"', body)
        target = named.group(1) if named else next(
            (m.group(1) for _, _, header in reversed(around) if (m := re.search(r"\bon\s+(\w+)$", header))), ""
        )
        found.append((target, duration.group(1).strip()))
    return found


def test_scene_animations_take_their_length_from_the_motion_tokens() -> None:
    timed = [
        (path.name, duration)
        for path in _QML_FILES
        for _, duration in _timed_animations(path.read_text(encoding="utf-8"))
    ]
    assert len(timed) >= 20
    # The meter's reading interval is the only length that is not a theme token.
    token = r"\bt\.motion[A-Z]\w*\b"
    assert [(name, d) for name, d in timed if not re.search(token, d) and "m.levelIntervalMs" not in d] == []
    assert {name for name, d in timed if "levelIntervalMs" in d} == {"Main.qml"}


def test_moving_animations_go_through_movement() -> None:
    moving = []
    for path in _QML_FILES:
        source = path.read_text(encoding="utf-8")
        for target, duration in _timed_animations(source):
            if target in _MOVING:
                moving.append((path.name, target, duration))
            elif target == "phase":
                # The busy sweep moves a block along its track; reduced motion stops it, and so does
                # hiding the bar.
                assert "running: bar.busy && bar.visible && !Look.reducedMotion" in source, path.name
    assert [item for item in moving if "Look.movement(" not in item[2]] == []
    # The inspector slide, the switch knob, the chevron, the fader handle and the ribbon are found.
    assert {target for _, target, _ in moving} >= {"x", "scale", "rotation", "progress"}


def test_no_scene_reads_the_retired_motion_key() -> None:
    retired = re.compile(r"\bt\.motion\b|\bt\[\"motion\"\]|\btheme\.motion\b")
    assert [path.name for path in _QML_FILES if retired.search(path.read_text(encoding="utf-8"))] == []
    assert "motion" not in quick_theme() and "motionColor" in quick_theme()


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


def test_the_top_bar_colours_follow_a_contrast_theme() -> None:
    contrast = build_palette(SYSTEM_COLORS)
    window, text = SYSTEM_COLORS["window"], SYSTEM_COLORS["window_text"]
    assert contrast.toolbar_control_top == contrast.toolbar_control_bottom == window
    assert contrast.toolbar_edge == text
    assert contrast.start_top == contrast.start_bottom == SYSTEM_COLORS["highlight"]
    assert (contrast.stop_top, contrast.stop_bottom, contrast.text_on_stop) == (text, text, window)
    assert contrast.brand_mark_glyph == SYSTEM_COLORS["highlight_text"]
    # Start and Stop differ by lightness as well as hue, for a colour-blind reader.
    assert abs(QColor(PALETTE.stop_bottom).lightness() - QColor(PALETTE.start_bottom).lightness()) >= 40
