"""Semantic visual tokens for AudioForge's explicit dark interface.

The token names describe intent rather than a particular widget.  Keeping the
palette here prevents dialogs, native Qt controls, and custom-painted widgets
from drifting apart across different Windows appearance settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
import sys

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPalette, QPen
from PySide6.QtWidgets import QApplication, QMenu, QWidget


@dataclass(frozen=True)
class SemanticPalette:
    """Colors used by the application chrome and its data canvases."""

    # Near-black inks in fine steps. Opaque tokens that stand for a translucent white
    # fill or hairline are that fill laid over the window ground, so contrast is measurable.
    app_surface: str = "#0f1013"
    pane_surface: str = "#0f1115"
    rail_surface: str = "#14161b"
    card_surface: str = "#18191d"
    toolbar_top: str = "#1b1c21"
    toolbar_bottom: str = "#15161a"
    sidebar_top: str = "#17181d"
    sidebar_bottom: str = "#131418"
    # The window ground. The sidebar and the main pane stand on it as raised, rounded
    # panels with a gap around them; their faint edge and lit top line say where they end.
    window_ground: str = "#07080a"
    panel_edge: str = "#0effffff"
    panel_highlight: str = "#13ffffff"
    pane_top: str = "#111319"
    pane_bottom: str = "#0c0d11"
    # A cool sheen across the top of the main pane.
    pane_glow: str = "#21788cdc"
    # A card on the main pane runs from card_top down to card_surface; the edge and the lit
    # top line are translucent white.
    card_top: str = "#202126"
    card_edge: str = "#13ffffff"
    card_highlight: str = "#17ffffff"
    # The tile behind a sidebar row's icon, and the icon drawn on it.
    tile_top: str = "#17ffffff"
    tile_bottom: str = "#08ffffff"
    tile_edge: str = "#1affffff"
    tile_icon: str = "#c9ced8"
    # The top bar is a rounded pane on the ground: a faint lit edge inside a dark rim.
    toolbar_edge: str = "#0effffff"
    toolbar_rim: str = "#8c000000"
    # A control in the top bar is a lighter step over the bar, top to bottom.
    toolbar_control_top: str = "#17ffffff"
    toolbar_control_bottom: str = "#0bffffff"
    # The app mark beside the name: an amber tile with a light waveform on it.
    brand_mark_top: str = "#f0ab72"
    brand_mark_bottom: str = "#c86d33"
    brand_mark_glyph: str = "#fff6ee"
    control_surface: str = "#1d1e21"
    control_surface_alt: str = "#191a1c"
    control_hover: str = "#27282b"
    border: str = "#202124"
    border_strong: str = "#2c2d2f"
    # The edge of a control, a switch track and a slider track: 3:1 against every surface
    # a control sits on (WCAG 1.4.11). The two borders above only divide surfaces.
    control_edge: str = "#7d818b"
    # A switch that is on: a neutral near-white track, so on and off differ by more than hue.
    switch_on: str = "#d9dbe1"
    focus_ring: str = "#cfe0ff"
    text_primary: str = "#ececf0"
    text_muted: str = "#9b9fa9"
    # The lowest text tier; it still carries meaning, so it keeps AA contrast.
    text_dim: str = "#8b8f99"
    text_on_emphasis: str = "#ececf0"
    text_on_accent: str = "#1b1005"
    # Dark text on a data color (EQ band, clip); a contrast theme keeps those colors.
    text_on_data_color: str = "#0f1013"
    # Text on a hovered or selected control; the dark look keeps the normal text color.
    hover_text: str = "#ececf0"
    # Amber is selection and the primary action, nothing else.
    accent: str = "#d98a4e"
    accent_hover: str = "#e29559"
    accent_pressed: str = "#c47c46"
    accent_text: str = "#e8a574"
    accent_soft: str = "#e8c5a6"
    accent_bright: str = "#ffd9b8"
    accent_fill: str = "#2ed98a4e"
    accent_edge: str = "#59d98a4e"
    # The selected sidebar row lights from its left end; its icon tile is amber too.
    accent_row_start: str = "#4dd98a4e"
    accent_row_end: str = "#1ad98a4e"
    accent_tile_top: str = "#57e8a574"
    accent_tile_bottom: str = "#1fe8a574"
    accent_tile_edge: str = "#80e8a574"
    # Violet is Mute and nothing else.
    mute: str = "#b9a3f5"
    mute_text: str = "#cdbcf8"
    text_on_mute: str = "#1a1230"
    mute_fill: str = "#29b9a3f5"
    mute_edge: str = "#73b9a3f5"
    # Rose-magenta says the call is not getting the voice; it is never the only sign,
    # since its words carry the meaning for colour-blind readers.
    callers_affected: str = "#e25a9a"
    callers_affected_text: str = "#f07ab4"
    callers_affected_fill: str = "#29e25a9a"
    callers_affected_edge: str = "#73e25a9a"

    action_primary: str = "#d98a4e"
    # Start and Stop in the top bar are lit from above. Stop while live is neutral: only the
    # action that begins something is amber.
    start_top: str = "#eea56b"
    start_bottom: str = "#d07c3e"
    stop_top: str = "#f3f4f7"
    stop_bottom: str = "#d9dbe2"
    text_on_stop: str = "#121318"
    # Laid over Start or Stop while the pointer is on it.
    transport_hover: str = "#14ffffff"
    # A button's "danger" look, for an action that ends or removes something: neutral, not amber.
    action_destructive: str = "#1d1e21"
    action_destructive_border: str = "#7d818b"
    action_secondary: str = "#1d1e21"
    action_disabled: str = "#2c2d2f"
    action_disabled_text: str = "#8b8f99"
    action_disabled_surface: str = "#191a1c"
    action_disabled_border: str = "#202124"

    status_ok_surface: str = "#172420"
    status_ok_text: str = "#8fcfa5"
    status_ok_border: str = "#254739"
    status_ok_dot: str = "#5fd39a"
    # The round mark behind a check that passed, or behind a note.
    status_ok_fill: str = "#298fcfa5"
    status_note_fill: str = "#14ffffff"
    status_warn_surface: str = "#26231a"
    status_warn_text: str = "#f5d97a"
    # Solid, not the soft edge: an empty picker is marked by it and needs 3:1.
    status_warn_border: str = "#f2cf5b"
    status_bad_surface: str = "#27191c"
    status_bad_text: str = "#ff8a80"
    status_bad_border: str = "#52292c"
    status_info_surface: str = "#1d1e21"
    status_info_text: str = "#ececf0"
    status_info_border: str = "#7d818b"
    status_idle_surface: str = "#1d1e21"
    status_idle_text: str = "#9b9fa9"
    status_idle_border: str = "#2c2d2f"

    data_surface: str = "#0c0d10"
    data_surface_raised: str = "#14161b"
    data_grid: str = "#242528"
    data_text: str = "#ececf0"
    data_text_muted: str = "#9b9fa9"
    data_curve: str = "#e6e8ee"
    data_marker: str = "#e8c5a6"
    data_warning: str = "#f2cf5b"
    # The ring around the selected EQ handle: the selection color.
    data_handle_selected: str = "#d98a4e"
    # The microphone (IN) and what callers hear (OUT), each one job only.
    signal_in: str = "#a9bbd4"
    signal_in_fill: str = "#7ea4dc"
    signal_out: str = "#e6e8ee"
    # The input analyser behind the EQ curves: a blue area that fades toward the floor, and its soft top edge.
    # Blue, so it stays apart from the curves for a red-green colour-blind eye.
    analyser_fill: str = "#5e9eff"
    analyser_line: str = "#b35e9eff"

    # Panels that dock beside live content; translucent over the ground behind them.
    glass: str = "#8c282c36"
    glass_solid: str = "#d122252e"
    glass_modal: str = "#b8242832"
    glass_edge: str = "#1affffff"
    glass_edge_modal: str = "#1fffffff"
    glass_highlight: str = "#12ffffff"
    glass_shadow: str = "#80000000"

    # One color per EQ band, in band order; data color, not accent.
    # Desaturated, so a band reads as data and not as a state; neighbours differ in hue.
    eq_band_colors: tuple[str, ...] = (
        "#a79ae0",
        "#8aa3dc",
        "#86b9cc",
        "#88c4a8",
        "#a8c98c",
        "#dad083",
        "#d9a974",
        "#d98f86",
        "#cf8aa8",
        "#b890cc",
    )

    # A level runs blue-grey, then yellow, then red at the hot end, so no step is told
    # from its neighbour by red against green.
    meter_safe: str = "#7ea4dc"
    meter_caution: str = "#f2cf5b"
    meter_danger: str = "#ff6b6b"
    meter_clip: str = "#ff6b6b"
    meter_peak: str = "#ececf0"
    meter_scale: str = "#9b9fa9"
    meter_reduction: str = "#a9bbd4"
    # The small bar under a sidebar row: how hard the stage works, from grey to near white.
    activity_track: str = "#14ffffff"
    activity_low: str = "#8a8f9a"
    activity_high: str = "#e8eaef"
    # The pill that says what an empty graph waits for.
    pill_surface: str = "#1b1f28"
    pill_edge: str = "#57ffffff"
    # A keycap in the shortcut row: a raised, lightly lit key with a bright label.
    key_top: str = "#24ffffff"
    key_bottom: str = "#0fffffff"
    key_edge: str = "#38ffffff"
    key_text: str = "#eef0f4"
    # The unlit steps of a segmented level bar.
    meter_segment_off: str = "#2a2c33"
    # The dark track behind the steps, ringed so the bar holds its shape against the pane.
    meter_track: str = "#73000000"
    meter_track_ring: str = "#0fffffff"


# Tokens with no reader yet, kept for work that is planned. test_every_palette_token_has_a_reader
# fails for any other token without a reader, and for an entry here once it gains one.
RESERVED_FOR_LATER: tuple[str, ...] = ()


def windows_contrast_colors() -> dict[str, str] | None:
    """Return the user's colors while a Windows contrast theme is on, else None."""

    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class HighContrast(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("dwFlags", ctypes.c_uint),
                ("lpszDefaultScheme", ctypes.c_wchar_p),
            ]

        info = HighContrast(ctypes.sizeof(HighContrast))
        spi_get_high_contrast = 0x0042
        hcf_high_contrast_on = 0x1
        user32 = ctypes.windll.user32
        if not user32.SystemParametersInfoW(
            spi_get_high_contrast, info.cbSize, ctypes.byref(info), 0
        ) or not info.dwFlags & hcf_high_contrast_on:
            return None

        def system_color(index: int) -> str:
            # GetSysColor returns 0x00BBGGRR.
            value = user32.GetSysColor(index)
            return f"#{value & 0xFF:02x}{value >> 8 & 0xFF:02x}{value >> 16 & 0xFF:02x}"

        color_window, color_window_text, color_gray_text = 5, 8, 17
        color_highlight, color_highlight_text = 13, 14
        return {
            "window": system_color(color_window),
            "window_text": system_color(color_window_text),
            "gray_text": system_color(color_gray_text),
            "highlight": system_color(color_highlight),
            "highlight_text": system_color(color_highlight_text),
        }
    except (AttributeError, OSError):
        return None


def build_palette(system: dict[str, str] | None) -> SemanticPalette:
    """Return the dark palette, or the user's contrast-theme colors when given."""

    if system is None:
        return SemanticPalette()
    window, text, disabled = system["window"], system["window_text"], system["gray_text"]
    accent, on_accent = system["highlight"], system["highlight_text"]
    # Surfaces are the window color and text the window text, so any pair stays readable;
    # a contrast theme has no muted tone, only GrayText for what cannot be used.
    return replace(
        SemanticPalette(),
        app_surface=window, rail_surface=window, card_surface=window,
        control_surface=window, control_surface_alt=window, control_hover=accent,
        border=text, border_strong=text,
        text_primary=text, text_muted=text, text_on_emphasis=window,
        text_on_accent=on_accent, hover_text=on_accent,
        accent=accent, accent_hover=accent, accent_pressed=accent,
        action_primary=accent, action_destructive=text, action_destructive_border=text,
        action_secondary=window, control_edge=text, switch_on=text,
        action_disabled=window, action_disabled_text=disabled,
        action_disabled_surface=window, action_disabled_border=disabled,
        status_ok_surface=window, status_ok_text=text, status_ok_border=text,
        status_warn_surface=window, status_warn_text=text, status_warn_border=text,
        status_bad_surface=window, status_bad_text=text, status_bad_border=text,
        status_info_surface=window, status_info_text=text, status_info_border=text,
        status_idle_surface=window, status_idle_text=text, status_idle_border=text,
        data_surface=window, data_surface_raised=window, data_grid=disabled,
        data_text=text, data_text_muted=text, data_curve=text,
        data_marker=accent, data_warning=accent, data_handle_selected=accent,
        analyser_fill=disabled, analyser_line=disabled,
        meter_scale=text, meter_peak=text, meter_segment_off=disabled, meter_track=window, meter_track_ring=text,
        toolbar_edge=text, toolbar_rim="#00000000",
        toolbar_control_top=window, toolbar_control_bottom=window,
        brand_mark_top=accent, brand_mark_bottom=accent, brand_mark_glyph=on_accent,
        start_top=accent, start_bottom=accent, stop_top=text, stop_bottom=text, text_on_stop=window,
        # Like the other buttons here, a contrast theme shows no hover tint on Start and Stop.
        transport_hover="#00000000",
        pane_surface=window, toolbar_top=window, toolbar_bottom=window,
        sidebar_top=window, sidebar_bottom=window, focus_ring=accent, text_dim=text,
        # Highlight is a fill colour in these themes, not one made to be read on the window.
        accent_text=text, accent_soft=on_accent, accent_bright=on_accent,
        accent_fill=accent, accent_edge=accent,
        mute=accent, text_on_mute=on_accent, mute_text=text, mute_fill=window, mute_edge=accent,
        callers_affected=text, callers_affected_text=text, callers_affected_fill=window,
        callers_affected_edge=text,
        status_ok_dot=text, signal_in=text, signal_in_fill=text, signal_out=text,
        glass=window, glass_solid=window, glass_modal=window,
        glass_edge=text, glass_edge_modal=text, glass_highlight=window, glass_shadow="#00000000",
        window_ground=window, panel_edge=text, panel_highlight=window,
        pane_top=window, pane_bottom=window, pane_glow="#00000000",
        card_top=window, card_edge=text, card_highlight=window,
        tile_top=window, tile_bottom=window, tile_edge=text, tile_icon=text,
        accent_row_start=accent, accent_row_end=accent,
        accent_tile_top=accent, accent_tile_bottom=accent, accent_tile_edge=accent,
        status_ok_fill=window, status_note_fill=window,
        activity_track=disabled, activity_low=text, activity_high=text,
        pill_surface=window, pill_edge=text,
        key_top=window, key_bottom=window, key_edge=text, key_text=text,
    )


# Read once at import: the styles below are built from it, so a contrast theme
# switched on while the app runs is picked up at the next start.
SYSTEM_CONTRAST_COLORS = windows_contrast_colors()
PALETTE = build_palette(SYSTEM_CONTRAST_COLORS)


def contrast_theme_changed() -> bool:
    """True when Windows contrast colors differ from the ones the styles were built with."""

    return windows_contrast_colors() != SYSTEM_CONTRAST_COLORS


RADIUS_CARD = 18
# The sidebar and the main pane.
RADIUS_PANEL = 20
RADIUS_CONTROL = 8
# A recessed well that holds a graph.
RADIUS_WELL = 14
# The top bar pane, and the controls in it.
RADIUS_TOOLBAR = 20
RADIUS_TOOLBAR_CONTROL = 12
RADIUS_MENU = 16
RADIUS_ITEM = 10
MENU_HOVER_WHITE = "#17ffffff"
UI_FONT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI")

# Text sizes in pixels at the default app font. Text that carries information is never
# below TYPE_FLOOR_PX; only axis ticks and keycaps use the smaller tick size.
TYPE_FLOOR_PX = 12
TYPE_TICK_PX = 11
TYPE_SCALE_PX = {"caption": 12, "body": 13, "lead": 15, "heading": 18, "title": 20, "display": 28}


def painter_font(
    px: int, *, weight: QFont.Weight | None = None, tracking: float = 0.0, floor: int = TYPE_FLOOR_PX
) -> QFont:
    """The app font at ``px`` for a QPainter, following the app font as the scenes' text does.

    The size scales with the app font (9 pt is the scale as written) and never drops below
    ``floor``; axis ticks pass ``floor=TYPE_TICK_PX``.
    """

    font = QFont(QGuiApplication.font())
    point_size = font.pointSizeF()
    zoom = point_size / 9 if point_size > 0 else 1.0
    font.setPixelSize(max(floor, round(px * zoom)))
    if weight is not None:
        font.setWeight(weight)
    if tracking:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking)
    return font


@dataclass(frozen=True)
class GlassRecipe:
    """How a docked glass panel is drawn; the colors are the palette's glass tokens."""

    radius: int = 22
    blur: int = 26
    saturation: float = 1.3
    modal_blur: int = 40
    modal_saturation: float = 1.4


GLASS = GlassRecipe()


@dataclass(frozen=True)
class MotionDurations:
    """Durations in milliseconds.

    Reduced motion swaps each for a short fade, except ``busy``, the period of a sweep that
    stops instead. Anything that moves takes 0 ms under reduced motion (``Look.movement``).
    """

    color: int = 150
    status: int = 150
    crossfade: int = 120
    switch: int = 120
    row: int = 120
    panel_open: int = 300
    panel_close: int = 200
    busy: int = 900


MOTION = MotionDurations()
REDUCED_MOTION_FADE_MS = 100
# Cubic-bezier control points: the panel decelerates as it opens and accelerates as it closes.
EASE_PANEL_OPEN = (0.16, 1.0, 0.3, 1.0)
EASE_PANEL_CLOSE = (0.42, 0.0, 1.0, 1.0)


def motion_durations(reduced: bool) -> MotionDurations:
    """The durations to use, with each a short fade while reduced motion is on."""

    if not reduced:
        return MOTION
    return MotionDurations(
        **{name: REDUCED_MOTION_FADE_MS for name in asdict(MOTION) if name != "busy"}
    )


def application_palette() -> QPalette:
    """Return the palette used by both the live app and screenshot harness."""

    palette = QPalette()
    role = QPalette.ColorRole
    palette.setColor(role.Window, qcolor(PALETTE.app_surface))
    palette.setColor(role.WindowText, qcolor(PALETTE.text_primary))
    palette.setColor(role.Base, qcolor(PALETTE.control_surface_alt))
    palette.setColor(role.AlternateBase, qcolor(PALETTE.control_surface))
    palette.setColor(role.ToolTipBase, qcolor(PALETTE.control_surface))
    palette.setColor(role.ToolTipText, qcolor(PALETTE.text_primary))
    palette.setColor(role.Text, qcolor(PALETTE.text_primary))
    palette.setColor(role.Button, qcolor(PALETTE.action_secondary))
    palette.setColor(role.ButtonText, qcolor(PALETTE.text_primary))
    palette.setColor(role.BrightText, qcolor(PALETTE.text_on_emphasis))
    palette.setColor(role.Highlight, qcolor(PALETTE.action_primary))
    palette.setColor(role.HighlightedText, qcolor(PALETTE.text_on_accent))
    palette.setColor(role.Link, qcolor(PALETTE.accent))
    palette.setColor(role.LinkVisited, qcolor(PALETTE.accent_pressed))
    palette.setColor(role.PlaceholderText, qcolor(PALETTE.text_muted))
    palette.setColor(role.Light, qcolor(PALETTE.border_strong))
    palette.setColor(role.Midlight, qcolor(PALETTE.control_hover))
    palette.setColor(role.Mid, qcolor(PALETTE.control_surface_alt))
    palette.setColor(role.Dark, qcolor(PALETTE.rail_surface))
    palette.setColor(role.Shadow, qcolor(PALETTE.rail_surface))

    disabled = QPalette.ColorGroup.Disabled
    for disabled_role in (
        role.WindowText,
        role.Text,
        role.ButtonText,
        role.PlaceholderText,
    ):
        palette.setColor(disabled, disabled_role, qcolor(PALETTE.action_disabled_text))
    palette.setColor(disabled, role.Button, qcolor(PALETTE.action_disabled_surface))
    palette.setColor(disabled, role.Base, qcolor(PALETTE.control_surface_alt))
    palette.setColor(disabled, role.Highlight, qcolor(PALETTE.border_strong))
    palette.setColor(disabled, role.HighlightedText, qcolor(PALETTE.text_muted))
    return palette


def qcolor(value: str, *, alpha: int | None = None) -> QColor:
    """Create a QColor from a semantic token, optionally overriding alpha."""

    color = QColor(value)
    if alpha is not None:
        color.setAlpha(max(0, min(255, int(alpha))))
    return color


def _linear_channel(channel: int) -> float:
    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(value: str) -> float:
    """Return WCAG relative luminance for an opaque RGB token."""

    color = QColor(value)
    if not color.isValid():
        raise ValueError(f"Invalid color token: {value!r}")
    return (
        0.2126 * _linear_channel(color.red())
        + 0.7152 * _linear_channel(color.green())
        + 0.0722 * _linear_channel(color.blue())
    )


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio of two opaque semantic colors."""

    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def flatten(value: str, ground: str) -> str:
    """The opaque color a translucent token shows over ``ground``, for measuring contrast."""

    color, behind = QColor(value), QColor(ground)
    if not color.isValid() or not behind.isValid():
        raise ValueError(f"Invalid color token: {value!r} over {ground!r}")
    alpha = color.alphaF()

    def mix(front: float, back: float) -> float:
        return alpha * front + (1 - alpha) * back

    return QColor.fromRgbF(
        mix(color.redF(), behind.redF()),
        mix(color.greenF(), behind.greenF()),
        mix(color.blueF(), behind.blueF()),
    ).name()


TEXT_CONTRAST_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("primary text", PALETTE.text_primary, PALETTE.app_surface),
    ("muted text", PALETTE.text_muted, PALETTE.app_surface),
    ("accent text", PALETTE.accent, PALETTE.app_surface),
    ("primary action", PALETTE.text_on_accent, PALETTE.action_primary),
    ("primary text on card", PALETTE.text_primary, PALETTE.card_surface),
    ("muted text on card", PALETTE.text_muted, PALETTE.card_surface),
    ("muted text on rail", PALETTE.text_muted, PALETTE.rail_surface),
    ("control text", PALETTE.text_primary, PALETTE.control_surface),
    (
        "destructive action",
        PALETTE.text_on_emphasis,
        PALETTE.action_destructive,
    ),
    ("secondary action", PALETTE.text_primary, PALETTE.action_secondary),
    (
        "disabled action",
        PALETTE.action_disabled_text,
        PALETTE.action_disabled_surface,
    ),
    ("success status", PALETTE.status_ok_text, PALETTE.status_ok_surface),
    ("warning status", PALETTE.status_warn_text, PALETTE.status_warn_surface),
    ("error status", PALETTE.status_bad_text, PALETTE.status_bad_surface),
    ("information status", PALETTE.status_info_text, PALETTE.status_info_surface),
    ("idle status", PALETTE.status_idle_text, PALETTE.status_idle_surface),
    ("data text", PALETTE.data_text, PALETTE.data_surface),
    ("muted data text", PALETTE.data_text_muted, PALETTE.data_surface),
    ("dim text", PALETTE.text_dim, PALETTE.app_surface),
    ("dim text on card", PALETTE.text_dim, PALETTE.card_surface),
    ("dim text on sidebar", PALETTE.text_dim, PALETTE.sidebar_top),
    ("muted text on toolbar", PALETTE.text_muted, PALETTE.toolbar_top),
    ("primary text on pane", PALETTE.text_primary, PALETTE.pane_surface),
    # The equalizer's footer: the calibration line and the mark beside it, in its state's colour.
    *(
        (f"{state} status text on pane", getattr(PALETTE, f"status_{state}_text"), PALETTE.pane_surface)
        for state in ("ok", "warn", "bad", "idle")
    ),
    # The Show choices and a field's unit, on a control; the chosen view on its hovered fill.
    ("muted text on a control", PALETTE.text_muted, PALETTE.control_surface),
    ("chosen text on a hovered control", PALETTE.hover_text, PALETTE.control_hover),
    ("accent link text", PALETTE.accent_text, PALETTE.app_surface),
    (
        "recommended action",
        PALETTE.accent_bright,
        flatten(PALETTE.accent_fill, PALETTE.app_surface),
    ),
    (
        "selected row secondary text",
        PALETTE.accent_soft,
        flatten(PALETTE.accent_fill, PALETTE.sidebar_top),
    ),
    ("pressed mute", PALETTE.mute_text, flatten(PALETTE.mute_fill, PALETTE.toolbar_top)),
    ("text on a top bar control", PALETTE.text_primary, flatten(PALETTE.toolbar_control_top, PALETTE.toolbar_top)),
    ("muted text on a top bar control", PALETTE.text_muted, flatten(PALETTE.toolbar_control_top, PALETTE.toolbar_top)),
    ("Start text, top", PALETTE.text_on_accent, PALETTE.start_top),
    ("Start text, bottom", PALETTE.text_on_accent, PALETTE.start_bottom),
    ("Stop text, top", PALETTE.text_on_stop, PALETTE.stop_top),
    ("Stop text, bottom", PALETTE.text_on_stop, PALETTE.stop_bottom),
    # Hover lays a light tint over the face.
    *(
        (f"{name} text on its hovered face, {end}", ink, flatten(PALETTE.transport_hover, face))
        for name, ink, faces in (
            ("Start", PALETTE.text_on_accent, (PALETTE.start_top, PALETTE.start_bottom)),
            ("Stop", PALETTE.text_on_stop, (PALETTE.stop_top, PALETTE.stop_bottom)),
        )
        for end, face in zip(("top", "bottom"), faces)
    ),
    ("text on solid mute", PALETTE.text_on_mute, PALETTE.mute),
    ("mute text", PALETTE.mute_text, PALETTE.app_surface),
    ("callers affected text", PALETTE.callers_affected_text, PALETTE.pane_surface),
    ("callers affected text on card", PALETTE.callers_affected_text, PALETTE.card_surface),
    ("callers affected text on sidebar", PALETTE.callers_affected_text, PALETTE.sidebar_top),
    (
        "callers affected text on its fill",
        PALETTE.callers_affected_text,
        flatten(PALETTE.callers_affected_fill, PALETTE.pane_surface),
    ),
    *(
        (f"{label} on {glass_name} over {surface_name}", foreground, flatten(glass, surface))
        for glass_name, glass in (("glass", PALETTE.glass), ("compact glass", PALETTE.glass_solid))
        for surface_name, surface in (
            ("app", PALETTE.app_surface),
            ("rail", PALETTE.rail_surface),
            ("pane", PALETTE.pane_surface),
            ("sidebar top", PALETTE.sidebar_top),
            ("sidebar bottom", PALETTE.sidebar_bottom),
            ("card", PALETTE.card_surface),
            ("control", PALETTE.control_surface),
        )
        for label, foreground in (
            ("primary text", PALETTE.text_primary),
            ("muted text", PALETTE.text_muted),
            ("dim text", PALETTE.text_dim),
        )
    ),
    ("muted text on modal glass", PALETTE.text_muted, flatten(PALETTE.glass_modal, PALETTE.app_surface)),
    # The command palette's rows, on its modal glass and on the selected row's fill over it.
    *(
        (f"palette {label} on modal glass", foreground, flatten(PALETTE.glass_modal, PALETTE.app_surface))
        for label, foreground in (
            ("primary text", PALETTE.text_primary),
            ("dim text", PALETTE.text_dim),
            ("disabled text", PALETTE.action_disabled_text),
            ("accent text", PALETTE.accent_text),
            ("callers text", PALETTE.callers_affected_text),
        )
    ),
    *(
        (
            f"palette {label} on the selected row",
            foreground,
            flatten(PALETTE.accent_fill, flatten(PALETTE.glass_modal, PALETTE.app_surface)),
        )
        for label, foreground in (
            ("title", PALETTE.hover_text),
            ("muted text", PALETTE.text_muted),
            ("accent text", PALETTE.accent_text),
            ("callers text", PALETTE.callers_affected_text),
        )
    ),
    ("signal in label", PALETTE.signal_in, PALETTE.data_surface),
    ("signal out label", PALETTE.signal_out, PALETTE.data_surface),
    *(
        (f"{label} on {ground_name}", foreground, ground)
        for ground_name, ground in (
            ("card top", PALETTE.card_top),
            ("pane top", PALETTE.pane_top),
            ("pane bottom", PALETTE.pane_bottom),
            ("sidebar top", PALETTE.sidebar_top),
            ("sidebar bottom", PALETTE.sidebar_bottom),
        )
        for label, foreground in (
            ("primary text", PALETTE.text_primary),
            ("muted text", PALETTE.text_muted),
            ("dim text", PALETTE.text_dim),
            ("success text", PALETTE.status_ok_text),
            ("warning text", PALETTE.status_warn_text),
            ("error text", PALETTE.status_bad_text),
            ("signal in label", PALETTE.signal_in),
            ("signal out label", PALETTE.signal_out),
        )
    ),
    ("icon on a tile", PALETTE.tile_icon, flatten(PALETTE.tile_top, PALETTE.sidebar_top)),
    ("selected row name", PALETTE.text_primary, flatten(PALETTE.accent_row_start, PALETTE.sidebar_top)),
    ("selected row line", PALETTE.accent_soft, flatten(PALETTE.accent_row_start, PALETTE.sidebar_top)),
    ("selected row icon", PALETTE.accent_bright, flatten(PALETTE.accent_tile_top, PALETTE.sidebar_top)),
    ("waiting pill text", PALETTE.text_primary, PALETTE.pill_surface),
    ("keycap label", PALETTE.key_text, flatten(PALETTE.key_top, PALETTE.pane_bottom)),
    # The play mark is text on the highlight in a contrast theme, so it takes accent_bright.
    ("waiting pill play mark", PALETTE.accent_bright, flatten(PALETTE.accent_fill, PALETTE.pill_surface)),
    # The route-check marks sit on their own tints over the card's lighter top.
    ("passed check mark", PALETTE.status_ok_text, flatten(PALETTE.status_ok_fill, PALETTE.card_top)),
    ("check to act on mark", PALETTE.status_warn_text, flatten(PALETTE.status_warn_surface, PALETTE.card_top)),
    ("note check mark", PALETTE.text_muted, flatten(PALETTE.status_note_fill, PALETTE.card_top)),
    # The sheen is brightest at the pane's top edge, where the headline and the headings are.
    *(
        (f"{label} under the pane's sheen", foreground, flatten(PALETTE.pane_glow, PALETTE.pane_top))
        for label, foreground in (
            ("primary text", PALETTE.text_primary),
            ("muted text", PALETTE.text_muted),
            ("success text", PALETTE.status_ok_text),
        )
    ),
    # The band strip's badge: the band's number, dark on the band's color.
    *(
        (f"EQ band {number} number on its badge", PALETTE.text_on_data_color, color)
        for number, color in enumerate(PALETTE.eq_band_colors, start=1)
    ),
)


def control_boundary_pairs(palette: SemanticPalette) -> tuple[tuple[str, str, str], ...]:
    """Non-text parts that identify a control, each with the surface behind it; 3:1 (WCAG 1.4.11)."""

    return (
        ("control edge on app surface", palette.control_edge, palette.app_surface),
        ("control edge on card", palette.control_edge, palette.card_surface),
        ("control edge on field", palette.control_edge, palette.control_surface),
        ("slider fill on card", palette.accent, palette.card_surface),
        ("switch on against its knob", palette.switch_on, palette.app_surface),
        ("switch on on the sidebar", palette.switch_on, palette.sidebar_top),
        ("switch on on card", palette.switch_on, palette.card_surface),
        ("focus ring on card", palette.focus_ring, palette.card_surface),
        ("focus ring on app surface", palette.focus_ring, palette.app_surface),
        ("focus ring on glass", palette.focus_ring, flatten(palette.glass, palette.app_surface)),
        ("danger button edge on card", palette.action_destructive_border, palette.card_surface),
        ("empty picker edge on card", palette.status_warn_border, palette.card_surface),
        ("callers affected mark on pane", palette.callers_affected, palette.pane_surface),
        ("callers affected mark on its fill", palette.callers_affected,
         flatten(palette.callers_affected_fill, palette.pane_surface)),
        ("callers affected edge on pane", palette.callers_affected_edge, palette.pane_surface),
        ("selected EQ handle ring on the graph", palette.data_handle_selected, palette.data_surface),
    )


CONTROL_BOUNDARY_PAIRS = control_boundary_pairs(PALETTE)


def muted_icon(icon: QIcon) -> QIcon:
    """The icon with a crossed-out badge in its corner, for the tray while output is muted."""

    muted = QIcon()
    for size in icon.availableSizes() or [QSize(32, 32)]:
        pixmap = icon.pixmap(size)
        if pixmap.isNull():
            continue
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = pixmap.width() * 0.6
        badge = QRectF(pixmap.width() - side, pixmap.height() - side, side, side).adjusted(1, 1, -1, -1)
        mark = QPen(qcolor(PALETTE.text_on_mute), max(1.0, side / 8))
        painter.setPen(mark)
        painter.setBrush(qcolor(PALETTE.mute))
        painter.drawEllipse(badge)
        centre, reach = badge.center(), side * 0.2
        painter.drawLine(centre + QPointF(reach, -reach), centre + QPointF(-reach, reach))
        painter.end()
        muted.addPixmap(pixmap)
    return muted


def prefers_reduced_motion() -> bool:
    """Return the user's reduced-motion preference.

    ``AUDIOFORGE_REDUCED_MOTION`` provides a deterministic override for tests
    and assistive setups.  On Windows the system client-animation preference is
    used when available.  Failure to query the OS keeps the existing behavior.
    """

    override = os.getenv("AUDIOFORGE_REDUCED_MOTION")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        enabled = ctypes.c_int(1)
        spi_get_client_area_animation = 0x1042
        succeeded = ctypes.windll.user32.SystemParametersInfoW(
            spi_get_client_area_animation,
            0,
            ctypes.byref(enabled),
            0,
        )
        return bool(succeeded) and not bool(enabled.value)
    except (AttributeError, OSError):
        return False


def prefers_reduced_transparency() -> bool:
    """Return whether the user turned Windows transparency effects off.

    ``AUDIOFORGE_REDUCED_TRANSPARENCY`` overrides it, as ``AUDIOFORGE_REDUCED_MOTION``
    does for motion. Windows keeps the switch (Personalization or Accessibility, Visual
    effects) as ``EnableTransparency`` in the user's Personalize key; a missing value
    means transparency is on.
    """

    override = os.getenv("AUDIOFORGE_REDUCED_TRANSPARENCY")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "EnableTransparency")
        return value == 0
    except OSError:
        return False


def themed_menu(parent: QWidget | None = None) -> QMenu:
    """A menu whose rounded corners show what is behind it, not a square frame."""

    menu = QMenu(parent)
    menu.setWindowFlags(
        menu.windowFlags() | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
    )
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    return menu


def application_stylesheet() -> str:
    """Return the one stylesheet that gives stock Qt widgets the app's look."""

    p = PALETTE
    control = f"border-radius: {RADIUS_CONTROL}px;"
    # Qt menus cannot blur, so the glass is laid over the window ground to stay opaque.
    menu_ground = flatten(p.glass_solid, p.app_surface)
    if QColor(p.glass_solid).alpha() < 255:
        menu_hover, menu_hover_text = flatten(MENU_HOVER_WHITE, menu_ground), p.text_primary
    else:  # a contrast theme: the highlight pair, as every other selected row
        menu_hover, menu_hover_text = p.control_hover, p.hover_text
    return f"""
QToolTip {{ background-color: {p.control_surface}; color: {p.text_primary};
    border: 1px solid {p.border_strong}; padding: 6px 8px; }}

QPushButton {{ background-color: {p.action_secondary}; color: {p.text_primary};
    border: 1px solid {p.control_edge}; {control}
    padding: 6px 10px; }}
QPushButton:hover {{ background-color: {p.control_hover}; color: {p.hover_text}; }}
QPushButton:pressed {{ background-color: {p.control_surface_alt}; }}
QPushButton:focus {{ border-color: {p.focus_ring}; }}
QPushButton:disabled {{ background-color: {p.action_disabled_surface};
    color: {p.action_disabled_text}; border-color: {p.action_disabled_border}; }}

QComboBox, QLineEdit {{ background-color: {p.control_surface};
    color: {p.text_primary}; border: 1px solid {p.control_edge}; {control}
    padding: 5px 8px; selection-background-color: {p.accent};
    selection-color: {p.text_on_accent}; }}
QComboBox:hover, QLineEdit:hover {{
    border-color: {p.text_muted}; }}
QComboBox:focus, QLineEdit:focus {{
    border-color: {p.focus_ring}; }}
QComboBox:disabled, QLineEdit:disabled {{
    background-color: {p.action_disabled_surface};
    color: {p.action_disabled_text}; }}
QComboBox QAbstractItemView {{ background-color: {p.control_surface};
    color: {p.text_primary}; border: 1px solid {p.border_strong};
    selection-background-color: {p.control_hover};
    selection-color: {p.hover_text}; outline: 0; }}

QScrollBar:vertical {{ width: 10px; margin: 0; background: transparent; }}
QScrollBar:horizontal {{ height: 10px; margin: 0; background: transparent; }}
QScrollBar::handle {{ background-color: {p.control_edge};
    border-radius: 4px; margin: 1px; }}
QScrollBar::handle:vertical {{ min-height: 32px; }}
QScrollBar::handle:horizontal {{ min-width: 32px; }}
QScrollBar::handle:hover {{ background-color: {p.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QDialog {{ background-color: {p.pane_surface}; color: {p.text_primary}; }}
QDialog QLabel {{ font-size: 13px; color: {p.text_primary}; }}
QDialog QPushButton {{ min-height: 22px; min-width: 56px; padding: 6px 16px;
    border-radius: {RADIUS_ITEM}px; }}
QDialog QPushButton:default {{ border-color: {flatten(p.accent_edge, p.pane_surface)}; }}
QDialog QPushButton:default:focus {{ border-color: {p.focus_ring}; }}

QMenu {{ background-color: {menu_ground}; color: {p.text_primary};
    border: 1px solid {flatten(p.glass_edge, menu_ground)};
    border-radius: {RADIUS_MENU}px; padding: 6px; }}
QMenu::item {{ min-height: 20px; padding: 8px 18px 8px 34px;
    border-radius: {RADIUS_ITEM}px; }}
QMenu::item:selected {{ background-color: {menu_hover}; color: {menu_hover_text}; }}
QMenu::item:disabled {{ color: {p.action_disabled_text}; }}
QMenu::indicator {{ width: 16px; height: 16px; left: 9px; }}
QMenu::separator {{ height: 1px; background-color: {flatten(p.glass_edge, menu_ground)};
    margin: 6px 8px; }}

QSplitter::handle {{ background: transparent; }}
"""


def apply_application_theme(app: QApplication) -> None:
    """Give the live app, the tests and the screenshot tool the same look."""

    app.setStyle("Fusion")
    app.setPalette(application_palette())
    font = app.font()
    font.setFamilies(list(UI_FONT_FAMILIES))
    app.setFont(font)
    app.setStyleSheet(application_stylesheet())
