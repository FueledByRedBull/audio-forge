"""Semantic visual tokens for AudioForge's explicit dark interface.

The token names describe intent rather than a particular widget.  Keeping the
palette here prevents dialogs, native Qt controls, and custom-painted widgets
from drifting apart across different Windows appearance settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
import sys

from PySide6.QtCore import QPointF, QRectF, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen
from PySide6.QtWidgets import QApplication


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
    sidebar_top: str = "#202127"
    sidebar_bottom: str = "#18191d"
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
    # Violet is Mute and nothing else.
    mute: str = "#b9a3f5"
    mute_text: str = "#cdbcf8"
    text_on_mute: str = "#1a1230"
    mute_fill: str = "#29b9a3f5"
    mute_edge: str = "#73b9a3f5"

    action_primary: str = "#d98a4e"
    # Stop while live is neutral: only the action that begins something is amber.
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
    data_handle_selected: str = "#ececf0"
    # The microphone (IN), what callers hear (OUT) and the EQ analyser, each one job only.
    signal_in: str = "#a9bbd4"
    signal_in_fill: str = "#7ea4dc"
    signal_out: str = "#e6e8ee"
    analyser: str = "#5e9eff"

    # Panels that dock beside live content; translucent over the ground behind them.
    glass: str = "#8c282c36"
    glass_solid: str = "#d122252e"
    glass_modal: str = "#b8242832"
    glass_edge: str = "#1affffff"
    glass_edge_modal: str = "#1fffffff"
    glass_highlight: str = "#12ffffff"
    glass_shadow: str = "#80000000"

    # One color per EQ band, in band order; data color, not accent.
    eq_band_colors: tuple[str, ...] = (
        "#a78bfa",
        "#6d8bff",
        "#f472b6",
        "#f87171",
        "#fb923c",
        "#fbbf24",
        "#a3e635",
        "#34d399",
        "#22d3ee",
        "#38bdf8",
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
        data_marker=accent, data_warning=accent, data_handle_selected=text,
        meter_scale=text, meter_peak=text,
        pane_surface=window, toolbar_top=window, toolbar_bottom=window,
        sidebar_top=window, sidebar_bottom=window, focus_ring=accent, text_dim=text,
        accent_text=accent, accent_soft=on_accent, accent_bright=on_accent,
        accent_fill=accent, accent_edge=accent,
        mute=accent, text_on_mute=on_accent, mute_text=text, mute_fill=window, mute_edge=accent,
        status_ok_dot=text, signal_in=text, signal_in_fill=text, signal_out=text, analyser=text,
        glass=window, glass_solid=window, glass_modal=window,
        glass_edge=text, glass_edge_modal=text, glass_highlight=window, glass_shadow="#00000000",
    )


# Read once at import: the styles below are built from it, so a contrast theme
# switched on while the app runs is picked up at the next start.
SYSTEM_CONTRAST_COLORS = windows_contrast_colors()
PALETTE = build_palette(SYSTEM_CONTRAST_COLORS)


def contrast_theme_changed() -> bool:
    """True when Windows contrast colors differ from the ones the styles were built with."""

    return windows_contrast_colors() != SYSTEM_CONTRAST_COLORS


RADIUS_CARD = 10
RADIUS_CONTROL = 8
UI_FONT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI")

# Text sizes in pixels at the default app font. Text that carries information is never
# below TYPE_FLOOR_PX; only axis ticks and keycaps use the smaller tick size.
TYPE_FLOOR_PX = 12
TYPE_TICK_PX = 11
TYPE_SCALE_PX = {"caption": 12, "body": 13, "lead": 15, "heading": 18, "title": 20, "display": 28}


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
    """Durations in milliseconds; reduced motion swaps each for a short opacity fade."""

    color: int = 150
    status: int = 150
    crossfade: int = 120
    switch: int = 120
    row: int = 120
    panel_open: int = 300
    panel_close: int = 200


MOTION = MotionDurations()
REDUCED_MOTION_FADE_MS = 100
# Cubic-bezier control points: the panel decelerates as it opens and accelerates as it closes.
EASE_PANEL_OPEN = (0.16, 1.0, 0.3, 1.0)
EASE_PANEL_CLOSE = (0.42, 0.0, 1.0, 1.0)


def motion_durations(reduced: bool) -> MotionDurations:
    """The durations to use, with every one a short fade while reduced motion is on."""

    if not reduced:
        return MOTION
    return MotionDurations(**{name: REDUCED_MOTION_FADE_MS for name in asdict(MOTION)})


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
    ("text on solid mute", PALETTE.text_on_mute, PALETTE.mute),
    ("mute text", PALETTE.mute_text, PALETTE.app_surface),
    ("primary text on glass", PALETTE.text_primary, flatten(PALETTE.glass, PALETTE.app_surface)),
    ("muted text on glass", PALETTE.text_muted, flatten(PALETTE.glass, PALETTE.app_surface)),
    ("muted text on compact glass", PALETTE.text_muted, flatten(PALETTE.glass_solid, PALETTE.app_surface)),
    ("muted text on modal glass", PALETTE.text_muted, flatten(PALETTE.glass_modal, PALETTE.app_surface)),
    ("signal in label", PALETTE.signal_in, PALETTE.data_surface),
    ("signal out label", PALETTE.signal_out, PALETTE.data_surface),
    # The number on the selected EQ handle, dark on the band's color.
    *(
        (f"EQ band {number} number on handle", PALETTE.text_on_data_color, color)
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


def application_stylesheet() -> str:
    """Return the one stylesheet that gives stock Qt widgets the app's look."""

    p = PALETTE
    control = f"border-radius: {RADIUS_CONTROL}px;"
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

QMenu {{ background-color: {p.control_surface}; color: {p.text_primary};
    border: 1px solid {p.border_strong}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 28px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {p.control_hover}; color: {p.hover_text}; }}
QMenu::item:disabled {{ color: {p.action_disabled_text}; }}
QMenu::separator {{ height: 1px; background-color: {p.border};
    margin: 4px 8px; }}

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
