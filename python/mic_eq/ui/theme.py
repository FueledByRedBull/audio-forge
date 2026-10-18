"""Semantic visual tokens for AudioForge's explicit dark interface.

The token names describe intent rather than a particular widget.  Keeping the
palette here prevents dialogs, native Qt controls, and custom-painted widgets
from drifting apart across different Windows appearance settings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class SemanticPalette:
    """Colors used by the application chrome and its data canvases."""

    app_surface: str = "#14171c"
    rail_surface: str = "#0f1115"
    card_surface: str = "#1b1f26"
    control_surface: str = "#262b34"
    control_surface_alt: str = "#20242c"
    control_hover: str = "#2f3540"
    border: str = "#2b313b"
    border_strong: str = "#3d4552"
    text_primary: str = "#eef1f5"
    text_muted: str = "#a2abb8"
    text_on_emphasis: str = "#ffffff"
    text_on_accent: str = "#14171c"
    # Dark text on a data color (EQ band, clip); a contrast theme keeps those colors.
    text_on_data_color: str = "#14171c"
    # Text on a hovered or selected control; the dark look keeps the normal text color.
    hover_text: str = "#eef1f5"
    accent: str = "#f2743a"
    accent_hover: str = "#ff8a55"
    accent_pressed: str = "#d9612b"

    action_primary: str = "#f2743a"
    action_destructive: str = "#b3261e"
    action_destructive_border: str = "#dc4a41"
    action_secondary: str = "#262b34"
    action_secondary_border: str = "#3d4552"
    action_disabled: str = "#232830"
    action_disabled_text: str = "#8f98a6"
    action_disabled_surface: str = "#1e2229"
    action_disabled_border: str = "#2b313b"

    status_ok_surface: str = "#16352b"
    status_ok_text: str = "#86efac"
    status_ok_border: str = "#2f855a"
    status_warn_surface: str = "#3b2e14"
    status_warn_text: str = "#fcd34d"
    status_warn_border: str = "#a16207"
    status_bad_surface: str = "#401d24"
    status_bad_text: str = "#fda4af"
    status_bad_border: str = "#be123c"
    status_info_surface: str = "#172d4d"
    status_info_text: str = "#93c5fd"
    status_info_border: str = "#2563eb"
    status_idle_surface: str = "#20242c"
    status_idle_text: str = "#cbd5e1"
    status_idle_border: str = "#3d4552"

    data_surface: str = "#111419"
    data_surface_raised: str = "#1b1f26"
    data_grid: str = "#333a45"
    data_text: str = "#d6dde7"
    data_text_muted: str = "#aeb8c5"
    data_curve: str = "#e6ebf2"
    data_marker: str = "#facc15"
    data_warning: str = "#fbbf24"
    data_handle_selected: str = "#ffffff"

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

    meter_safe: str = "#3ecf8e"
    meter_caution: str = "#f5c542"
    meter_danger: str = "#ef5350"
    meter_clip: str = "#ff3b30"
    meter_peak: str = "#ffffff"
    meter_scale: str = "#a2abb8"
    meter_reduction: str = "#f5c542"



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
        action_secondary=window, action_secondary_border=text,
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
    )


# Read once at import: the styles below are built from it, so a contrast theme
# switched on while the app runs is picked up at the next start.
SYSTEM_CONTRAST_COLORS = windows_contrast_colors()
PALETTE = build_palette(SYSTEM_CONTRAST_COLORS)


def contrast_theme_changed() -> bool:
    """True when Windows contrast colors differ from the ones the styles were built with."""

    return windows_contrast_colors() != SYSTEM_CONTRAST_COLORS


RADIUS_CARD = 10
RADIUS_CONTROL = 6
UI_FONT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI")


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
    # The number on the selected EQ handle, dark on the band's color.
    *(
        (f"EQ band {number} number on handle", PALETTE.text_on_data_color, color)
        for number, color in enumerate(PALETTE.eq_band_colors, start=1)
    ),
)


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


SUBDUED_TEXT_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
DESCRIPTION_LABEL_STYLE = f"color: {PALETTE.text_muted}; font-size: 9pt; padding: 5px;"
PROGRESS_LABEL_STYLE = f"font-size: 12pt; color: {PALETTE.text_primary}; font-weight: bold;"

def _button_style(
    surface: str, text: str, border: str, hover: str, hover_text: str | None = None
) -> str:
    return (
        f"QPushButton {{ background-color: {surface}; color: {text}; "
        f"font-weight: 600; border: 1px solid {border}; "
        f"border-radius: {RADIUS_CONTROL}px; padding: 6px 10px; }} "
        f"QPushButton:hover {{ background-color: {hover}; color: {hover_text or text}; }} "
        f"QPushButton:focus {{ border-color: {PALETTE.text_primary}; }} "
        f"QPushButton:disabled {{ background-color: {PALETTE.action_disabled_surface}; "
        f"color: {PALETTE.action_disabled_text}; "
        f"border-color: {PALETTE.action_disabled_border}; }}"
    )


STEP_STYLE = f"color: {PALETTE.text_muted}; padding: 0 0 6px 0;"
STEP_CURRENT_STYLE = (
    f"color: {PALETTE.text_primary}; font-weight: 600; padding: 0 0 4px 0; "
    f"border-bottom: 2px solid {PALETTE.accent};"
)

PRIMARY_ACTION_BUTTON_STYLE = _button_style(
    PALETTE.action_primary,
    PALETTE.text_on_accent,
    PALETTE.action_primary,
    PALETTE.accent_hover,
)

SECONDARY_ACTION_BUTTON_STYLE = _button_style(
    PALETTE.action_secondary,
    PALETTE.text_primary,
    PALETTE.action_secondary_border,
    PALETTE.control_hover,
    PALETTE.hover_text,
)

CARD_STYLE = (
    f"QFrame#card {{ background-color: {PALETTE.card_surface}; "
    f"border: 1px solid {PALETTE.border}; border-radius: {RADIUS_CARD}px; }}"
)

PROGRESS_BAR_STYLE = (
    f"QProgressBar {{ border: none; border-radius: 4px; "
    f"background-color: {PALETTE.control_surface}; }} "
    f"QProgressBar::chunk {{ background-color: {PALETTE.accent}; "
    "border-radius: 4px; }"
)


_STATUS_COLORS = {
    "ok": (
        PALETTE.status_ok_surface,
        PALETTE.status_ok_text,
        PALETTE.status_ok_border,
    ),
    "warn": (
        PALETTE.status_warn_surface,
        PALETTE.status_warn_text,
        PALETTE.status_warn_border,
    ),
    "bad": (
        PALETTE.status_bad_surface,
        PALETTE.status_bad_text,
        PALETTE.status_bad_border,
    ),
    "info": (
        PALETTE.status_info_surface,
        PALETTE.status_info_text,
        PALETTE.status_info_border,
    ),
    "idle": (
        PALETTE.status_idle_surface,
        PALETTE.status_idle_text,
        PALETTE.status_idle_border,
    ),
}


def status_chip_style(state: str) -> str:
    background, foreground, border = _STATUS_COLORS.get(
        state,
        _STATUS_COLORS["idle"],
    )
    return (
        "QLabel { "
        f"background-color: {background}; "
        f"color: {foreground}; "
        f"border: 1px solid {border}; "
        f"border-radius: {RADIUS_CONTROL}px; "
        "padding: 6px 10px; "
        "font-size: 9pt; "
        "font-weight: 600; }"
    )


def message_text_style(state: str, *, strong: bool = False) -> str:
    """Style transient dialog text using the semantic status foreground."""

    _background, foreground, _border = _STATUS_COLORS.get(
        state,
        _STATUS_COLORS["idle"],
    )
    weight = " font-weight: bold;" if strong else ""
    return f"color: {foreground}; font-size: 11pt;{weight}"


def application_stylesheet() -> str:
    """Return the one stylesheet that gives stock Qt widgets the app's look."""

    p = PALETTE
    control = f"border-radius: {RADIUS_CONTROL}px;"
    return f"""
QToolTip {{ background-color: {p.control_surface}; color: {p.text_primary};
    border: 1px solid {p.border_strong}; padding: 6px 8px; }}

QGroupBox {{ background-color: {p.card_surface}; border: 1px solid {p.border};
    border-radius: {RADIUS_CARD}px; margin-top: 0; padding: 34px 0 4px 0;
    font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: border; subcontrol-position: top left;
    padding: 12px 12px 0 12px; color: {p.text_primary}; }}
QGroupBox QGroupBox {{ background-color: transparent; border: none;
    border-top: 1px solid {p.border}; border-radius: 0; padding: 36px 0 0 0; }}
QGroupBox QGroupBox::title {{ padding: 10px 0 0 0; color: {p.text_muted}; }}

QPushButton {{ background-color: {p.action_secondary}; color: {p.text_primary};
    border: 1px solid {p.action_secondary_border}; {control}
    padding: 6px 10px; }}
QPushButton:hover {{ background-color: {p.control_hover}; color: {p.hover_text}; }}
QPushButton:pressed {{ background-color: {p.control_surface_alt}; }}
QPushButton:checked {{ border-color: {p.accent}; color: {p.accent}; }}
QPushButton:focus {{ border-color: {p.text_primary}; }}
QPushButton:disabled {{ background-color: {p.action_disabled_surface};
    color: {p.action_disabled_text}; border-color: {p.action_disabled_border}; }}

QComboBox, QAbstractSpinBox, QLineEdit {{ background-color: {p.control_surface};
    color: {p.text_primary}; border: 1px solid {p.border}; {control}
    padding: 5px 8px; selection-background-color: {p.accent};
    selection-color: {p.text_on_accent}; }}
QComboBox:hover, QAbstractSpinBox:hover, QLineEdit:hover {{
    border-color: {p.border_strong}; }}
QComboBox:focus, QAbstractSpinBox:focus, QLineEdit:focus {{
    border-color: {p.accent}; }}
QComboBox:disabled, QAbstractSpinBox:disabled, QLineEdit:disabled {{
    background-color: {p.action_disabled_surface};
    color: {p.action_disabled_text}; }}
QAbstractSpinBox {{ qproperty-buttonSymbols: NoButtons; }}
QComboBox QAbstractItemView {{ background-color: {p.control_surface};
    color: {p.text_primary}; border: 1px solid {p.border_strong};
    selection-background-color: {p.control_hover};
    selection-color: {p.hover_text}; outline: 0; }}

QTextEdit {{ background-color: {p.data_surface};
    color: {p.text_primary}; border: 1px solid {p.border}; {control}
    padding: 8px; }}

QSlider::groove:horizontal {{ height: 4px; border-radius: 2px;
    background-color: {p.border_strong}; }}
QSlider::sub-page:horizontal {{ border-radius: 2px;
    background-color: {p.accent}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0;
    border-radius: 7px; background-color: {p.text_primary}; }}
QSlider::handle:hover, QSlider::handle:focus {{
    background-color: {p.accent_hover}; }}
QSlider::sub-page:disabled, QSlider::add-page:disabled {{
    background-color: {p.border_strong}; }}
QSlider::handle:disabled {{ background-color: {p.action_disabled_text}; }}

QScrollArea {{ border: none; }}
QScrollBar:vertical {{ width: 10px; margin: 0; background: transparent; }}
QScrollBar:horizontal {{ height: 10px; margin: 0; background: transparent; }}
QScrollBar::handle {{ background-color: {p.border_strong};
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

QProgressBar {{ border: none; border-radius: 4px; text-align: center;
    background-color: {p.control_surface}; color: {p.text_primary}; }}
QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 4px; }}

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
