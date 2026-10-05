"""Semantic visual tokens for AudioForge's explicit dark interface.

The token names describe intent rather than a particular widget.  Keeping the
palette here prevents dialogs, native Qt controls, and custom-painted widgets
from drifting apart across different Windows appearance settings.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    warning_banner_surface: str = "#d97706"
    warning_banner_text: str = "#111827"

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


PALETTE = SemanticPalette()

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
    ("warning banner", PALETTE.warning_banner_text, PALETTE.warning_banner_surface),
    ("success status", PALETTE.status_ok_text, PALETTE.status_ok_surface),
    ("warning status", PALETTE.status_warn_text, PALETTE.status_warn_surface),
    ("error status", PALETTE.status_bad_text, PALETTE.status_bad_surface),
    ("information status", PALETTE.status_info_text, PALETTE.status_info_surface),
    ("idle status", PALETTE.status_idle_text, PALETTE.status_idle_surface),
    ("data text", PALETTE.data_text, PALETTE.data_surface),
    ("muted data text", PALETTE.data_text_muted, PALETTE.data_surface),
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


PRIMARY_LABEL_STYLE = "font-size: 11pt;"
METER_LABEL_STYLE = f"font-size: 10pt; font-weight: bold; color: {PALETTE.accent};"
INFO_LABEL_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
SUBDUED_TEXT_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
DESCRIPTION_LABEL_STYLE = f"color: {PALETTE.text_muted}; font-size: 9pt; padding: 5px;"
PROGRESS_LABEL_STYLE = f"font-size: 12pt; color: {PALETTE.accent}; font-weight: bold;"

def _button_style(
    surface: str, text: str, border: str, hover: str, *, selector: str = "QPushButton"
) -> str:
    return (
        f"{selector} {{ background-color: {surface}; color: {text}; "
        f"font-weight: 600; border: 1px solid {border}; "
        f"border-radius: {RADIUS_CONTROL}px; padding: 6px 10px; }} "
        f"{selector}:hover {{ background-color: {hover}; }} "
        f"{selector}:focus {{ border-color: {PALETTE.text_primary}; }} "
        f"{selector}:disabled {{ background-color: {PALETTE.action_disabled_surface}; "
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

DESTRUCTIVE_ACTION_BUTTON_STYLE = _button_style(
    PALETTE.action_destructive,
    PALETTE.text_on_emphasis,
    PALETTE.action_destructive,
    PALETTE.action_destructive_border,
)

SECONDARY_ACTION_BUTTON_STYLE = _button_style(
    PALETTE.action_secondary,
    PALETTE.text_primary,
    PALETTE.action_secondary_border,
    PALETTE.control_hover,
)

CARD_STYLE = (
    f"QFrame#card {{ background-color: {PALETTE.card_surface}; "
    f"border: 1px solid {PALETTE.border}; border-radius: {RADIUS_CARD}px; }}"
)

WARNING_BANNER_STYLE = (
    f"QLabel {{ background-color: {PALETTE.warning_banner_surface}; "
    f"color: {PALETTE.warning_banner_text}; padding: 10px 12px; "
    f"font-weight: 600; border-radius: {RADIUS_CONTROL}px; }}"
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
QPushButton:hover {{ background-color: {p.control_hover}; }}
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
    selection-color: {p.text_primary}; outline: 0; }}

QTextEdit, QPlainTextEdit {{ background-color: {p.data_surface};
    color: {p.text_primary}; border: 1px solid {p.border}; {control}
    padding: 8px; }}

QSlider::groove:horizontal {{ height: 4px; border-radius: 2px;
    background-color: {p.border_strong}; }}
QSlider::sub-page:horizontal {{ border-radius: 2px;
    background-color: {p.accent}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0;
    border-radius: 7px; background-color: {p.text_primary}; }}
QSlider::groove:vertical {{ width: 4px; border-radius: 2px;
    background-color: {p.border_strong}; }}
QSlider::add-page:vertical {{ border-radius: 2px;
    background-color: {p.accent}; }}
QSlider::handle:vertical {{ width: 14px; height: 14px; margin: 0 -5px;
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

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {p.text_muted}; border: none;
    border-bottom: 2px solid transparent; padding: 8px 14px;
    font-weight: 600; }}
QTabBar::tab:hover {{ color: {p.text_primary}; }}
QTabBar::tab:selected {{ color: {p.text_primary};
    border-bottom-color: {p.accent}; }}

QMenuBar {{ background-color: {p.app_surface}; color: {p.text_primary}; }}
QMenuBar::item {{ background: transparent; padding: 6px 10px;
    border-radius: 4px; }}
QMenuBar::item:selected {{ background-color: {p.control_surface}; }}
QMenu {{ background-color: {p.control_surface}; color: {p.text_primary};
    border: 1px solid {p.border_strong}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 28px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {p.control_hover}; }}
QMenu::item:disabled {{ color: {p.action_disabled_text}; }}
QMenu::separator {{ height: 1px; background-color: {p.border};
    margin: 4px 8px; }}

QProgressBar {{ border: none; border-radius: 4px; text-align: center;
    background-color: {p.control_surface}; color: {p.text_primary}; }}
QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 4px; }}

QSplitter::handle {{ background: transparent; }}
QStatusBar {{ background-color: {p.rail_surface}; color: {p.text_muted}; }}
QStatusBar::item {{ border: none; }}
"""


def apply_application_theme(app: QApplication) -> None:
    """Give the live app, the tests and the screenshot tool the same look."""

    app.setStyle("Fusion")
    app.setPalette(application_palette())
    font = app.font()
    font.setFamilies(list(UI_FONT_FAMILIES))
    app.setFont(font)
    app.setStyleSheet(application_stylesheet())
