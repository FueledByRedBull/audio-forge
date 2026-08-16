"""Semantic visual tokens for AudioForge's explicit dark interface.

The token names describe intent rather than a particular widget.  Keeping the
palette here prevents dialogs, native Qt controls, and custom-painted widgets
from drifting apart across different Windows appearance settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

from PyQt6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class SemanticPalette:
    """Colors used by the application chrome and its data canvases."""

    app_surface: str = "#191c20"
    control_surface: str = "#2b3139"
    control_surface_alt: str = "#20242a"
    text_primary: str = "#f1f5f9"
    text_muted: str = "#b8c0cc"
    text_on_emphasis: str = "#ffffff"
    accent: str = "#7db7ff"

    action_primary: str = "#2563eb"
    action_primary_border: str = "#60a5fa"
    action_destructive: str = "#b91c1c"
    action_destructive_border: str = "#ef4444"
    action_secondary: str = "#2b3139"
    action_secondary_border: str = "#566170"
    action_disabled: str = "#343a43"
    action_disabled_text: str = "#a4adba"
    action_disabled_surface: str = "#252a31"
    action_disabled_border: str = "#4b5563"

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
    status_idle_surface: str = "#262b33"
    status_idle_text: str = "#cbd5e1"
    status_idle_border: str = "#566170"

    warning_banner_surface: str = "#d97706"
    warning_banner_text: str = "#111827"

    data_surface: str = "#15181c"
    data_surface_raised: str = "#20252b"
    data_grid: str = "#59636f"
    data_text: str = "#d6dde7"
    data_text_muted: str = "#aeb8c5"
    data_curve: str = "#22d3ee"
    data_marker: str = "#facc15"
    data_warning: str = "#fbbf24"
    data_handle_disabled: str = "#9ca3af"
    data_handle_outline: str = "#d7f8ff"
    data_handle_selected: str = "#ffffff"
    data_handle_selected_outline: str = "#111827"

    meter_safe: str = "#4caf50"
    meter_caution: str = "#ffeb3b"
    meter_danger: str = "#f44336"
    meter_clip: str = "#ff0000"
    meter_peak: str = "#ffffff"
    meter_scale: str = "#c8c8c8"
    meter_reduction: str = "#ff9800"


PALETTE = SemanticPalette()


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
    palette.setColor(role.HighlightedText, qcolor(PALETTE.text_on_emphasis))
    palette.setColor(role.Link, qcolor(PALETTE.accent))
    palette.setColor(role.LinkVisited, qcolor("#c4b5fd"))
    palette.setColor(role.PlaceholderText, qcolor(PALETTE.text_muted))
    palette.setColor(role.Light, qcolor("#3b424c"))
    palette.setColor(role.Midlight, qcolor("#333a44"))
    palette.setColor(role.Mid, qcolor("#252a31"))
    palette.setColor(role.Dark, qcolor("#111318"))
    palette.setColor(role.Shadow, qcolor("#0d0f12"))

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
    palette.setColor(disabled, role.Highlight, qcolor("#38465c"))
    palette.setColor(disabled, role.HighlightedText, qcolor("#d1d5db"))
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
    ("primary action", PALETTE.text_on_emphasis, PALETTE.action_primary),
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
COMPACT_CONTROL_STYLE = "font-size: 9pt;"
METER_LABEL_STYLE = f"font-size: 10pt; font-weight: bold; color: {PALETTE.accent};"
INFO_LABEL_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
SUBDUED_TEXT_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
DESCRIPTION_LABEL_STYLE = f"color: {PALETTE.text_muted}; font-size: 9pt; padding: 5px;"
PROGRESS_LABEL_STYLE = f"font-size: 12pt; color: {PALETTE.accent}; font-weight: bold;"

PRIMARY_ACTION_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {PALETTE.action_primary}; "
    f"color: {PALETTE.text_on_emphasis}; font-weight: 600; "
    f"border: 1px solid {PALETTE.action_primary_border}; "
    "border-radius: 6px; padding: 8px 16px; } "
    f"QPushButton:disabled {{ background-color: {PALETTE.action_disabled}; "
    f"color: {PALETTE.action_disabled_text}; "
    f"border-color: {PALETTE.action_disabled}; }}"
)

DESTRUCTIVE_ACTION_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {PALETTE.action_destructive}; "
    f"color: {PALETTE.text_on_emphasis}; font-weight: 600; "
    f"border: 1px solid {PALETTE.action_destructive_border}; "
    "border-radius: 6px; padding: 8px 16px; } "
    f"QPushButton:disabled {{ background-color: {PALETTE.action_disabled}; "
    f"color: {PALETTE.action_disabled_text}; "
    f"border-color: {PALETTE.action_disabled}; }}"
)

SECONDARY_ACTION_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {PALETTE.action_secondary}; "
    f"color: {PALETTE.text_primary}; font-weight: 600; "
    f"border: 1px solid {PALETTE.action_secondary_border}; "
    "border-radius: 6px; padding: 8px 16px; } "
    f"QPushButton:disabled {{ background-color: {PALETTE.action_disabled_surface}; "
    f"color: {PALETTE.action_disabled_text}; "
    f"border-color: {PALETTE.action_disabled_border}; }}"
)

WARNING_BANNER_STYLE = (
    f"QLabel {{ background-color: {PALETTE.warning_banner_surface}; "
    f"color: {PALETTE.warning_banner_text}; padding: 10px 12px; "
    "font-weight: 600; border-radius: 6px; }"
)

PROGRESS_BAR_STYLE = (
    f"QProgressBar {{ border: 2px solid {PALETTE.data_grid}; "
    f"border-radius: 5px; background-color: {PALETTE.data_surface}; }} "
    f"QProgressBar::chunk {{ background-color: {PALETTE.meter_safe}; "
    "border-radius: 3px; }}"
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
        "border-radius: 6px; "
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
