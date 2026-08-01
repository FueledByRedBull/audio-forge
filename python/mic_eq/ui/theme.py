"""Semantic visual tokens for the current AudioForge light interface.

The token names describe intent rather than a particular widget.  Keeping the
palette here prevents dialogs and custom-painted widgets from drifting apart,
while deliberately leaving a future dark-theme redesign as separate work.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

from PyQt6.QtGui import QColor


@dataclass(frozen=True)
class SemanticPalette:
    """Colors used by the current light UI and its dark data canvases."""

    app_surface: str = "#ffffff"
    text_primary: str = "#1f2933"
    text_muted: str = "#475569"
    text_on_emphasis: str = "#ffffff"
    accent: str = "#1d4ed8"

    action_primary: str = "#2563eb"
    action_primary_border: str = "#1d4ed8"
    action_destructive: str = "#dc2626"
    action_destructive_border: str = "#b91c1c"
    action_secondary: str = "#eef2f7"
    action_secondary_border: str = "#c8cdd4"
    action_disabled: str = "#cbd5e1"
    action_disabled_text: str = "#64748b"
    action_disabled_surface: str = "#f8fafc"
    action_disabled_border: str = "#e2e8f0"

    status_ok_surface: str = "#ecfdf5"
    status_ok_text: str = "#047857"
    status_ok_border: str = "#a7f3d0"
    status_warn_surface: str = "#fffbeb"
    status_warn_text: str = "#92400e"
    status_warn_border: str = "#fcd34d"
    status_bad_surface: str = "#fef2f2"
    status_bad_text: str = "#b91c1c"
    status_bad_border: str = "#fecaca"
    status_info_surface: str = "#eff6ff"
    status_info_text: str = "#1d4ed8"
    status_info_border: str = "#bfdbfe"
    status_idle_surface: str = "#f8fafc"
    status_idle_text: str = "#475569"
    status_idle_border: str = "#cbd5e1"

    warning_banner_surface: str = "#f59e0b"
    warning_banner_text: str = "#111827"

    data_surface: str = "#20242a"
    data_surface_raised: str = "#2a3038"
    data_grid: str = "#66707d"
    data_text: str = "#d6dde7"
    data_text_muted: str = "#aeb8c5"
    data_curve: str = "#22d3ee"
    data_curve_overlay: str = "#fb923c"
    data_curve_current: str = "#86efac"
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
METER_LABEL_STYLE = (
    f"font-size: 10pt; font-weight: bold; color: {PALETTE.accent};"
)
INFO_LABEL_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
SUBDUED_TEXT_STYLE = f"font-size: 9pt; color: {PALETTE.text_muted};"
DESCRIPTION_LABEL_STYLE = (
    f"color: {PALETTE.text_muted}; font-size: 9pt; padding: 5px;"
)
PROGRESS_LABEL_STYLE = (
    f"font-size: 12pt; color: {PALETTE.accent}; font-weight: bold;"
)

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
