"""
Shared spacing and typography constants for consistent UI design.

This module provides a single source of truth for spacing values and
font styling across all UI panels to ensure visual consistency.
"""


# Standard spacing constants for consistent UI design
SPACING_TIGHT = 4      # Very related items (label + value)
SPACING_NORMAL = 8     # Related controls in a group
SPACING_SECTION = 16   # Between major sections
MARGIN_PANEL = 12      # Panel content margins


# Typography styles for visual hierarchy
PRIMARY_LABEL_STYLE = "font-size: 11pt;"
METER_LABEL_STYLE = "font-size: 10pt; font-weight: bold; color: #4a90d9;"
INFO_LABEL_STYLE = "font-size: 9pt; color: gray;"
SUBDUED_TEXT_STYLE = "font-size: 10px; color: #6f7782;"

PRIMARY_ACTION_BUTTON_STYLE = (
    "QPushButton { background-color: #2563eb; color: white; font-weight: 600; "
    "border: 1px solid #1d4ed8; border-radius: 6px; padding: 8px 16px; } "
    "QPushButton:disabled { background-color: #cbd5e1; color: #64748b; border-color: #cbd5e1; }"
)

DESTRUCTIVE_ACTION_BUTTON_STYLE = (
    "QPushButton { background-color: #dc2626; color: white; font-weight: 600; "
    "border: 1px solid #b91c1c; border-radius: 6px; padding: 8px 16px; } "
    "QPushButton:disabled { background-color: #cbd5e1; color: #64748b; border-color: #cbd5e1; }"
)

SECONDARY_ACTION_BUTTON_STYLE = (
    "QPushButton { background-color: #eef2f7; color: #1f2933; font-weight: 600; "
    "border: 1px solid #c8cdd4; border-radius: 6px; padding: 8px 16px; } "
    "QPushButton:disabled { background-color: #f8fafc; color: #94a3b8; border-color: #e2e8f0; }"
)

WARNING_BANNER_STYLE = (
    "QLabel { background-color: #f59e0b; color: #111827; padding: 10px 12px; "
    "font-weight: 600; border-radius: 6px; }"
)


def status_chip_style(state: str) -> str:
    palette = {
        "ok": ("#ecfdf5", "#047857", "#a7f3d0"),
        "warn": ("#fffbeb", "#b45309", "#fcd34d"),
        "bad": ("#fef2f2", "#b91c1c", "#fecaca"),
        "info": ("#eff6ff", "#1d4ed8", "#bfdbfe"),
        "idle": ("#f8fafc", "#475569", "#cbd5e1"),
    }
    background, foreground, border = palette.get(state, palette["idle"])
    return (
        "QLabel { "
        f"background-color: {background}; "
        f"color: {foreground}; "
        f"border: 1px solid {border}; "
        "border-radius: 6px; "
        "padding: 6px 10px; "
        "font-size: 11px; "
        "font-weight: 600; }"
    )
