"""Shared layout constants and compatibility exports for UI theme styles."""

from .theme import (
    DESTRUCTIVE_ACTION_BUTTON_STYLE,
    INFO_LABEL_STYLE,
    METER_LABEL_STYLE,
    PRIMARY_ACTION_BUTTON_STYLE,
    PRIMARY_LABEL_STYLE,
    SECONDARY_ACTION_BUTTON_STYLE,
    SUBDUED_TEXT_STYLE,
    WARNING_BANNER_STYLE,
    status_chip_style,
)

__all__ = [
    "DESTRUCTIVE_ACTION_BUTTON_STYLE",
    "INFO_LABEL_STYLE",
    "MARGIN_PANEL",
    "METER_LABEL_STYLE",
    "PRIMARY_ACTION_BUTTON_STYLE",
    "PRIMARY_LABEL_STYLE",
    "SECONDARY_ACTION_BUTTON_STYLE",
    "SPACING_NORMAL",
    "SPACING_SECTION",
    "SPACING_TIGHT",
    "SUBDUED_TEXT_STYLE",
    "WARNING_BANNER_STYLE",
    "status_chip_style",
]


# Standard spacing constants for consistent UI design
SPACING_TIGHT = 4      # Very related items (label + value)
SPACING_NORMAL = 8     # Related controls in a group
SPACING_SECTION = 16   # Between major sections
MARGIN_PANEL = 12      # Panel content margins
