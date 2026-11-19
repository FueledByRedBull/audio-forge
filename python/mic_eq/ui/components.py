"""Shared building blocks for the card-based interface."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

ICON_FONT_FAMILIES = ("Segoe Fluent Icons", "Segoe MDL2 Assets")


class Glyph:
    """Code points shared by both Windows system icon fonts."""

    MICROPHONE = ""
    HEALTH = ""
    SETTINGS = ""
    MORE = ""
    HELP = ""
    PREVIOUS = ""
    NEXT = ""
    EXPAND = ""
    REFRESH = ""
    UNDO = ""
    REDO = ""


def icon_font(point_size: int = 12) -> QFont | None:
    """Return the system icon font, or ``None`` when Windows does not ship one."""

    for family in ICON_FONT_FAMILIES:
        if QFontDatabase.hasFamily(family):
            font = QFont(family)
            font.setPointSize(point_size)
            return font
    return None


def plain_label(text: str) -> str:
    """Menu text as a label: no mnemonic marker, no trailing ellipsis."""

    return text.replace("&&", "\0").replace("&", "").replace("\0", "&").removesuffix("...")


def session_is_ending() -> bool:
    """Whether Windows is signing out or shutting down, which the main window says on the app."""

    app = QApplication.instance()
    return app is not None and app.property("audioforge_session_ending") is True
