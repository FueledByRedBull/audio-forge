"""Shared building blocks for the card-based interface."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .capture_session import LIVE_OUTPUT_MUTED_NOTICE
from .layout_constants import MARGIN_PANEL, SPACING_SECTION
from .theme import (
    CARD_STYLE,
    STEP_CURRENT_STYLE,
    STEP_STYLE,
    status_chip_style,
)

SPACING_CARD = 16
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


def live_output_muted_notice() -> QLabel:
    """The warning chip a dialog shows while its recording mutes the live output."""

    label = QLabel(LIVE_OUTPUT_MUTED_NOTICE)
    label.setWordWrap(True)
    label.setStyleSheet(status_chip_style("warn"))
    return label


def session_is_ending() -> bool:
    """Whether Windows is signing out or shutting down, which the main window says on the app."""

    app = QApplication.instance()
    return app is not None and app.property("audioforge_session_ending") is True


def update_label(label: QLabel, text: str, style: str) -> None:
    """Set a label's text and style only when they changed, for callers that run on a timer."""

    if label.text() != text:
        label.setText(text)
    if label.styleSheet() != style:
        label.setStyleSheet(style)


class StepHeader(QWidget):
    """The row of step names at the top of a guided dialog."""

    def __init__(self, steps: tuple[str, ...], parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(MARGIN_PANEL, MARGIN_PANEL, MARGIN_PANEL, 0)
        layout.setSpacing(SPACING_SECTION)
        self._labels = [QLabel(name) for name in steps]
        for label in self._labels:
            layout.addWidget(label)
        layout.addStretch(1)
        self.set_current(0)

    def set_current(self, step: int) -> None:
        if 0 <= step < len(self._labels):
            self.setAccessibleName(f"Step {step + 1} of {len(self._labels)}: {self._labels[step].text()}")
        for index, label in enumerate(self._labels):
            label.setStyleSheet(STEP_CURRENT_STYLE if index == step else STEP_STYLE)


class Card(QFrame):
    """A titled surface: a title in the header, then content."""

    def __init__(
        self,
        title: str,
        *,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(CARD_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING_CARD, 12, SPACING_CARD, SPACING_CARD)
        outer.setSpacing(12)

        self.header = QHBoxLayout()
        self.header.setSpacing(10)
        self.title_label = QLabel(title.upper())
        title_font = self.title_label.font()
        title_font.setPointSize(10)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
        self.title_label.setFont(title_font)
        self.header.addWidget(self.title_label)
        self.header.addStretch(1)
        outer.addLayout(self.header)

        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.body)
