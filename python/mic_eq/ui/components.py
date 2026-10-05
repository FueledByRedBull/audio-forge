"""Shared building blocks for the card-based interface."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QVariantAnimation
from PySide6.QtGui import QFont, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .layout_constants import MARGIN_PANEL, SPACING_SECTION
from .theme import (
    CARD_STYLE,
    PALETTE,
    RADIUS_CONTROL,
    STEP_CURRENT_STYLE,
    STEP_STYLE,
    prefers_reduced_motion,
    qcolor,
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


# Short stand-ins for machines without a system icon font.
_GLYPH_TEXT = {
    Glyph.MORE: "...",
    Glyph.HELP: "?",
    Glyph.PREVIOUS: "<",
    Glyph.NEXT: ">",
    Glyph.REFRESH: "Refresh",
}


def icon_font(point_size: int = 12) -> QFont | None:
    """Return the system icon font, or ``None`` when Windows does not ship one."""

    for family in ICON_FONT_FAMILIES:
        if QFontDatabase.hasFamily(family):
            font = QFont(family)
            font.setPointSize(point_size)
            return font
    return None


def form_layout(parent: QWidget | None = None) -> QFormLayout:
    """Return a label-and-field layout with the spacing every card uses."""

    form = QFormLayout(parent)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setSpacing(8)
    form.setContentsMargins(0, 0, 0, 0)
    return form


class ToggleSwitch(QCheckBox):
    """A check box painted as a switch; signals and state API are unchanged."""

    TRACK_WIDTH = 36
    TRACK_HEIGHT = 20
    TEXT_GAP = 10

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._position = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(120)
        self._animation.valueChanged.connect(self._set_position)
        self.toggled.connect(self._animate)

    def _set_position(self, value: float) -> None:
        self._position = float(value)
        self.update()

    def _animate(self, checked: bool) -> None:
        target = 1.0 if checked else 0.0
        self._animation.stop()
        if prefers_reduced_motion() or not self.isVisible():
            self._set_position(target)
            return
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(target)
        self._animation.start()

    def sizeHint(self) -> QSize:
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        width = self.TRACK_WIDTH + (self.TEXT_GAP + text_width if text_width else 0)
        return QSize(width + 4, max(self.TRACK_HEIGHT + 4, self.fontMetrics().height()))

    def minimumSizeHint(self) -> QSize:
        # A long label is elided rather than forcing its container wider.
        return QSize(self.TRACK_WIDTH + 4, self.sizeHint().height())

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        track = QRectF(
            2,
            (self.height() - self.TRACK_HEIGHT) / 2,
            self.TRACK_WIDTH,
            self.TRACK_HEIGHT,
        )
        off = qcolor(PALETTE.border_strong)
        on = qcolor(PALETTE.accent)
        blend = self._position
        color = qcolor(PALETTE.border_strong)
        color.setRgbF(
            off.redF() + (on.redF() - off.redF()) * blend,
            off.greenF() + (on.greenF() - off.greenF()) * blend,
            off.blueF() + (on.blueF() - off.blueF()) * blend,
        )
        if not enabled:
            color = qcolor(PALETTE.action_disabled)
        painter.setPen(Qt.PenStyle.NoPen)
        if self.hasFocus():
            painter.setPen(QPen(qcolor(PALETTE.text_primary), 1.5))
        painter.setBrush(color)
        radius = self.TRACK_HEIGHT / 2
        painter.drawRoundedRect(track, radius, radius)

        thumb_radius = radius - 3
        travel = self.TRACK_WIDTH - self.TRACK_HEIGHT
        center = QPointF(track.left() + radius + travel * blend, track.center().y())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(
            qcolor(PALETTE.text_primary if enabled else PALETTE.action_disabled_text)
        )
        painter.drawEllipse(center, thumb_radius, thumb_radius)

        if self.text():
            painter.setPen(
                qcolor(PALETTE.text_primary if enabled else PALETTE.action_disabled_text)
            )
            text_rect = self.rect().adjusted(
                int(track.right()) + self.TEXT_GAP, 0, 0, 0
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self.fontMetrics().elidedText(
                    self.text(), Qt.TextElideMode.ElideRight, text_rect.width()
                ),
            )


def plain_label(text: str) -> str:
    """Menu text as a label: no mnemonic marker, no trailing ellipsis."""

    return text.replace("&&", "\0").replace("&", "").replace("\0", "&").removesuffix("...")


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
        for index, label in enumerate(self._labels):
            label.setStyleSheet(STEP_CURRENT_STYLE if index == step else STEP_STYLE)


class IconButton(QToolButton):
    """A borderless glyph button that always carries an accessible name."""

    def __init__(self, glyph: str, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        font = icon_font()
        if font is None:
            self.setText(_GLYPH_TEXT.get(glyph, name))
        else:
            self.setFont(font)
            self.setText(glyph)
        self.setAccessibleName(name)
        self.setToolTip(name)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoRaise(True)
        self.setStyleSheet(
            f"QToolButton {{ border: none; border-radius: {RADIUS_CONTROL}px; "
            f"padding: 6px; color: {PALETTE.text_muted}; }} "
            f"QToolButton:hover {{ background-color: {PALETTE.control_hover}; "
            f"color: {PALETTE.text_primary}; }} "
            f"QToolButton:focus {{ border: 1px solid {PALETTE.text_primary}; }} "
            "QToolButton::menu-indicator { image: none; }"
        )


class Card(QFrame):
    """A titled surface: optional switch, help and menu in the header, then content."""

    def __init__(
        self,
        title: str,
        *,
        switch: QCheckBox | None = None,
        help_text: str = "",
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
        self.switch = switch
        if switch is not None:
            self.header.addWidget(switch)
        self.title_label = QLabel(title.upper())
        title_font = self.title_label.font()
        title_font.setPointSize(10)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
        self.title_label.setFont(title_font)
        self.header.addWidget(self.title_label)
        self.help_button: IconButton | None = None
        self.menu_button: IconButton | None = None
        if help_text:
            self.help_button = IconButton(Glyph.HELP, f"About {title}")
            self.help_button.setToolTip(help_text)
            self.help_button.clicked.connect(self._show_help)
            self.header.addWidget(self.help_button)
        self.header.addStretch(1)
        outer.addLayout(self.header)

        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        outer.addLayout(self.body)

        self._advanced: QWidget | None = None
        self.advanced_button: QPushButton | None = None
        self._title = title

    def _show_help(self) -> None:
        button = self.help_button
        assert button is not None
        QToolTip.showText(button.mapToGlobal(button.rect().bottomLeft()), button.toolTip())

    def add_header_widget(self, widget: QWidget) -> None:
        self.header.addWidget(widget)

    def set_menu(self, menu: QMenu) -> None:
        """Attach an overflow menu to the right end of the header."""

        button = IconButton(Glyph.MORE, f"{self._title} options")
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button = button
        self.header.addWidget(button)

    def add_advanced(self, widget: QWidget, *, label: str = "Advanced") -> QPushButton:
        """Put ``widget`` behind a disclosure row at the bottom of the card."""

        button = QPushButton(label)
        button.setCheckable(True)
        button.setAccessibleName(f"Show {self._title} {label.lower()} settings")
        button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        button.setStyleSheet(
            f"QPushButton {{ border: none; background: transparent; padding: 4px 0; "
            f"color: {PALETTE.text_muted}; }} "
            f"QPushButton:hover, QPushButton:checked {{ color: {PALETTE.text_primary}; }} "
            f"QPushButton:focus {{ color: {PALETTE.accent}; }}"
        )
        widget.setVisible(False)
        button.toggled.connect(widget.setVisible)
        self.body.addWidget(button)
        self.body.addWidget(widget)
        self._advanced = widget
        self.advanced_button = button
        return button


class NavButton(QPushButton):
    """A navigation rail entry: glyph above a short label."""

    def __init__(self, glyph: str, label: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._glyph = glyph
        self._label = label
        self.setCheckable(True)
        self.setAccessibleName(f"{label} page")
        self.setToolTip(label)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(76, 64)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = self.isChecked()
        if active or self.underMouse() or self.hasFocus():
            painter.fillRect(self.rect(), qcolor(PALETTE.card_surface))
        if active:
            painter.fillRect(0, 0, 3, self.height(), qcolor(PALETTE.accent))
        painter.setPen(qcolor(PALETTE.text_primary if active else PALETTE.text_muted))
        label_rect = self.rect().adjusted(0, 38, 0, -8)
        glyph_font = icon_font(15)
        if glyph_font is None:
            label_rect = self.rect()
        else:
            painter.setFont(glyph_font)
            painter.drawText(
                self.rect().adjusted(0, 10, 0, -26),
                Qt.AlignmentFlag.AlignCenter,
                self._glyph,
            )
        label_font = self.font()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self._label)
