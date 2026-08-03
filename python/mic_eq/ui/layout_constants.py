"""Shared layout constants and compact sizing helpers."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QVBoxLayout,
    QWidget,
)

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
    "configure_resizable_dialog",
    "configure_responsive_combo",
    "create_scrollable_dialog_body",
    "fit_spinbox_to_contents",
    "status_chip_style",
]


# Standard spacing constants for consistent UI design
SPACING_TIGHT = 4  # Very related items (label + value)
SPACING_NORMAL = 8  # Related controls in a group
SPACING_SECTION = 16  # Between major sections
MARGIN_PANEL = 12  # Panel content margins


def _spinbox_text_from_value(
    spinbox: QDoubleSpinBox | QSpinBox,
    value: float | int,
) -> str:
    if isinstance(spinbox, QDoubleSpinBox):
        return spinbox.textFromValue(float(value))
    return spinbox.textFromValue(int(value))


def fit_spinbox_to_contents(spinbox: QDoubleSpinBox | QSpinBox) -> int:
    """Give a numeric spin box enough room for its widest legal value and unit."""

    values = (spinbox.minimum(), spinbox.maximum(), spinbox.value())
    texts = (
        f"{spinbox.prefix()}{_spinbox_text_from_value(spinbox, value)}"
        f"{spinbox.suffix()}"
        for value in values
    )
    text_width = max(spinbox.fontMetrics().horizontalAdvance(text) for text in texts)
    option = QStyleOptionSpinBox()
    option.initFrom(spinbox)
    content = QSize(text_width + 10, spinbox.fontMetrics().height() + 8)
    style = spinbox.style()
    if style is None:
        raise RuntimeError("Qt spin box style is unavailable")
    width = style.sizeFromContents(
        QStyle.ContentsType.CT_SpinBox,
        option,
        content,
        spinbox,
    ).width()
    spinbox.setMinimumWidth(width)
    spinbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return width


def configure_responsive_combo(combo: QComboBox, *, minimum_chars: int = 12) -> None:
    """Let a combo elide long options instead of forcing its container wider."""

    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(minimum_chars)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def create_scrollable_dialog_body(
    parent: QDialog,
) -> tuple[QScrollArea, QVBoxLayout]:
    """Create a vertically scrollable dialog body without horizontal overflow."""

    scroll_area = QScrollArea(parent)
    scroll_area.setWidgetResizable(True)
    scroll_area.setFrameShape(QFrame.Shape.NoFrame)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    body = QWidget(scroll_area)
    body_layout = QVBoxLayout(body)
    scroll_area.setWidget(body)
    return scroll_area, body_layout


def configure_resizable_dialog(
    dialog: QDialog,
    *,
    preferred_width: int,
    preferred_height: int,
    minimum_width: int = 420,
    minimum_height: int = 320,
) -> None:
    """Clamp a dialog's initial and minimum size to its current screen."""

    screen = dialog.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        dialog.setMinimumSize(minimum_width, minimum_height)
        dialog.resize(preferred_width, preferred_height)
        dialog.setSizeGripEnabled(True)
        return
    available = screen.availableGeometry()
    max_width = max(320, available.width() - 48)
    max_height = max(240, available.height() - 48)
    width = min(preferred_width, max_width)
    height = min(preferred_height, max_height)
    dialog.setMinimumSize(min(minimum_width, width), min(minimum_height, height))
    dialog.resize(width, height)
    dialog.setSizeGripEnabled(True)
