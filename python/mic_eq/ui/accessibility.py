"""Small accessibility helpers and deterministic widget-tree auditing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QProgressBar,
    QSlider,
    QSpinBox,
    QTextEdit,
    QWidget,
)


NAMED_CONTROL_TYPES = (
    QComboBox,
    QDoubleSpinBox,
    QProgressBar,
    QSlider,
    QSpinBox,
    QTextEdit,
)


@dataclass(frozen=True)
class AccessibilityIssue:
    kind: str
    widget_type: str
    object_name: str


def set_accessible(
    widget: QWidget,
    name: str,
    description: str | None = None,
) -> None:
    """Assign a concise name and optional longer assistive description."""

    normalized = " ".join(name.replace("&", "").split())
    if not normalized:
        raise ValueError("Accessible names must not be empty")
    widget.setAccessibleName(normalized)
    if description:
        widget.setAccessibleDescription(" ".join(description.split()))


def bind_label(
    label: QLabel,
    widget: QWidget,
    *,
    name: str | None = None,
    description: str | None = None,
) -> QLabel:
    """Bind a visible label to a control and expose the same accessible name."""

    label.setBuddy(widget)
    set_accessible(widget, name or label.text(), description)
    return label


def set_accessible_group(
    controls: Iterable[tuple[QWidget, str, str | None]],
) -> None:
    for widget, name, description in controls:
        set_accessible(widget, name, description)


def _button_has_name(widget: QAbstractButton) -> bool:
    return bool(widget.accessibleName().strip() or widget.text().replace("&", "").strip())


def audit_widget_tree(root: QWidget) -> tuple[AccessibilityIssue, ...]:
    """Return missing-name/focus issues for user-operable descendants."""

    issues: list[AccessibilityIssue] = []
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if widget.objectName().startswith("qt_"):
            continue
        if isinstance(widget, QAbstractButton):
            named = _button_has_name(widget)
        elif isinstance(widget, NAMED_CONTROL_TYPES):
            named = bool(widget.accessibleName().strip())
        else:
            continue
        if not named:
            issues.append(
                AccessibilityIssue(
                    kind="missing-accessible-name",
                    widget_type=type(widget).__name__,
                    object_name=widget.objectName(),
                )
            )
    return tuple(issues)
