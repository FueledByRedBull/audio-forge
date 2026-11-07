"""Deterministic widget-tree audit shared by the accessibility tests."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
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
