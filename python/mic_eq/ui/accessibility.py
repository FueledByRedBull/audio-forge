"""Small accessibility helpers."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtWidgets import QLabel, QWidget


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
