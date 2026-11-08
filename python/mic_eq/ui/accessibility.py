"""Small accessibility helpers."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


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
