"""Shared shell presentation; widgets and Quick consume the same notifications."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Property, QPoint, QSignalBlocker, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import QComboBox, QLabel, QMenu

from .components import plain_label
from .theme import status_chip_style


class TextState(QObject):
    """Application-owned text, visibility and status, independent of its views."""

    changed = Signal()

    def __init__(self, parent: QObject, *, text: str = "", name: str = "", tip: str = "", shown: bool = True):
        super().__init__(parent)
        self._text, self._name, self._tip = text, name, tip
        self._shown, self._status, self._description = shown, "", ""
        self._data: dict[str, Any] = {}

    text = Property(str, lambda self: self._text, notify=changed)
    name = Property(str, lambda self: self._name, constant=True)
    toolTip = Property(str, lambda self: self._tip, notify=changed)
    shown = Property(bool, lambda self: self._shown, notify=changed)
    enabled = Property(bool, lambda self: True, constant=True)
    state = Property(str, lambda self: self._status, notify=changed)
    data = Property(dict, lambda self: self._data, notify=changed)

    def update(self, *, text: str | None = None, tip: str | None = None,
               shown: bool | None = None, status: str | None = None,
               description: str | None = None, data: dict | None = None) -> None:
        changed = False
        for key, value in (("_text", text), ("_tip", tip), ("_shown", shown),
                           ("_status", status), ("_description", description), ("_data", data)):
            if value is not None and getattr(self, key) != value:
                setattr(self, key, value)
                changed = True
        if changed:
            self.changed.emit()

    def bind_label(self, label: QLabel) -> None:
        def render() -> None:
            label.setText(self._text)
            label.setToolTip(self._tip)
            label.setVisible(self._shown)
            label.setAccessibleDescription(self._description)
            if self._status and label.property("health_state") != self._status:
                label.setStyleSheet(status_chip_style(self._status))
                label.setProperty("health_state", self._status)
        self.changed.connect(render)
        render()


class StatusState(TextState):
    """Transient status messages with one shared expiry for both views."""

    def __init__(self, parent: QObject, text: str):
        super().__init__(parent, text=text)
        self._expiry = QTimer(self)
        self._expiry.setSingleShot(True)
        self._expiry.timeout.connect(self.clearMessage)

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self._expiry.stop()
        self.update(text=message)
        if timeout > 0:
            self._expiry.start(timeout)

    def clearMessage(self) -> None:
        self._expiry.stop()
        self.update(text="")


class ChoiceState(QObject):
    """Choice identities and selected index; a combo box is only a renderer."""

    changed = Signal()
    currentIndexChanged = Signal(int)

    def __init__(self, parent: QObject, *, name: str = "", tip: str = ""):
        super().__init__(parent)
        self._name, self._tip = name, tip
        self._options: list[tuple[str, Any]] = []
        self._index, self._enabled, self._placeholder = -1, True, ""

    name = Property(str, lambda self: self._name, constant=True)
    toolTip = Property(str, lambda self: self._tip, constant=True)
    enabled = Property(bool, lambda self: self._enabled, notify=changed)
    shown = Property(bool, lambda self: True, constant=True)
    items = Property(list, lambda self: [label for label, _ in self._options], notify=changed)
    index = Property(int, lambda self: self._index, notify=changed)
    text = Property(str, lambda self: self.currentText(), notify=changed)

    def count(self) -> int:
        return len(self._options)

    def itemData(self, index: int) -> Any:
        return self._options[index][1] if 0 <= index < self.count() else None

    def itemText(self, index: int) -> str:
        return self._options[index][0] if 0 <= index < self.count() else ""

    def currentData(self) -> Any:
        return self.itemData(self._index)

    def currentText(self) -> str:
        return self.itemText(self._index)

    def currentIndex(self) -> int:
        return self._index

    def findData(self, value: Any) -> int:
        return next((i for i, (_, data) in enumerate(self._options) if data == value), -1)

    def setCurrentIndex(self, index: int) -> None:
        index = index if 0 <= index < self.count() else -1
        if index != self._index:
            self._index = index
            self.changed.emit()
            self.currentIndexChanged.emit(index)

    @Slot(int)
    def setIndex(self, index: int) -> None:
        if self._enabled and 0 <= index < self.count():
            self.setCurrentIndex(index)
        # Read back even if application validation restored the previous value.
        self.changed.emit()

    def clear(self) -> None:
        self._options.clear()
        self.setCurrentIndex(-1)
        self.changed.emit()

    def addItem(self, text: str, userData: Any = None) -> None:
        self._options.append((text, userData))
        if self._index < 0 and self.count() == 1:
            self.setCurrentIndex(0)
        self.changed.emit()

    def setEnabled(self, enabled: bool) -> None:
        if self._enabled != enabled:
            self._enabled = enabled
            self.changed.emit()

    def setPlaceholderText(self, text: str) -> None:
        self._placeholder = text
        self.changed.emit()

    def blockSignals(self, block: bool) -> bool:
        previous = super().blockSignals(block)
        if previous and not block:
            self.changed.emit()
        return previous

    def bind_combo(self, combo: QComboBox) -> None:
        def render() -> None:
            with QSignalBlocker(combo):
                items = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
                if items != self._options:
                    combo.clear()
                    for label, data in self._options:
                        combo.addItem(label, data)
                combo.setCurrentIndex(self._index)
                combo.setEnabled(self._enabled)
                combo.setPlaceholderText(self._placeholder)
        self.changed.connect(render)
        combo.currentIndexChanged.connect(self.setIndex)
        render()


class ActionControl(QObject):
    """Expose Qt's shared command primitive to Quick, including anchored menus."""

    changed = Signal()

    def __init__(self, action: QAction, parent: QObject, *, name: str = "", text: str | None = None,
                 menu: QMenu | None = None):
        super().__init__(parent)
        self.action, self._name, self._menu = action, name, menu
        self._text = text
        action.changed.connect(self.changed)

    text = Property(str, lambda self: self._text if self._text is not None else plain_label(self.action.text()), notify=changed)
    name = Property(str, lambda self: self._name, constant=True)
    toolTip = Property(str, lambda self: self.action.toolTip(), notify=changed)
    enabled = Property(bool, lambda self: self.action.isEnabled(), notify=changed)
    shown = Property(bool, lambda self: self.action.isVisible(), notify=changed)
    checked = Property(bool, lambda self: self.action.isChecked(), notify=changed)

    @Slot()
    def click(self) -> None:
        self._activate(QCursor.pos())

    @Slot(float, float)
    def clickAt(self, x: float, y: float) -> None:
        self._activate(QPoint(round(x), round(y)))

    def _activate(self, position: QPoint) -> None:
        if self.action.isEnabled():
            if self._menu is not None:
                self._menu.popup(position)
            else:
                self.action.trigger()
        self.changed.emit()


class PageState(QObject):
    changed = Signal()
    selected = Signal(int)

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._index = 0

    index = Property(int, lambda self: self._index, notify=changed)

    @Slot(int)
    def setIndex(self, index: int) -> None:
        if 0 <= index < 3 and index != self._index:
            self._index = index
            self.changed.emit()
            self.selected.emit(index)
