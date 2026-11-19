"""A small window to keep beside a call: the status, the levels, Mute and the preset.

It renders the main window's own scene model (``QuickBridge.model``), so it adds no engine
and no state; it only knows whether it stays on top. The main window opens and closes it.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Property, QEvent, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtQml import QQmlProperty
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .quick_scene import QML_DIR, load_scene
from .quick_shell import register_qml_types

COMPACT_SHORTCUT = "Ctrl+Shift+A"
MINIMUM_WIDTH = 320
MINIMUM_HEIGHT = 200


class CompactWindowLoadError(RuntimeError):
    """The compact scene did not load."""


class CompactWindow(QWidget):
    """The compact call window over the main window's scene model.

    A top-level window of its own, so the taskbar and Alt+Tab list it while the main
    window is hidden; the main window owns its lifetime.
    """

    keepOnTopChanged = Signal()
    expandRequested = Signal()
    closed = Signal()
    # Shown, hidden, minimized or restored: the meters run only while it is on screen.
    shownChanged = Signal()

    def __init__(self, bridge) -> None:
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("AudioForge")
        # Closing it while the main window is hidden must not quit the app.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        register_qml_types()
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view, QML_DIR / "CompactWindow.qml", bridge.model["theme"], {"ui": bridge, "host": self}
        )
        if reason is not None:
            raise CompactWindowLoadError(f"The compact window did not load: {reason}")
        self.quick_view.setAccessibleName("Compact window content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)
        # The shortcut that opened it closes it again, back to where it came from.
        self.close_action = QAction("Close compact window", self)
        self.close_action.setShortcut(COMPACT_SHORTCUT)
        self.close_action.setAutoRepeat(False)
        self.close_action.triggered.connect(self.close)
        self.addAction(self.close_action)
        scene = self.quick_view.rootObject()
        follow = self.metaObject().indexOfMethod("_follow_scene_minimum()")
        for name in ("minimumWidth", "minimumHeight"):
            QQmlProperty(scene, name).connectNotifySignal(self, follow)
        self._follow_scene_minimum()
        self.resize(self.minimumSize())

    @Slot()
    def _follow_scene_minimum(self) -> None:
        scene = self.quick_view.rootObject()
        self.setMinimumSize(
            max(MINIMUM_WIDTH, math.ceil(scene.property("minimumWidth"))),
            max(MINIMUM_HEIGHT, math.ceil(scene.property("minimumHeight"))),
        )

    keepOnTop = Property(
        bool,
        lambda self: bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint),
        notify=keepOnTopChanged,
    )

    @Slot(bool)
    def setKeepOnTop(self, on: bool) -> None:
        if on == self.keepOnTop:
            return
        flags = self.windowFlags()
        flags = flags | Qt.WindowType.WindowStaysOnTopHint if on else flags & ~Qt.WindowType.WindowStaysOnTopHint
        handle = self.windowHandle()
        if handle is None:
            self.setWindowFlags(flags)
        else:
            # QWidget.setWindowFlags would recreate the native window, which hides it and loses
            # its place; the native window takes the hint as it is.
            self.overrideWindowFlags(flags)
            handle.setFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.keepOnTopChanged.emit()

    @Slot()
    def expand(self) -> None:
        self.expandRequested.emit()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        self.closed.emit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.shownChanged.emit()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.shownChanged.emit()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self.shownChanged.emit()
