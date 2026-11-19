"""A small window to keep beside a call: the status, the levels, Mute and the preset.

It renders the main window's own scene model (``QuickBridge.model``), so it adds no engine
and no state; it only knows where it was opened from and whether it stays on top.
"""

from __future__ import annotations

import logging
import math

from PySide6.QtCore import Property, QEvent, Qt, Signal, Slot
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .desktop_integration import activate_window
from .quick_scene import QML_DIR, load_scene
from .quick_shell import register_qml_types

COMPACT_SHORTCUT = "Ctrl+Shift+A"
MINIMUM_WIDTH = 320
MINIMUM_HEIGHT = 200
_LOG = logging.getLogger(__name__)


class CompactWindowLoadError(RuntimeError):
    """The compact scene did not load."""


class CompactWindow(QWidget):
    """The compact call window over the main window's scene model."""

    keepOnTopChanged = Signal()

    def __init__(self, window, bridge) -> None:
        super().__init__(window, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self._window = window
        self._show_main_on_close = False
        self.setWindowTitle("AudioForge")
        # The main window hides in the tray while this one shows; closing this one never quits.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        register_qml_types()
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view, QML_DIR / "CompactWindow.qml", bridge.model["theme"], {"ui": bridge, "host": self}
        )
        if reason is not None:
            self.setParent(None)
            raise CompactWindowLoadError(f"The compact window did not load: {reason}")
        self.quick_view.setAccessibleName("Compact window content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)
        scene = self.quick_view.rootObject()
        self.setMinimumSize(
            max(MINIMUM_WIDTH, math.ceil(scene.property("minimumWidth"))),
            max(MINIMUM_HEIGHT, math.ceil(scene.property("minimumHeight"))),
        )
        self.resize(self.minimumSize())

    keepOnTop = Property(
        bool,
        lambda self: bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint),
        notify=keepOnTopChanged,
    )

    @Slot(bool)
    def setKeepOnTop(self, on: bool) -> None:
        if on == self.keepOnTop:
            return
        shown = self.isVisible()
        # Changing a window flag hides the window; show it again where it was.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        if shown:
            self.show()
        self.keepOnTopChanged.emit()

    def open(self) -> None:
        """Show this window in place of the main one; closing it goes back to how things were."""

        window = self._window
        if not self.isVisible():
            self._show_main_on_close = window.isVisible()
        activate_window(self)
        if window.isVisible():
            tray = window.__dict__.get("_tray_icon")
            if tray is not None and tray.isVisible():
                window._hide_to_tray()
            else:
                window.hide()

    @Slot()
    def expand(self) -> None:
        self._show_main_on_close = True
        self.close()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        if self._show_main_on_close and not self._window.__dict__.get("_quitting"):
            activate_window(self._window)
        self._show_main_on_close = False

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._window._sync_meter_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._window._sync_meter_timer()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._window._sync_meter_timer()

    def release(self) -> None:
        """Close for good: the main scene, whose proxies this one shows, is being dropped."""

        if self.isVisible():
            self.expand()
        self.deleteLater()


def open_compact_window(window) -> None:
    """Open the window's compact window, making it the first time."""

    compact = window.compact_window
    if compact is None:
        bridge = window.__dict__.get("quick_bridge")
        if bridge is None:
            window.status_message.showMessage("The compact window needs the main view, which is not running", 5000)
            return
        try:
            compact = CompactWindow(window, bridge)
        except CompactWindowLoadError:
            _LOG.exception("Compact window could not be shown")
            window.status_message.showMessage("The compact window could not be shown", 5000)
            return
        window.compact_window = compact
    compact.open()
