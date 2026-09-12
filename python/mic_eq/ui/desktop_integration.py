"""Small Windows desktop integrations used by the main window.

The audio engine stays independent of this module.  The only native hook is
the Windows global hotkey, which is delivered to Qt's main thread and cleaned
up when the window closes.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from ctypes import wintypes

from PyQt6.QtCore import QAbstractNativeEventFilter, QCoreApplication


logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
_HOTKEY_MODIFIERS = {
    "ALT": MOD_ALT,
    "CTRL": MOD_CONTROL,
    "CONTROL": MOD_CONTROL,
    "SHIFT": MOD_SHIFT,
    "WIN": MOD_WIN,
    "WINDOWS": MOD_WIN,
    "META": MOD_WIN,
}
_SPECIAL_KEYS = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "RETURN": 0x0D,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "DELETE": 0x2E,
    "INSERT": 0x2D,
    "HOME": 0x24,
    "END": 0x23,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "UP": 0x26,
    "DOWN": 0x28,
    "LEFT": 0x25,
    "RIGHT": 0x27,
}


def parse_global_hotkey(shortcut: str) -> tuple[int, int]:
    """Parse a compact ``Ctrl+Alt+M`` shortcut into Win32 constants."""

    parts = [part.strip().upper() for part in str(shortcut).split("+")]
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("shortcut must contain a modifier and a key")

    modifiers = 0
    for part in parts[:-1]:
        modifier = _HOTKEY_MODIFIERS.get(part)
        if modifier is None:
            raise ValueError(f"unsupported hotkey modifier: {part}")
        modifiers |= modifier

    key = parts[-1]
    if len(key) == 1 and (key.isascii() and key.isalnum()):
        virtual_key = ord(key)
    elif key in _SPECIAL_KEYS:
        virtual_key = _SPECIAL_KEYS[key]
    elif key.startswith("F") and key[1:].isdigit():
        function_number = int(key[1:])
        if not 1 <= function_number <= 24:
            raise ValueError("function key must be F1 through F24")
        virtual_key = 0x70 + function_number - 1
    else:
        raise ValueError(f"unsupported hotkey key: {key}")
    return modifiers, virtual_key


class _WindowsHotkeyFilter(QAbstractNativeEventFilter):
    """Forward WM_HOTKEY messages without adding a native message loop."""

    def __init__(self, hotkey_id: int, callback: Callable[[], None]) -> None:
        super().__init__()
        self.hotkey_id = hotkey_id
        self.callback = callback

    def nativeEventFilter(self, event_type, message):  # noqa: N802 - Qt API
        if event_type not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            return False, 0
        try:
            msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
            if msg.message == WM_HOTKEY and int(msg.wParam) == self.hotkey_id:
                self.callback()
                return True, 0
        except Exception:
            logger.exception("Global mute hotkey callback failed")
        return False, 0


class GlobalMuteHotkey:
    """Register one process-owned global Windows hotkey."""

    _next_id = 0xAF01

    def __init__(self, shortcut: str, callback: Callable[[], None]) -> None:
        self.shortcut = str(shortcut)
        self.callback = callback
        self._hotkey_id: int | None = None
        self._event_filter: _WindowsHotkeyFilter | None = None

    @property
    def registered(self) -> bool:
        return self._hotkey_id is not None

    def register(self) -> tuple[bool, str | None]:
        """Register the shortcut and return ``(success, error)``."""

        if self.registered:
            return True, None
        if os.name != "nt":
            return False, "global mute shortcuts are supported on Windows only"
        try:
            modifiers, virtual_key = parse_global_hotkey(self.shortcut)
        except ValueError as error:
            return False, str(error)
        app = QCoreApplication.instance()
        if app is None:
            return False, "Qt application is not running"

        hotkey_id = self._allocate_id()
        register_hot_key = ctypes.windll.user32.RegisterHotKey
        register_hot_key.argtypes = [wintypes.HWND, wintypes.INT, wintypes.UINT, wintypes.UINT]
        register_hot_key.restype = wintypes.BOOL
        if not register_hot_key(None, hotkey_id, modifiers | 0x4000, virtual_key):
            error = ctypes.WinError(ctypes.windll.kernel32.GetLastError())
            return False, str(error)

        event_filter = _WindowsHotkeyFilter(hotkey_id, self.callback)
        app.installNativeEventFilter(event_filter)
        self._hotkey_id = hotkey_id
        self._event_filter = event_filter
        return True, None

    def unregister(self) -> None:
        """Release the native registration and Qt event filter."""

        hotkey_id = self._hotkey_id
        if hotkey_id is None:
            return
        app = QCoreApplication.instance()
        if app is not None and self._event_filter is not None:
            app.removeNativeEventFilter(self._event_filter)
        try:
            if os.name == "nt":
                ctypes.windll.user32.UnregisterHotKey(None, hotkey_id)
        finally:
            self._hotkey_id = None
            self._event_filter = None

    @classmethod
    def _allocate_id(cls) -> int:
        hotkey_id = cls._next_id
        cls._next_id += 1
        if cls._next_id > 0xBFFF:
            cls._next_id = 0xAF01
        return hotkey_id


__all__ = ["GlobalMuteHotkey", "parse_global_hotkey"]
