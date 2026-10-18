"""Small desktop integrations used by the main window.

The audio engine stays independent of this module.  Native work is limited to
the Windows global hotkey and the Qt single-instance activation channel.
"""

from __future__ import annotations

import ctypes
import functools
import getpass
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QLockFile, Qt, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QMainWindow

from .login_startup import LOGIN_STARTUP_ARGUMENT


logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
# Windows shows at most 127 characters of a tray tooltip.
TRAY_TOOLTIP_LIMIT = 127


def _windows_user_sid() -> str:
    """Return the account SID of this process, which no environment variable can change."""

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)
    ]
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    ]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))  # TokenUser
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER starts with the SID pointer.
        sid = ctypes.c_void_p.from_buffer(buffer).value
        text = ctypes.c_void_p()
        if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.wstring_at(text.value or 0)
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.CloseHandle(token)


def _user_scope() -> str:
    """Name part that keeps one account's session lock and activation pipe apart from another's."""

    if os.name == "nt":
        try:
            return _windows_user_sid()
        except OSError:
            logger.warning("Could not read the account SID; scoping by user name", exc_info=True)
    # getuser() trusts LOGNAME and USER, which any process can set.
    try:
        return getpass.getuser().encode("utf-8").hex()
    except Exception:
        return "default"


_USER_SCOPE = _user_scope()
SINGLE_INSTANCE_SERVER = f"AudioForge.SingleInstance.v1.{_USER_SCOPE}"
# Set in the environment of a copy started after the renderer failed.
RENDERER_RELAUNCH_ENV = "AUDIOFORGE_RENDERER_RELAUNCH"
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


@functools.cache
def _user32() -> ctypes.WinDLL:
    # Only a handle opened with use_last_error keeps the error code of the failing call.
    return ctypes.WinDLL("user32", use_last_error=True)


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
        register_hot_key = _user32().RegisterHotKey
        register_hot_key.argtypes = [wintypes.HWND, wintypes.INT, wintypes.UINT, wintypes.UINT]
        register_hot_key.restype = wintypes.BOOL
        if not register_hot_key(None, hotkey_id, modifiers | 0x4000, virtual_key):
            return False, str(ctypes.WinError(ctypes.get_last_error()))

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
                _user32().UnregisterHotKey(None, hotkey_id)
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


def build_tray_tooltip(
    *,
    processing: bool,
    muted: bool,
    health: str,
    route: tuple[str, str] | None = None,
) -> str:
    """Build the short state summary shown by the tray icon.

    The state comes first. The route is last and its device names are shortened
    (or it is left out) so that the whole text fits the tray tooltip.
    """

    state = "Running" if processing else "Stopped"
    if muted:
        state += " / Muted"
    parts = [f"AudioForge: {state}", f"Health: {str(health).strip() or 'unknown'}"]
    summary = " | ".join(parts)
    if route:
        source, destination = (str(name).strip() for name in route)
        room = TRAY_TOOLTIP_LIMIT - len(summary) - len(" |  -> ")
        half = room // 2
        if len(source) <= half:
            destination = _shorten(destination, room - len(source))
        elif len(destination) <= half:
            source = _shorten(source, room - len(destination))
        else:
            source, destination = _shorten(source, half), _shorten(destination, room - half)
        if room >= 8:
            summary += f" | {source} -> {destination}"
    return summary[:TRAY_TOOLTIP_LIMIT]


def _shorten(text: str, room: int) -> str:
    return text if len(text) <= room else text[: max(room - 3, 0)] + "..."


def activate_window(window: QMainWindow) -> None:
    """Show and focus a hidden/minimized Qt window after a second launch."""

    # Only un-minimize; showNormal() would also shrink a maximized window.
    window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
    window.show()
    window.raise_()
    window.activateWindow()


class SingleInstanceCoordinator:
    """Own one process lock and forward later launches to its Qt window.

    ``QLockFile`` supplies stale-lock handling across supported platforms.  The
    local server is only the activation channel; failure to contact an existing
    owner fails closed instead of starting a second audio session.
    """

    ACQUIRED = "acquired"
    FORWARDED = "forwarded"
    ALREADY_RUNNING = "already_running"
    FAILED = "failed"

    def __init__(
        self,
        *,
        server_name: str = SINGLE_INSTANCE_SERVER,
        lock_path: Path | None = None,
    ) -> None:
        self.server_name = str(server_name)
        self._lock_path = lock_path or (
            Path(tempfile.gettempdir()) / f"AudioForge.SingleInstance.{_USER_SCOPE}.lock"
        )
        self._server = QLocalServer()
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._on_new_connection)
        self._server_owned = False
        self._lock_file: QLockFile | None = None
        self._activation_callback: Callable[[], None] | None = None
        self._pending_activation = False
        self.last_error: str | None = None
        self._closed = False

    def acquire(self, *, activate_existing: bool = True) -> str:
        """Acquire this instance or forward activation to the current owner."""

        lock_file = QLockFile(str(self._lock_path))
        lock_file.setStaleLockTime(0)
        if not lock_file.tryLock(0):
            if not activate_existing:
                if lock_file.error() == QLockFile.LockError.LockFailedError:
                    return self.ALREADY_RUNNING
                self.last_error = f"could not acquire the session lock: {lock_file.error().name}"
                return self.FAILED
            return self.FORWARDED if self._request_activation() else self.FAILED
        self._lock_file = lock_file

        if self._listen_for_activation():
            self._closed = False  # a closed coordinator can take the guard again
            return self.ACQUIRED
        lock_file = self._lock_file
        self._lock_file = None
        if lock_file is not None and lock_file.isLocked():
            lock_file.unlock()
        return self.FAILED

    def set_activation_callback(self, callback: Callable[[], None] | None) -> None:
        """Set the primary-window activation hook and flush queued launches."""

        self._activation_callback = callback
        if callback is not None and self._pending_activation:
            self._pending_activation = False
            QTimer.singleShot(0, callback)

    def close(self) -> None:
        """Release the activation channel and process lock exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._server_owned:
            self._server.close()
            QLocalServer.removeServer(self.server_name)
            self._server_owned = False
        lock_file = self._lock_file
        self._lock_file = None
        if lock_file is not None and lock_file.isLocked():
            lock_file.unlock()

    def _listen_for_activation(self) -> bool:
        if self._server.listen(self.server_name):
            self._server_owned = True
            return True

        # The process lock proves no live owner exists here.  A leftover Qt
        # endpoint can therefore be removed safely before retrying once.
        QLocalServer.removeServer(self.server_name)
        if self._server.listen(self.server_name):
            self._server_owned = True
            return True
        self.last_error = f"could not listen for activation: {self._server.errorString()}"
        return False

    def _request_activation(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self.server_name)
        if socket.waitForConnected(500):
            socket.disconnectFromServer()
            return True
        socket.abort()
        self.last_error = (
            f"could not contact the existing AudioForge instance: "
            f"{self.server_name}"
        )
        return False

    def _on_new_connection(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                return
            callback = self._activation_callback
            if callback is None:
                self._pending_activation = True
            else:
                QTimer.singleShot(0, callback)
            socket.disconnected.connect(socket.deleteLater)
            socket.disconnectFromServer()


def relaunch_with_software_rendering(instance: SingleInstanceCoordinator | None) -> bool:
    """Start this app again with software rendering; False if it could not be started.

    The instance guard is released first, or the new copy would only activate this one;
    if the copy cannot be started, the guard is taken back.
    """

    # PyInstaller's argv already starts with the executable; a source run needs Python's own flags.
    arguments = sys.argv if getattr(sys, "frozen", False) else sys.orig_argv
    command = [sys.executable, *(arg for arg in arguments[1:] if arg != LOGIN_STARTUP_ARGUMENT)]
    environment = {
        **os.environ,
        "QT_QUICK_BACKEND": "software",
        RENDERER_RELAUNCH_ENV: "1",
        # Without it a frozen copy takes itself for this one's child and reuses its bundle state.
        "PYINSTALLER_RESET_ENVIRONMENT": "1",
    }
    if instance is not None:
        instance.close()
    try:
        subprocess.Popen(command, env=environment)
    except Exception:  # any failure must end on the error screen, not a silent exit
        logger.exception("Could not start AudioForge with software rendering")
        if instance is not None:
            status = instance.acquire(activate_existing=False)
            if status != instance.ACQUIRED:
                logger.error("Could not take the single-instance guard back: %s", status)
        return False
    return True


__all__ = [
    "GlobalMuteHotkey",
    "SingleInstanceCoordinator",
    "activate_window",
    "build_tray_tooltip",
    "parse_global_hotkey",
    "relaunch_with_software_rendering",
]
