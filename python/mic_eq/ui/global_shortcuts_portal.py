# jeepney is a Linux-only dependency; Windows type checks run without it.
# pyright: reportMissingImports=false
"""Bind the mute shortcut through the XDG GlobalShortcuts portal on Linux.

KDE Plasma, GNOME 48 and later, and Hyprland implement the portal. Each
session lives on its own D-Bus connection in a worker thread; closing the
connection ends the session, because the portal drops a session whose client
has disconnected.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import secrets
import socket
import threading
from typing import Any

from jeepney import DBusAddress, DBusErrorResponse, MatchRule, Message, MessageType, Properties, new_method_call
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import DBusConnection, open_dbus_connection


logger = logging.getLogger(__name__)

PORTAL_PATH = "/org/freedesktop/portal/desktop"
PORTAL_SERVICE = "org.freedesktop.portal.Desktop"
GLOBAL_SHORTCUTS = DBusAddress(PORTAL_PATH, PORTAL_SERVICE, "org.freedesktop.portal.GlobalShortcuts")
REGISTRY = DBusAddress(PORTAL_PATH, PORTAL_SERVICE, "org.freedesktop.host.portal.Registry")
SHORTCUT_ID = "toggle-mute"
# The desktop may ask the user to confirm or pick the key before it answers.
RESPONSE_TIMEOUT = 300.0

# Win32-style names from parse_global_hotkey, as XDG shortcut modifiers and xkb key names.
_MODIFIERS = {
    "ALT": "ALT", "CTRL": "CTRL", "CONTROL": "CTRL", "SHIFT": "SHIFT",
    "WIN": "LOGO", "WINDOWS": "LOGO", "META": "LOGO",
}
_KEYS = {
    "BACKSPACE": "BackSpace", "TAB": "Tab", "ENTER": "Return", "RETURN": "Return",
    "ESC": "Escape", "ESCAPE": "Escape", "SPACE": "space", "DELETE": "Delete",
    "INSERT": "Insert", "HOME": "Home", "END": "End", "PAGEUP": "Page_Up",
    "PAGEDOWN": "Page_Down", "UP": "Up", "DOWN": "Down", "LEFT": "Left", "RIGHT": "Right",
}


class PortalError(RuntimeError):
    """The portal refused or cancelled a request."""


def portal_trigger(shortcut: str) -> str:
    """Translate a ``Ctrl+Alt+M`` shortcut into the portal's ``CTRL+ALT+m`` form."""

    parts = [part.strip().upper() for part in str(shortcut).split("+")]
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("shortcut must contain a modifier and a key")
    modifiers = []
    for part in parts[:-1]:
        if part not in _MODIFIERS:
            raise ValueError(f"unsupported hotkey modifier: {part}")
        modifiers.append(_MODIFIERS[part])
    key = parts[-1]
    if len(key) == 1 and key.isascii() and key.isalnum():
        name = key.lower()
    elif key in _KEYS:
        name = _KEYS[key]
    elif key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        name = key
    else:
        raise ValueError(f"unsupported hotkey key: {key}")
    return "+".join([*modifiers, name])


def _checked(reply: Message) -> Message:
    if reply.header.message_type == MessageType.error:
        raise DBusErrorResponse(reply)
    return reply


def portal_available(connection: DBusConnection | None = None) -> bool:
    """Whether the session's portal implements GlobalShortcuts."""

    try:
        own = connection is None
        connection = connection or open_dbus_connection("SESSION")
        try:
            _checked(connection.send_and_get_reply(Properties(GLOBAL_SHORTCUTS).get("version"), timeout=5))
        finally:
            if own:
                connection.close()
    except (OSError, DBusErrorResponse, TimeoutError) as error:
        logger.info("GlobalShortcuts portal is unavailable: %s", error)
        return False
    return True


class GlobalShortcutsSession:
    """Bind one shortcut and report each activation until closed.

    ``on_activated`` and ``on_failed`` run on the worker thread.
    """

    def __init__(
        self,
        app_id: str,
        trigger: str,
        description: str,
        on_activated: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        self.app_id = app_id
        self.trigger = trigger
        self.description = description
        self.on_activated = on_activated
        self.on_failed = on_failed
        self._connection: DBusConnection | None = None
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, connection: DBusConnection | None = None) -> None:
        self._connection = connection or open_dbus_connection("SESSION")
        self._thread = threading.Thread(target=self._run, name="global-shortcuts-portal", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._connection is None:
            return
        try:
            # Wakes the worker's blocking receive; closing the socket alone does not.
            self._connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._connection.close()

    def _run(self) -> None:
        connection = self._connection
        assert connection is not None
        try:
            self._register_app_id(connection)
            activated = MatchRule(
                type="signal", interface=GLOBAL_SHORTCUTS.interface, member="Activated", path=PORTAL_PATH
            )
            _checked(connection.send_and_get_reply(message_bus.AddMatch(activated)))
            with connection.filter(activated, bufsize=16) as queue:
                session = self._create_session(connection)
                self._bind(connection, session)
                while not self._closed.is_set():
                    signal = connection.recv_until_filtered(queue)
                    if signal.body[0] == session and signal.body[1] == SHORTCUT_ID:
                        self.on_activated()
        except Exception as error:  # noqa: BLE001 - report every failure to the window
            if not self._closed.is_set():
                logger.warning("GlobalShortcuts portal session ended: %s", error)
                self.on_failed(str(error) or type(error).__name__)

    def _register_app_id(self, connection: DBusConnection) -> None:
        # A host app names itself so the desktop can store its shortcuts; older
        # portals lack the registry and identify the app on their own.
        reply = connection.send_and_get_reply(new_method_call(REGISTRY, "Register", "sa{sv}", (self.app_id, {})))
        if reply.header.message_type == MessageType.error:
            logger.info("Portal app registration failed: %s", reply.body)

    def _request(self, connection: DBusConnection, method: str, signature: str, body: tuple) -> dict[str, Any]:
        """Call a portal method that answers through a Request object's Response signal."""

        token = f"audioforge_{secrets.token_hex(8)}"
        sender = (connection.unique_name or "").lstrip(":").replace(".", "_")
        response = MatchRule(
            type="signal",
            interface="org.freedesktop.portal.Request",
            member="Response",
            path=f"{PORTAL_PATH}/request/{sender}/{token}",
        )
        _checked(connection.send_and_get_reply(message_bus.AddMatch(response)))
        options = {**body[-1], "handle_token": ("s", token)}
        with connection.filter(response) as queue:
            _checked(connection.send_and_get_reply(
                new_method_call(GLOBAL_SHORTCUTS, method, signature, (*body[:-1], options))
            ))
            code, results = connection.recv_until_filtered(queue, timeout=RESPONSE_TIMEOUT).body
        if code != 0:
            raise PortalError(f"{method} was {'cancelled' if code == 1 else 'refused'} by the desktop")
        return results

    def _create_session(self, connection: DBusConnection) -> str:
        token = f"audioforge_{secrets.token_hex(8)}"
        results = self._request(connection, "CreateSession", "a{sv}", ({"session_handle_token": ("s", token)},))
        return str(results["session_handle"][1])

    def _bind(self, connection: DBusConnection, session: str) -> None:
        shortcut = (SHORTCUT_ID, {
            "description": ("s", self.description),
            "preferred_trigger": ("s", self.trigger),
        })
        results = self._request(
            connection, "BindShortcuts", "oa(sa{sv})sa{sv}", (session, [shortcut], "", {})
        )
        bound = [identifier for identifier, _ in results.get("shortcuts", ("a(sa{sv})", []))[1]]
        if SHORTCUT_ID not in bound:
            raise PortalError("the desktop did not bind the shortcut")
