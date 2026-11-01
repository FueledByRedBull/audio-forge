# jeepney is a Linux-only dependency; Windows type checks run without it.
# pyright: reportMissingImports=false
"""Serve org.freedesktop.Application for the running AudioForge on Linux.

Desktops, ``gapplication`` and D-Bus activation reach the app through it:
``Activate`` and ``Open`` raise the window, and ``ActivateAction`` runs one of
the desktop entry's actions, such as ``toggle-mute``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import logging
import socket
import threading

from jeepney import (
    DBusErrorResponse, HeaderFields, MatchRule, MessageFlag, MessageType, new_error, new_method_return,
)
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import DBusConnection
from PySide6.QtCore import QObject, Signal

from .session_bus import open_session_bus


logger = logging.getLogger(__name__)

INTERFACE = "org.freedesktop.Application"
_PRIMARY_OWNER = 1
_DO_NOT_QUEUE = 4
INTROSPECTION = """<!DOCTYPE node PUBLIC "-//freedesktop//DTD D-BUS Object Introspection 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd">
<node>
  <interface name="org.freedesktop.Application">
    <method name="Activate"><arg type="a{sv}" name="platform_data" direction="in"/></method>
    <method name="Open">
      <arg type="as" name="uris" direction="in"/>
      <arg type="a{sv}" name="platform_data" direction="in"/>
    </method>
    <method name="ActivateAction">
      <arg type="s" name="action_name" direction="in"/>
      <arg type="av" name="parameter" direction="in"/>
      <arg type="a{sv}" name="platform_data" direction="in"/>
    </method>
  </interface>
</node>
"""


def object_path(app_id: str) -> str:
    """The object path the Desktop Entry spec derives from an app id."""

    return "/" + app_id.replace(".", "/").replace("-", "_")


class _Bridge(QObject):
    called = Signal(str)


class ApplicationBusService:
    """Own ``app_id`` on the session bus and answer org.freedesktop.Application.

    ``on_activate`` and the ``actions`` callbacks run on the Qt thread that
    started the service.
    """

    def __init__(self, app_id: str, on_activate: Callable[[], None], actions: dict[str, Callable[[], None]]):
        self.app_id = app_id
        self.path = object_path(app_id)
        self.on_activate = on_activate
        self.actions = actions
        self._bridge = _Bridge()
        self._bridge.called.connect(self._dispatch)
        self._connection: DBusConnection | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._early: deque = deque(maxlen=64)

    def start(self) -> bool:
        """Claim the bus name; False when the bus or the name is unavailable."""

        self._closed.clear()  # a service closed for a relaunch can take its name back
        self._early.clear()
        try:
            connection = open_session_bus()
        except OSError as error:
            logger.info("No session bus for %s: %s", INTERFACE, error)
            return False
        try:
            # The bus hands a D-Bus launch's queued calls over with the name;
            # send_and_get_reply would drop them, so they wait here for _serve.
            with connection.filter(MatchRule(type="method_call"), queue=self._early):
                reply = connection.send_and_get_reply(
                    message_bus.RequestName(self.app_id, _DO_NOT_QUEUE), timeout=5
                )
            if reply.header.message_type == MessageType.error:
                raise DBusErrorResponse(reply)
        except (OSError, DBusErrorResponse, TimeoutError) as error:
            logger.info("Could not claim %s on the session bus: %s", self.app_id, error)
            connection.close()
            return False
        if reply.body[0] != _PRIMARY_OWNER:
            logger.info("%s is already owned on the session bus", self.app_id)
            connection.close()
            return False
        self._connection = connection
        self._thread = threading.Thread(target=self._serve, name="application-bus", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        self._closed.set()
        if self._connection is None:
            return
        try:
            # Wakes the worker's blocking receive; closing the socket alone does not.
            self._connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._connection.close()
        self._connection = None

    def _dispatch(self, action: str) -> None:
        callback = self.on_activate if not action else self.actions[action]
        try:
            callback()
        except Exception:
            logger.exception("%s call %r failed", INTERFACE, action or "Activate")

    def _serve(self) -> None:
        connection = self._connection
        assert connection is not None
        while not self._closed.is_set():
            try:
                message = self._early.popleft() if self._early else connection.receive()
            except (OSError, ConnectionError):
                return
            header = message.header
            if header.message_type != MessageType.method_call:
                continue
            try:
                reply, action = self._answer(message)
            except Exception:  # a bad call must not end the service
                logger.exception("%s call %r failed", INTERFACE, header.fields.get(HeaderFields.member))
                reply, action = new_error(message, "org.freedesktop.DBus.Error.Failed"), None
            if not header.flags & MessageFlag.no_reply_expected:
                try:
                    connection.send(reply)
                except (OSError, ConnectionError):
                    return
            # Only after the reply: the callback may quit the app and close the bus.
            if action is not None:
                self._bridge.called.emit(action)

    def _answer(self, message):
        fields = message.header.fields
        interface = fields.get(HeaderFields.interface)
        member = fields.get(HeaderFields.member)
        if fields.get(HeaderFields.path) != self.path:
            return new_error(message, "org.freedesktop.DBus.Error.UnknownObject"), None
        if member == "Introspect" and interface in (None, "org.freedesktop.DBus.Introspectable"):
            return new_method_return(message, "s", (INTROSPECTION,)), None
        if member == "Ping" and interface in (None, "org.freedesktop.DBus.Peer"):
            return new_method_return(message), None
        if interface not in (None, INTERFACE):
            return new_error(message, "org.freedesktop.DBus.Error.UnknownInterface"), None
        if member in ("Activate", "Open"):
            return new_method_return(message), ""
        if member == "ActivateAction":
            action = message.body[0] if message.body else None
            if not isinstance(action, str) or action not in self.actions:
                return new_error(message, "org.freedesktop.DBus.Error.InvalidArgs", "s", (f"no action {action!r}",)), None
            return new_method_return(message), action
        return new_error(message, "org.freedesktop.DBus.Error.UnknownMethod"), None
