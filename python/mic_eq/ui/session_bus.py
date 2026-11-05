# jeepney is a Linux-only dependency; Windows type checks run without it.
# pyright: reportMissingImports=false
"""Open the D-Bus session bus, and report a missing bus as an ordinary OSError."""

from __future__ import annotations

import os

from jeepney.io.blocking import DBusConnection, open_dbus_connection


def open_session_bus() -> DBusConnection:
    """Connect to the session bus; no usable bus raises OSError.

    jeepney reads DBUS_SESSION_BUS_ADDRESS itself and fails with KeyError when it is
    unset, as it is on a login from a console, ssh or sudo. libdbus then tries the
    bus the login manager runs at $XDG_RUNTIME_DIR/bus, so this does too.
    """

    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    fallback = f"unix:path={_escape_address_value(runtime)}/bus" if os.path.isabs(runtime) else ""
    try:
        return open_dbus_connection(os.environ.get("DBUS_SESSION_BUS_ADDRESS") or fallback)
    except (ValueError, RuntimeError) as error:
        # jeepney's complaint about an empty address or one it has no transport for;
        # a socket that is missing or refuses already raises an OSError.
        raise ConnectionError(f"no usable session bus: {error}") from error


def _escape_address_value(value: str) -> str:
    """Percent-escape a D-Bus address value, so ',', ';', '=' and '%' in a path survive.

    Only ASCII is escaped: jeepney turns each %XX back into one character, which
    would split a multi-byte character.
    """

    safe = r"-_/.\*"
    return "".join(
        char if not char.isascii() or char.isalnum() or char in safe else f"%{ord(char):02x}"
        for char in value
    )
