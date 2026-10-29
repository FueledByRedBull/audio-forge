"""The GlobalShortcuts portal client binds the mute shortcut and reports activations."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading

import pytest

if sys.platform != "linux":
    pytest.skip("the GlobalShortcuts portal is Linux-only", allow_module_level=True)

from jeepney import DBusAddress, HeaderFields, MessageType, new_error, new_method_return, new_signal  # noqa: E402
from jeepney.bus_messages import message_bus  # noqa: E402
from jeepney.io.blocking import open_dbus_connection  # noqa: E402

from mic_eq.ui import global_shortcuts_portal as portal  # noqa: E402


@pytest.mark.parametrize(("shortcut", "trigger"), [
    ("Ctrl+Alt+M", "CTRL+ALT+m"),
    ("win+shift+f9", "LOGO+SHIFT+F9"),
    ("Ctrl+PageUp", "CTRL+Page_Up"),
])
def test_shortcuts_translate_to_the_portal_trigger_format(shortcut, trigger):
    assert portal.portal_trigger(shortcut) == trigger


def test_an_unsupported_shortcut_is_refused():
    with pytest.raises(ValueError):
        portal.portal_trigger("M")
    with pytest.raises(ValueError):
        portal.portal_trigger("Ctrl+F25")


# A private bus with no service directories, so nothing on the host gets activated on it.
BUS_CONFIG = """<busconfig>
  <type>session</type>
  <listen>unix:dir={directory}</listen>
  <auth>EXTERNAL</auth>
  <policy context="default">
    <allow send_destination="*" eavesdrop="true"/>
    <allow eavesdrop="true"/>
    <allow own="*"/>
  </policy>
</busconfig>
"""


@pytest.fixture
def session_bus(monkeypatch, tmp_path):
    if shutil.which("dbus-daemon") is None:
        pytest.skip("needs dbus-daemon")
    config = tmp_path / "bus.conf"
    config.write_text(BUS_CONFIG.format(directory=tmp_path), encoding="utf-8")
    daemon = subprocess.Popen(
        ["dbus-daemon", f"--config-file={config}", "--nofork", "--print-address"],
        stdout=subprocess.PIPE, text=True,
    )
    assert daemon.stdout is not None
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", daemon.stdout.readline().strip())
    yield
    daemon.terminate()
    daemon.wait(timeout=5)


class FakePortal:
    """Answers the GlobalShortcuts calls the way xdg-desktop-portal does."""

    def __init__(self, *, bind_code=0, activations=1):
        self.bind_code = bind_code
        self.activations = activations
        self.bound = []
        self.session_closed = threading.Event()
        self.connection = open_dbus_connection("SESSION")
        self.connection.send_and_get_reply(message_bus.RequestName(portal.PORTAL_SERVICE))
        self.connection.send_and_get_reply(message_bus.AddMatch(
            "type='signal',sender='org.freedesktop.DBus',member='NameOwnerChanged'"
        ))
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _respond(self, sender, options, code, results):
        path = f"{portal.PORTAL_PATH}/request/{sender[1:].replace('.', '_')}/{options['handle_token'][1]}"
        self.connection.send(new_signal(
            DBusAddress(path, interface="org.freedesktop.portal.Request"), "Response", "ua{sv}", (code, results)
        ))
        return path

    def _serve(self):
        session = f"{portal.PORTAL_PATH}/session/client/1"
        while True:
            try:
                message = self.connection.receive()
            except (OSError, ConnectionError):
                return
            header = message.header
            if header.message_type == MessageType.signal:
                # The portal ends a session when its client leaves the bus.
                if header.fields.get(HeaderFields.member) == "NameOwnerChanged" and message.body[2] == "":
                    self.session_closed.set()
                continue
            if header.message_type != MessageType.method_call:
                continue
            member = header.fields[HeaderFields.member]
            sender = header.fields[HeaderFields.sender]
            if member == "Get":
                reply = new_method_return(message, "v", (("u", 1),))
            elif member == "Register":
                reply = new_method_return(message)
            elif member == "CreateSession":
                path = self._respond(sender, message.body[0], 0, {"session_handle": ("s", session)})
                reply = new_method_return(message, "o", (path,))
            elif member == "BindShortcuts":
                handle, shortcuts, _parent, options = message.body
                self.bound.extend(shortcuts)
                path = new_method_return(message, "o", ("/",))
                self.connection.send(path)
                self._respond(sender, options, self.bind_code, {"shortcuts": ("a(sa{sv})", shortcuts)})
                for _ in range(self.activations if self.bind_code == 0 else 0):
                    self.connection.send(new_signal(
                        DBusAddress(portal.PORTAL_PATH, interface=portal.GLOBAL_SHORTCUTS.interface),
                        "Activated", "osta{sv}", (handle, portal.SHORTCUT_ID, 0, {}),
                    ))
                continue
            else:
                reply = new_error(message, "org.freedesktop.DBus.Error.UnknownMethod")
            self.connection.send(reply)

    def close(self):
        self.connection.close()


def _session(on_activated, on_failed):
    return portal.GlobalShortcutsSession(
        "io.github.FueledByRedBull.AudioForge", "CTRL+ALT+m", "Toggle Output Mute", on_activated, on_failed
    )


def test_a_bound_shortcut_reports_activations_until_the_session_closes(session_bus):
    fake = FakePortal(activations=2)
    activated = threading.Semaphore(0)
    failures = []
    session = _session(activated.release, failures.append)
    try:
        assert portal.portal_available()
        session.start()
        assert activated.acquire(timeout=5) and activated.acquire(timeout=5)
        assert fake.bound[0][0] == portal.SHORTCUT_ID
        assert fake.bound[0][1]["preferred_trigger"] == ("s", "CTRL+ALT+m")
    finally:
        session.close()
    assert fake.session_closed.wait(timeout=5)
    assert failures == []
    fake.close()


def test_a_cancelled_bind_is_reported_as_a_failure(session_bus):
    fake = FakePortal(bind_code=1)
    failed = threading.Event()
    errors = []
    session = _session(lambda: None, lambda error: (errors.append(error), failed.set()))
    try:
        session.start()
        assert failed.wait(timeout=5)
    finally:
        session.close()
        fake.close()
    assert errors == ["BindShortcuts was cancelled by the desktop"]


def test_a_desktop_without_the_portal_is_reported_as_unavailable(session_bus):
    assert not portal.portal_available()


def _wait_for(qapp, condition, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return condition()


def test_the_mute_hotkey_toggles_on_the_qt_thread_through_the_portal(qapp, session_bus):
    from mic_eq.ui.desktop_integration import GlobalMuteHotkey

    fake = FakePortal(activations=1)
    threads = []
    hotkey = GlobalMuteHotkey("Ctrl+Alt+M", lambda: threads.append(threading.current_thread()))
    try:
        assert hotkey.register() == (True, None)
        assert hotkey.registered
        assert _wait_for(qapp, lambda: threads)
        assert threads == [threading.main_thread()]
    finally:
        hotkey.unregister()
        fake.close()
    assert not hotkey.registered


def test_a_refused_portal_bind_disables_the_hotkey(qapp, session_bus):
    from mic_eq.ui.desktop_integration import GlobalMuteHotkey

    fake = FakePortal(bind_code=2)
    errors = []
    hotkey = GlobalMuteHotkey("Ctrl+Alt+M", lambda: None, errors.append)
    try:
        assert hotkey.register() == (True, None)
        assert _wait_for(qapp, lambda: errors)
    finally:
        fake.close()
    assert errors == ["BindShortcuts was refused by the desktop"]
    assert not hotkey.registered


def test_without_the_portal_the_hotkey_asks_for_a_desktop_key_binding(qapp, session_bus):
    from mic_eq.ui.desktop_integration import GlobalMuteHotkey

    registered, error = GlobalMuteHotkey("Ctrl+Alt+M", lambda: None).register()
    assert not registered
    assert error is not None and "--toggle-mute" in error
