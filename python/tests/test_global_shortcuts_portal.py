# jeepney is a Linux-only dependency; Windows type checks run without it.
# pyright: reportMissingImports=false
"""The GlobalShortcuts portal client binds the mute shortcut and reports activations."""

from __future__ import annotations

import queue
import sys
import threading
import time

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


class FakePortal:
    """Answers the GlobalShortcuts calls the way xdg-desktop-portal does."""

    def __init__(self, *, bind_code=0, activations=1, silent=False, ignoring=(), name_flags=0):
        self.bind_code = bind_code
        self.activations = activations
        self.silent = silent  # owns the name but answers nothing, like a portal still starting up
        self.ignoring = set(ignoring)  # method names it stalls on
        self.session_path = f"{portal.PORTAL_PATH}/session/client/1"
        self.bound = []
        self.session_closed = threading.Event()
        self.connection = open_dbus_connection("SESSION")
        # The name last: a client that sees it calls at once, and a call arriving during a
        # blocking setup call here would be dropped.
        self.connection.send_and_get_reply(message_bus.AddMatch(
            "type='signal',sender='org.freedesktop.DBus',member='NameOwnerChanged'"
        ))
        self.connection.send_and_get_reply(message_bus.RequestName(portal.PORTAL_SERVICE, name_flags))
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _respond(self, sender, options, code, results):
        path = f"{portal.PORTAL_PATH}/request/{sender[1:].replace('.', '_')}/{options['handle_token'][1]}"
        self.connection.send(new_signal(
            DBusAddress(path, interface="org.freedesktop.portal.Request"), "Response", "ua{sv}", (code, results)
        ))
        return path

    def _serve(self):
        session = self.session_path
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
            if header.message_type != MessageType.method_call or self.silent:
                continue
            member = header.fields[HeaderFields.member]
            sender = header.fields[HeaderFields.sender]
            if member in self.ignoring:
                continue
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

    def close_session(self):
        self.connection.send(new_signal(
            DBusAddress(self.session_path, interface=portal.SESSION_INTERFACE), "Closed", "a{sv}", ({},)
        ))

    def close(self):
        self.connection.close()


def _session(on_activated, on_failed, on_availability=lambda _reason: None):
    return portal.GlobalShortcutsSession(
        "io.github.FueledByRedBull.AudioForge", "CTRL+ALT+m", "Toggle Output Mute",
        on_activated, on_failed, on_availability,
    )


@pytest.fixture
def no_session_bus(monkeypatch, tmp_path):
    """A login from a console, ssh or sudo: no bus address, and no bus socket at the runtime dir."""
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))


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


def test_a_session_without_a_bus_is_unavailable_not_failed(no_session_bus):
    assert not portal.portal_available()
    reasons = queue.Queue()
    failures = []
    session = _session(lambda: None, failures.append, reasons.put)
    try:
        session.start()
        assert reasons.get(timeout=5)
    finally:
        session.close()
    assert failures == []


def test_a_session_the_desktop_closes_is_reported_as_unavailable(session_bus):
    fake = FakePortal(activations=0)
    states = queue.Queue()
    session = _session(lambda: None, lambda error: None, states.put)
    try:
        session.start()
        assert states.get(timeout=5) == ""
        fake.close_session()
        assert states.get(timeout=5) == "the desktop closed the shortcut session"
    finally:
        session.close()
        fake.close()


def test_a_portal_that_leaves_and_returns_gets_the_shortcut_bound_again(session_bus):
    fake = FakePortal(activations=0)
    states = queue.Queue()
    activated = threading.Semaphore(0)
    session = _session(activated.release, lambda error: None, states.put)
    try:
        session.start()
        assert states.get(timeout=5) == ""
        fake.close()
        assert states.get(timeout=5) == "the desktop portal went away"
        fake = FakePortal(activations=1)
        assert states.get(timeout=10) == ""
        assert activated.acquire(timeout=5)
    finally:
        session.close()
        fake.close()


def test_a_portal_replaced_in_one_step_gets_the_shortcut_bound_again(session_bus):
    # xdg-desktop-portal --replace: the name passes straight to the new portal, and the
    # old portal's session goes with it.
    allow_replacement, replace_existing = 1, 2
    old = FakePortal(activations=0, name_flags=allow_replacement)
    states = queue.Queue()
    activated = threading.Semaphore(0)
    session = _session(activated.release, lambda error: None, states.put)
    new = None
    try:
        session.start()
        assert states.get(timeout=5) == ""
        new = FakePortal(activations=1, name_flags=replace_existing)
        assert states.get(timeout=10) == ""
        assert activated.acquire(timeout=5)
        assert new.bound
    finally:
        session.close()
        old.close()
        if new is not None:
            new.close()


def test_a_portal_that_starts_after_the_session_gets_the_shortcut_bound(session_bus):
    states = queue.Queue()
    session = _session(lambda: None, lambda error: None, states.put)
    fake = None
    try:
        session.start()
        assert states.get(timeout=5) == "no GlobalShortcuts portal answered"
        fake = FakePortal(activations=0)
        assert states.get(timeout=10) == ""
    finally:
        session.close()
        if fake is not None:
            fake.close()


def test_a_portal_that_owns_its_name_but_answers_late_gets_the_shortcut_bound(session_bus, monkeypatch):
    # Nothing announces when a portal still starting up begins to answer.
    monkeypatch.setattr(portal, "PORTAL_RETRY_SECONDS", 0.1)
    states = queue.Queue()
    session = _session(lambda: None, lambda error: None, states.put)
    fake = FakePortal(activations=0, silent=True)
    try:
        session.start()
        assert states.get(timeout=10) == "TimeoutError"
        fake.silent = False
        assert _next_state_other_than("TimeoutError", states, timeout=15) == ""
    finally:
        session.close()
        fake.close()


def test_a_portal_that_stalls_after_registering_is_asked_again(session_bus, monkeypatch):
    monkeypatch.setattr(portal, "PORTAL_RETRY_SECONDS", 0.1)
    states = queue.Queue()
    session = _session(lambda: None, lambda error: None, states.put)
    fake = FakePortal(activations=0, ignoring={"Get"})
    try:
        session.start()
        assert states.get(timeout=10) == "TimeoutError"
        fake.ignoring.clear()
        assert _next_state_other_than("TimeoutError", states, timeout=15) == ""
    finally:
        session.close()
        fake.close()


def _next_state_other_than(ignored, states, timeout):
    import time

    deadline = time.monotonic() + timeout
    while True:
        state = states.get(timeout=max(0.0, deadline - time.monotonic()))
        if state != ignored:
            return state


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


@pytest.mark.parametrize("bus", ["session_bus", "no_session_bus"], ids=["no-portal", "no-session-bus"])
def test_without_the_portal_the_hotkey_asks_for_a_desktop_key_binding(qapp, request, bus):
    request.getfixturevalue(bus)
    from mic_eq.ui.desktop_integration import GlobalMuteHotkey

    reasons, failures = [], []
    hotkey = GlobalMuteHotkey("Ctrl+Alt+M", lambda: None, failures.append, reasons.append)
    try:
        assert hotkey.register() == (True, None)
        assert _wait_for(qapp, lambda: reasons)
    finally:
        hotkey.unregister()
    assert "--toggle-mute" in reasons[0] and failures == []


def test_registering_does_not_wait_for_a_portal_that_never_answers(qapp, session_bus):
    from mic_eq.ui.desktop_integration import GlobalMuteHotkey

    fake = FakePortal(silent=True)
    hotkey = GlobalMuteHotkey("Ctrl+Alt+M", lambda: None)
    started = time.monotonic()
    try:
        assert hotkey.register() == (True, None)
        assert time.monotonic() - started < 1.0
    finally:
        hotkey.unregister()
        fake.close()
