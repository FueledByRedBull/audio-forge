# jeepney is a Linux-only dependency; Windows type checks run without it.
# pyright: reportMissingImports=false
"""The running app answers org.freedesktop.Application on the session bus."""

from __future__ import annotations

import sys
import time

import pytest

if sys.platform != "linux":
    pytest.skip("org.freedesktop.Application is served on Linux only", allow_module_level=True)

from jeepney import DBusAddress, HeaderFields, MessageType, new_method_call  # noqa: E402
from jeepney.io.blocking import open_dbus_connection  # noqa: E402

from mic_eq.ui.application_bus import ApplicationBusService, object_path  # noqa: E402

APP_ID = "io.github.FueledByRedBull.AudioForge"


def _wait_for(qapp, condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    return condition()


def _call(connection, method, signature=None, body=()):
    address = DBusAddress(object_path(APP_ID), APP_ID, "org.freedesktop.Application")
    return connection.send_and_get_reply(new_method_call(address, method, signature, body), timeout=5)


def test_the_object_path_follows_the_desktop_entry_spec():
    assert object_path(APP_ID) == "/io/github/FueledByRedBull/AudioForge"
    assert object_path("org.example.my-app") == "/org/example/my_app"


def test_activate_and_the_toggle_mute_action_reach_the_qt_thread(qapp, session_bus):
    calls = []
    service = ApplicationBusService(
        APP_ID, lambda: calls.append("activate"), {"toggle-mute": lambda: calls.append("toggle-mute")}
    )
    client = open_dbus_connection("SESSION")
    try:
        assert service.start()
        assert _call(client, "Activate", "a{sv}", ({},)).header.message_type == MessageType.method_return
        assert _call(client, "ActivateAction", "sava{sv}", ("toggle-mute", [], {})).header.message_type == (
            MessageType.method_return
        )
        unknown = _call(client, "ActivateAction", "sava{sv}", ("quit", [], {}))
        assert unknown.header.message_type == MessageType.error
        assert _wait_for(qapp, lambda: len(calls) == 2)
        assert calls == ["activate", "toggle-mute"]
    finally:
        service.close()
        client.close()


def test_a_second_copy_does_not_take_the_name(qapp, session_bus):
    first = ApplicationBusService(APP_ID, lambda: None, {})
    second = ApplicationBusService(APP_ID, lambda: None, {})
    try:
        assert first.start()
        assert not second.start()
    finally:
        first.close()
        second.close()


def test_without_a_session_bus_the_service_stays_off(qapp):
    assert not ApplicationBusService(APP_ID, lambda: None, {}).start()


def test_with_no_session_bus_address_at_all_the_service_stays_off(qapp, monkeypatch, tmp_path):
    # A login from a console, ssh or sudo: jeepney alone fails with a KeyError here.
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert not ApplicationBusService(APP_ID, lambda: None, {}).start()


def test_a_malformed_action_call_is_refused_and_the_service_keeps_answering(qapp, session_bus):
    calls = []
    service = ApplicationBusService(APP_ID, lambda: calls.append("activate"), {"toggle-mute": lambda: None})
    client = open_dbus_connection("SESSION")
    try:
        assert service.start()
        # No arguments, a dictionary, and a list where the action name belongs.
        for signature, body in (("", ()), ("a{sv}", ({},)), ("as", (["toggle-mute"],))):
            reply = _call(client, "ActivateAction", signature or None, body)
            assert reply.header.message_type == MessageType.error
            assert reply.header.fields[HeaderFields.error_name] == "org.freedesktop.DBus.Error.InvalidArgs"
        assert _call(client, "Activate", "a{sv}", ({},)).header.message_type == MessageType.method_return
        assert _wait_for(qapp, lambda: calls == ["activate"])
    finally:
        service.close()
        client.close()


def test_a_call_that_breaks_the_handler_is_answered_with_an_error_and_logged(qapp, session_bus, caplog):
    service = ApplicationBusService(APP_ID, lambda: None, {})
    answer = service._answer
    calls = []

    def answer_once_broken(message):
        calls.append(message)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return answer(message)

    service._answer = answer_once_broken
    client = open_dbus_connection("SESSION")
    try:
        assert service.start()
        failed = _call(client, "Activate", "a{sv}", ({},))
        assert failed.header.fields[HeaderFields.error_name] == "org.freedesktop.DBus.Error.Failed"
        assert _call(client, "Activate", "a{sv}", ({},)).header.message_type == MessageType.method_return
        assert "boom" in caplog.text
    finally:
        service.close()
        client.close()


def test_a_service_closed_for_a_relaunch_gives_the_name_up_and_can_take_it_back(qapp, session_bus):
    service = ApplicationBusService(APP_ID, lambda: None, {})
    other = ApplicationBusService(APP_ID, lambda: None, {})
    try:
        assert service.start()
        service.close()
        assert _takes_the_name(other)  # the relaunched copy
        other.close()
        assert _takes_the_name(service)  # the relaunch failed
    finally:
        service.close()
        other.close()


SERVICE_SCRIPT = """
import sys
sys.path.insert(0, {python!r})
from PySide6.QtCore import QCoreApplication, QTimer
from mic_eq.ui.application_bus import ApplicationBusService

app = QCoreApplication([])
def activated():
    open({marker!r}, "w").write("activated")
    app.quit()
service = ApplicationBusService({app_id!r}, activated, {{}})
assert service.start()
QTimer.singleShot(20000, app.quit)
app.exec()
service.close()
"""


def test_a_d_bus_launch_answers_the_call_that_started_it(session_bus, tmp_path):
    from pathlib import Path

    marker = tmp_path / "activated"
    script = tmp_path / "service.py"
    python = str(Path(__file__).resolve().parents[1])
    script.write_text(SERVICE_SCRIPT.format(python=python, marker=str(marker), app_id=APP_ID))
    (session_bus / f"{APP_ID}.service").write_text(
        f"[D-BUS Service]\nName={APP_ID}\nExec={sys.executable} {script}\n"
    )

    with open_dbus_connection("SESSION") as client:
        reply = _call(client, "Activate", "a{sv}", ({},))

    assert reply.header.message_type == MessageType.method_return
    assert _wait_until(lambda: marker.exists())


def _takes_the_name(service, timeout=5.0):
    # The bus frees a name once it has read the closed connection that held it.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if service.start():
            return True
        time.sleep(0.05)
    return False


def _wait_until(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return condition()
