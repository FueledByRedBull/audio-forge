"""A session without a D-Bus session bus reads as "no bus", not as a crash."""

from __future__ import annotations

import os
import sys

import pytest

if sys.platform != "linux":
    pytest.skip("D-Bus is used on Linux only", allow_module_level=True)

from mic_eq.ui.session_bus import open_session_bus  # noqa: E402


def _bus_socket() -> str:
    address = os.environ["DBUS_SESSION_BUS_ADDRESS"].partition(":")[2]
    return dict(item.split("=", 1) for item in address.split(","))["path"]


@pytest.fixture
def no_session_address(monkeypatch, tmp_path):
    """A login from a console, ssh or sudo: no address, and no bus socket at the runtime dir."""
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def test_an_unset_address_without_a_bus_at_the_runtime_directory_is_an_os_error(no_session_address):
    with pytest.raises(OSError):
        open_session_bus()


@pytest.mark.parametrize("address", ["", "tcp:host=localhost,port=1"])
def test_an_empty_or_unsupported_address_is_an_os_error(monkeypatch, address):
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", address)
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    with pytest.raises(OSError):
        open_session_bus()


@pytest.mark.parametrize("address", [None, ""])
def test_the_login_managers_bus_at_the_runtime_directory_is_used_without_an_address(
    session_bus, monkeypatch, tmp_path, address
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "bus").symlink_to(_bus_socket())
    if address is None:
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS")
    else:
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", address)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    with open_session_bus() as connection:
        assert connection.unique_name


def test_a_runtime_directory_with_address_syntax_in_its_name_still_finds_the_bus(session_bus, monkeypatch, tmp_path):
    runtime = tmp_path / "run,time;=100%"
    runtime.mkdir()
    (runtime / "bus").symlink_to(_bus_socket())
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    with open_session_bus() as connection:
        assert connection.unique_name
