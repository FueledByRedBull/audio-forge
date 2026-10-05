"""Login registration tests never access the host's Startup folder or COM."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from mic_eq.ui import login_startup


class _ShellLink:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def QueryInterface(self, _iid):
        return self

    def Load(self, path):
        self.values = json.loads(Path(path).read_text(encoding="utf-8"))

    def Save(self, path, _remember):
        Path(path).write_text(json.dumps(self.values), encoding="utf-8")

    def SetPath(self, value):
        self.values["target"] = value

    def GetPath(self, _flags):
        return self.values["target"], None

    def SetArguments(self, value):
        self.values["arguments"] = value

    def GetArguments(self):
        return self.values["arguments"]

    def SetDescription(self, value):
        self.values["description"] = value

    def GetDescription(self):
        return self.values["description"]

    def SetWorkingDirectory(self, value):
        self.values["working_directory"] = value


@pytest.fixture
def registration(tmp_path, monkeypatch):
    folder = tmp_path / "Startup folder"
    folder.mkdir()
    path = folder / "AudioForge Login.lnk"
    monkeypatch.setattr(login_startup, "startup_shortcut_path", lambda: path)
    monkeypatch.setattr(login_startup, "_shell_link", _ShellLink)
    executable = tmp_path / "Portable app" / "AudioForge.exe"
    executable.parent.mkdir()
    executable.touch()
    return executable, path


def test_registration_is_default_off_and_reading_never_creates_it(registration):
    executable, path = registration
    assert login_startup.registration_state(executable) == "absent"
    assert not path.exists()


def test_explicit_registration_is_idempotent_and_separates_path_from_arguments(registration):
    executable, path = registration
    assert login_startup.set_login_startup(executable, True)
    contents = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    values = json.loads(contents)
    assert values["target"] == str(executable)
    assert values["arguments"] == "--login-startup"
    assert values["working_directory"] == str(executable.parent)
    assert login_startup.registration_state(executable) == "configured"
    assert not login_startup.set_login_startup(executable, True)
    assert path.read_bytes() == contents
    assert path.stat().st_mtime_ns == stamp  # Do not undo Windows disablement by rewriting.
    assert login_startup.set_login_startup(executable, False)
    assert not path.exists()


@pytest.mark.parametrize("changed_field", ["target", "arguments", "description"])
def test_opt_out_and_uninstall_preserve_reassigned_or_unrecognized_links(
    registration, changed_field
):
    executable, path = registration
    login_startup.set_login_startup(executable, True)
    values = json.loads(path.read_text(encoding="utf-8"))
    values[changed_field] += " different"
    path.write_text(json.dumps(values), encoding="utf-8")
    contents = path.read_bytes()
    assert login_startup.registration_state(executable) == "other"
    assert not login_startup.set_login_startup(executable, False)
    with pytest.raises(RuntimeError, match="another|unrecognized"):
        login_startup.set_login_startup(executable, True)
    assert path.read_bytes() == contents


def test_unreadable_shortcut_is_never_repaired_or_deleted(registration):
    executable, path = registration
    path.write_bytes(b"invalid shortcut")
    assert login_startup.registration_state(executable) == "unreadable"
    for enabled in (False, True):
        with pytest.raises(RuntimeError, match="read"):
            login_startup.set_login_startup(executable, enabled)
    assert path.read_bytes() == b"invalid shortcut"


def test_directory_at_shortcut_path_is_preserved(registration):
    executable, path = registration
    path.mkdir()
    assert login_startup.registration_state(executable) == "unreadable"
    with pytest.raises(RuntimeError, match="read"):
        login_startup.set_login_startup(executable, False)
    assert path.is_dir()


def test_uninstall_launcher_calls_ownership_helper_before_any_qt_import(monkeypatch, registration):
    import builtins
    import sys

    executable, path = registration
    login_startup.set_login_startup(executable, True)
    launcher_path = Path(__file__).parents[2] / "launcher.py"
    spec = importlib.util.spec_from_file_location("startup_test_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("PySide6", "mic_eq.ui.main_window", "mic_eq.ui.app_bootstrap")):
            raise AssertionError("Uninstall must not construct or import the Qt application")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "argv", [str(executable), "--remove-login-startup"])
    assert launcher._run() == 0
    assert not path.exists()
