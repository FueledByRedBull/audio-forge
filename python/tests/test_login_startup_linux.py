"""XDG autostart registration tests never touch the developer's autostart folder."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from mic_eq.ui import login_startup

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="XDG autostart backend")


@pytest.fixture
def registration(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    executable = tmp_path / "Portable app" / "AudioForge"
    executable.parent.mkdir()
    executable.touch()
    path = tmp_path / "config" / "autostart" / f"{login_startup.APP_ID}.desktop"
    assert login_startup.startup_shortcut_path() == path
    return executable, path


def test_registration_is_default_off_and_reading_never_creates_it(registration):
    executable, path = registration
    assert login_startup.registration_state(executable) == "absent"
    assert not path.parent.exists()


def test_explicit_registration_is_idempotent_and_quotes_the_executable(registration):
    executable, path = registration
    assert login_startup.set_login_startup(executable, True)
    contents = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    assert f'Exec="{executable}" --login-startup\n'.encode() in contents
    assert login_startup.registration_state(executable) == "configured"
    assert not login_startup.set_login_startup(executable, True)
    assert path.read_bytes() == contents
    # Rewriting would undo a desktop's own disablement of the entry.
    assert path.stat().st_mtime_ns == stamp
    assert login_startup.set_login_startup(executable, False)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


def test_exec_quoting_follows_the_desktop_entry_spec():
    exec_line = login_startup._autostart_exec(Path('/opt/A "b"/$c%d\\e'))
    assert exec_line == '"/opt/A \\\\"b\\\\"/\\\\$c%%d\\\\\\\\e" --login-startup'


@pytest.mark.parametrize(
    ("old", "new"),
    [("--login-startup", "--other"), ("Comment=AudioForge login startup", "Comment=Mine")],
)
def test_opt_out_and_uninstall_preserve_reassigned_or_unrecognized_entries(
    registration, old, new
):
    executable, path = registration
    login_startup.set_login_startup(executable, True)
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    contents = path.read_bytes()
    assert login_startup.registration_state(executable) == "other"
    assert not login_startup.set_login_startup(executable, False)
    with pytest.raises(RuntimeError, match="another|unrecognized"):
        login_startup.set_login_startup(executable, True)
    assert path.read_bytes() == contents


def test_entry_for_another_copy_is_not_this_copys(registration, tmp_path):
    executable, _path = registration
    other = tmp_path / "Other copy" / "AudioForge"
    login_startup.set_login_startup(executable, True)
    assert login_startup.registration_state(other) == "other"


def test_unreadable_entry_is_never_repaired_or_deleted(registration):
    executable, path = registration
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe not utf-8")
    assert login_startup.registration_state(executable) == "unreadable"
    for enabled in (False, True):
        with pytest.raises(RuntimeError, match="read"):
            login_startup.set_login_startup(executable, enabled)
    assert path.read_bytes() == b"\xff\xfe not utf-8"


def test_directory_at_entry_path_is_preserved(registration):
    executable, path = registration
    path.mkdir(parents=True)
    assert login_startup.registration_state(executable) == "unreadable"
    with pytest.raises(RuntimeError, match="read"):
        login_startup.set_login_startup(executable, False)
    assert path.is_dir()


def test_appimage_registers_the_image_not_its_mount(monkeypatch, tmp_path):
    image = tmp_path / "AudioForge-x86_64.AppImage"
    monkeypatch.setenv("APPIMAGE", str(image))
    assert login_startup.launch_executable() == image
    monkeypatch.setenv("APPIMAGE", "relative.AppImage")
    assert login_startup.launch_executable() == Path(sys.executable)
