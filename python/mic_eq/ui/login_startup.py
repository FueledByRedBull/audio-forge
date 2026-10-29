"""Explicit, per-user login registration; the shortcut is the sole authority.

Windows may disable a configured shortcut. Never infer its effective state or
rewrite it during launch/upgrade: that could undo the user's Windows preference.
On Linux the registration is an XDG autostart entry, held to the same rules.
This module has no Qt or audio construction, including during MSI removal.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Literal


LOGIN_STARTUP_ARGUMENT = "--login-startup"
_DESCRIPTION = "AudioForge login startup"
RegistrationState = Literal["absent", "configured", "other", "unreadable"]
APP_ID = "io.github.FueledByRedBull.AudioForge"


def launch_executable() -> Path:
    """The file a login entry should start: the AppImage itself, not its mount."""
    appimage = os.environ.get("APPIMAGE", "")
    if sys.platform != "win32" and os.path.isabs(appimage):
        return Path(appimage)
    return Path(sys.executable)


def startup_shortcut_path() -> Path:
    if sys.platform != "win32":
        return _autostart_dir() / f"{APP_ID}.desktop"
    from win32com.shell import shell, shellcon  # pyright: ignore[reportMissingModuleSource]

    return Path(shell.SHGetKnownFolderPath(shellcon.FOLDERID_Startup)) / "AudioForge Login.lnk"


def _shell_link() -> Any:
    import pythoncom
    from win32com.shell import shell  # pyright: ignore[reportMissingModuleSource]

    return pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink
    )


def registration_state(executable: Path) -> RegistrationState:
    """Inspect our stable filename without resolving targets or changing Windows state."""
    if sys.platform != "win32":
        return _autostart_state(executable)
    import pythoncom
    from win32com.shell import shell  # pyright: ignore[reportMissingModuleSource]

    try:
        path = startup_shortcut_path()
        try:
            info = path.lstat()
        except FileNotFoundError:
            return "absent"
        if not stat.S_ISREG(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            return "unreadable"
        link = _shell_link()
        link.QueryInterface(pythoncom.IID_IPersistFile).Load(str(path))
        target, _ = link.GetPath(shell.SLGP_RAWPATH)
        matches = (
            Path(target).is_absolute()
            and os.path.normcase(os.path.normpath(target))
            == os.path.normcase(str(executable.absolute()))
            and link.GetArguments() == LOGIN_STARTUP_ARGUMENT
            and link.GetDescription() == _DESCRIPTION
        )
        return "configured" if matches else "other"
    except Exception:
        return "unreadable"


def set_login_startup(executable: Path, enabled: bool) -> bool:
    """Change only an explicitly requested registration for this executable.

    Returns whether the shortcut changed. An unrecognized or reassigned link is
    never removed; an unreadable link is never repaired. No Windows approval
    registry entries are inspected or modified.
    """
    executable = executable.absolute()
    state = registration_state(executable)
    if state == "unreadable":
        raise RuntimeError("Could not read the login shortcut; it was left unchanged.")
    if not enabled:
        if state != "configured":
            return False
        startup_shortcut_path().unlink()
        return True
    if state == "configured":
        return False
    if state == "other":
        raise RuntimeError("The login shortcut belongs to another copy or is unrecognized.")
    if not executable.is_file():
        raise RuntimeError("The AudioForge executable is unavailable.")

    if sys.platform != "win32":
        _write_autostart_entry(executable)
        return True

    import pythoncom

    path = startup_shortcut_path()
    link = _shell_link()
    link.SetPath(str(executable))
    link.SetArguments(LOGIN_STARTUP_ARGUMENT)
    link.SetDescription(_DESCRIPTION)
    link.SetWorkingDirectory(str(executable.parent))
    fd, temporary_name = tempfile.mkstemp(suffix=".lnk", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(temporary), True)
        # Windows rename fails if another process created the destination first.
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _autostart_dir() -> Path:
    # The XDG Base Directory spec ignores a relative XDG_CONFIG_HOME.
    config_home = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(config_home) if os.path.isabs(config_home) else Path.home() / ".config"
    return base / "autostart"


def _desktop_exec_argument(value: str) -> str:
    """Quote one Exec argument as the Desktop Entry spec requires.

    The quoting escapes come first; the string escape then doubles every
    backslash, and ``%`` is doubled so it is not read as a field code.
    """
    quoted = "".join("\\" + char if char in '"`$\\' else char for char in value)
    return '"' + quoted.replace("\\", "\\\\").replace("%", "%%") + '"'


def _autostart_exec(executable: Path) -> str:
    return f"{_desktop_exec_argument(str(executable))} {LOGIN_STARTUP_ARGUMENT}"


def _autostart_entry(executable: Path) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=AudioForge\n"
        f"Comment={_DESCRIPTION}\n"
        f"Exec={_autostart_exec(executable)}\n"
        f"Icon={APP_ID}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def _autostart_state(executable: Path) -> RegistrationState:
    path = startup_shortcut_path()
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(info.st_mode):
        return "unreadable"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "unreadable"
    keys: dict[str, str] = {}
    in_entry = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_entry = stripped == "[Desktop Entry]"
        elif in_entry and "=" in stripped:
            key, _, value = stripped.partition("=")
            keys.setdefault(key.strip(), value.strip())
    matches = (
        keys.get("Exec") == _autostart_exec(executable.absolute())
        and keys.get("Comment") == _DESCRIPTION
    )
    return "configured" if matches else "other"


def _write_autostart_entry(executable: Path) -> None:
    path = startup_shortcut_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(suffix=".desktop", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(_autostart_entry(executable))
        # A hard link refuses a name another process created first; rename would replace it.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
