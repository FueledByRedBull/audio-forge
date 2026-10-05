"""Explicit, per-user login registration; the shortcut is the sole authority.

Windows may disable a configured shortcut. Never infer its effective state or
rewrite it during launch/upgrade: that could undo the user's Windows preference.
This module has no Qt or audio construction, including during MSI removal.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Literal


LOGIN_STARTUP_ARGUMENT = "--login-startup"
_DESCRIPTION = "AudioForge login startup"
RegistrationState = Literal["absent", "configured", "other", "unreadable"]


def startup_shortcut_path() -> Path:
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
    import pythoncom

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
