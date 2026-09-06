"""Validate an AudioForge MSI against the portable bundle it installs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from package_smoke import _bundle_version, check_dist_bundle
from release_provenance import build_bundle_manifest
from release_version import parse_version


class MsiCommandError(RuntimeError):
    """A Windows Installer invocation returned an unexpected exit code."""

    def __init__(self, args: list[str], returncode: int, detail: str) -> None:
        self.args_list = args
        self.returncode = returncode
        self.detail = detail
        super().__init__(
            f"msiexec {' '.join(args[:2])} failed with exit code "
            f"{returncode}: {detail}"
        )


def _msiexec() -> str:
    path = shutil.which("msiexec.exe") or shutil.which("msiexec")
    if path is None:
        raise RuntimeError("Windows Installer (msiexec.exe) is not available")
    return path


def _run_msiexec(
    msiexec: str,
    args: list[str],
    *,
    allowed_exit_codes: tuple[int, ...] = (0, 3010),
) -> int:
    result = subprocess.run(
        [msiexec, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode not in allowed_exit_codes:
        detail = (result.stderr or result.stdout).strip()
        raise MsiCommandError(args, result.returncode, detail)
    return result.returncode


def _find_bundle(root: Path) -> Path:
    matches = list(root.rglob("AudioForge.exe"))
    if len(matches) != 1:
        raise RuntimeError(
            f"MSI extraction must contain exactly one AudioForge.exe, found {len(matches)}"
        )
    return matches[0].parent


def _compare_payload(expected: Path, actual: Path) -> None:
    expected_manifest = build_bundle_manifest(expected)
    actual_manifest = build_bundle_manifest(actual)
    for field in ("schema_version", "file_count", "total_bytes", "files"):
        if expected_manifest.get(field) != actual_manifest.get(field):
            raise RuntimeError("MSI payload differs from the portable bundle")


def _read_msi_product_version(msi: Path) -> str:
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("MSI ProductVersion validation requires pywin32") from exc

    installer = win32com.client.Dispatch("WindowsInstaller.Installer")
    database = installer.OpenDatabase(str(msi), 0)
    view = database.OpenView(
        "SELECT Value FROM Property WHERE Property='ProductVersion'"
    )
    view.Execute()
    record = view.Fetch()
    if record is None:
        raise RuntimeError(f"MSI ProductVersion property is missing: {msi}")
    # StringData is an indexed COM property, not a method.
    string_data_id = record._oleobj_.GetIDsOfNames(0, "StringData")
    value = record._oleobj_.Invoke(
        string_data_id,
        0,
        pythoncom.DISPATCH_PROPERTYGET,
        True,
        1,
    )
    if not value:
        raise RuntimeError(f"MSI ProductVersion property is empty: {msi}")
    return str(value)


def _assert_msi_product_version(msi: Path, payload: Path) -> None:
    bundle_version = _bundle_version(payload)
    if bundle_version is None:
        raise RuntimeError(f"Portable bundle version is missing: {payload}")
    try:
        expected = parse_version(bundle_version).msi
    except ValueError as exc:
        raise RuntimeError(
            f"Portable bundle version is invalid: {bundle_version!r}"
        ) from exc
    actual = _read_msi_product_version(msi)
    if actual != expected:
        raise RuntimeError(
            f"MSI ProductVersion {actual!r} does not match portable bundle version "
            f"{bundle_version!r} mapped to {expected!r}"
        )


def _shortcut_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is required for shortcut validation")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "AudioForge" / "AudioForge.lnk"


def _assert_shortcut(shortcut: Path) -> None:
    if not shortcut.is_file():
        raise RuntimeError(f"MSI did not create the per-user Start Menu shortcut: {shortcut}")


def _assert_expected_downgrade(error: MsiCommandError, log_path: Path) -> None:
    """Accept only the installer failure that explicitly reports a downgrade."""
    if error.returncode not in {1603, 1638}:
        raise error
    try:
        raw_log = log_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"MSI downgrade failed without a readable installer log: {log_path}"
        ) from exc
    decoded_logs = (
        raw_log.decode("utf-16", errors="replace"),
        raw_log.decode("utf-8", errors="replace"),
    )
    has_major_upgrade_message = any(
        re.search(
            r"Product:\s*AudioForge\s*--\s*A newer version of AudioForge is already installed\.",
            log,
            re.IGNORECASE | re.DOTALL,
        )
        is not None
        for log in decoded_logs
    )
    if not has_major_upgrade_message:
        raise RuntimeError(
            "MSI downgrade failed without the expected MajorUpgrade message: "
            f"{error}"
        )


def validate_msi(
    msi: Path,
    payload: Path,
    *,
    install_uninstall: bool = False,
    upgrade_from: Path | None = None,
) -> None:
    if os.name != "nt":
        raise RuntimeError("MSI smoke validation requires Windows")
    msi = msi.resolve(strict=True)
    payload = payload.resolve(strict=True)
    if not payload.is_dir():
        raise RuntimeError(f"Portable payload directory is missing: {payload}")
    errors = check_dist_bundle(payload)
    if errors:
        raise RuntimeError("Portable bundle is invalid:\n  " + "\n  ".join(errors))
    _assert_msi_product_version(msi, payload)

    msiexec = _msiexec()
    with tempfile.TemporaryDirectory(prefix="audioforge-msi-smoke-") as temp_name:
        temp_root = Path(temp_name)
        admin_root = temp_root / "admin"
        _run_msiexec(
            msiexec,
            ["/a", str(msi), f"TARGETDIR={admin_root}", "/qn", "/norestart"],
        )
        _compare_payload(payload, _find_bundle(admin_root))

        if upgrade_from is None and not install_uninstall:
            return

        if upgrade_from is not None:
            upgrade_from = upgrade_from.resolve(strict=True)
            old_admin_root = temp_root / "upgrade-from-admin"
            _run_msiexec(
                msiexec,
                ["/a", str(upgrade_from), f"TARGETDIR={old_admin_root}", "/qn", "/norestart"],
            )
            old_payload = _find_bundle(old_admin_root)

        local_appdata = os.environ.get("LOCALAPPDATA")
        appdata = os.environ.get("APPDATA")
        if not local_appdata or not appdata:
            raise RuntimeError("LOCALAPPDATA and APPDATA are required for install smoke")
        install_root = Path(local_appdata) / "AudioForge"
        config_root = Path(appdata) / "AudioForge"
        shortcut = _shortcut_path()
        if install_root.exists():
            raise RuntimeError(
                f"Refusing to run MSI install smoke over an existing directory: {install_root}"
            )
        if shortcut.exists():
            raise RuntimeError(f"Refusing to overwrite existing shortcut: {shortcut}")
        sentinel = config_root / ".msi-smoke-config-sentinel"
        if sentinel.exists():
            raise RuntimeError(f"Refusing to overwrite existing sentinel: {sentinel}")
        config_root.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("preserve\n", encoding="utf-8")
        attempted_current = False
        attempted_upgrade = False
        try:
            if upgrade_from is not None:
                attempted_upgrade = True
                _run_msiexec(msiexec, ["/i", str(upgrade_from), "/qn", "/norestart"])
                if not install_root.is_dir():
                    raise RuntimeError(f"Upgrade baseline MSI did not install to {install_root}")
                _compare_payload(old_payload, install_root)
                _assert_shortcut(shortcut)

            attempted_current = True
            _run_msiexec(msiexec, ["/i", str(msi), "/qn", "/norestart"])
            if not install_root.is_dir():
                raise RuntimeError(f"MSI did not install to {install_root}")
            _compare_payload(payload, install_root)
            _assert_shortcut(shortcut)
            if sentinel.read_text(encoding="utf-8") != "preserve\n":
                raise RuntimeError("MSI upgrade changed the per-user configuration sentinel")

            if upgrade_from is not None:
                try:
                    _run_msiexec(
                        msiexec,
                        [
                            "/i",
                            str(upgrade_from),
                            "/qn",
                            "/norestart",
                            "/l*v",
                            str(temp_root / "downgrade.log"),
                        ],
                        allowed_exit_codes=(),
                    )
                except MsiCommandError as error:
                    _assert_expected_downgrade(error, temp_root / "downgrade.log")
                else:
                    raise RuntimeError("MSI downgrade unexpectedly succeeded")
                if not install_root.is_dir():
                    raise RuntimeError("MSI downgrade removed the upgraded installation")
                _compare_payload(payload, install_root)
                _assert_shortcut(shortcut)
                if sentinel.read_text(encoding="utf-8") != "preserve\n":
                    raise RuntimeError("MSI downgrade changed the per-user configuration sentinel")

            _run_msiexec(msiexec, ["/x", str(msi), "/qn", "/norestart"])
            attempted_current = False
            if install_root.exists():
                raise RuntimeError(f"MSI uninstall left files under {install_root}")
            if shortcut.exists():
                raise RuntimeError(f"MSI uninstall left the Start Menu shortcut: {shortcut}")
            if sentinel.read_text(encoding="utf-8") != "preserve\n":
                raise RuntimeError("MSI uninstall changed the per-user configuration sentinel")
        finally:
            if attempted_current:
                try:
                    _run_msiexec(msiexec, ["/x", str(msi), "/qn", "/norestart"])
                except Exception:
                    pass
            if attempted_upgrade and upgrade_from is not None:
                try:
                    _run_msiexec(msiexec, ["/x", str(upgrade_from), "/qn", "/norestart"])
                except Exception:
                    pass
            sentinel.unlink(missing_ok=True)
            try:
                config_root.rmdir()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msi", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--install-uninstall",
        action="store_true",
        help="also install and uninstall the MSI, preserving a config sentinel",
    )
    parser.add_argument(
        "--upgrade-from",
        type=Path,
        help="older MSI used to exercise per-user upgrade, downgrade blocking, and uninstall",
    )
    args = parser.parse_args()
    validate_msi(
        args.msi,
        args.payload,
        install_uninstall=args.install_uninstall,
        upgrade_from=args.upgrade_from,
    )
    print("MSI smoke check passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MSI smoke check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
