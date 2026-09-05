"""Tests for MSI payload validation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).parent.parent / "tools"
for name in ("prune_bundle", "package_smoke", "release_provenance"):
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location("msi_smoke", TOOLS_DIR / "msi_smoke.py")
assert spec is not None and spec.loader is not None
msi_smoke = importlib.util.module_from_spec(spec)
sys.modules["msi_smoke"] = msi_smoke
spec.loader.exec_module(msi_smoke)


def _bundle(root: Path, marker: bytes = b"payload") -> Path:
    bundle = root / "AudioForge"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "AudioForge.exe").write_bytes(b"exe")
    (bundle / "_internal" / "asset.bin").write_bytes(marker)
    return bundle


def test_compare_payload_accepts_identical_trees(tmp_path: Path) -> None:
    expected = _bundle(tmp_path / "expected")
    actual = _bundle(tmp_path / "actual")
    msi_smoke._compare_payload(expected, actual)


def test_compare_payload_rejects_changed_file(tmp_path: Path) -> None:
    expected = _bundle(tmp_path / "expected")
    actual = _bundle(tmp_path / "actual", marker=b"changed")
    with pytest.raises(RuntimeError, match="differs"):
        msi_smoke._compare_payload(expected, actual)


def test_msi_command_error_preserves_exit_code() -> None:
    error = msi_smoke.MsiCommandError(["/i", "AudioForge.msi"], 1638, "downgrade")

    assert error.returncode == 1638
    assert "1638" in str(error)


def test_downgrade_gate_requires_major_upgrade_message(tmp_path: Path) -> None:
    log = tmp_path / "downgrade.log"
    log.write_text(
        "MSI (s) (10:20) [12:34:56]: Product: AudioForge -- "
        "A newer version of AudioForge is already installed.\n",
        encoding="utf-8",
    )
    error = msi_smoke.MsiCommandError(["/i", "AudioForge.msi"], 1603, "")

    msi_smoke._assert_expected_downgrade(error, log)

    log.write_text(
        "MSI (s) (10:20) [12:34:56]: /l*v C:\\temp\\downgrade.log "
        "DowngradeErrorMessage property was present, but installation failed.\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="MajorUpgrade message"):
        msi_smoke._assert_expected_downgrade(error, log)


def test_msi_defines_per_user_shortcut_and_upgrade_policy() -> None:
    source = (Path(__file__).parents[2] / "installer" / "AudioForge.wxs").read_text(
        encoding="utf-8"
    )

    assert 'Scope="perUser"' in source
    assert 'UpgradeCode="' in source
    assert 'MajorUpgrade ' in source
    assert 'Id="WINDOWSBUILDNUMBER"' in source
    assert 'Name="CurrentBuildNumber"' in source
    assert 'Condition="Installed OR (VersionNT64 AND WINDOWSBUILDNUMBER &gt;= 17763)"' in source
    assert "Windows 10 version 1809" in source
    assert 'StandardDirectory Id="ProgramMenuFolder"' in source
    assert '<Shortcut' in source
    assert '<RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />' in source


def test_msi_build_script_targets_x64() -> None:
    source = (Path(__file__).parents[2] / "build_msi.ps1").read_text(
        encoding="utf-8"
    )

    assert "& $WixPath build `" in source
    assert "        -arch x64 `" in source
