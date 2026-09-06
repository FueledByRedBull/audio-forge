"""Tests for MSI payload validation helpers."""

from __future__ import annotations

import importlib.util
import shutil
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


def test_msi_product_version_uses_release_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _bundle(tmp_path / "payload")
    (payload / "_internal" / "audioforge-build.json").write_text(
        '{"version": "1.12.0"}\n', encoding="utf-8"
    )
    msi = tmp_path / "AudioForge.msi"
    msi.write_bytes(b"msi")
    monkeypatch.setattr(msi_smoke, "_read_msi_product_version", lambda _msi: "1.12.0")

    with pytest.raises(RuntimeError, match="mapped to '1.12.99'"):
        msi_smoke._assert_msi_product_version(msi, payload)

    monkeypatch.setattr(msi_smoke, "_read_msi_product_version", lambda _msi: "1.12.99")
    msi_smoke._assert_msi_product_version(msi, payload)


@pytest.mark.skipif(sys.platform != "win32", reason="MSI flow is Windows-only")
def test_validate_msi_rejects_wrong_product_version_before_msiexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _bundle(tmp_path / "payload")
    (payload / "_internal" / "audioforge-build.json").write_text(
        '{"version": "1.12.0"}\n', encoding="utf-8"
    )
    msi = tmp_path / "AudioForge.msi"
    msi.write_bytes(b"msi")
    monkeypatch.setattr(msi_smoke, "check_dist_bundle", lambda _payload: [])
    monkeypatch.setattr(msi_smoke, "_read_msi_product_version", lambda _msi: "1.12.0")
    monkeypatch.setattr(
        msi_smoke,
        "_msiexec",
        lambda: pytest.fail("ProductVersion mismatch must precede msiexec"),
    )

    with pytest.raises(RuntimeError, match="mapped to '1.12.99'"):
        msi_smoke.validate_msi(msi, payload)


@pytest.mark.skipif(sys.platform != "win32", reason="MSI flow is Windows-only")
def test_validate_msi_accepts_distinct_upgrade_payload_and_checks_current_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_payload = _bundle(tmp_path / "old", marker=b"old-payload")
    current_payload = _bundle(tmp_path / "current", marker=b"new-payload")
    old_msi = tmp_path / "AudioForge-old.msi"
    current_msi = tmp_path / "AudioForge-current.msi"
    old_msi.write_bytes(b"old-msi")
    current_msi.write_bytes(b"current-msi")
    local_appdata = tmp_path / "local-appdata"
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(msi_smoke, "_msiexec", lambda: "fake-msiexec")
    comparisons: list[tuple[Path, Path, bytes]] = []
    original_compare = msi_smoke._compare_payload

    def record_compare(expected: Path, actual: Path) -> None:
        marker = (actual / "_internal" / "asset.bin").read_bytes()
        comparisons.append((expected.resolve(), actual.resolve(), marker))
        original_compare(expected, actual)

    monkeypatch.setattr(msi_smoke, "_compare_payload", record_compare)

    def fake_msiexec(
        _executable: str,
        args: list[str],
        *,
        allowed_exit_codes: tuple[int, ...] = (0, 3010),
    ) -> int:
        del allowed_exit_codes
        action = args[0]
        if action == "/a":
            target = Path(next(value.split("=", 1)[1] for value in args if value.startswith("TARGETDIR=")))
            source = old_payload if str(old_msi) in args else current_payload
            shutil.copytree(source, target, dirs_exist_ok=True)
            return 0
        if action == "/i":
            install_root = local_appdata / "AudioForge"
            if "/l*v" in args:
                log_path = Path(args[args.index("/l*v") + 1])
                log_path.write_text(
                    "Product: AudioForge -- A newer version of AudioForge is already installed.\n",
                    encoding="utf-8",
                )
                raise msi_smoke.MsiCommandError(args, 1638, "downgrade")
            source = old_payload if str(old_msi) in args else current_payload
            shutil.rmtree(install_root, ignore_errors=True)
            shutil.copytree(source, install_root)
            shortcut = msi_smoke._shortcut_path()
            shortcut.parent.mkdir(parents=True, exist_ok=True)
            shortcut.write_bytes(b"shortcut")
            return 0
        if action == "/x":
            shutil.rmtree(local_appdata / "AudioForge", ignore_errors=True)
            msi_smoke._shortcut_path().unlink(missing_ok=True)
            return 0
        raise AssertionError(f"unexpected fake msiexec action: {args}")

    monkeypatch.setattr(msi_smoke, "_run_msiexec", fake_msiexec)
    monkeypatch.setattr(msi_smoke, "check_dist_bundle", lambda _payload: [])
    monkeypatch.setattr(msi_smoke, "_bundle_version", lambda _payload: "1.12.0")
    monkeypatch.setattr(msi_smoke, "_read_msi_product_version", lambda _msi: "1.12.99")

    msi_smoke.validate_msi(
        current_msi,
        current_payload,
        upgrade_from=old_msi,
    )

    assert [marker for _, _, marker in comparisons] == [
        b"new-payload",
        b"old-payload",
        b"new-payload",
        b"new-payload",
    ]
    install_root = (local_appdata / "AudioForge").resolve()
    assert any(
        actual == install_root and marker == b"old-payload"
        for _, actual, marker in comparisons
    )
    assert any(
        expected == current_payload.resolve()
        and actual == install_root
        and marker == b"new-payload"
        for expected, actual, marker in comparisons
    )


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
