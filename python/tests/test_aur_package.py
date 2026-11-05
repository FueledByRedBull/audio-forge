"""The AUR package files follow final releases and the tarball's own checksum."""

from __future__ import annotations

import pytest

import aur_package

DIGEST = "a" * 64
TARBALL = "AudioForge-v1.2.3-linux-x86_64.tar.gz"


def test_a_release_renders_matching_pkgbuild_and_srcinfo():
    pkgbuild, srcinfo = aur_package.render("v1.2.3", DIGEST)

    assert "pkgver=1.2.3\n" in pkgbuild
    assert f"sha256sums=('{DIGEST}')" in pkgbuild
    assert "\tpkgver = 1.2.3\n" in srcinfo
    assert f"\tsource = {aur_package.REPOSITORY}/releases/download/v1.2.3/{TARBALL}\n" in srcinfo
    assert srcinfo.count("\tdepends = ") == len(aur_package.DEPENDS)
    # The system install, unlike the tarball and AppImage, can be launched over D-Bus.
    assert "DBusActivatable=true" in pkgbuild
    assert '"$pkgdir/usr/share/dbus-1/services/$_app_id.service"' in pkgbuild


def test_a_release_candidate_is_refused():
    with pytest.raises(ValueError, match="final releases"):
        aur_package.render("v1.2.3-rc.1", DIGEST)


def test_the_tarball_digest_is_read_from_the_release_checksums(tmp_path):
    checksums = tmp_path / "SHA256SUMS.txt"
    # Promotion writes this file on Windows.
    checksums.write_bytes(f"{'b' * 64}  AudioForge-v1.2.3-x86_64.AppImage\r\n{DIGEST}  {TARBALL}\r\n".encode())
    assert aur_package.read_checksum(checksums, TARBALL) == DIGEST

    with pytest.raises(ValueError, match="exactly once"):
        aur_package.read_checksum(checksums, "AudioForge-v1.2.4-linux-x86_64.tar.gz")


def test_the_files_are_written_with_unix_line_endings(tmp_path, monkeypatch):
    # makepkg refuses CRLF, which Windows' default text mode would write.
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{DIGEST}  {TARBALL}\n", encoding="utf-8")
    output = tmp_path / "aur"
    monkeypatch.setattr(
        "sys.argv", ["aur_package.py", "--tag", "v1.2.3", "--checksums", str(checksums), "--output", str(output)]
    )
    assert aur_package.main() == 0
    for name in ("PKGBUILD", ".SRCINFO"):
        assert b"\r" not in (output / name).read_bytes()
