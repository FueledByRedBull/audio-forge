"""Write the PKGBUILD and .SRCINFO of the audioforge-bin AUR package for a release.

The package repackages the release tarball: the bundle goes to /opt/audioforge,
with an audioforge launcher, the desktop entry, metainfo and icons under /usr.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from release_version import parse_tag

PKGNAME = "audioforge-bin"
APP_ID = "io.github.FueledByRedBull.AudioForge"
REPOSITORY = "https://github.com/FueledByRedBull/audio-forge"
DESCRIPTION = "Remove background noise and polish your voice for calls, streams and recordings"
LICENSES = ("MIT", "LGPL-3.0-only")
# The libraries linux_package.SYSTEM_LIBRARIES leaves to the system, by Arch package.
DEPENDS = (
    "alsa-lib", "dbus", "fontconfig", "freetype2", "glib2", "glibc", "krb5", "libdrm", "libglvnd",
    "libpipewire", "libpulse", "libstdc++", "libx11", "libxcb", "libxkbcommon",
    "libxkbcommon-x11", "mesa", "pipewire", "systemd-libs", "wayland",
)
OPTDEPENDS = ("rtkit: realtime priority for the audio thread",)

PACKAGE_FUNCTION = """package() {
    cd "AudioForge-v$pkgver-linux-x86_64"
    install -d "$pkgdir/opt/audioforge" "$pkgdir/usr/bin" "$pkgdir/usr/share/licenses"
    cp -a AudioForge/. "$pkgdir/opt/audioforge/"
    cp -a share "$pkgdir/usr/"
    printf '#!/bin/sh\\nexec /opt/audioforge/AudioForge "$@"\\n' > "$pkgdir/usr/bin/audioforge"
    chmod 755 "$pkgdir/usr/bin/audioforge"
    ln -s /opt/audioforge/_internal/licenses "$pkgdir/usr/share/licenses/$pkgname"
    # A system install can be launched over D-Bus; the release entry stays Exec-only
    # because AppImage and tarball copies have no D-Bus service file.
    sed -i 's/^StartupWMClass=.*/&\\nDBusActivatable=true/' "$pkgdir/usr/share/applications/$_app_id.desktop"
    install -d "$pkgdir/usr/share/dbus-1/services"
    printf '[D-BUS Service]\\nName=%s\\nExec=/usr/bin/audioforge\\n' "$_app_id" \\
        > "$pkgdir/usr/share/dbus-1/services/$_app_id.service"
}
"""


def read_checksum(path: Path, tarball: str) -> str:
    """Return the tarball's digest from a `sha256sum` file such as the release's SHA256SUMS.txt."""
    digests = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](\S+)\s*", line)
        if match is None:
            raise ValueError(f"{path} has a line that is not a sha256sum line: {line!r}")
        if match[2] == tarball:
            digests.append(match[1])
    if len(digests) != 1:
        raise ValueError(f"{path} must list {tarball} exactly once")
    return digests[0]


def _array(values: tuple[str, ...]) -> str:
    return "(" + " ".join(f"'{value}'" for value in values) + ")"


def render(tag: str, sha256: str, pkgrel: int = 1) -> tuple[str, str]:
    """Return the PKGBUILD and .SRCINFO texts for a final release tag."""
    version = parse_tag(tag)
    if version.is_prerelease:
        raise ValueError("the AUR package follows final releases only")
    source = f"{REPOSITORY}/releases/download/v$pkgver/AudioForge-v$pkgver-linux-x86_64.tar.gz"
    pkgbuild = "\n".join((
        f"pkgname={PKGNAME}",
        f"_app_id={APP_ID}",
        f"pkgver={version.base}",
        f"pkgrel={pkgrel}",
        f'pkgdesc="{DESCRIPTION}"',
        "arch=('x86_64')",
        f'url="{REPOSITORY}"',
        f"license={_array(LICENSES)}",
        f"depends={_array(DEPENDS)}",
        f"optdepends={_array(OPTDEPENDS)}",
        "provides=('audioforge')",
        "conflicts=('audioforge')",
        # The bundle is PyInstaller output: stripping it breaks the archive appended to the executable.
        "options=('!strip' '!debug')",
        f'source=("{source}")',
        f"sha256sums=('{sha256}')",
        "",
        PACKAGE_FUNCTION,
    ))
    fields = [
        ("pkgdesc", DESCRIPTION),
        ("pkgver", version.base),
        ("pkgrel", str(pkgrel)),
        ("url", REPOSITORY),
        ("arch", "x86_64"),
        *(("license", value) for value in LICENSES),
        *(("depends", value) for value in DEPENDS),
        *(("optdepends", value) for value in OPTDEPENDS),
        ("provides", "audioforge"),
        ("conflicts", "audioforge"),
        ("options", "!strip"),
        ("options", "!debug"),
        ("source", source.replace("$pkgver", version.base)),
        ("sha256sums", sha256),
    ]
    srcinfo = "".join((
        f"pkgbase = {PKGNAME}\n",
        *(f"\t{key} = {value}\n" for key, value in fields),
        f"\npkgname = {PKGNAME}\n",
    ))
    return pkgbuild, srcinfo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="final release tag, such as v1.19.0")
    parser.add_argument("--checksums", type=Path, required=True, help="the release's SHA256SUMS.txt")
    parser.add_argument("--pkgrel", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="AUR repository checkout to write into")
    args = parser.parse_args()
    tarball = f"AudioForge-{args.tag}-linux-x86_64.tar.gz"
    try:
        pkgbuild, srcinfo = render(args.tag, read_checksum(args.checksums, tarball), args.pkgrel)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "PKGBUILD").write_text(pkgbuild, encoding="utf-8")
    (args.output / ".SRCINFO").write_text(srcinfo, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
