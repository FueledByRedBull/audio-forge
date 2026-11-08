"""Package the Linux bundle from build_linux.sh as a tarball and an AppImage.

dist/AudioForge becomes, named after the release tag like the Windows files:
  dist/AudioForge-<tag>-linux-x86_64.tar.gz   extract anywhere; install.sh adds it to the menu
  dist/AudioForge-<tag>-x86_64.AppImage        when --appimagetool is given
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tomllib

from release_version import parse_version

ROOT = Path(__file__).resolve().parents[2]
APP_ID = "io.github.FueledByRedBull.AudioForge"
DESKTOP_DIR = ROOT / "installer" / "linux"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
ORT_LIBRARY = "libonnxruntime.so.1.23.2"
MODELS = ("DeepFilterNet3_ll_onnx.tar.gz", "DeepFilterNet3_onnx.tar.gz", "silero_vad.onnx")

# AudioForge.spec leaves these out of the bundle. Every supported desktop has
# them, and they must match it: the audio, D-Bus, Kerberos (Qt Network's
# GSSAPI), graphics, display and font stacks, glib, and the C++ runtime. Releases are built on the oldest supported
# distribution, so the system copy is never older than the one the build linked.
SYSTEM_LIBRARIES = (
    "libpipewire-0.3.so", "libasound.so", "libpulse", "libdbus-1.so", "libsystemd.so",
    "libgssapi_krb5.so", "libkrb5.so", "libk5crypto.so", "libkrb5support.so", "libcom_err.so",
    "libkeyutils.so",
    "libstdc++.so", "libgcc_s.so",
    "libGL", "libEGL.so", "libOpenGL.so", "libdrm.so", "libgbm.so", "libwayland-",
    "libX11.so", "libX11-xcb.so", "libxcb.so", "libxkbcommon",
    "libfontconfig.so", "libfreetype.so",
    "libglib-2.0.so", "libgobject-2.0.so", "libgio-2.0.so", "libgmodule-2.0.so",
    "libgthread-2.0.so",
)
# What Qt's xcb platform plugin, libQt6XcbQpa and the GLX integration link and
# the system policy above does not cover (read from the pinned PySide6 6.11.1
# wheel). Most desktops lack them, and a bundle without them cannot start under
# X11, so the build host provides them and the bundle must carry them.
XCB_LIBRARIES = (
    "libxcb-cursor.so.0", "libxcb-glx.so.0", "libxcb-icccm.so.4", "libxcb-image.so.0",
    "libxcb-keysyms.so.1", "libxcb-randr.so.0", "libxcb-render-util.so.0", "libxcb-render.so.0",
    "libxcb-shape.so.0", "libxcb-shm.so.0", "libxcb-sync.so.1", "libxcb-util.so.1",
    "libxcb-xfixes.so.0", "libxcb-xkb.so.1",
)
# The app has no use for FFmpeg, and a build of it can carry GPL code.
FFMPEG_LIBRARIES = (
    "libavcodec", "libavdevice", "libavfilter", "libavformat", "libavutil", "libpostproc",
    "libswresample", "libswscale", "libffmpegmediaplugin",
)
# The Debian packages the bundle may copy a library from, each with the license
# of that library. Only permissive ones belong here: a package that is not
# listed fails the check until its copyright file has been read.
HOST_LIBRARY_PACKAGES = {
    "libbrotli1": "MIT", "libbz2-1.0": "bzip2-1.0.6", "libexpat1": "MIT", "libffi8": "MIT",
    "liblzma5": "public domain", "libmpdec3": "BSD-2-Clause", "libncursesw6": "X11",
    "libsqlite3-0": "public domain", "libtinfo6": "X11", "libuuid1": "BSD-3-Clause",
    "libxcb-cursor0": "MIT", "libxcb-glx0": "MIT", "libxcb-icccm4": "MIT",
    "libxcb-image0": "MIT", "libxcb-keysyms1": "MIT", "libxcb-randr0": "MIT",
    "libxcb-render-util0": "MIT", "libxcb-render0": "MIT", "libxcb-shape0": "MIT",
    "libxcb-shm0": "MIT", "libxcb-sync1": "MIT", "libxcb-util1": "MIT", "libxcb-xfixes0": "MIT",
    "libxcb-xkb1": "MIT", "libzstd1": "BSD-3-Clause", "zlib1g": "Zlib",
}
# Qt plugins the app never loads; the GTK theme alone drags in GTK's stack.
# QAudioSink and QMediaDevices work without a multimedia backend, and the
# FFmpeg one would ship FFmpeg's LGPL libraries for nothing.
UNUSED_QT_PLUGINS = (
    "multimedia/",
    "platformthemes/libqgtk3.so",
    "platforminputcontexts/libqtvirtualkeyboardplugin.so",
    "egldeviceintegrations/", "generic/", "networkinformation/", "qmltooling/", "tls/",
    *(f"imageformats/libq{name}.so" for name in ("gif", "icns", "jpeg", "pdf", "tga", "tiff", "wbmp", "webp")),
)

INSTALL_SH = """#!/bin/sh
# Add this copy of AudioForge to your application menu; --uninstall removes it.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
app_id=@APP_ID@
data=${XDG_DATA_HOME:-}
case $data in /*) ;; *) data=$HOME/.local/share ;; esac

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$data/applications/$app_id.desktop" "$data/metainfo/$app_id.metainfo.xml"
    for icon in "$data"/icons/hicolor/*/apps/"$app_id".png; do
        rm -f "$icon"
    done
    echo "AudioForge was removed from the application menu. Delete this folder to remove the app."
    exit 0
fi

# The Exec line quotes the path; the Desktop Entry spec reserves these characters there.
case $here in
    *'"'* | *'`'* | *'$'* | *'\\'* | *'%'*) # nosemgrep: generic-obfuscation-operators
        echo "Move AudioForge to a folder whose path has no \\" \\` \\$ \\\\ or % characters." >&2
        exit 1
        ;;
esac
mkdir -p "$data/applications" "$data/metainfo"
entry=$data/applications/$app_id.desktop
# Point the entry's commands, its actions' included, at this copy.
while IFS= read -r line; do
    case $line in
        Exec=audioforge*) printf 'Exec="%s/AudioForge/AudioForge"%s\\n' "$here" "${line#Exec=audioforge}" ;;
        *) printf '%s\\n' "$line" ;;
    esac
done < "$here/share/applications/$app_id.desktop" > "$entry"
cp "$here/share/metainfo/$app_id.metainfo.xml" "$data/metainfo/"
for icon in "$here"/share/icons/hicolor/*/apps/"$app_id".png; do
    size=$(basename "$(dirname "$(dirname "$icon")")")
    mkdir -p "$data/icons/hicolor/$size/apps"
    cp "$icon" "$data/icons/hicolor/$size/apps/"
done
command -v update-desktop-database >/dev/null && update-desktop-database -q "$data/applications" || true
echo "AudioForge is in your application menu. Run ./install.sh --uninstall before moving this folder."
""".replace("@APP_ID@", APP_ID)

APPRUN = """#!/bin/sh
here=$(dirname "$(readlink -f "$0")")
exec "$here/usr/lib/audioforge/AudioForge" "$@"
"""


def _owning_package(library: str) -> tuple[str, list[Path]]:
    """The build host's package that installs ``library``, and its copyright files."""
    if shutil.which("dpkg-query"):
        listing = subprocess.run(
            ["dpkg-query", "-S", f"*/{Path(library).name}"], capture_output=True, text=True, check=True
        ).stdout
        packages = {line.split(":", 1)[0] for line in listing.splitlines() if not line.startswith("diversion")}
        if len(packages) != 1:
            raise SystemExit(f"{library} belongs to {sorted(packages) or 'no package'}, not one package")
        package = packages.pop()
        return package, [Path("/usr/share/doc", package, "copyright")]
    package = subprocess.run(["pacman", "-Qqo", library], capture_output=True, text=True, check=True).stdout.strip()
    return package, sorted(path for path in Path("/usr/share/licenses", package).glob("*") if path.is_file())


def host_library_notices(libraries: list[str]) -> list[tuple[str, str, str]]:
    """PyInstaller DATA entries for the copyright files of the packages that own ``libraries``."""
    entries = set()
    for library in libraries:
        package, notices = _owning_package(library)
        notices = [notice for notice in notices if notice.is_file()]
        if not notices:
            raise SystemExit(f"{package}, which installs {library}, has no copyright file")
        entries.update((f"licenses/system/{package}/{notice.name}", str(notice), "DATA") for notice in notices)
    return sorted(entries)


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


PT_LOAD, PT_DYNAMIC = 1, 2
DT_NULL, DT_STRTAB, DT_STRSZ, DT_RPATH, DT_RUNPATH = 0, 5, 10, 15, 29


def elf_search_paths(path: Path) -> list[tuple[str, str]]:
    """The DT_RPATH and DT_RUNPATH strings of the ELF file ``path`` as (tag, value); none for other files."""
    with path.open("rb") as handle:
        ident = handle.read(16)
        if len(ident) < 16 or ident[:4] != b"\x7fELF":
            return []
        order = "<" if ident[5] == 1 else ">"
        if ident[4] == 2:
            header, program, dynamic, picks = "HHIQQQIHHHHHH", "IIQQQQQQ", "qQ", (0, 2, 3, 5)
        else:
            header, program, dynamic, picks = "HHIIIIIHHHHHH", "IIIIIIII", "iI", (0, 1, 2, 4)
        fields = struct.unpack(order + header, handle.read(struct.calcsize(order + header)))
        phoff, phentsize, phnum = fields[4], fields[8], fields[9]
        segments = []  # (type, file offset, virtual address, file size)
        for index in range(phnum):
            handle.seek(phoff + index * phentsize)
            entry = struct.unpack(order + program, handle.read(struct.calcsize(order + program)))
            segments.append(tuple(entry[pick] for pick in picks))
        dynamic_segment = next((segment for segment in segments if segment[0] == PT_DYNAMIC), None)
        if dynamic_segment is None:
            return []
        handle.seek(dynamic_segment[1])
        raw = handle.read(dynamic_segment[3])
        size = struct.calcsize(order + dynamic)
        entries = []
        for tag, value in struct.iter_unpack(order + dynamic, raw[: len(raw) - len(raw) % size]):
            if tag == DT_NULL:
                break
            entries.append((tag, value))
        wanted = [(tag, value) for tag, value in entries if tag in (DT_RPATH, DT_RUNPATH)]
        if not wanted:
            return []
        values = dict(entries)
        strtab = values.get(DT_STRTAB)
        load = next((segment for segment in segments if segment[0] == PT_LOAD
                     and strtab is not None and segment[2] <= strtab < segment[2] + segment[3]), None)
        if load is None or DT_STRSZ not in values:
            raise ValueError(f"{path} has a search path but no readable string table")
        handle.seek(load[1] + strtab - load[2])
        table = handle.read(values[DT_STRSZ])
    names = {DT_RPATH: "RPATH", DT_RUNPATH: "RUNPATH"}
    return [(names[tag], os.fsdecode(table[value:].split(b"\0", 1)[0])) for tag, value in wanted]


def _outside_bundle(entry: str, origin: Path, bundle: Path) -> bool:
    """Whether the library search path ``entry`` of a file in ``origin`` leaves ``bundle``."""
    for token in ("$ORIGIN", "${ORIGIN}"):
        if entry == token or entry.startswith(token + "/"):
            target = os.path.normpath(os.path.join(origin, entry[len(token):].lstrip("/")))
            return not Path(target).is_relative_to(os.path.normpath(bundle))
    return True


def check_bundle(bundle: Path, version: str) -> list[str]:
    """Problems that make ``bundle`` unfit to ship; empty when it is ready."""
    internal = bundle / "_internal"
    errors = [
        f"missing {relative}"
        for relative in (
            "AudioForge",
            f"_internal/{ORT_LIBRARY}",
            "_internal/libdf.so",
            "_internal/AudioForge.ico",
            *(f"_internal/models/{model}" for model in MODELS),
            *(f"_internal/{library}" for library in XCB_LIBRARIES),
        )
        if not (bundle / relative).is_file()
    ]
    if not list((internal / "mic_eq").glob("mic_eq_core.cpython-*-linux-gnu.so")):
        errors.append("missing the mic_eq_core extension")
    try:
        built = json.loads((internal / "audioforge-build.json").read_text(encoding="utf-8"))["version"]
    except (OSError, KeyError, ValueError):
        built = None
    if built != version:
        errors.append(f"audioforge-build.json names version {built!r}, not {version}")
    for path in sorted(internal.rglob("*.so*")):
        relative = path.relative_to(bundle).as_posix()
        if path.name.startswith(SYSTEM_LIBRARIES):
            errors.append(f"bundles a library that must come from the system: {relative}")
        if path.name.startswith("libreadline.so"):
            errors.append(f"bundles GPL readline, which the app never uses: {relative}")
        if path.name.startswith(FFMPEG_LIBRARIES):
            errors.append(f"bundles FFmpeg, which the app never uses: {relative}")
        if any(f"PySide6/Qt/plugins/{plugin}" in relative for plugin in UNUSED_QT_PLUGINS):
            errors.append(f"bundles an unused Qt plugin: {relative}")
    # A search path outside the bundle makes the loader look on the user's
    # machine (a build host's home or toolcache) before the system libraries.
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        try:
            search_paths = elf_search_paths(path)
        except (OSError, OverflowError, ValueError, struct.error) as exc:
            errors.append(f"cannot read the library search paths of {relative}: {exc}")
            continue
        for tag, value in search_paths:
            outside = [entry for entry in value.split(":") if _outside_bundle(entry, path.parent, bundle)]
            if outside:
                shown = ":".join(entry or "''" for entry in outside)
                errors.append(f"{relative} has {tag} entries outside the bundle: {shown}")
    # Releases are built on Ubuntu; the table names Debian packages, so a pacman
    # host (a local Arch build) cannot be checked against it.
    notices = internal / "licenses" / "system"
    packages = sorted(path.name for path in notices.iterdir()) if notices.is_dir() else []
    for package in packages if shutil.which("dpkg-query") else []:
        if package not in HOST_LIBRARY_PACKAGES:
            errors.append(
                f"copies a library from the host package {package}, which is not known to be "
                f"permissive: read _internal/licenses/system/{package}/copyright and add it "
                "to HOST_LIBRARY_PACKAGES only if it is"
            )
    return errors


def export_icons(share: Path) -> None:
    """Write the hicolor PNGs from AudioForge.ico, which holds every size we install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImageReader

    _app = QGuiApplication.instance() or QGuiApplication([])
    reader = QImageReader(str(ROOT / "AudioForge.ico"))
    images = {}
    for index in range(reader.imageCount()):
        reader.jumpToImage(index)
        image = reader.read()
        images[image.width()] = image
    missing = [size for size in ICON_SIZES if size not in images]
    if missing:
        raise SystemExit(f"AudioForge.ico has no {missing} px images")
    for size in ICON_SIZES:
        target = share / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{APP_ID}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not images[size].save(str(target), "PNG"):
            raise SystemExit(f"could not write {target}")


def share_tree(share: Path) -> None:
    (share / "applications").mkdir(parents=True)
    (share / "metainfo").mkdir(parents=True)
    shutil.copy2(DESKTOP_DIR / f"{APP_ID}.desktop", share / "applications")
    shutil.copy2(DESKTOP_DIR / f"{APP_ID}.metainfo.xml", share / "metainfo")
    export_icons(share)


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def build_tarball(bundle: Path, work: Path, output: Path, version: str) -> Path:
    name = f"AudioForge-{parse_version(version).tag}-linux-x86_64"
    staging = work / name
    shutil.copytree(bundle, staging / "AudioForge", symlinks=True)
    share_tree(staging / "share")
    write_executable(staging / "install.sh", INSTALL_SH)
    archive = output / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname=name, filter=_normalized)
    return archive


def build_appimage(
    bundle: Path, work: Path, output: Path, version: str, appimagetool: Path, runtime: Path | None
) -> Path:
    appdir = work / "AudioForge.AppDir"
    usr = appdir / "usr"
    shutil.copytree(bundle, usr / "lib" / "audioforge", symlinks=True)
    (usr / "bin").mkdir(parents=True)
    (usr / "bin" / "audioforge").symlink_to("../lib/audioforge/AudioForge")
    share_tree(usr / "share")
    shutil.copy2(usr / "share" / "applications" / f"{APP_ID}.desktop", appdir)
    shutil.copy2(usr / "share" / "icons" / "hicolor" / "256x256" / "apps" / f"{APP_ID}.png", appdir)
    (appdir / ".DirIcon").symlink_to(f"{APP_ID}.png")
    write_executable(appdir / "AppRun", APPRUN)
    image = output / f"AudioForge-{parse_version(version).tag}-x86_64.AppImage"
    # Without --runtime-file, appimagetool downloads whatever runtime is newest.
    runtime_args = ["--runtime-file", str(runtime)] if runtime else []
    subprocess.run(
        [str(appimagetool), "--no-appstream", *runtime_args, str(appdir), str(image)],
        check=True,
        env={**os.environ, "ARCH": "x86_64"},
    )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / "AudioForge")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--appimagetool", type=Path, help="appimagetool executable; omit to skip the AppImage")
    parser.add_argument("--appimage-runtime", type=Path, help="pinned AppImage runtime for appimagetool to embed")
    parser.add_argument("--check-only", action="store_true", help="check the bundle without packaging it")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if sys.platform != "linux":
        parser.error("Linux packages are built on Linux")
    version = project_version()
    errors = check_bundle(bundle, version)
    if errors:
        print(f"{bundle} is not ready to ship:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    if args.check_only:
        print(f"{bundle} is ready to ship")
        return 0
    work = args.output / "linux-package"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    print(build_tarball(bundle, work, args.output, version))
    if args.appimagetool:
        print(build_appimage(bundle, work, args.output, version, args.appimagetool, args.appimage_runtime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
