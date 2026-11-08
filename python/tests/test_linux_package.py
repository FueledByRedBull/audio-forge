"""The Linux bundle check refuses bundles that would misbehave on a user's desktop."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import linux_package


def _bundle(tmp_path, version="1.2.3"):
    bundle = tmp_path / "AudioForge"
    internal = bundle / "_internal"
    for relative in (
        "AudioForge",
        f"_internal/{linux_package.ORT_LIBRARY}",
        "_internal/libdf.so",
        "_internal/AudioForge.ico",
        "_internal/mic_eq/mic_eq_core.cpython-313-x86_64-linux-gnu.so",
        *(f"_internal/{library}" for library in linux_package.XCB_LIBRARIES),
        *(f"_internal/models/{model}" for model in linux_package.MODELS),
    ):
        (bundle / relative).parent.mkdir(parents=True, exist_ok=True)
        (bundle / relative).write_bytes(b"")
    (internal / "audioforge-build.json").write_text(json.dumps({"version": version}))
    return bundle


def test_a_complete_bundle_is_ready(tmp_path):
    assert linux_package.check_bundle(_bundle(tmp_path), "1.2.3") == []


def test_system_libraries_readline_unused_plugins_and_a_stale_version_are_refused(tmp_path):
    bundle = _bundle(tmp_path, version="1.2.2")
    (bundle / "_internal/libpipewire-0.3.so.0").write_bytes(b"")
    (bundle / "_internal/libreadline.so.8").write_bytes(b"")
    theme = bundle / "_internal/PySide6/Qt/plugins/platformthemes/libqgtk3.so"
    theme.parent.mkdir(parents=True)
    theme.write_bytes(b"")
    (bundle / "_internal/libdf.so").unlink()

    errors = linux_package.check_bundle(bundle, "1.2.3")

    assert "missing _internal/libdf.so" in errors
    assert any("version '1.2.2'" in error for error in errors)
    assert "bundles a library that must come from the system: _internal/libpipewire-0.3.so.0" in errors
    assert any(error.startswith("bundles GPL readline") for error in errors)
    assert any(error.startswith("bundles an unused Qt plugin") for error in errors)
    assert len(errors) == 5


def test_ffmpeg_and_a_bundle_without_the_xcb_libraries_are_refused(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "_internal/PySide6/Qt/lib").mkdir(parents=True)
    (bundle / "_internal/PySide6/Qt/lib/libavcodec.so.61").write_bytes(b"")
    (bundle / "_internal/libswresample.so.5").write_bytes(b"")
    (bundle / "_internal/libxcb-cursor.so.0").unlink()

    errors = linux_package.check_bundle(bundle, "1.2.3")

    assert "missing _internal/libxcb-cursor.so.0" in errors
    assert "bundles FFmpeg, which the app never uses: _internal/PySide6/Qt/lib/libavcodec.so.61" in errors
    assert "bundles FFmpeg, which the app never uses: _internal/libswresample.so.5" in errors
    assert len(errors) == 3


def _elf(path, *search_paths, wide=True, order="<"):
    """Write a minimal shared object whose dynamic section holds ``search_paths`` (tag, value)."""
    strtab = b"\0"
    entries = []
    for tag, value in search_paths:
        entries.append(({"RPATH": 15, "RUNPATH": 29}[tag], len(strtab)))
        strtab += value.encode() + b"\0"
    header, program, dynamic = ("HHIQQQIHHHHHH", "IIQQQQQQ", "qQ") if wide else ("HHIIIIIHHHHHH", "IIIIIIII", "iI")
    base = 0x10000
    program_offset = 16 + struct.calcsize(order + header)
    dynamic_offset = program_offset + 2 * struct.calcsize(order + program)
    entries += [(5, base + dynamic_offset + (len(entries) + 3) * struct.calcsize(order + dynamic)), (10, len(strtab)), (0, 0)]
    table = b"".join(struct.pack(order + dynamic, tag, value) for tag, value in entries)
    total = dynamic_offset + len(table) + len(strtab)

    def segment(kind, offset, size):
        if wide:
            return struct.pack(order + program, kind, 4, offset, base + offset, base + offset, size, size, 8)
        return struct.pack(order + program, kind, offset, base + offset, base + offset, size, size, 4, 8)

    ident = b"\x7fELF" + bytes((2 if wide else 1, 1 if order == "<" else 2, 1)) + bytes(9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        ident
        + struct.pack(order + header, 3, 62, 1, 0, program_offset, 0, 0, program_offset,
                      struct.calcsize(order + program), 2, 0, 0, 0)
        + segment(1, 0, total) + segment(2, dynamic_offset, len(table)) + table + strtab
    )


@pytest.mark.parametrize(("wide", "order"), [(True, "<"), (False, ">")])
def test_the_library_search_paths_are_read_from_the_dynamic_section(tmp_path, wide, order):
    library = tmp_path / "libexample.so"
    _elf(library, ("RPATH", "/usr/lib"), ("RUNPATH", "$ORIGIN/../lib"), wide=wide, order=order)
    assert linux_package.elf_search_paths(library) == [("RPATH", "/usr/lib"), ("RUNPATH", "$ORIGIN/../lib")]
    _elf(library)
    assert linux_package.elf_search_paths(library) == []


def test_library_search_paths_outside_the_bundle_are_refused(tmp_path):
    bundle = _bundle(tmp_path)
    module = bundle / "_internal/mic_eq/mic_eq_core.cpython-313-x86_64-linux-gnu.so"
    # What `maturin develop` left in the 1.22.0 module on a runner with patchelf.
    leaked = "/home/runner/work/audio-forge/audio-forge/target/release/build/libspa-sys-78b153ad82a8aae9/out"
    _elf(module, ("RPATH", f"/usr/lib/x86_64-linux-gnu:$ORIGIN:{leaked}"))
    _elf(bundle / "_internal/libpython3.13.so.1.0", ("RUNPATH", "/opt/hostedtoolcache/Python/3.13.17/x64/lib"))
    _elf(bundle / "_internal/PySide6/Qt/lib/libQt6Core.so.6", ("RUNPATH", "$ORIGIN"))
    _elf(bundle / "_internal/PySide6/Qt/plugins/platforms/libqxcb.so", ("RUNPATH", "${ORIGIN}/../../lib"))
    _elf(bundle / "_internal/numpy/core/_umath.so", ("RPATH", "$ORIGIN/../../../../lib"))
    (bundle / "_internal/libtruncated.so").write_bytes(b"\x7fELF\x02\x01\x01" + bytes(11))

    errors = linux_package.check_bundle(bundle, "1.2.3")

    assert errors == [
        "_internal/libpython3.13.so.1.0 has RUNPATH entries outside the bundle: "
        "/opt/hostedtoolcache/Python/3.13.17/x64/lib",
        errors[1],
        f"_internal/mic_eq/mic_eq_core.cpython-313-x86_64-linux-gnu.so has RPATH entries outside the bundle: "
        f"/usr/lib/x86_64-linux-gnu:{leaked}",
        "_internal/numpy/core/_umath.so has RPATH entries outside the bundle: $ORIGIN/../../../../lib",
    ]
    assert errors[1].startswith("cannot read the library search paths of _internal/libtruncated.so: ")


def test_the_xcb_libraries_are_the_ones_qt_links_and_no_system_policy_covers():
    assert all(not name.startswith(linux_package.SYSTEM_LIBRARIES) for name in linux_package.XCB_LIBRARIES)


def test_only_permissive_host_packages_may_ship_their_libraries(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    for package in ("zlib1g", "libgdbm6"):
        notice = bundle / "_internal/licenses/system" / package / "copyright"
        notice.parent.mkdir(parents=True)
        notice.write_text("")

    monkeypatch.setattr(linux_package.shutil, "which", lambda name: f"/usr/bin/{name}")
    errors = linux_package.check_bundle(bundle, "1.2.3")
    assert len(errors) == 1 and errors[0].startswith("copies a library from the host package libgdbm6,")

    # A pacman host names its packages differently, so the Debian table cannot judge them.
    monkeypatch.setattr(linux_package.shutil, "which", lambda name: None)
    assert linux_package.check_bundle(bundle, "1.2.3") == []


def test_no_host_package_is_listed_under_a_copyleft_license():
    assert linux_package.HOST_LIBRARY_PACKAGES
    assert not [
        package for package, license_name in linux_package.HOST_LIBRARY_PACKAGES.items()
        if "GPL" in license_name.upper()
    ]


def _dpkg_host(monkeypatch, listing):
    monkeypatch.setattr(linux_package.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        linux_package.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=listing),
    )


def test_a_host_library_ships_with_its_debian_copyright_file(monkeypatch):
    _dpkg_host(monkeypatch, "libffi8:amd64: /usr/lib/x86_64-linux-gnu/libffi.so.8\n")
    monkeypatch.setattr(linux_package.Path, "is_file", lambda path: True)

    [(destination, source, kind)] = linux_package.host_library_notices(["/lib/x86_64-linux-gnu/libffi.so.8"])
    assert (destination, Path(source).as_posix(), kind) == (
        "licenses/system/libffi8/copyright", "/usr/share/doc/libffi8/copyright", "DATA"
    )


def test_a_host_library_without_one_owner_or_a_copyright_file_stops_the_build(monkeypatch):
    _dpkg_host(monkeypatch, "")
    with pytest.raises(SystemExit, match="no package"):
        linux_package.host_library_notices(["/usr/lib/libmystery.so.1"])

    _dpkg_host(monkeypatch, "libffi8:amd64: /usr/lib/x86_64-linux-gnu/libffi.so.8\n")
    monkeypatch.setattr(linux_package.Path, "is_file", lambda path: False)
    with pytest.raises(SystemExit, match="no copyright file"):
        linux_package.host_library_notices(["/usr/lib/x86_64-linux-gnu/libffi.so.8"])


@pytest.mark.skipif(sys.platform != "linux", reason="install.sh runs on Linux")
def test_install_sh_points_every_desktop_command_at_the_extracted_copy(tmp_path):
    copy = tmp_path / "My & Copy|x"
    shutil.copytree(
        linux_package.DESKTOP_DIR, copy / "share" / "applications",
        ignore=lambda _dir, names: [name for name in names if not name.endswith(".desktop")],
    )
    (copy / "share" / "metainfo").mkdir(parents=True)
    (copy / "share" / "metainfo" / f"{linux_package.APP_ID}.metainfo.xml").write_text("")
    icon = copy / "share" / "icons" / "hicolor" / "48x48" / "apps" / f"{linux_package.APP_ID}.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"")
    (copy / "install.sh").write_text(linux_package.INSTALL_SH)

    subprocess.run(["sh", str(copy / "install.sh")], check=True, capture_output=True,
                   env={**os.environ, "XDG_DATA_HOME": str(tmp_path / "data")})

    entry = (tmp_path / "data" / "applications" / f"{linux_package.APP_ID}.desktop").read_text()
    commands = [line for line in entry.splitlines() if line.startswith("Exec=")]
    executable = f'"{copy}/AudioForge/AudioForge"'
    assert commands == [f"Exec={executable}", f"Exec={executable} --toggle-mute"]
