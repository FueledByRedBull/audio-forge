"""The Linux bundle check refuses bundles that would misbehave on a user's desktop."""

from __future__ import annotations

import json
import subprocess

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
        "_internal/libxcb-cursor.so.0",
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


def _dpkg_host(monkeypatch, listing):
    monkeypatch.setattr(linux_package.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        linux_package.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=listing),
    )


def test_a_host_library_ships_with_its_debian_copyright_file(monkeypatch):
    _dpkg_host(monkeypatch, "libffi8:amd64: /usr/lib/x86_64-linux-gnu/libffi.so.8\n")
    monkeypatch.setattr(linux_package.Path, "is_file", lambda path: True)

    assert linux_package.host_library_notices(["/lib/x86_64-linux-gnu/libffi.so.8"]) == [
        ("licenses/system/libffi8/copyright", "/usr/share/doc/libffi8/copyright", "DATA")
    ]


def test_a_host_library_without_one_owner_or_a_copyright_file_stops_the_build(monkeypatch):
    _dpkg_host(monkeypatch, "")
    with pytest.raises(SystemExit, match="no package"):
        linux_package.host_library_notices(["/usr/lib/libmystery.so.1"])

    _dpkg_host(monkeypatch, "libffi8:amd64: /usr/lib/x86_64-linux-gnu/libffi.so.8\n")
    monkeypatch.setattr(linux_package.Path, "is_file", lambda path: False)
    with pytest.raises(SystemExit, match="no copyright file"):
        linux_package.host_library_notices(["/usr/lib/x86_64-linux-gnu/libffi.so.8"])
