# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import hashlib
import json
import sys
import sysconfig


repo_root = Path(globals().get("SPECPATH", ".")).resolve()
python_source = repo_root / "python"
hook_source = repo_root / "pyinstaller-hooks"

extension_dir = python_source / "mic_eq"
if extension_dir.is_dir():
    extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".pyd")
    current_extension = extension_dir / f"mic_eq_core{extension_suffix}"
    if not current_extension.is_file():
        raise ValueError(
            f"Current Python ABI extension is missing: {current_extension.name}"
        )

binaries = []
datas = []

# The Qt Quick scene is loaded from disk next to the package.
datas.append((str(python_source / "mic_eq" / "ui" / "qml"), "mic_eq/ui/qml"))

project_license = repo_root / "LICENSE"
if project_license.exists():
    datas.append((str(project_license), "licenses"))

licenses_dir = repo_root / "licenses"
if licenses_dir.exists():
    for license_file in sorted(licenses_dir.iterdir()):
        if license_file.is_file():
            datas.append((str(license_file), "licenses"))

# Bundle only notices named by this build's inventory, not stale cache files.
dependency_licenses = repo_root / "build" / "dependency-licenses"
inventory_file = dependency_licenses / "inventory.json"
inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
datas.append((str(inventory_file), "licenses/dependencies"))
components = [
    inventory["python"],
    *inventory["python_components"],
    *inventory["rust_components"],
    *inventory["native_components"],
]
for component in components:
    for notice in component["notices"]:
        path = (dependency_licenses / notice["file"]).resolve()
        relative = path.relative_to(dependency_licenses.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != notice["sha256"]:
            raise ValueError(f"Dependency notice digest mismatch: {relative}")
        datas.append((str(path), (Path("licenses/dependencies") / relative.parent).as_posix()))

windows = sys.platform == "win32"

df_dll = repo_root / ("df.dll" if windows else "libdf.so")
if df_dll.exists():
    binaries.append((str(df_dll), "."))

ort_lib_dir = repo_root / "target" / "onnxruntime-cpu" / "lib"
ort_names = (
    ("onnxruntime.dll", "onnxruntime_providers_shared.dll")
    if windows
    else ("libonnxruntime.so.1.23.2",)
)
for ort_dll_name in ort_names:
    ort_dll = ort_lib_dir / ort_dll_name
    if ort_dll.exists():
        binaries.append((str(ort_dll), "."))

models_dir = repo_root / "models"
for model_name in (
    "DeepFilterNet3_ll_onnx.tar.gz",
    "DeepFilterNet3_onnx.tar.gz",
    "silero_vad.onnx",
):
    model_path = models_dir / model_name
    if model_path.exists():
        datas.append((str(model_path), "models"))
icon_file = repo_root / "AudioForge.ico"
if not windows and icon_file.exists():
    # Windows reads the icon from the executable; an ELF file has none.
    datas.append((str(icon_file), "."))


a = Analysis(
    ["launcher.py"],
    pathex=[str(repo_root.resolve()), str(python_source.resolve())],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "json",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        # libshiboken imports it when the Qt Quick bindings load; the frozen app fails to start without it.
        "PySide6.QtOpenGL",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.support.deprecated",
        "mic_eq.mic_eq_core",
        "mic_eq",
        "mic_eq.ui",
        # Windows only; elsewhere PyInstaller notes them as not found and moves on.
        "pythoncom",
        "pywintypes",
        # ShellLink.GetPath creates timestamps through this dynamic import.
        "win32timezone",
        "win32com.propsys.propsys",
        "win32com.propsys.pscon",
    ],
    hookspath=[str(hook_source.resolve())],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # The COM shell/property APIs do not use Pythonwin's MFC UI bindings.
        "win32ui",
        "PyQt6",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtVirtualKeyboard",
        # Linux builds would ship GPL libreadline for an interactive prompt the app never shows.
        "readline",
        "ssl",
        "_ssl",
        "_hashlib",
        "pytest",
        "setuptools",
        "wheel",
        # Development-only (requirements/dev.txt installs it in the build environment).
        "scipy",
        # Optional scientific-library helpers installed by development tools.
        "cffi",
        "charset_normalizer",
        "typing_extensions",
        "yaml",
    ],
    noarchive=False,
    optimize=0,
)

if not windows:
    import os
    import re
    import subprocess

    sys.path.insert(0, str(python_source / "tools"))
    from linux_package import SYSTEM_LIBRARIES as system_libraries
    from linux_package import host_library_notices
    from linux_package import UNUSED_QT_PLUGINS as unused_qt_plugins

    host_library_dirs = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")

    def unused_plugin(entry):
        dest = entry[0].replace("\\", "/")
        return any(f"PySide6/Qt/plugins/{plugin}" in dest for plugin in unused_qt_plugins)

    def from_host(entry):
        return entry[1].startswith(host_library_dirs) and entry[2] != "EXTENSION"

    a.binaries = [entry for entry in a.binaries if not unused_plugin(entry)]
    a.datas = [entry for entry in a.datas if not unused_plugin(entry)]
    def direct_needs(path):
        # Direct DT_NEEDED entries only; ldd-style resolution would report the
        # system libraries' own dependencies as if the app linked them.
        dynamic = subprocess.run(
            ["readelf", "-d", path], capture_output=True, text=True, check=True,
            env={**os.environ, "LC_ALL": "C"},
        ).stdout
        return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dynamic)

    def qt_library(entry):
        return "/PySide6/Qt/lib/" in entry[1]

    # Keep a build-host or Qt library only when something the app ships links
    # it, without following links into the system libraries above: a stray copy
    # would shadow the system's own through the bundle's library path.
    collected = {Path(entry[0]).name: entry[1] for entry in a.binaries}
    pending = [
        entry[1] for entry in a.binaries
        if not (from_host(entry) or qt_library(entry)) or Path(entry[0]).name.startswith("libpython")
    ]
    needed = set()
    while pending:
        for name in direct_needs(pending.pop()):
            if name in needed or name.startswith(system_libraries) or name not in collected:
                continue
            needed.add(name)
            pending.append(collected[name])
    a.binaries = [
        entry for entry in a.binaries
        if not Path(entry[0]).name.startswith(system_libraries)
        and (not (from_host(entry) or qt_library(entry)) or Path(entry[0]).name in needed
             or Path(entry[0]).name.startswith("libpython"))
    ]
    # PyInstaller files the top-level links to Qt's libraries with the data.
    kept = {entry[0] for entry in a.binaries}
    a.datas = [
        entry for entry in a.datas
        if entry[2] != "SYMLINK"
        or os.path.normpath(os.path.join(os.path.dirname(entry[0]), entry[1])) in kept
    ]
    # libpython's terms come with the CPython inventory.
    a.datas += host_library_notices([
        entry[1] for entry in a.binaries
        if from_host(entry) and not Path(entry[0]).name.startswith("libpython")
    ])

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AudioForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Linux takes the icon from the desktop entry, not the executable.
    icon=[str(icon_file)] if windows and icon_file.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AudioForge",
)
