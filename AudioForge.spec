# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import hashlib
import json
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

df_dll = repo_root / "df.dll"
if df_dll.exists():
    binaries.append((str(df_dll), "."))

ort_lib_dir = repo_root / "target" / "onnxruntime-cpu" / "lib"
for ort_dll_name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
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


a = Analysis(
    ["launcher.py"],
    pathex=[str(repo_root.resolve()), str(python_source.resolve())],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "json",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "mic_eq.mic_eq_core",
        "mic_eq",
        "mic_eq.ui",
        "pythoncom",
        "pywintypes",
        "win32com.propsys.propsys",
        "win32com.propsys.pscon",
    ],
    hookspath=[str(hook_source.resolve())],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt6.QtPdf",
        "PyQt6.QtPdfWidgets",
        "ssl",
        "_ssl",
        "_hashlib",
        "pytest",
        "setuptools",
        "wheel",
    ],
    noarchive=False,
    optimize=0,
)

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
    icon=[str(icon_file)] if icon_file.exists() else None,
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
