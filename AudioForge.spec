# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


repo_root = Path(globals().get("SPECPATH", ".")).resolve()
python_source = repo_root / "python"

binaries = []
datas = []

df_dll = repo_root / "df.dll"
if df_dll.exists():
    binaries.append((str(df_dll), "."))

models_dir = repo_root / "models"
if models_dir.exists():
    datas.append((str(models_dir), "models"))
icon_file = repo_root / "mic_eq.ico"


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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt6.QtPdf",
        "PyQt6.QtPdfWidgets",
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name="AudioForge",
)
