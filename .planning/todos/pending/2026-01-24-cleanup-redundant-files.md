---
created: 2026-01-24T21:47
title: Clean up redundant files and bloat from MicEq directory
area: tooling
files:
  - target/ (4.8GB Rust build artifacts)
  - build/ (94MB PyInstaller intermediate files)
  - dist/models/ (90MB duplicate models)
  - deepfilter_impl.md (research doc for implemented feature)
  - DEEPFILTER_TEST_REPORT.md (test report for completed feature)
  - ICON_GUIDE.md (completed task instructions)
  - test_deepfilter.py (dev test script)
  - test_output.txt, nul, err.txt, out.txt (empty test files)
  - df.dll (15.9MB unused DeepFilterNet C library)
  - mic_eq.spec (superseded by MicEq.spec)
---

## Problem

Comprehensive scan revealed significant bloat in MicEq directory:

**Build Artifacts (~5GB):**
- `target/` directory (4.8GB) - Rust build artifacts that can be regenerated
- `build/` directory (94MB) - PyInstaller intermediate files

**Duplicate Files (90MB+):**
- `dist/models/` duplicates of source models
- Multiple similar PyInstaller spec files (mic_eq.spec, MicEq.spec, MicEq_Debug.spec)

**Research/Development Files (30KB):**
- `deepfilter_impl.md` - Research doc for feature already implemented via C FFI
- `DEEPFILTER_TEST_REPORT.md` - Test report for completed DeepFilterNet integration
- `ICON_GUIDE.md` - Icon creation instructions (icon already exists)
- `test_deepfilter.py` - Development test script

**Test/Temporary Files:**
- `test_output.txt`, `nul`, `err.txt`, `out.txt` - Empty or temporary files
- `df.dll` (15.9MB) - Unused DeepFilterNet C library (superseded by ONNX implementation)

These files clutter the project, consume 5.1GB of disk space, and create confusion about which files are current.

## Solution

Execute cleanup commands in order:

**1. Build Artifacts (~5GB savings):**
```powershell
rmdir /s /q "target"
rmdir /s /q "build"
```

**2. Duplicate Models (90MB savings):**
```powershell
rmdir /s /q "dist\models"
```

**3. Research Files (completed features):**
```powershell
del "deepfilter_impl.md"
del "DEEPFILTER_TEST_REPORT.md"
del "ICON_GUIDE.md"
```

**4. Test and Temporary Files:**
```powershell
del "test_deepfilter.py"
del "test_output.txt"
del "nul"
del "dist\err.txt"
del "dist\out.txt"
```

**5. Unused Binary:**
```powershell
del "df.dll"
```

**6. Old Spec File:**
```powershell
del "mic_eq.spec"
```

**After cleanup:**
- 5.1GB disk space savings
- 12 files removed
- 3 directories removed
- Cleaner project structure
- Keep `build.ps1` as primary build script, verify `build_exe.ps1` is still needed

**Note:** All removed files are properly ignored by `.gitignore` and won't affect the repository or ability to rebuild.
