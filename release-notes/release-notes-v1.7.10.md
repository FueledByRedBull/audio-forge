# AudioForge v1.7.10

## Highlights

- Fixed the packaged EXE startup failure caused by stripping SciPy's `_highspy` dependency out of the Windows bundle.
- Restored the required SciPy optimize payload in the PyInstaller spec and removed the bad prune rule from the post-build cleanup step.
- Rebuilt and smoke-tested the portable app so `dist/AudioForge/AudioForge.exe` starts normally again.

## Validation

- `python -m pytest python/tests -q`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- direct packaged EXE startup check from `dist/AudioForge`

## Artifact

- `AudioForge-v1.7.10-win64-ultra.7z`
- Size: `99,025,746` bytes (`94.44 MiB`)
- SHA-256: `0316B944A4B918F6B6B97262B96FF1D086AB8D8746DB49F3C0D1C64563D0A467`
