# AudioForge v1.8.3

## Highlights

- Completed the public MicEq-to-AudioForge identity cleanup by renaming the root icon asset, updating package references, and removing the superseded `build.ps1` helper.
- Clarified Windows-only support, DeepFilterNet opt-in behavior, packaged DeepFilter auto-enable behavior, and the PyInstaller launcher role in the README.
- Added rotating desktop app logs under `%APPDATA%/AudioForge/logs/app.log` so runtime errors are easier to collect from packaged builds.
- Moved stream recovery and device-selection helpers out of the main window, and split high-frequency meter polling from slower diagnostics/recovery polling.

## Validation

- `.\\.venv\\Scripts\\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\\.venv\\Scripts\\python.exe -m pyright`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests -q`
- `.\\.venv\\Scripts\\python.exe python\\tools\\check_versions.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py --source-only`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py`

## Artifact

- `AudioForge-v1.8.3-win64-ultra.7z`
- SHA-256: published as `AudioForge-v1.8.3-win64-ultra.7z.sha256`
