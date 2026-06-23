# AudioForge v1.8.2

## Highlights

- Improved Auto-EQ confidence reporting so capture confidence, EQ confidence, and post-validation confidence are separated instead of collapsing into a single misleading score.
- Added a new `Auto Voice Setup` wizard that records room noise plus speech and recommends EQ, gate/VAD, de-esser, and compressor settings in one pass while leaving limiter settings unchanged.
- Fixed preset restore for the noise gate so saved `VAD Assisted` and `VAD Only` modes no longer revert to `Threshold Only` during load when VAD availability has not been reported yet.
- Kept the Windows taskbar/app identity metadata fixes and startup backlog recovery fixes that are already queued on `master` ahead of the last published release.

## Validation

- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `.\\.venv\\Scripts\\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\\.venv\\Scripts\\python.exe -m pyright`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests -q`
- `.\\.venv\\Scripts\\python.exe python\\tools\\check_versions.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\verify_release_assets.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py --source-only`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py`

## Artifact

- `AudioForge-v1.8.2-win64-ultra.7z`
- SHA-256: published as `AudioForge-v1.8.2-win64-ultra.7z.sha256`
