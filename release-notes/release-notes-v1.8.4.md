# AudioForge v1.8.4

## Highlights

- Hardened config startup and stream recovery against corrupt persisted state, stale device IDs, and malformed runtime diagnostics.
- Fixed VAD startup status so restored `VAD Assisted` and `VAD Only` gate modes report active state after the backend becomes ready.
- Improved dynamics and analysis correctness with post-compression auto makeup targeting, bounded EQ boost/cut selection, and spectral-tilt regression.
- Split realtime processor routing, resampling, diagnostics, and output writing into focused modules with regression coverage for drift retime and multichannel mixdown behavior.
- Refined diagnostics so historical underrun/recovery totals stay visible without keeping the UI in a warning state after recovery.
- Reduced the portable package footprint by pruning the duplicate native-extension payload and making package smoke reject that duplicate.

## Validation

- `.\\.venv\\Scripts\\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\\.venv\\Scripts\\python.exe -m pyright`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests -q`
- `.\\.venv\\Scripts\\python.exe python\\tools\\check_versions.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py --source-only`
- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `cargo clippy -p mic_eq_core --all-targets -- -D warnings`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\health_check.py --duration 30 --allow-recovery`

## Artifact

- `AudioForge-v1.8.4-win64-ultra.7z`
- SHA-256: published as `AudioForge-v1.8.4-win64-ultra.7z.sha256`
