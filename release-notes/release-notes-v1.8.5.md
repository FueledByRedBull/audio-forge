# AudioForge v1.8.5

## Highlights

- Added input-channel intelligence with left/right/average/max-RMS/phase-safe mono modes and negative-correlation warnings for phase-cancellation-prone stereo inputs.
- Added careful output protection with a lower limiter ceiling, output clip diagnostics, and true-peak-style warning telemetry.
- Improved gate behavior with smoothed decision hysteresis, chatter detection, and health-panel warnings for rapid open/close transitions.
- Added a compact health panel for input level, output clipping/true peak, gate chatter, backend state, callback age, and underruns.
- Improved voice dynamics with compressor sidechain high-pass detection, de-esser detector confidence, and RNNoise model-input soft clipping for hot input.
- Made Auto-EQ more controllable with explicit adaptive/static target modes, smoothing strength controls, and regularized narrow-residual correction diagnostics.

## Validation

- `.\\.venv\\Scripts\\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\\.venv\\Scripts\\python.exe -m pyright`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests -q --basetemp=.pytest-tmp -o cache_dir=.pytest-tmp/cache`
- `.\\.venv\\Scripts\\python.exe python\\tools\\check_versions.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py --source-only`
- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `cargo clippy -p mic_eq_core --all-targets -- -D warnings`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py`

## Artifact

- `AudioForge-v1.8.5-win64-ultra.7z`
- SHA-256: published as `AudioForge-v1.8.5-win64-ultra.7z.sha256`
