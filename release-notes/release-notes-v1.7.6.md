## What's Changed
- Reduced the packaged Windows archive further by removing the duplicated runtime icon payload from the frozen app and pruning unused Qt translations/PDF binaries after the PyInstaller build.
- Prefer bundled `_internal/models` and `_internal/df.dll` more aggressively in frozen runtime lookup so the packaged asset layout stays canonical.
- Kept the full feature set intact: RNNoise, VAD, DeepFilter LL, and DeepFilter Standard remain bundled in the release build.

## Build
- Built from `master` with the configured full feature set: `pyo3/extension-module`, `vad`, and `deepfilter`.
- Rust extension built with `maturin develop --release`.
- Windows package built from `AudioForge.spec` via `build_exe.ps1`.

## Validation
- `cargo test -p mic_eq_core --lib -- --nocapture`
- `python -m pytest python/tests -q`
- `python -m maturin develop --release`
- `powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`
- `python python/tools/self_test.py`
  - Processor startup and capture succeeded, but this machine still finished below the default confidence threshold on March 9, 2026 (`0.246` vs `0.25`).

## Asset
- `AudioForge-v1.7.6-win64-ultra.7z`
- Size: `94.49 MiB`
- SHA256: `C4D6A6217A6A3D4DF50A1C77220A9271D6421044137DA1B82EE25434A602C236`
