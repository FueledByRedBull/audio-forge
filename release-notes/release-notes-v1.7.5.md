## What's Changed
- Hardened DeepFilter runtime handling so packaged builds prefer the bundled `df.dll`, runtime failures become sticky backend errors, and UI diagnostics report fallback state explicitly.
- Added richer processor diagnostics for backend health, non-finite suppressor resets, recovery state, underruns, dropped samples, and output device sample rate.
- Moved calibration and latency-capture processor control back to the main Qt thread; background workers now handle analysis only.
- Tuned watchdog recovery suppression for calibration/self-test workflows and required consecutive stall checks before auto-restart.
- Made the headless self-test device-rate aware and added an automatic retry window for timing-sensitive capture paths.
- Reduced the packaged Windows archive size by making `AudioForge.spec` canonical and removing the duplicated `python/mic_eq` source payload from the frozen app.

## Build
- Built from `master` with the configured full feature set: `pyo3/extension-module`, `vad`, and `deepfilter`.
- Rust extension built with `maturin develop --release`.
- Windows package built from `AudioForge.spec` via `build_exe.ps1`.

## Validation
- `cargo test -p mic_eq_core --lib -- --nocapture`
- `python -m pytest python/tests -q`
- `python -c "import mic_eq.ui.calibration_dialog, mic_eq.ui.latency_calibration_dialog, mic_eq.ui.main_window, mic_eq.config"`
- `python python/tools/self_test.py`
  - Passed on retry with `rt=134.29ms`, `confidence=0.284`.

## Asset
- `AudioForge-v1.7.5-win64-ultra.7z`
- Size: `103.88 MiB`
- SHA256: `3A48C783F468B639366B2FE3AE6E58003FBB63E5E497D987647E3CDEE68B5661`
