## What's Changed
- Fixed CPAL input and output stream setup to honor device sample formats (`f32`, `i16`, `u16`) instead of always assuming `f32`.
- Added output-side resampling so the 48 kHz DSP engine plays correctly on non-48 kHz output devices.
- Removed DeepFilter steady-state heap churn in the realtime path by reusing fixed buffers instead of allocating per frame.
- Reduced the packaged Windows archive size by dropping duplicate copied DeepFilter assets and using the bundled `_internal` runtime assets directly.

## Build
- Built from `master` with the full packaged feature set, including RNNoise, VAD, and DeepFilter support.
- Rust extension built in release mode with `maturin develop --release`.
- PyInstaller package built with `build_exe.ps1`.

## Validation
- `cargo test -p mic_eq_core --lib -- --nocapture`
- `cargo test -p mic_eq_core --tests`
- `python -m pytest python/tests -v`
- `python python/tools/self_test.py`
  - The self-test started the processor successfully but reported low confidence (`0.219`) against the default `0.25` threshold on this machine.

## Asset
- `AudioForge-v1.7.4-win64-ultra.7z`
- Size: `112 MiB`
- SHA256: `60E7BA5E98FF91BED0FA23F8E46F1CBA72DE804652CB572FF7C7065C2DB0FB4D`
