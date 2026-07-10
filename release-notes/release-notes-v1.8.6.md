# AudioForge v1.8.6

## Highlights

- True-peak protection now uses a band-limited 4x detector validated against an independent reference oversampler instead of cubic interpolation.
- Phase-safe mono keeps fractional-delay state across callback blocks, and adaptive cleanup tracks drifting 49-61 Hz mains hum and harmonics without stacking high-pass filters.
- Auto-EQ reports Python fallback headroom checks as advisory and reserves authoritative safety decisions for the native Rust chain simulator.
- Auto Voice Setup now uses VAD-masked BS.1770 short-term loudness, loudness range, robust band energy, offline chain validation, and visible uncertainty before applying weak captures.
- Rapid UI automation uses time-based biquad morphs, with seeded concurrent control/DSP stress coverage in release mode.

## Security and reproducibility

- DeepFilter only trusts canonical paths registered by the application bootstrap unless `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` explicitly enables `DEEPFILTER_LIB_PATH` or `DEEPFILTER_MODEL_PATH`.
- GitHub Actions are pinned to reviewed commit SHAs and use read-only repository permissions except for the isolated release publishing job.
- Python runtime/build dependencies are hash locked, audited with pip-audit, and updated through Dependabot.
- CI preserves Semgrep SARIF and runs RustSec against `Cargo.lock`.
- PyO3/numpy were upgraded to 0.29 to resolve the current RustSec advisories.

## Validation

- Hash-verified installation from `requirements/dev.txt`
- `pip-audit --require-hashes` on both runtime and development/build locks
- Reviewed Semgrep Python, Rust, secrets, GitHub Actions, and third-party audit rules with telemetry disabled
- `cargo audit`
- Ruff, Pyright, pytest, version consistency, and source package smoke checks
- Rust formatting, full tests, Clippy with warnings denied, and release-mode contention stress tests
- Explicit hardware-only CPAL/WASAPI enumeration and lifecycle smoke checks on a Windows machine with real audio endpoints
- Verified runtime assets, a full PyInstaller portable build, and packaged-bundle smoke checks

Release-mode microbenchmarks on the validation machine measured phase-safe mono
at 40.34 ns/frame for 128-frame callbacks (average mixdown: 1.63 ns/frame) and
automated biquad processing at 8.26 ns/sample (steady: 7.26 ns/sample, 1.14x).

## Artifact

The release workflow produces `AudioForge-v1.8.6-win64-ultra.7z` and the matching `AudioForge-v1.8.6-win64-ultra.7z.sha256` checksum sidecar.
