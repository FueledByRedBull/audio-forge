# AudioForge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

AudioForge is a Windows microphone processor for people who want a cleaner live mic without sending audio through a cloud service. It combines a Rust realtime audio core with a PyQt desktop UI for noise suppression, smart gating, Auto-EQ, latency calibration, and dynamics control.

Current version: `v1.7.18`

## Download

The latest portable build is available on the GitHub releases page:

- [AudioForge v1.7.18](https://github.com/FueledByRedBull/audio-forge/releases/tag/v1.7.18)
- Artifact: `AudioForge-v1.7.18-win64-ultra.7z`
- SHA-256: `17F844C8FBF8E4D79F0A3617969D637ECA89005DD853C902039CEEE19570D987`

The portable bundle is self-contained. Extract it and run `AudioForge.exe`.

## What It Does

AudioForge sits between your microphone and your output/virtual routing path. It is built for voice work where reliability matters: streaming, calls, recording chains, monitoring, and calibration-heavy setups.

User-facing tools:

- AI noise suppression with RNNoise and optional DeepFilterNet backends.
- Smart noise gate with threshold-only, VAD-assisted, and VAD-only modes.
- Auto thresholding that tracks the live noise floor in VAD modes.
- 10-band parametric EQ with gain, Q, and per-band center frequencies.
- Auto-EQ calibration that records your voice, analyzes the spectrum, and applies a bounded correction.
- Dynamic-EQ de-esser, compressor, auto makeup gain, and lookahead limiter.
- Per device-pair latency calibration profiles.
- Raw monitor and bypass paths for troubleshooting.

Operational tools:

- Input/output meters and runtime diagnostics.
- Dropped-sample, backlog, callback-stall, and recovery counters.
- Stream restart/backoff handling.
- Device refresh that preserves current selections when possible.
- Portable PyInstaller packaging with bundled runtime assets.

## Status

AudioForge is Windows-first. The repository supports source builds, local development, and portable `dist/AudioForge` packaging. It is not currently shipped as a signed installer.

DeepFilterNet support is intentionally opt-in at runtime. Use `AUDIOFORGE_ENABLE_DEEPFILTER=1` when you want to exercise the DeepFilter backend; RNNoise remains the default safe path.
DeepFilter model/DLL initialization and Silero VAD inference are prepared off the realtime DSP loop; the audio path only swaps ready suppressor state and consumes cached VAD probabilities.

## DSP Chain

Normal processing path:

```text
Mic Input -> Pre-Filter (DC block + 80 Hz HP) -> Noise Gate -> Noise Suppression
-> De-Esser -> 10-Band EQ -> Compressor -> Limiter -> Output
```

Special paths:

- `Bypass` keeps the transport path active while skipping the main DSP stages.
- `Raw Monitor` uses the clean write path and skips the pre-filter and downstream DSP chain for diagnostics.

Latency labels in the UI describe suppressor/DSP behavior, not full round-trip latency. End-to-end latency still depends on the selected devices, driver mode, buffer sizing, and routing path.

## Requirements

- Windows 10/11
- Python 3.10+
- Rust 1.70+
- `maturin`
- A virtual environment in `.venv` is assumed by the packaging script.

## Quick Start From Source

```powershell
git clone https://github.com/FueledByRedBull/audio-forge.git
cd audio-forge

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]

.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m mic_eq
```

You can also use the installed console entrypoint:

```powershell
.\.venv\Scripts\mic-eq.exe
```

## Using The App

1. Select input and output devices.
2. Start processing.
3. Choose a suppressor backend and gate mode.
4. Tune EQ/dynamics manually or run Auto-EQ.
5. Run latency calibration if the current device route needs compensation.

Useful behavior to know:

- Device refresh keeps the current selection when the same device is still available.
- Input/output stream setup prefers 48 kHz configs when available.
- In VAD modes, auto threshold is the default path; the UI shows live noise floor and effective threshold.
- Diagnostics separate input drops, backlog recovery, output recovery, and short-write loss.

## Development Assets

Full-feature development and release builds use the tracked `release-assets.json` manifest. Obtain each listed asset from the documented source, place it at the manifest `path`, and verify before packaging:

```powershell
.\.venv\Scripts\python.exe python/tools/verify_release_assets.py
```

Create `models/` in the repo root for local runtime discovery:

- `models/DeepFilterNet3_ll_onnx.tar.gz`
- `models/DeepFilterNet3_onnx.tar.gz`
- `models/silero_vad.onnx`

DeepFilter runtime library:

- `df.dll` in the repo root for development runs.
- `target/release/DirectML.dll` from the pinned DirectML redistributable package for full-feature packaging.
- Bundled under `dist/AudioForge/_internal` for portable builds.

Environment variables:

- `AUDIOFORGE_ENABLE_DEEPFILTER=1`
- `AUDIOFORGE_ALLOW_EXTERNAL_DF=1`
- `DEEPFILTER_MODEL_PATH`
- `DEEPFILTER_LIB_PATH`
- `VAD_MODEL_PATH`

Packaged builds prefer bundled DeepFilter assets. `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` should only be used when you intentionally want a packaged build to resolve `df.dll` externally.

## Build Portable EXE

Build the Rust extension first, then package:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Packaging script behavior:

- Uses the locally built `python/mic_eq/mic_eq_core*.pyd`.
- Fails if the local native extension is older than Rust sources.
- Validates required full-feature runtime assets against `release-assets.json`.
- Bundles the Python runtime with PyInstaller.
- Prunes unused Qt payload with `python/tools/prune_bundle.py` while retaining dependency metadata and licenses.
- Keeps the application self-contained in `dist/AudioForge`.

Portable output:

- `dist/AudioForge/AudioForge.exe`
- Bundled assets and runtime files under `dist/AudioForge/_internal`

## Create Release Archive

The portable folder is intended to be archived as a single distributable:

```powershell
& "C:/Program Files/7-Zip/7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.7.18-win64-ultra.7z .\dist\AudioForge\*
```

This uses LZMA2 with max compression and solid mode, which is appropriate for the PyInstaller bundle.

## Testing

CI-equivalent checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe python/tools/check_versions.py
.\.venv\Scripts\python.exe python/tools/package_smoke.py --source-only
cargo fmt --check
cargo test -p mic_eq_core
cargo clippy -p mic_eq_core --all-targets -- -D warnings
```

Packaged-build smoke check after `build_exe.ps1`:

```powershell
.\.venv\Scripts\python.exe python/tools/verify_release_assets.py
.\.venv\Scripts\python.exe python/tools/package_smoke.py
```

Headless runtime checks:

```powershell
.\.venv\Scripts\python.exe python/tools/health_check.py --duration 1800
.\.venv\Scripts\python.exe python/tools/self_test.py
```

## Repository Layout

- `python/mic_eq`: PyQt application, analysis code, persistence, packaging entrypoints.
- `rust-core`: Rust audio engine exposed through PyO3.
- `python/tests`: Python test suite.
- `python/tools`: health, package, and release validation helpers.
- `.github/workflows/ci.yml`: Windows CI for Python and Rust checks.
- `build_exe.ps1`: PyInstaller packaging script.
- `AudioForge.spec`: canonical portable package definition.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- RNNoise by Jean-Marc Valin
- DeepFilterNet by Hendrik Schroter and contributors
- Silero VAD contributors
