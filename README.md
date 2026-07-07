# AudioForge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

AudioForge is a Windows microphone processor for people who want a cleaner live mic without sending audio through a cloud service. It combines a Rust realtime audio core with a PyQt desktop UI for noise suppression, smart gating, Auto-EQ, Auto Voice Setup, latency calibration, and dynamics control.

Current version: `v1.8.5`

## Download

The latest portable build is available on the GitHub releases page:

- [AudioForge v1.8.5](https://github.com/FueledByRedBull/audio-forge/releases/tag/v1.8.5)
- Artifact: `AudioForge-v1.8.5-win64-ultra.7z`
- SHA-256: published with the release as `AudioForge-v1.8.5-win64-ultra.7z.sha256`

The portable bundle is self-contained. Extract it and run `AudioForge.exe`.

## What It Does

AudioForge sits between your microphone and your output/virtual routing path. It is built for voice work where reliability matters: streaming, calls, recording chains, monitoring, and calibration-heavy setups.

User-facing tools:

- AI noise suppression with RNNoise and optional DeepFilterNet backends.
- Smart noise gate with threshold-only, VAD-assisted, and VAD-only modes.
- Auto thresholding that tracks the live noise floor in VAD modes.
- 10-band parametric EQ with gain, Q, and per-band center frequencies.
- Auto-EQ calibration that records your voice, analyzes the spectrum, and applies a bounded correction.
- Auto Voice Setup wizard that records room noise plus speech and recommends EQ, gate/VAD, de-esser, and compressor settings in one pass.
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

AudioForge currently supports Windows 10/11 only. Source builds, CI, portable `dist/AudioForge` packaging, runtime assets, device recovery behavior, and desktop identity integration are validated on Windows. Linux and macOS builds are not supported today, even though parts of the Rust audio stack use cross-platform libraries.

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

## Configuration

RNNoise is the default safe noise-suppression backend. DeepFilterNet is opt-in for source and development runs; set `AUDIOFORGE_ENABLE_DEEPFILTER=1` when you want to use the DeepFilter backend with local `df.dll` and model assets. Packaged builds auto-enable DeepFilterNet when the bundled DeepFilter DLL and model assets are present.

See [Development Assets](#development-assets) for the full runtime asset and environment-variable list.

## Using The App

1. Select input and output devices.
2. Start processing.
3. Choose a suppressor backend and gate mode.
4. Tune EQ/dynamics manually, run Auto-EQ, or run Auto Voice Setup for a broader voice-chain calibration.
5. Run latency calibration if the current device route needs compensation.

Useful behavior to know:

- Device refresh keeps the current selection when the same device is still available.
- Input/output stream setup prefers 48 kHz configs when available.
- In VAD modes, auto threshold is the default path; the UI shows live noise floor and effective threshold.
- Preset loading preserves saved `VAD Assisted` and `VAD Only` gate modes instead of collapsing them back to `Threshold Only`.
- Diagnostics separate input drops, backlog recovery, output recovery, output short-write loss, and active output underrun streaks. Historical output underrun and recovery totals stay visible without forcing the health chip into a warning state after the stream has recovered.

## Development Assets

Full-feature development and release builds use the tracked `release-assets.json` manifest. Obtain each listed asset from the documented source, place it at the manifest `path`, and verify before packaging:

```powershell
.\.venv\Scripts\python.exe python/tools/verify_release_assets.py
```

For a cleaner fresh-clone setup, you can hydrate those assets from the matching GitHub release:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py --release-tag v1.8.5
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

Packaged builds prefer bundled DeepFilter assets. By default, bundled `df.dll` and the bundled DeepFilter model override external environment paths. Set `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` only when you intentionally want a packaged build to honor external `DEEPFILTER_LIB_PATH` and/or `DEEPFILTER_MODEL_PATH`; any missing path still defaults to the bundled asset.

## Build Portable EXE

Build the Rust extension first, then package:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py --release-tag v1.8.5
.\.venv\Scripts\python.exe -m maturin develop --release
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Packaging script behavior:

- Uses the locally built `python/mic_eq/mic_eq_core*.pyd`.
- Fails if the local native extension is older than Rust sources.
- Validates required full-feature runtime assets against `release-assets.json`.
- Reuses PyInstaller's analysis cache by default; pass `-Clean` for a cold PyInstaller rebuild.
- Bundles the Python runtime with PyInstaller.
- Prunes unused Qt payload and duplicate native-extension payload with `python/tools/prune_bundle.py` while retaining dependency metadata and licenses.
- Keeps the application self-contained in `dist/AudioForge`.

Portable output:

- `dist/AudioForge/AudioForge.exe`
- Bundled assets and runtime files under `dist/AudioForge/_internal`

## Create Release Archive

The portable folder is intended to be archived as a single distributable:

```powershell
& "C:/Program Files/7-Zip/7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.8.5-win64-ultra.7z .\dist\AudioForge\*
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

- `python/mic_eq`: PyQt application, analysis code, persistence, and source/development entrypoints.
- `rust-core`: Rust audio engine exposed through PyO3.
- `python/tests`: Python test suite.
- `python/tools`: health, package, and release validation helpers.
- `.github/workflows/ci.yml`: Windows CI for Python and Rust checks.
- `build_exe.ps1`: PyInstaller packaging script.
- `AudioForge.spec`: canonical portable package definition.
- `launcher.py`: PyInstaller/frozen-app launcher used by `AudioForge.spec`; source/development runs use `python -m mic_eq` or the `mic-eq` console entrypoint.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- RNNoise by Jean-Marc Valin
- DeepFilterNet by Hendrik Schroter and contributors
- Silero VAD contributors
