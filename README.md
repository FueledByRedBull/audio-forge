# AudioForge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

Low-latency Windows microphone processor with AI noise suppression, smart gating, Auto-EQ, latency calibration, and a portable desktop build.

Current version: `v1.7.14`

## Status

AudioForge is a Windows-first desktop app with a Python/PyQt UI and a Rust real-time audio core. The repository is set up for local source builds and portable `dist/AudioForge` packaging.

## Features

- Noise suppression backends:
  - RNNoise
  - DeepFilterNet LL
  - DeepFilterNet Standard
- Noise gate modes:
  - Threshold-only gate
  - VAD-assisted gate
  - VAD-only gate
  - Auto threshold defaults on in VAD modes and tracks the estimated noise floor plus margin
- 10-band parametric EQ with per-band frequency, gain, and Q
- Auto-EQ workflow with recording, spectral analysis, bounded center-frequency nudging, and one-click apply/undo
- Split-band de-esser with manual and auto amount control
- Compressor with soft knee, adaptive release, and optional auto makeup
- Lookahead limiter
- Raw monitor mode for direct diagnostic monitoring
- Real-time health and diagnostics:
  - input/output meters
  - callback stall detection
  - dropped-sample counters
  - backlog and recovery counters
  - backend status/error reporting
- Callback watchdog with stream restart/backoff handling
- Device persistence and refresh that preserves the current selection when possible
- Per device-pair latency calibration profiles with migration from older saved keys
- Portable PyInstaller build with bundled Python runtime and model assets

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
- Python 3.9+
- Rust 1.70+
- `maturin`
- A virtual environment in `.venv` is assumed by the packaging script

## Repository Layout

- `python/mic_eq`: PyQt application, analysis code, persistence, packaging entrypoints
- `rust-core`: Rust audio engine exposed through PyO3
- `python/tests`: Python test suite
- `dist/AudioForge`: packaged portable application output
- `build_exe.ps1`: PyInstaller packaging script

## Build And Run From Source

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

## Runtime Assets

Create `models/` in the repo root if you want local runtime discovery during development:

- `models/DeepFilterNet3_ll_onnx.tar.gz`
- `models/DeepFilterNet3_onnx.tar.gz`
- `models/silero_vad.onnx`

DeepFilter runtime library:

- `df.dll` in the repo root for development runs
- bundled under `dist/AudioForge/_internal` for portable builds

Environment variables:

- `DEEPFILTER_MODEL_PATH`
- `DEEPFILTER_LIB_PATH`
- `AUDIOFORGE_ALLOW_EXTERNAL_DF=1`
- `AUDIOFORGE_ENABLE_DEEPFILTER`
- `VAD_MODEL_PATH`

Packaged builds prefer bundled DeepFilter assets. `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` should only be used when you intentionally want a packaged build to resolve `df.dll` externally.

## Using The App

1. Select the input and output devices.
2. Start processing.
3. Choose a suppressor backend and gate mode.
4. Tune EQ/dynamics manually or run Auto-EQ from the calibration flow.
5. If the route needs compensation, run latency calibration for the current device pair.

Operational notes:

- Device refresh keeps the current selection when the same device is still available.
- Input/output stream setup prefers 48 kHz configs when available.
- In VAD modes, auto gate threshold is the default path; the UI shows the live noise floor and effective threshold while the manual threshold remains available as fallback.
- Runtime diagnostics expose input drops, backlog recovery, output recovery, and short-write loss separately.
- Dropped-sample counters can be reset from the UI.

## Build Portable EXE

Build the Rust extension first, then package:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Packaging script behavior:

- uses the locally built `python/mic_eq/mic_eq_core*.pyd`
- bundles the Python runtime with PyInstaller
- prunes the final bundle with `python/tools/prune_bundle.py`
- keeps the entire application self-contained in `dist/AudioForge`

Portable output:

- `dist/AudioForge/AudioForge.exe`
- bundled assets and runtime files under `dist/AudioForge/_internal`

## Create Release Archive

The portable folder is intended to be archived as a single distributable:

```powershell
& "C:/Program Files/7-Zip/7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.7.14-win64-ultra.7z .\dist\AudioForge\*
```

This uses LZMA2 with max compression and solid mode, which is appropriate for the PyInstaller bundle.

## Testing

Rust:

```powershell
cd rust-core
cargo test -p mic_eq_core --tests
```

Python:

```powershell
cd ..
.\.venv\Scripts\python.exe -m pytest python/tests -v
```

Targeted checks used frequently during development:

```powershell
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests
.\.venv\Scripts\python.exe -m pytest python/tests/test_auto_eq.py python/tests/test_spectrum.py
.\.venv\Scripts\python.exe -m pytest python/tests/test_config_v17.py python/tests/test_ui_sample_rate_and_diagnostics.py
```

Headless checks:

```powershell
.\.venv\Scripts\python.exe python/tools/health_check.py --duration 1800
.\.venv\Scripts\python.exe python/tools/self_test.py
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- RNNoise by Jean-Marc Valin
- DeepFilterNet by Hendrik Schroter and contributors
- Silero VAD contributors
