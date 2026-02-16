# AudioForge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

Low-latency microphone audio processor with AI noise suppression, smart gating, Auto-EQ, and 10-band parametric EQ.

Current version: `v1.7.0`

## Status

AudioForge is currently focused on Windows desktop use and portable distribution. It is optimized for personal use and small friend-group sharing.

## Features

- AI noise suppression backends:
  - RNNoise (fast baseline)
  - DeepFilterNet LL (low-latency quality mode)
  - DeepFilterNet Standard (highest quality, higher latency)
- Smart gate modes:
  - Threshold-only gate
  - VAD-assisted gate
  - VAD-only gate
- 10-band parametric EQ with per-band frequency, gain, and Q controls
- Wideband de-esser with manual controls and auto amount mode
- Auto-EQ recording/analysis flow with validation and one-click apply/undo
- Dynamics:
  - Compressor (soft-knee, adaptive release, auto makeup support)
  - Hard limiter (final safety ceiling)
- Real-time metering and DSP telemetry:
  - Input/output level meters
  - Buffer and processing health metrics
  - Gate/VAD and gain-reduction indicators
- Preset save/load with migration support
- Latency calibration dialog for per device-pair compensation profiles

## DSP Chain

Runtime processing chain:

```text
Mic Input -> Pre-Filter (DC block + 80 Hz HP) -> Noise Gate -> Noise Suppression
-> De-Esser -> 10-Band EQ -> Compressor -> Limiter -> Output
```

Note: model latency labels describe suppressor and DSP behavior only. End-to-end round-trip latency also depends on device driver, buffer size, OS mixer path, and routing setup.

## Quick Start (Windows)

### Prerequisites

- Python 3.9+
- Rust 1.70+
- `maturin`

### Build and run from source

```bash
git clone https://github.com/FueledByRedBull/audio-forge.git
cd audio-forge

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]

.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m mic_eq
```

## Optional Runtime Assets

Create `models/` in repo root for runtime model discovery:

- `models/DeepFilterNet3_ll_onnx.tar.gz`
- `models/DeepFilterNet3_onnx.tar.gz` (optional if you only use LL)
- `models/silero_vad.onnx` (for VAD gate modes)

DeepFilter runtime library:

- `df.dll` in repo root (development)
- bundled next to `AudioForge.exe` for portable builds

You can also use environment variables:

- `DEEPFILTER_MODEL_PATH`
- `VAD_MODEL_PATH`

## Build Portable EXE

Use the provided packaging script:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Output:

- `dist\AudioForge\AudioForge.exe`
- `dist\AudioForge\df.dll` (if present)
- `dist\AudioForge\models\...`

Create archive:

```powershell
Compress-Archive -Path .\dist\AudioForge -DestinationPath .\AudioForge-win64-fresh.zip -CompressionLevel Optimal
```

## Testing

Rust tests:

```bash
cd rust-core
cargo test -p mic_eq_core --tests
```

Python tests:

```bash
cd ..
.\.venv\Scripts\python.exe -m pytest python/tests -v
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- RNNoise by Jean-Marc Valin
- DeepFilterNet by Hendrik Schroter and contributors
- Silero VAD contributors
