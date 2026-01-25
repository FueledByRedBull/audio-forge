# AudioForge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

> Low-latency microphone audio processor with AI noise suppression and 10-band parametric EQ.

Forge your sound in real-time. Built for streamers, content creators, and anyone who wants professional microphone processing without the complexity.

## Features

- **AI Noise Suppression** - Three models to choose from:
  - RNNoise (~10ms latency)
  - DeepFilterNet LL (~10ms latency, better quality)
  - DeepFilterNet Standard (~40ms latency, best quality)

- **10-Band Parametric EQ** - Shaping with presets:
  - Voice, Bass Cut, Presence, Warm & Clear
  - Per-band frequency, gain, and Q control

- **Full DSP Chain** - All in one place:
  - Noise Gate with IIR envelope follower
  - AI Noise Suppression (selectable model)
  - 10-Band Parametric EQ
  - Compressor with soft-knee and makeup gain
  - Hard Limiter for clipping prevention

- **Real-Time Monitoring** - See what's happening:
  - OBS-style visual level meters (input/output)
  - DSP performance metrics (latency, processing time)
  - Buffer health monitoring
  - Compressor gain reduction meter

## Screenshots

*Coming soon*

## Quick Start

### Prerequisites

- Python 3.9+
- Rust 1.70+ (with maturin)
- Audio interface (VB Audio Cable recommended for routing)

### Installation

**1. Clone the repo:**
```bash
git clone https://github.com/FueledByRedBull/audio-forge.git
cd audio-forge
```

**2. Build the Rust core:**

| Feature | Flag | Description | Required for |
|---------|------|-------------|--------------|
| RNNoise | *(default)* | Fast noise suppression | ✅ Basic usage |
| VAD | `--features vad` | Silero VAD-assisted gate | Smart gate modes |
| DeepFilterNet | `--features deepfilter` | Advanced noise suppression | DF models |

**Build commands:**

```bash
# Minimal build (RNNoise only)
cd rust-core
maturin develop --release

# Full feature build (VAD + DeepFilterNet)
maturin develop --release --features vad,deepfilter
```

**3. Run the application:**
```bash
python -m mic_eq
```

---

### Optional: DeepFilterNet Support

DeepFilterNet requires both the **C library** and **model file**.

#### Step 1: Build the C library (df.dll)

```bash
# Clone DeepFilterNet
git clone https://github.com/Rikorose/DeepFilterNet.git
cd DeepFilterNet

# Build the C library
cargo build --release

# Copy the DLL to AudioForge
cp target/release/df.dll ../audio-forge/
```

#### Step 2: Download the model

```bash
# Create models directory in AudioForge
mkdir audio-forge/models

# Download DeepFilterNet3 LL model (recommended for real-time)
# Place in: audio-forge/models/DeepFilterNet3_ll_onnx.tar.gz
```

Or set `DEEPFILTER_MODEL_PATH` environment variable to point to your model file.

---

### Optional: Silero VAD (for VAD-assisted gate)

VAD-assisted gate modes require the Silero VAD model:

```bash
# Create models directory in AudioForge
mkdir audio-forge/models

# Download Silero VAD model
# Visit: https://github.com/snakers4/silero-vad/tree/master/files
# Download: silero_vad.onnx (v4 or v5)
# Place in: audio-forge/models/silero_vad.onnx
```

Or set `VAD_MODEL_PATH` environment variable.

## Usage

1. **Select Audio Devices**
   - Input: Your microphone
   - Output: VB Audio Cable (for routing to Discord/OBS)

2. **Choose AI Model**
   - RNNoise - Fastest, good quality
   - DeepFilterNet LL - Best for real-time
   - DeepFilterNet - Best quality, higher latency

3. **Adjust Settings**
   - Use presets or tweak individual bands
   - Adjust gate threshold, compressor, limiter as needed

4. **Start Processing**
   - Click "Start Processing"
   - Monitor levels in real-time

## Roadmap

### v1.5.0 (Current Release)
- [x] RNNoise integration
- [x] Silero VAD-assisted noise gate
- [x] DeepFilterNet integration (experimental, C FFI)
- [x] 10-band parametric EQ
- [x] Noise gate, compressor, limiter
- [x] Real-time level meters
- [x] DSP performance metrics
- [x] Preset system
- [x] Scrollable panels
- [x] Adaptive release compressor

### v1.6.0 (Planned)
- [ ] Linux/macOS builds
- [ ] Installation script
- [ ] More EQ presets
- [ ] VST plugin format

### v2.0.0 (Future)
- [ ] WebRTC/NDI support
- [ ] Multiple input/output profiles
- [ ] Scene-based preset switching
- [ ] Cloud preset sharing

## Architecture

**Processing Chain:**
```
Mic Input → Pre-Filter (DC Block + 80Hz HP) → Noise Gate → AI Noise (RNNoise/DeepFilter) → 10-Band EQ → Compressor → Limiter → Output
```

**Note:** DeepFilterNet is experimental. RNNoise is recommended for production use.

**Target Latency:** <30ms total (DeepFilterNet LL), <50ms (Standard)

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Inspired by SteelSeries GG Sonar's ClearCast AI
- RNNoise by Jean-Marc Valin
- DeepFilterNet by Hendrik Schröter
- Spectral Workbench (reference implementation)

---

**Made with :heart: by FueledByRedBull**
