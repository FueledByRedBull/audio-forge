# MicForge

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

**Windows:**
```powershell
# Clone the repo
git clone https://github.com/FueledByRedBull/micforge.git
cd micforge

# Build Rust core
cd rust-core
maturin develop --release --features deepfilter

# Run
python -m mic_eq
```

**Linux/macOS:**
```bash
# Clone the repo
git clone https://github.com/FueledByRedBull/micforge.git
cd micforge

# Build Rust core
cd rust-core
maturin develop --release --features deepfilter

# Run
python -m mic_eq
```

### Models (Optional)

For DeepFilterNet support, download the model file:

```bash
# Create models directory
mkdir models

# Download DF3 LL model (recommended for real-time)
# Place in: models/DeepFilterNet3_ll_onnx.tar.gz
```

Or set `DEEPFILTER_MODEL_PATH` environment variable.

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

### v0.2 (Current)
- [x] RNNoise integration
- [x] DeepFilterNet integration (3 models)
- [x] 10-band parametric EQ
- [x] Noise gate, compressor, limiter
- [x] Real-time level meters
- [x] DSP performance metrics
- [x] Preset system

### v0.3 (Planned)
- [ ] Linux/macOS builds
- [ ] Installation script
- [ ] More EQ presets
- [ ] APU (audio processing unit) rebroadcast mode
- [ ] VST plugin format

### v0.4 (Future)
- [ ] WebRTC/NDI support
- [ ] Multiple input/output profiles
- [ ] Scene-based preset switching
- [ ] Cloud preset sharing

## Architecture

```
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → DeepFilter → EQ → Compressor → Limiter → Output
          ~                        ~           ~10-40ms    <1ms    ~       <1ms    ~
```

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
