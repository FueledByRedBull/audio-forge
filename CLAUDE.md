# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MicEq is a low-latency microphone audio processor inspired by SteelSeries GG Sonar's ClearCast AI. It provides real-time noise suppression and equalization for voice communication (Discord, etc.).

**Processing Chain:**
```
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → RNNoise → 10-Band EQ → Compressor → Hard Limiter → VB Audio Cable Output
```

## Architecture

- **Rust Core** (`rust-core/`): High-performance DSP engine with PyO3 bindings
  - IIR biquad filters for 10-band parametric EQ
  - Noise gate with IIR envelope follower
  - RNNoise integration via `nnnoiseless` crate
  - Compressor with soft-knee and makeup gain
  - Hard limiter (brick-wall ceiling)
  - Lock-free audio I/O via `cpal` (WASAPI on Windows)
  - Real-time DSP performance metrics (lock-free atomics)

- **Python GUI** (`python/`): PyQt6 interface for real-time control
  - 10-band EQ controls (frequency, gain, Q per band)
  - Noise gate threshold/attack/release
  - RNNoise enable/disable
  - Compressor controls (threshold, ratio, attack, release, makeup gain)
  - Hard limiter controls (ceiling, release)
  - OBS-style visual level meters (input/output)
  - DSP performance display (latency, DSP time, buffer health)
  - Input/output device selection (VB Audio Cable routing)
  - Preset save/load system

## Build Commands

```powershell
# Build Rust core with Python bindings
cd rust-core
maturin develop --release

# Run the application
python -m mic_eq

# Run Rust tests
cargo test

# Build release wheel
maturin build --release
```

## Key Dependencies

- `nnnoiseless`: Pure Rust RNNoise port for ML-based noise suppression
- `cpal`: Cross-platform audio I/O
- `pyo3`: Rust-Python bindings
- `PyQt6`: GUI framework

## Reference Implementation

The Spectral Workbench project (`C:\Users\anchatzigiannakis\Documents\Projects\Experiments\Spectral Workbench`) contains reference implementations for:
- FIR filter design and processing
- Noise gate with IIR envelope follower
- Lock-free ring buffers
- PyO3 binding patterns
- PyQt6 real-time audio GUI

## Performance Targets

- Total latency: <30ms (Gate: <1ms, RNNoise: 10ms, EQ: <1ms, VB Cable: ~10ms)
- CPU usage: <10%
- Sample rate: 48kHz internal

## Implementation Plan

### Phase 1: Project Setup
- [x] Create workspace Cargo.toml
- [x] Create rust-core/Cargo.toml with dependencies:
  - `cpal` 0.15 (audio I/O)
  - `nnnoiseless` (RNNoise)
  - `pyo3` 0.20 + `numpy` 0.20 (Python bindings)
  - `ringbuf` 0.3 (lock-free buffers)
- [x] Create pyproject.toml with maturin build config
- [x] Set up Python package structure

### Phase 2: Rust Core - IIR Biquad Filters
- [x] Implement `Biquad` struct with coefficients (b0, b1, b2, a1, a2) and state (z1, z2)
- [x] Implement filter types: lowshelf, highshelf, peaking (parametric)
- [x] Create `ParametricEQ` with 10 bands:
  - Band 1: Low shelf (80 Hz)
  - Bands 2-9: Peaking (160, 320, 640, 1.2k, 2.5k, 5k, 8k, 12k Hz)
  - Band 10: High shelf (16 kHz)
- [x] Each band: frequency, gain (dB), Q factor configurable

### Phase 3: Rust Core - Noise Gate
- [x] Port IIR envelope follower from Spectral Workbench
- [x] Parameters: threshold (dB), attack (ms), release (ms)
- [x] Smooth gain transitions to avoid clicks

### Phase 4: Rust Core - RNNoise Integration
- [x] Integrate `nnnoiseless` crate
- [x] **Critical**: Implement 480-sample frame buffering (see below)
- [x] Add bypass toggle

#### RNNoise Frame Buffering (Critical)

RNNoise requires **exactly 480 samples** per call (10ms at 48kHz). However, cpal delivers variable chunk sizes (e.g., 432, 1024). Solution:

```
┌─────────────────────────────────────────────────────────────┐
│                    AudioProcessor                           │
│                                                             │
│  cpal input ──► [RNNoise Ring Buffer] ──► RNNoise ──► EQ   │
│  (variable)      (accumulates until     (exact 480)        │
│                   ≥480 samples)                             │
└─────────────────────────────────────────────────────────────┘
```

#### Considerations
User Experience: In your GUI (Phase 7), ensure the "Output Device" dropdown explicitly filters for or highlights the VB Audio Cable driver if found. If the user selects their actual speakers as the output, they will hear themselves (monitoring), but Discord won't hear them. You might want a "Monitor" toggle (output to speakers) and a "Main Output" (output to VB Cable).

If you implement a "Bypass" feature, do not drop the RNNoise struct or reset it constantly. Just skip calling the process function. If you reset the state, the model takes ~200ms to "converge" on the background noise level again, causing a glitch.

Instead of failing if the default isn't 48kHz, you should ask cpal if the device supports 48kHz and request that specific configuration.

Use Direct Form II Transposed as biquad implementation form

**Implementation:**
1. **Input**: Push cpal samples into intermediate ring buffer
2. **Process**: While buffer ≥ 480 samples:
   - Pop exactly 480 samples
   - Run RNNoise on the 480-sample chunk
   - Push processed samples to output buffer
3. **Output**: Drain output buffer to cpal

This adds minimal latency (worst case: 479 samples = ~10ms) but ensures RNNoise always receives valid frame sizes.

### Phase 5: Rust Core - Audio Processor
- [x] Create unified `AudioProcessor` struct combining:
  - Noise Gate → RNNoise → 10-Band EQ
- [x] Implement lock-free ring buffer for audio thread safety
- [x] Audio I/O via cpal:
  - Input: System microphone (with resampling to 48kHz if needed)
  - Output: VB Audio Cable virtual device
- [x] Device enumeration and selection

### Phase 6: Python Bindings (PyO3)
- [x] Expose `AudioProcessor` to Python
- [x] Expose device enumeration
- [x] Expose real-time parameter updates:
  - Gate: threshold, attack, release, enable
  - RNNoise: enable
  - EQ: per-band gain, frequency, Q, enable

### Phase 7: PyQt6 GUI
- [x] Main window with device selection (input/output dropdowns)
- [x] Noise gate panel: threshold slider, attack/release knobs, enable checkbox
- [x] RNNoise panel: enable checkbox, status indicator
- [x] 10-band EQ panel:
  - Vertical sliders for gain (-12 to +12 dB)
  - Optional: frequency/Q adjustment per band
  - Per-band bypass
- [x] Master bypass toggle
- [x] Start/Stop processing button
- [x] OBS-style level meters (input/output) with peak hold and clipping indicators
- [x] DSP performance metrics display (latency, DSP time, buffer health)

### Phase 8: Build & Polish
- [x] Create build.ps1 for Windows
- [x] Test full pipeline with VB Audio Cable
- [x] Profile and optimize for <30ms latency
- [x] Add preset save/load (JSON config)

## Latency Analysis

The total processing latency is well within the <30ms target:

| Stage | Latency | Notes |
|-------|---------|-------|
| Noise Gate | <0.1ms | IIR envelope follower - instantaneous |
| RNNoise | 10ms (max) | 480-sample frame buffering at 48kHz |
| 10-Band EQ | <0.1ms | IIR biquad filters - instantaneous |
| Ring Buffer | ~0.1ms | Lock-free, negligible overhead |
| **DSP Total** | **~10ms** | Dominated by RNNoise frame buffering |
| VB Audio Cable | ~10-15ms | Depends on buffer settings |
| **End-to-End** | **~20-25ms** | Well under 30ms target |

**Notes:**
- RNNoise latency varies from 0-10ms depending on when samples arrive relative to frame boundaries
- VB Audio Cable latency can be adjusted via its control panel (trade-off: lower latency = higher CPU)
- Processing thread polls every 100μs when idle (negligible CPU impact)

## Completed Additions

- [x] **Compressor with Hard Limiter** - Soft-knee compressor + brick-wall limiter
- [x] **Visual Level Meters** - OBS-style meters with green/yellow/red gradients, peak hold, clipping indicators
- [x] **DSP Performance Metrics** - Real-time display of DSP time, latency, and buffer health
- [x] **Gain Reduction Meter** - Shows compressor activity

## Future Plans

### Plan 1: Noise Suppression "Strength" Slider (Next)

Add wet/dry mix control for RNNoise to allow blending between original and processed signal.

**Formula:** `output = (strength * processed) + ((1.0 - strength) * original)`

**Files to modify:**
- `rust-core/src/dsp/rnnoise.rs` - Add strength parameter and mixing logic
- `rust-core/src/audio/processor.rs` - Add control methods + PyO3 bindings
- `python/mic_eq/ui/main_window.py` - Add strength slider to RNNoise panel
- `python/mic_eq/config.py` - Add strength to preset system

**Latency impact:** None

---

### Plan 2: DeepFilterNet Migration (Future/Optional)

Replace RNNoise with DeepFilterNet v3 for improved noise suppression quality.

**Trade-offs:**
| Aspect | RNNoise (Current) | DeepFilterNet |
|--------|-------------------|---------------|
| Latency | ~10ms | ~30-40ms |
| Quality | Good | Better |
| CPU | ~0.1ms | ~2-5ms |

**Prerequisites:**
1. Research `df` crate API and compatibility (https://github.com/Rikorose/DeepFilterNet)
2. Test DeepFilterNet latency on target hardware
3. Consider dual-mode toggle (RNNoise vs DeepFilter)

**Decision point:** Only proceed if RNNoise quality is insufficient and ~100ms total latency is acceptable.

**Implementation approach:**
- Create `rust-core/src/dsp/deepfilter.rs` with same interface as rnnoise.rs
- Add "AI Model" dropdown in GUI: RNNoise (Low Latency) / DeepFilter (High Quality)
- Strength slider works with both models
