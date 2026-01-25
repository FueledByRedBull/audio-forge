# Roadmap: AudioForge

## Overview

AudioForge is a low-latency real-time microphone audio processor inspired by SteelSeries GG Sonar's ClearCast AI. It provides real-time noise suppression and equalization for voice communication (Discord, etc.).

**Processing Chain:**
```
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → RNNoise/DeepFilterNet → 10-Band EQ → Compressor → Hard Limiter → VB Audio Cable Output
```

## Milestones

- ✅ **v1.0.0 Production Release** — Phases 1-7 (shipped 2026-01-22)
  - Full details: `.planning/milestones/v1.0.0-ROADMAP.md`
  - Requirements: `.planning/milestones/v1.0.0-REQUIREMENTS.md`
- ✅ **v1.1.0 Enhancement** — Phases 8-9 (shipped 2026-01-23)
  - Full details: `.planning/milestones/v1.1.0-ROADMAP.md`
  - Requirements: `.planning/milestones/v1.1.0-REQUIREMENTS.md`
- ✅ **v1.1.1 Enhancement** — Phase 10 (completed 2026-01-24)
  - DeepFilterNet integration via C FFI (experimental due to upstream library bug)
- ✅ **v1.2.0 Advanced DSP** — Phases 11-13 (completed 2026-01-24)
  - Silero VAD Auto-Gate, Adaptive Release Compressor, Auto Makeup Gain (LUFS-based)
- ✅ **v1.3.0 GUI Polish** — Phase 14 (completed 2026-01-24)
  - Spacing, layout, compressor panel reorganization, visual hierarchy improvements
- ✅ **v1.4.0 Scrollable Panels** — Phase 15 (completed 2026-01-24)
  - Scroll areas for panels with overflow, increased minimum window height
- ✅ **v1.5.0 VAD Runtime Fix** — Phase 16 (shipped 2026-01-24)
  - Fixed ONNX Runtime API integration for Silero VAD

## Phases

<details>
<summary>✅ v1.0.0 Production Release (Phases 1-7) - SHIPPED 2026-01-22</summary>

**Phase 1: GUI Fixes** (1/1 plans) — Completed 2026-01-22
- Fixed PyQt6 layout issues causing overlapping/truncated controls

**Phase 2: User Experience** (1/1 plans) — Completed 2026-01-22
- Device and preset persistence across sessions

**Phase 3: EQ Enhancement** (2/2 plans) — Completed 2026-01-22
- Semi-parametric EQ with visual feedback and macro presets

**Phase 4: Robustness** (2/2 plans) — Completed 2026-01-22
- Stability fixes for edge cases and error handling

**Phase 5: Performance** (2/2 plans) — Completed 2026-01-22
- Lock-free audio thread optimization

**Phase 6: Code Quality** (2/2 plans) — Completed 2026-01-22
- Input validation and security hardening

**Phase 7: Validation** (1/1 plans) — Completed 2026-01-22
- Stress testing and hotswap reliability

</details>

<details>
<summary>✅ v1.1.0 Enhancement (Phases 8-9) - SHIPPED 2026-01-23</summary>

**Phase 8: RNNoise Enhancement** (4/4 plans) — Completed 2026-01-23
RNNoise wet/dry mix control with 0-100% strength slider, 15ms EMA smoothing, preset version migration

**Phase 9: Metrics Enhancement** — Pre-existing
Dropped samples counter display (already fully implemented)

</details>

<details>
<summary>✅ Phase 10: DeepFilterNet Integration - COMPLETED 2026-01-24</summary>

**Goal:** Integrate DeepFilterNet Low Latency (LL) variant as alternative noise suppression model

**Plans:**
- [x] 10-01-PLAN.md — Add git dependency for DeepFilterNet to Cargo.toml
- [x] 10-02-PLAN.md — Implement DeepFilterNet processor wrapper with libDF
- [x] 10-03-PLAN.md — Update UI latency display and model names for LL variant
- [x] 10-04-PLAN.md — Testing and validation

**Implementation:**
C FFI approach using runtime dynamic loading via libloading. Resolved `Rc/Send` issues with Rust crate by bypassing it entirely. Created 579-line FFI wrapper with graceful passthrough fallback.

**Status:** Experimental - C library loads but crashes during initialization due to upstream tract-core bug. RNNoise remains recommended for production use.

**Details:**
- Created `rust-core/src/dsp/deepfilter_ffi.rs` (579 lines)
- Runtime dynamic loading via `libloading` crate
- Send-safe wrapper resolves Rc/Send issues
- All 8 unit tests pass
- Application works in passthrough mode when C library unavailable

</details>

### ✅ v1.2.0 Advanced DSP - COMPLETED 2026-01-24

**Milestone Goal:** Implement three professional DSP features: Auto Makeup Gain (LUFS-based), Adaptive Release compressor, and Silero VAD Auto-Gate completion.

#### ✅ Phase 11: Silero VAD Integration
**Goal**: Complete VAD auto-gate integration with GUI controls and preset persistence
**Depends on**: Phase 10
**Requirements**: VAD-01, VAD-02, VAD-03, VAD-04, VAD-05, VAD-06, VAD-07, VAD-08, VAD-09, VAD-10, VAD-11
**Success Criteria** (what must be TRUE):
  1. User can select gate mode (Threshold-only / VAD-Assisted / VAD-Only) from dropdown
  2. User can adjust VAD probability threshold (0.3-0.8) and gate hold time (0-500ms)
  3. VAD confidence meter displays speech probability in real-time (0.0-1.0)
  4. Gate opens smoothly when threshold exceeded OR speech detected (VAD-assisted mode)
  5. VAD settings (threshold, mode, hold time) persist across preset save/load

**Plans:**
- [x] 11-01-PLAN.md — VAD gate mode controls and metering (PyO3 bindings, three-mode logic, VAD probability atomic)
- [x] 11-02-PLAN.md — Gate panel GUI with mode selection, VAD controls, and confidence meter
- [x] 11-03-PLAN.md — Preset system integration for VAD settings with backward compatibility
- [x] 11-04-PLAN.md — Wire VAD confidence meter to main window timer (gap closure)
- [x] 11-05-PLAN.md — Instantiate VadAutoGate in AudioProcessor and wire to NoiseGate (gap closure)

**Status:** Complete — All 5 plans executed including gap closure. VAD integration fully functional with real-time confidence meter and speech detection.

#### ✅ Phase 12: Adaptive Release Compressor
**Goal**: Implement adaptive release with overage detection and dynamic scaling
**Depends on**: Phase 11
**Requirements**: ADP-01, ADP-02, ADP-03, ADP-04, ADP-05, ADP-06, ADP-07, ADP-08
**Success Criteria** (what must be TRUE):
  1. User can set base release time (20-200ms) and toggle adaptive release mode
  2. Release time dynamically scales from 50ms to 400ms based on overage amount
  3. Release changes smoothly with 100ms hysteresis to prevent rapid toggling
  4. Current release time displays in real-time on compressor panel
  5. Adaptive release settings persist across preset save/load

**Plans:**
- [x] 12-01-PLAN.md — Compressor adaptive release logic with overage detection
- [x] 12-02-PLAN.md — PyO3 bindings for base release and adaptive mode controls
- [x] 12-03-PLAN.md — Compressor GUI with adaptive controls and real-time release meter

**Status:** Complete — Adaptive release scales from 50ms to 400ms based on sustained overage with 100ms hysteresis smoothing.

#### ✅ Phase 13: Auto Makeup Gain (LUFS-based)
**Goal**: Implement EBU R128 loudness measurement with automatic makeup gain
**Depends on**: Phase 12
**Requirements**: LUF-01, LUF-02, LUF-03, LUF-04, LUF-05, LUF-06, LUF-07, LUF-08, LUF-09, LUF-10
**Success Criteria** (what must be TRUE):
  1. User can set target LUFS (-24 to -12, default -18) and toggle auto/manual mode
  2. System measures current loudness using EBU R128 algorithm with momentary integration (400ms)
  3. Makeup gain is calculated as (target_lufs - current_lufs) with 200ms smoothing
  4. Gain clamped to 0-12 dB range to prevent excessive amplification
  5. Auto makeup settings persist across preset save/load
  6. User can see current LUFS and makeup gain values in real-time

**Plans:**
- [x] 13-01-PLAN.md — EBU R128 loudness meter with ebur128 crate integration
- [x] 13-02-PLAN.md — Auto makeup gain calculation and smoothing in compressor
- [x] 13-03-PLAN.md — PyO3 bindings, GUI controls, and preset integration (gap closure)

**Status:** Complete — EBU R128 loudness measurement with auto makeup gain. Uses momentary integration (400ms) for real-time control, 200ms smoothing for gain transitions. Full PyO3 bindings and GUI integration with preset persistence.

#### ✅ Phase 14: GUI Polish
**Goal**: Improve spacing, layout, and visual hierarchy to reduce congestion and improve usability
**Depends on**: Phase 13
**Plans:** 4 plans

**Issues Identified:**
- Compressor panel is overcrowded with too many controls stacked (threshold, ratio, attack, release, adaptive, auto makeup, gain reduction, makeup gain)
- Noise Gate section has VAD controls tightly packed with insufficient spacing
- Overall vertical spacing inconsistent - some sections cramped while others have room
- Poor visual hierarchy - no clear distinction between primary and secondary controls
- Small text size makes labels and values difficult to read
- Control grouping issues - related controls not positioned together efficiently

**Plans:**
- [x] 14-01-PLAN.md — Create shared spacing and typography constants
- [x] 14-02-PLAN.md — Reorganize compressor panel with two-column layout
- [x] 14-03-PLAN.md — Apply consistent spacing and typography to gate panel
- [x] 14-04-PLAN.md — Apply spacing and typography to EQ panel and main window

**Status:** Complete — Created shared spacing and typography constants. Reorganized compressor panel to two-column layout with visual grouping for advanced settings. Applied consistent spacing (SPACING_NORMAL=8px) and typography (11pt/10pt/9pt) across all UI panels. Visual consistency achieved across entire application.

#### Phase 15: Scrollable Panels
**Goal**: Add scroll areas for panels with overflow content and increase minimum window height
**Depends on**: Phase 14
**Plans:** 3 plans

**Issues Identified:**
- Gate panel has many controls (gate mode, threshold, attack, release, VAD threshold, hold time, confidence meter)
- Compressor panel also has many controls despite two-column layout
- Current minimum window height (750px) may be insufficient for all controls to be visible
- Users need ability to scroll panels independently when content overflows

**Plans:**
- [x] 15-01-PLAN.md — Wrap gate panel in scroll area
- [x] 15-02-PLAN.md — Wrap compressor panel in scroll area
- [x] 15-03-PLAN.md — Increase minimum window height

**Status:** Complete — Gate and compressor panels wrapped in QScrollArea with horizontal scrollbar disabled (ScrollBarAlwaysOff), widgetResizable(true) for proper sizing. Minimum window height increased from 750px to 850px. Scroll areas use NoFrame shape for cleaner visual integration.

#### Phase 16: Fix VAD ONNX Runtime API Integration
**Goal**: Fix VAD ONNX Runtime API integration issues to make Silero VAD functional
**Depends on**: Phase 15

**Issues to Fix:**
1. ONNX Runtime API mismatch (ort 2.0 API vs older API)
2. Wrong model input shape (needs [1, 512] not [1, 1, 512])
3. Missing LSTM hidden states (h/c state must persist between calls)
4. Wrong tensor extraction method (try_extract_tensor vs extract_scalar)

**Plans:**
- [x] 16-01-PLAN.md — Replace vad.rs with fixed ONNX API implementation
- [x] 16-02-PLAN.md — Update Cargo.toml dependencies for ort
- [x] 16-03-PLAN.md — Add Silero VAD model download instructions
- [x] 16-04-PLAN.md — Test VAD functionality end-to-end

**Status:** ✅ SHIPPED — Fixed VAD ONNX Runtime API integration with ort 2.0. Updated ndarray to 0.17, fixed input shapes [1,512], added LSTM state management with combined state [2,1,128] (not separate h/c), proper tensor extraction with try_extract_tensor(). All 4 plans executed and verified. **Runtime fix applied 2026-01-24**: Changed from separate h/c inputs to combined "state" input to match actual Silero VAD model.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16

| Phase             | Milestone | Plans Complete | Status      | Completed  |
|------------------|-----------|----------------|-------------|-----------|
| 1. GUI Fixes       | v1.0.0    | 1/1            | Complete    | 2026-01-22 |
| 2. User Experience  | v1.0.0    | 1/1            | Complete    | 2026-01-22 |
| 3. EQ Enhancement   | v1.0.0    | 2/2            | Complete    | 2026-01-22 |
| 4. Robustness      | v1.0.0    | 2/2            | Complete    | 2026-01-22 |
| 5. Performance     | v1.0.0    | 2/2            | Complete    | 2026-01-22 |
| 6. Code Quality    | v1.0.0    | 2/2            | Complete    | 2026-01-22 |
| 7. Validation      | v1.0.0    | 1/1            | Complete    | 2026-01-22 |
| 8. RNNoise Enhancement | v1.1.0 | 4/4            | Complete | 2026-01-23 |
| 9. Metrics Enhancement | v1.1.0 | N/A            | Complete (pre-existing) | - |
| 10. DeepFilterNet Integration | v1.1.1 | 4/4            | Complete (experimental) | 2026-01-24 |
| 11. Silero VAD Integration | v1.2.0 | 5/5            | Complete | 2026-01-24 |
| 12. Adaptive Release Compressor | v1.2.0 | 3/3            | Complete | 2026-01-24 |
| 13. Auto Makeup Gain (LUFS-based) | v1.2.0 | 3/3            | Complete | 2026-01-24 |
| 14. GUI Polish | v1.3.0 | 4/4            | Complete | 2026-01-24 |
| 15. Scrollable Panels | v1.4.0 | 3/3            | Complete | 2026-01-24 |
| 16. Fix VAD ONNX Runtime API Integration | v1.5.0 | 4/4            | Complete | 2026-01-24 |

**Overall Progress: 47/47 plans complete (100%)**
