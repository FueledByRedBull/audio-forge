# AudioForge Plan: De-Esser + Real Latency Calibration

Status: parked for later implementation  
Created: 2026-02-15

## Goals

1. Add a real-time de-esser to reduce harsh sibilance (`s`, `sh`, `t`) without dulling voice.
2. Add a latency calibration workflow that measures actual round-trip latency, not estimated latency.

## Why These Two

- De-esser is one of the biggest remaining voice-quality wins in the current chain.
- Real calibration improves user trust and setup quality across different devices/drivers.

---

## Feature A: De-Esser

### Target Behavior

- User can enable/disable de-esser.
- User can control:
  - Detection frequency band (default around `4kHz-9kHz`)
  - Threshold (dB)
  - Ratio/intensity
  - Attack/release
  - Max reduction cap
- Meter shows current de-ess gain reduction in dB.
- Processing stays real-time safe (no allocations/blocking in audio thread).

### DSP Design (MVP)

1. Build sidechain detector:
   - Band-pass filter for sibilance range.
   - Envelope follower from sidechain energy.
2. Compute dynamic gain reduction:
   - Above threshold -> apply downward gain.
   - Smooth with attack/release.
3. Apply reduction:
   - Start with wideband gain reduction (simpler, safer MVP).
   - Optional later: split-band de-esser.

### Implementation Steps

1. Rust DSP module
   - Add `rust-core/src/dsp/deesser.rs`.
   - Export in `rust-core/src/dsp/mod.rs`.
2. Processor wiring
   - Add de-esser state/config in `rust-core/src/audio/processor.rs`.
   - Place module after noise suppression and before EQ (initial placement).
3. PyO3 API
   - Add setters/getters on `PyAudioProcessor` for de-esser params and reduction meter.
4. Python UI
   - Add `python/mic_eq/ui/deesser_panel.py`.
   - Integrate panel into `python/mic_eq/ui/main_window.py`.
5. Presets/config
   - Extend preset schema in `python/mic_eq/config.py`.
   - Add migration `v1.6 -> v1.7` including safe defaults.
6. Tests
   - Unit tests in Rust for envelope/reduction behavior.
   - Python preset migration/load-save tests.
   - One end-to-end chain test that confirms no crash and stable output.

### Acceptance Criteria

- Audible reduction of harsh sibilance on test speech clips.
- No crackles/dropouts at normal buffer sizes.
- Added CPU overhead is small (target: < 0.3 ms DSP time increase on baseline machine).
- Presets save/load and migrate correctly.

### Risks

- Over-de-essing causes lisp artifacts.
- Wideband method can darken voice if aggressive.

### Risk Mitigation

- Conservative defaults.
- Max reduction clamp.
- Clear UI guidance and reduction meter.

---

## Feature B: Real Latency Calibration

### Target Behavior

- User can run a calibration wizard from UI.
- App emits a short test signal and records the return path.
- App estimates round-trip latency via correlation and computes output/input compensation.
- Measured value displayed as:
  - `Measured Round-Trip`
  - `Estimated One-Way`
  - `Applied Compensation`
- User can accept/re-run/reset calibration.

### Measurement Approach (MVP)

1. Generate deterministic probe signal (chirp or pseudo-random sequence).
2. Play probe through selected output while recording input.
3. Cross-correlate recorded data vs reference probe.
4. Peak offset -> round-trip latency samples -> ms.
5. Store compensation in config.

### Implementation Steps

1. Rust/Python boundary decision
   - Keep correlation analysis in Python first (faster iteration).
   - Revisit Rust move only if performance needs it.
2. Data capture
   - Reuse existing recording hooks in `rust-core/src/audio/processor.rs` where possible.
3. Analysis module
   - Add `python/mic_eq/analysis/latency_calibration.py`.
4. Wizard UI
   - Add `python/mic_eq/ui/latency_calibration_dialog.py`.
   - Hook into `main_window.py` settings/tools menu.
5. Config persistence
   - Store calibration profile per input/output device pair in `config.py`.
6. Display integration
   - Update latency labels to show measured vs estimated.
7. Tests
   - Synthetic signal tests for correlation accuracy.
   - Config persistence tests for per-device mapping.

### Acceptance Criteria

- Repeatability: repeated runs within small tolerance (target: +-2 ms).
- Correctly identifies large latency differences across different device setups.
- User can revert to estimated mode at any time.

### Risks

- Acoustic loopback environments are noisy.
- Users may not physically route output to input correctly.

### Risk Mitigation

- Wizard with clear setup instructions.
- Signal quality check + retry prompt.
- Fallback to current estimated latency when confidence is low.

---

## Suggested Release Sequence

1. `v1.7.0`: De-esser (full UI + preset + tests).
2. `v1.8.0`: Latency calibration wizard + measured latency display.

## Time Boxes (Realistic)

- De-esser MVP: 4-6 focused dev days.
- Latency calibration MVP: 3-5 focused dev days.
- Hardening + polish: 2-3 days each feature.

## Out of Scope for MVP

- Split-band multiband de-esser.
- Auto de-esser parameter learning.
- Full hardware loopback auto-detection without user setup.

---

## Quick Restart Checklist (When You Revisit)

1. Implement `deesser.rs` and wire into processor chain.
2. Expose PyO3 controls + reduction meter.
3. Add UI panel + preset migration.
4. Build latency analysis module + wizard.
5. Persist per-device calibration and update latency display.
6. Add end-to-end smoke tests before release.
