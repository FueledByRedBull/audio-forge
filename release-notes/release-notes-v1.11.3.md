# AudioForge v1.11.3

This patch fixes streaming audio state and timing defects found during a
whole-repository audit, strengthens release qualification, and removes unused
internal surface.

## Audio correctness

- New and invalid multichannel input settings now use phase-safe mono instead
  of arithmetic channel averaging. Saved explicit channel choices are kept.
- Output drift correction now preserves interpolation phase and sample history
  across callback boundaries, avoiding repeated block-edge timing errors.
- Filter resets clear stored signal history, and adaptive cleanup now uses
  elapsed audio time and restores its actual high-pass state after bypass,
  Raw Monitor, recording, or cleanup-mode changes.
- RNNoise starts at the requested strength instead of briefly starting fully
  wet. The final true-peak limiter follows the configured release time in both
  realtime and offline paths.
- VAD processing drains complete backlogs, separates worker and display state,
  publishes freshness correctly, and provides the full configured hold time.
- The primary limiter lookahead is now 0.5 ms after corrected paired,
  delay-flushed evidence passed every predefined quality and safety gate.

## Release and maintenance hardening

- Exact-artifact health checks explicitly select Standard DeepFilter and
  require 1,440-sample latency, successful inference frames, and finite,
  non-silent output.
- PyInstaller failures now propagate their exit code instead of allowing a
  misleading successful build step.
- A native production-output regression covers product resampling, drift
  retiming, and final true-peak limiter recovery together.
- Unused Rust and Python APIs, package-root evaluation exports, and the retired
  local TODO-index generator were removed. Evaluation tools import native hooks
  directly from `mic_eq.mic_eq_core`.

## Validation and compatibility

All source-bound evaluators were rerun. Apart from the corrected limiter result,
retained and rejected decisions are unchanged. Existing presets remain
compatible and saved explicit input-channel choices are unchanged. The removed
package-root names were internal evaluation hooks; those tools now import the
native module directly.

Exact package sizes, hashes, and file identities are published only in the
generated checksum, metadata, and manifest sidecars for the qualified release
artifact.
