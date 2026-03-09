# Changelog

## v1.7.7 - 2026-03-09

- Hardened the headless self-test by using expected playback windows, a normalized correlation score, and a wider retry delay ladder while keeping the `0.25` confidence threshold unchanged.
- Removed steady-state `try_lock()` use from the downstream DSP chain and raw-recording tap, and mirrored suppressor latency into atomics for lock-free reporting.
- Deleted the unused legacy `RecordingWorker` path and tightened the packaged build flow around the checked-in spec and bundle-pruning helper.

## v1.7.6 - 2026-03-09

- Trimmed the packaged Windows app further by dropping the duplicated runtime icon payload and pruning unused Qt translations/PDF binaries after build.
- Prefer bundled `_internal/models` and `_internal/df.dll` more aggressively in the frozen launcher/runtime path to keep packaged asset lookup canonical.
- Kept the full-feature release payload intact: RNNoise, VAD, DeepFilter LL, and DeepFilter Standard remain bundled.

## v1.7.5 - 2026-03-09

- Hardened DeepFilter runtime loading and surfaced backend availability/error state in the UI.
- Added runtime diagnostics for suppressor non-finite resets, stream recovery, underruns, and dropped samples.
- Moved calibration and latency-calibration processor access back to the main Qt thread to avoid `unsendable` PyO3 cross-thread calls.
- Tuned watchdog recovery suppression for recording/calibration workflows.
- Switched the Windows package build to the checked-in `AudioForge.spec` and removed the redundant packaged `python/mic_eq` source tree.

## v1.7.4 - 2026-03-09

- Fixed CPAL stream setup for `f32`, `i16`, and `u16` devices so startup no longer depends on float-native hardware.
- Added output-side resampling so non-48 kHz playback devices receive correctly timed audio.
- Removed steady-state per-frame DeepFilterNet allocations from the realtime processing path.
- Trimmed packaged release size by relying on bundled internal DeepFilter assets instead of duplicating them next to the executable.

## v1.7.3 - 2026-02-28

- Fixed suppressor non-finite output poisoning the DSP/output pipeline after extended silence.
- Added runtime suppressor output sanitization and automatic suppressor reinitialization on detection.
- Reworked stop/start suppressor reset path to rebuild suppressor state for reliable in-app recovery.

## v1.7.2 - 2026-02-25

- Guarded biquad Q from zero to prevent NaNs.

## v1.7.1 - 2026-02-25

Note: releases were behind master; this rollup includes changes since v1.5.3.

- Added callback-based stream supervisor with auto-recovery and backoff.
- Added headless health check and self-test tools.
- Improved DeepFilterNet auto-enable when assets are present.
- Refactored downstream DSP chain to remove duplicated logic.
- Split output mute flag from recording state.
- Switched recording level meter to a sliding RMS window.
- Added/updated tests for preset VAD persistence.
- Misc: health check warmup handling, clearer docs, packaging notes.
