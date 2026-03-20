# Changelog

## v1.7.12 - 2026-03-20

- Completed DSP redesign updates with canonical compressor knee/detector semantics, sample-rate-aware limiter lookahead latency reporting, split-band de-esser recombination, and percentile-based VAD floor tracking.
- Fixed limiter lookahead peak planning to include the active output decision window and preserved gate gain smoothing during VAD force-close transitions.
- Upgraded Auto-EQ to a two-stage dense-grid optimizer (gain-only then gain+Q) with bounded Q regularization and gain-ripple penalties, and made Auto-EQ calibration follow the user-selected UI input/output devices.
- Added/updated regression coverage across gate/compressor/limiter/VAD/EQ/resampler paths and refreshed release packaging hooks for SciPy hidden-import handling.

## v1.7.11 - 2026-03-19

- Preferred native `48 kHz` input configs when available and made required input/output resampler setup fail fast at startup instead of falling through to wrong-rate processing.
- Reworked the real-time reliability path with proactive input backlog shedding, gentler output catch-up, shorter underrun tails, and deferred gate/suppressor control updates so hot audio blocks stop depending on `try_lock()` fallbacks.
- Standardized calibration and analysis sample-rate handling around the runtime processor rate and surfaced new backlog, clip, and resampler diagnostics in the main window.

## v1.7.10 - 2026-03-15

- Fixed the packaged Windows build startup failure caused by excluding and pruning SciPy's `_highspy` runtime payload.
- Restored the required SciPy optimize module in `AudioForge.spec` and stopped the bundle-pruning step from deleting it.
- Rebuilt the portable EXE and verified that `AudioForge.exe` starts normally from `dist/AudioForge`.

## v1.7.9 - 2026-03-13

- Fixed the main-window dark-theme regression by removing the broad forced-light styling and limiting custom styling to explicit action buttons and health chips.
- Rebalanced the splitter layout for the tabbed control column and EQ pane, with clamped persisted sizes and wider pane floors so labels and EQ controls stop clipping on 1366-wide displays.
- Polished action-row spacing and tab-page margins to make the reworked `Cleanup` and `Dynamics` views read cleanly without changing DSP behavior.

## v1.7.8 - 2026-03-13

- Removed steady-state VAD buffer draining and per-window scratch allocation by switching Silero VAD to reusable cursor-based buffers.
- Hard-gated audio-adjacent debug logging in the VAD, gate, and processor paths so release builds stop printing from those hot sections.
- Pruned the unused bundled `scipy.optimize._highspy` payload to trim the Windows package further without dropping features.

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
