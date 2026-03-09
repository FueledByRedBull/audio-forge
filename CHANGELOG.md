# Changelog

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
