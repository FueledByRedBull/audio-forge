# Changelog

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
