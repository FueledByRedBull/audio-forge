# AudioForge 1.13.0 (unreleased)

PR #66 targets this version. It has not been tagged or published; the latest
published release remains at 1.12.1.

## Changes

- Preserve voice character during Auto-EQ, add the Warm / Full Voice target,
  and keep microphone correction separate from editable tone.
- Compare the full processing chain on one recording before applying settings.
- Calibrate Auto Voice Setup with the actual auto-makeup settings, explain
  headroom constraints, and keep neutral EQ free of false abstention warnings.
- Reduce repeated suppression work, run compressor candidates in bounded parallel
  batches, and keep analysis responsive with stage and candidate progress.
- Recheck adjusted Voice Setup settings on the same verification take, with
  specific failure guidance and bounded retries that restore the previous sound.
- Recover VAD worker state safely across dropped analysis blocks.
- Keep VAD Assisted automatic thresholds consistent, reset VAD history across
  bypass/raw input gaps, and show the actual selected speech threshold.
- Recover speech detection after sustained near-full-scale audio and quieter
  clipped passages using speech-ending boundaries, without restarting
  processing or resetting the audio buffer and source clock.
- Add tray/background controls, a mute shortcut, unified processing modes,
  and safer preset, device and calibration handling.

See the [1.13.0 changelog](../CHANGELOG.md) for the complete change list.

## Validation limits

Recorded-audio, native DSP, UI and portable-package checks cover the local
development build. They do not qualify a future release artifact or establish
live hardware recovery across every microphone and route. Release validation
and publication still follow [RELEASING.md](../RELEASING.md).
