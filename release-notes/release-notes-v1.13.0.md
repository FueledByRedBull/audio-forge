# AudioForge 1.13.0

AudioForge 1.13.0 improves voice calibration, full-chain comparisons, peak
protection, and everyday device and preset handling.

## Changes

- Improve short-burst true-peak limiting and preview safety. Final peak protection
  adds 6.25 ms at 48 kHz; ringbuf 0.5.2 addresses the previous queue dependency's
  RustSec advisory.
- Preserve voice character in Auto-EQ, add Warm / Full Voice, and separate
  calibration from editable tone.
- Compare the complete processing chain on one recording. Voice Setup calibrates
  against the applied input processing and auto makeup, validates combined
  headroom, and reuses verification takes for bounded adjustments.
- Keep calibration responsive with shared renders, bounded candidate batches,
  progress reporting, and sound rankings independent of CPU timing noise.
- Improve VAD recovery, de-essing, sample-rate latency reporting, saved calibration
  evidence, and device/preset error handling. Add optional tray operation, a mute
  shortcut, and unified processing modes.

See the [1.13.0 changelog](https://github.com/FueledByRedBull/audio-forge/blob/master/CHANGELOG.md#v1130)
for the complete change list.

## Downloads

Choose the per-user MSI installer or extract the complete portable archive.
The release also includes corresponding source, one evidence archive containing
validation and provenance records, and one SHA256SUMS file for the downloads.

## Validation and compatibility

The release candidate at `fd02f793b11c90618461132e341efa4107a6c984` passed CI
and exact-artifact qualification, including portable startup, installer lifecycle,
payload provenance, and corresponding-source checks. Publication promotes those
same validated bytes without rebuilding them.

Windows 10/11 is the supported target. Automated checks and recorded-audio
evaluations do not establish physical reconnect, sleep recovery, accessibility,
or listening results for every microphone, device route, and display setup.
Those configurations remain outside this release's qualification claims.
