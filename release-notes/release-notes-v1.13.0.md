AudioForge 1.13.0 improves voice calibration, full-chain comparisons, peak
protection, and everyday device and preset handling.

## Which file do I need?

Download **one** of these. Not sure? Choose the installer.

| File | Choose it if |
| --- | --- |
| [AudioForge-v1.13.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.13.0/AudioForge-v1.13.0-win64.msi) (about 110 MB) | You want a normal install for your Windows user. **Recommended.** |
| [AudioForge-v1.13.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.13.0/AudioForge-v1.13.0-win64-ultra.7z) (about 97 MB) | You want a portable folder. Extract everything, then run `AudioForge.exe`. "Ultra" is the compression level. |

You don't need the other files to run AudioForge: the SHA256SUMS file checks
your download, the evidence archive holds validation records, and the source
archive is the corresponding source for redistribution.

To use AudioForge in Discord, OBS, Zoom, or Teams you also need a virtual audio
cable such as [VB-CABLE](https://vb-audio.com/Cable/); see the
[setup steps](https://github.com/FueledByRedBull/audio-forge#set-up).
AudioForge isn't code-signed, so Windows SmartScreen may warn before it runs.
Compare `Get-FileHash <file>` with `AudioForge-v1.13.0-SHA256SUMS.txt`, then
choose **More info > Run anyway**.

## Changes

- Improve short-burst true-peak limiting and preview safety. Final peak
  protection adds 6.25 ms of latency at 48 kHz compared with 1.12.1; ringbuf
  0.5.2 addresses the previous queue dependency's RustSec advisory.
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

## Validation and compatibility

The release candidate at `fd02f793b11c90618461132e341efa4107a6c984` passed CI
and exact-artifact qualification, including portable startup, installer lifecycle,
payload provenance, and corresponding-source checks. Publication promotes those
same validated bytes without rebuilding them.

AudioForge is built for Windows 10 (1809 or later) and Windows 11, 64-bit.
Automated checks and recorded-audio evaluations do not establish physical
reconnect, sleep recovery, accessibility, or listening results for every
microphone, device route, and display setup. Those configurations remain outside
this release's qualification claims.
