AudioForge 1.14.0 corrects compressor, limiter, and de-esser detection, keeps
device recovery on the devices you chose, and makes calibration and settings
handling more predictable.

## Which file do I need?

Download **one** of these. Not sure? Choose the installer.

| File | Choose it if |
| --- | --- |
| [AudioForge-v1.14.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.14.0/AudioForge-v1.14.0-win64.msi) | You want a normal install for your Windows user. **Recommended.** |
| [AudioForge-v1.14.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.14.0/AudioForge-v1.14.0-win64-ultra.7z) | You want a portable folder. Extract everything, then run `AudioForge.exe`. "Ultra" is the compression level. |

You don't need the other files to run AudioForge: the SHA256SUMS file checks
your download, the evidence archive holds validation records, and the source
archive is the corresponding source for redistribution.

To use AudioForge in Discord, OBS, Zoom, or Teams you also need a virtual audio
cable such as [VB-CABLE](https://vb-audio.com/Cable/); see the
[setup steps](https://github.com/FueledByRedBull/audio-forge#set-up).
AudioForge isn't code-signed, so Windows SmartScreen may warn before it runs.
Compare `Get-FileHash <file>` with `AudioForge-v1.14.0-SHA256SUMS.txt`, then
choose **More info > Run anyway**.

## Changes

- Measure compressor presence from a real 2 kHz band, catch short peaks at any
  sample rate, and make Adaptive Release's Base Release and manual makeup
  changes take effect smoothly.
- Ramp limiter gain across its 0.5 ms lookahead instead of stepping it, and base
  de-esser detection on a separate voice-body band.
- Recover only the devices you selected, apply configuration while muted with
  reliable rollback, and show unavailable meters instead of stale values.
- Audition and apply one verified EQ candidate, with warnings that follow each
  band's real filter type. Auto Voice Setup now changes gate and noise
  suppression settings less often; every safety check still passes.
- Validate presets strictly while keeping exact values, record every edit in
  undo history, and verify downloaded runtime assets before replacing them.

See the [1.14.0 changelog](https://github.com/FueledByRedBull/audio-forge/blob/master/CHANGELOG.md#v1140)
for the complete change list.

## Validation and compatibility

The release candidate at `804f728cd0591d9e30178e308661f72452a80739` passed CI
and exact-artifact qualification, including portable startup, installer smoke
and upgrade, payload and installer provenance, package smoke, and
corresponding-source checks. Publication promotes those same validated bytes
without rebuilding them.

Before tagging, the existing objective evaluators were re-run on this source:
automatic makeup, dynamics aliasing, limiter lookahead, processing order, voice
preservation against 1.13.0, and joint gate/suppression tuning all pass their
predefined gates. The expanded compressor search again failed qualification
and stays disabled. See [evaluation/README.md](https://github.com/FueledByRedBull/audio-forge/blob/master/evaluation/README.md)
for the reports and their limits.

AudioForge is built for Windows 10 (1809 or later) and Windows 11, 64-bit.
Automated checks and recorded-audio evaluations do not establish physical
reconnect, sleep recovery, accessibility, or listening results for every
microphone, device route, and display setup. Those configurations remain outside
this release's qualification claims.
