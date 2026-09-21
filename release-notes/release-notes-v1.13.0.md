# AudioForge 1.13.0

AudioForge 1.13.0 improves voice calibration, full-chain comparisons, peak
protection, and everyday device and preset handling.

## Changes

- Correct short-burst true-peak limiting and match preview safety behavior to
  live processing. Final peak protection adds 6.25 ms of latency at 48 kHz
  compared with the previous implementation.
- Keep de-esser bands within their selected interval, remove the zero-gain
  dead zone from EQ fitting, and reject stale VAD results after stream gaps.
- Upgrade ringbuf to 0.5.2 to address the RustSec advisory affecting the previous
  queue dependency.
- Preserve voice character during Auto-EQ, add the Warm / Full Voice target,
  and keep microphone correction separate from editable tone.
- Compare the full processing chain on one recording before applying settings.
- Calibrate Auto Voice Setup with the actual auto-makeup settings, explain
  headroom constraints, and keep neutral EQ free of false abstention warnings.
- Fit compression after selecting input processing, reusing its rendered audio
  and speech/noise confidence in scoring and calibration.
- Reduce repeated suppression work, run compressor candidates in bounded parallel
  batches, and keep analysis responsive with stage and candidate progress.
- Keep feasible gate and suppression rankings independent of CPU timing noise.
- Validate the combined cleanup, gate, suppression, EQ and dynamics chain before
  offering the candidate, and recheck the actual applied settings on verification.
- Correct automatic loudness feedback and include speech-aware gain control in
  full-chain previews and verification.
- Save complete calibration evidence, allow verification after leaving Raw Monitor,
  and handle startup/device settings write failures without false success.
- Include input sample-rate conversion in the engine latency estimate.
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
