AudioForge 1.18.1 is a fix release. Switching the de-esser, muting and
recovering from a dropout no longer click, one bad sample no longer leaves the
compressor off for the rest of the session, 24-bit integer input devices work,
and the setup dialogs no longer freeze while they check a recording. It changes
no Voice Setup or Auto-EQ decision and no sound default.

**Close AudioForge 1.18.0 or earlier before you start 1.18.1.** The
single-instance lock and the activation pipe now carry your Windows account's
SID in their names, so the two versions do not see each other and both can run at
once.

## Downloads

- **Installer:** [AudioForge-v1.18.1-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.18.1/AudioForge-v1.18.1-win64.msi)
- **Portable:** [AudioForge-v1.18.1-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.18.1/AudioForge-v1.18.1-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## Fixes

- **Clicks and steps.** The de-esser fades over 10 ms when you switch it on or
  off. An EQ or de-esser filter changed while an earlier change is still
  fading finishes that fade first. A hum notch moved by less than half its
  bandwidth keeps its filter state, which leaves less residual hum. Audio after
  an output dropout fades in over 64 frames, and mute, unmute and the end of a
  recording or probe fade the same way instead of stepping. Each has a test on
  fixed input in which the steady-state output is unchanged and the largest
  step is smaller: mute from 0.45 to 0.011 and unmute from 0.44 to 0.056 of
  full scale, the de-esser from 0.15 to 0.0006, the first audio after a dropout
  from 0.48 to 0.055.
- A NaN after the downstream chain could leave the compressor not compressing
  for the rest of the session. Those samples are now zeroed and counted, and
  the de-esser, EQs and compressor detector are reset.
- Health could show "Input: ALREADY COMPRESSED (0.0 dB)" on plain room noise,
  because the crest factor compared one short audio block with a 100 ms RMS. It
  now uses the meter window for both. The meters' peak reading is unchanged.
- 24-bit integer input devices (I24 and U24) work. A device that reports zero
  channels or a zero sample rate now fails at start with an error.
- A panic on the audio thread is recorded and requests a restart. After a
  panic, stall, stream error or output-resampler failure the restart now names
  that cause, where a suppressor rebuild used to take its place. A failed
  start cleans up after itself, and a refused multimedia-scheduling request
  runs the engine at normal priority instead of failing the start.
- The voice activity worker waits between load attempts, reports why loading
  failed and stops promptly. It accepts only the pinned Silero VAD v6.2.1 model
  (checked by SHA-256), and a `VAD_MODEL_PATH` that does not exist is an
  error. `df.dll` loads without the current directory or `PATH` in its search.
- The gate, de-esser and compressor apply their settings in one step. The
  engine ignores NaN and infinite values for its settings.
- The recording check in Voice Setup and the calibration dialogs ran on the
  window thread and could freeze it. It now runs on a worker and shows
  "Checking the recording..."; the result is dropped if you close the dialog.
- Quit from the tray rolls back every open setup dialog before it asks about
  unsaved changes. A minimized window no longer trips the renderer watchdog. A
  preset that cannot be read because its file is locked stays your last preset.
- An invalid or newer-schema `config.json` keeps one recovery copy instead of
  a new one on every launch. A save retries a brief sharing violation and gives
  up at once on a read-only file, so the window no longer stalls.
- A preset name that turns into the same file name as another preset gets its
  own file, so the overwrite prompt names the right preset. Windows device
  names and very long names are handled. A deeply nested route key or an
  oversized saved window geometry no longer discards the whole `config.json`.

## Changes

- The diagnostics export (still schema version 3) gains
  `downstream_non_finite_count`, `output_resampler_error_count`,
  `rejected_control_value_count`, `dsp_scheduling_degraded`,
  `auto_makeup_error_present` and `vad_worker_error_present`. Its
  `rt_error_name` now carries the engine's own error names; before, it read
  `other` for every error.
- `app.log` now records uncaught exceptions and Qt or QML warnings with
  tracebacks. A message that repeats is counted and logged at the 1st, 10th,
  100th and every 1000th time, with the total at exit. A second launch no
  longer opens the log.
- The limiter's duplicate output oversampler is gone: about 27% less true-peak
  CPU for the same output.
- Settings from earlier versions load as before, and so do presets.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

Steady-state audio is the same as in 1.18.0: the golden outputs for the chain
on the public bench set match unchanged. The click fixes were measured on
synthetic signals and on that bench set, not by ear on a live call.

No audio-hardware run of this build is claimed. The hour-long soak and the
hands-on checks in the 1.18.0 notes were of the 1.18.0 candidate and are not
attributed to this one.

I24 and U24 input is covered by sample-conversion tests. No device that
delivers those formats is claimed to have been tried.

Voice Setup can still refuse a setup that would have been no worse than the
settings it would replace; that is unchanged from 1.18.0.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
