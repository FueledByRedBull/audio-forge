AudioForge 1.19.0 moves DeepFilterNet inference to a worker thread of its own,
makes the audio engine the only watcher and restarter of the streams, and lets
Voice Setup go on when the room-noise recording is questionable. The window
shows the mute state more clearly, and a Windows sign-out or shutdown no longer
waits on a dialog. DeepFilterNet stays on by default, and its latency labels now
include the worker's hand-off, about 5 ms.

**Settings format.** The settings file (`config.json`) stays at schema version 2
and the preset format is unchanged. Presets saved by 1.19.0 are stamped 1.19.0,
and 1.18.2 and older refuse them as newer.

## Downloads

- **Installer:** [AudioForge-v1.19.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.19.0/AudioForge-v1.19.0-win64.msi)
- **Portable:** [AudioForge-v1.19.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.19.0/AudioForge-v1.19.0-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## Changes

- **DeepFilterNet worker.** Inference runs on a worker thread, so a slow frame
  can no longer stall the audio path. The labels read about 15 ms for
  DeepFilterNet LL and about 35 ms for DeepFilterNet, up from 10 and 30. RNNoise
  is unchanged.
- **DeepFilterNet failures.** If the worker returns a bad frame, panics, exits
  or stops answering, audio passes through unsuppressed and diagnostics give the
  reason. Each model is checked against its pinned size and SHA-256 before it
  loads, and non-finite input is zeroed.
- **Recovery.** The audio engine is the only stall detector and restarter. Waits
  between restarts run 0, 2, 5, 10, 20, then 30 s, and reset after both streams
  have delivered for 10 s. The engine also restarts when the output holds exact
  digital zeros for 1.5 s while the input is above -35 dB. A stall restart reads
  "Audio restarted automatically after: audio stalled; you may have heard a
  short gap". Recovery pauses while the latency, Auto-EQ and Voice Setup dialogs
  are open.
- **Engine calls.** Start, Stop, device listing and recovery run off the window
  thread. The window stays responsive and the engine controls lock until the
  call finishes.
- **Voice Setup.** A questionable room-noise recording (unsteady, partly
  matching the voice, or with speech in it) no longer refuses the whole setup.
  The gate and noise suppressor keep your current settings, Auto-EQ, the
  de-esser and the compressor are judged as for a usable recording, and
  automatic loudness stays off. With no current gate, the factory gate stands.
  The review says why. An invalid recording, a weak voice-to-noise ratio, a low
  capture score, the final headroom check, and a silent recording with no pauses
  to measure the room from still refuse. Analysis runs candidates on a worker
  pool and finishes sooner with the same decisions.
- **Mute.** A Muted pill sits in the top bar and the tray icon shows a muted
  variant. **Ctrl+M** mutes inside the window; **Ctrl+Alt+M** stays the global
  shortcut. The mode picker takes an alert look in Bypass and Raw.
- **Sign-out and shutdown.** A Windows sign-out or shutdown never waits on the
  unsaved-changes question or a failed save, and the setup dialogs close without
  asking. Quitting from the hidden tray runs the same shutdown as closing the
  window.
- **Devices.** Labels start with the instance number and `(Default)`, as in
  `[#2] (Default) Name`, so identical devices stay apart when a picker cuts off
  the end. Saved device keys are unchanged.
- **Problems stay visible.** A failed save, a missing or broken startup preset
  and an unplugged saved device stay in the warning banner until resolved. The
  startup-failure dialog shows the log file in use, and the renderer failure
  screen says whether audio is still running and how to stop it. Open Presets
  Folder warns when the folder is missing.
- **Accessibility.** Hints open on keyboard focus as well as hover and close on
  Esc, and every control exposes its tooltip as its accessible description.
  Control edges and switch and slider tracks keep at least 3.35:1 contrast. One
  focus ring marks every focusable control and the rail. Stage cards follow
  signal order for Tab and screen readers in every layout, and dialog buttons
  follow their visual order. Preset names, device names and status text show as
  plain text, never as markup.
- **Release workflows.** Promote and both hardware workflows fail unless they
  start from `master` and the person who triggered them has the admin or
  maintain role. Promote runs in two jobs: `verify-candidate` has a read-only
  token and runs the candidate's code, and `publish-release` alone can write,
  sits behind the `release` environment and re-hashes the staged files before
  publishing.

## Fixes

- A leftover migration marker no longer blocks saves on every launch. Logging
  now resolves the config folder first, so it can no longer make a pending
  migration look finished. While a migration is pending the log goes to
  `%TEMP%\AudioForge\logs` and its first line gives the reason. No user files
  are deleted, and a config folder stranded as a partial copy is moved back.
- A corrupt DeepFilterNet model or a failing frame no longer ends the app.
- A device that opens but stays silent is no longer restarted every 6 s, and
  loud steady noise or quiet speech is no longer taken for a silent output.
- The first small block after a pause no longer raises a false suppressor
  failure, and a noise model chosen while the stream is starting now reaches the
  running engine.
- Changing one device picker no longer overwrites the other picker's device with
  its temporary fallback.
- Undo and redo of a preset load restore that preset's name, file and saved
  state along with its settings.
- Stop stays available while a restart is pending.
- A slider value set through an accessibility tool now commits.
- Voice Setup shows "Analysis in progress" instead of "Complete!" while it
  analyses.

## Limits

Each live DeepFilterNet worker wakes about 1000 times a second, including while
noise suppression is off.

Voice Setup still refuses an invalid room-noise recording and a weak
voice-to-noise ratio, for example TV or music close to the level of your voice.
Retaking the recording does not help while the sound keeps playing. It can also
refuse a setup that would have been no worse than the settings it would
replace. Its 35 ms latency budget compares the offline engine, not the live
DeepFilterNet worker.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
